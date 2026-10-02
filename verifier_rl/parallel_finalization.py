"""Bounded read-ahead for the unchanged, sequential evidence verifier.

Only JSON reads run in threads. Decisions, equality checks, reward replay,
ordering, and aggregation remain in the frozen verifier's calling thread.
No candidate program is executed, and this module never writes evidence.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

VERSION = "parallel-json-finalization-0.2"
WORKERS = 16


def journal_records(key, raw):
    """Yield exactly the paths/order used by the frozen verifier."""
    for sid, inputs in raw["entries"].items():
        for input_hash, entry in inputs.items():
            for name, value in entry.items():
                for component in (key, sid, input_hash, name):
                    if (not isinstance(component, str) or not component or
                            component in {".", ".."} or "/" in component or "\\" in component):
                        raise ValueError("unsafe execution journal component")
                yield f"grading/{key}/inputs/{sid}/{input_hash}/{name}.json", value


def read_record(path):
    data = Path(path).read_bytes()
    return json.loads(data), len(data), hashlib.sha256(data).hexdigest()


class PrefetchJSONReader:
    """At most `workers` active reads, 2*workers queued; consume in original order.

Raw batch records provide filenames, NOT replacement contents. Every individual
file is opened and parsed independently. Exceptions propagate to the original
verifier. There is no cache across batches, no skipping, and no retry policy.
"""

    def __init__(self, root, *, workers=WORKERS, read=read_record):
        if type(workers) is not int or not 1 <= workers <= WORKERS:
            raise ValueError("parallel reads must be bounded to 1..16")
        self.root, self.workers, self.read = Path(root), workers, read
        self.pending = deque()
        self.pool = None
        self.records = iter(())
        self.batch = None
        self.batches = []
        self.max_pending = 0
        self.synchronous_reads = 0
        self.completed = False

    def __enter__(self):
        if self.pool is not None:
            raise RuntimeError("reader cannot be entered twice")
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="journal-read")
        return self

    def _fill(self):
        while len(self.pending) < 2 * self.workers:
            try:
                relative, _ = next(self.records)
            except StopIteration:
                break
            self.pending.append((relative, self.pool.submit(self.read, self.root / relative)))
            self.max_pending = max(self.max_pending, len(self.pending))

    def _finish_batch(self):
        if self.pending:
            raise ValueError("not all prefetched journal records were checked")
        if self.batch is not None:
            record = dict(self.batch)
            record["read_order_sha256"] = self.transcript.hexdigest()
            record["elapsed_seconds"] = time.monotonic() - self.started
            self.batches.append(record)
            self.batch = None

    def __call__(self, path):
        if self.pool is None:
            raise RuntimeError("reader must be used as a context manager")
        relative = Path(path).relative_to(self.root).as_posix()
        parts = PurePosixPath(relative).parts
        if ".." in parts:
            raise ValueError("reader path outside evidence root")
        if len(parts) >= 3 and parts[0] == "grading" and parts[2] == "inputs":
            if not self.pending or self.pending[0][0] != relative:
                raise ValueError("journal read order differs from frozen verifier")
            _, future = self.pending.popleft()
            value, size, sha = future.result()
            self.batch["files"] += 1
            self.batch["bytes"] += size
            self.transcript.update(json.dumps([relative, sha], separators=(",", ":")).encode() + b"\n")
            self._fill()
            return value
        value, _, _ = self.read(path)
        self.synchronous_reads += 1
        if len(parts) == 3 and parts[0] == "grading" and parts[2] == "raw.json":
            self._finish_batch()
            self.batch = {"key": parts[1], "files": 0, "bytes": 0}
            self.started, self.transcript = time.monotonic(), hashlib.sha256()
            self.records = iter(journal_records(parts[1], value))
            self._fill()
        return value

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self._finish_batch()
                self.completed = True
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        return False

    def receipt(self):
        if self.pool is not None or not self.completed:
            raise RuntimeError("read receipt requires completed verification")
        return {"version": VERSION, "workers": self.workers, "max_pending": self.max_pending,
                "prefetch_limit": 2 * self.workers,
                "synchronous_reads": self.synchronous_reads, "batches": self.batches,
                "journal_files": sum(b["files"] for b in self.batches)}


@contextmanager
def parallel_reader(frozen, root, *, workers=WORKERS):
    """Replace only the authored launcher's JSON reader, restoring on all exits.

The caller must own its process and not run two verifiers concurrently in it.
The frozen verifier and all scoring functions remain byte-for-byte unchanged.
"""
    original = frozen.read_json
    with PrefetchJSONReader(root, workers=workers) as reader:
        frozen.read_json = reader
        try:
            yield reader
        finally:
            frozen.read_json = original
