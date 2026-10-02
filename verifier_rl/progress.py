"""Best-effort stage logs and liveness heartbeats; never evidence or scoring."""

from datetime import datetime, timezone
import json
import math
import threading
import time


class ProgressLog:
    """Report long synchronous work, including time with no measured progress.

    Only the calling thread advances counters. The heartbeat thread reports the
    last observed state; it does not inspect artifacts or imply work advanced.
    Use identifiers/counts only, never candidate source or expected answers.
    """

    def __init__(self, run_id, *, label="FINALIZATION", interval_seconds=30,
                 emit=None, clock=None):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("heartbeat interval must be finite and positive")
        self.run_id, self.label = run_id, label
        self.interval_seconds = interval_seconds
        self._emit = emit or self._print
        self._clock = clock or time.monotonic
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = None
        self._phase, self._details = "starting", {}
        self._completed, self._total, self._unit, self._item = 0, None, None, None

    def _print(self, record):
        print(self.label, json.dumps(record, sort_keys=True), flush=True)

    def _log(self, event, **extra):
        # Caller holds the lock so a terminal event cannot race a heartbeat.
        now = self._clock()
        record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                  "run_id": self.run_id, "phase": self._phase,
                  "elapsed_seconds": round(now - self._started, 1),
                  "stage_elapsed_seconds": round(now - self._stage_started, 1),
                  "seconds_since_progress": round(now - self._last_progress, 1),
                  "completed": self._completed, "total": self._total,
                  "unit": self._unit, "current_item": self._item,
                  **self._details, **extra}
        try:
            self._emit(record)
        except Exception:
            # An unavailable log sink must not change rewards or replay results.
            pass

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("progress log cannot be reused")
        self._started = self._stage_started = self._last_progress = self._clock()
        with self._lock:
            self._log("started")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                if not self._stop.is_set():
                    self._log("heartbeat")

    def stage(self, name, *, total=None, unit=None, **details):
        with self._lock:
            if self._phase != "starting":
                self._log("stage_completed")
            self._phase, self._details = name, details
            self._completed, self._total, self._unit, self._item = 0, total, unit, None
            self._stage_started = self._last_progress = self._clock()
            self._log("stage_started")

    def item(self, identifier):
        with self._lock:
            self._item = identifier

    def advance(self, amount=1):
        with self._lock:
            self._completed += amount
            self._item = None
            self._last_progress = self._clock()

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()
        with self._lock:
            if exc_type is None:
                self._log("stage_completed")
                self._log("completed")
            else:
                self._log("failed", error_type=exc_type.__name__)
        return False
