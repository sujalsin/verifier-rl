from contextlib import redirect_stdout
import io
import json
import threading
import unittest

from verifier_rl.progress import ProgressLog


class ProgressTests(unittest.TestCase):
    def test_heartbeat_reports_stalled_work_without_advancing_it(self):
        records, now, observed = [], [100.0], threading.Event()

        def emit(record):
            records.append(record)
            if record["event"] == "heartbeat" and record["elapsed_seconds"] == 35:
                observed.set()

        with ProgressLog("fixture", interval_seconds=.01, emit=emit, clock=lambda: now[0]) as progress:
            progress.stage("journals", total=10, unit="files", batch_index=2, batches_total=9)
            progress.advance(3)
            progress.item("pending.json")
            now[0] += 35
            self.assertTrue(observed.wait(2), "no heartbeat while main thread waited")
        heartbeat = next(r for r in records if r["event"] == "heartbeat" and r["elapsed_seconds"] == 35)
        self.assertEqual(heartbeat["completed"], 3)
        self.assertEqual(heartbeat["total"], 10)
        self.assertEqual(heartbeat["current_item"], "pending.json")
        self.assertEqual(heartbeat["seconds_since_progress"], 35)
        self.assertEqual(heartbeat["batch_index"], 2)
        self.assertEqual(records[-1]["event"], "completed")
        self.assertFalse(progress._thread.is_alive())

    def test_stage_durations_and_counters_reset_without_per_file_logs(self):
        records, now = [], [10.0]
        with ProgressLog("fixture", emit=records.append, clock=lambda: now[0]) as progress:
            progress.stage("journals", total=20, unit="files", batch="batch-1")
            for i in range(20):
                progress.item(f"{i}.json")
                progress.advance()
            self.assertEqual(len(records), 2)
            now[0] += 12
            progress.stage("commit_artifacts")
            now[0] += 8
        done = [r for r in records if r["event"] == "stage_completed"]
        self.assertEqual([r["stage_elapsed_seconds"] for r in done], [12, 8])
        self.assertEqual(done[0]["completed"], 20)
        self.assertEqual(done[1]["completed"], 0)
        self.assertNotIn("batch", done[1])
        self.assertEqual(records[-1]["elapsed_seconds"], 20)

    def test_failure_propagates_and_never_logs_completion(self):
        records = []
        failure = ValueError("private contents must not be logged")
        with self.assertRaises(ValueError) as raised:
            with ProgressLog("fixture", emit=records.append) as progress:
                progress.stage("journals", total=10, unit="files")
                progress.item("failed.json")
                raise failure
        self.assertIs(raised.exception, failure)
        self.assertEqual(records[-1]["event"], "failed")
        self.assertEqual(records[-1]["error_type"], "ValueError")
        self.assertEqual(records[-1]["current_item"], "failed.json")
        self.assertNotIn("completed", [r["event"] for r in records])
        self.assertNotIn("private contents", json.dumps(records))
        self.assertFalse(progress._thread.is_alive())

    def test_broken_log_sink_does_not_fail_work(self):
        def broken_sink(record):
            raise OSError("logging unavailable")

        with ProgressLog("fixture", emit=broken_sink) as progress:
            progress.stage("summary")
            progress.advance()
        self.assertFalse(progress._thread.is_alive())

    def test_default_output_is_structured_and_terminal(self):
        output = io.StringIO()
        with redirect_stdout(output), ProgressLog("fixture") as progress:
            progress.stage("commit_artifacts")
        records = [json.loads(line.removeprefix("FINALIZATION ")) for line in output.getvalue().splitlines()]
        self.assertEqual(records[-1]["event"], "completed")
        self.assertEqual(records[-1]["phase"], "commit_artifacts")
        self.assertTrue(all(r["run_id"] == "fixture" and r["utc"] for r in records))

    def test_invalid_interval_and_reuse_rejected(self):
        for interval in (0, -1, float("inf"), float("nan")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                ProgressLog("fixture", interval_seconds=interval)
        with ProgressLog("fixture", emit=lambda record: None) as progress:
            pass
        with self.assertRaises(RuntimeError):
            with progress:
                pass


if __name__ == "__main__":
    unittest.main()
