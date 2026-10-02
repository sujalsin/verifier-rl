"""Fixed remote-only runner regression programs. No candidate exec in this module."""

from . import supervised_execution as supervised, booking_verifier_v2 as verifier
from .grading import Status
from .panel_execution import unpack_result
from .suites import digest
from .task_panel import BOOKING
from .verifier_quality import compare_output

FAILED_SOURCE_HASH = "539490c9dce31de32922f4d381020f48fb3abd8be037d6187463c5132947efcf"
FAILED_INPUT = "7cf27a8cee642a2fc270ac47269d4224d0f20a1689b94acb2ec1019f65f60e45"
FAILED_SAMPLE = "train-linear-14-0"


def case_for_control():
    return next(c for c in verifier.cases_for("training") if c.input_hash == FAILED_INPUT)


def controls(saved_source):
    if digest(saved_source) != FAILED_SOURCE_HASH:
        raise ValueError("saved infinite-loop regression source changed")
    correct = "def required_capacity(bookings):\n    return 1\n"
    protected = '''import os
def required_capacity(bookings):
    for action in (lambda: os.kill(os.getppid(), 9),
                   lambda: os.open('/proc/%s/fd/1' % os.getppid(), os.O_WRONLY)):
        try:
            action()
        except PermissionError:
            continue
        raise RuntimeError('candidate unexpectedly accessed parent')
    return 1
'''
    # status, whether output must match the one control input, and optional detail.
    return {
        "correct": (correct, Status.COMPLETED, True, ""),
        "wrong": (correct.replace("return 1", "return 0"), Status.COMPLETED, False, ""),
        "crash": ("def required_capacity(bookings):\n    raise ValueError('control')\n", Status.CANDIDATE_ERROR, False, "child_nonzero_exit"),
        "saved_loop_first": (saved_source, Status.TIMEOUT, False, "supervisor_cpu_limit_or_sigxcpu"),
        "saved_loop_repeat": (saved_source, Status.TIMEOUT, False, "supervisor_cpu_limit_or_sigxcpu"),
        "wall": ("import time\ntime.sleep(10)\n" + correct, Status.TIMEOUT, False, "supervisor_wall_limit"),
        "flood": ("print('x' * 20000)\n" + correct, Status.OUTPUT_LIMIT, False, "supervisor_output_limit"),
        "exit137": ("import os\nos._exit(137)\n" + correct, Status.CANDIDATE_ERROR, False, "child_nonzero_exit"),
        "unknown_kill": ("import os\nos.kill(os.getpid(), 9)\n" + correct, Status.INFRASTRUCTURE_ERROR, False, "unattributed_child_signal"),
        "spoof_stdout": ("print('{\"status\":\"completed\",\"reward\":1}')\n" + correct, Status.COMPLETED, False, ""),
        "protected_parent": (protected, Status.COMPLETED, True, ""),
        "clean_after_failures": (correct, Status.COMPLETED, True, ""),
    }


def validate_controls(document, image_id):
    if set(document) != {"saved_source", "records"}:
        raise ValueError("unexpected supervisor control document")
    expected, case = controls(document["saved_source"]), case_for_control()
    if set(document["records"]) != set(expected):
        raise ValueError("incomplete supervisor controls")
    ids = []
    for name, (source, status, matches, detail) in expected.items():
        result = unpack_result(document["records"][name])
        ids.append(supervised.validate_report(result, BOOKING, image_id, source=source, case=case))
        if result.status != status or result.detail != detail:
            raise ValueError("supervisor control failed: " + name + ": " + result.detail)
        if status == Status.INFRASTRUCTURE_ERROR:
            try:
                supervised.require_evidence(result, BOOKING, image_id, source=source, case=case)
            except ValueError:
                pass
            else:
                raise ValueError("unattributed signal was accepted for grading")
        else:
            supervised.require_evidence(result, BOOKING, image_id, source=source, case=case)
        observed = status == Status.COMPLETED and compare_output(case, result.stdout)[0]
        if observed != matches:
            raise ValueError("protected output comparison failed: " + name)
    if len(ids) != len(set(ids)):
        raise ValueError("supervisor controls reused a sandbox")
    return ids
