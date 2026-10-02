"""Fixed CPU-only controls and throughput benchmark; no research samples/updates."""

import asyncio

from . import booking_baseline_comparison as booking, booking_replication as replication
from . import program_execution as execution
from .panel_execution import fixture_source
from .suites import digest
from .task_panel import BOOKING

VERSION = "booking-program-benchmark-0.1"
RUN_ID = "booking-program-sandbox-benchmark-20260930-v1"
FUNCTION_SECONDS = 3600
RESERVATION_USD = "5"
SAVED_POOL = ("runs/qwen-booking-matched-training-20260929-v1/analysis/weak-only-001/evidence/"
              "qwen-booking-matched-training-20260929-v1/arms/reference/evaluation-24/result.json")


def control_cases():
    cases = booking.cases_for("training")
    return (next(c for c in cases if not c.arguments["bookings"]),
            next(c for c in cases if len(c.arguments["bookings"])==1))


def controls():
    correct = fixture_source(BOOKING,"correct")
    conditional = "\ndef required_capacity(bookings):\n    if not bookings:\n        ACTION\n    return authored_booking(bookings, 'correct')\n"
    controls = {"correct":(correct,["completed","completed"],[True,True]),
                "wrong":(fixture_source(BOOKING,"constant"),["completed","completed"],[True,False])}
    for name,action,status in (
        ("crash","raise ValueError('control')","candidate_error"),
        ("cpu","while True: pass","timeout"),
        ("wall","__import__('time').sleep(10)","timeout"),
        ("flood","print('x' * 50000)","output_limit"),
        ("exit137","__import__('os')._exit(137)","candidate_error"),
        ("self_signal","__import__('os').kill(__import__('os').getpid(),9)","infrastructure_error"),
        ("memory","bytearray(512 * 1024 * 1024)","candidate_error"),
    ):
        controls[name]=(correct+conditional.replace("ACTION",action),[status,"completed"],
                        [None if name=="self_signal" else False,True])
    controls["globals_reset"]=(correct+'''
calls = 0
def required_capacity(bookings):
    global calls
    calls += 1
    assert calls == 1, "Python globals leaked between inputs"
    return authored_booking(bookings, 'correct')
''',["completed"]*2,[True]*2)
    controls["stdlib_imports"]=(correct+'''
import math, bisect, heapq, collections, itertools, functools, decimal, fractions, re, typing
assert math.sqrt(4)==2 and bisect.bisect_left([1,3],2)==1
assert decimal.Decimal('1.5')+1==decimal.Decimal('2.5')
''',["completed"]*2,[True]*2)
    controls["permissions_and_persistence"]=(correct+'''
import os, socket, subprocess, ctypes
marker = '/dev/shm/verifier_rl_program_marker'
assert not os.path.exists(marker), 'file persisted between tests'
for action in (
    lambda: os.open(marker, os.O_WRONLY|os.O_CREAT, 0o600),
    lambda: os.open('/dev/null', os.O_RDWR),
    lambda: os.mkdir('/tmp/verifier_rl_program_directory'),
    lambda: os.fork(),
    lambda: subprocess.Popen(['/bin/true']),
    lambda: socket.socket(),
    lambda: os.kill(os.getppid(),9),
    lambda: os.open('/proc/%s/fd/1'%os.getppid(),os.O_WRONLY),
):
    try:
        action()
    except PermissionError:
        pass
    else:
        raise AssertionError('forbidden side effect allowed')
libc = ctypes.CDLL(None,use_errno=True)
assert libc.prctl(21,0,0,0,0)==2, 'filter not active'
''',["completed"]*2,[True]*2)
    controls["spoof_stdout"]=(correct+"\nprint('{\"reward\":1,\"status\":\"completed\"}')\n",
                               ["completed"]*2,[False]*2)
    return controls


def programs(saved):
    selected = saved["samples"][:4]
    expected = [f"eval-reference-24-{i}" for i in range(18000,18004)]
    if [s["sample_id"] for s in selected]!=expected or any(not s.get("source") for s in selected):
        raise ValueError("fixed saved-program pool changed")
    authored = [{"id":"authored-"+fault,"source":fixture_source(BOOKING,fault)}
                for fault in ("correct","inclusive_boundary","constant")]
    return authored+[{"id":s["sample_id"],"source":s["source"]} for s in selected]


def comparison_cases():
    return booking.cases_for("training")[::8]


def manifest(saved):
    pool = programs(saved)
    return {"version":VERSION,"run_id":RUN_ID,"execution_version":execution.VERSION,
        "profile":execution.PROFILE,"runner_hash":digest(execution.runner()),
        "controls":{n:{"source_hash":digest(s),"statuses":statuses,"passes":passes}
                    for n,(s,statuses,passes) in controls().items()},
        "pool":[{"id":p["id"],"source_hash":digest(p["source"])} for p in pool],
        "comparison_inputs":[c.input_hash for c in comparison_cases()],
        "benchmark_inputs":[c.input_hash for c in booking.cases_for("evaluation")],
        "parallel_programs":4,"owning_controllers":1,"creation_interval_seconds":.30,
        "function_seconds":FUNCTION_SECONDS,"reservation_usd":RESERVATION_USD,
        "research_updates":0,"new_model_samples":0,"automatic_study_launch":False,
        "retry_policy":"none; preserve every partial result and stop on ambiguity",
        "benchmark_modes":["sequential","parallel"],
        "limitations":["Small fixed pool: three authored programs and first four saved reference-final programs.",
            "Not an independent model evaluation or estimate of the policy distribution.",
            "Readonly/no-descendants permissions are stricter than the historical runner.",
            "Clock, PID and other read-only kernel state are not identical between cases.",
            "No security proof and no measured full training-study runtime."]}


def maximum_sandbox_seconds():
    return (len(controls())*execution.lifetime_for(2) + 3*execution.lifetime_for(96)
            + 7*execution.lifetime_for(len(comparison_cases()))
            + 2*7*execution.lifetime_for(287) + 7*len(comparison_cases())*120)


def grader_counts(outcomes):
    train = booking.cases_for("training")
    weak,_ = booking.contrast.partition(train)
    return [sum(outcomes[c.input_hash]["passed"] is True for c in selected)
            for selected in (train,weak,replication.repair()[2])]


def validate_control(name,outcomes):
    _,statuses,passes=controls()[name]
    cases=control_cases()
    if ([outcomes[c.input_hash]["status"] for c in cases]!=statuses
            or [outcomes[c.input_hash]["passed"] for c in cases]!=passes):
        raise ValueError("program control failed: "+name)


def summarize_timing(seconds,program_count,case_count):
    if seconds<=0:
        raise ValueError("invalid benchmark timing")
    rate=program_count*case_count/seconds
    return {"seconds":seconds,"programs":program_count,"inputs":program_count*case_count,
            "inputs_per_second":rate,"programs_per_second":program_count/seconds,
            "full_research_grading_hours_at_this_rate":replication.workload()["research_input_slots"]/rate/3600,
            "projection_is_not_total_training_runtime":True}


async def bounded_map(items, worker, concurrency):
    """Stop queued work on failure, but await cleanup of every in-flight item."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("positive concurrency required")
    limit, stop = asyncio.Semaphore(concurrency), asyncio.Event()
    async def run(item):
        async with limit:
            if stop.is_set():
                return None
            try:
                return await worker(item)
            except Exception:
                stop.set()
                raise
    results = await asyncio.gather(*(run(item) for item in items), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results
