"""Run untrusted candidate code in an isolated subprocess.

Candidate patches are LLM output and must never be executed in-process. Each
call spawns ``python -I`` (isolated mode), applies resource limits where the
platform supports them, enforces a per-input alarm and an overall wall-clock
timeout, and communicates through JSON on stdin/stdout. Inputs are passed as
``repr`` strings and rebuilt with ``ast.literal_eval`` in the child, so only
literal values (numbers, strings, lists, dicts, tuples, sets, None) are supported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence

from acrdl.types import ExecResult

_RUNNER = r'''
import ast, json, signal, sys
cfg = json.loads(sys.stdin.read())
try:
    import resource
    mem = cfg["mem_mb"] * 1024 * 1024
    for lim in ("RLIMIT_AS", "RLIMIT_DATA"):
        if hasattr(resource, lim):
            try:
                resource.setrlimit(getattr(resource, lim), (mem, mem))
            except (ValueError, OSError):
                pass
except ImportError:
    pass

class _Timeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _Timeout()

has_alarm = hasattr(signal, "SIGALRM")
if has_alarm:
    signal.signal(signal.SIGALRM, _alarm)

out = []
ns = {"__name__": "__candidate__"}
try:
    exec(compile(cfg["source"], "<candidate>", "exec"), ns)
    fn = ns[cfg["entry"]]
except BaseException as e:  # noqa: BLE001
    print(json.dumps({"load_error": type(e).__name__}))
    sys.exit(0)

n_timeouts = 0
for arg_repr in cfg["inputs"]:
    if n_timeouts >= cfg["max_timeouts"]:
        # a candidate that keeps looping would cost per_input_s on every
        # remaining input; record the rest as timeouts instead
        out.append(["timeout", ""])
        continue
    args = ast.literal_eval(arg_repr)
    try:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, cfg["per_input_s"])
        val = fn(*args)
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            text = repr(val)
        except BaseException:
            text = "<unrepr>"
        out.append(["ok", text[:2000]])
    except _Timeout:
        n_timeouts += 1
        out.append(["timeout", ""])
    except BaseException as e:  # noqa: BLE001
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        out.append(["error", type(e).__name__])
print(json.dumps({"results": out}))
'''


def run_function(
    source: str,
    entry_point: str,
    inputs: Sequence[tuple],
    per_input_timeout_s: float = 2.0,
    memory_limit_mb: int = 512,
    max_timeouts: int = 3,
) -> list[ExecResult]:
    """Execute ``entry_point`` from ``source`` on every input tuple.

    Returns one :class:`ExecResult` per input. A source that fails to load
    yields ``error:<ExceptionName>`` for every input.
    """
    inputs = list(inputs)
    if not inputs:
        return []
    payload = json.dumps(
        {
            "source": source,
            "entry": entry_point,
            "inputs": [repr(tuple(a)) for a in inputs],
            "per_input_s": float(per_input_timeout_s),
            "mem_mb": int(memory_limit_mb),
            "max_timeouts": int(max_timeouts),
        }
    )
    overall = per_input_timeout_s * len(inputs) + 5.0
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=overall,
        )
    except subprocess.TimeoutExpired:
        return [ExecResult("timeout") for _ in inputs]

    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return [ExecResult("error", "Crash") for _ in inputs]
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError:
        return [ExecResult("error", "Crash") for _ in inputs]
    if "load_error" in data:
        return [ExecResult("error", f"Load{data['load_error']}") for _ in inputs]
    results = [ExecResult(k, v) for k, v in data["results"]]
    # A hard crash mid-run (e.g. memory limit) truncates output: pad.
    results += [ExecResult("error", "Crash")] * (len(inputs) - len(results))
    return results


def expected_signature(expected: object) -> str:
    """Signature of an expected test value, comparable to ``ExecResult.signature``."""
    return f"ok:{expected!r}"
