from acrdl.evidence.sandbox import run_function
from acrdl.evidence.testing import run_tests, values_equal
from acrdl.types import TestCase


def test_values_and_exceptions():
    src = "def f(x):\n    return 10 // x\n"
    res = run_function(src, "f", [(2,), (0,)])
    assert res[0].signature == "ok:5"
    assert res[1].signature == "error:ZeroDivisionError"


def test_timeout_is_reported_and_later_inputs_still_run():
    src = "def f(x):\n    while x:\n        pass\n    return 1\n"
    res = run_function(src, "f", [(1,), (0,)], per_input_timeout_s=0.3)
    assert res[0].kind == "timeout"
    assert res[1].signature == "ok:1"


def test_load_error_and_missing_entry():
    assert run_function("def f(:\n", "f", [(1,)])[0].value == "LoadSyntaxError"
    assert run_function("x = 1\n", "f", [(1,)])[0].value == "LoadKeyError"


def test_arguments_are_isolated_between_calls():
    src = "def f(xs):\n    xs.append(1)\n    return len(xs)\n"
    assert [r.value for r in run_function(src, "f", [([],), ([],)])] == ["1", "1"]


def test_run_tests_uses_float_tolerance():
    src = "def f(x):\n    return x / 3 * 3\n"
    assert run_tests(src, "f", [TestCase((1.0,), 1.0)]) == [True]
    assert values_equal(1, 1.0) and not values_equal(True, 1)


def test_stops_after_repeated_timeouts():
    import time

    src = "def f(x):\n    while True:\n        pass\n"
    t0 = time.time()
    res = run_function(src, "f", [(i,) for i in range(20)], per_input_timeout_s=0.2,
                       max_timeouts=2)
    assert [r.kind for r in res] == ["timeout"] * 20
    assert time.time() - t0 < 3.0  # 2 real timeouts, not 20
