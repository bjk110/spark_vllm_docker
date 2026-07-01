#!/usr/bin/env python3
"""Offline unit tests for scripts/diag/soak_mem_snapshot.py.

These tests never contact spark01 or spark02: the SSH runner and the sleep
function are injected. Real sleeping never occurs.

Run: python3 scripts/diag/tests/test_soak_mem_snapshot.py
  or: python3 -m pytest scripts/diag/tests/test_soak_mem_snapshot.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soak_mem_snapshot import (  # noqa: E402
    GateDecision,
    MemParseError,
    MemSnapshot,
    RunResult,
    SSHTransportError,
    SnapshotStatus,
    EmptyOutputError,
    evaluate_stage_c_gate,
    format_gate_log,
    parse_mem_available_kib,
    snapshot_node,
)

GIB = 1048576  # kB per GiB


# ---------------------------------------------------------------------------
# Test doubles (dependency injection): no real SSH, no real sleep.
# ---------------------------------------------------------------------------

class FakeRunner:
    """Return queued RunResults or raise queued exceptions, one per attempt."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def __call__(self, node):
        self.calls += 1
        item = self._script.pop(0) if self._script else self._script_default()
        if isinstance(item, BaseException):
            raise item
        return item

    def _script_default(self):
        raise AssertionError("FakeRunner called more times than scripted")


class FakeSleep:
    """Record delays instead of sleeping."""

    def __init__(self):
        self.delays = []

    def __call__(self, d):
        self.delays.append(d)


def _snap(node, script, **kw):
    kw.setdefault("delay", 0.5)
    sleep = FakeSleep()
    runner = FakeRunner(script)
    result = snapshot_node(node, runner, sleep_fn=sleep, clock=lambda: 123.0, **kw)
    return result, runner, sleep


def ok_out(gib):
    return RunResult(stdout=f"{int(gib * GIB)}\n", exit_code=0)


# ---------------------------------------------------------------------------
# Parser rules
# ---------------------------------------------------------------------------

def test_parse_valid_bare_integer():
    assert parse_mem_available_kib("27000000\n") == 27000000

def test_parse_valid_meminfo_dump():
    dump = "MemTotal:      100 kB\nMemAvailable:   29360128 kB\nSwapFree: 0 kB\n"
    assert parse_mem_available_kib(dump) == 29360128

def test_parse_trailing_newline_ok():
    assert parse_mem_available_kib("28000000\n") == 28000000

def test_parse_extra_whitespace_ok():
    assert parse_mem_available_kib("   28000000   \n") == 28000000

def test_parse_genuine_zero():
    assert parse_mem_available_kib("0") == 0

def test_parse_empty_raises():
    for bad in ("", "   ", "\n", "\t \n"):
        try:
            parse_mem_available_kib(bad)
            assert False, f"expected EmptyOutputError for {bad!r}"
        except EmptyOutputError:
            pass

def test_parse_malformed_token():
    for bad in ("abc", "12x", "1.2.3", "27000000 extra"):
        try:
            parse_mem_available_kib(bad)
            assert False, f"expected MemParseError for {bad!r}"
        except MemParseError:
            pass

def test_parse_nan_infinity_negative_decimal():
    for bad in ("nan", "NaN", "inf", "Infinity", "-1", "-27000000", "27000000.5"):
        try:
            parse_mem_available_kib(bad)
            assert False, f"expected MemParseError for {bad!r}"
        except MemParseError:
            pass

def test_parse_meminfo_present_but_malformed():
    try:
        parse_mem_available_kib("MemAvailable: notanumber kB")
        assert False
    except MemParseError:
        pass


# ---------------------------------------------------------------------------
# snapshot_node: single-attempt statuses
# ---------------------------------------------------------------------------

def test_snapshot_valid_spark01():
    r, runner, sleep = _snap("spark01", [ok_out(26.0)])
    assert r.status is SnapshotStatus.OK
    assert abs(r.mem_available_gib - 26.0) < 1e-6
    assert r.attempts == 1
    assert runner.calls == 1
    assert sleep.delays == []

def test_snapshot_valid_spark02():
    r, _, _ = _snap("spark02", [ok_out(28.0)])
    assert r.status is SnapshotStatus.OK and abs(r.mem_available_gib - 28.0) < 1e-6

def test_snapshot_genuine_zero_not_retried():
    r, runner, sleep = _snap("spark01", [RunResult(stdout="0\n", exit_code=0)])
    assert r.status is SnapshotStatus.GENUINE_ZERO
    assert r.mem_available_gib == 0.0
    assert runner.calls == 1  # NOT retried
    assert sleep.delays == []

def test_snapshot_empty_stdout_exhausts_retries():
    r, runner, sleep = _snap("spark01", [RunResult(stdout="", exit_code=0)] * 3)
    assert r.status is SnapshotStatus.EMPTY_OUTPUT
    assert r.mem_available_gib is None            # never becomes zero
    assert r.attempts == 3 and runner.calls == 3
    assert sleep.delays == [0.5, 0.5]             # slept between attempts only

def test_snapshot_whitespace_only_stdout():
    r, _, _ = _snap("spark01", [RunResult(stdout="   \n", exit_code=0)] * 3)
    assert r.status is SnapshotStatus.EMPTY_OUTPUT and r.mem_available_gib is None

def test_snapshot_command_error_nonzero_exit():
    r, runner, _ = _snap("spark01",
                         [RunResult(stdout="", stderr="awk: cannot open", exit_code=2)] * 3)
    assert r.status is SnapshotStatus.COMMAND_ERROR
    assert r.exit_code == 2 and "awk" in r.raw_stderr and r.mem_available_gib is None

def test_snapshot_ssh_exception():
    r, runner, _ = _snap("spark01", [SSHTransportError("connection reset")] * 3)
    assert r.status is SnapshotStatus.SSH_ERROR and r.mem_available_gib is None
    assert r.attempts == 3

def test_snapshot_timeout():
    r, runner, sleep = _snap("spark01", [TimeoutError("deadline")] * 3)
    assert r.status is SnapshotStatus.TIMEOUT and r.mem_available_gib is None
    assert sleep.delays == [0.5, 0.5]

def test_snapshot_malformed_not_retried():
    r, runner, sleep = _snap("spark01", [RunResult(stdout="garbage", exit_code=0)])
    assert r.status is SnapshotStatus.PARSE_ERROR
    assert runner.calls == 1                      # malformed stable output: NO retry
    assert sleep.delays == [] and r.mem_available_gib is None

def test_snapshot_negative_is_parse_error_not_zero():
    r, _, _ = _snap("spark01", [RunResult(stdout="-5\n", exit_code=0)])
    assert r.status is SnapshotStatus.PARSE_ERROR and r.mem_available_gib is None


# ---------------------------------------------------------------------------
# snapshot_node: retry sequences
# ---------------------------------------------------------------------------

def test_retry_first_empty_then_valid():
    r, runner, sleep = _snap("spark01",
                            [RunResult(stdout="", exit_code=0), ok_out(27.0)])
    assert r.status is SnapshotStatus.OK
    assert abs(r.mem_available_gib - 27.0) < 1e-6
    assert r.attempts == 2 and runner.calls == 2
    assert sleep.delays == [0.5]

def test_retry_two_timeouts_then_valid():
    r, runner, sleep = _snap("spark01",
                            [TimeoutError("t1"), TimeoutError("t2"), ok_out(28.0)])
    assert r.status is SnapshotStatus.OK and r.attempts == 3
    assert sleep.delays == [0.5, 0.5]

def test_retry_all_empty_stays_unavailable():
    r, _, _ = _snap("spark01", [RunResult(stdout="", exit_code=0)] * 3)
    assert r.status is SnapshotStatus.EMPTY_OUTPUT and r.mem_available_gib is None

def test_retry_count_respected():
    r, runner, sleep = _snap("spark01", [TimeoutError("t")] * 5, retries=5)
    assert r.attempts == 5 and runner.calls == 5
    assert sleep.delays == [0.5, 0.5, 0.5, 0.5]   # retries-1 sleeps

def test_retry_delay_abstraction_invoked():
    sleep = FakeSleep()
    snapshot_node("spark01", FakeRunner([RunResult(stdout="", exit_code=0)] * 3),
                  retries=3, delay=1.25, sleep_fn=sleep, clock=lambda: 0.0)
    assert sleep.delays == [1.25, 1.25]           # exact injected delay, no real sleep


# ---------------------------------------------------------------------------
# Stage C gate
# ---------------------------------------------------------------------------

def _ok(node, gib):
    return MemSnapshot(node=node, status=SnapshotStatus.OK, mem_available_gib=gib)

def _unavail(node, status):
    return MemSnapshot(node=node, status=status)


def test_gate_both_valid_above_threshold_runs():
    g = evaluate_stage_c_gate({"spark01": _ok("spark01", 26.0),
                               "spark02": _ok("spark02", 28.0)},
                              threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.RUN and g.run_stage_c
    assert g.stage_c_classification == "CONCURRENCY4_ATTEMPTED"

def test_gate_one_below_threshold_low_memory():
    g = evaluate_stage_c_gate({"spark01": _ok("spark01", 12.0),
                               "spark02": _ok("spark02", 28.0)},
                              threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_LOW_MEMORY and not g.run_stage_c
    assert g.below_threshold == {"spark01": 12.0}
    assert g.stage_c_classification == "CONCURRENCY4_NOT_ATTEMPTED_LOW_MEMORY"

def test_gate_genuine_zero_is_safety_failure_not_unavailable():
    g = evaluate_stage_c_gate(
        {"spark01": MemSnapshot("spark01", SnapshotStatus.GENUINE_ZERO, 0.0),
         "spark02": _ok("spark02", 28.0)},
        threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_GENUINE_ZERO
    assert g.stage_c_classification == "CONCURRENCY4_NOT_ATTEMPTED_GENUINE_ZERO"

def test_gate_all_unavailable_is_data_unavailable_not_zero():
    g = evaluate_stage_c_gate(
        {"spark01": _unavail("spark01", SnapshotStatus.EMPTY_OUTPUT),
         "spark02": _unavail("spark02", SnapshotStatus.TIMEOUT)},
        threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_DATA_UNAVAILABLE and not g.run_stage_c
    assert g.stage_c_classification == "CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE"
    assert "spark01" in g.unavailable_nodes and "spark02" in g.unavailable_nodes

def test_gate_partial_preserves_valid_value():
    g = evaluate_stage_c_gate(
        {"spark01": _unavail("spark01", SnapshotStatus.EMPTY_OUTPUT),
         "spark02": _ok("spark02", 28.0)},
        threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_PARTIAL
    assert g.ok_nodes == {"spark02": 28.0}          # successful data NOT discarded
    assert g.unavailable_nodes == {"spark01": "EMPTY_OUTPUT"}
    assert g.stage_c_classification == "CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE"

def test_gate_missing_required_node():
    g = evaluate_stage_c_gate({"spark02": _ok("spark02", 28.0)},
                              threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_PARTIAL
    assert g.unavailable_nodes == {"spark01": "MISSING_RESULT"}

def test_gate_log_has_no_secrets_and_states_reason():
    g = evaluate_stage_c_gate(
        {"spark01": _unavail("spark01", SnapshotStatus.EMPTY_OUTPUT),
         "spark02": _ok("spark02", 28.0)},
        threshold_gib=20, required_nodes=["spark01", "spark02"])
    line = format_gate_log(g)
    assert "DATA_UNAVAILABLE" in line and "spark02" in line
    for secret in ("PASSWORD", "TOKEN", "ssh-rsa", "BEGIN OPENSSH"):
        assert secret not in line


# ---------------------------------------------------------------------------
# Regression: the exact H1Z-B1AC Stage C failure
# ---------------------------------------------------------------------------

def test_regression_b1ac_stage_c_empty_spark01():
    """H1Z-B1AC Stage C: spark01 returned empty output, spark02 ~28 GiB.

    The old driver turned spark01 into MemAvailable 0 and skipped Stage C as a
    (false) low-memory safety bound. The hardened path must instead classify
    spark01 as data-unavailable after bounded retries, preserve spark02's real
    value, and skip with a DATA_UNAVAILABLE reason — never a zero / low-memory
    verdict.
    """
    s01, runner01, sleep01 = _snap("spark01", [RunResult(stdout="", exit_code=0)] * 3)
    s02, runner02, _ = _snap("spark02", [ok_out(28.3)])

    assert s01.status is SnapshotStatus.EMPTY_OUTPUT
    assert s01.mem_available_gib is None            # NOT zero
    assert s01.attempts == 3 and sleep01.delays == [0.5, 0.5]
    assert s02.status is SnapshotStatus.OK and abs(s02.mem_available_gib - 28.3) < 1e-6

    g = evaluate_stage_c_gate({"spark01": s01, "spark02": s02},
                              threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.SKIP_PARTIAL, g.decision
    assert not g.run_stage_c
    assert g.stage_c_classification == "CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE"
    # spark02 real value preserved (kB round-trip, compare with tolerance)
    assert list(g.ok_nodes) == ["spark02"], g.ok_nodes
    assert abs(g.ok_nodes["spark02"] - 28.3) < 1e-3, g.ok_nodes
    assert g.below_threshold == {}                  # NOT a low-memory verdict
    assert g.zero_nodes == []                       # NOT a genuine zero


def test_regression_b1ac_later_attempt_recovers():
    """If spark01's telemetry recovers on a later attempt, Stage C may run."""
    s01, _, sleep01 = _snap("spark01",
                           [RunResult(stdout="", exit_code=0), ok_out(26.4)])
    s02, _, _ = _snap("spark02", [ok_out(28.3)])
    assert s01.status is SnapshotStatus.OK and s01.attempts == 2
    g = evaluate_stage_c_gate({"spark01": s01, "spark02": s02},
                              threshold_gib=20, required_nodes=["spark01", "spark02"])
    assert g.decision is GateDecision.RUN and g.run_stage_c


# ---------------------------------------------------------------------------
# Runner (stdlib, no pytest dependency)
# ---------------------------------------------------------------------------

def run_tests():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
