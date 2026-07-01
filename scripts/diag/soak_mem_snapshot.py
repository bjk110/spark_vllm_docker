#!/usr/bin/env python3
"""Hardened per-node memory-snapshot handling for concurrency soak gating.

Root cause (H1Z-B1AC Stage C, 2026-07-01)
-----------------------------------------
A bespoke soak driver read per-node ``MemAvailable`` over SSH and gated the
concurrency-4 stage with roughly::

    ms = mem_snapshot()                       # {"spark01": "<awk output>", ...}
    try:
        m1 = float(ms["spark01"].split()[0])  # <-- empty string -> exception
        m2 = float(ms["spark02"].split()[0])
    except Exception:
        m1 = m2 = 0                            # <-- unavailable data becomes 0
    if m1 >= 20 and m2 >= 20 and health == "200":
        stage_c(...)
    else:
        skip("safety_bound")                  # <-- reported as low memory

A transient EMPTY ssh result from spark01 raised inside ``float(...)`` and was
converted to ``0``, so the gate concluded memory was below the concurrency-4
threshold and skipped Stage C. Live verification showed ~26/28 GiB available and
a bounded manual concurrency-4 observation completed 8/8.

This module makes snapshot results EXPLICIT and never turns unavailable or
invalid telemetry into a numeric zero. A genuine numeric zero stays
distinguishable from "we could not read the value". Safety-first gating is
preserved: uncertain memory data never authorizes higher concurrency.

Metric definition
-----------------
``MemAvailable`` is read from ``/proc/meminfo`` (kB) and converted to GiB. The
canonical remote command is :data:`MEMINFO_CMD`. The parser also accepts a full
``/proc/meminfo`` dump (it locates the ``MemAvailable:`` line).

This module performs no I/O of its own: the SSH runner and the sleep function are
injected, so it is fully unit-testable offline (see
``scripts/diag/tests/test_soak_mem_snapshot.py``).
"""

from __future__ import annotations

import enum
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

KIB_PER_GIB = 1048576  # /proc/meminfo reports kB; 1 GiB == 1048576 kB

# Canonical, deterministic, locale-independent remote command: emit only the
# MemAvailable value in kB as a single integer token.
MEMINFO_CMD = "awk '/^MemAvailable:/{print $2}' /proc/meminfo"

_MEMINFO_LINE_RE = re.compile(r"^MemAvailable:\s+(\d+)\s*kB\s*$", re.MULTILINE)
_INT_TOKEN_RE = re.compile(r"^\d+$")


class SnapshotStatus(enum.Enum):
    """Explicit outcome of a single-node memory snapshot."""

    OK = "OK"                       # parsed a valid positive value
    GENUINE_ZERO = "GENUINE_ZERO"   # parsed a real numeric zero (data present)
    EMPTY_OUTPUT = "EMPTY_OUTPUT"   # stdout empty / whitespace-only
    COMMAND_ERROR = "COMMAND_ERROR"  # remote command nonzero exit
    SSH_ERROR = "SSH_ERROR"         # SSH transport failure
    TIMEOUT = "TIMEOUT"             # snapshot timed out
    PARSE_ERROR = "PARSE_ERROR"     # malformed / nonnumeric / NaN / inf / negative
    MISSING_RESULT = "MISSING_RESULT"   # no result for a required node
    PARTIAL_RESULT = "PARTIAL_RESULT"   # gate-level: some nodes ok, some not


# Transient statuses are eligible for bounded retry. GENUINE_ZERO, OK,
# PARSE_ERROR and MISSING_RESULT are stable and must NOT be retried.
_TRANSIENT = frozenset({
    SnapshotStatus.EMPTY_OUTPUT,
    SnapshotStatus.TIMEOUT,
    SnapshotStatus.SSH_ERROR,
    SnapshotStatus.COMMAND_ERROR,
})

# Statuses that mean "we hold a trustworthy reading" (value present, incl. real 0).
_AVAILABLE = frozenset({SnapshotStatus.OK, SnapshotStatus.GENUINE_ZERO})


class SSHTransportError(Exception):
    """Raised by a runner when SSH transport itself fails (connect/auth/reset)."""


@dataclass
class RunResult:
    """Result of one remote command execution (what a runner returns)."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class MemSnapshot:
    """Explicit, structured result of a single-node memory snapshot.

    ``mem_available_gib`` is populated ONLY for ``OK`` (positive) and
    ``GENUINE_ZERO`` (exactly ``0.0``). For every unavailable/invalid status it
    stays ``None`` — unavailable data is never represented as a number.
    """

    node: str
    status: SnapshotStatus
    mem_available_gib: Optional[float] = None
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: Optional[int] = None
    attempts: int = 0
    timestamp: Optional[float] = None
    error: str = ""
    attempt_log: List[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.status is SnapshotStatus.OK

    @property
    def is_available(self) -> bool:
        """True when a trustworthy reading exists (positive value OR genuine 0)."""
        return self.status in _AVAILABLE


class EmptyOutputError(ValueError):
    """Stdout was empty or whitespace-only."""


class MemParseError(ValueError):
    """Stdout was non-empty but not a valid MemAvailable reading."""


def parse_mem_available_kib(stdout: str) -> int:
    """Parse ``MemAvailable`` (kB) from a snapshot's stdout.

    Rules: trim; reject empty/whitespace-only (:class:`EmptyOutputError`);
    accept either a full ``/proc/meminfo`` dump or a single integer token;
    reject missing fields, extra tokens, nonnumeric tokens, NaN, infinity, and
    negatives (:class:`MemParseError`); distinguish a genuine ``0``. Integer,
    locale-independent; never indexes an empty split.
    """
    if stdout is None:
        raise EmptyOutputError("stdout is None")
    text = stdout.strip()
    if not text:
        raise EmptyOutputError("empty or whitespace-only stdout")

    # Full /proc/meminfo dump: locate the MemAvailable line explicitly.
    if "MemAvailable:" in text:
        m = _MEMINFO_LINE_RE.search(text)
        if not m:
            raise MemParseError("MemAvailable line present but malformed")
        return int(m.group(1))

    # Otherwise expect exactly one bare integer token (kB). Validate full shape.
    tokens = text.split()
    if len(tokens) != 1:
        raise MemParseError(f"expected one integer token, got {len(tokens)}: {tokens!r}")
    tok = tokens[0]
    if not _INT_TOKEN_RE.match(tok):
        # Catches malformed tokens, signs (negative), NaN, inf, decimals, etc.
        raise MemParseError(f"nonnumeric or invalid MemAvailable token: {tok!r}")
    return int(tok)


def snapshot_node(
    node: str,
    runner: Callable[[str], RunResult],
    *,
    retries: int = 3,
    delay: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Optional[Callable[[str], None]] = None,
) -> MemSnapshot:
    """Take a hardened single-node memory snapshot with bounded retry.

    ``runner(node)`` returns a :class:`RunResult`, or raises ``TimeoutError``
    (-> ``TIMEOUT``) or :class:`SSHTransportError` (-> ``SSH_ERROR``). A nonzero
    ``exit_code`` -> ``COMMAND_ERROR``. Transient statuses are retried up to
    ``retries`` attempts with ``delay`` seconds between attempts (via the
    injected ``sleep_fn``); ``GENUINE_ZERO``, ``OK`` and ``PARSE_ERROR`` return
    immediately. Every attempt is recorded in ``attempt_log``. Never returns a
    numeric value for an unavailable/invalid status.
    """
    if retries < 1:
        raise ValueError("retries must be >= 1")

    attempt_log: List[str] = []
    last: Optional[MemSnapshot] = None

    for attempt in range(1, retries + 1):
        status: SnapshotStatus
        stdout = stderr = ""
        exit_code: Optional[int] = None
        mem_gib: Optional[float] = None
        error = ""

        try:
            rr = runner(node)
        except TimeoutError as exc:
            status, error = SnapshotStatus.TIMEOUT, f"timeout: {exc}"
        except SSHTransportError as exc:
            status, error = SnapshotStatus.SSH_ERROR, f"ssh transport: {exc}"
        else:
            stdout, stderr, exit_code = rr.stdout, rr.stderr, rr.exit_code
            if rr.exit_code != 0:
                status = SnapshotStatus.COMMAND_ERROR
                error = f"exit={rr.exit_code} stderr={stderr.strip()[:200]!r}"
            else:
                try:
                    kib = parse_mem_available_kib(rr.stdout)
                except EmptyOutputError as exc:
                    status, error = SnapshotStatus.EMPTY_OUTPUT, str(exc)
                except MemParseError as exc:
                    status, error = SnapshotStatus.PARSE_ERROR, str(exc)
                else:
                    if kib == 0:
                        status, mem_gib = SnapshotStatus.GENUINE_ZERO, 0.0
                    else:
                        status, mem_gib = SnapshotStatus.OK, kib / KIB_PER_GIB

        will_retry = status in _TRANSIENT and attempt < retries
        line = (f"node={node} attempt={attempt}/{retries} status={status.value}"
                + (f" mem={mem_gib:.1f}GiB" if mem_gib is not None else "")
                + (f" exit={exit_code}" if exit_code is not None else "")
                + (f" err={error}" if error else "")
                + (" retrying" if will_retry else ""))
        attempt_log.append(line)
        if log is not None:
            log(line)

        last = MemSnapshot(
            node=node, status=status, mem_available_gib=mem_gib,
            raw_stdout=stdout, raw_stderr=stderr, exit_code=exit_code,
            attempts=attempt, timestamp=clock(), error=error,
            attempt_log=list(attempt_log),
        )

        if status not in _TRANSIENT:
            # OK / GENUINE_ZERO / PARSE_ERROR: stable, do not retry.
            return last
        if attempt < retries:
            sleep_fn(delay)

    assert last is not None
    return last


class GateDecision(enum.Enum):
    RUN = "RUN"
    SKIP_LOW_MEMORY = "SKIP_LOW_MEMORY"
    SKIP_GENUINE_ZERO = "SKIP_GENUINE_ZERO"
    SKIP_DATA_UNAVAILABLE = "SKIP_DATA_UNAVAILABLE"
    SKIP_PARTIAL = "SKIP_PARTIAL"


_STAGE_C_CLASS = {
    GateDecision.RUN: "CONCURRENCY4_ATTEMPTED",
    GateDecision.SKIP_LOW_MEMORY: "CONCURRENCY4_NOT_ATTEMPTED_LOW_MEMORY",
    GateDecision.SKIP_GENUINE_ZERO: "CONCURRENCY4_NOT_ATTEMPTED_GENUINE_ZERO",
    GateDecision.SKIP_DATA_UNAVAILABLE: "CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE",
    GateDecision.SKIP_PARTIAL: "CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE",
}


@dataclass
class StageCGateResult:
    decision: GateDecision
    reason: str
    threshold_gib: float
    ok_nodes: Dict[str, float] = field(default_factory=dict)
    zero_nodes: List[str] = field(default_factory=list)
    unavailable_nodes: Dict[str, str] = field(default_factory=dict)  # node -> status
    below_threshold: Dict[str, float] = field(default_factory=dict)

    @property
    def run_stage_c(self) -> bool:
        return self.decision is GateDecision.RUN

    @property
    def stage_c_classification(self) -> str:
        return _STAGE_C_CLASS[self.decision]


def evaluate_stage_c_gate(
    snapshots: Dict[str, MemSnapshot],
    *,
    threshold_gib: float,
    required_nodes: Sequence[str],
) -> StageCGateResult:
    """Safety-first concurrency-4 gate over per-node snapshots.

    Runs Stage C ONLY when every required node reports ``OK`` at or above
    ``threshold_gib``. A genuine zero is a real safety failure
    (``SKIP_GENUINE_ZERO``). Any node that is missing or unavailable after
    bounded retries yields ``SKIP_DATA_UNAVAILABLE`` (all unavailable) or
    ``SKIP_PARTIAL`` (some valid, some not) — never interpreted as zero memory.
    Successful node values are always preserved in the result.
    """
    ok_nodes: Dict[str, float] = {}
    zero_nodes: List[str] = []
    unavailable: Dict[str, str] = {}

    for node in required_nodes:
        snap = snapshots.get(node)
        if snap is None:
            unavailable[node] = SnapshotStatus.MISSING_RESULT.value
        elif snap.status is SnapshotStatus.OK:
            ok_nodes[node] = float(snap.mem_available_gib)
        elif snap.status is SnapshotStatus.GENUINE_ZERO:
            zero_nodes.append(node)
        else:
            unavailable[node] = snap.status.value

    below = {n: v for n, v in ok_nodes.items() if v < threshold_gib}

    if zero_nodes:
        decision = GateDecision.SKIP_GENUINE_ZERO
        reason = f"genuine zero MemAvailable on {zero_nodes} (real safety failure)"
    elif unavailable:
        if len(unavailable) == len(required_nodes):
            decision = GateDecision.SKIP_DATA_UNAVAILABLE
            reason = f"memory telemetry unavailable after retries: {unavailable}"
        else:
            decision = GateDecision.SKIP_PARTIAL
            reason = (f"partial telemetry: ok={ok_nodes} unavailable={unavailable} "
                      "(data unavailable, not low memory)")
    elif below:
        decision = GateDecision.SKIP_LOW_MEMORY
        reason = f"MemAvailable below {threshold_gib} GiB on {below}"
    else:
        decision = GateDecision.RUN
        reason = f"all nodes OK and >= {threshold_gib} GiB: {ok_nodes}"

    return StageCGateResult(
        decision=decision, reason=reason, threshold_gib=threshold_gib,
        ok_nodes=ok_nodes, zero_nodes=zero_nodes,
        unavailable_nodes=unavailable, below_threshold=below,
    )


def format_gate_log(result: StageCGateResult) -> str:
    """One concise structured log line for the final Stage C gate decision.

    Contains no credentials, tokens, or environment values.
    """
    return (f"stage_c_gate decision={result.decision.value} "
            f"classification={result.stage_c_classification} "
            f"threshold_gib={result.threshold_gib} ok={result.ok_nodes} "
            f"zero={result.zero_nodes} unavailable={result.unavailable_nodes} "
            f"reason={result.reason!r}")
