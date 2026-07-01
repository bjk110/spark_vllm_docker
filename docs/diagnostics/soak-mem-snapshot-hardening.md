# Soak memory-snapshot hardening (H1Z-B1AF)

Helper: [`scripts/diag/soak_mem_snapshot.py`](../../scripts/diag/soak_mem_snapshot.py)
Tests: [`scripts/diag/tests/test_soak_mem_snapshot.py`](../../scripts/diag/tests/test_soak_mem_snapshot.py)

## Why

During **H1Z-B1AC Stage C** (2026-07-01) a concurrency-4 soak gate read per-node
`MemAvailable` over SSH and computed roughly `float(ms["spark01"].split()[0])`. A
transient **empty** SSH result from spark01 raised inside `float(...)` and was caught
into `m1 = 0`, so the gate concluded memory was below the concurrency-4 threshold and
skipped Stage C as a low-memory "safety bound". Live verification showed ~26/28 GiB
available, and a bounded manual concurrency-4 observation completed 8/8.

The defect: **unavailable telemetry was silently converted into a numeric zero**, which
is indistinguishable from a genuine reading of 0.

## What changed

The hardened helper makes every snapshot outcome explicit and never turns
unavailable/invalid data into a number. It performs no I/O itself — the SSH runner and
the sleep function are injected, so it is fully unit-testable offline.

### Snapshot statuses (`SnapshotStatus`)

| Status | Meaning | Retryable | Value populated |
|---|---|---|---|
| `OK` | valid positive `MemAvailable` | no | yes (GiB) |
| `GENUINE_ZERO` | parsed a real numeric `0` | no | `0.0` |
| `EMPTY_OUTPUT` | empty / whitespace-only stdout | yes | none |
| `COMMAND_ERROR` | remote command nonzero exit | yes | none |
| `SSH_ERROR` | SSH transport failure | yes | none |
| `TIMEOUT` | snapshot timed out | yes | none |
| `PARSE_ERROR` | malformed / nonnumeric / NaN / inf / negative | no | none |
| `MISSING_RESULT` | no result for a required node (gate level) | — | none |
| `PARTIAL_RESULT` | some nodes valid, some not (gate level) | — | — |

A `GENUINE_ZERO` is deliberately distinct from every unavailable status: a real zero is
a safety failure, while unavailable data is "unknown".

### Parser rules (`parse_mem_available_kib`)

Trim; reject empty/whitespace-only; accept either a full `/proc/meminfo` dump (locates
the `MemAvailable:` line) or exactly one bare integer token (kB); reject missing fields,
extra tokens, nonnumeric tokens, NaN, infinity, decimals, and negatives; distinguish a
genuine `0`; integer and locale-independent; never indexes an empty split. Metric:
`MemAvailable` from `/proc/meminfo` (kB → GiB); canonical remote command
`awk '/^MemAvailable:/{print $2}' /proc/meminfo`.

### Bounded retry (`snapshot_node`)

Default max 3 attempts with a short injected delay between attempts (no unbounded loop,
no background retry). Only transient statuses (`EMPTY_OUTPUT`, `TIMEOUT`, `SSH_ERROR`,
`COMMAND_ERROR`) are retried; `OK`, `GENUINE_ZERO`, and `PARSE_ERROR` return immediately
(a genuine zero and stable malformed output are never retried). Every attempt is recorded
in `attempt_log`. Retry count and delay are parameters, and the sleep function is
injected, so retries are deterministic in tests (no real sleeping).

### Stage C gate (`evaluate_stage_c_gate`)

Safety-first. Runs Stage C only when **every required node is `OK` and at or above the
threshold**. Otherwise:

| Situation | Decision | Stage C classification |
|---|---|---|
| all nodes OK, all ≥ threshold | `RUN` | `CONCURRENCY4_ATTEMPTED` |
| all nodes OK, some below threshold | `SKIP_LOW_MEMORY` | `CONCURRENCY4_NOT_ATTEMPTED_LOW_MEMORY` |
| any node genuine zero | `SKIP_GENUINE_ZERO` | `CONCURRENCY4_NOT_ATTEMPTED_GENUINE_ZERO` |
| all required nodes unavailable after retries | `SKIP_DATA_UNAVAILABLE` | `CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE` |
| some valid, some unavailable/missing | `SKIP_PARTIAL` | `CONCURRENCY4_NOT_ATTEMPTED_DATA_UNAVAILABLE` |

Uncertain memory data never authorizes higher concurrency, and successful node values are
always preserved (never discarded, never zeroed). The gate makes it explicit whether
Stage C was skipped for **low memory**, a **genuine zero**, **unavailable telemetry**, or
partial data.

### Logging

`format_gate_log` emits one concise structured line (decision, classification, threshold,
ok/zero/unavailable nodes, reason). It contains no credentials, tokens, or environment
values. Per-attempt lines are recorded in each snapshot's `attempt_log`.

## Regression coverage

`test_regression_b1ac_stage_c_empty_spark01` reproduces the exact H1Z-B1AC failure —
spark01 empty output, spark02 ~28 GiB — and asserts the corrected behavior: spark01 is
classified data-unavailable after the retry sequence (never zero), spark02's real value
is preserved, and Stage C is skipped with a `DATA_UNAVAILABLE` reason rather than a false
low-memory verdict. `test_regression_b1ac_later_attempt_recovers` covers a later attempt
succeeding.

## Running the tests (offline)

```
python3 scripts/diag/tests/test_soak_mem_snapshot.py
```

No spark01/spark02 access; no real sleeping. 33 tests, stdlib only (pytest optional).

## Scope note

The H1Z-B1AC driver that carried the defect was a throwaway per-run script (not tracked
in this repository). This module is the tracked, reusable, tested replacement for the
memory-snapshot + concurrency-gate logic that future soak drivers should import. The
existing tracked long-soak driver
[`scripts/diag/dsv4_mtp1_fullgraph_long_soak.py`](../../scripts/diag/dsv4_mtp1_fullgraph_long_soak.py)
reads local `/proc/meminfo` (not per-node SSH) and already guards `None`, so it has no
concurrency-4 SSH gate to migrate.
