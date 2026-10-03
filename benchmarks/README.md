# Benchmark and recipe evidence

This directory distinguishes **canonical recipe evidence** from historical benchmark output.

- [`evidence/`](evidence/README.md) — authority for the 14 production/rollback/production-qualified and
  validated non-production presets in catalog sections 1–2. It contains a per-recipe evidence map,
  small retained decision summaries, immutable hashes, and the external-archive ledger.
- [`llama-benchy/`](llama-benchy/README.md) — tracked historical benchmark summaries. They remain contextual
  records and must not be treated as recommended runtime settings by filename alone.

Raw benchmark runs, repeated dry runs, logs, profiler captures, and superseded experiment trees are
not stored in the active Git working tree. They are preserved in an external archive whose file-level
SHA-256 ledger is tracked at [`evidence/archive-manifest.sha256`](evidence/archive-manifest.sha256).

Interpreted model-serving results and operational status remain under [`docs/`](../docs/README.md). For the
interpreted unholy-fusion / DSV4 comparison, see
[`docs/unholy-fusion-benchmark.md`](../docs/unholy-fusion-benchmark.md).
