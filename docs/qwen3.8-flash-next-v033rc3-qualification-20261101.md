# Qwen3.8 Flash Next Uncensored FP8 — vLLM v0.33.0rc3 qualification

| Field | Value |
|---|---|
| Status | **Qualification record — bounded pass, runtime not adopted** |
| Date | 2026-11-01 UTC |
| Model revision | `8bebbc05cbe65099f37c520865463c1377d8050d` |
| Existing runtime | NGC 26.09, vLLM `e77daef89e18e08321ae7b8b24827eedd5fe8673` |
| Evaluated runtime | vLLM `v0.33.0rc3`, local image ID `sha256:d4c2ace4e497…` |
| Runtime contract | dual-Spark Ray TP2/EP2, eager, MTP k=2, FP8 KV 8 GiB/rank, 262,144 context |
| Decision | Keep the existing immutable e77 recipe; do not publish or activate the RC3 image |

## Scope

This evaluation asked whether v0.33.0rc3 should replace the established uncensored Qwen3.8 e77
runtime. The model bytes and serving contract were held constant. The RC3 image was built from the
pinned source archive on the existing CUDA 13.4/Torch stack, deployed to both nodes under separate
container names, tested, stopped, and followed by an e77 restoration check.

Build provenance and the exact Dockerfile are retained under
[`dockerfiles/experimental/`](../dockerfiles/experimental/qwen38-v033rc3-PROVENANCE.md). They do not
constitute a supported build or image publication.

## Results

Correctness and bounded stability passed:

- canonical and semantic probes;
- required tool call and JSON Schema response;
- synchronized C2 execution;
- 30K and 120K retrieval;
- two simultaneous exact 262,112-token retrievals with HTTP 200 and `KESTREL-74291`;
- no OOM or container restart during the bounded run.

The operational comparison did not justify replacement:

- 512/C2 decode: -8.86% versus e77;
- 2048/C2 decode: -13.81% versus e77;
- KV capacity: 911,141 versus 943,718 tokens (-3.45%);
- cold start: 2,239.3 versus 1,663.7 seconds;
- exact 262,112 C2 makespan: 186.48 versus 175.33 seconds (+6.36%);
- inherited optional `b12x_fp6` plugin entry point logged a non-fatal missing import.

The C1 decode rows improved and 2048/C2 TTFR improved, but those gains did not offset the C2 decode,
capacity, startup, and unresolved-plugin results for this deployment.

## Decision boundary

The existing immutable e77 recipe remains authoritative. This record does not change its preset,
Compose overlay, source manifest, startup policy, or automatic-start state. The local RC3 image may be
kept for future regression work, but it is not a repository serving recipe and must not be inferred
from the experimental Dockerfile alone.

Compact evidence:
[`benchmarks/evidence/retained/qwen38-v033rc3-20261101/report.md`](../benchmarks/evidence/retained/qwen38-v033rc3-20261101/report.md).
