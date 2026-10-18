# Qwen3.8 NGC 26.09/e77 Ray vs MP matched-C2 qualification — 2026-10-18

## Classification

`MATCHED_C2_PERFORMANCE_PASS`

The Ray and MP arms used the same Qwen3.8 model, TP=2, MTP k=2, 262,144-token limit,
`MAX_NUM_SEQS=2`, BT8K, 8 GiB KV/rank, eager execution, source overlay, RoCE interfaces, and
request matrix. The intended variable was the distributed executor/control plane; the Ray child image
adds `ray[cgraph]==2.59.0` to the immutable MP parent.

## Functional and stability gates

- Ray two-node readiness: PASS.
- Deterministic short/tool suite: PASS.
- MTP draft and accepted-token activity: PASS.
- 262,112-token semantic retrieval: PASS.
- Synchronized C2 pairs: 9/9 PASS, with server running peak 2 in every pair.
- Container restart/OOM and CUDA/NCCL/Xid/traceback: zero.
- MP clean-boot restoration: PASS; health and model endpoint returned HTTP 200.

The first fatal scanner classified four startup lines containing `Failed to import Triton kernels`
as EngineCore fatal. The missing optional module was `triton_kernels.matmul_ogs`; the same warning
exists on the MP path and did not coincide with process death, restart, OOM, CUDA/NCCL/Xid, health
failure, or request failure. It is retained as an adjudicated scanner false positive, not silently
deleted.

## Matched performance

Predeclared gates required Ray aggregate decode ≥90% of MP, mean request E2E ≤115% of MP, and mean
request TTFT ≤120% of MP for every workload.

| Workload | MP aggregate decode tok/s | Ray aggregate decode tok/s | Ray/MP decode | MP mean E2E s | Ray mean E2E s | Ray/MP E2E | MP mean TTFT s | Ray mean TTFT s | Ray/MP TTFT | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prose | 41.034 | 40.607 | 98.96% | 26.498 | 26.755 | 100.97% | 2.129 | 2.095 | 98.40% | PASS |
| code | 52.720 | 51.648 | 97.97% | 21.319 | 21.543 | 101.05% | 2.153 | 2.071 | 96.18% | PASS |
| JSON | 57.655 | 55.046 | 95.47% | 19.552 | 20.104 | 102.82% | 2.068 | 2.010 | 97.20% | PASS |

Machine-readable values and exact identities are in [`comparison.json`](comparison.json).

## Decision

The matched performance, correctness, activity, long-context, stability, and rollback gates passed.
The canonical Qwen3.8 recipe may use Ray while retaining the qualified MP path as the explicit rollback.
This record does not configure autostart or claim that Ray is faster; measured aggregate decode was
1.04–4.53% lower while TTFT was 1.60–3.82% lower and E2E was 0.97–2.82% higher.
