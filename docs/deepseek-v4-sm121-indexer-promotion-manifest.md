# DeepSeek-V4-Flash SM121 DeepGEMM indexer — promotion manifest

Machine-oriented promotion record for the current DeepSeek-V4-Flash production baseline. Companion
to [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md). Contains no
secrets, credentials, tokens, or machine-specific data.

| Field | Value |
|---|---|
| Promotion date | 2026-07-01 |
| Overall outcome | `H1Z_B1AE_PROMOTION_CUTOVER_PASS` |
| Source candidate tag | `vllm-spark:v023-dsv4-72261a7-h1z-b1aa-sm121-mqa-source-clone-marlin-exp-501043b38bcb` |
| Immutable promoted tag | `ghcr.io/bjk110/vllm-spark:v023-dsv4-72261a7-sm121-deepgemm-indexer-prod-fa83457d` |
| Immutable manifest digest (runtime pin) | `sha256:ade810fd637e30922a30d09f0fcf128fbeb2a757a27a64f8a77e3646fae223a7` |
| Config digest (whole-image identity anchor) | `sha256:fa83457d35c1cd91c511d7ae88dbce7966f0667bf6dd3219026940d93618f459` |
| Ordered layers | 106 (first 105 == H1Z-B1V parent) |
| Mutable alias (provenance only, NOT runtime pin) | `ghcr.io/bjk110/vllm-spark:dsv4-sm121-indexer-production` |
| Running preset path | `presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env` |
| Running preset SHA (full file) | `f1b049d5` |
| Running preset SHA (runtime payload, non-comment) | `b986d87a` |
| vLLM source commit | `72261a7af149fa5d3fe2ed2b9956e92590731012` |
| DeepGEMM source commit | `1f2f161dba747b7c12671d017f7c88e1249c3d3e` |
| Patched `vllm/utils/deep_gemm.py` SHA | `caacc1b0` |
| Authoritative production-ABI binary | `vllm/_C_stable_libtorch.abi3.so` = `09f2696b` |
| — companion ABI binaries | `_moe_C_stable_libtorch.abi3.so` = `c3c24daf`; `cumem_allocator.abi3.so` = `44710ceb` |
| Stale ABI label (superseded) | `7d16d0aa` (was mislabeled as `_C.so`; authoritative direct hash is `09f2696b`) |
| SM120 source — prefill MQA | `cedcce47` (`sm120_fp8_mqa_logits.cuh`) |
| SM120 source — paged MQA | `b3a5d236` (`sm120_fp8_paged_mqa_logits.cuh`) |
| SM121 clone — prefill MQA | `0e1d9bac` (`sm121_fp8_mqa_logits.cuh`) |
| SM121 clone — paged MQA | `34145813` (`sm121_fp8_paged_mqa_logits.cuh`) |

## Validation tasks and outcomes

| Task | Outcome |
|---|---|
| H1Z-B1AB | `H1Z_B1AB_INDEXER_UPLIFT_VIABLE` |
| H1Z-B1AC | `H1Z_B1AC_PROMOTION_READY` |
| H1Z-B1AD | `H1Z_B1AD_PROMOTION_PLAN_READY` / `SOAK_60MIN_CONFIRMED` / `PROMOTION_EVIDENCE_COMPLETE` |
| H1Z-B1AE | `H1Z_B1AE_PROMOTION_CUTOVER_PASS` |

## Rollback

| Field | Value |
|---|---|
| Rollback image | `vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019` (config `sha256:4c41950c47ecb771…`) |
| Rollback registry manifest digest | `sha256:2f4a96283fc5b491…` |
| Rollback preset | `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env` (SHA `593ba898`) |
| Monitoring gate rule SHA | `b91951bc` |

## Maintenance guard

SM121 clones valid ONLY while SM120 sources retain `cedcce47` (prefill) and `b3a5d236` (paged). Any
change requires the full 11-step revalidation in
[`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md) §8 before the
promotion is eligible again.
