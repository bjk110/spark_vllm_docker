# DeepSeek-V4-Flash — SM121 DeepGEMM FP8-Q indexer production baseline

**Status: `Current production baseline` (`H1Z_B1AE_PROMOTION_CUTOVER_PASS`, 2026-07-01).**

This is the current accepted DeepSeek-V4-Flash production serving path on the dual-GB10 DGX Spark
TP=2 cluster. It is the immutable digest-pinned promotion of the validated H1Z-B1AA SM121 DeepGEMM
FP8-Q prefill-indexer candidate. Repository production status does **not** auto-activate the runtime;
live activation/rollback follows the maintenance-window discipline used by H1Z-B1AE and the
prefill8192 runbook.

---

## 1. Runtime image identity

| Field | Value |
|---|---|
| **Immutable runtime reference (the runtime pin)** | `ghcr.io/bjk110/vllm-spark@sha256:ade810fd637e30922a30d09f0fcf128fbeb2a757a27a64f8a77e3646fae223a7` |
| Immutable promoted tag (provenance) | `ghcr.io/bjk110/vllm-spark:v023-dsv4-72261a7-sm121-deepgemm-indexer-prod-fa83457d` |
| Mutable human-readable alias | `ghcr.io/bjk110/vllm-spark:dsv4-sm121-indexer-production` |
| Image config (whole-image identity anchor) | `sha256:fa83457d35c1cd91c511d7ae88dbce7966f0667bf6dd3219026940d93618f459` |
| Ordered layers | 106 (first 105 == H1Z-B1V parent; 106th = SM121 source-clone `COPY`) |
| Source candidate tag | `vllm-spark:v023-dsv4-72261a7-h1z-b1aa-sm121-mqa-source-clone-marlin-exp-501043b38bcb` |

**The runtime MUST be pinned to the immutable manifest digest, not the mutable alias.** The alias
`dsv4-sm121-indexer-production` resolves to the same digest today, but a mutable tag can be
repointed; only the digest guarantees the exact validated bytes. The whole-image config
`fa83457d` and the ordered 106-layer identity are the primary immutable promotion anchors — any
config ID other than `fa83457d` invalidates the promotion.

The image was promoted with `docker tag` + `docker push` only (no rebuild); all layers are byte
reuse of the validated candidate.

## 2. Active production preset

`presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env` (full-file SHA `f1b049d5`; runtime
payload — non-comment lines — SHA `b986d87a`). Its only semantic delta from the validated H1Z-B1AA
preset is the `VLLM_IMAGE` value (mutable local tag → immutable GHCR digest). Serve with:

```
docker compose --env-file presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env up -d worker   # spark02
docker compose --env-file presets/deepseek-v4-h1z-b1ae-sm121-indexer-production-tp2.env up -d head     # spark01
```

## 3. Runtime architecture

| Setting | Value |
|---|---|
| Entrypoint | `./entrypoints/entrypoint.sh` |
| Tensor parallel | `TP_SIZE=2`, one rank per node |
| Distributed backend | multiprocessing (`mp`), dual-RDMA |
| Head / worker RoCE IP | `10.10.10.1` / `10.10.10.2` (`enp1s0f0np0`, HCA `rocep1s0f0`, MTU 9000) |
| MTP | `n=1` (`deepseek_mtp`, `num_speculative_tokens=1`) |
| KV cache | fixed 4 GiB (`--kv-cache-memory-bytes 4294967296`), dtype fp8; measured capacity 159,445 tokens |
| Graph | `FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[2]`, warmups 2 |
| Context / concurrency | `MAX_MODEL_LEN=135168`, `MAX_NUM_SEQS=1`, prompts up to 131,072 tokens, prefix cache disabled |

### Routing isolation (the whole point of this baseline)

| Path | Backend |
|---|---|
| MoE | **MARLIN** (explicit `--moe-backend marlin`; `VLLM_MOE_USE_DEEP_GEMM=0`) |
| Dense FP8 | production Triton (`VLLM_USE_DEEP_GEMM_E8M0=0`, UE8M0 disabled) |
| Sparse MLA | production Triton (`sparse_mla_prefill_mg_dual` JIT absent) |
| **FP8-Q prefill indexer** | **DeepGEMM SM121 entry symbol with internal SM120-family implementation** (cold-JIT `sm121_fp8_mqa_logits`, `#include <deep_gemm/impls/sm121_fp8_mqa_logits.cuh>`; no include-parser assertion, no Triton indexer fallback) |
| B12X | off |
| B2 | absent |

## 4. Performance validation (reference ranges, not universal guarantees)

Concurrency-1 TTFT-derived prefill throughput. Values are validated reference ranges under the
approved envelope (concurrency 1, prompts up to 131K); they are **not** universal guarantees and are
context-dependent.

| depth | Candidate / promoted production | Fresh former-production reference (rollback baseline) | Recovered H1C uplift |
|---|---|---|---|
| 32K | ~1818.7–1832.1 t/s | 1661.3 t/s | 47.6% |
| 64K | ~1732.3–1758.0 t/s | 1503.4 t/s | 54.8% |
| 131K | ~1575.2–1594.9 t/s | 1337.6 t/s | 52.1% |

H1Z-B1AE cutover medians (CV<0.6%) landed within ~1.5% of the H1Z-B1AB reference (32K −0.7%,
64K −1.5%, 131K −1.2%).

## 5. Promotion evidence chain

| Task | Outcome |
|---|---|
| H1Z-B1AB | `H1Z_B1AB_INDEXER_UPLIFT_VIABLE` — cold-JIT SM121 clone compile+execute, correctness 10/10, retrieval matrix PASS |
| H1Z-B1AC | `H1Z_B1AC_PROMOTION_READY` — 60-min soak, mandatory concurrency-2, 100% request success, memory stable |
| H1Z-B1AD | `H1Z_B1AD_PROMOTION_PLAN_READY` / `SOAK_60MIN_CONFIRMED` / `PROMOTION_EVIDENCE_COMPLETE` |
| H1Z-B1AE | `H1Z_B1AE_PROMOTION_CUTOVER_PASS` — immutable promotion + digest-pinned cutover, 10/10 correctness, retrieval matrix ALL PASS, 15-min observation PASS |

External evidence workspaces (immutable, not part of this repo):
`h1z-b1ab-sm121-source-clone-runtime-20260701T010000Z/`,
`h1z-b1ac-promotion-evaluation-20260701T040000Z/`,
`h1z-b1ad-promotion-plan-20260701T093000Z/`,
`h1z-b1ae-immutable-promotion-cutover-20260701T094500Z/`.

## 6. Source and binary provenance

| Item | Value |
|---|---|
| vLLM source commit | `72261a7af149fa5d3fe2ed2b9956e92590731012` |
| DeepGEMM source commit | `1f2f161dba747b7c12671d017f7c88e1249c3d3e` |
| Patched `vllm/utils/deep_gemm.py` | SHA `caacc1b0` |
| SM120 source — prefill MQA (`sm120_fp8_mqa_logits.cuh`) | SHA `cedcce47` |
| SM120 source — paged MQA (`sm120_fp8_paged_mqa_logits.cuh`) | SHA `b3a5d236` |
| SM121 clone — prefill MQA (`sm121_fp8_mqa_logits.cuh`) | SHA `0e1d9bac` |
| SM121 clone — paged MQA (`sm121_fp8_paged_mqa_logits.cuh`) | SHA `34145813` |
| **Authoritative production-ABI custom-ops binary** | `vllm/_C_stable_libtorch.abi3.so` SHA **`09f2696b`** |

### ABI-hash provenance correction

Earlier B1V/B1Y/B1AB working notes propagated the SHA prefix **`7d16d0aa`** as an "`_C.so`"
identity. H1Z-B1AE Phase 0 directly extracted and hashed the actual production-ABI custom-ops
library from the candidate image and established the authoritative facts:

- The authoritative production-ABI custom-ops file is `vllm/_C_stable_libtorch.abi3.so`, SHA
  **`09f2696b`** (with `vllm/_moe_C_stable_libtorch.abi3.so` `c3c24daf` and
  `cumem_allocator.abi3.so` `44710ceb`).
- This binary is **byte-identical between the promoted candidate and the former production image
  `4c41950c`** — the candidate carries the same production-ABI binaries (PyTorch 2.12, no PyTorch
  2.11 / no H1C transplant).
- `7d16d0aa` was a **stale provenance label** and is superseded by `09f2696b` for the authoritative
  direct file hash. (Note: `7d16d0aa` still legitimately appears in the untracked historical
  H1X-B1R record as the SHA of a *different* artifact — the rebuilt DeepGEMM host `_C` binding from
  donor `1f2f161d` — which is not this file; that historical record is preserved unchanged.)
- The **whole-image config `fa83457d` and ordered 106-layer identity remain the primary immutable
  promotion anchors**; the ABI hash is a supporting identity, not the promotion anchor.

## 7. Rollback baseline

The immediate rollback target is the prior production baseline, preserved unchanged:

| Item | Value |
|---|---|
| Rollback image | `vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`, config `sha256:4c41950c47ecb771…` |
| Rollback registry manifest digest | `sha256:2f4a96283fc5b491…` |
| Rollback preset | `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env` (SHA `593ba898`) |
| Monitoring gate rule | SHA `b91951bc` |

### Rollback procedure (mirror of the validated restoration path)

1. Monitoring gate 1 → 0 (SIGHUP reload; no Prometheus restart).
2. Stop promoted head, then promoted worker.
3. Reboot each node **once** only if UVM is retained and MemAvailable < 110 GiB; otherwise skip.
   **No third reboot** within a maintenance window.
4. Clear ONLY the allowlisted compile caches; preserve model/tokenizer/FlashInfer AOT.
5. Restore the rollback preset `593ba898` as the active env on both nodes.
6. Start rollback **worker (spark02) first**, then **head (spark01)**, each exactly once, no retry.
7. Accept: HTTP 200; config `4c41950c`; MARLIN both ranks; DeepGEMM absent; KV ~159,445.
8. Generation check: "The capital of France is" → " Paris.".
9. Gate 0 → 1 after rollback acceptance (verify gate=1, targets 5/5, alerts inactive).

## 8. Source-clone maintenance guard

The production SM121 source clones are byte copies of the SM120 DeepGEMM FP8 MQA sources with only
the exported `CUTLASS_GLOBAL` kernel entry-point identifier renamed `sm120_*`→`sm121_*` (exactly one
line per file). They remain valid **only while the SM120 source SHAs retain**:

- prefill MQA `sm120_fp8_mqa_logits.cuh` = `cedcce47`
- paged MQA `sm120_fp8_paged_mqa_logits.cuh` = `b3a5d236`

Any change to those SM120 sources (DeepGEMM bump, kernel signature / template-param change, or a real
upstream sm121 impl landing) **invalidates the promoted clone validation status** until ALL of the
following are repeated:

1. Regenerate the SM121 clones from the new SM120 sources.
2. Verify exactly one exported identifier replacement per file.
3. Verify reverse-substitution byte identity against the SM120 source.
4. Verify no forwarding `#include` and no macro alias.
5. Verify no nested `deep_gemm/impls/` include (no `impls/`→`impls/` edge).
6. Run include-parser validation (no `include_parser.hpp:62` circular, no `:69` missing).
7. Run preprocessing validation with the SM121 target (guard active at `__CUDA_ARCH__` 1210).
8. Perform cold-JIT runtime validation (B1AB-equivalent).
9. Run correctness and retrieval.
10. Run performance attribution.
11. Run the promotion soak evaluation (B1AC-equivalent).

This guard is also summarized in the preset header.

## 9. Known limitations

- **Source-clone maintenance burden** — the SM121 clones must be re-validated on any SM120-source
  change (see §8). There is no automated tracking of the SM120 SHAs yet.
- **Sustained concurrency 4 not formally soak-validated** — H1Z-B1AC validated mandatory
  concurrency-2; concurrency-4 was only a bounded observation. Increasing concurrency requires KV
  headroom re-validation and a concurrency-4 soak.
- **B2 absent** — the second-stage indexer optimization (B2) is not implemented.
- **Performance is context-dependent** — the §4 values are validated reference ranges under the
  approved envelope (concurrency 1, prompts up to 131K), not universal guarantees.

## 10. Operational notes

- Clean-boot + dedicated-cache-clear startup gate (not automated by the preset): require MemAvailable
  ≥ 110 GiB per node before full model load; clear only the allowlisted compile caches.
- OpenWebUI backend timeout `AIOHTTP_CLIENT_TIMEOUT ≥ 180` (confirmed effective 240).
- Monitoring: Prometheus + both Spark node-exporters healthy; gate rule `b91951bc`. Prometheus 9090
  is not host-published — query via `docker exec prometheus wget -qO- http://localhost:9090/...`.
