# DeepSeek V4 prefill8192 Production Activation Record

## Status

- **Final classification:** `PRODUCTION_RUNTIME_ACTIVE_ACCEPTED`
- **Activation date:** 2026-06-25
- **Runtime status at completion:** running
- **Repository package HEAD at activation:** `1c899d7d5769391bbfbefdaad6931cc85169a1d2`
- **Production preset:**
  [`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env`](../presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env)
  (SHA-256 `fdf765ec5f6c8abaea4f4a302ef62f28add80d1fe906e5d0417ae277acb5deda`)

## Scope

This is the final activation and acceptance record for the DeepSeek-V4-Flash
prefill8192 production serving path.

- Detailed activation/rollback procedures remain in the
  [production runbook](deepseek-v4-prefill8192-production-runbook.md).
- Detailed candidate testing remains in the
  [validated-candidate record](deepseek-v4-prefill8192-validated-candidate.md).
- Raw runtime artifacts (startup logs, per-request captures, telemetry) remain
  local on the operator host and are intentionally **not** committed.
- Post-activation long-duration observation is **out of scope** of this record. No
  24-hour or longer stability campaign was performed or is claimed.

## Configuration identity

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash` (official checkpoint, 46 shards) |
| Image config | `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105` |
| Image manifest | `sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44` |
| Source pin | `72261a7af149fa5d3fe2ed2b9956e92590731012` |
| Cluster | dual DGX Spark |
| Tensor parallel | 2 |
| Distributed backend | multiprocessing (one rank per node) |
| Transport | NET/IB over RoCE |
| MTP | n=1 |
| Graph mode | FULL_DECODE_ONLY, capture `[2]` |
| max-num-batched-tokens | 8192 |
| max model length | 135168 |
| max-num-seqs | 1 |
| Fixed KV | 4 GiB, dtype fp8 |
| Prefix cache | disabled |
| Approved prompt envelope | up to 131,072 tokens |
| Typical approved output | 128 tokens |

## Activation summary

- Exactly one clean reboot per Spark node.
- Clean-memory gate passed: MemAvailable ~117.7 GiB (spark01) and ~118.0 GiB
  (spark02); swap zero after reboot.
- Dedicated vLLM cache cleared (model, tokenizer, AOT, and unrelated caches preserved).
- RoCE UP; RDMA ACTIVE / LINK_UP; MTU 9000; native NET/IB confirmed; NET/Socket
  fallback zero.
- Worker started first (spark02), then head (spark01); a single engine
  initialization; both ranks alive.
- MTP active on both ranks; CUDA graph captured on both ranks.
- Runtime KV capacity 159,445 tokens (1.18x at max model length 135,168); observed
  margin against the previously tested maximum sequence ~28,274 tokens.
- API health HTTP 200; graph fallback zero; graph recapture zero; NCCL errors zero;
  CUDA errors zero.

## Acceptance summary

Bounded acceptance (concurrency 1, one request at a time) passed:

- arithmetic, English, Korean, and Unicode correctness
- speculative rejection-heavy request
- token accounting (total == prompt + completion)
- short streaming generation
- exact-token 64K English retrieval
- direct API streaming
- OpenWebUI-path streaming

64K retrieval evidence: expected identifier `ZEBRA-7741` recovered correctly; exact
local/API prompt tokens `65511`; first-line contract followed; no prompt truncation.

Performance smoke: decode ~21.7 tokens/s, consistent with the validated-candidate
range; no performance-collapse indication.

Resource evidence: minimum MemAvailable during acceptance ~26.7 GiB; sustained
swap-out none; PSI zero; both ranks healthy.

## Monitoring acceptance

Prometheus targets at acceptance: `node-local` UP, `node-spark01` UP,
`node-spark02` UP, `prometheus` UP, `vllm-spark01` UP.

- Monitoring activation gate enabled: `vllm_serving_activation_gate = vector(1)`.
- Availability alert `VLLMSpark01Down` is **armed**: currently inactive because the
  target is UP; it fires after the configured 120-second duration when the target
  remains down.
- OpenWebUI healthy; endpoint `http://192.168.0.200:8000/v1`;
  `AIOHTTP_CLIENT_TIMEOUT=240`.

The Prometheus rule file lives **outside this repository** at
`/home/bjk110/docker/monitoring/rules/vllm_spark.rules.yml` (operational path, not a
repository file; current SHA-256
`b91951bcbd38e60dba4ef2e0aaf08ebf952b72befd2d6ae4b5e6004e5fd703da`).

**Remaining log-only gaps** (no confirmed exported Prometheus metric): graph
fallback, graph recapture, per-rank health, NCCL errors, and CUDA errors. These
remain operator log checks; this record does not claim complete metric coverage.
Spark cAdvisor, Spark DCGM exporter, and homeserver cAdvisor are intentionally
deferred and are not active.

## Rollback readiness

- Committed 2048 validated baseline
  [`presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env`](../presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env)
  remains available.
- Graph-only L1
  [`presets/deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env`](../presets/deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env)
  remains available.
- Eager U0-RDMA L2
  [`presets/deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env`](../presets/deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env)
  remains available.
- No image rebuild is required for rollback.
- UVM reclamation generally requires a reboot before another full model load.

## Final declaration

> **Status update (2026-07-01): SUPERSEDED — now the rollback baseline.** This
> activation record is preserved as the historical acceptance of the prefill8192
> baseline (config `4c41950c`). The current accepted DeepSeek-V4-Flash production
> serving path is now `dsv4-sm121-indexer` (digest `ade810fd`, config `fa83457d`;
> `H1Z_B1AE_PROMOTION_CUTOVER_PASS`), for which this prefill8192 baseline is the
> immediate rollback target. See
> [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md).
> The present-tense wording below reflects the state at original activation.

The prefill8192 configuration is the **current accepted DeepSeek V4 production
serving path** within the validated envelope (concurrency 1; prompts up to 131,072
tokens; typical output 128 tokens; fixed 4 GiB fp8 KV; prefix cache disabled; MTP
n=1; FULL_DECODE_ONLY capture `[2]`; TP=2 multiprocessing; NET/IB over RoCE).

This record does **not** claim: concurrency above one; context above 131K;
prefix-cache validation; B12X validation; DCGM monitoring coverage; cAdvisor
monitoring coverage; or any 24-hour or longer post-activation stability observation.
