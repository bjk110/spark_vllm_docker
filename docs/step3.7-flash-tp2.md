# stepfun-ai/Step-3.7-Flash — Dual-Spark TP=2 Guide (FP8 / NVFP4)

Procedure for serving `stepfun-ai/Step-3.7-Flash` (198B-param sparse MoE VLM,
196B LM + 1.8B vision encoder, ~11B active params/token, 288 experts top-8)
across two DGX Spark nodes (head node + worker node) at TP=2, plus a
configuration/benchmark comparison of the two quantization variants (FP8,
NVFP4).

The FP8 variant uses:
`vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7`
(NGC 26.05 + vLLM 0.22.1 + `patches/patch_step3p7_nvfp4_input_scale.py`)

The **NVFP4 variant requires a separate image** with an additional workaround
for the ModelOpt NVFP4 MoE post-load memory issue (see §4.2):
`vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`
(all of the above + `patches/patch_step3p7_modelopt_cache_release.py`)

## 0. Configuration Summary

| Item | FP8 | NVFP4 |
|---|---|---|
| Preset | [`presets/step37-flash-fp8-tp2.env`](../presets/step37-flash-fp8-tp2.env) | [`presets/step37-flash-nvfp4-tp2.env`](../presets/step37-flash-nvfp4-tp2.env) |
| Quantization | FP8 block (e4m3, 128×128, dynamic activation — DeepSeek-V3 style) | NVFP4 (modelopt) + FP8 KV cache |
| Weight size | ~97.2-97.3 GB/GPU (TP=2) | ~50 GB/GPU (TP=2) |
| `MAX_MODEL_LEN` | 32,768 | 8,192 (preset default; the benchmarked session ran with a larger value, see §4) |
| `MAX_NUM_SEQS` | **1** (memory ceiling, see §3) | 4 |
| `GPU_MEMORY_UTILIZATION` | 0.87 | 0.88 |
| `MAX_NUM_BATCHED_TOKENS` | 2,048 | 8,192 |
| `--enforce-eager` | required (to preserve memory headroom) | required |
| `--quantization` | (default fp8.py path, no flag needed) | `modelopt` |
| `--kv-cache-dtype` | (default auto) | `fp8` |
| `--enable-expert-parallel` | not used | used (288 experts / 2 ranks) |
| Verified | 2026-06-10 | 2026-06-10 |

Common args: `--trust-remote-code --reasoning-parser step3p5
--enable-auto-tool-choice --tool-call-parser step3p5`.

## 1. Serving Procedure

### 1.1. Apply the preset

From your git working copy (the canonical location where `.env` is edited):

```bash
cd vllm-spark
cp presets/step37-flash-fp8-tp2.env .env      # or step37-flash-nvfp4-tp2.env
```

Edit `MODEL_PATH` in `.env` to point at your local model weights:

```
MODEL_PATH=/path/to/models/stepfun-ai/Step-3.7-Flash-FP8
```

### 1.2. Sync to both nodes

`.env` must be synced to both the head and worker node working trees (both
need the new config for the head to boot correctly):

```bash
scp .env <head_node>:/path/to/vllm-spark/.env
scp .env <worker_node>:/path/to/vllm-spark/.env
```

Model weights also need to be fully copied to both nodes (even at TP=2, each
node reads its own shard directly from local disk).

### 1.3. Startup order

Start the worker node first, then the head node — this matches the Ray
rendezvous order:

```bash
# worker node
ssh <worker_node> "cd /path/to/vllm-spark && docker compose --profile worker up -d"

sleep 15

# head node
ssh <head_node> "cd /path/to/vllm-spark && docker compose --profile head up -d"
```

`entrypoint.sh` runs `ray start --head` on the head node, waits for the
worker to join, then starts `vllm serve --distributed-executor-backend ray`.
Total boot time scales with weight size (FP8 ~6 min, NVFP4 ~3-4 min
estimated).

### 1.4. Verify boot

```bash
ssh <head_node> "docker logs --tail 100 vllm-spark-head" | grep -E "GPU KV cache size|Maximum concurrency|Application startup complete"
```

Check for NaN (logprobs should be finite):

```bash
curl -s http://<head_node_ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stepfun-ai/Step-3.7-Flash-FP8","prompt":"The capital of France is","max_tokens":20,"temperature":0,"logprobs":1}'
```

## 2. GB10 UMA Memory Pitfalls (common to both variants)

- `nvidia-smi` reports `Memory-Usage: Not Supported` on GB10 unified memory —
  check actual availability via vLLM's pre-init log
  (`Free memory on device cuda:0 (X/121.63 GiB)`) or `free -h`.
- After a crashed **or graceful** container stop, `docker compose down` does
  not release GPU memory from the driver.  For this two-node TP=2 NVFP4
  configuration, host MemAvailable remains ~19 GiB (~102 GiB below the clean
  baseline) even after all vLLM and Ray processes terminate.  Only
  `systemctl reboot` reliably reclaims UMA (~10-12 min).  Reboot both DGX
  Spark nodes before loading another large model if memory does not recover
  spontaneously.  (This is specific to this high-utilization configuration —
  smaller workloads may recover without a reboot.)
  Don't combine reboot with `pkill` in the same ssh command (it can kill your
  own shell before `reboot` runs) — use a plain
  `ssh <host> "sudo systemctl reboot"`.
- vLLM 0.22.1 (this image has no `VLLM_SKIP_INIT_MEMORY_CHECK` patch) performs
  a two-stage memory check:
  1. **Pre-init** (`request_memory()`): if `util * 121.63 > free at startup`,
     it fails immediately with `ValueError: Free memory on device cuda:0
     (.../121.63 GiB) on startup is less than desired GPU memory
     utilization`.
  2. **Post-profile** (`_check_enough_kv_cache_memory`): if `Available KV
     cache memory = util*total - (weights+activation)` is negative, it fails
     with `ValueError: No available memory for the cache blocks.`

## 3. FP8: Why MAX_NUM_SEQS=1

FP8 weights (~97GB/GPU) leave almost no headroom against the idle free ceiling
(106.41/121.63 GiB). Attempt history:

| Config | rank0 (head) | rank1 (worker, RDMA) | Result |
|---|---:|---:|---|
| util=0.85, batched=16384, seqs=4 | -1.2 GiB | -10.83 GiB | **failed** (post-profile) |
| util=0.87, batched=2048, seqs=1 | +6.6 GiB | +2.27 GiB | **succeeded** |

- `rank1` (the cross-node RDMA Ray worker) consistently has 4-10GiB more
  activation/comm overhead than `rank0` — likely caused by NCCL/RDMA
  cross-node communication buffer asymmetry (unconfirmed).
- Lowering `MAX_NUM_BATCHED_TOKENS`/`MAX_NUM_SEQS` shrinks the dummy batch
  (activation memory) used during profiling, directly increasing "Available
  KV cache memory" — this lever flipped both ranks positive.
- **What `MAX_NUM_SEQS=1` means**: the reported KV cache of 80,308 tokens /
  "max concurrency 2.45x @ 32768" is just a memory-headroom calculation;
  `MAX_NUM_SEQS=1` hard-caps actual concurrent request processing to 1 — the
  2.45x headroom can't be used for concurrency (it's only meaningful for a
  single very long sequence).
- **Conclusion**: on a 121GiB GB10 UMA, FP8 TP=2 is structurally tight on
  concurrency. If concurrency matters, NVFP4 (~50GB/GPU, `MAX_NUM_SEQS=4`
  verified) is the better choice.

## 4. NVFP4 NaN Bug (resolved)

The NVFP4 variant previously produced NaN logits on every output before
2026-06-10.

- **Cause**: the checkpoint's NVFP4 per-expert input scales
  (`.moe.{gate,up,down}_proj.input_scale`, shape `[288]`) were not mapped in
  `Step3p5Model.load_weights()`'s `expert_params_mapping`, leaving
  `w13_input_scale`/`w2_input_scale` as uninitialized `torch.empty()` garbage
  → the entire NVFP4 MoE GEMM produced NaN. Independent of MoE backend
  (CUTLASS/Marlin) selection.
- **Fix**: `patches/patch_step3p7_nvfp4_input_scale.py` (2 files):
  1. Add 3 `.input_scale` mappings to `expert_params_mapping` in
     `step3p5.py`.
  2. Add a ModelOpt dual-shard branch to the input-scale loading in
     `fused_moe/layer.py` (`w13_input_scale` has shape `[num_experts, 2]` —
     write into the w1/w3 slots separately).
- **Verification**: after rebuilding
  `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7`, `/v1/completions`
  returned `"Paris."` with finite token_logprobs (range -0.03 to -1.9). No
  NaN.
- When adding similar NVFP4 ModelOpt MoE models (old packed 3D checkpoint
  format), check first whether `expert_params_mapping` is missing
  `.input_scale` entries.

## 4.2. NVFP4 ModelOpt MoE Post-Load OOM (resolved)

Without the additional workaround, serving Step-3.7-Flash-NVFP4 on dual
DGX Spark GB10 (121.63 GiB UMA per node) fails at startup with a Ray OOM
kill during `process_weights_after_loading()`.

- **Cause**: each of the 42 `ModelOptNvFp4FusedMoE` modules retains ~6.68 GiB
  in the CUDA caching-allocator reserved pool after NVFP4→MARLIN repacking
  completes.  Active allocations return to baseline, but the reserved pool
  grows monotonically: 42 × 6.68 GiB ≈ 280 GiB reserved in total.  On a
  121.63 GiB UMA node, this exhausts host MemAvailable well before all
  modules complete, triggering Ray's OOM monitor.
- **Fix**: `patches/patch_step3p7_modelopt_cache_release.py` inserts
  `torch.cuda.empty_cache()` after each completed `ModelOptNvFp4FusedMoE`
  module when `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=1`.  This resets
  the reserved pool back to the stable post-weight-load baseline (~65 GiB)
  after each conversion, preventing cumulative growth.
  Feature is disabled by default (`:-0` in docker-compose.yml) and enabled
  only in `presets/step37-flash-nvfp4-tp2.env`.
- **Additional requirement**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
  — expandable segments alone reduces but does not eliminate the cumulative
  growth; both settings together are required.
- **Overhead**: < 0.5 s cumulative per rank (validated: rank0 427 ms,
  rank1 297 ms over 42 modules on dual DGX Spark GB10).
- **Verified**: 2026-06-14, dual DGX Spark GB10, TP=2 EP=2, Ray, vLLM 0.22.1.
  See `docs/diagnostics/dgx-spark-uma-memory-freeze.md` §32 for full record.

## 5. Benchmark Results

Tool: [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.7, endpoint
`http://<head_node_ip>:8000/v1`, depth-sweep format (pp2048, tg32, runs=3,
latency-mode=generation — the standard format for DSV4/Step-3.7
benchmarks).

### 5.1. FP8 (MAX_MODEL_LEN=32768, MAX_NUM_SEQS=1, c=1)

Depths were chosen under the constraint `pp2048 + depth + tg32 ≤ 32768`:
d0/4096/8192/16384/28672 (NVFP4's d32768/d65536 exceed the 32768 limit and are
excluded).

Results file: [`benchmarks/llama-benchy/results_step37-flash-fp8-tp2-DEPTH.md`](../benchmarks/llama-benchy/results_step37-flash-fp8-tp2-DEPTH.md)

| depth | pp2048 t/s | tg32 t/s | peak tg t/s |
|---|---:|---:|---:|
| 0 | 1084.4 ± 51.7 | 13.32 ± 0.12 | 14.00 |
| 4096 | 1099.2 ± 5.7 | 13.12 ± 0.05 | 14.00 |
| 8192 | 1055.4 ± 11.6 | 13.12 ± 0.12 | 14.00 |
| 16384 | 1053.8 ± 1.0 | 13.18 ± 0.05 | 14.00 |
| 28672 | 1031.9 ± 0.3 | 13.25 ± 0.01 | 14.00 |

- Prefill: ~1030-1100 t/s, decreasing gently with depth (-5% even at d28672)
- Decode: ~13.1-13.3 t/s, flat across the whole range, peak 14.0 t/s
  throughout

### 5.2. NVFP4 (c=1, depth sweep)

Results file: [`benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-DEPTH.md`](../benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-DEPTH.md)

| depth | pp2048 t/s | tg32 t/s | peak tg t/s |
|---|---:|---:|---:|
| 0 | 1251.42 ± 3.22 | 13.35 ± 0.71 | 14.00 ± 0.82 |
| 4096 | 1299.69 ± 1.11 | 12.84 ± 0.34 | 14.00 ± 0.82 |
| 8192 | 1289.83 ± 3.11 | 11.90 ± 0.20 | 12.67 ± 0.47 |
| 16384 | 1267.43 ± 1.14 | 12.11 ± 0.36 | 12.67 ± 0.47 |
| 32768 | 1235.27 ± 16.01 | 12.37 ± 0.55 | 13.33 ± 0.47 |
| 65536 | 1148.67 ± 1.10 | 12.03 ± 0.09 | 13.00 ± 0.00 |

### 5.3. NVFP4 concurrency sweep (c=1/2/4)

Results file: [`benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-c1to4.md`](../benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-c1to4.md)

| concurrency | pp2048 t/s (total) | tg32 t/s (total) | peak tg t/s |
|---|---:|---:|---:|
| c1 | 1247.40 ± 5.60 | 13.23 ± 0.27 | 14.33 ± 0.47 |
| c2 | 1196.81 ± 42.81 | 22.85 ± 5.05 | 28.00 ± 0.00 |
| c4 | 1230.07 ± 2.61 | 21.18 ± 0.28 | 45.33 ± 1.89 |

`MAX_NUM_SEQS=4` was confirmed to handle up to c=4 concurrent requests (FP8
can't be swept for concurrency since `MAX_NUM_SEQS=1`, see §3).

### 5.4. FP8 vs NVFP4 comparison (c=1, d0)

| Metric | FP8 | NVFP4 | Note |
|---|---:|---:|---|
| Prefill (pp2048) | 1084.4 | 1251.4 | NVFP4 +15% |
| Decode (tg32) | 13.32 | 13.35 | roughly equal |
| Max concurrency | 1 (structural limit) | 4 (verified) | NVFP4 ahead |
| Weights/GPU | ~97GB | ~50GB | NVFP4 has much more memory headroom |

At c=1 single-stream, the two variants perform similarly. If concurrent users
(c≥2) are needed, NVFP4 is clearly the better choice.

## 6. Operational Recommendations

| Scenario | Recommendation |
|---|---|
| Single user, long context (up to 32K) priority | FP8 (`MAX_NUM_SEQS=1`, KV pool 80,308 tokens) |
| Multiple concurrent users (2-4) | **NVFP4** (`MAX_NUM_SEQS=4`, FP8 KV cache) |
| Memory headroom priority | NVFP4 (~50GB/GPU vs FP8 ~97GB/GPU) |

## 7. Known Issue: Korean / Non-ASCII Output Garbling (resolved)

Symptom (observed via OpenWebUI, 2026-06-10): Korean prompts produced
unrelated/hallucinated output, and the `reasoning`/`content` fields contained
literal byte-level BPE markers (`Ġ`, `Ċ`) instead of spaces/newlines —
rendering as garbled text.

- **Cause**: the checkpoint's `tokenizer_config.json` declares
  `"tokenizer_class": "LlamaTokenizerFast"`. Under transformers 5.10.2 (this
  image's version), `AutoTokenizer.from_pretrained()` detects an "incorrect
  regex pattern" (same family as the
  [mistralai/Mistral-Small-3.1-24B-Instruct-2503 tokenizer issue](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e))
  and silently falls back to the **slow** `LlamaTokenizer`
  (SentencePiece-based). The slow tokenizer (a) drops non-ASCII (Korean) text
  during encoding — `/tokenize` on `"한국의 수도는 어디?"` returned only
  `[0, 33]` (BOS + trailing `?`, the Korean text vanished) — and (b) doesn't
  reverse GPT2's byte-to-unicode mapping during decoding, leaking `Ġ`/`Ċ`
  markers into the output.
- **Fix**: run
  [`patches/patch_step3p7_tokenizer_class.py`](../patches/patch_step3p7_tokenizer_class.py)
  against `tokenizer_config.json` on the model weight directory, on every node
  that holds a copy of the weights:
  ```bash
  python3 patches/patch_step3p7_tokenizer_class.py /path/to/Step-3.7-Flash-FP8/tokenizer_config.json
  ```
  This changes only `"tokenizer_class": "LlamaTokenizerFast"` to
  `"tokenizer_class": "PreTrainedTokenizerFast"` (with a `.bak-tokenizer-class`
  backup), making `AutoTokenizer` resolve to the Rust fast tokenizer instead.
  The underlying `tokenizer.json` is correct and does not need changes.
  Restart the vLLM containers (worker then head) after editing.
- **Verification**:
  ```bash
  curl -s http://<head_node_ip>:8000/tokenize \
    -H "Content-Type: application/json" \
    -d '{"model":"stepfun-ai/Step-3.7-Flash-FP8","prompt":"한국의 수도는 어디?"}'
  # expect ~8 tokens, not [0, 33]
  ```
  A `/v1/chat/completions` request with a Korean prompt should return relevant
  Korean `reasoning`/`content` with no `Ġ`/`Ċ` artifacts.
- The `[transformers] ... incorrect regex pattern ... fix_mistral_regex=True`
  warning still prints at startup even after this fix — it appears benign
  (vLLM resolves the fast tokenizer via a different path regardless).
- Verified on the FP8 variant (2026-06-11). Apply the same edit to the NVFP4
  checkpoint if/when it's deployed, since it likely ships the same
  `tokenizer_config.json`.

## 8. References

- [`presets/step37-flash-fp8-tp2.env`](../presets/step37-flash-fp8-tp2.env)
- [`presets/step37-flash-nvfp4-tp2.env`](../presets/step37-flash-nvfp4-tp2.env)
- [`patches/patch_step3p7_nvfp4_input_scale.py`](../patches/patch_step3p7_nvfp4_input_scale.py)
- [`docs/dsv4-flash-tp2.md`](dsv4-flash-tp2.md) — same GB10 UMA memory
  pitfalls/mitigations (DeepSeek-V4-Flash)
