# stepfun-ai/Step-3.7-Flash — Dual-Spark TP=2 Guide (FP8 / NVFP4)

Procedure for serving `stepfun-ai/Step-3.7-Flash` (198B-param sparse MoE VLM,
196B LM + 1.8B vision encoder, ~11B active params/token, 288 experts top-8)
across two DGX Spark nodes (head node + worker node) at TP=2, plus a
configuration/benchmark comparison of the two quantization variants (FP8,
NVFP4).

Both variants use the same image
`vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7`
(NGC 26.05 + vLLM 0.22.1 + `patches/patch_step3p7_nvfp4_input_scale.py`). The
patch only affects the NVFP4-specific code path (`step3p5.py`
`expert_params_mapping` + `fused_moe/layer.py` dual-shard write), so the FP8
variant can use the same image **without rebuilding** (FP8 uses vLLM's
standard `fp8.py` path).

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
- After a crashed container, even `docker compose down` does not release the
  GPU memory (~97-110GiB) held by the driver — only `systemctl reboot`
  reliably reclaims it (~10-12 min). Don't combine reboot with `pkill` in the
  same ssh command (it can kill your own shell before `reboot` runs) — use a
  plain `ssh <host> "sudo systemctl reboot"`.
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

## 7. References

- [`presets/step37-flash-fp8-tp2.env`](../presets/step37-flash-fp8-tp2.env)
- [`presets/step37-flash-nvfp4-tp2.env`](../presets/step37-flash-nvfp4-tp2.env)
- [`patches/patch_step3p7_nvfp4_input_scale.py`](../patches/patch_step3p7_nvfp4_input_scale.py)
- [`docs/dsv4-flash-tp2.md`](dsv4-flash-tp2.md) — same GB10 UMA memory
  pitfalls/mitigations (DeepSeek-V4-Flash)
