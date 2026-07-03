# DeepSeek-V4-Flash — B3I 64K c1 A/B: FlashInfer sparse-MLA prefill vs MARLIN

**Purpose:** decide whether the experimental FlashInfer CUDA sparse-MLA *prefill-only*
candidate (H1Z-B3H vendorfix) delivers a material 64K prefill uplift over the current
MARLIN + SM121-DeepGEMM-indexer production baseline. **Result: no uplift (parity,
−0.91%). Candidate NOT promoted.** Closure:
[`../../docs/dsv4-sparse-mla-b3-investigation-closure.md`](../../docs/dsv4-sparse-mla-b3-investigation-closure.md).

## Configuration

- Date: 2026-07-03 (H1Z-B3I).
- Cluster: dual GB10 (spark01 head + spark02 worker), TP=2, `mp` backend, RoCE.
- Model: `deepseek-v4-flash`, MTP n=1, fixed 4 GiB FP8 KV (159,445 tokens),
  `MAX_NUM_SEQS=1`, FULL_DECODE_ONLY graph capture `[2]`.
- Reference (MARLIN): production route — MARLIN MoE, native SM121 DeepGEMM FP8-Q
  prefill indexer, Triton sparse-MLA prefill/decode. Launched via
  `docker-compose.b3d.yml` (no prefill flag).
- Candidate: same image family with vendored FlashInfer sparse-MLA prefill kernel
  active (`VLLM_DSV4_SPARSE_MLA_PREFILL=flashinfer_sm12x`,
  `VLLM_DEEPSEEK_V4_FLASHINFER_SM120_PREFILL=1`); MARLIN MoE + SM121 indexer + dense
  FP8 preserved; Triton sparse-MLA prefill invoked = 0 (no silent fallback); official
  FlashInfer DECODE class disabled. Image
  `vllm-spark:v023-dsv4-72261a7-sparse-mla-prefill-vendorfix-exp-ff4477f4878c`
  (id `f17c8d51`). Env from
  `presets/deepseek-v4-b3h-sparse-mla-prefill-vendorfix-exp-tp2.env` via `--env-file`.
- Harness: llama-benchy 0.3.8, depth-sweep — `--pp 2048 --tg 32 --depth 65536
  --runs 3 --concurrency 1 --latency-mode generation --exact-tg --no-cache`.
- Both sides measured in the **same boot window**, identical harness, warmup + 3 reps,
  diagnostics off. Median of 3 reported.

> **Note:** this is a **targeted 64K c1 depth-sweep A/B sentinel**, not the standard
> 24-row full matrix. It measures a single point (64K, c1) chosen as the decision
> gate for the B3 investigation. It is deliberately narrow and must not be read as a
> full-matrix benchmark.

## Result (median of 3)

| model | test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-v4-flash (MARLIN ref) | pp2048 @ d65536 (c1) | 1746.56 | 1746.56 | — | — | — | — | 38896.64 |
| deepseek-v4-flash (MARLIN ref) | tg32 (c1) | 40.99 | 40.99 | — | — | — | — | — |
| deepseek-v4-flash (FI sparse-MLA cand) | pp2048 @ d65536 (c1) | 1730.61 | 1730.61 | — | — | — | — | 39249.73 |
| deepseek-v4-flash (FI sparse-MLA cand) | tg32 (c1) | 37.35 | 37.35 | — | — | — | — | — |

Prefill spread: MARLIN pp 1744.88 ± 3.74 vs candidate 1730.66 ± 3.87 (near-parity,
barely non-overlapping).

## A/B deltas and acceptance gates

| metric | MARLIN ref | candidate | delta | gate | verdict |
|---|---:|---:|---:|---|---|
| pp (prefill) t/s | 1746.56 | 1730.61 | **−0.91%** | ≥ +5% uplift | **FAIL** |
| tg (decode) t/s | 40.99 | 37.35 | −8.88% | ≤ 3% regression | FAIL (noisy/overlapping — not decisive) |
| e2e_ttft ms | 38896.64 | 39249.73 | +0.91% | ≤ 5% regression | PASS |

Arithmetic: `(1730.61/1746.56 − 1)×100 = −0.91%`; `(37.35/40.99 − 1)×100 = −8.88%`;
`(39249.73/38896.64 − 1)×100 = +0.91%`.

## Stability / health during the window

- Health 200; head+worker restart 0/0; PSI ~0; swap flat; memory stable.
- Route proof both ranks symmetric: MARLIN active, fp8_ds_mla, DeepGEMM PDL (SM121
  indexer), MTP n=1, KV 159,445, graph `[2]`; FlashInfer sparse-MLA prefill route
  marker present; Triton sparse-MLA prefill invoked = 0.
- Correctness (EN/KO/ZH/arithmetic/reasoning) all correct, deterministic, valid UTF-8,
  no garble/NaN.

## Conclusion

The FlashInfer sparse-MLA prefill candidate is **performance-neutral** at 64K c1
(prefill parity, −0.91%). It clears functional/stability gates but fails the +5%
prefill-uplift gate. **Not promoted.** The decode figure is noisy and not a decisive
regression claim. Raw llama-benchy output and sealed runtime evidence are preserved
locally on the build host and are not committed to this repository.
