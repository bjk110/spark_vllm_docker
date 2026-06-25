# DeepSeek-V4-Flash `max-num-batched-tokens=8192` — Validated Candidate

**Status: VALIDATED_CANDIDATE (not production).** Production promotion remains a separate,
explicitly-authorized step and has not occurred.

This document records the independent, harness-corrected cold-start reproduction and the
60-minute mixed-context stability validation that promote the `max-num-batched-tokens=8192`
prefill configuration from experimental to validated candidate.

## Lineage

| Stage | `max-num-batched-tokens` | 131K prefill (TTFT-derived) | Classification |
|---|---|---|---|
| 2048 baseline (LC3 reference) | 2048 | ~1264 t/s | committed validated preset `d116f132…` |
| 4096 variant | 4096 | ~1315.3 t/s (+4.06%) | `PREFILL_VARIANT_NO_MATERIAL_GAIN` |
| 8192 variant (campaign) | 8192 | ~1342.5 t/s (+6.21%) | `PREFILL_VARIANT_MODERATE_GAIN` → `PROMOTE_PREFILL8192_AS_VALIDATED_CANDIDATE` |

The single semantic change from the committed validated preset is the three context/KV deltas:
`MAX_MODEL_LEN 8192→135168`, `--kv-cache-memory-bytes 2147483648→4294967296`,
`MAX_NUM_BATCHED_TOKENS 2048→8192`. Nothing else differs.

## Previous inconclusive run and exact cause

The first independent cold reproduction (2026-06-25, classification `PREFILL8192_INCONCLUSIVE`)
is preserved. The original inconclusive raw artifacts are preserved locally and are
intentionally not committed to the repository. All four required retrieval
probes passed, but a **non-required 64K Korean probe** — auto-run by the previous retrieval
harness, which ran en/ko/multi per depth — missed its needle (HTTP 200, accounting valid,
graph fallback/recapture 0, memory/paging stable). The completion was off-task Python code,
not the identifier.

Root cause (`HARNESS_PROMPT_CONSTRUCTION_LIKELY`, with `HARNESS_OUTPUT_BUDGET` contributing):
the retrieval prompt used a weak response contract in raw `/v1/completions` mode and a 32-token
single-needle output budget. The successful probes happened to emit the needle first; the
64K-ko prompt let the model wander into code with no budget to recover. The harder 131K Korean
probe passed, confirming the candidate's Korean retrieval capability.

## Harness correction

Modifications to the untracked driver (`scripts/diag/dsv4_mtp1_fullgraph_long_context_probe.py`,
new SHA-256 `d67075b46a1bfd06e380d6db6baefa40b23129b850372a9a96d39d385e76f2b1`):

- **Probe matrix** — a new `retrieval_matrix` phase runs exactly the required probes:
  64K English, 131K English, 131K Korean/Unicode, 131K multi-position (20/50/80%). No
  automatic 64K Korean probe.
- **Output budget** — every retrieval probe uses `max_tokens=128` (was 32 single / 64 multi).
- **Response contract** — `NEEDLE_QUESTION` / `NEEDLE_QUESTION_KO` / `MULTI_QUESTION` now require
  the identifier(s) only, on the first output line, no code, no reasoning, no prose.
- **Validator** — `validate_single` / `validate_multi`: NFKC + whitespace normalization, exact
  identifier match, reject absent / partial / wrong / unexpected-synthetic; record
  first-line / extra-text / contract-followed (and multi ordering) **separately**, so format
  compliance never relaxes retrieval correctness. Multi ordering is a format signal, not a
  correctness gate.
- **Unit tests** — `selftest` phase: 14 offline cases (exact, whitespace, first-line+extra,
  code-block, missing, wrong, partial, unexpected-synthetic, Korean normalization, multi
  exact/reorder/missing/unexpected) plus the previous failed completion as a negative
  regression fixture. No API calls. All 14 pass.

The 60-minute stability runner (`scripts/diag/dsv4_prefill8192_stability_runner.py`) is a
separate orchestration harness; it imports the frozen driver read-only and does not modify it.

## Runtime identity (independent cold start, 2026-06-25)

- image config `sha256:4c41950c…` on both nodes; manifest `sha256:2f4a9628…`; source `72261a7…`
- effective `max-num-batched-tokens` 8192, MML 135168, fixed KV 4,294,967,296 bytes, fp8
- KV capacity **159,445 tokens** (1.18x @135,168); margin over the largest 131K sequence
  ~28,274 tokens (~21.56%), 109 spare 256-token blocks; no capacity warning
- graph FULL_DECODE_ONLY, capture `[2]`, both ranks captured, fallback 0 / recapture 0
- MTP n=1, 39 params both ranks, fresh init
- transport NET/IB `rocep1s0f0:1/RoCE`, GID index 3, NET/Socket fallback 0

## Cold reproduction — `PREFILL8192_COLD_REPRO_PASS`

- correctness 6/6 + 131K needle; retrieval matrix 4/4 needles recovered (identifier on first
  line, `contract_followed=true`; each then emitted filler within the 128-token budget —
  recorded as a format warning, not a correctness failure)
- performance within ±5% of the campaign: 32K 1657.7 (+0.27%), 64K 1494.8 (+0.21%),
  131K 1344.3 t/s (+0.13%); 131K decode 21.38 t/s (−0.28%); all CV <1%
- accounting 0 violations; measured-window memory stable (MemAvailable 26.8–27.4 GiB, swap
  Δ0, PSI 0.00, pswpout 0)

## 60-minute stability — `PREFILL8192_60MIN_STABILITY_PASS`

- 19 cycles, 79/79 requests ok, retrieval 3/3
- prefill drift −0.39%..+0.15% (no degradation, no collapse); decode flat 21.4–21.7 t/s
- MTP acceptance per-context mean 85.7 / 86.9 / 93.1% (min 77.8 / 80.3 / 84.1%, all ≥75%)
- graph fallback/recapture 0; accounting 0 violations
- pswpout 0, PSI some10 max 0.00, MemAvailable flat (spark01 26.7–27.4, spark02 28.7), both
  ranks healthy

## Validated envelope

concurrency 1 only · prompt up to 131,072 tokens (max tested ~131,171) · output allowance
~128 tokens · prefix cache disabled · fixed KV 4 GiB fp8 · MTP n=1 · FULL_DECODE_ONLY ·
capture `[2]` · TP=2 mp · NET/IB over RoCE.

## KV headroom caveat

The 8192 KV token capacity (159,445) is tighter than 4096 (275,742, 2.04x) and 2048 (433,860).
It is sufficient for the validated single-concurrency ≤131K envelope, but **runtime KV
headroom must be revalidated for any concurrency increase or context expansion**.

## Rollback

The formal rollback chain is the committed validated baseline and its committed rollback
ladder. A 4096 `max-num-batched-tokens` configuration was a tested safe intermediate
experiment (`PREFILL_VARIANT_NO_MATERIAL_GAIN`); it is retained locally as disposable
experimental material and is not part of the formal candidate package.

- to 2048 committed validated baseline:
  `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env` (SHA-256 `d116f132…`)
- committed rollback ladder: L1 `87ad3920…`, L2 `bffea158…`

## Promotion note

This candidate is validated for the envelope above only. Production promotion is a separate,
explicitly-authorized decision and has not been made here.
