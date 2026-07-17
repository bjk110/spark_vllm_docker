# DeepSeek-V4-Flash-DSpark — speculative-decoding investigation on vLLM `72261a7` (closure)

> **Status: Experimental / NOT VALIDATED — investigation CLOSED, no candidate promoted.**
> The DS2 arc enabled DSpark speculative decoding (`method=dspark`) end-to-end on dual DGX Spark
> (GB10 sm_121, TP=2 `mp`/RoCE) via incremental installed-package patches, found and fixed two
> confirmed semantic defects, and could not move speculative acceptance. Position-1 acceptance
> stayed at approximately **8-10%** across four controlled runs, three images and two `k` values.
> **DS2D14 is the preferred experimental baseline**, **DS2D13 is the fallback**, **DS2D12 is the
> rollback**, and **k=3 is the only validated runtime envelope**. **No production or benchmark
> claim is made.** **Production serving does not use DSpark speculative decoding** and was untouched
> throughout. Further behavioral work requires a new separately approved experimental arc.

## Result

`DS2ARC_CLOSED_EXPERIMENTAL_NOT_VALIDATED` — this describes the **investigation status only**. It is
not a Git, image, or production promotion statement.

The arc's two shipped fixes are **semantically correct and runtime regression-free**. Neither was the
dominant cause of the low acceptance rate. The remaining leading risk (R1, below) is **not a precise
patch target**, so the arc closes with no evidence-backed narrow change left to make.

## Scope and identity

- **vLLM lineage:** `v0.24.0.dev0+dsv4.pr41834.72261a7` (jasl PR #41834 @ `72261a7`). The pinned
  commit was later force-pushed away upstream, so from-source rebuilds of this lineage are blocked;
  all DS2 images are installed-package patches layered on the existing chain.
- **Model:** `DeepSeek-V4-Flash-DSpark` (`deepseek-ai/deepseek-ai_DeepSeek-V4-Flash-DSpark`, 48
  shards). Declares `dspark_block_size=5`, `n_predict=4`, `dspark_target_layer_ids=[40,41,42]`,
  `num_hidden_layers=43` ⇒ 3 draft blocks (layers 43/44/45).
- **Images are local experimental artifacts on spark01/spark02 only.** No GHCR promotion occurred;
  none was pushed, retagged, or published.
- **Production baseline (untouched):** the SM121 DeepGEMM FP8-Q indexer path — see
  [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md). It does not
  enable speculative decoding, so **no production configuration change is required by any finding
  here.**

## Baseline hierarchy

| Role | Image tag | spark01 | spark02 | Notes |
|---|---|---|---|---|
| **Preferred experimental baseline** | `vllm-spark:ds2d14-context-kv-per-group-slotmap-static-exp-3259041` | `cd679693ecc3` | `c920fb82a4b3` | Semantically corrected; runtime regression-free. **Experimental, not production.** |
| **Fallback** | `vllm-spark:ds2d13-eagle3-aux-offbyone-static-exp-3259041` | `5619be6c3c99` | `b948748d7483` | Carries the C1 aux correction; runtime validated. **The layer-44 context-KV mismatch remains present in this image.** |
| **Rollback** | `vllm-spark:ds2d12-dspark-draft-precompute-reduce-static-exp-3259041` | `a101c35c9e55` | `43b3f400d34d` | Pre-DS2D13/DS2D14 state; preserves the earlier shape-path corrections. |

Image IDs differ per node **by design**: `docker save | docker load` fails with "max depth exceeded"
on these 126-layer images, so each node builds independently from a byte-identical context and
equivalence is proven by **content hash**, not image ID.

"Preferred experimental baseline" means *the best image to start from for future DSpark research*. It
does **not** mean production baseline, and it does not imply readiness, validation, or promotion.

### Validated runtime envelope — k=3 only

`k=3` · `MAX_MODEL_LEN=16384` · `MAX_NUM_BATCHED_TOKENS=8192` · TP=2 · `DISTRIBUTED_BACKEND=mp` ·
RoCE (10.10.10.1/.2) · `MAX_NUM_SEQS=1` · KV **4 GiB** · KV dtype **fp8** · `--enforce-eager` ·
`--moe-backend marlin` · `draft_sample_method: probabilistic`. GPU KV capacity **21,431 tokens**.

**k=4 is non-baseline**: runtime-functional but with no quality benefit and a ~54% larger draft cost.
Not recommended. Omitting `num_speculative_tokens` is **structurally impossible** in this build (see
superseded conclusion #2).

## Stage-by-stage summary

| Stage | Objective | Outcome | Runtime? | Evidence (local, not committed) |
|---|---|---|---|---|
| DS2D2–D12 | Enable DSpark e2e (config mapping, proposer allowlist, EAGLE3 aux, multigroup KV, slot-map guard, copy alias, reduce/project) | e2e works; accept ~3% | yes | per-stage dirs |
| DS2J | Limited perf observation, DS2D12, k=3 | `DS2J_LIMITED_PERF_OBSERVATION_PASS_LOW_ACCEPT_RATE` | yes | `ds2j-dspark-limited-perf-observation-20260716T152429` |
| DS2A11 | Read-only upstream/reference comparison | `DS2A11_IMPLEMENTATION_MISMATCH_CANDIDATE_FOUND` — **C1**: EAGLE3 aux captured one layer late | no | `ds2a11-dspark-upstream-reference-comparison-20260716T193805` |
| DS2D13 (static) | Fix C1 (`idx` → `idx + 1`) | `DS2D13_EAGLE3_AUX_OFFBYONE_STATIC_BUILD_PASS` | no | `ds2d13-eagle3-aux-offbyone-static-build-20260716T203458` |
| DS2D13-SMOKE | Measure C1 fix, k=3 | `DS2D13_SMOKE_PASS_ACCEPT_RATE_WEAKLY_IMPROVED` — non-material | yes | `ds2d13-smoke-accept-remeasure-20260716T221604` |
| DS2A12 | Read-only semantic analysis of draft-KV / proposal inputs | `DS2A12_TRAINING_ASSUMPTION_MISMATCH_RISK_FOUND` — **R1**; refuted 6 other hypotheses | no | `ds2a12-dspark-draft-kv-proposal-alignment-20260716T233129` |
| DS2K4 | Config-only k=4 (block-layout hypothesis) | `DS2K4_CONFIG_ONLY_K4_PASS_ACCEPT_UNCHANGED_LOW` + `K4_AUTO_DEFAULT_RUNTIME_REFUTED` | yes | `ds2k4-dspark-config-only-k4-20260717T000754` |
| DS2A13 | Read-only WRITE/READ KV key-alignment verification | `DS2A13_LAYER44_GROUP1_CACHE_MISMATCH_FOUND` — **a precise source defect** | no | `ds2a13-dspark-kv-write-read-key-alignment-20260717T090751` |
| DS2D14 (static) | Fix the layer-44 context-KV routing | `DS2D14_CONTEXT_KV_PER_GROUP_SLOTMAP_STATIC_BUILD_PASS`; static routing **24/24** | no | `ds2d14-context-kv-per-group-slotmap-static-build-20260717T101500` |
| **DS2D14-SMOKE** | Measure the layer-44 fix, k=3 | **`DS2D14_SMOKE_PASS_ACCEPT_UNCHANGED_LOW`**; mapping contract PASS; 0 regressions | yes | `ds2d14-smoke-accept-remeasure-20260717T133500` |

All evidence directories carry verified `SHA256SUMS.txt` manifests and are retained on the build host.
They are **local-only** and are not committed to this repository.

## Acceptance summary

| Run | Image | k | Aggregate | **Position-1** | Pos-2 | Pos-3 | Pos-4 | Mean accepted length | Accepted / Drafted |
|---|---|---|---|---|---|---|---|---|---|
| DS2J | DS2D12 | 3 | 3.10% | 8.36% | 0.85% | 0.10% | — | 1.093 | 166 / 5,355 |
| DS2D13-SMOKE | DS2D13 | 3 | 3.41% | **9.46%** | 1.03% | 0.08% | — | 1.106 | 179 / 5,244 |
| DS2K4 | DS2D13 | 4 | 2.59% | 9.65% | 0.79% | 0.21% | 0.04% | 1.107 | 210 / 8,100 |
| **DS2D14-SMOKE** | **DS2D14** | 3 | **3.62%** | **9.42%** | 1.77% | 0.19% | — | **1.114** | 234 / 6,462 |

**Position-1 acceptance remained approximately 8-10%** — the 8.36–9.65% band, across four runs, three
images and two `k` values.

### Interpretation caveats

- **DS2D14 did not materially change position-1**: −0.04 pp vs DS2D13. Aggregate +0.21 pp and mean
  accepted length +0.008 are likewise immaterial.
- **Sub-1-percentage-point n=1 differences are non-attributable.** This path is **nondeterministic
  run-to-run even at temperature 0** (different outputs, different finish-reason mixes, 5,244 vs
  6,462 drafted tokens on byte-identical prompts).
- **The k=4 aggregate is not directly comparable.** Its decline (3.41% → 2.59%) is **denominator
  dilution**: k=4 adds a fourth drafted position accepted 0.04% of the time, so Drafted rose ~54%
  while Accepted rose ~17%.
- **Convention:** aggregate = `Accepted/Drafted` (weighted); per-position and mean accepted length =
  the unweighted mean of per-interval `metrics.py` values. Verdicts were re-checked under both
  conventions and are robust (position-1 steps-weighted: 8.35 / 9.04 / 9.28 / 9.10).
- At mean accepted length ≈ 1.11 the drafter accepts ~0.11 extra tokens per step and **does not pay
  for its own draft cost**, irrespective of any fix.
- **No benchmark conclusion is justified.** These are limited internal observations, n=1, concurrency
  1, short outputs only. No long-context or concurrency testing was performed.

## Confirmed fixes

Two defects were **confirmed and corrected**. Neither is characterised as a failed patch: both are
semantically correct and runtime regression-free; what they did not do is explain the acceptance rate.

### DS2D13 — EAGLE3 aux capture off-by-one (C1)

- **Defect:** the aux backport captured hidden states **after** `layer()` while testing
  `idx in aux_hidden_state_layers`, so with `aux_hidden_state_layers=(40,41,42)` it captured the
  output of layers 40/41/42 instead of the residual stream **entering** them (= output of 39/40/41).
  Independently corroborated by the repository's own `config/speculative.py`, which derives the
  intended 0-based set `[39,40,41]` via `target_layer_ids = [i-1 for i in dspark_target_layer_ids]`.
- **Correction:** one condition — `if idx + 1 in self.aux_hidden_state_layers` — in
  `models/deepseek_v4/nvidia/model.py` (hash `b09b9eed` → `42b9dc42`), plus a regression-guard
  comment.
- **Runtime:** aggregate 3.10 → 3.41%, position-1 8.36 → 9.46% — **not material, not separable from
  run-to-run variation at n=1**. Zero regressions.
- **Causal impact:** C1 was **necessary but insufficient**; it was **not** the dominant cause.
- **Retention:** **fallback (DS2D13)**. Retained in DS2D14 as well.

### DS2D14 — layer-44 context-KV per-group slot routing

- **Defect (`DS2A13_LAYER44_GROUP1_CACHE_MISMATCH_FOUND`):** the active DFlash context-precompute
  caller passed a **single Tensor** slot mapping, so the draft model's Mapping-aware per-prefix branch
  was unreachable and **every** draft layer wrote context KV through the **primary group's** slot
  mapping (derived from one block table, captured only for the primary KV group). Draft READ is
  per-group. Draft layers span KV groups **[1, 2]** with **primary = group 2**, so layers 43 and 45
  were aligned while **layer 44 (group 1) wrote context KV at group-2 addresses and read at group-1
  addresses**. Uniform KV specs and a globally shared block pool kept the wrong write **in bounds**,
  hiding it from shape checks and runtime exceptions. DS2D9 had routed **query** KV per group but left
  **context** KV on the single-group path — an incomplete fix.
- **Correction:** one file — `v1/spec_decode/dspark.py` (`fbc68e0b` → `e2465b83`). `DSparkProposer`
  gains `_get_context_slot_mapping` and a `build_model_inputs_first_pass` override that passes a
  per-layer Mapping. It **reuses DS2D9's authoritative per-group slot mappings**, **preserves the
  primary-group kernel-buffer path byte-identically** (layers 43/45 unchanged), **routes layer 44
  through group 1**, **preserves generic DFlash** and **DS2D10 query routing**, and **fails fast** on
  a missing group or insufficient mapping length (no silent fallback). The draft model needed no
  change — it already accepted `Mapping[str, Tensor]`; only the caller was wrong.
- **Static:** routing **24/24 PASS**; image equivalence PASS on both nodes; no runtime occurred in the
  static-build task.
- **Runtime:** `DS2D14_RUNTIME_MAPPING_CONTRACT_PASS` — **0 fail-fast triggers**, with patch-path
  execution confirmed **after actual drafting** (37 SpecDecoding intervals plus the DS2D12 precompute
  marker), which closes the one layout assumption static analysis could not. **0 runtime regressions**;
  output correctness, route stability and memory stability all PASS.
- **Causal impact:** **`LAYER44_CONTEXT_KV_MISMATCH_NOT_DOMINANT`** — position-1 −0.04 pp.
  **The fix did not materially improve speculative acceptance.**
- **Retention:** **preferred experimental baseline (DS2D14)**.

> **The unchanged acceptance does not make DS2D14 unnecessary.** A silent wrong-slot write is a defect
> regardless of its weight on any metric; leaving it in place would keep a known-incorrect path in the
> experimental baseline and would contaminate every future measurement on this lineage. What the
> result changed is our belief about **causal weight**, not about correctness.

## Remaining unresolved risk — R1 (not a patch target)

The leading unresolved risk is the DS2A12 **R1** finding:

- the model declares **`dspark_block_size = 5`**, which the current draft model **does not
  structurally consume** (it has zero references to it);
- the current implementation **inherits the DFlash architecture** — its query block is `1 + k` slots,
  while the evolved upstream builds exactly `k`;
- DFlash attention is **non-causal**, so block width perturbs slot 0 — the slot that produces the
  position-1 draft token;
- the **evolved upstream rewrites proposer, draft model and sampler as a matched set**;
- **neither implementation has a validated reference acceptance baseline**
  (`DEAD_UPSTREAM_CODE_HIGH_REFERENCE_RISK`: the shipped DSpark was never runnable, and the evolved
  reference has itself never produced a validated accept rate and is internally inconsistent).

**Why R1 does not justify further work now:**

- R1 is **not a precise patch target** — it names an architectural mismatch, not a located line with a
  known-correct replacement.
- **DS2K4 tested only R1's numeric `1+k = 5 == dspark_block_size = 5` subset** and found no benefit.
- **No evidence-backed narrow behavioral patch remains.**
- A matched-pair rewrite would be a **separate high-risk project** against an unvalidated reference,
  with unknown expected benefit.

Two independent attempts have now failed to move position-1 (DS2K4 on R1's config subset; DS2D14 by
removing the strongest competing KV explanation). Eliminating the KV candidate leaves R1 standing with
less competition — it does **not** prove R1.

## Explicit non-claims

This document does **not** claim, and no evidence here supports:

- production readiness, or any production change;
- validated performance, or any benchmark result;
- that low acceptance has been fixed or explained;
- upstream compatibility;
- repeatability (every runtime result is **n=1**);
- suitability for users, or any recommendation to adopt DSpark;
- long-context or concurrency behaviour (neither was tested).

## Future re-entry criteria

A future DSpark arc on this lineage should start **only** if at least one of these exists:

- a **validated upstream/reference implementation**;
- **official runtime acceptance data** for this model;
- a **precise new semantic mismatch** (a located source target with a known-correct replacement);
- a **reproducible first-token trace** demonstrating incorrect architecture;
- a **complete proposer/draft/sampler matched-set design** with a rollback plan;
- a **strong upstream change** that explicitly supports this model and runtime version.

Any such task must define: exact hypothesis · exact source target · independent baseline ·
single-variable validation · rollback image · memory guard · stop conditions · no production impact.

**Low acceptance alone is not a reason to re-open this arc.** Further behavioral work requires a new
separately approved experimental arc.

## Artifact disposition (statement of current state — nothing executed by this document)

| Artifact class | Disposition |
|---|---|
| **Documentation** | This closure document — **keep + commit** (documentation-only). |
| **Experimental images** (DS2D12 / DS2D13 / DS2D14, both nodes) | **Retain, local-only.** Do not promote, retag, or push. |
| **Local evidence directories** | **Local only** — retained with verified `SHA256SUMS.txt` manifests; not committed. |
| **Disposable runtime configs** (DS2 `.env` files inside evidence dirs) | **Disposable** — kept for reproducibility; not part of any serving path. |
| **Production presets** | **Untouched.** Production does not use speculative decoding. |

## Cross-references

- Production identity/routing/rollback:
  [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md).
- Closure precedent and method (attribution arc closed without promotion):
  [`dsv4-sparse-mla-b3-investigation-closure.md`](dsv4-sparse-mla-b3-investigation-closure.md).
- v0.23-stack / PR #41834 build context:
  [`deepseek-v4-v023-stack-pr41834.md`](deepseek-v4-v023-stack-pr41834.md).

## Durable engineering lessons from this arc

1. **"No crash + valid shapes + no exception" is not evidence of semantic correctness.** Uniform KV
   specs plus a globally shared block-id space made a wrong-group slot **in-range**, so the defective
   write was well-formed and landed where nothing would read it. Three separate tasks rated that path
   "runtime-safe, quality-unproven" on exactly this non-evidence before DS2A13 located the defect.
2. **A startup-only gate is not a gate** when the patched code runs later. Verify **after** the path
   executes, and prove that it executed (a marker or metric), rather than concluding from an absence
   of errors.
3. **A static conclusion can be wrong at runtime.** DS2K4 overturned a confident static claim
   (`K4_AUTO_DEFAULT_RUNTIME_REFUTED`) with one startup. Prefer a cheap runtime probe when the cost is
   a single start.
4. **A negative result on a proven defect is still a real result** — record it as "not the dominant
   cause", never as "the patch was unnecessary".
5. **Record the command, not just the number.** An image-count figure became unverifiable because its
   counting convention was not recorded.
6. **A hash match on empty input is a false pass.** Verify the input is non-empty before trusting a
   hash comparison.
7. This path is **nondeterministic run-to-run even at temperature 0** ⇒ n=1 deltas under ~1 pp are not
   attributable.
