# DeepSeek-V4-Flash — B3 sparse-attention performance investigation (closure)

> **Status: CLOSED — investigation complete, candidate NOT promoted.** The B3 arc
> built, repaired, and GPU-validated an experimental FlashInfer CUDA sparse-MLA
> *prefill-only* candidate against the current MARLIN + SM121-DeepGEMM-indexer
> production baseline, to test whether backporting the historical "unholy-fusion"
> sparse-MLA prefill kernel yields a material (≥ +5%) 64K prefill uplift. The
> candidate is **functionally correct and stable** but **performance-neutral**
> (64K c1 prefill parity, −0.91% vs MARLIN). It is **not promoted**. Production was
> untouched throughout and is restored/verified. This document is the authoritative
> record; per-stage documents remain for detail.

## Result

`H1Z_B3_INVESTIGATION_CLOSED` — primary `FLASHINFER_SPARSE_MLA_PREFILL_PERFORMANCE_NEUTRAL`.

The B3 conclusion **confirms** the earlier H1C / H1O attribution: the historical
unholy prefill advantage is **not** attributable to the sparse-MLA prefill kernel
alone. The current native SM121 DeepGEMM indexer explains a substantial portion of
the performance recovered since the historical baseline, and it is **already in the
current production baseline**. However, the remaining historical unholy advantage is
not attributable to either B12X MoE or FlashInfer sparse-MLA prefill in isolation;
its exact source remains unverified and may involve broader runtime, scheduler,
chunking, or pipeline interactions. What B3I establishes directly is narrower:
swapping only the sparse-MLA prefill kernel from the production Triton path to the
vendored FlashInfer CUDA path buys no prefill time at 64K c1.

## Scope and baseline

- **Production baseline (unchanged):** image digest `ade810fd`, config `fa83457d`,
  vLLM `72261a7`, FlashInfer 0.6.12, MARLIN MoE, native SM121 DeepGEMM FP8-Q prefill
  indexer, Triton sparse-MLA prefill/decode, dual-node TP=2, `mp` backend, MTP n=1,
  fixed 4 GiB FP8 KV (159,445 tokens), `MAX_NUM_SEQS=1`, FULL_DECODE_ONLY graph
  capture `[2]`.
- **Candidate (experimental, default-OFF):** the production image with a vendored
  FlashInfer sparse-MLA prefill kernel (`dsv4_sparse_mla_sm120`, adapted to
  FlashInfer 0.6.12 JIT, arch-adaptive `compute_121a`) plus a narrow gated vLLM
  branch that redirects **only** DeepSeek-V4 sparse-MLA prefill when
  `VLLM_DSV4_SPARSE_MLA_PREFILL=flashinfer_sm12x`. MARLIN MoE, the SM121 indexer,
  dense FP8, and the Triton sparse-MLA decode/default path are preserved. The
  official model decode class (`VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE`, requires
  FlashInfer ≥ 0.6.13) was kept **disabled** throughout.
- Candidate image: `vllm-spark:v023-dsv4-72261a7-sparse-mla-prefill-vendorfix-exp-ff4477f4878c`
  (id `f17c8d51`).

## Stage-by-stage summary

| Stage | Objective | Outcome | Detail doc |
|---|---|---|---|
| B3A | Unholy performance forensics / attribution | Residual unholy prefill advantage ~+11.5% at 64K attributed (via H1W matched-nsys) to the SM120 sparse-attention pipeline, NOT B12X/MoE | [`dsv4-unholy-perf-forensics-b3a.md`](dsv4-unholy-perf-forensics-b3a.md) |
| B3B | Feasibility of a prefill-only FlashInfer sparse-MLA backport | Feasible; vendoring + gated-branch approach chosen | (in B3C doc) |
| B3C | Build + static validation of the prefill-only candidate | Built from production image; no-model static validation passed; synthetic GPU test resource-blocked (prod live) | [`dsv4-sparse-mla-prefill-b3c.md`](dsv4-sparse-mla-prefill-b3c.md) |
| B3D | First GPU runtime attempt | Crash in dummy warmup: `assert swa_metadata.prefill_swa_indices is not None`; also CUDA 701 register-overflow when JIT verbose (-O0) was set (a kernel launch-resource error, not a memory failure) | (superseded by B3F) |
| B3E | Audit official upstream `_forward_prefill` contract | Contract identified (self-compute-on-None SWA indices, C4A cu_base rebase, query slice, uint8 cache view, field-specific validation) | (folded into B3F) |
| B3F | Implement upstream-aligned warmup/metadata contract | Adapter re-implemented against PR #41834 @ `444fe3ac8b`; fixes the B3D warmup crash; 30 CPU contract tests pass; synthetic GPU resource-blocked | [`dsv4-sparse-mla-prefill-warmup-b3f.md`](dsv4-sparse-mla-prefill-warmup-b3f.md) |
| B3G | GPU runtime of the warmup-fixed candidate | Crash: `ModuleNotFoundError: dsv4_sparse_mla_sm120.jit` from a function-local import in the ≤64-row decode-dispatch cache loader | (root-caused in B3H) |
| B3H | Repair + audit the vendored package namespace | Two stale donor-relative function-local imports in `wrapper.py` (`.jit.env`→`flashinfer.jit.env`; `.autotuner`→`flashinfer.autotuner`) re-anchored to stock FlashInfer 0.6.12; AST namespace-integrity test added; candidate image rebuilt on both nodes | [`dsv4-sparse-mla-vendor-import-audit-b3h.md`](dsv4-sparse-mla-vendor-import-audit-b3h.md) |
| B3I | Clean-GPU dispatch validation + 64K c1 A/B sentinel | Full model started for the first time in the arc; dispatch matrix all-pass; **64K prefill parity (−0.91%) → below the +5% gate** | this document + benchmark record |

## B3I decisive result — 64K c1 A/B sentinel

Same-window median of 3 runs, llama-benchy 0.3.8, depth 65536, pp 2048, tg 32,
concurrency 1, latency-mode generation. MARLIN reference and candidate measured in
the same boot window, identical harness. Full record:
[`../benchmarks/llama-benchy/results_dsv4-b3i-sparse-mla-prefill-vs-marlin-64k-c1.md`](../benchmarks/llama-benchy/results_dsv4-b3i-sparse-mla-prefill-vs-marlin-64k-c1.md).

| metric | MARLIN ref | candidate | delta | acceptance gate | verdict |
|---|---|---|---|---|---|
| pp (prefill) t/s | 1746.56 | 1730.61 | **−0.91%** | ≥ +5% uplift | **FAIL** |
| tg (decode) t/s | 40.99 | 37.35 | −8.88% (noisy, overlapping) | ≤ 3% regression | FAIL (not decisive) |
| e2e_ttft ms | 38896.64 | 39249.73 | +0.91% | ≤ 5% regression | PASS |

MARLIN pp 1744.88 ± 3.74 vs candidate 1730.66 ± 3.87 (near-parity, barely
non-overlapping). Health 200, restart 0/0, PSI ~0, memory stable throughout.

**Interpretation caveats (do not overstate):**

- The prefill result is a **near-parity / no-uplift** finding. The candidate is
  **not** faster; treat −0.91% as parity, **not** as an improvement.
- The decode −8.88% figure is **noisy and overlapping** across runs and is **not**
  a precisely established regression. The decisive B3 outcome is the prefill
  near-parity, not a decode claim.
- This experiment kept the official FlashInfer sparse-MLA **decode** class disabled
  (needs FlashInfer ≥ 0.6.13). Nothing here implies FlashInfer ≥ 0.6.13 would be
  automatically faster.

## Functional validation (candidate is correct, just not faster)

The B3H vendoring fix was **definitively validated on GPU** in the exact
decode-dispatch path that crashed B3G:

- Dispatch matrix T=1/16/64/65/128 all pass — the ≤64-row decode-dispatch cache
  loader (`_decode_dsv4_maybe_load_cache`) runs with **no ModuleNotFoundError**;
  outputs finite + deterministic; Triton fallback = 0; JIT compiled once then reused.
- Full-model startup succeeded (first readiness in the entire B3 arc); the
  "mixed tokens=16, prefill 8192" warmup (the B3G crash shape) **completed**.
- Route proof, both ranks symmetric: **MARLIN active, fp8_ds_mla, DeepGEMM PDL
  (SM121 indexer), MTP n=1, KV 159,445, graph capture `[2]`**; FlashInfer sparse-MLA
  prefill route marker present; **Triton sparse-MLA prefill invoked = 0** (no silent
  fallback); official DECODE flag off; B12X env unset.
- Correctness: EN/KO/ZH/arithmetic/reasoning all correct, deterministic repeats,
  valid UTF-8, no garble/NaN.

## Verdict

**Do NOT promote.** The candidate is a sound, correct, stable implementation of a
FlashInfer CUDA sparse-MLA prefill path for DeepSeek-V4-Flash on SM121, but it
provides **no prefill performance advantage** over the current production Triton
sparse-MLA prefill + native SM121 DeepGEMM indexer at 64K c1. There is no
performance case for adopting it, and adopting it would add a vendored CUDA-kernel
maintenance surface for zero benefit. The B3 arc is closed.

This closes the B3 arc on the specific question it set out to answer (does the
FlashInfer sparse-MLA prefill kernel add prefill throughput at 64K c1 — no). It does
not fully explain the historical unholy advantage. The current native SM121 DeepGEMM
indexer explains a substantial portion of the performance recovered since the
historical baseline. However, the remaining historical unholy advantage is not
attributable to either B12X MoE or FlashInfer sparse-MLA prefill in isolation. Its
exact source remains unverified and may involve broader runtime, scheduler,
chunking, or pipeline interactions.

## Future reconsideration conditions

Re-open this investigation only if one of the following changes the premise:

1. **FlashInfer ≥ 0.6.13 ships the official `_sparse_mla_sm120` runner**, enabling
   the official `DeepseekV4FlashInferSM120Attention` prefill **and** decode path
   (PR #41834 lineage). The decode path was out of scope here; a combined
   prefill+decode FlashInfer route is a different measurement and could shift the
   balance. Re-run the 64K c1 A/B (and add a decode-focused window) at that point.
2. **A larger concurrency / longer-context envelope** (concurrency > 1, or context
   > 131K) where the sparse-MLA prefill kernel's scaling differs from Triton. B3I
   measured only 64K c1; a different envelope is a separate question.
3. **A profiling result that localizes the residual unholy delta to sparse-MLA
   compute specifically** (as opposed to the indexer or the scheduler). B3A/H1W
   attributed the delta to the SM120 sparse-attention *pipeline*; if a future
   profile isolates a sparse-MLA-*compute* component that this prefill kernel
   actually accelerates, the trade-off changes.
4. **An upstream vLLM change** that makes the FlashInfer sparse-MLA prefill path
   the default/maintained path for DeepSeek-V4 on SM121, removing the vendoring
   maintenance burden.

Absent one of these, the vendored candidate should remain experimental/archived.

## Artifact disposition plan (proposal — not executed here)

None of the runtime artifacts below are integrated into the production path by this
closure. They are retained as experimental references for a possible future
re-open. Disposition is a proposal for a later, separately authorized action.

| Artifact class | Files | Proposed disposition |
|---|---|---|
| **Documentation (this arc)** | `docs/dsv4-unholy-perf-forensics-b3a.md`, `docs/dsv4-sparse-mla-prefill-b3c.md`, `docs/dsv4-sparse-mla-prefill-warmup-b3f.md`, `docs/dsv4-sparse-mla-vendor-import-audit-b3h.md`, this closure doc | **Keep + commit** (documentation-only integration) |
| **Benchmark record** | `benchmarks/llama-benchy/results_dsv4-b3i-sparse-mla-prefill-vs-marlin-64k-c1.md` | **Keep + commit** |
| **Reusable diagnostic tests** | `scripts/diag/tests/test_dsv4_sparse_mla_namespace.py`, `test_dsv4_sparse_mla_prefill_contract.py`, `test_dsv4_sparse_mla_prefill_gate.py`, `test_dsv4_sparse_mla_prefill_warmup.py` | **Retain** (CPU-only, no runtime footprint); commit optional in a later authorized change |
| **Experimental runtime source** | `patches/vllm/dsv4-sparse-mla-sm120/` (vendored package), `patches/vllm/dsv4-sparse-mla-prefill/` (adapter/env/gate) | **Retain as experimental**; do NOT integrate into the production build path |
| **Experimental Dockerfiles** | `dockerfiles/active/Dockerfile.v023-dsv4-sparse-mla-prefill-exp`, `…-warmup-exp`, `…-vendorfix-exp` | **Retain as experimental** |
| **Disposable runtime config** | `presets/deepseek-v4-b3{a,c,d,f,h}-*.env`, `docker-compose.b3d.yml`, `docker-compose.b3f.yml`, `dockerfiles/active/b3c_protected_hashes.txt` | **Disposable** — keep locally for reproducibility; not required in the production path |
| **Local-only evidence** | Local build-host evidence directories outside the repository (per-stage B3 run dirs) | **Local only** — retained with SHA256 manifests, not committed |
| **Candidate image** | `vllm-spark:…-vendorfix-exp-ff4477f4878c` (both nodes) | **Retain untagged/experimental**; do not promote, retag, or push |

## Cross-references

- Attribution precedents: H1C (B12X not primary), H1O (A4 DeepGEMM runtime partial),
  H1W (matched-nsys sparse-MLA metadata group), B2C9 (B12X MoE perf attribution).
- Production identity/routing/rollback:
  [`deepseek-v4-production.md`](deepseek-v4-production.md),
  [`deepseek-v4-production.md`](deepseek-v4-production.md).
