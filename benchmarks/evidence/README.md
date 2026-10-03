# Canonical recipe evidence

This directory is the evidence authority for the **14 presets in catalog sections 1–2**:
8 production/rollback/production-qualified presets and 6 validated non-production presets.
The machine-readable contract is [`manifest.json`](manifest.json).

## Retention rule

Keep in Git only material that directly establishes at least one of the following for a final recipe:

- immutable image/source provenance;
- promotion, rollback, correctness, stability, or bounded performance gates;
- a known limit that changes how the recipe may be used;
- the authoritative runbook or activation contract.

Raw requests/responses, container logs, profiler captures, repeated dry runs, transient telemetry,
intermediate A/B attempts, and superseded experimental presets are not recipe authority. They are
kept in the external archive and represented here by [`archive-manifest.sha256`](archive-manifest.sha256).

## Evidence map

| Family | Final recipe coverage | Canonical evidence |
|---|---|---|
| DeepSeek-V4 | v0.27 active production, optional attended MS4, v0.25 primary rollback, MTP1 legacy rollback | [`docs/deepseek-v4-production.md`](../../docs/deepseek-v4-production.md), [`docs/deepseek-v4-v027-b43s-promotion-candidate.md`](../../docs/deepseek-v4-v027-b43s-promotion-candidate.md), retained MTP1 cold/soak/long-context summaries |
| Solar-Open2 | v0.25.1 r4 active production and v0.22.1 rollback | [`docs/solar-open2-production.md`](../../docs/solar-open2-production.md), tracked r3/r4 source provenance, retained promotion/readiness/build/cache/performance summaries |
| Qwen3.8 | production-qualified c1/c2 manual baseline and PR55122 MTP1 c1 opt-in | [`docs/qwen3.8-flash-next-tp2.md`](../../docs/qwen3.8-flash-next-tp2.md), [`docs/qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md`](../../docs/qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md) |
| Step-3.7 | NVFP4 v0.23 latency, FP8 v0.23 tokenizer-overlay, FP8 v0.22 bounded paths | tracked benchmark and model guides under [`docs/`](../../docs/README.md) |
| Gemma 4 / Qwen3.6 | bounded validated non-production presets | [`docs/model-serving-validation-history.md`](../../docs/model-serving-validation-history.md), [`docs/flashinfer-aot-prebake.md`](../../docs/flashinfer-aot-prebake.md) |

The exact per-preset evidence paths and explicit limits are in `manifest.json`. Retained summary files
are byte-pinned by SHA-256 there; they are intentionally small and do not include raw prompt/response
corpora or runtime logs.

## External archive

- Archive ID: `2026-10-03-final-recipe-curation`
- Maintainer-local location: `/home/bjk110/benchmark-archive/vllm-spark/2026-10-03-final-recipe-curation`
- Files: 5,135
- Bytes: 388,800,633
- Maintainer-local directory permissions at curation: `0700`
- Integrity ledger: [`archive-manifest.sha256`](archive-manifest.sha256)

The local archive is **not required for clone-time verification**. Its ledger is tracked so a recovered
copy can be authenticated file-by-file. To verify a local copy:

```bash
python scripts/diag/verify_recipe_evidence.py \
  --archive-root /home/bjk110/benchmark-archive/vllm-spark/2026-10-03-final-recipe-curation
```

Without `--archive-root`, CI validates the 14-recipe scope, all tracked evidence links, retained-file
hashes, and archive-ledger format/count without requiring machine-local benchmark data.

## Status boundaries

- A retained report does not promote a recipe. `presets/README.md` and each preset's metadata remain
  the status authority.
- Failed or mixed results are retained only when they explain a current limit or rollback decision.
- July Solar summaries predate the authoritative August production fast-track; the production runbook
  remains authoritative for the six-gate promotion.
- MTP1 staged long-context passes do not authorize repeated large-context production operation.
