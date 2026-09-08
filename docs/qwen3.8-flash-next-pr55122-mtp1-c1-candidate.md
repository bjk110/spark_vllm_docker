# Qwen3.8 Flash Next PR55122 MTP1 interactive-c1 candidate

**Status: MAIN_MERGE_APPROVED_OPT_IN — opt-in interactive c1 only; NOT the production default; no auto-start.**

This Gate4 recipe pins local image ID `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610` and enables the deterministic `persistent_topk` kernel from vLLM PR #55122 (`1c4caa3dbe1c3c616f63810fd6fa60877105132d`). The kernel is pinned by SHA256 `803037813e307a7c5fa05efee5e874695a0620a336dd798800ccd51d566377b3`; patched QSA SHA256 is `7e6aff79a37b866b130de65bd483a6526d9e8ae14cc45733c2deb080ff2f2c47`.

## Qualified envelope and exclusions

TP2, `MAX_NUM_SEQS=2`, client concurrency **c1 only**, prefix cache OFF, MTP depth 1, and CUDA graph `FULL_DECODE_ONLY` capture `[1,2]`. MTP1 c2/c8, prefix-cache ON, and `MAX_NUM_SEQS=8` performance are excluded and are not production-qualified. Commit, push, and main merge are explicitly authorized. Image tag/publication, build, launch, service activation, auto-start, and production-default change remain out of scope.

## Production suitability decision

- `MAIN_REPOSITORY_INTEGRATION_APPROVED`: the bounded opt-in c1 profile, immutable provenance artifacts, documentation, and fail-closed verifier may be committed, pushed, and merged into `main`.
- `PRODUCTION_RUNTIME_PROMOTION_BLOCKED`: do not make this the production default or auto-start it. PR #55122 remains open/unmerged without maintainer approval and with a pre-run-check failure; the base image full vLLM commit remains unresolved; epoch 3 retained an unresolved `NV_ERR_NO_MEMORY` residual.
- The existing Qwen3.8 c1/c2 preset remains the production-qualified baseline. Repository integration does not authorize image publication/rebuild or service activation.

## Qualification ledger

- Matched c1 performance/correctness: `/home/bjk110/docker-build/qwen38-pr55122-perf-mtp1-20260905-r2/`.
- Gate1 30-minute c1 qualification PASS: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-soak30-20260905-r2/`; the initial r1 role-listener harness false negative remains preserved and is not a PASS claim.
- Gate2 4-hour c1 soak PASS: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-soak4h-20260905-r1/` (`14400.000459282892s`, 800 request records, 0 fatal signal, MTP acceptance 1804/1828).
- Gate3 three clean-boot cold-start epochs PASS: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-coldstart-e1-20260905-r1/`, `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-coldstart-e2-20260906-r1/`, and `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-coldstart-e3-20260906-r1/`.
- Gate4 zero-start static closure PASS decision: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate4-20260906-r1/gate4-decision.json`, SHA256 `19520f6abaff5fb4a0bcad29e3afd78b131a2f732e397cc67eff5d22f8f275d5`; independent review: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate4-20260906-r1/gate4-independent-review.json`, SHA256 `b126037c2be0300c3fb62b2d6b3b5dca5fbfc6a1f141bdb3dce3274b8007f928`. The verifier performs no `up`, build, pull, tag, push, or service mutation.
- Gate5 independent decision: `/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate5-20260906-r1/gate5-independent-review.json`, SHA256 `05feba33f0f68f0dd3802c442be56ed04986623640889ba949a286cfe2355680`. It qualified only explicit user decision for local working-tree product/config promotion; it did not authorize commit, push, release, image tag/publication, build, launch, or service activation. Commit, push, and main merge are explicitly authorized. Image tag/publication, build, launch, service activation, auto-start, and production-default change remain out of scope.

## Provenance closure and unresolved upstream state

The authoritative local image inspection on **each** target node separately observed embedded _C_det.so SHA256 `803037813e307a7c5fa05efee5e874695a0620a336dd798800ccd51d566377b3` and separately observed embedded QSA SHA256 `7e6aff79a37b866b130de65bd483a6526d9e8ae14cc45733c2deb080ff2f2c47`. These are observations of files embedded in the already-qualified image, distinct from hashing the repository copies. The pinned authoritative upstream kernel source is retained as `patches/qwen/pr55122/persistent_topk.cuh` at SHA256 `db6f9c2b2c580ddb97b7026e01100ae23b2e2bfa5dbaa6396ba61189cc33fe6c`; its bytes match tested PR head `1c4caa3dbe1c3c616f63810fd6fa60877105132d`.

As rechecked before repository merge, the current PR head is `995cd99fa7d47834c0d89e038c2e07324a0065ac`. Its `persistent_topk.cuh` SHA256 is `b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7`, which differs from the tested and pinned source SHA256 `db6f9c2b2c580ddb97b7026e01100ae23b2e2bfa5dbaa6396ba61189cc33fe6c`; the current PR head is not locally qualified. This profile remains pinned to tested head `1c4caa3dbe1c3c616f63810fd6fa60877105132d`. PR #55122 is **open and unmerged**, has **no maintainer approval**, and has a **pre-run-check failure**. The base image exposes only the unresolved short vLLM revision `8e685d198` (`0.1.dev20073+g8e685d198`); its full commit identity remains unresolved. None of these facts is promoted into stronger provenance.

The reconstruction-only Dockerfile mirrors the authoritative Dockerfile's exact labels, values, instruction counts, and COPY semantics, changing only COPY source paths so they resolve from this repository's build context. Its immutable image/Dockerfile label `LOCAL NON-PRODUCTION experimental deterministic persistent_topk evaluation; not promoted or pushed` is deliberately preserved as build-time provenance. That preserved label does not negate the externally recorded local product/config promotion and does not mean the image was rebuilt or retagged. The Dockerfile **does not reproduce the image ID**: rebuilding is not claimed to yield `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610`, and no rebuild is authorized.

## Residuals and decision integrity

Epoch 3 passed the scoped cold-start gate, but an `NV_ERR_NO_MEMORY` residual was observed and remains unresolved; the PASS must not erase or downgrade that residual. Commit, push, and main merge are explicitly authorized. Image tag/publication, build, launch, service activation, auto-start, and production-default change remain out of scope. This remains opt-in interactive c1 only, is NOT the production default, and has no auto-start.

### Decision-integrity correction history

The Gate1 r1 run was a role-listener harness false negative; the corrected r2 run and both records remain preserved rather than rewriting history.

Separately, after epoch 3 completed, its launcher-produced `decision-summary.json` was observed by the first independent reviewer at SHA256 `8404c8fc051d12e797d8dd61fba8682372661f6e3353317a9b7e96a1041bff40`. It was then accidentally overwritten by a manual summary that copied epoch 2 readiness `1056s` instead of epoch 3 raw readiness `1072s`. Those invalid bytes are preserved as `decision-summary.invalid-5deb6a53.json` at SHA256 `5deb6a537c3c6bf89807f5bbc4990d29676c9c1839f1b63c3549ca49a549e4cf`; the original launcher bytes could not be recovered and are not claimed restored. A corrected decision rebuilt from raw epoch 3 evidence has SHA256 `0d6ac07e1e7f5f28954e27fc22c1d436d15da20d378a90cfcb4e60a55a35bdab` and passed a fresh independent review. The separate three-epoch `gate3-overall-decision.json` is SHA256 `3253f2fa00b45e37c4cf1131a7303c6aa9ca5323d72bcde35b9152b161bd5777`.

Gate0–Gate5 qualification and the explicit local promotion apply only to MTP1/c1 and do not authorize c2/c8, prefix-cache ON, or `MAX_NUM_SEQS=8` performance. Commit, push, and main merge are explicitly authorized. Image tag/publication, build, launch, service activation, auto-start, and production-default change remain out of scope. The corrected record carries forward the upstream PR/pre-run-check limitations, unresolved base-image revision, and epoch-3 memory residual instead of treating a scoped PASS as global clearance.

## Gate4 vertical TDD evidence

Each blocker mutation was added and observed RED before its verifier/recipe slice was changed, then rerun GREEN. Commands used the targeted `python3 -m unittest ... -v` test names. Observed transitions:

- COPY flag after positional: RED `FAILED (failures=1)`; GREEN `OK`.
- Numeric YAML `&123`/`*123`: RED `FAILED (failures=1)`; GREEN `OK`.
- Extra service child `foo`, quoted `foo`, and multiline `foo`: each RED `FAILED (failures=1)`; each GREEN `OK`.
- Extra environment entry: RED `FAILED (failures=1)`; GREEN `OK`.
- Arbitrary non-approved preset value and unapproved extra-args token: each RED `FAILED (failures=1)`; each GREEN `OK`.
- Closed LABEL/value set, pinned upstream source hash mutation, and wrong forced-node provenance label: each RED `FAILED (failures=1)`; each GREEN `OK`.
- Documentation closure check: RED with 12 missing requirements; GREEN with zero missing requirements.

## Static verification

```bash
python3 scripts/diag/verify_qwen38_pr55122_mtp1_c1_candidate_recipe.py
QWEN38_PR55122_REQUIRE_LOCAL_IMAGE_ID=1 QWEN38_PR55122_IMAGE_ID_HOSTS=spark01,spark02 python3 scripts/diag/verify_qwen38_pr55122_mtp1_c1_candidate_recipe.py
```

The verifier rejects ambient candidate-preset value drift and any non-exact ambient `COMPOSE_PROJECT_NAME`. Forced remote identity mode requires the exact ordered host value `QWEN38_PR55122_IMAGE_ID_HOSTS=spark01,spark02`; only an absent host variable selects local forced mode. Run the forced identity form independently on both target nodes before any separately authorized launch. A bare local image ID is intentionally used: this image has no registry digest, and a fake `repo@sha256:<local-ID>` must never be manufactured.

## Separately authorized worker-first invocation

Use the candidate preset plus `docker-compose.yml` plus the candidate overlay, worker first and then head, always with `--no-deps`. Merely checking this recipe must use `docker compose ... config`, never `up`.

## Rollback

Stop both candidate roles with the exact same preset/base/overlay/project inputs and `docker compose --profile head --profile worker down`. Verify no candidate containers/listeners remain. The tracked production recipe remains `presets/qwen3.8-flash-next-fp8-tp2-candidate.env` with `compose/qwen3.8-flash-next/docker-compose.candidate.yml`; this opt-in recipe does not modify it. Because GB10 UMA may remain retained after teardown, reboot both nodes before restoring a memory-heavy production workload when `MemAvailable` has not recovered.
