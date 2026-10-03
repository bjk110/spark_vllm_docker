# Solar Open 2 — vLLM 0.25.1 r4 Promotion Readiness (PR arc)

- Date: 2026-07-27 (UTC)
- **1. Final state: `SOLAR_OPEN2_V0251_PROMOTION_READY_WITH_C2_CAVEAT`**
- No production promotion occurred in this task.

## 2–4. Evidence, mirror, verification

- **2.** spark01: `/home/bjk110/docker-build/solar-open2-v0251-promotion-readiness-20260727T065646Z/`
- **3.** Homeserver mirror: `/home/bjk110/docker/vllm-spark/benchmarks/results/solar-open2-v0251-promotion-readiness-20260727T065646Z/` (untracked path)
- **4.** SHA-256: all mirrored files verified against `SHA256SUMS.txt` (0 mismatches; see mirror log).

## 5. Repository

Root `/home/bjk110/docker/vllm-spark` (spark01), branch `main`, HEAD `fd61254`, tracked state
clean at preflight and completion; `git diff --check` clean. No commit, push, or staging. No
tracked file modified; no worktree needed (all proposed docs live in the evidence directory).

## 6–7. Images (identical IDs on spark01 and spark02, arm64; never rebuilt or retagged)

- Baseline: `vllm-spark:solar-open2-nvfp4-v022d568-vllm0221-upstage00907fc-ecfix-exp`
  `sha256:1873d2174691f67e16b5588fcef01680d21f1e7b42ac5587bd23d7503cae1366` (created 2026-07-24)
- r4: `vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp`
  `sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774` (created 2026-07-26);
  byte-identical (same image ID) to the image used in MI/AC/LS/SF/R4 arcs.

## 8. r4 provenance (complete; `01_provenance/provenance.md`)

NGC 26.05 (PyTorch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2) → r2 `7e282758328c` (vLLM 0.25.1
@752a3a504485 + FlashInfer 0.6.15 @8eccd0c1 + Upstage Solar overlay rebased 00907fc) → r3
`001dcd2fb66d` (+raw-g1 fix, +ST_PREAD gate) → r4 `ecb7bfe3978a` (+B12X shared-workspace gate).
All patches applied as SHA-256-verified COPYs; canonical .patch files baked at
`/opt/spark-patches/{r3,r4}`. Probed versions: vllm 0.25.1, flashinfer 0.6.15, triton 3.7.0,
transformers 5.10.2, compressed-tensors 0.15.0.1, safetensors 0.8.0. Build evidence dirs preserved.

## 9. Model

`nota-ai/Solar-Open2-250B-Nota-NVFP4` @ `de88f6226788077e2d340204fd79d37720c9eda0`;
`/home/bjk110/Documents/Models/upstage/nota-ai_Solar-Open2-250B-Nota-NVFP4`;
153,347,363,609 bytes / 42 files identical on both nodes. Not modified.

## 10–11. Rendered commands

- **10.** r4: `vllm serve /models/Solar-Open2-250B-Nota-NVFP4 --served-model-name solar-open2-250b
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.80 --max-num-batched-tokens 2048
  --trust-remote-code --host 0.0.0.0 --port 8000 --dtype auto --enable-prefix-caching
  --tensor-parallel-size 2 --distributed-executor-backend ray --enforce-eager
  --kv-cache-memory-bytes 4294967296 --moe-backend flashinfer_b12x
  --default-chat-template-kwargs {"think_render_option":"preserved"} --reasoning-parser solar_open2
  --tool-call-parser solar_open2 --enable-auto-tool-choice
  --logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor`
  + env `VLLM_SPARK_ST_PREAD=1`, `VLLM_SPARK_B12X_SHARED_WORKSPACE=1`
  (preset `solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-exp-tp2.env`, compose stack with
  pread-r3 + b12xsw-r4 + b12x-cache overlays, launcher `launch-sf.sh`).
- **11.** Rollback: identical `vllm serve` line under the baseline image via preset
  `solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` (base compose stack only);
  `launch-sf.sh head|worker <ev> baseline`.

## 12. Initial clean-state memory

Controlled reboot (spark02→spark01; required because idle MemAvailable was 33/28 GiB — the
documented GB10 UMA residual). Post-reboot: s01 123,426,652 kB / s02 123,739,944 kB (~117.7 GiB),
swap ≈0 — matching the MI clean envelope.

## 13. r4 startup

Launch → health 200 in **479 s** (validated range 457–501 s). Milestones in `03_r4_startup/`.
Gates: 2 Ray nodes, one rank/node, numeric RoCE, NVFP4 compressed-tensors path, pread markers 2,
shared-workspace entries 3/rank (6 total), fallback 0, KV `GPU KV cache size: 66,764 tokens`
(= validated r4 value), /health + /v1/models 200, zero CUDA/NCCL/Gloo/Ray/OOM/restart errors.

## 14. Correctness and API smoke tests (all PASS; `04_smoke_tests/`)

- Conformance C1–C5 (validated AC suite): 33/33 http 200. C1 8/8 deterministic (finish=length,
  16 tok); C3 tool_calls finish + valid `get_weather` JSON args + reasoning field present;
  C2/C4/C5 in the documented low-margin variance class; C5 length-finish per preserved-think
  contract. Matches the accepted conformance classes exactly.
- Multi-turn Korean: 2 turns, both 200; turn 2 explicitly references turn 1's question and
  answers with 서울-context (경복궁 task) — context retained, no template leakage, no parser error.
- Coherence: llama-benchy coherence test PASSED.

## 15–20. Four-hour soak (`05_soak/`; all gates PASS)

- **15.** Duration: exactly 14,400 s (07:29:30→11:29:30Z), 48 cycles.
- **16.** Requests: 288 OK / **0 failures** (mix per cycle: Korean, reasoning, tool-call,
  pp512-class c1, pp512-class c2 pair; hourly summaries 72/144/216/288 all fail=0).
- **17.** MemAvailable: min s01 22,197,776 kB (21.2 GiB) / s02 22,539,404 kB (21.5 GiB); final
  ≈22.8 GiB — flat for 4 h, no decline trend (samplers every 30 s + fastguard 20 ms).
- **18.** Swap: peak s01 352 kB / s02 344 kB; final same — swap-free class throughout.
- **19.** Guard trips 0, OOM 0, worker restarts 0, worker loss 0, request failures 0; API health
  200 every cycle; final head-log error grep 0.
- **20.** Shared workspace: 3 entries/rank, fallback 0 at every hourly audit and at final audit.

## 21. Post-soak clean restart test (`06_restart_test/`; PASS)

Clean stop (ports 8000/6379 freed, memory recovery recorded: ma 33.4 GiB residual per documented
GB10 behavior) → controlled reboot cycle → same image/preset relaunch → health in **458 s** →
all r4 gates re-passed (pread 2, shared 6, fallback 0, KV 66,764) → smoke 3/3 (Korean, reasoning,
tool-call) → clean stop, ports free. Validated state is reproducible, not a one-time condition.

## 22–23. Baseline rollback rehearsal (`07_rollback/`; PASS)

- Attempt 1 (recorded, preserved): startup 866 s, health 200, reasoning 200, benchy pp512c1 +
  coherence PASSED; the basic-Korean request hit the harness's 180 s client timeout against the
  baseline's documented cold first-shape specialization (~186 s, SF arc) — an artifact of the test
  client, not a baseline failure; documented per protocol.
- Attempt 2 (clean PASS): startup **867 s** (validated 866–867 s range), image IDs verified both
  nodes, KV 66,764 tok, Korean 200 (48 tok), reasoning 200, benchy pp512c1 coherence PASSED,
  clean stop, ports free.
- **23.** Baseline swap during rehearsal: ≈5.6 GiB (s01) — the documented accepted baseline
  behavior, recorded not failed. Rollback required no rebuild, no model recopy, no patch.

## 24. AC / LS / MI interpretation (formalized in `02_protocol_review/`)

AC observed −12~14% (single-instance) → LS rejected scheduler geometry → MI attributed the
magnitude primarily to fresh-engine variance. c1 = parity class (median 0.9863, CI [0.937, 1.011]);
c2 = possible real, directionally consistent ~1–3% deficit (median 0.9702, CI [0.917, 0.987],
6/6 < 1.0). AC's figure must not be quoted as the expected r4 penalty; MI supersedes AC for
promotion interpretation; AC preserved as history. Kernel isolation not justified without new
evidence.

## 25. Proposed multi-instance acceptance protocol

In `02_protocol_review/performance-evidence-review.md` (proposed only): ≥5 (prefer 6
counterbalanced) fresh-engine pairs, per-instance median-of-3, paired ratios + 100k paired
bootstrap CI, no outlier deletion, continuity gates, image-ID/model-revision equality; reusable
harness `mi_driver.sh`.

## 26–27. Performance conclusions

- **26. c1**: parity — median paired ratio 0.9863, CI includes 1.000. Meets proposed ≥0.97.
- **27. c2**: NOT full parity — median 0.9702 (≥0.95 proposed gate: PASS), CI [0.917, 0.987]
  excludes parity, 6/6 pairs below 1.0. **Caveat: a real ~1–3% c2 deficit likely persists.** No
  reproducible >5% regression established.

## 28. Operational trade-off (full table in `08_documentation/tradeoff-and-promotion-plan.md`)

r4 advantages (confirmed): ~1.8× faster startup (457–501 s vs 866–867 s), swap-free operation
(≤352 kB vs 5.5–5.6 GiB every baseline start), stable post-idle behavior (no 26–41 s re-spec),
stable c8 (70.2 t/s CV 0.6% vs 41 t/s CV 54.5%), raw-g1 KDA correctness fix included, 4 h soak
0-failure. Baseline advantages: accepted history; c2 ~1–3% median edge. Promotion requires
explicit user approval.

## 29–30. Proposed promotion plan and rollback plan (PREPARED, NOT EXECUTED)

`08_documentation/tradeoff-and-promotion-plan.md`: exact artifact (image ID above), preset
candidate path, required service changes (port-8000 takeover; OWUI/Traefik untouched), backup
steps, health checks, abort conditions, ≥30 min + 24 h observation. Rollback: stop r4 → reboot
cycle → `launch-sf.sh … baseline` with `sha256:1873d217…` + di-matched preset → health + smoke
(re-rehearsed twice this arc, both functional). No promoted preset file was created.

## 31. Created/modified files

- No tracked repository file modified. Untracked: evidence dir (this arc), harness scripts
  `~/docker-build/c2-instr/{pr_soak.sh,pr_sampler.sh}` (spark01/02), scratchpad scripts
  (`pr_start_r4.sh`, `pr_cycle.sh`) copied into `08_documentation/`.

## 32. Final service/port state

No test containers; ports 8000 and 6379 free on both nodes; guards and samplers stopped; both
hosts responsive; images, model, caches, and all evidence preserved. Final memory: s01 ma
34.5 GiB (post-rollback UMA residual, documented; swap used ≈5.6 GB from the baseline rehearsal —
baseline-inherent, clears on next reboot), s02 nominal.

## 33. Git

`main@fd61254`, tracked clean, `git diff --check` clean, nothing staged, no commit, no push.

## 34. Promotion confirmation

**No production promotion occurred.** Production preset, promoted images, v0.22.1 baseline,
OpenWebUI, Traefik, GHCR: all untouched. Promotion requires a separate, explicitly authorized
task.

---

Final state: **`SOLAR_OPEN2_V0251_PROMOTION_READY_WITH_C2_CAVEAT`**
