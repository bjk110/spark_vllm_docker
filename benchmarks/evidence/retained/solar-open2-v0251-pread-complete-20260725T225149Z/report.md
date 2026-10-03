# Solar-Open2 v0.25.1 pread completion diagnostic — final report
Arc: solar-open2-v0251-pread-complete-20260725T225149Z

## FINAL STATE: SOLAR_OPEN2_V0251_PREAD_RUNTIME_SAFE_PHASE2_REGRESSION_REMAINS

The pread runtime COMPLETED the diagnostic: engine-ready reached, API health
passed, C3 correctness intact (no garble), 5-min idle stable, second request
OK, and the 4.0 GiB hard floor was never crossed on either node
(min MemAvailable: spark01 4,581,512 kB = 4.37 GiB; spark02 4,918,356 kB =
4.69 GiB). Loader-phase swap remains ELIMINATED (pread Phase-1 result
reproduced: swap <= 36-348 kB through the entire load window on both nodes;
load 141/175 s, 71.6 GiB weights). HOWEVER a NEW sustained swap growth was
observed on spark01 during engine-init warmup (0 -> 6.2 GB by engine-ready,
peak 6.64 GB, persistent thereafter), and decode is catastrophically slow
(~0.3 t/s vs 12.8 t/s on 0.22.1) — the Phase-2 anon/UVM regression is
unmitigated and now partially spills to swap instead of tripping the old
5.0 floor. v0.25.1 migration remains BLOCKED.

## Two attempts (both from controlled clean reboots of both nodes)
- attempt-1 (23:01Z): engine-ready 23:11:31Z; first C3 request killed at
  23:14:12Z NOT by memory exhaustion but by the spark02 guard tripping its
  legacy 5.0 GiB MemAvailable FLOOR by 4 MB (ma 5,238,736 kB, swap 348 kB,
  gentle slope ~30 MB/s) during rank1 first-request b12x JIT -> docker stop
  worker -> RayWorkerProc rank1 died -> EngineDead. Evidence attempt1/.
- Guard change (documented, attempt1/GUARD_CHANGE_RATIONALE.md): spark02
  worker guard aligned to the spark01 policy = fastguard3 ma_soft 5.0 /
  ma_hard 4.0 GiB + swap_hard 4.0 GB (spec: spark02 must not be stricter
  than spark01 without documented reason; hard floor kept at 4.0). The
  3.2 GB swap parity mark is derived post-hoc from hfmon3 (never approached:
  s02 peak swap 348 kB).
- attempt-2 (23:27Z): full clean-reboot cycle repeated, engine-ready
  23:39:35Z, all functional gates completed, clean shutdown 00:02Z.

## Attempt-2 timeline (UTC 2026-07-25/26)
| stage | time | ma (s01) | swap (s01) |
|---|---|---|---|
| prestart (post-reboot) | 23:27:16 | 117.6 GiB | 0 |
| container start | 23:27:17 | 117.6 GiB | 0 |
| load done TP0/TP1 (attempt-1 offsets: +141/+175 s) | ~23:29:40/23:30:30 | ~23 GiB floor of phase | <=36 kB |
| swap onset >100 MB | 23:34:23 | 7.35 GiB | 0.46 GB |
| SOFT 5.0 crossed (snapshot captured) | 23:34:26 | 4.98 GiB | 0.46 GB |
| swap burst 2.6->5.5 GB in 6 s | 23:36:08-14 | 7.2->8.1 GiB | 5.5 GB |
| engine-ready (health 200) | 23:39:35 | 7.20 GiB | 6.18 GB |
| C3 request (16 tok, incl JIT) | 23:39:35->23:43:40 | end 5.47 GiB | 6.63 GB peak |
| s02 SOFT 5.0 crossed (first-request JIT) | 23:40:58 | s02 5.00 GiB | s02 348 kB |
| Korean request (120 tok, 6m05s) | 23:47:33->23:53:38 | ~5.7 GiB | stable |
| 5-min idle (10 samples, health 200 all) | 23:53:57->23:58:57 | 5.69-5.73 GiB stable | 6.62 GB stable, no growth |
| request 2 (40 tok, 2m28s) | 23:59:10->00:01:38 | 5.67 GiB | 6.62 GB |
| shutdown + teardown complete | 00:02->00:07 | final 11.4 GiB | 0.19 GB |

Stage sub-decomposition within engine init (from attempt-1 head-full.log,
identical config/trajectory; attempt-2 in-container startup log lost at
compose down — known evidence gap, covered by hfmon3 1-s host telemetry +
in-container 2-s smaps sampler which BOTH cover attempt-2 fully):
load done 23:04:40/23:05:14 -> CT post-load releases 23:04:39+ (upstream
release_device_memory_under_pressure ACTIVE) -> b12x init + FlashInfer
autotune (autotuner start 23:10:14 TP1) + dummy-run window 23:05-23:10 (this
is where ALL swap growth occurs) -> KV reservation 23:08:56/23:09:57 (fixed
4 GiB/rank, KV 66,764 tokens) -> init engine total 336.53 s -> ready.

## Gates
1. Engine-ready: PASS (both attempts; first time for v0.25.1 on this stack).
2. /health 200, /v1/models solar-open2-250b maxlen 4096: PASS.
3. C3 deterministic request (temp=0, max_tokens=16, logprobs top-10,
   "대한민국의 수도는 어디인가요?"): TOKEN_MATCH 14/16 vs C3 marlin/v0.22.1
   reference. First 14 tokens identical with healthy logprobs (tok0 -0.0318
   vs ref -0.0226; broken-class was -0.322); divergence at pos 15
   (" Question" -> " User", logprob -0.247) continuing coherently as
   " User Question". NO garble; NOT the C1 Class-A from-first-token
   signature; consistent with MoE-backend numerics flipping a low-margin
   token. NOTE: this run used b12x (the C3 16/16 was the marlin route) —
   this is the first coherent b12x full-model output on v0.25.1, softening
   the Class-A b12x suspicion (raw-g1 fix appears to cover it).
4. Korean request: coherent bilingual reasoning output, PASS.
5. 5-min idle: ma flat, swap flat (slow decline), health 200 x10: PASS.
6. Second request: coherent: PASS.
7. Safety: NO hard trip, NO OOM, NO NVRM/Xid, NO Ray loss (attempt-2),
   hosts responsive throughout.
8. Performance (observational, not a gate): decode ~0.27-0.33 t/s — ~40x
   slower than the 0.22.1 baseline benchmark (12.8 t/s c1). Swap-resident
   UVM working set thrashing suspected.

## Phase-2 causal ranking (attempt-2 telemetry)
1. FlashInfer b12x autotune/JIT + engine-init dummy-run [HIGH]:
   23:34:23-23:37 window: s01 swap 0 -> 6.2 GB (2.6->5.5 GB in 6 s),
   ma min 4.37 GiB. In-container anon at soft snapshot only ~4.5 GiB total
   (Worker 2.72 + APIServer 1.08 + EngineCore 0.67) => bulk of pressure is
   CUDA UVM/driver-side, outside cgroup (confirms MEM-arc). Persistent
   after stage completion (swap never reclaimed while serving).
2. First-request b12x JIT on rank1/spark02 [HIGH, s02-specific]:
   ma 7.0 -> 4.69 GiB during C3 request, zero swap; killed attempt-1 via
   legacy 5.0 floor; completes safely with 4.0 hard floor.
3. CT NVFP4 post-load conversion [MEDIUM-LOW]: transient; upstream
   release_device_memory_under_pressure confirmed active; ma recovers.
4. KV reservation [LOW]: fixed 4 GiB/rank by design, deterministic.
5. KDA/GDN warmup [LOW]: dead code under kv-cache-memory-bytes (profile
   run skipped); its cost moves into first-request JIT (see 2).
6. Unresolved residual: why v0.25.1 UVM residency exceeds 0.22.1 by
   ~8-9 GiB at same settings — still open (upstream-side).

## Provenance
- Image both nodes (full ID identical):
  sha256:7e282758328c7dab8fea50b3fcac18ae0eb9f9e05b661f208c9ca4bb1c37d29c
- vLLM 0.25.1@752a3a504485; pread overlay weight_utils_pread.py sha
  9354fd0b… (exactly one backend="pread", lazy branch line 949; original
  in-image ad98e404…); C3 raw-g1 overlay solar_open2_model_c3fix.py sha
  77b081ea…; preset solar-open2-250b-nota-nvfp4-v0251-kv4g-diag-tp2.env sha
  28efc9af… (unchanged, both nodes); compose stack unchanged from prior
  pread run; model 29 shards, config.json sha a5045fec… both nodes.
- pread ACTIVE proof: same overlay+mount as prior arc (rendered-compose
  saved); loader signature reproduced exactly (load 2.5x fast, no loader
  swap, Inactive_file-dominant).
- Guards: s01 fastguard3 5.0/4.0/7.9 (sha a33dcb42…); s02 attempt-1
  fastguard2 5/3.2/4.0, attempt-2 fastguard3 5.0/4.0/4.0.
- Big rank instrumentation jsonls retained on-node in ~/docker-build/c2-obs/
  (rank-spark01-6847.jsonl, rank-spark02-1492.jsonl, ~76 MB each).

## Permanent pread patch readiness
READY for a separate opt-in implementation phase (VLLM_SPARK_ST_PREAD=1,
default off, fail-fast without safetensors>=0.8.0, lazy-loader-path only,
reversible). NOT implemented in this arc.

## Remaining v0.25.1 migration blockers
1. Phase-2 anon/UVM regression (~8-9 GiB vs 0.22.1) — now manifests as
   6.6 GB persistent swap on the head + ma floor grazing 4.4-4.7 GiB.
2. Decode performance collapse under this pressure (~0.3 t/s) — unusable.
3. b12x 14/16 vs marlin reference — acceptable-looking but needs a proper
   conformance decision if promoted.
4. Repeated/long-context behavior untested at 4096 maxlen diag config.
