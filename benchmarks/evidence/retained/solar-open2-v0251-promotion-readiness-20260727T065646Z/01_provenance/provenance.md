# r4 Artifact Provenance (Promotion Readiness arc, 2026-07-27)

## Image lineage (all arm64, per-node builds, never `docker save`-transferred)

```
nvcr.io NGC 26.05 (PyTorch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2)
  └─ vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rebased-ecfix-upstream-r2-exp
     (ID 7e282758328c, built 2026-07-25)
     = vLLM v0.25.1 @ 752a3a504485 + FlashInfer 0.6.15 @ 8eccd0c1
       + Upstage Solar-Open2 overlay rebased from 00907fc (incl. r2 KDA warmup fix;
         upstream replaced the standalone ecfix)
  └─ vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-r3-exp
     (ID 001dcd2fb66d, built 2026-07-26; build evidence
      ~/docker-build/solar-open2-v0251-pread-r3-build-20260726T002411Z)
     + patch 2: C3 raw-g1 KDA contract fix   -> vllm/model_executor/models/solar_open2.py
       (4e5c9071… -> 56b48747…; patches/solar/solar-open2-rawg1-contract-v025.patch,
        sha f252383c…)
     + patch 3: VLLM_SPARK_ST_PREAD env gate -> vllm/model_executor/model_loader/weight_utils.py
       (ad98e404… -> de48cbef…; patches/solar/vllm-safetensors-pread-env-gate.patch,
        sha 39b26822…)
  └─ vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-b12xsw-r4-exp
     (ID sha256:ecb7bfe3978a5241c5c304d52ce91e061e22b750178d21a4ef7788a08e86e774,
      built 2026-07-26T14:39:54+09:00; build evidence
      ~/docker-build/solar-open2-v0251-b12xsw-r4-20260726T053457Z)
     + patch 4: VLLM_SPARK_B12X_SHARED_WORKSPACE env gate
       -> flashinfer/fused_moe/cute_dsl/b12x_moe.py (bcac8067… -> 38c253a3…;
          patches/solar/vllm-flashinfer-b12x-shared-workspace-env-gate.patch,
          sha 2b31f3a8…)
```

Patch application method: SHA-256-verified COPY of externally patched files (pre- and
post-image hashes asserted in the Dockerfile RUN; AST parse gate; stale .pyc removed).
Canonical .patch files baked in-image at `/opt/spark-patches/r3/` and `/opt/spark-patches/r4/`.

## Component versions (probed inside the r4 image, `component-versions.txt`)

vllm 0.25.1 · torch 2.12.0a0+5aff3928d8.nv26.05 · CUDA 13.2 · flashinfer 0.6.15 ·
triton 3.7.0 · transformers 5.10.2 · compressed_tensors 0.15.0.1 · safetensors 0.8.0

## Feature-to-artifact mapping

| Feature | Artifact |
|---|---|
| raw-g1 KDA contract correction | r3 layer: `vllm/model_executor/models/solar_open2.py` (patch 2) |
| pread loader path | r3 layer: `weight_utils.py` gate, activated by `VLLM_SPARK_ST_PREAD=1` (preset) |
| b12x shared workspace | r4 layer: `flashinfer/fused_moe/cute_dsl/b12x_moe.py` (patch 4), activated by `VLLM_SPARK_B12X_SHARED_WORKSPACE=1` (preset) |
| shared-workspace entry allocation | `_sw_shared_get()` in patched b12x_moe.py; logs "allocated shared …" (expect 3/rank) |
| shared-workspace fallback detection | fallback log lines in patched b12x_moe.py (expect 0) |
| fixed 4 GiB KV per rank | preset/rendered flag `--kv-cache-memory-bytes 4294967296` |

## Runtime configuration

- Preset: `presets/solar-open2-250b-nota-nvfp4-v0251-r4-b12xsw-kv4g-exp-tp2.env`
  (sha256 recorded in provenance-raw.txt; sets `VLLM_SPARK_ST_PREAD=1`,
  `VLLM_SPARK_B12X_SHARED_WORKSPACE=1`)
- Compose stack: `docker-compose.yml` + `docker-compose.solar-open2-hc-exp.yml`
  + `docker-compose.pread-r3.yml` + `docker-compose.b12xsw-r4.yml` + `docker-compose.b12x-cache.yml`
  (launcher `launch-sf.sh`, guards-only)
- Rendered vLLM command (verbatim, from MI evidence and re-verified this arc):
  `vllm serve /models/Solar-Open2-250B-Nota-NVFP4 --served-model-name solar-open2-250b
   --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.80
   --max-num-batched-tokens 2048 --trust-remote-code --host 0.0.0.0 --port 8000 --dtype auto
   --enable-prefix-caching --tensor-parallel-size 2 --distributed-executor-backend ray
   --enforce-eager --kv-cache-memory-bytes 4294967296 --moe-backend flashinfer_b12x
   --default-chat-template-kwargs {"think_render_option":"preserved"}
   --reasoning-parser solar_open2 --tool-call-parser solar_open2 --enable-auto-tool-choice
   --logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor`
- Memory guards: fastguard3.py pure-/proc 20 ms (head: ma soft 5.0 / hard 4.0 GiB, swap 7.9 GiB;
  worker: swap 4.0 GiB) + dmesg OOM/NVRM/Xid/UVM watcher
- Ray/RoCE: head 10.10.10.1, worker 10.10.10.2, Ray port 6379, one rank per node,
  NCCL_NVLS_ENABLE=0

## Model

- HF `nota-ai/Solar-Open2-250B-Nota-NVFP4`, pinned revision `de88f6226788077e2d340204fd79d37720c9eda0`
  (downloaded+verified on homeserver 2026-07-24, distributed homeserver→spark01→spark02)
- Spark path `/home/bjk110/Documents/Models/upstage/nota-ai_Solar-Open2-250B-Nota-NVFP4`
- Integrity this arc: byte count 153,347,363,609 and file count 42 identical on both nodes
  (preflight); same values as at deployment.

## Identity assertions

- r4 image ID identical on spark01/spark02 AND byte-identical (same sha256 ID) to the image used
  throughout MI, AC, LS, SF, R4 arcs: `sha256:ecb7bfe3978a…86e774`.
- Baseline image ID identical on both nodes: `sha256:1873d2174691…ae1366`.
- Tag equality was NOT relied upon; all comparisons by full image ID.
