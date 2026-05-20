# DeepSeek-V4-Flash (Official FP8) — Dual-Spark TP=2 가이드

`deepseek-ai/DeepSeek-V4-Flash` (공식 FP8 체크포인트) 를 DGX Spark 두 대
(`spark01`, `spark02`) 위에서 TP=2 로 서빙하는 절차와 성능 측정 결과 정리.

NVIDIA 개발자 포럼 [DeepSeek V4 Flash official FP8 running across 2× DGX Spark, TP=2, MTP, 200K ctx — recipe + numbers](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309) post #43 의 레시피를 기반으로, jasl/vllm 포크 + 우리 v022-d568 베이스 이미지로 재구성. 9가지 설정 조합을 벤치마크한 결과 우리 환경(GB10 / SM12.1a / 200 Gbps RoCE) 의 운영 최적값을 도출.

## 0. 구성 요약

| 항목 | 값 |
|---|---|
| 모델 | `deepseek-ai/DeepSeek-V4-Flash` (공식 FP8, E4M3 128×128 block) |
| 디스크 크기 | ~149 GB safetensors (46 shards) |
| 모델 아키텍처 | DeepseekV4ForCausalLM (MoE + MLA, MTP heads 포함) |
| TP | 2 (노드당 1 GPU, cross-node) |
| Pipeline parallel | 1 |
| Expert parallel | enabled (256 experts split 128/rank) |
| KV cache dtype | fp8 (DeepSeek's fp8_ds_mla format) |
| Attention | sparse MLA + Lightning Indexer (FP8 cache) |
| `MAX_MODEL_LEN` | 200,000 |
| GPU mem util | 0.85 |
| 운영 best `MAX_NUM_SEQS` | **4** (forum 기본 2, 우리 환경 4 가 peak) |
| Compose 파일 | `docker-compose.yml` (공용) |
| Env 프리셋 | `models/dsv4-flash-fp8-tp2.env` |
| 이미지 | `ghcr.io/bjk110/vllm-spark:dsv4-d568` |
| 컨테이너 | `vllm-spark-head` (spark01) / `vllm-spark-worker` (spark02) |

## 1. 이미지 빌드

### 1.1. 이미지 체인

```
ghcr.io/bjk110/vllm-spark:dsv4-d568
 │ digest: sha256:b18da2a01146a8fd527bd77758ac8d14d4497a697777ada2796eab32ef32d574
 │
 └─FROM ghcr.io/bjk110/vllm-spark:v022-d568
     (NGC 26.04 + PyTorch 2.12.0a0 + FlashInfer 0.6.11.post3
      + Triton 3.7.0 + NCCL 2.30.4 + Transformers 5.8.1)
```

`Dockerfile.dsv4-d568` 가 v022-d568 베이스 위에:
1. **vllm-builder stage**: `jasl/vllm:codex/ds4-sm120-min-enable` @ `edc82b614f51f4f9ce16c7010e879571e5c46136` (2026-05-19 HEAD, +249 commits over forum-pinned `dda4668b`) 휠 빌드
2. **runner stage**: v022-d568 의 기존 vLLM (v0.21.0+PR#35568) 제거 → jasl 휠 설치
3. `patches/apply_dsv4_packed_mapping.py` 적용 (defensive, jasl branch 가 자체 정의 시 skip)
4. `patches/patch_split_module_compat.py` 재적용 (jasl 휠은 base 의 패치를 상속 못 함)
5. `patches/moe_config_e256/e512.json` 재배치 (GB10 튜닝, vLLM 재설치로 사라진 것 복원)
6. `instanttensor` 설치 (eugr PR #219 recipe 요구사항)

### 1.2. jasl/vllm @ edc82b614f51 의 핵심 변경 (vs dda4668b)

**MTP 관련 fix:**
- `662b07732d3e` Fix DeepSeek V4 MTP small-batch graph hangs
- `beb72fe2f050` Fix DeepSeek V4 MTP sparse SWA reordering
- `296b5bff3aff` [Bugfix] Fix SWA cache block mask breaking prefix caching with Eagle/MTP
- `0e9bc3702e32` Remove ineffective DeepSeek V4 mHC warmup

**SM12x (GB10) 성능 튜닝:**
- FP8 MQA logits tile widening + row tile tune (`e7d91051971d`, `0c2ac46b3d84`, `edc82b614f51`)
- Sparse MLA accumulate autotune num_warps/num_stages (`05830a5d496c`)
- fp8_einsum + fused_indexer_q num_warps tune (`7f817da476d6`)
- C128A prefill KV gather overlap on aux stream (`25b90652afa4`)
- per-token early-loop-exit on sparse MLA (`48f8874d3897`)
- harden `sparse_attn_indexer` seq_lens slice with `.contiguous()` (`b035280bba14`)

**예상 효과 (벤치마크 결과로 확인):** Prefill +50-77%, Decode peak +6-7%.

### 1.3. 빌드 명령

> **모든 빌드는 spark01 또는 spark02 에서.** homeserver (32 GiB RAM) 에서 빌드 시
> OOM cascade + freeze 가능 (2026-05-07 사례).

```bash
ssh spark01 'docker buildx build --progress=plain \
  -f <repo>/Dockerfile.dsv4-d568 \
  -t vllm-spark:dsv4-d568 --load \
  <repo>'
```

빌드 시간 ~22 분 (vLLM C++/CUDA 휠 빌드 1344s + runner stage 50s).
ccache 가 비어 있으면 휠 빌드 풀 컴파일, ccache hit 시 ~3 분.

### 1.4. 양 노드 배포

GHCR 매개로 레이어 dedup (v022-d568 공통 레이어 재사용, 신규 ~1.5 GB 만 전송):

```bash
# spark01 에서
docker tag vllm-spark:dsv4-d568 ghcr.io/bjk110/vllm-spark:dsv4-d568
docker push ghcr.io/bjk110/vllm-spark:dsv4-d568

# spark02 에서
docker pull ghcr.io/bjk110/vllm-spark:dsv4-d568
```

## 2. 모델 배포

### 2.1. 경로

| 위치 | 경로 |
|---|---|
| homeserver | `<source_dir>/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash` |
| spark01 / spark02 host | `<spark_model_dir>/deepseek-ai/DeepSeek-V4-Flash` |
| 컨테이너 내부 | `/models/DeepSeek-V4-Flash` |

### 2.2. 디스크 요구사항

- 모델 149 GB × 2 노드 = ~300 GB
- 이미지 ~31 GB × 2 노드 = ~62 GB (v022-d568 와 레이어 공유로 실제는 ~32 GB)
- 빌드 임시 + ccache: ~15 GB
- 권장 free disk: 노드당 200 GB+

### 2.3. 양 노드 동시 동기화 (homeserver → 각 spark)

```bash
# spark01 으로
rsync -av --info=progress2 --no-i-r \
  <source_dir>/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash/ \
  spark01:<spark_model_dir>/deepseek-ai/DeepSeek-V4-Flash/ &

# spark02 으로 (병렬)
rsync -av --info=progress2 --no-i-r \
  <source_dir>/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash/ \
  spark02:<spark_model_dir>/deepseek-ai/DeepSeek-V4-Flash/ &
wait
```

전송 시간 ~20 분 (128 MB/s).

## 3. 부팅 절차

### 3.1. Page cache 비우기 (GB10 unified memory 함정 회피)

모델 149 GB rsync 후 OS page cache 에 잔존 → GPU 메모리 풀 가용량 산정에 영향.
부팅 전 반드시 drop_caches:

```bash
ssh spark01 'sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"'
ssh spark02 'sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"'
```

확인: `free -h` 의 `available` 이 ~117 GiB 가 되어야 정상.

### 3.2. Ray 모드 (운영 권장)

```bash
# spark01 (head 먼저)
ssh spark01 'docker compose \
  --project-directory <repo> \
  -f <repo>/docker-compose.yml \
  --env-file <repo>/models/dsv4-flash-fp8-tp2.env \
  --profile head up -d'

# head 의 Ray runtime 이 뜰 때까지 대기 후 worker
ssh spark02 'docker compose \
  --project-directory <repo> \
  -f <repo>/docker-compose.yml \
  --env-file <repo>/models/dsv4-flash-fp8-tp2.env \
  --profile worker up -d'
```

`entrypoint.sh` 가 `ray start --head` → 노드 join 대기 → `vllm serve --distributed-executor-backend ray` 진행.

### 3.3. MP 모드 (Ray Compiled DAG 우회용)

`models/dsv4-flash-fp8-tp2.env` 에서:
```
DISTRIBUTED_BACKEND=mp
MASTER_PORT=29501
```

기동 순서는 동일. `entrypoint.sh` 가 양 노드에서 `vllm serve` 를 SPMD 로 실행:
- head: `--nnodes 2 --node-rank 0 --master-addr <HEAD_ROCE_IP> --master-port <MASTER_PORT>`
- worker: 위에 `--headless` 추가

eugr/spark-vllm-docker `launch-cluster.sh` 의 `--no-ray` 모드와 동일 패턴.

### 3.4. 부팅 진행 단계 (정상 시)

총 ~6-9 분.

| Phase | 시간 | 로그 키워드 |
|---|---|---|
| Container start + Ray/MP rendezvous | 30-60s | `CLUSTER_MODE=dual-rdma`, `Ray runtime started` / `Inferred data_parallel_rank` |
| Engine init + arch resolve | 10s | `Resolved architecture: DeepseekV4ForCausalLM`, `TritonFp8BlockScaledMMKernel`, `fp8_ds_mla KV cache` |
| Safetensors 로딩 (46 shards) | 1:45 | `Loading safetensors checkpoint shards: 100% Completed | 46/46` |
| torch.compile + AOT cache | 35-65s | `torch.compile took ... s`, `saved AOT compiled function` |
| TileLang `mhc_*` 커널 컴파일 | ~80s | `mhc_pre_big_fuse_tilelang`, `mhc_post_tilelang`, `hc_head_fuse_tilelang`, `mhc_fused_tilelang` |
| Initial profiling/warmup | 60-85s | `Initial profiling/warmup run took` |
| KV cache + CUDA graph capture | ~120s | `GPU KV cache size: 1,919,636 tokens`, `Graph capturing finished` |
| API server ready | - | `Application startup complete`, `/health` HTTP 200 |

### 3.5. 부팅 성공 시 KV cache + concurrency

```
GPU KV cache size: 1,919,636 tokens
Maximum concurrency for 200,000 tokens per request: 9.60x
```

200K 시퀀스 9.6 개 동시 보관 가능. `max_num_seqs=4` × 200K = 21% 풀 사용 (여유 많음).

## 4. 알려진 함정과 픽스

### 4.1. PyTorch 2.12.0a0 alpha 의 `split_module(tuple_return=...)` 결여

증상: 부팅 중 컴파일 단계에서
```
TypeError: split_module() got an unexpected keyword argument 'tuple_return'
```

원인: vLLM 이 `is_torch_equal_or_newer("2.12.0.dev")` 정적 체크로 `tuple_return=True` 를 패스하지만 NGC 26.04 의 PyTorch 2.12.0a0 alpha 는 이 kwarg 미지원.

수정: `patches/patch_split_module_compat.py` — 정적 버전 체크를 런타임 signature probe 로 교체. Dockerfile 의 runner stage 에서 새 vLLM 설치 직후 적용.

### 4.2. Forum 권장 `dda4668b` pin 의 MTP graph hang

`dda4668b` (forum 권장 commit) 부팅 시 MTP small-batch 에서 hang. `edc82b614f51` (HEAD) 에 픽스 (`662b07732d3e`, `beb72fe2f050`, `296b5bff3aff`) 포함.

### 4.3. `max_num_scheduled_tokens` 경고 (MTP 사용 시)

부팅 로그:
```
max_num_scheduled_tokens is set to 4192 based on the speculative decoding settings.
This may lead to suboptimal performance. Consider increasing max_num_batched_tokens.
```

`MAX_NUM_BATCHED_TOKENS=8192` 로 올리면 경고 해소 + prefill 더 향상 (벤치 §6 참고). 단 KV pool 이 1.82M → 1.04M 으로 줄어들어 trade-off.

### 4.4. Page cache 점유로 인한 GPU 메모리 부족

증상: 부팅 중 `Free memory on device cuda:0 ... is less than desired GPU memory utilization`. GB10 unified memory 에서 OS page cache 가 가용 GPU 메모리에서 차감됨.

수정: 위 §3.1 참고.

## 5. 백엔드 선택 (Ray vs mp)

`DISTRIBUTED_BACKEND` env 변수 한 줄로 결정. 같은 이미지에서 재기동만 하면 됨.

| 항목 | `ray` (기본) | `mp` |
|---|---|---|
| 프로세스 모델 | Ray actor handle | torch.distributed SPMD |
| 부트스트랩 | head Ray 시작 → worker `ray start --address` join → vLLM 기동 | head/worker 동시에 `vllm serve --nnodes 2 --node-rank N --master-addr ... --master-port ...` (worker 는 `--headless`) |
| 콜드 스타트 | 느림 (~30s Ray init 추가) | 빠름 |
| 스텝 오버헤드 | Ray Compiled DAG (CUDAGraph + zero-copy IPC) | NCCL all-reduce 직접 |
| 알려진 이슈 | cross-node Compiled DAG bug [#36237](https://github.com/vllm-project/vllm/issues/36237) — 일부 모델 cudagraph 깨짐 | (없음, vLLM 0.20+ 1-class 지원) |
| DSV4 우리 환경 측정 | **Test C (§6) 참고** | **Test A (§6) — 차이 거의 없음** |

본 환경(GB10 SM12.1a + jasl@edc82b614f51) 측정상 **Ray vs mp 성능 차이는 무시할 만함** (peak t/s ±2%). Ray 권장 (기존 entrypoint 로직 그대로).

## 6. 벤치마크 결과

### 6.1. 측정 환경

| 항목 | 값 |
|---|---|
| 도구 | [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4 |
| 클라이언트 | homeserver (관리 LAN) |
| 서버 | spark01 head 컨테이너 (`HOST_PORT=8000`, 호스트 네트워크) |
| 토크나이저 | `deepseek-ai/DeepSeek-V4-Flash` (필요시 명시) |
| 측정 항목 | `pp 512/1024/2048` × `tg 32/128` × `concurrency 1/2/3/4` × 3 runs |
| 결과 위치 | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-*.md` |

`llama-benchy` 로컬 토크나이저는 `gpt2` 로 fallback (DSV4 tokenizer 가 `PreTrainedConfig.max_position_embeddings` 미지원). t/s 절대값은 ~10-18% 저평가 가능; 서버측 `peak t/s` 컬럼은 정확.

### 6.2. 9-way 비교 (peak t/s)

| # | vLLM commit | Backend | max_seq | MTP | batched_tok | tg128 c=4 peak | pp2048 c=4 |
|---|---|---|---:|---|---:|---:|---:|
| 1 | `dda4668b` | Ray | 2 | off | 4192 | 40.67 | 527 |
| 2 | `dda4668b` | Ray | 4 | off | 4192 | **68.00** | 527 |
| 3 | `dda4668b` | Ray | 4 | n=2 | 4192 | 33.00 | - |
| 4 | `dda4668b` | Ray | 4 | n=1 | 4192 | 48.00 | - |
| 5 | `dda4668b` | mp | 4 | n=2 | 4192 | 30.33 | - |
| 6 | `edc82b6` | mp | 4 | n=2 | 4192 | 40.00 | 931 |
| 7 | **`edc82b6`** | **Ray** | **4** | **off** | **4192** | **66.67** | **850** |
| 8 | `edc82b6` | mp | 4 | off | 4192 | 69.33 | similar |
| 9 | `edc82b6` | Ray | 4 | n=2 | **8192** | 40.00 | **🚀 1100** |

> #2 와 #7/#8 이 decode peak 동률 권장. #7 (Ray) 가 entrypoint 변경 최소화.
> #9 가 prefill 신기록 (1099.94 t/s) — MTP 가 prefill 만 보면 도움됨.

### 6.3. 결과 파일 매핑

| 설정 # | 파일명 |
|---|---|
| 1 | `results_dsv4-flash-fp8-tp2.md` (초기 c=1-2) + `*-c1to4.md` |
| 2 | `results_dsv4-flash-fp8-tp2-maxseq4-c1to4.md` |
| 3 | `results_dsv4-flash-fp8-tp2-maxseq4-mtp2-c1to4.md` |
| 4 | `results_dsv4-flash-fp8-tp2-maxseq4-mtp1-c1to4.md` |
| 5 | `results_dsv4-flash-fp8-tp2-mp-maxseq4-mtp2-c1to4.md` |
| 6 | `results_dsv4-flash-fp8-tp2-edc82b6-mp-maxseq4-mtp2-c1to4.md` |
| **7** | `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-nomtp-c1to4.md` |
| 8 | `results_dsv4-flash-fp8-tp2-edc82b6-mp-maxseq4-nomtp-c1to4.md` |
| 9 | `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-c1to4.md` |

### 6.4. 운영 best 상세 (설정 #7)

**Config**: `edc82b6` + Ray + `MAX_NUM_SEQS=4` + MTP off + `MAX_NUM_BATCHED_TOKENS=4192`

**Decode peak t/s (server-reported):**

| Concurrency | tg32 | tg128 |
|---|---:|---:|
| 1 | 25 | 25 |
| 2 | 42 | 42.67 |
| 3 | 48.33 | 49 |
| **4** | **69.33** | **66.67** |

**Prefill t/s (server-reported total):**

| Concurrency | pp512 | pp1024 | pp2048 |
|---|---:|---:|---:|
| 1 | ~308 | 404 | **665** |
| 2 | 369 | 660 | 913 |
| 3 | 358 | 532 | 763 |
| 4 | 405 | 656 | 800 |

**TTFT (c=1, ms)**:

| | pp512 | pp1024 | pp2048 |
|---|---:|---:|---:|
| TTFT | 2,024 | 2,242 | 2,785 |

## 7. MTP 가 우리 환경에서 손해인 이유

DSV4-Flash 는 MTP heads (Multi-Token Prediction) 를 공식 체크포인트에 포함. 포럼 #43 보고는 ~44 t/s c=1 (decode warm) 달성, MTP 효과 추정. 우리 환경:

| 설정 | Decode c=1 peak | tg128 c=4 peak |
|---|---:|---:|
| MTP off | **25** | **66.67** |
| MTP n=1 | 19 | 48 |
| MTP n=2 | 16 | 33-40 |

**측정된 MTP 동작 (정상):**
- 1st draft acceptance rate: 78-83% (포럼 보고와 일치)
- 2nd draft acceptance rate: 36-50% (depth 2 효과 미미)
- Mean acceptance length: 2.14-2.50 tokens/step
- Avg acceptance: 55-69%

**손해 원인 추정:**
1. GB10 SM12.1a 에서 draft forward (sparse MLA + Lightning Indexer + FP8 GEMM 풀가동) 비용이 H100 / RTX Pro 6000 대비 상대적으로 큼
2. cross-node TP=2 에서 draft 도 양 노드 동기 실행 → 단일 노드 + TP=1 대비 sync 비용 증폭
3. MTP step cost 가 acceptance gain 을 압도

**대응 옵션 (미시도)**: jasl 가 별도로 준비 중인 "preview branch" (포럼 #43 jasl 본인 언급, 더 강한 GB10 최적화 포함 예정) 시점에 재시험.

## 8. Prefill 최고 기록 (설정 #9)

**Config**: `edc82b6` + Ray + `MAX_NUM_SEQS=4` + MTP n=2 + `MAX_NUM_BATCHED_TOKENS=8192`

| pp2048 c=4 | 1,099.94 t/s |
|---|---|

부팅 로그의 명시 경고 (`Consider increasing max_num_batched_tokens`) 에 따라 4192 → 8192 로 올리면 prefill 처리량 +29% 추가 향상. 단 KV pool 이 1.82M → 1.04M 으로 줄어들어 동시 시퀀스 한계가 5.2x@200K 로 감소 (여전히 c=4 운영에 충분).

**용도별 선택:**

| 운영 시나리오 | 권장 |
|---|---|
| 일반 대화/응답 (decode 중심) | **설정 #7** (peak 66 t/s decode) |
| 긴 컨텍스트 prefill 빈도 높음 | **설정 #9** (peak 1100 t/s prefill, decode 40 t/s) |
| 다중 동시 사용자 + 짧은 답변 | 설정 #7 (max_seq 더 올릴 여지) |

## 9. 참고 링크

- NVIDIA 개발자 포럼 [post #43](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309/43)
- eugr/spark-vllm-docker [PR #219 (DeepSeek V4 Flash recipe)](https://github.com/eugr/spark-vllm-docker/pull/219)
- eugr/spark-vllm-docker [PR #244 (configurable vllm/flashinfer URLs)](https://github.com/eugr/spark-vllm-docker/pull/244)
- vllm-project/vllm [PR #41834 (jasl SM12x DSV4 support)](https://github.com/vllm-project/vllm/pull/41834)
- jasl/vllm [branch codex/ds4-sm120-min-enable](https://github.com/jasl/vllm/tree/codex/ds4-sm120-min-enable)
- vllm-project/vllm [issue #36237 (Ray Compiled DAG cross-node)](https://github.com/vllm-project/vllm/issues/36237)
