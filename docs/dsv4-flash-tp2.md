# DeepSeek-V4-Flash (Official FP8) — Dual-Spark TP=2 가이드

`deepseek-ai/DeepSeek-V4-Flash` (공식 FP8 체크포인트) 를 DGX Spark 두 대
(`<head_node>`, `<worker_node>`) 위에서 TP=2 로 서빙하는 절차와 성능 측정 결과 정리.

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
| 컨테이너 | `vllm-spark-head` (head 노드) / `vllm-spark-worker` (worker 노드) |

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

### 1.3. 빌드 요구사항

- 빌드는 충분한 RAM(≥48 GiB 권장) 이 있는 호스트에서. 저메모리 호스트는 vLLM C++/CUDA 컴파일 중 OOM 위험.
- 빌드 시간 ~22 분 (vLLM 휠 빌드 ~1344s + runner stage ~50s, ccache miss 기준). ccache hit 시 ~3 분.
- 명령은 [`README`](../README.md#소스에서-빌드) 의 일반 패턴과 동일 — `docker buildx build -f Dockerfile.dsv4-d568 ...`.

### 1.4. 양 노드 배포

같은 이미지를 양 spark 노드에 배포해야 함. 권장 경로: GHCR 매개로 레이어 dedup (v022-d568 공통 레이어 재사용, 신규 ~1.5 GB 만 전송):

1. 빌드 호스트에서 `docker tag` → `docker push ghcr.io/<account>/vllm-spark:dsv4-d568`
2. 두 spark 노드에서 `docker pull ghcr.io/<account>/vllm-spark:dsv4-d568`

(또는 `docker save | ssh ... docker load` 로도 가능 — GHCR 미사용 환경.)

## 2. 모델 배포

### 2.1. 컨테이너 내부 경로

| 위치 | 경로 |
|---|---|
| 호스트 (양 spark) | `${MODEL_PATH}` 환경변수가 가리키는 디렉토리 |
| 컨테이너 내부 | `/models/DeepSeek-V4-Flash` (`${MODEL_CONTAINER_PATH}`) |

호스트 경로는 [`models/dsv4-flash-fp8-tp2.env`](../models/dsv4-flash-fp8-tp2.env) 의 `MODEL_PATH` 로 지정. compose 가 그 디렉토리를 컨테이너의 `/models/DeepSeek-V4-Flash` 에 read-only 바인드.

### 2.2. 디스크 요구사항

- 모델 149 GB × 2 노드 = ~300 GB
- 이미지 ~31 GB × 2 노드 = ~62 GB (v022-d568 와 레이어 공유로 실제 증분은 ~1.5 GB)
- 빌드 임시 + ccache: ~15 GB (빌드 노드에만)
- 권장 free disk: 노드당 200 GB+

### 2.3. 양 노드 모델 동기화

DSV4-Flash 체크포인트는 각 노드에 (TP=2 라도) 전체 가중치가 필요. 사용자 환경에 맞는 방법으로 동기화 (rsync over 관리 LAN, HF Hub 직접 다운로드, NFS 등). 전송량 ~149 GB / 노드.

## 3. 부팅 절차

### 3.1. Page cache 비우기 (GB10 unified memory 함정 회피)

모델 149 GB 동기화 후 OS page cache 에 잔존 → GPU 메모리 풀 가용량 산정에 영향. 부팅 전 반드시 양 노드에서 drop_caches:

```bash
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

확인: `free -h` 의 `available` 이 ~117 GiB 가 되어야 정상.

### 3.2. Ray 모드 (운영 권장)

기동 명령은 [`README`](../README.md#dual-spark--tp2-ray--roce) 의 일반 패턴과 동일:

1. spark head 노드에서 `docker compose ... --profile head up -d`
2. head 의 Ray runtime 이 뜨면 spark worker 노드에서 `docker compose ... --profile worker up -d`

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

200K 시퀀스 9.6 개 동시 보관 가능. `max_num_seqs=4` × 200,000 ≈ 800K tokens — 1,919,636 token KV pool 의 **약 42%** 사용 (여유 큼).

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

### 4.4. `--attention_config.use_fp4_indexer_cache=True` 는 GB10 에서 활성화 불가

jasl/vllm 의 `vllm/v1/attention/backends/mla/indexer.py` 가 이 옵션을 시도하면 *"requires Blackwell datacenter GPUs"* 검증을 수행. GB10 (SM121, consumer Blackwell) 은 datacenter 클래스가 아니므로 이 가드에 걸려 활성화 실패. 부팅 로그가 자동으로 `Using FP8 indexer cache for Lightning Indexer` 로 fallback 되는 것이 정상 동작. 따라서 `models/dsv4-flash-fp8-tp2.env` 에 이 플래그를 **추가하지 않음**. 환경변수 `VLLM_TRITON_MLA_SPARSE=1` 가 GB10 sparse MLA 활성화 경로의 핵심.

### 4.5. Page cache 점유로 인한 GPU 메모리 부족

증상: 부팅 중 `Free memory on device cuda:0 ... is less than desired GPU memory utilization`. GB10 unified memory 에서 OS page cache 가 가용 GPU 메모리에서 차감됨.

수정: 위 §3.1 참고.

### 4.6. KV pool 사용률 계산

`MAX_MODEL_LEN=200000` × `MAX_NUM_SEQS=4` = 800,000 tokens 최악치, KV pool ≈ 1,919,636 tokens, **800,000 / 1,919,636 ≈ 41.7% (≈42%)** 사용. 안전 마진 큼. (이전 21% 표기는 오류였음.)

## 5. 백엔드 선택 (Ray vs mp)

`DISTRIBUTED_BACKEND` env 변수 한 줄로 결정. 같은 이미지에서 재기동만 하면 됨.

| 항목 | `ray` (기본) | `mp` |
|---|---|---|
| 프로세스 모델 | Ray actor handle | torch.distributed SPMD |
| 부트스트랩 | head Ray 시작 → worker `ray start --address` join → vLLM 기동 | head/worker 동시에 `vllm serve --distributed-executor-backend mp --nnodes 2 --node-rank N --master-addr ... --master-port ...` (worker 는 `--headless`) |
| 콜드 스타트 | 느림 (~30s Ray init 추가) | 빠름 |
| 스텝 오버헤드 | Ray Compiled DAG (CUDAGraph + zero-copy IPC) | NCCL all-reduce 직접 |
| 알려진 이슈 | cross-node Compiled DAG bug [#36237](https://github.com/vllm-project/vllm/issues/36237) — 일부 모델 cudagraph 깨짐 | vLLM 0.20+ SPMD 1-class 지원, 명령 형태 안정화 중 (재검증 권장) |
| DSV4 우리 환경 측정 | 본 문서 §6 Test #2/#7 (Ray, no-MTP) | §6 Test #5/#6/#8 (mp) |

본 환경(GB10 SM12.1a + jasl@edc82b614f51) 의 측정 범위(no-MTP, c=1-4 peak t/s) 안에서 **Ray 와 mp 의 decode peak 는 유사** (Test #2 68.00 vs Test #8 69.33, Test #7 66.67 vs Test #8 69.33). 더 강한 결론을 내리려면 latency 분포·prefill·다른 MTP 설정 등 추가 metric 비교가 필요합니다. 현재 운영 권장은 **Ray** (기존 entrypoint 로직 그대로, 회귀 risk 최소). **mp + MTP off** 도 유효한 후보이며 entrypoint 의 mp 명령 형태가 안정화된 뒤 재측정 권장.

## 6. 벤치마크 결과

### 6.1. 측정 환경

| 항목 | 값 |
|---|---|
| 도구 | [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4 |
| 클라이언트 | 외부 호스트, 관리 LAN 으로 접속 |
| 서버 | spark head 컨테이너 (`HOST_PORT=8000`, 호스트 네트워크) |
| 토크나이저 | `deepseek-ai/DeepSeek-V4-Flash` (필요시 명시) |
| 측정 항목 | `pp 512/1024/2048` × `tg 32/128` × `concurrency 1/2/3/4` × 3 runs |
| 결과 위치 | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-*.md` |

`llama-benchy` 의 로컬 토크나이저는 `gpt2` 로 fallback (DSV4 tokenizer 가 `PreTrainedConfig.max_position_embeddings` 미지원). 이 때문에 클라이언트 측에서 표시되는 t/s 절대값과 다른 모델 간 직접 비교는 정밀하지 않습니다. 본 문서의 비교는 같은 클라이언트 토크나이저로 측정된 동일 모델 내 설정 간 상대 비교로 제한됩니다. 서버측 `peak t/s` 컬럼은 vLLM 이 직접 보고하므로 토크나이저 영향 없음.

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

## 7. MTP 가 본 측정에서 throughput 을 떨어뜨림

DSV4-Flash 는 MTP heads (Multi-Token Prediction) 를 공식 체크포인트에 포함. 포럼 #43 보고는 ~44 t/s c=1 (decode warm) 달성, MTP 효과 추정. 본 환경 측정값:

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

**관찰 — 본 GB10 TP=2 cross-node 측정에서 MTP 는 전반 throughput 을 떨어뜨림.** Acceptance rate 자체는 정상 범위지만 step time 이 늘어나서 net loss. 가능한 원인 (확정하려면 nsight / pytorch profiler 등으로 step breakdown 측정 필요):

1. draft forward 오버헤드 (sparse MLA + Lightning Indexer + FP8 GEMM)
2. cross-node TP 에서 draft step 의 NCCL 동기 비용 증폭
3. cudagraph 동작 또는 acceptance gain 을 압도하는 fixed overhead

**대응 옵션 (미시도)**: jasl 가 별도로 준비 중인 "preview branch" (포럼 #43 jasl 본인 언급, 더 강한 GB10 최적화 포함 예정) 시점에 재시험. 위 원인 가설은 profiling 없이는 확정할 수 없으므로 이후 작업으로 남겨둠.

## 8. Prefill 최고 기록 (설정 #9)

**Config**: `edc82b6` + Ray + `MAX_NUM_SEQS=4` + MTP n=2 + `MAX_NUM_BATCHED_TOKENS=8192`

| pp2048 c=4 | 1,099.94 t/s |
|---|---|

부팅 로그의 명시 경고 (`Consider increasing max_num_batched_tokens`) 에 따라 4192 → 8192 로 올리면 prefill 처리량 +29% 추가 향상. 단 KV pool 이 1.82M → 1.04M 으로 줄어들어 동시 시퀀스 한계가 5.2x@200K 로 감소 (여전히 c=4 운영에 충분).

**용도별 선택:**

| 운영 시나리오 | 권장 |
|---|---|
| 일반 대화/응답 (decode 중심, 1-4 동시) | **설정 #7** (peak 66 t/s decode @ c=4) |
| 대화 + 다중 동시 사용자 (5-8 동시) | **설정 #10** (peak 61 t/s decode @ c=8, +54% vs #9) |
| 긴 컨텍스트 prefill 빈도 높음 | **설정 #9** (peak 1100 t/s prefill, decode 40 t/s) |
| 다중 동시 사용자 + 짧은 답변 | **설정 #10** 또는 #7 |

## 9. Decode 최고 기록 (설정 #10) — max_num_seqs=8 (2026-05-22)

**Config**: 설정 #9 동일 + `MAX_NUM_SEQS=4 → 8` (admission queue 확장)

| pp2048 c=8 tg32 peak | **61.67 ± 3.30 t/s** (Run A c=4 의 40 대비 +54%) |
|---|---|
| pp2048 c=8 tg128 peak | 58.67 ± 3.77 t/s (+47%) |
| pp2048 c=7 tg128 prefill | 1085.20 ± 77 t/s (설정 #9 peak 1099 와 동등) |

### 9.1. 측정 결과 (server-reported peak, pp2048)

| c | 설정 #9 tg32 peak | 설정 #10 tg32 peak | Δ | 설정 #9 tg128 peak | 설정 #10 tg128 peak | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 38.33 ± 0 | 40.00 ± 0 | +4.4% | 40.00 ± 0 | 40.00 ± 0 | 0% |
| 5 | — | 45.00 ± 0 | new | — | 45.00 ± 0 | new |
| 6 | — | 49.33 ± 3.68 | new | — | 48.67 ± 0.47 | new |
| 7 | — | 56.00 ± 0 | new | — | 53.33 ± 2.49 | new |
| **8** | — | **61.67 ± 3.30** | **+61% vs c=4** | — | **58.67 ± 3.77** | **+47%** |

### 9.2. KV pool 영향

- pool: 1.04M tokens (설정 #9 와 동일, bt=8192 변경 없음)
- max concurrency @ 200K: **5.48x** (c=6+ 는 admission queue 의 부분 직렬화 시작)
- 실효 효과: c=4 이상 동시 요청이 있을 때 decode token 처리량이 거의 선형 스케일링

### 9.3. Trade-off

| 측면 | 설정 #9 (c=4) | 설정 #10 (c=8) |
|---|---|---|
| Decode peak | 40 t/s | **62 t/s** (+54%) |
| Prefill peak @ c=4 | **1099 ± 4 t/s (안정)** | 949 ± 190 t/s (σ ×47, 회귀) |
| Prefill peak @ c=7 | — | 1085 ± 77 t/s |
| TTFT (pp2048 c=8 tg128) | — | 11066 ms ± 4210 (높음) |
| 어떤 워크로드 | 단일·소규모 prefill 중심 | **다중 동시 decode 중심** |

설정 #10 의 prefill 은 c=4 단일 측정에서 회귀 (variance 폭증) 하지만 admission 확장으로 인한 스케줄러 동작 변화 부수효과. **c=7 영역의 prefill (1085 t/s) 은 설정 #9 peak 와 거의 동일**. 

### 9.4. 결과 파일

`benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq8-mtp2-bt8192-c1to8.md`

## 10. Tier-1 env tuning evaluation (negative findings, 2026-05-22)

NVIDIA 개발자 포럼 [post #53 Serapis](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309/53) 의 2× DGX Spark TP=2 recipe 에서 권장된 env 변수와 자체 후보 한 항목을 우리 환경(jasl/vllm @ `edc82b614f51`, 설정 #9 baseline) 에 적용 후 A/B 측정. **모두 운영 best 에 추가하지 않음.**

### 10.1. 시험 대상

| 변수 | 출처 / 권장값 | 효과 가설 |
|---|---|---|
| `OMP_NUM_THREADS=8` | forum #53 / 8 | PyTorch CPU thread pool 캡으로 GPU dispatch 와의 contention 감소 |
| `VLLM_USE_FLASHINFER_SAMPLER=1` | forum #53 / 1 | sampling 단계를 FlashInfer 로 라우팅 (PyTorch native 대체) |
| `--disable-custom-all-reduce` | forum #53 / enable | vLLM custom all-reduce kernel 비활성, NCCL 기본 경로 사용 |
| `MAX_NUM_BATCHED_TOKENS=12288` | 자체 (vLLM 부팅 hint) | bt 8192 → 12288 로 더 큰 prefill 배치 |

### 10.2. 측정 (server-reported peak/prefill, pp2048, Test #9 baseline)

| Run | Config | Decode peak (c4 tg128) | Prefill (c4 tg128) | Δ vs A |
|---|---|---:|---:|---:|
| A | baseline (Test #9, 변경 없음) | 40.00 ± 0 | **1099.94 ± 4** | — |
| B | + OMP=8 + SAMPLER=1 | 40.00 | 987.69 ± 123 | **−10.2%** prefill |
| C | + OMP=8 only (SAMPLER=0) | 40.00 | 999.36 ± 102 | **−9.1%** prefill |
| D | + `--disable-custom-all-reduce` | 41.33 ± 1.89 | 868.37 ± 157 | **−21.1%** prefill, σ ×40 |
| E | + `bt=12288` (KV pool 0.73M, max c=3.67x) | 39.67 ± 0.47 | 1077.87 ± 30 | -2.0% (σ 범위, **flat**) |

### 10.3. 결론

- **`OMP_NUM_THREADS=8` 이 prefill 회귀의 주범** (run C). GB10 의 20-core CPU 를 8 thread 로 축소하면 chunked-prefill admission path 가 CPU 병목. forum #53 의 RTX PRO 6000 (다른 코어 구조) 권장값이 GB10 에 부적합.
- **`VLLM_USE_FLASHINFER_SAMPLER=1` 은 거의 중립** (B vs C 차이 ~1%, σ 범위). 본 벤치는 temperature=0 기본 → sampler 경로가 가벼움. greedy 워크로드에서 의미 없음. 또한 jasl/vllm 에서 **이 옵션은 기본 ON** 이므로 명시 `=1` 은 no-op.
- **`--disable-custom-all-reduce` 는 prefill peak (c=4) 손상** (run D). σ 4 → 157 로 variance 40배 증가. vLLM custom all-reduce kernel 이 우리 GB10 + 200 Gbps RoCE + TP=2 cross-node 구성에 잘 최적화되어 있음. forum #53 의 권장은 다른 토폴로지 기준.
- **`bt=12288` 은 평형 (saturation 도달)** (run E). 4192 → 8192 단계의 +29% 와 달리 8192 → 12288 은 -2% (σ 범위). vLLM 의 "Consider increasing" hint 는 bt=4192 시점 경고였고 8192 가 이미 sweet spot.
- **Decode peak 는 모든 run 에서 ≈40 t/s** (GPU-bound 영역). 본 섹션의 변경 후보들은 모두 c=4 admission ceiling 자체를 깰 수 없음 — admission 자체를 확장한 [설정 #10](#9-decode-최고-기록-설정-10--max_num_seqs8-2026-05-22) 만이 decode 60+ t/s 영역을 열었음.

→ 네 변경 모두 `models/dsv4-flash-fp8-tp2.env` 와 `docker-compose.yml` 에 적용하지 않음. 본 negative finding 은 동일 시도 재발 방지용 기록.

### 10.4. 결과 파일

| 설정 | 파일 |
|---|---|
| Run A (baseline) | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-c1to4.md` (= 설정 #9) |
| Run B (OMP+SAMPLER) | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-OMP8-flashsampler-c1to4.md` |
| Run C (OMP only) | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-OMP8-only-c1to4.md` |
| Run D (no-custom-allreduce) | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-no-custom-allreduce-c1to4.md` |
| Run E (bt=12288) | `benchmarks/llama-benchy/results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt12288-c1to4.md` |

## 11. vLLM commit bump 시도 (negative, 2026-05-23)

`edc82b614f51` → `dad6ff885838` (branch HEAD 시점, "Limit long prefill chunks behind active decode" 안정성 fix 등 포함) 시도. 결과: **GB10 UMA 사전 점유와 vLLM strict free-memory check 비호환으로 채택 보류.**

### 11.1. 빌드/배포
- spark02 빌드 → 이미지 `vllm-spark:dsv4-d568-dad6ff8` (31.6 GB), spark01 로 docker save | ssh docker load 전송 (~27분)
- Dockerfile 변경: `git fetch origin ${VLLM_BRANCH}` → `git fetch origin ${VLLM_COMMIT}` (jasl 브랜치 force-rebase 대응)
- 양 노드 `.env` 및 `models/dsv4-flash-fp8-tp2.env` 의 `VLLM_IMAGE` 갱신
- Dry-run import 검증 (`docker run --rm` 안에서 `import vllm` + `DeepseekV4ForCausalLM` + `EngineCoreProc` + MTP + FlashInfer) 모두 통과 → 빌드 결함 아님

### 11.2. 실패 모드
양 노드 깨끗한 리부팅 후 컨테이너 시작 → vLLM `request_memory()` (`vllm/v1/worker/utils.py:413`) 가 init 시점에 거부:
```
ValueError: Free memory on device cuda:0 (36.75/121.63 GiB) on startup is less than
desired GPU memory utilization (0.85, 103.38 GiB).
```
2회 시도 모두 동일 — 양 노드 host RAM 의 `used` 가 idle 상태에서 ~78 GiB (`free -h`), nvidia driver/UVM 이 컨테이너 시작 직후 host RAM 의 ~85 GiB 를 pre-reserve. vLLM의 init 시점 free-memory 사전체크는 가중치 로드/UMA commit 이전 시점이므로, 이 reserve를 "사용 중"으로 인식.

### 11.3. Root cause
- GB10 UMA의 nvidia driver는 컨테이너 spawn 직후 host RAM 의 큰 영역(~85 GiB)을 미리 잡고, 가중치 로드 단계에서 GPU 메모리로 commit. vLLM이 init 시점에 검사하는 `free` 값은 이 reserve 이후의 수치다.
- 동일 환경에서 `edc82b614f51` 는 정상 작동 (Run F 검증) — `dad6ff885838` 의 변경 어딘가에서 init 시점 free-memory 사전체크 path가 추가/강화돼 GB10 UMA pre-reserve 와 비호환이 된 것으로 추정. upstream vLLM `request_memory()` 의 SM12x/UMA 특수 케이스 처리 부재.

### 11.4. 부수 피해
- compose `restart: unless-stopped` 가 EngineCore fail 후 자동 재시작 → 첫 시도 1회차에 누적 RestartCount=35 → host RAM thrashing 발생 → 양 노드 sshd banner exchange timeout (TCP accept 는 됨, 커널은 살아있으나 userspace 부하). 물리적 reset 필요.
- 추후 비슷한 commit bump 테스트 시 임시 `restart: no` 권장 + `docker logs -f | grep -m 1 'EngineCore failed|Uvicorn running'` 단일 ssh stream 으로 첫 fail/ready 신호 즉시 감지하여 stop 권장.

### 11.5. 결정
`edc82b614f51` 유지. 이미지 `vllm-spark:dsv4-d568-dad6ff8` 는 spark01/02 로컬에 남겨두되 `.env` 미참조. dad6ff8 단독 디버깅(또는 `request_memory()` 우회 패치)은 별도 세션.

### 11.6. 후속 검증 (2026-05-23 G6 + G7)
**§11.1-11.5 의 "dad6ff8 strict free-memory check" 가설은 G7 검증으로 무효화됨.** 진짜 root cause 는 commit 차이가 아니다.

원인 가설 점검 결과:
- `request_memory()` (`vllm/v1/worker/utils.py:413`), `MemorySnapshot.measure()` (`vllm/utils/mem_utils.py`), `is_integrated_gpu()` (`vllm/platforms/cuda.py`), `gpu_worker.init_device()` 부근 — **edc82b614f51 ↔ dad6ff885838 라인-identical** (G6).
- **G7 결정적 finding (2026-05-23 15:00)**: 운영 복귀 시도 중 **edc82b6 도 동일 ValueError 거부** (`Free memory 36.85/121.63 GiB`). 즉 dad6ff8 vs edc82b6 차이가 아니라 **reboot 후 경과 시간** 이 진짜 root cause.
  - reboot 직후 (uptime 2분): host RAM free 117 GiB, psutil available 117 GiB → **통과**
  - 시간 경과 (uptime 14분): host RAM used 78 GiB / available 42 GiB → **거부** (요구 103 GiB 미달)
- 우리 이전 "edc82b6 정상" 운영 사례는 모두 reboot 직후 즉시 시작이었음. jasl/vllm version 과 무관한 **GB10 UMA platform-wide issue**.
- 메커니즘: nvidia driver / kernel buffers (buff/cache 포함) / nx server / dockerd 등 user-space + kernel-side process 가 reboot 후 시간 지남에 따라 host RAM 누적 점유 → `psutil.virtual_memory().available` 감소 → vLLM UMA 분기의 free memory check 거부. `buff/cache` 가 reclaimable 임에도 psutil `available` 계산에서 부분 제외되는 것으로 추정.
- 운영 규칙: **reboot 후 5분 이내에 vLLM 컨테이너 시작 필수**. 그 이상 지나면 어느 commit 이든 동일 거부 가능.
- 부수 finding (G6): docker compose v2 `restart: "on-failure:N"` 의 max-retries 부분이 **swarm-only 로 처리되어 일반 deploy 에서 무시됨**. 결과적으로 무한 재시작 = `unless-stopped` 와 동일 → sshd-storm 재발. `restart: "no"` 로 강화 (commit `4b848a1`). 재시작이 필요하면 `docker compose up -d` 수동 호출.
- 진짜 해결 후보:
  1. **`vllm/utils/mem_utils.py` UMA 분기 패치**: `psutil.virtual_memory().available` 대신 `available + (buff/cache reclaimable 부분)` 사용. reclaimable kernel memory 를 free 로 인식하도록.
  2. `--gpu-memory-utilization` 을 측정된 available 기반 동적 산정 (단순 0.85 곱 아닌 `available - safety_margin`).
  3. 운영 정책으로 reboot 직후 즉시 시작 (workaround, 현재 채택).
- §11.1-11.5 의 dad6ff8 거부 사례는 위 동일 원인. dad6ff8 코드 자체는 문제 없음 — 재시도 시 reboot 직후 즉시 시작하면 통과 가능성 큼.

**G9 검증 (mp backend 도 거부, 2026-05-23):**
- ray 운영 stop → DISTRIBUTED_BACKEND=mp 변경 → reboot 없이 즉시 시도 (host RAM 누적 상태) → 동일 `ValueError: Free memory 14.11/121.63 GiB` 거부.
- Path: ray `RayWorkerProc.initialize_worker()` vs mp `multiproc_executor.py:870 (Worker pid=...)` — 다른 path 로 같은 `request_memory()` 호출.
- 측정값: ray 36.85 GiB > mp 14.11 GiB → **mp 가 더 비관적**. Ray object store ~70 GiB 점유가 host RAM 측정에 영향 주는 게 아님. **backend 무관 host RAM 누적이 진짜 원인 최종 확정.**
- 컨테이너 stop 후 host RAM 회수: used 110 → 104 GiB (단 ~7 GiB만 회수) → GB10 UMA quirk [[gb10-uma-memory-needs-reboot-to-reclaim]] 재확인. reboot 만이 유일한 회수 방법.

### 11.7. G8 — `VLLM_SKIP_INIT_MEMORY_CHECK` 패치 검증 (✅ 통과, 2026-05-24)

§11.6 의 운영 5분 제약을 영구 해소하기 위한 escape-hatch 패치 검증.

**패치 디자인** (`patches/patch_skip_init_memory_check.py`, commit `82d013a`):
- `vllm/v1/worker/utils.py:413` `request_memory()` 함수에 env-var anchor 추가
- `VLLM_SKIP_INIT_MEMORY_CHECK=1` 설정 시 pre-check 건너뛰고 경고 로그 + 정상 진행
- 진짜 OOM 은 weight load 단계에서 natural failure site 로 surface
- Dockerfile.dsv4-d568 STAGE 2 에 idempotent 적용 (env-var 미설정 시 동작 동일)
- vLLM `envs.py` 미등록 var 라서 시작 시 `Unknown vLLM environment variable detected` 경고 출력 (무해, 향후 vLLM 패치로 등록 가능)

**빌드 및 배포**:
- 이미지: `vllm-spark:dsv4-d568-skipmem` (image ID `de4c3ba4661e`, VLLM_COMMIT `edc82b614f51` 유지)
- spark02 빌드 ~50분 (NGC 26.04 base + 패치 line 추가 → STAGE 2 영향만)
- spark02 → spark01 transfer 6분 15초 (대부분 layer 가 기존 dsv4-d568 와 공유)
- `docker-compose.yml` head + worker `environment` 에 `VLLM_SKIP_INIT_MEMORY_CHECK=${VLLM_SKIP_INIT_MEMORY_CHECK:-0}` passthrough 추가

**검증 조건** (host RAM 누적 상태):
- spark01 head: uptime 21h 16m, used 82 GiB, **available 39 GiB** ≪ 요구 103 GiB → 패치 없으면 거부될 정확한 조건
- spark02 worker: uptime 1h 08m, available 115 GiB (빌드 후 회수) → 자연 통과 가능

**핵심 검증 결과**:
1. **패치 활성화 로그 양 노드 출현** @ 13:16:52:
   ```
   WARNING [utils.py:419] VLLM_SKIP_INIT_MEMORY_CHECK=1 — skipping startup
     free-memory check (free_memory=35785916416, requested=111006700749 on cuda:0).
   ```
   spark01: free 33.3 GiB ≪ requested 103.4 GiB → **패치 효과 결정적 입증**
   spark02: free 115.1 GiB > requested 103.4 GiB → patch 무관 통과
2. **가중치 46/46 shards 로드 완료** — TP=2 분산이 실제 가중치를 적절히 분배해 weight load 단계 OOM 없음
3. **API server `Application startup complete`** + `/v1/models` 정상 응답 (deepseek-v4-flash, max_len=200000)
4. **Token gen sanity**: `"What is 2+2?"` → content `'4'`, finish_reason=stop, completion_tokens=26

**운영 변경 (영구)**:
- `.env` 양 노드: `VLLM_IMAGE=ghcr.io/bjk110/vllm-spark:dsv4-d568-skipmem` + `VLLM_SKIP_INIT_MEMORY_CHECK=1`
- 백업 자동 생성: `.env.bak-pre-g8-skipmem-*`
- **운영 5분 제약 해제** — reboot 후 시간 경과 무관 안전한 컨테이너 시작 가능

**Rollback 절차** (필요 시):
1. `.env` 의 `VLLM_IMAGE` 를 `ghcr.io/bjk110/vllm-spark:dsv4-d568` 로 복원 (양 노드 dsv4-d568 이미지 보존됨)
2. `VLLM_SKIP_INIT_MEMORY_CHECK=0` 또는 제거
3. 또는 `.env.bak-pre-g8-skipmem-*` 직접 복원
4. `docker compose --profile head/worker up -d` 재시작 → 5분 제약 운영 복귀

**Follow-up 후보** (관련 메모: [[upstream-updates-2026-05-24]]):
- vLLM `envs.py` 에 `VLLM_SKIP_INIT_MEMORY_CHECK` registration 추가 → "Unknown env var" 경고 제거
- PR #35929 (`posix_fadvise(POSIX_FADV_DONTNEED)`) 자체 패치 이식 → 운영 중 host RAM page cache 회수
- jasl `c4fc1d2` (SM120 sparse MLA → DSV4) commit bump 재도전 — 이제 host RAM 누적 변수 없이 공정 A/B 가능

## 12. 참고 링크

- NVIDIA 개발자 포럼 [post #43](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309/43)
- eugr/spark-vllm-docker [PR #219 (DeepSeek V4 Flash recipe)](https://github.com/eugr/spark-vllm-docker/pull/219)
- eugr/spark-vllm-docker [PR #244 (configurable vllm/flashinfer URLs)](https://github.com/eugr/spark-vllm-docker/pull/244)
- vllm-project/vllm [PR #41834 (jasl SM12x DSV4 support)](https://github.com/vllm-project/vllm/pull/41834)
- jasl/vllm [branch codex/ds4-sm120-min-enable](https://github.com/jasl/vllm/tree/codex/ds4-sm120-min-enable)
- vllm-project/vllm [issue #36237 (Ray Compiled DAG cross-node)](https://github.com/vllm-project/vllm/issues/36237)
