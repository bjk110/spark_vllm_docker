# vLLM Spark — DGX Spark (GB10) 통합 서빙

한국어 | **[English](README.md)**

NVIDIA DGX Spark 듀얼 노드 클러스터(GB10 x 2)를 위한 통합 vLLM 서빙 구성입니다.
여러 모델(Qwen3.5, Gemma 4)을 다양한 양자화 방식으로 `.env` 프리셋 하나로 전환할 수 있습니다 — 리포 하나, Dockerfile 하나, compose 파일 하나.

## 하드웨어

| 노드 | 역할 | GPU | 메모리 | 인터커넥트 |
|---|---|---|---|---|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |

## 소프트웨어 스택

### v020-ngc2603 (최신, NGC 26.03)

주요 업데이트: vLLM main에 포함된 **upstream TurboQuant KV 캐시 압축** (PR #38479), FlashInfer v0.6.8 SM121/GB10 최적화 (NVFP4 group GEMM, tile filtering, FP4 CUTLASS). upstream 반영된 패치 3개 제거 (cuMemcpyBatch, RoPE fix, PR #38423). `--kv-cache-dtype turboquant_k8v4`로 KV 캐시 용량 2-4배 확장 가능.

| 구성요소 | 버전 |
|---|---|
| 베이스 이미지 | NGC PyTorch 26.03 |
| vLLM | 0.20.0.dev (main 978a4462, 소스 빌드, TurboQuant 포함) |
| FlashInfer | v0.6.8 (SM121 tile filtering, NVFP4 group GEMM, 소스 빌드) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (네이티브) |
| NCCL | 2.29.7 |
| Python | 3.12 |
| Transformers | 5.5.4 |
| `_C_stable_libtorch` | 포함 (NVFP4/FP8/CUTLASS 전체 op) |

### v019-ngc2603 (이전, NGC 26.03)

vLLM 0.19.1 Gemma 4 지원, 비동기 스케줄링. Transformers 5.5.0. TTFT v018 대비 ~2배 향상. TurboQuant와 FlashInfer v0.6.8이 포함된 v020-ngc2603으로 대체됨.

| 구성요소 | 버전 |
|---|---|
| 베이스 이미지 | NGC PyTorch 26.03 |
| vLLM | 0.19.1 (main a7d79fa, 소스 빌드) |
| FlashInfer | v0.6.7.post3 (CUTLASS 4.4.2, SM121 소스 빌드) |
| PyTorch | 2.11.0a0 |
| CUDA | 13.2 (네이티브) |
| Transformers | 5.5.0 |

## 지원 모델

| 프리셋 | 모델 | 양자화 | TP | 이미지 |
|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE (26B/4B active) | 1 | v020-ngc2603 |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (멀티모달) | 2 | v020-ngc2603 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (사전 양자화) | 1 | v020-ngc2603 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | 1 | v020-ngc2603 |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (텍스트 전용, 탈검열) | 2 | v020-ngc2603 |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (텍스트 전용, 탈검열) | 1 | v020-ngc2603 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | 2 | v020-ngc2603 |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 (런타임) | 1 | v020-ngc2603 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 (런타임) | 2 | v020-ngc2603 |
| `qwen3.6-35b-fp16.env` ⚗️ | Qwen/Qwen3.6-35B-A3B | **FP16 원본** (KV fp8) | 1 | v020-ngc2603 |

## 빠른 시작

### 0. Docker 이미지 준비

#### 방법 A: GHCR에서 빌드된 이미지 Pull

```bash
# NGC 26.03 + vLLM 0.20.0.dev (TurboQuant + Gemma 4 + Qwen3.5)
docker pull ghcr.io/bjk110/vllm-spark:v020-ngc2603
```

#### 방법 B: 소스에서 빌드

```bash
# NGC 26.03 소스 빌드 (vLLM main, TurboQuant 포함)
docker buildx build -f Dockerfile.gemma4 \
  -t vllm-spark:v020-ngc2603 --load .
```

빌드 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `BUILD_JOBS` | 16 | 병렬 빌드 작업 수 |
| `FLASHINFER_REF` | v0.6.8 | FlashInfer git ref |
| `VLLM_COMMIT` | 978a4462 | vLLM 소스 커밋 |
| `TORCH_CUDA_ARCH` | 12.1a | 타겟 CUDA 아키텍처 (Blackwell) |

### 1. 모델 프리셋 선택

```bash
cp models/qwen3.5-397b-int4.env .env
```

`.env`의 `MODEL_PATH`를 실제 모델 가중치 경로로 수정합니다:

```bash
# [model_path]를 실제 경로로 변경
sed -i 's|\[model_path\]|/home/user/models|' .env
```

### 2. 서비스 시작

#### TP2 멀티노드 (예: 397B INT4)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

Head 노드는 Worker가 Ray 클러스터에 참여할 때까지 자동으로 대기한 후 vLLM을 시작합니다.

#### TP1 싱글노드 (예: NVFP4 122B)

```bash
cp models/qwen3.5-122b-nvfp4.env .env
docker compose --profile head up -d
```

`TP_SIZE=1`이면 Ray 없이 `vllm serve`를 직접 실행합니다.

### 3. 동작 확인

```bash
curl http://spark01:8000/health
```

## 아키텍처

```
spark01 (head)                    spark02 (worker)
┌─────────────────────┐          ┌─────────────────────┐
│  Ray Head (6379)    │          │  Ray Worker          │
│  vLLM API (:8000)   │◄────────►│                      │
│  GB10 GPU            │ 200Gbps │  GB10 GPU            │
│  TP rank 0           │  RoCE   │  TP rank 1           │
└─────────────────────┘          └─────────────────────┘
```

### 엔트리포인트 동작 방식

`entrypoint.sh`는 `ROLE`과 `TP_SIZE`에 따라 자동으로 분기합니다:

| ROLE | TP_SIZE | 동작 |
|---|---|---|
| `head` | 1 | `vllm serve` 직접 실행 (Ray 없음) |
| `head` | 2+ | Ray head 시작 → 워커 대기 → `vllm serve --distributed-executor-backend ray` |
| `worker` | any | `ray start --block` (head에 참여) |

### 리포지토리 구조

```
vllm-spark/
├── docker-compose.yml          # 통합 compose (head + worker 프로필)
├── entrypoint.sh               # 스마트 엔트리포인트 (TP1/TP2 자동 분기)
├── .env.example                # 전체 설정 템플릿
├── Dockerfile.gemma4           # v020-ngc2603 (NGC 26.03, 최신)
├── Dockerfile.ngc2603-v3       # v018-ngc2603 (NGC 26.03, 아카이브)
├── models/                     # 검증된 모델 프리셋
│   ├── gemma4-26b-a4b.env      # Gemma 4 26B MoE (TP1)
│   ├── redhatai-122b-nvfp4.env # RedHatAI NVFP4 (TP1)
│   ├── intel-122b-int4.env     # Intel INT4 AutoRound (TP1)
│   ├── wangzhang-122b-fp8.env  # 탈검열 FP8 (TP2)
│   ├── wangzhang-122b-nvfp4.env # 탈검열 NVFP4 (TP1)
│   ├── qwen3.5-397b-int4.env   # 397B INT4 (TP2)
│   ├── qwen3.5-122b-fp8.env
│   ├── qwen3.5-122b-nvfp4.env
│   └── qwen3.5-122b-nvfp4-tp2.env
├── benchmarks/                 # llama-benchy 벤치마크 결과
│   ├── results_intel-int4-tp1.json
│   ├── results_wangzhang-fp8-tp2.json
│   └── results_wangzhang-nvfp4-tp1.json
├── patches/                    # SM121 / PyTorch 2.11 호환성 패치
│   ├── fix_pytorch211_compat.py   # hoist=True 제거 (PyTorch 2.11)
│   └── ...
└── scripts/
    ├── run-cluster-node.sh     # 수동 Ray 클러스터 부트스트랩
    ├── verify_imports.py       # 빌드/런타임 검증
    └── verify_runtime.sh       # GPU 포함 전체 런타임 검증
```

## 설정

모든 설정은 `.env`를 통해 관리합니다. 전체 문서는 [`.env.example`](.env.example)을 참고하세요.

### 주요 변수

| 변수 | 설명 | 예시 |
|---|---|---|
| `VLLM_IMAGE` | Docker 이미지 (로컬 또는 GHCR) | `ghcr.io/bjk110/vllm-spark:v020-ngc2603` |
| `MODEL_PATH` | 호스트의 모델 가중치 경로 | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | 컨테이너 내 마운트 경로 | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API 모델 이름 | `Qwen/Qwen3.5-397B-...` |
| `TP_SIZE` | 텐서 병렬 크기 (1=단독, 2+=Ray) | `2` |
| `VLLM_EXTRA_ARGS` | 모델별 vllm serve 추가 플래그 | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | INT4 AutoRound 활성화 | `1` (비활성화: 빈 값) |

## 패치

Dockerfile에서 적용하는 SM121 (Blackwell) 호환성 패치:

| 패치 | 목적 | 상태 |
|---|---|---|
| `fix_pytorch211_compat` | `hoist=True` 제거 (PyTorch 2.11) | 활성 |
| `fastsafetensors_natural_sort` | 멀티노드 가중치 로딩 순서 수정 | 활성 |
| `aot_cache_fix` | AOT 캐시 torch.fx.Node pickling 수정 | 활성 |
| `nogds_force` | `nogds=True` 강제 (GB10은 GDS 미지원) | 활성 |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 분리, TRITON_PTXAS | 활성 |
| `moe_config_e256/e512` | GB10 튜닝 MoE 커널 설정 | 활성 |
| ~~`fix_cuda13_memcpy_batch`~~ | `cuMemcpyBatchAsync` API 수정 | 제거 (upstream 반영) |
| ~~`qwen3_5_moe_rope_fix`~~ | RoPE 검증 수정 | 제거 (upstream 반영) |
| ~~`pr38423_nvfp4_spark`~~ | NVFP4 DGX Spark 수정 | 제거 (upstream 반영) |

## 벤치마크 결과

모든 벤치마크는 [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4로 측정했습니다.

### Gemma 4 — 싱글 노드 (TP1, BF16)

| 동시 요청 수 | 26B MoE (4B active) | 31B Dense |
|---|---|---|
| 1 | 25.0 (피크 26) | 4.0 (피크 5) |
| 2 | 45.9 (피크 49) | 7.9 (피크 8) |
| 4 | 67.2 (피크 77) | 14.1 (피크 17) |

| 지표 | 26B MoE | 31B Dense |
|---|---|---|
| TTFT c=1 | 417 ms | 653 ms |
| KV 캐시 | 224K 토큰 (51.3 GiB) | 77K 토큰 (35.2 GiB, FP8) |

### Qwen3.5 122B — Decode 처리량 비교 (t/s)

| 동시 요청 수 | FP8 TP2 (탈검열) | INT4 TP1 (Intel) | NVFP4 TP1 (탈검열) |
|---|---|---|---|
| 1 | 31.5 (피크 32.5) | 29.7 (피크 30) | 17.0 (피크 18) |
| 2 | 42.4 (피크 54) | 57.6 (피크 59) | 33.3 (피크 35) |
| 4 | 59.7 (피크 91) | 52.1 (피크 97) | 55.2 (피크 65) |

| 지표 | FP8 TP2 | INT4 TP1 | NVFP4 TP1 |
|---|---|---|---|
| TTFT c=1 | 1,989 ms | 1,098 ms | 984 ms |
| KV 캐시 | 839K 토큰 (38.5 GiB/노드) | 789K 토큰 (36.2 GiB) | 155K 토큰 (14.3 GiB) |

### 397B INT4 TP2

#### 단일 요청 (concurrency=1)

| 테스트 | 처리량 (t/s) | TTFT (ms) |
|---|---|---|
| pp512 | 967 ± 33 | 543 ± 25 |
| pp1024 | 1,349 ± 2 | 776 ± 2 |
| pp2048 | 1,704 ± 9 | 1,224 ± 7 |
| tg128 | 27.0 ± 0.1 | — |

#### 동시 요청 — 총 Decode 처리량 (t/s)

| 동시 요청 수 | tg128 총합 | tg128 피크 |
|---|---|---|
| 1 | 27.0 | 28 |
| 2 | 45.3 | 52 |
| 4 | 60~67 | 85~88 |
| 8 | 59~91 | 152~160 |

### Qwen3.6-35B-A3B — 싱글 노드 (TP1, FP16 + fp8 KV) ⚗️

실험용 테스트 프리셋 (아래 [실험: Qwen3.6-35B-A3B FP16 테스트 프리셋](#실험-qwen36-35b-a3b-fp16-테스트-프리셋) 참고).
원본 bf16/fp16 가중치, fp8 KV cache, 32K 컨텍스트, `spark01` 단일 노드 측정.

| 동시 요청 수 | pp2048 총 t/s | tg32 총 t/s | tg32 요청당 t/s | 피크 tg t/s |
|---|---|---|---|---|
| 1 | 3,032 ± 825 | 32.4 ± 0.1 | 32.4 | 33 |
| 2 | 4,724 ± 75 | 63.9 ± 2.2 | 32.0 | 66 |
| 3 | 4,783 ± 439 | 61.1 ± 10.8 | 21.5 | 72 |
| 4 | 5,206 ± 444 | 80.1 ± 19.2 | 22.4 | 101 |

TTFT c=1: 약 746 ms (pp2048).

## 시스템 튜닝

DGX Spark에 권장하는 OS 수준 설정:

```bash
# Swap 부담 감소 (통합 메모리)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## 실험: Qwen3.6-35B-A3B FP16 테스트 프리셋

> Qwen3.6 원본 가중치를 단일 DGX Spark에서 빠르게 평가하기 위한 **실험용 테스트
> 프리셋**입니다. **베이스 스택 변경 아님** — 메인 이미지, vLLM, FlashInfer,
> transformers, CUDA 버전은 모두 그대로입니다.

- **프리셋 파일**: `models/qwen3.6-35b-fp16.env`
- **범위**: `단일 DGX Spark / TP=1` (GB10 1대에 여유를 가지고 올라가도록 설계)
- **모델**: Qwen3.6-35B-A3B **원본 가중치** (bf16/fp16, **양자화 아님**).
  `--kv-cache-dtype fp8`은 **KV cache 전용 최적화**일 뿐이며 모델 가중치를
  바꾸지 않습니다.
- **권장 옵션** (프리셋에 이미 포함):
  - `--kv-cache-dtype fp8` (KV cache 압축만)
  - `--reasoning-parser qwen3`
  - `--enable-chunked-prefill`
  - `--enable-prefix-caching` (entrypoint 기본값)

### 기동 전: 현재 구동 중인 397B TP=2 스택 중지

```bash
ssh spark01 'cd ~/docker/vllm-spark && docker compose --profile head down'
ssh spark02 'cd ~/docker/vllm-spark && docker compose --profile worker down'
# GB10 unified memory 잔여 정리 (모델 전환 시 필수)
ssh spark01 'sync && sudo sysctl -w vm.drop_caches=3'
```

### 모델 위치

homeserver의 `/mnt/data/llm-models/Qwen/Qwen_Qwen3.6-35B-A3B`에 다운로드되어
있다는 전제입니다. 테스트 대상 Spark 노드(권장: `spark01`, 기존 397B head와
동일)로 복사 후 `MODEL_PATH`를 로컬 경로로 지정해 사용합니다.

```bash
# homeserver에서 (~67 GB, RoCE 링크로 ~6분)
rsync -av /mnt/data/llm-models/Qwen/Qwen_Qwen3.6-35B-A3B/ \
    spark01:/home/bjk110/Documents/Models/Qwen/Qwen_Qwen3.6-35B-A3B/

# spark01: 프리셋 복사 + 로컬 경로 치환
ssh spark01 'cd ~/docker/vllm-spark && \
    cp models/qwen3.6-35b-fp16.env .env && \
    sed -i "s|\[model_path\]|/home/bjk110/Documents/Models/Qwen|" .env'
```

### 기동 (단일 Spark, TP=1)

```bash
ssh spark01 'cd ~/docker/vllm-spark && \
    docker compose --env-file .env --profile head up -d'
```

### 초기 기동 실패 시 조정 순서

`qwen3.6-35b-fp16.env` 안에서 메모리 압박을 낮추는 순서대로 조정합니다.

1. `GPU_MEMORY_UTILIZATION=0.80`
2. `MAX_MODEL_LEN=16384`
3. `MAX_NUM_SEQS=4`
4. 그래도 실패 시 `spark01` + `spark02` TP=2 구성을 검토 (현재 프리셋은
   TP=1 전용 — 이번 실험 프리셋 범위를 넘어섬).

## 브랜치 구조

이 저장소는 현재 두 개의 주 브랜치로 관리됩니다:

- **`main`**: 현재 베이스 브랜치
  vLLM / FlashInfer / Transformers / 컨테이너 기준이 갱신된 refresh 스택을 포함합니다.

- **`feat/turboquant-rebase-20260417`**: 활성 TurboQuant 브랜치
  현재 베이스 브랜치 위에서 TurboQuant 통합·검증·후속 실험을 진행하는 브랜치입니다.

### 아카이브된 브랜치 이력

이전의 실험용 브랜치들은 `main` 에 머지되었거나 현재의 TurboQuant 리베이스 작업으로 대체된 후 정리되었습니다.

구형 TurboQuant 브랜치는 태그로 보존되어 있습니다:

- **`archive/feat-turboquant`**

필요하면 다음과 같이 복원할 수 있습니다:

```bash
git checkout -b feat/turboquant archive/feat-turboquant
```

## 라이선스

설정 파일은 참고용으로 제공됩니다. 모델은 해당 라이선스를 따릅니다 ([Qwen 라이선스](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [Gemma 라이선스](https://ai.google.dev/gemma/terms)).
