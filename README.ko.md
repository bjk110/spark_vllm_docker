# vLLM Spark — DGX Spark (GB10) 통합 서빙

한국어 | **[English](README.md)**

NVIDIA DGX Spark (GB10) 통합 vLLM 서빙 구성입니다. 동일한 리포 / Dockerfile /
compose 파일로 두 가지 토폴로지를 지원합니다:

- **단일 Spark** (기본값, RDMA 설정 불필요) — GB10 한 대, TP=1.
- **듀얼 Spark + 200 Gbps RoCE/IB** — GB10 두 대, Ray, TP=2.

`.env`에서 `CLUSTER_MODE=single` (기본) 또는 `CLUSTER_MODE=dual-rdma`로 토폴로지를
선택합니다. 자세한 내용은 아래 [`빠른 시작`](#빠른-시작)을 참고하세요.

릴리스별 변경 이력은 [`CHANGELOG.md`](CHANGELOG.md), 패치별 상태/제거 조건은
[`PATCH_STATUS.md`](PATCH_STATUS.md)를 참고하세요.

## 하드웨어

| 토폴로지 | 노드 | 역할 | GPU | 메모리 | 인터커넥트 |
|---|---|---|---|---|---|
| single | Spark 한 대 | vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 해당 없음 |
| dual-rdma | spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |
| dual-rdma | spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |

## 소프트웨어 스택

### v022-d568 (현재 메인 — NGC 26.04, vLLM v0.21.0+#35568, FlashInfer 0.6.11.post3, Transformers 5.8.1, Triton 3.7.0, NCCL 2.30.4)

v022 forward-stack 의 최종층 (2026-05-18 빌드). PrismaSCOUT NVFP4 TP=2 (텍스트 + 이미지, MTP n=3) 로 각 중간 단계 검증, 최종 `-d568` 층에서 `wangzhang-122b-abliterix-fp8-tp2` (SM121 FP8 커널 경로 활성 확인) + `wangzhang-122b-abliterix-nvfp4-tp2` (FlashInfer-CUTLASS NVFP4 GEMM + MoE 경로) + `gemma4-31b-it` (dense BF16 멀티모달) + `qwen3.6-35b-a3b` (하이브리드 Mamba/Attention MoE) 검증 완료.

| 구성요소 | 버전 |
|---|---|
| 베이스 이미지 | NGC PyTorch **26.04-py3** |
| vLLM | **0.21.0 + PR #35568** (릴리스 태그 `ad7125a4` + commit `06d020bb6` cherry-pick, 소스 재빌드) |
| FlashInfer | **v0.6.11.post3** (SM120/121 XQA MLA 버그픽스 #2689, CUTLASS Small Tile N #3152, Blackwell GDN 정확도 #3156, SM120 cuDNN NaN #3192, NVFP4 KV prefill #3097) |
| PyTorch | **2.12.0a0** |
| CUDA | 13.2 (네이티브) |
| Transformers | **5.8.1** |
| Triton | **3.7.0** (vanilla PyPI; NGC 26.04 번들은 3.6.0) |
| NCCL | **2.30.4** (`nvidia-nccl-cu13` pip + `LD_LIBRARY_PATH` 런타임 override; NGC 26.04 시스템 NCCL 은 2.29.7 유지) |
| 이미지 태그 | `ghcr.io/bjk110/vllm-spark:v022-d568` (**GHCR**, digest `sha256:88b544ed`) |

**런타임 패치:**
- `patches/patch_split_module_compat.py`: vLLM 의 정적 `is_torch_equal_or_newer("2.12.0.dev")` gate 를 런타임 signature probe 로 교체 (NGC 26.04 PyTorch 2.12 alpha 가 `tuple_return` kwarg 부재)
- `patches/apply_sm121_fp8_pr35568.py` (`-d568` 전용): vLLM PR #35568 빌드 타임 cherry-pick. Marlin / CUTLASS FP8 codepath 의 SM120-only / `[89, 120]` gate 를 SM12x family 로 확장하여 GB10 (SM121) 포함

**검증된 preset overrides:**
- `models/qwen3.6-27b-prismascout-nvfp4-tp2-v022-{fi0611,ngc2604,tx581,trt37,nccl234,d568}.env` — PrismaSCOUT NVFP4 (텍스트 + 이미지)
- `models/wangzhang-122b-abliterix-fp8-tp2-v022-d568.env` — abliterix FP8 (FP8 커널 경로 활성 확인)
- `models/wangzhang-122b-abliterix-nvfp4-tp2.env` — abliterix NVFP4 (fused-group 공유 `weight_global_scale`)
- `models/gemma4-31b-it.env` — Gemma 4 31B IT (dense BF16 멀티모달, single TP=1)
- `models/qwen3.6-35b-a3b.env` — Qwen3.6-35B-A3B (하이브리드 Mamba/Attention MoE)

### dsv4-d568 (v022-d568 파생 — GB10 의 DeepSeek-V4-Flash)

v022-d568 의 vLLM 을 **jasl/vllm @ `edc82b614f51`** (branch `codex/ds4-sm120-min-enable` HEAD, 2026-05-19) 로 교체. SM12x DSV4 지원 (sparse MLA, Lightning Indexer, fp8_ds_mla KV cache, MTP heads). 나머지 레이어는 v022-d568 동일.

| 구성요소 | 버전 |
|---|---|
| 베이스 이미지 | `ghcr.io/bjk110/vllm-spark:v022-d568` |
| vLLM | **jasl/vllm @ `edc82b614f51`** (소스 재빌드, v0.20.0.dev) |
| 추가 패치 | `apply_dsv4_packed_mapping.py`, `patch_split_module_compat.py` (재적용), `moe_config_e256/e512.json` (재배치), `instanttensor` pip dep |
| 이미지 태그 | `ghcr.io/bjk110/vllm-spark:dsv4-d568` (**GHCR**, digest `sha256:b18da2a0`) |

검증된 preset: `models/dsv4-flash-fp8-tp2.env` — DeepSeek-V4-Flash dual-rdma TP=2, 200K ctx, fp8 KV cache + Lightning Indexer.

**전체 가이드 + 9-way 벤치마크 sweep + MTP/backend 분석**: [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md).

### 이전 / 레거시 스택

이전 이미지와 v022 중간 단계는 별도 문서로 분리되어 있습니다:

| 스택 | 사용 시점 | 상세 |
|---|---|---|
| `v021-ngc2603` / `v021-tq` | 대부분 프리셋의 운영 기본값 (`models/*.env` 의 이미지 컬럼 = `v021-ngc2603`); `*-tq` (TurboQuant) 프리셋 필수 | [`docs/stack-v021.md`](docs/stack-v021.md) |
| `v022-vllm021` / `v022-tx581` / `v022-{fi0611,ngc2604,trt37,nccl234}` | v022 스택 중간 단계 (local-build only, `v022-d568` 와의 bisection / rollback 용도로 보존) | [`docs/stack-v022.md`](docs/stack-v022.md) |
| `v019-ngc2603` | 아카이브 (vLLM 0.19.1 + Gemma 4 + async scheduling). 역사적 재현용. | [`docs/stack-v019.md`](docs/stack-v019.md) |

릴리스별 변경 이력은 [`CHANGELOG.md`](CHANGELOG.md), 패치별 upstream 추적은 [`PATCH_STATUS.md`](PATCH_STATUS.md) 참고.

## 지원 모델

| 프리셋 | 모델 | 양자화 / dtype | 토폴로지 | TP | 이미지 | 비고 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b.env` | google/gemma-4-26B-A4B-it | BF16 MoE (26B/4B active) | single | 1 | v021-ngc2603 | — |
| `gemma4-26b-a4b-tq.env` | google/gemma-4-26B-A4B-it | BF16 MoE + **TurboQuant KV** (`turboquant_k8v4`) | single | 1 | v021-tq | TQ 빌드 포함 |
| `qwen3.5-122b-fp8.env` | Qwen/Qwen3.5-122B-A10B-FP8 | FP8 (멀티모달) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-122b-nvfp4.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 (런타임, FlashInfer) | single | 1 | v021-ngc2603 | — |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen/Qwen3.5-122B-A10B | NVFP4 (런타임, FlashInfer) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-122b-prismaquant.env` | rdtand/Qwen3.5-122B-A10B-PrismaQuant-4.75bit-vllm | PrismaQuant 4.76bpp (NVFP4+MXFP8+BF16 혼합) | single | 1 | v021-ngc2603 | MTP `n=1` 기본 |
| `redhatai-122b-nvfp4.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 (사전 양자화) | single | 1 | v021-ngc2603 | — |
| `redhatai-122b-nvfp4-tq.env` | RedHatAI/Qwen3.5-122B-A10B-NVFP4 | NVFP4 + **TurboQuant KV** | single | 1 | v021-tq | TQ 빌드 포함 |
| `intel-122b-int4.env` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | INT4 AutoRound (Marlin) | single | 1 | v021-ngc2603 | — |
| `wangzhang-122b-fp8.env` | wangzhang/Qwen3.5-122B-A10B-abliterated | FP8 (텍스트 전용, 탈검열) | dual-rdma | 2 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1` |
| `wangzhang-122b-nvfp4.env` | wangzhang/Qwen3.5-122B-A10B-abliterated-NVFP4 | NVFP4 (텍스트 전용, 탈검열) | single | 1 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1` |
| `wangzhang-122b-abliterix-fp8-tp2.env` | wangzhang/Qwen3.5-122B-A10B-abliterix | FP8 W8A8 (텍스트 전용, 자체 safetensors 양자화) | dual-rdma | 2 | v021-ngc2603 | `APPLY_TEXT_ONLY_SHIM=1`; BF16→FP8 변환은 `quantize_qwen35_abliterix_fp8_direct.py` 사용 |
| `wangzhang-122b-abliterix-nvfp4-tp2.env` | wangzhang/Qwen3.5-122B-A10B-abliterix | NVFP4 W4A4 (텍스트 전용, fused-group 공유 `weight_global_scale`) | dual-rdma | 2 | v022-d568 | `APPLY_TEXT_ONLY_SHIM=1`; BF16→NVFP4 변환은 `fp8-quantizer/convert_bf16_to_nvfp4.py`. q/k/v_proj, expert gate/up, shared_expert gate/up가 fused group마다 단일 `weight_global_scale` 공유 — FlashInfer NVFP4 fused Linear에서 scale 불일치 시 정밀도 손실 경고 + 출력 깨짐. NGC 26.04 스택(v022-d568)에서 검증. `model_type=qwen3_5_moe_text` (flat config)로 wrapper의 default `text_config.hidden_size=2048` 함정 회피 |
| `qwen3.5-397b-int4.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound (Marlin) | dual-rdma | 2 | v021-ngc2603 | — |
| `qwen3.5-397b-int4-tq.env` | Intel/Qwen3.5-397B-A17B-int4-AutoRound | INT4 AutoRound + **TurboQuant KV** (`turboquant_3bit_nc` 캐스케이드) | dual-rdma | 2 | v021-tq | TQ 빌드 포함; `--compilation-config {"use_inductor_graph_partition":true}` 사용 |
| `qwen3.6-35b-fp16.env` ⚗️ | Qwen/Qwen3.6-35B-A3B | **FP16 원본** (KV fp8) | single | 1 | v021-ngc2603 | 실험용 |
| `qwen3.6-35b-a3b.env` | Qwen/Qwen3.6-35B-A3B | BF16 하이브리드 Mamba/Attention MoE (KV fp8) | single | 1 | v022-d568 | `--reasoning-parser qwen3` + 하이브리드 아키텍처용 `--compilation-config {"use_inductor_graph_partition":true}` |
| `gemma4-31b-it.env` | google/gemma-4-31B-it | BF16 dense 멀티모달 | single | 1 | v022-d568 | `--limit-mm-per-prompt {"image":1,"audio":0,"video":0}` (audio는 vLLM 0.21에서 아직 beta) |
| `dsv4-flash-fp8-tp2.env` | deepseek-ai/DeepSeek-V4-Flash | FP8 (E4M3 128×128 block, 공식) | dual-rdma | 2 | **dsv4-d568** | DSV4 sparse MLA + Lightning Indexer + fp8_ds_mla KV + MTP heads. 빌드/레시피/9-way 벤치마크 상세는 [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md). 운영 best: `MAX_NUM_SEQS=4`, MTP off, Ray (peak 69 t/s decode, 800 t/s prefill at c=4). |

## 빠른 시작

### 0. Docker 이미지 준비

#### 방법 A: GHCR에서 빌드된 이미지 Pull

```bash
# 기본 이미지 (모든 모델, TQ 패치 미포함)
docker pull ghcr.io/bjk110/vllm-spark:v021-ngc2603

# TurboQuant 이미지 (기본 + 하이브리드 모델 TQ 버그픽스 패치)
docker pull ghcr.io/bjk110/vllm-spark:v021-tq
```

#### 방법 B: 소스에서 빌드

```bash
# NGC 26.03 소스 빌드 (vLLM main, TurboQuant 포함)
docker buildx build -f dockerfiles/Dockerfile.gemma4 \
  -t vllm-spark:v021-ngc2603 --load .
```

빌드 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `BUILD_JOBS` | 16 | 병렬 빌드 작업 수 |
| `FLASHINFER_REF` | v0.6.9 | FlashInfer git ref |
| `VLLM_COMMIT` | 95995bbe | vLLM 소스 커밋 |
| `TORCH_CUDA_ARCH` | 12.1a | 타겟 CUDA 아키텍처 (Blackwell) |

### 1. 모델 프리셋 선택

단일 Spark 프리셋은 `CLUSTER_MODE=single`, TP=1로 출고됩니다 (RDMA 설정 불필요).
듀얼 Spark 프리셋은 `CLUSTER_MODE=dual-rdma`, TP=2로 출고됩니다.

```bash
# 단일 Spark (RDMA 불필요):
cp models/redhatai-122b-nvfp4.env .env

# 듀얼 Spark + RoCE:
cp models/qwen3.5-397b-int4.env .env
```

`.env`의 `MODEL_PATH`를 실제 모델 가중치 경로로 수정합니다:

```bash
sed -i 's|\[model_path\]|/home/user/models|' .env
```

### 2. 서비스 시작

#### 단일 Spark — TP=1 (기본, Ray/RDMA 없음)

```bash
docker compose --profile head up -d
docker logs -f vllm-spark-head
```

`entrypoint.sh`가 `CLUSTER_MODE=single`을 읽고 **강제로** `VLLM_HOST_IP=127.0.0.1`,
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `UCX_NET_DEVICES` / `NCCL_IB_HCA` unset,
`NCCL_IB_DISABLE=1`을 적용합니다. 이게 단일 Spark에서 발생하는
`tcp://10.10.10.1:<port> server socket has timed out` c10d hang을 막아주는 핵심입니다
(아래 [트러블슈팅](#트러블슈팅) 참조).

#### 듀얼 Spark — TP=2 (Ray + RoCE)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

Head는 Worker가 Ray 클러스터에 참여할 때까지 대기한 후
`--distributed-executor-backend ray`로 vLLM을 시작합니다. `.env`에 `HEAD_ROCE_IP`,
`WORKER_ROCE_IP`, `ROCE_IF_NAME`, `IB_HCA_NAME`, `RAY_PORT`가 필요합니다
(`CLUSTER_MODE=dual-rdma` 프리셋은 해당 블록을 활성화한 상태로 출고됩니다).

#### 백엔드 선택 — `DISTRIBUTED_BACKEND=ray | mp`

`dual-rdma` 배포에서 두 노드 간 조율 방식을 `DISTRIBUTED_BACKEND` 환경변수로 선택 (기본 `ray`):

| 모드 | 동작 방식 | 사용 시점 |
|---|---|---|
| `ray` (기본) | head 가 `ray start --head` 실행, worker 가 `ray start --address=…` 로 join, vLLM 이 `--distributed-executor-backend ray` 로 서빙 | 기존 모든 멀티노드 프리셋의 검증된 경로 |
| `mp` | head + worker 모두 `vllm serve` 동시 실행 (SPMD); head 는 `--nnodes N --node-rank 0 --master-addr <head> --master-port <port>`; worker 는 추가로 `--headless`. Ray 없음. eugr/spark-vllm-docker `--no-ray` 경로와 동일. | Ray Compiled DAG 멀티노드 버그 [#36237](https://github.com/vllm-project/vllm/issues/36237) 회피 필요 시; 일부 forum recipe (예: DeepSeek-V4-Flash) 가 요구 |

전환은 env 한 줄 — 이미지 / compose 동일, 재기동만 하면 됨:

```
# models/<preset>.env 에서
DISTRIBUTED_BACKEND=mp   # 또는 ray (기본)
MASTER_PORT=29501        # mp 모드에서만 사용
```

DSV4 측정에서 GB10 환경 Ray vs mp 성능 차이 ±2% — 자세한 비교는 [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md) §5 참조.

### 3. 동작 확인

```bash
curl http://localhost:8000/health      # 단일
curl http://spark01:8000/health        # 듀얼-rdma
```

## 트러블슈팅

### `SyntaxError: invalid syntax` (`compilation/codegen.py`, EngineCore init 시 — Qwen3.5 하이브리드 + torch.compile)

```
File "<string>", line 5
    gdn_attention_core = torch.ops.vllm.gdn_attention_core(..., <vllm.utils.torch_utils.LayerName object at 0x...>)
                                                                ^
SyntaxError: invalid syntax
```

vLLM main `951dca80` (PR #38657 "[compile] Invoke split FX graph by codegen") 이후 codegen 단계에서 `LayerName` 같은 opaque 인자에 default `repr()`을 사용해 생성 코드 안에 박음. Qwen3.5 하이브리드의 GDN attention 경로가 `LayerName`을 인자로 받아 cold start마다 트리거됨.

**권장 우회 (소스 패치 불필요)** — `use_inductor_graph_partition=True` 로 torch.compile이 Inductor 자체 partition을 쓰게 해 vLLM split-by-codegen 경로를 우회:

```
VLLM_EXTRA_ARGS=... --compilation-config {"use_inductor_graph_partition":true}
```

torch.compile + CUDAGraph (`FULL_AND_PIECEWISE`) 모두 유지. cold-start engine init이 약 2배 (397B INT4 TP=2 기준 ≈ 440s vs `--enforce-eager` 250s) 길어지지만, 정상 추론은 CUDA graph + Inductor 최적화 활용.

**최후 수단** — `--enforce-eager`. torch.compile/CUDAGraph 모두 끔. 성능 손실 크지만 codegen 경로 자체를 우회하므로 확실.

**핫패치 (대기 중)** — `patches/patch_codegen_fx_repr.py` 가 `_node_ref()` 를 `__fx_repr__()` 인지하도록 재작성하고 namespace 병합. Inductor partition 회귀나 다른 opaque type 도입 시에만 적용:

```bash
docker exec vllm-spark-head python3 /patches/patch_codegen_fx_repr.py
docker compose --profile head restart
```

### `[c10d] The server socket on [::ffff:10.10.10.1]:<port> has timed out, will retry.`

단일 Spark 환경에서 RDMA 환경변수(`VLLM_HOST_IP=10.10.10.1`,
`GLOO_SOCKET_IFNAME=enp1s0f0np0` 등)가 컨테이너로 새어 들어가, PyTorch c10d가
이 호스트에 존재하지 않는 RoCE IP에 바인드를 시도하기 때문에 발생합니다. 해결:

1. `.env` (또는 복사한 프리셋)에 `CLUSTER_MODE=single`이 있는지 확인합니다.
2. 단일 모드 프리셋의 `HEAD_ROCE_IP=…` / `ROCE_IF_NAME=…` / `IB_HCA_NAME=…` 행이
   **주석 처리** (`#`로 시작)되어 있는지 확인합니다.
3. 컨테이너를 재생성합니다:
   ```bash
   docker compose --profile head down
   docker compose --profile head up -d
   ```
   정상 시작 시 `entrypoint.sh`가
   `CLUSTER_MODE=single: VLLM_HOST_IP=127.0.0.1, NCCL_IB_DISABLE=1, NCCL/GLOO/UCX ifname cleared`
   로그를 출력합니다.

## 아키텍처

### 단일 Spark (CLUSTER_MODE=single, TP=1)

```
single Spark
┌──────────────────────────────────┐
│  vLLM API (:8000)                │
│  c10d binds 127.0.0.1            │
│  GB10 GPU, TP rank 0             │
└──────────────────────────────────┘
```

### 듀얼 Spark (CLUSTER_MODE=dual-rdma, TP=2)

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

`entrypoint.sh`는 `CLUSTER_MODE`에 따라 환경변수를 정규화한 다음 `ROLE` × `TP_SIZE`로 분기합니다:

| CLUSTER_MODE | ROLE | TP_SIZE | 동작 |
|---|---|---|---|
| `single`     | `head`   | 1   | `VLLM_HOST_IP=127.0.0.1` 강제, NCCL/GLOO/UCX ifname 제거, `NCCL_IB_DISABLE=1`, 그 후 `vllm serve` 직접 실행 (Ray 없음) |
| `single`     | `head`   | 2+  | Fail-fast (`single`은 TP≥2를 호스팅 불가) |
| `single`     | `worker` | any | Fail-fast (단일 모드에서 worker는 의미 없음) |
| `dual-rdma`  | `head`   | 1   | 거부 (TP=1은 `single`을 사용) |
| `dual-rdma`  | `head`   | 2+  | RDMA 환경변수 검증 → Ray head → 워커 대기 → `vllm serve --distributed-executor-backend ray` |
| `dual-rdma`  | `worker` | any | `ray start --address=$HEAD_ROCE_IP:$RAY_PORT --block` |

### 리포지토리 구조

```
vllm-spark/
├── docker-compose.yml             # 통합 compose (head + worker 프로필)
├── entrypoint.sh                  # CLUSTER_MODE 인지 entrypoint
├── .env.example                   # 전체 설정 템플릿
├── Dockerfile.v022-d568           # 현재 베이스 이미지 빌드 (NGC 26.04 스택)
├── Dockerfile.dsv4-d568           # DeepSeek-V4-Flash 파생 (FROM v022-d568)
├── dockerfiles/                   # 레거시 / 중간 빌드 (bisection 보존용)
│   ├── Dockerfile                     # NGC 26.01 시대 (vLLM 0.18.x, 역사적)
│   ├── Dockerfile.gemma4              # v021-ngc2603 통합 빌드
│   ├── Dockerfile.ngc2603-v3          # v018-ngc2603 아카이브 빌드
│   ├── Dockerfile.nvfp4               # NVFP4 런타임 기본값 오버레이
│   └── Dockerfile.v022(-fi0611/-ngc2604/-tx581/-trt37/-nccl234)  # v022 스택 중간 단계
├── CHANGELOG.md                   # 릴리스별 변경 이력
├── PATCH_STATUS.md                # 패치별 목적/상태/제거 조건
├── models/                        # 검증된 모델 프리셋
│   ├── gemma4-26b-a4b.env             # Gemma 4 26B MoE (single, TP1)
│   ├── gemma4-26b-a4b-tq.env          # Gemma 4 + TurboQuant KV (single, TP1)
│   ├── redhatai-122b-nvfp4.env        # RedHatAI NVFP4 (single, TP1)
│   ├── redhatai-122b-nvfp4-tq.env     # RedHatAI NVFP4 + TurboQuant (single, TP1)
│   ├── intel-122b-int4.env            # Intel INT4 AutoRound (single, TP1)
│   ├── wangzhang-122b-fp8.env         # 탈검열 FP8 (dual-rdma, TP2)
│   ├── wangzhang-122b-nvfp4.env       # 탈검열 NVFP4 (single, TP1)
│   ├── qwen3.5-397b-int4.env          # 397B INT4 (dual-rdma, TP2)
│   ├── qwen3.5-397b-int4-tq.env       # 397B INT4 + TurboQuant (dual-rdma, TP2)
│   ├── qwen3.5-122b-fp8.env           # 122B FP8 멀티모달 (dual-rdma, TP2)
│   ├── qwen3.5-122b-nvfp4.env         # 122B NVFP4 런타임 (single, TP1)
│   ├── qwen3.5-122b-nvfp4-tp2.env     # 122B NVFP4 런타임 (dual-rdma, TP2)
│   ├── qwen3.5-122b-prismaquant.env   # PrismaQuant 4.76bpp 혼합 (single, TP1)
│   ├── wangzhang-122b-abliterix-fp8-tp2.env  # abliterix FP8 W8A8 텍스트 전용 (dual-rdma, TP2)
│   ├── wangzhang-122b-abliterix-nvfp4-tp2.env # abliterix NVFP4 W4A4 텍스트 전용 (dual-rdma, TP2; v022-d568)
│   ├── gemma4-31b-it.env             # Gemma 4 31B IT BF16 dense 멀티모달 (single, TP1; v022-d568)
│   ├── qwen3.6-35b-a3b.env           # Qwen3.6-35B-A3B BF16 하이브리드 MoE (single, TP1; v022-d568)
│   ├── dsv4-flash-fp8-tp2.env        # DeepSeek-V4-Flash 공식 FP8 (dual-rdma, TP2; dsv4-d568)
│   └── qwen3.6-35b-fp16.env           # ⚗️ Qwen3.6 FP16 실험 (single, TP1)
├── docs/                          # 모델별 상세 가이드
│   └── dsv4-flash-tp2.md             # DSV4-Flash 빌드/레시피/9-way 벤치마크 sweep
├── benchmarks/                    # llama-benchy 벤치마크 결과
├── patches/                       # SM121 / PyTorch 2.11 / TurboQuant 패치
│   ├── fix_pytorch211_compat.py       # hoist=True 제거 (PyTorch 2.11)
│   ├── fastsafetensors_natural_sort.patch
│   ├── aot_cache_fix.patch
│   ├── nogds_force.patch
│   ├── apply_sm121_patches.py
│   ├── moe_config_e256.json / moe_config_e512.json
│   ├── apply_turboquant_fixes.py      # v021-tq 전용
│   ├── patch_qwen35_moe_text.py       # APPLY_TEXT_ONLY_SHIM=1 시
│   ├── patch_codegen_fx_repr.py       # 대기 중 핫패치 (Troubleshooting 참고)
│   └── ...                            # 전체 목록은 PATCH_STATUS.md
└── scripts/
    ├── run-cluster-node.sh        # 수동 Ray 클러스터 부트스트랩
    ├── verify_imports.py          # 빌드 시점 import 검증
    └── verify_runtime.sh          # GPU 포함 전체 런타임 검증
```

## 설정

모든 설정은 `.env`를 통해 관리합니다. 전체 문서는 [`.env.example`](.env.example)을 참고하세요.

### 주요 변수

| 변수 | 설명 | 예시 |
|---|---|---|
| `VLLM_IMAGE` | Docker 이미지 (로컬 또는 GHCR) | `ghcr.io/bjk110/vllm-spark:v021-ngc2603` |
| `MODEL_PATH` | 호스트의 모델 가중치 경로 | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | 컨테이너 내 마운트 경로 | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API 모델 이름 | `Qwen/Qwen3.5-397B-...` |
| `CLUSTER_MODE` | 토폴로지: `single` (기본) 또는 `dual-rdma` | `single` |
| `TP_SIZE` | 텐서 병렬 크기 (1=single, 2+=dual-rdma) | `1` |
| `HEAD_ROCE_IP` | (`dual-rdma` 전용) head 노드 RoCE IP | `10.10.10.1` |
| `WORKER_ROCE_IP` | (`dual-rdma` 전용) worker 노드 RoCE IP | `10.10.10.2` |
| `ROCE_IF_NAME` | (`dual-rdma` 전용) RoCE 인터페이스 이름 | `enp1s0f0np0` |
| `IB_HCA_NAME` | (`dual-rdma` 전용) InfiniBand HCA 이름 | `rocep1s0f0` |
| `RAY_PORT` | (`dual-rdma` + `ray` 백엔드 전용) Ray head 포트 | `6379` |
| `DISTRIBUTED_BACKEND` | (`dual-rdma` 전용) `ray` (기본) 또는 `mp` (Ray 없는 SPMD) | `ray` |
| `MASTER_PORT` | (`mp` 백엔드 전용) torch.distributed master 포트 | `29501` |
| `VLLM_EXTRA_ARGS` | 모델별 vllm serve 추가 플래그 | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | INT4 AutoRound 활성화 | `1` (비활성화: 빈 값) |

## 패치

요약 — 패치별 목적·범위·upstream 추적·제거 조건은
[`PATCH_STATUS.md`](PATCH_STATUS.md) 참고.

| 패치 | 목적 | 상태 |
|---|---|---|
| `fix_pytorch211_compat` | `hoist=True` 제거 (PyTorch 2.11) | 활성 (build) |
| `fastsafetensors_natural_sort` | 멀티노드 가중치 로딩 순서 수정 | 활성 (build) |
| `aot_cache_fix` | AOT 캐시 torch.fx.Node pickling 수정 | 활성 (build) |
| `nogds_force` | `nogds=True` 강제 (GB10은 GDS 미지원) | 활성 (build) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 분리, TRITON_PTXAS | 활성 (build) |
| `moe_config_e256/e512` | GB10 튜닝 MoE 커널 설정 | 활성 (build) |
| `apply_turboquant_fixes` | TurboQuant KV 보정 (PR #40074, #39988, #39931) | 활성 (`v021-tq` 전용) |
| `patch_qwen35_moe_text` | 탈검열 Qwen3.5 MoE text-only shim | 조건부 (`APPLY_TEXT_ONLY_SHIM=1`) |
| `patch_codegen_fx_repr` | `compilation/codegen.py`에서 `__fx_repr__()` 인지 | 대기 중 (핫패치 — Troubleshooting 참조) |
| ~~`fix_cuda13_memcpy_batch`~~ | `cuMemcpyBatchAsync` API 수정 | 제거 (upstream 반영 — base-refresh-20260417) |
| ~~`qwen3_5_moe_rope_fix`~~ | RoPE 검증 수정 | 제거 (upstream 반영 — base-refresh-20260417) |
| ~~`pr38423_nvfp4_spark`~~ | NVFP4 DGX Spark 수정 | 제거 (upstream 반영 — base-refresh-20260417) |

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

### Qwen3.5-122B-A10B PrismaQuant — 싱글 노드 (TP1, 혼합 정밀도 + fp8 KV)

Fisher 민감도 기반 per-Linear 혼합 정밀도 체크포인트 (NVFP4 MoE 본체 / MXFP8 고민감 Linear / BF16 라우터·임베드).
가중치 72 GB, 피크 VRAM 약 86 GB (fp8 KV @ 32k) — 단일 GB10 한 장에 탑재.
모델에 MTP speculative-decoding 헤드가 포함되어 있으며, 본 프리셋은 로컬 튜닝 결과에 따라 **`n=1`을 기본값**으로 사용합니다.

**디코드 처리량 — MTP 설정별 비교 (llama-benchy tg32, 각 3회 실행):**

| 동시 요청 수 | MTP=3 총 / 피크 | MTP=1 총 / 피크 | MTP=0 총 / 피크 |
|---|---:|---:|---:|
| 1 | 11.2 / 12.5 | 15.7 / 16.4 | **19.1 / 20.0** |
| 2 | 20.5 / 23.0 | 25.7 / 28.7 | **30.4 / 38.0** |
| 3 | 21.1 / 24.0 | 30.3 / 34.0 | **39.8 / 49.0** |
| 4 | 29.2 / 33.7 | 45.1 / 50.7 | **65.1 / 72.3** |

**프리필 처리량 (pp2048 총 t/s) 및 TTFT (c=1):**

| MTP | pp c=1 | pp c=4 | TTFT c=1 |
|---|---:|---:|---:|
| n=3 | 1,744 | 2,262 | 1,026 ms |
| n=1 | 1,825 | 2,318 | 1,033 ms |
| n=0 | **1,989** | **2,555** | **947 ms** |

MTP speculative decoding은 스텝당 오버헤드가 있어 tg32 마이크로버스트(32 토큰 생성)에서는 오버헤드가 이득을 넘어 **MTP=0이 승**. 긴 자연문 생성에서는 수용률이 오르며 MTP=1이 MTP=0과 동등 또는 앞섬. 모델 카드에서 제시한 `n=3`은 본 하드웨어의 모든 처리량 구간에서 최악 — 추가 스펙큘레이션 토큰이 수용률을 낮추고 GB10에서 잘 상쇄되지 않음.

**Intel INT4 / RedHatAI NVFP4와 비교 (동일 TP=1, c=1):**

| 양자화 | 디스크 | pp2048 c=1 | tg32 c=1 | tg32 c=4 피크 |
|---|---:|---:|---:|---:|
| Intel INT4 AutoRound | ~65 GB | 2,084 | 29.8 | 96.0 |
| RedHatAI NVFP4 | ~60 GB | 2,027 | 16.2 | 60.0 |
| PrismaQuant (MTP=1) | 72 GB | 1,825 | 15.7 | 50.7 |
| PrismaQuant (MTP=0) | 72 GB | 1,989 | 19.1 | 72.3 |

GB10에서 순수 처리량은 Intel INT4가 여전히 가장 빠름. PrismaQuant의 강점은 Fisher 가중치 기반 per-Linear 할당(NVFP4 본체 + 고민감 Linear의 MXFP8 + 라우터/임베드의 BF16)에 의한 **bit당 품질** — 방법론은 모델 카드 참조.

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

### 397B INT4 TP2 — TurboQuant KV 캐시 모드 비교

동일 397B INT4 AutoRound 모델, `v021-tq` 이미지, TP=2 (spark01+spark02, 200Gbps RoCE), `max_model_len=32768`, `gpu_memory_utilization=0.90`. `--kv-cache-dtype` 만 변경. 측정일 2026-04-17.

#### 용량·품질 프로파일

| 모드 | 압축비 | KV tokens | Max conc @ 32K | PPL vs bf16* |
|---|---:|---:|---:|---:|
| `turboquant_3bit_nc` | 4.9x | 75,488 | 3.00x | +20.6% |
| `turboquant_k3v4_nc` | 3.5x | 64,960 | 3.00x | +10.6% |
| `turboquant_4bit_nc` | 3.8x | 57,120 | 2.82x | +2.7% |
| `turboquant_k8v4`    | 2.6x | 38,528 | 2.50x | +1.2% |

*PPL 수치는 upstream `TurboQuantConfig` docstring 기준값.

참고: `k3v4_nc` 는 `4bit_nc` 에 모든 축에서 열세 — 압축비(3.8x > 3.5x)·품질(+2.7% < +10.6%) 둘 다 `4bit_nc` 가 우세. 3-bit 키의 품질 손실이 너무 큼.

#### Prefill 처리량 — `t/s (total)`

| 모드 | pp512 c1 | pp1024 c1 | pp2048 c1 | pp2048 c4 |
|---|---:|---:|---:|---:|
| 3bit_nc | 916.1 | 1,313.4 | 1,673.4 | 1,928.9 |
| k3v4_nc | 898.0 | 1,304.1 | 1,663.2 | 2,013.1 |
| 4bit_nc | 873.8 | 1,300.7 | 1,642.7 | 1,930.8 |
| k8v4    | 901.8 | 1,295.4 | 1,662.7 | 1,931.7 |

전체 표: `benchmarks/llama-benchy/results_397b-int4-tq-*-c1to4.md`

#### Decode 처리량 — tg128 `t/s (total)` / peak

| 모드 | c1 | c2 | c3 | c4 peak |
|---|---:|---:|---:|---:|
| 3bit_nc | 26.7 | 42.1 | 50.1 | 72.0 |
| k3v4_nc | 26.8 | 44.4 | 55.4 | 80.0 |
| 4bit_nc | 26.6 | 44.7 | 55.2 | **84.0** |
| k8v4    | 26.7 | 45.0 | 56.1 | 78.7 |

#### 분석

- **단일 요청(c1) decode 처리량은 모드 무관 동일** (26.6-26.8 t/s). 단일 스트림은 MoE matmul compute-bound 라 KV 메모리 대역폭 영향 없음.
- **고부하(c4)에서 차이 발생**: `4bit_nc` 가 tg128 c4 peak 84 t/s 로 최고 — `3bit_nc` 대비 **+17%**. 4-bit value dequant 의 arithmetic intensity 가 3-bit 보다 효율적.
- **KV 용량과 처리량은 비례하지 않음**: `3bit_nc` 는 `k8v4` 대비 KV 용량 2배지만 peak 처리량은 오히려 낮음. Dequant 비용이 지배적.
- **Prefill 은 모드 무관 거의 평행** (±3%). 이 모델은 prefill 전체 compute 에서 attention R/W 비중이 작음.

#### 한국어 QA 품질 (12 문항, mt=30000, thinking 제거)

각 답변의 사실 정확성 기준 채점 (O=정답, △=부분정답, X=오답). 상세 답변은 `benchmarks/results/*_Qwen3.5-397B-A17B-int4-AutoRound_mt30000_*.txt` 참조.

| 모드 | O | △ | X | Timeout | 정답률 |
|---|---:|---:|---:|---:|---:|
| `3bit_nc` | 7 | 2 | 3 | 0 | **66.7%** |
| `k3v4_nc` | 8 | 3 | 1 | 0 | 79.2% |
| `4bit_nc` | 8 | 3 | 1 | 0 | 79.2% |
| `k8v4`    | 8 | 3 | 0 | 1 | 79.2% (Q6 제외) |

`3bit_nc` 는 논리/자모분해 문항에서 실제 품질 저하 관측 — PPL +20.6% 예측치와 부합. 나머지 3개 모드는 이 벤치에서 구별 불가(12 문항은 +1% vs +10% PPL 차이를 잡기엔 부족). `k8v4` 의 1건 실패는 해마 이모지 답변이 과도하게 길어져 urllib 900초 클라이언트 타임아웃에 걸린 케이스 — vLLM/모델 문제 아님.

#### 권장

**`turboquant_4bit_nc` 가 운영 기본값으로 최적**:
- c4 고부하에서 tg128 peak 최고 (84 t/s)
- 3.8x KV 압축 (bf16 대비 ~2배 동시성 여유)
- PPL +2.7% 만 — 실제 응답에서 체감 불가
- 모든 축에서 `k3v4_nc` 우세

`k8v4` 는 최고 품질이 필수이고 KV 용량이 병목이 아닌 경우에만. `3bit_nc` 는 품질 저하가 실측되므로 회피.

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

homeserver의 `<source_dir>/Qwen/Qwen_Qwen3.6-35B-A3B`에 다운로드되어
있다는 전제입니다. 테스트 대상 Spark 노드(권장: `spark01`, 기존 397B head와
동일)로 복사 후 `MODEL_PATH`를 로컬 경로로 지정해 사용합니다.

```bash
# homeserver에서 (~67 GB, RoCE 링크로 ~6분)
rsync -av <source_dir>/Qwen/Qwen_Qwen3.6-35B-A3B/ \
    spark01:<spark_model_dir>/Qwen/Qwen_Qwen3.6-35B-A3B/

# spark01: 프리셋 복사 + 로컬 경로 치환
ssh spark01 'cd ~/docker/vllm-spark && \
    cp models/qwen3.6-35b-fp16.env .env && \
    sed -i "s|\[model_path\]|<spark_model_dir>/Qwen|" .env'
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

## 이미지 태그와 Git 태그

GHCR 이미지 태그(`ghcr.io/bjk110/vllm-spark:<tag>`)와 Git 태그는 아직 1:1로
정렬되어 있지 않습니다. 현재 Git에 존재하는 태그는 `v018-ngc2603` 한 개뿐이며,
나머지는 GHCR 이미지로만 발행되어 있습니다. 아래 표는 각 이미지 태그가 Git
이력의 어떤 시점에 대응하는지 정리한 것입니다. 특정 이미지를 재현하거나
롤백할 때 참고하세요.

| 이미지 태그 | Git ref (commit) | 스택 | 비고 |
|---|---|---|---|
| `v021-tq` (최신) | `3070f9a` | base + TurboQuant 패치 + Inductor graph partition fix | 모든 `*-tq.env` 프리셋이 사용 |
| `v021-ngc2603` (최신) | `8623187` | vLLM `95995bbe` + FlashInfer `v0.6.9` | TQ 미사용 프리셋 전체가 사용 |
| `v020-ngc2603` (전환기) | `8efdf0b` (base-refresh-20260417 base bump) | vLLM `978a4462` + FlashInfer `v0.6.8` | v021로 대체됨; 재현용으로만 GHCR에 유지 |
| `v019-ngc2603` | `7736716` (Gemma 4 + vLLM 0.19.1 업그레이드) | vLLM `0.19.1` `a7d79fa` + FlashInfer `v0.6.7.post3` | v021로 대체됨 |
| `v018-ngc2603` | `feb5993` (NGC 26.03 source build 시작) — Git 태그 `v018-ngc2603` 존재 | vLLM `0.18.3` `c494977` + FlashInfer `v0.6.7` | 현재 Git에 존재하는 유일한 릴리스 태그 |

### 권장 Git 태그 (생성 가이드)

현재 Git 태그는 `v018-ngc2603` 하나뿐입니다. 메인테이너가 GHCR 이미지 태그와
Git 태그를 일치시키려면 아래 커맨드로 태그를 생성·푸시하세요. 실행 전 반드시
SHA를 검증하세요.

    git tag -a v019-ngc2603 7736716 -m "v019-ngc2603 — Gemma 4 + vLLM 0.19.1"
    git tag -a v020-ngc2603 8efdf0b -m "v020-ngc2603 — base-refresh-20260417 (vLLM 978a4462, FlashInfer 0.6.8)"
    git tag -a v021-ngc2603 8623187 -m "v021-ngc2603 — vLLM 95995bbe + FlashInfer v0.6.9"
    git tag -a v021-tq      3070f9a -m "v021-tq — base + TurboQuant cherry-picks + codegen workaround"
    git push origin v019-ngc2603 v020-ngc2603 v021-ngc2603 v021-tq

**Verify commit before tagging.** 위 4개 SHA는 본 README 갱신 시점의
`git log --oneline` 결과입니다. 이후 `main`이 재정리되었다면 다음 명령으로
경계 커밋을 다시 찾으세요:

    git log --oneline --grep='base.refresh\|bump base.*v021\|0.19.1\|use Inductor graph partition'

## 브랜치 구조

`main`이 유일한 장기 브랜치입니다. 이전의 별도 작업 흐름(베이스 스택 갱신,
TurboQuant 리베이스, 단일 Spark CLUSTER_MODE)은 모두 머지되어 feature
브랜치는 정리된 상태입니다.

### 아카이브된 브랜치 이력

구형 TurboQuant 브랜치는 참고용 태그로 보존되어 있습니다:

- **`archive/feat-turboquant`**

필요하면 다음과 같이 복원할 수 있습니다:

```bash
git checkout -b feat/turboquant archive/feat-turboquant
```

## 라이선스

설정 파일은 참고용으로 제공됩니다. 모델은 해당 라이선스를 따릅니다 ([Qwen 라이선스](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [Gemma 라이선스](https://ai.google.dev/gemma/terms)).
