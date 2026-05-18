# Qwen3.6-27B PrismaSCOUT NVFP4-BF16 — Dual-Spark TP=2 가이드

`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` 모델을 DGX Spark 두
대(`spark01`, `spark02`) 위에서 RDMA + Ray + TP=2 로 서빙하는 절차.

## 0. 구성 요약

| 항목 | 값 |
|---|---|
| 모델 | `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| 양자화 | compressed-tensors (NVFP4 + BF16 sidecars) |
| TP | 2 |
| Speculative (기본) | **MTP `n=3`** |
| Speculative (실험) | DFlash `z-lab/Qwen3.6-27B-DFlash` `k=8` ※ 본 문서 §6 |
| KV cache dtype | fp8 |
| `MAX_MODEL_LEN` | 32768 |
| GPU mem util | 0.90 |
| NVFP4 GEMM 백엔드 | `flashinfer-cutlass` |
| Compose 파일 | `docker-compose.qwen36-prismascout-tp2.yml` |
| Env 프리셋 | `models/qwen3.6-27b-prismascout-nvfp4-tp2.env` |
| Compose project | `qwen36-prismascout-tp2` |
| 컨테이너 | `vllm-qwen36-head` (spark01) / `vllm-qwen36-worker` (spark02) |

기존 `docker-compose.yml` (서비스 이름: `vllm-spark-head` / `vllm-spark-worker`,
project 이름 미지정) 과 컨테이너 이름·project 이름이 모두 다르므로 충돌 없이
공존할 수 있다.

## 1. 호스트 / 네트워크

| 호스트 | 관리 LAN | RDMA | 역할 |
|---|---|---|---|
| homeserver | `192.168.0.8` | — | 모델 다운로드 + 배포 origin |
| spark01 | `192.168.0.200` | `10.10.10.1` (`enp1s0f0np0`, HCA `rocep1s0f0`) | Ray head + vLLM API |
| spark02 | `192.168.0.201` | `10.10.10.2` (`enp1s0f0np0`, HCA `rocep1s0f0`) | Ray worker |

## 2. 경로

| 위치 | 경로 |
|---|---|
| homeserver | `/mnt/data/llm-models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| spark01 / spark02 host | `/home/bjk110/Documents/Models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| 컨테이너 내부 | `/models/google/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |

호스트의 `/home/bjk110/Documents/Models` 를 컨테이너의 `/models` 에 read-only
바인드 마운트. `google/` 하위 폴더는 사용자의 저장 규칙.

## 3. 다운로드 (homeserver)

```bash
cd /home/bjk110/docker/vllm-spark
# (gated/private 모델인 경우) export HF_TOKEN=hf_...
bash scripts/download_qwen36_prismascout.sh
```

- `huggingface-cli` 또는 `hf` CLI 사용 (PATH 에서 자동 선택)
- 실제 파일 다운로드 (`--local-dir-use-symlinks False` — CLI 버전에 따라
  flag 가 없으면 기본값이 실파일이므로 생략)
- 중단 후 재실행 시 자동 resume

## 4. 배포 (homeserver → spark01 → spark02)

```bash
bash scripts/sync_qwen36_prismascout_to_sparks.sh
```

- 1단계: homeserver → spark01, 관리 LAN `192.168.0.200` 경유
- 2단계: spark01 → spark02, **RDMA `10.10.10.2`** 경유 (spark01 위에서
  ssh 로 두 번째 rsync 를 띄워 bytes 가 RDMA 링크로 흐르도록 함)
- rsync flag: `-avh --info=progress2 --partial --append-verify`
- SSH 키 인증이 (a) homeserver→spark01, (b) spark01→spark02 RDMA 양쪽에서
  미리 설정되어 있어야 한다.

## 5. 실행

### 5.1. orchestrate 한 번에 (homeserver 또는 어디서든)

```bash
# 같은 repo가 양쪽 spark에 동일 경로로 checkout 되어 있다는 전제.
HOST=orchestrate bash scripts/start_qwen36_prismascout_tp2.sh
```

- 내부적으로 worker → head 순으로 SSH 띄움
- `entrypoint.sh` 가 head 쪽에서 Ray 조인 대기 후 `vllm serve` 기동

### 5.2. 노드별로 직접

```bash
# spark01 (head)
cd /home/bjk110/docker/vllm-spark
bash scripts/start_qwen36_prismascout_tp2.sh   # auto: hostname=spark01 → head

# spark02 (worker)
cd /home/bjk110/docker/vllm-spark
bash scripts/start_qwen36_prismascout_tp2.sh   # auto: hostname=spark02 → worker
```

내부적으로 실행되는 명령:

```bash
docker compose \
  -p qwen36-prismascout-tp2 \
  -f docker-compose.qwen36-prismascout-tp2.yml \
  --env-file models/qwen3.6-27b-prismascout-nvfp4-tp2.env \
  --profile head up -d         # spark01
  --profile worker up -d       # spark02
```

### 5.3. 정지

```bash
HOST=orchestrate bash scripts/stop_qwen36_prismascout_tp2.sh
```

### 5.4. 로그 / health

```bash
# 각 노드에서
bash scripts/logs_qwen36_prismascout_tp2.sh           # docker logs -f 로컬

# 한 번만 health 확인
bash scripts/logs_qwen36_prismascout_tp2.sh --health

# 또는 직접
curl http://192.168.0.200:8000/health
curl http://192.168.0.200:8000/v1/models
```

엔진 init (NVFP4 GEMM 빌드 + CUDA graph capture + MTP 헤드 로딩) 까지
대략 **5~10분**. `/v1/models` 가 200 으로 응답하기 전까지는 정상적인
워밍업으로 간주.

## 6. MTP 기본 vs DFlash 실험 옵션

이번 1차 구성의 **기본값은 MTP `n=3`** 입니다. 안정성/회귀 확인을 먼저 마친
후에 DFlash 로 교체 비교합니다.

| 모드 | speculative-config | 비고 |
|---|---|---|
| MTP n=3 (기본) | `'{"method":"mtp","num_speculative_tokens":3}'` | PrismaSCOUT 체크포인트에 MTP 헤드 포함 |
| DFlash k=8 (실험) | `'{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":8}'` | vLLM 버전/PR 의존성 있음. 본 preset 기본값에서 제외 |
| DFlash k=4 / k=12 / k=15 (실험) | 동일, `num_speculative_tokens` 만 교체 | acceptance rate vs latency trade-off 측정용 |

DFlash 로 전환하려면 `models/qwen3.6-27b-prismascout-nvfp4-tp2.env` 의
`VLLM_EXTRA_ARGS` 라인에서 `--speculative-config ...` 부분만 교체:

```bash
# DFlash k=8 예시 (환경변수에 직접 export 해도 compose 가 우선 적용)
sed -i \
  's|--speculative-config {"method":"mtp"[^}]*}|--speculative-config {"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":8}|' \
  models/qwen3.6-27b-prismascout-nvfp4-tp2.env
```

DFlash 모델 본체는 별도 다운로드/배포가 필요. 본 가이드의 download/sync 스크립트는
PrismaSCOUT 메인 체크포인트만 처리한다.

## 7. llama-benchy 템플릿

[llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4 기준. homeserver
또는 spark01 에서 실행. `--model` 은 vLLM 이 노출하는 `served-model-name`
(`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`) 또는 모델
컨테이너 경로 둘 다 동작.

```bash
BASE=http://192.168.0.200:8000/v1
MODEL=rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm

# 핵심 3 조합 × concurrency 1/2/4
for c in 1 2 4; do
  for pp_tg in "128 32" "512 128" "2048 128"; do
    read -r pp tg <<< "$pp_tg"
    llama-benchy \
      --base-url "$BASE" --model "$MODEL" \
      --concurrency "$c" \
      --pp "$pp" --tg "$tg" \
      --runs 3
  done
done
```

결과 표 템플릿 (MTP n=3 결과를 채우고, 추후 DFlash 행 추가):

| pp / tg | c=1 | c=2 | c=4 |
|---|---|---|---|
| 128 / 32  (MTP n=3) | | | |
| 512 / 128 (MTP n=3) | | | |
| 2048 / 128 (MTP n=3) | | | |
| 128 / 32  (DFlash k=4) | | | |
| 128 / 32  (DFlash k=8) | | | |
| 128 / 32  (DFlash k=12) | | | |
| 128 / 32  (DFlash k=15) | | | |

## 8. 문제 발생 시 확인 순서

`docker logs vllm-qwen36-head` 에서 아래 단계를 순서대로 확인:

1. **NCCL / Ray 조인** — `[entrypoint] CLUSTER_MODE=dual-rdma: head=10.10.10.1
   worker=10.10.10.2 ...` 와 `All 2 nodes joined!` 메시지가 5~30초 안에 떠야 함.
   안 뜨면 spark02 worker 컨테이너가 죽었거나 RDMA 라우팅 문제.
2. **compressed-tensors 로딩** — `Loading model weights ...` 단계에서 NVFP4
   sidecar 가 모두 발견되는지. `KeyError: '...weight_scale'` → 양자화 메타파일
   누락 또는 체크포인트 불완전.
3. **NVFP4 backend** — `FlashInfer CUTLASS NVFP4 GEMM` 또는 동등한 백엔드 로그.
   `--quantization compressed-tensors` + `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`
   가 둘 다 인식되었는지 확인.
4. **MTP tensor 로딩** — `Loading speculator weights...` / `mtp` 키워드.
   tensor 누락 시 MTP 비활성화하고 다시 시도.
5. **KV cache dtype** — `KV cache dtype: fp8`. 다른 값이면 `--kv-cache-dtype fp8`
   가 누락됐거나 모델 config 가 강제 override 한 상황.
6. **OOM** — `out of memory ... in profiling stage` → `GPU_MEMORY_UTILIZATION`
   을 `0.85` 또는 `0.80` 으로 낮추기. GB10 unified memory 121 GiB 공유 풀이므로
   OS buff/cache 가 많이 잡혀 있으면 일시적으로 OOM 가능.
7. **RDMA interface 실사용** — `NCCL_DEBUG=INFO` 기본값이므로 NCCL 이 선택한
   transport 가 head 로그에 출력됨. IB/RoCE 가 아닌 SHM/TCP 으로 빠진 경우
   `IB_HCA_NAME=rocep1s0f0`, `ROCE_IF_NAME=enp1s0f0np0` 가 실제 디바이스명과
   일치하는지 양 노드에서 `ibstat` / `ip a` 로 재확인.

## 9. 본 구성이 기존 setup 에 영향 없음

- 기존 `docker-compose.yml` 의 services (`vllm-spark-head`, `vllm-spark-worker`),
  shared `.env` 흐름, 다른 preset (`qwen3.5-397b-int4*`, `qwen3.5-122b-*-tq` 등)
  모두 그대로 유지.
- 본 구성은 별도 compose 파일 + 별도 project name (`qwen36-prismascout-tp2`)
  + 별도 컨테이너 이름 (`vllm-qwen36-*`) + 별도 cache 디렉토리
  (`./.cache/vllm-qwen36`) 사용 → 동시 실행도 가능 (단, GPU 메모리 121 GiB
  공유 풀이므로 두 모델 동시 서빙은 메모리 한도 안에서만).
