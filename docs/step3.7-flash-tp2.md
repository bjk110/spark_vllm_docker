# stepfun-ai/Step-3.7-Flash — Dual-Spark TP=2 가이드 (FP8 / NVFP4)

`stepfun-ai/Step-3.7-Flash` (198B-param sparse MoE VLM, 196B LM + 1.8B vision
encoder, ~11B active params/token, 288 experts top-8) 를 DGX Spark 두 대
(`spark01`=head, `spark02`=worker) 위에서 TP=2 로 서빙하는 절차와 두 양자화
변형(FP8, NVFP4)의 설정/벤치마크 비교.

두 변형 모두 같은 이미지 `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7`
(NGC 26.05 + vLLM 0.22.1 + `patches/patch_step3p7_nvfp4_input_scale.py`) 사용.
패치는 NVFP4 전용 코드 경로(`step3p5.py` `expert_params_mapping` +
`fused_moe/layer.py` dual-shard write)에만 영향을 주므로, FP8 변형은 같은
이미지를 **재빌드 없이** 그대로 사용 가능 (FP8은 vLLM 표준 `fp8.py` 경로).

## 0. 구성 요약

| 항목 | FP8 | NVFP4 |
|---|---|---|
| Preset | [`presets/step37-flash-fp8-tp2.env`](../presets/step37-flash-fp8-tp2.env) | [`presets/step37-flash-nvfp4-tp2.env`](../presets/step37-flash-nvfp4-tp2.env) |
| 양자화 | FP8 block (e4m3, 128×128, dynamic activation — DeepSeek-V3 style) | NVFP4 (modelopt) + FP8 KV cache |
| 가중치 크기 | ~97.2-97.3 GB/GPU (TP=2) | ~50 GB/GPU (TP=2) |
| `MAX_MODEL_LEN` | 32,768 | 8,192 (preset 기본값; 세션 벤치마크는 더 큰 값으로 운영, §4 참고) |
| `MAX_NUM_SEQS` | **1** (메모리 한계, §3 참고) | 4 |
| `GPU_MEMORY_UTILIZATION` | 0.87 | 0.88 |
| `MAX_NUM_BATCHED_TOKENS` | 2,048 | 8,192 |
| `--enforce-eager` | 필요 (메모리 마진 확보) | 필요 |
| `--quantization` | (기본 fp8.py 경로, 명시 불요) | `modelopt` |
| `--kv-cache-dtype` | (기본 auto) | `fp8` |
| `--enable-expert-parallel` | 미사용 | 사용 (288 experts / 2 ranks) |
| 검증일 | 2026-06-10 | 2026-06-10 |

공통 인자: `--trust-remote-code --reasoning-parser step3p5
--enable-auto-tool-choice --tool-call-parser step3p5`.

## 1. 서빙 방법

### 1.1. Preset 적용

homeserver(canonical git 작업 위치)에서:

```bash
cd /home/bjk110/docker/vllm-spark
cp presets/step37-flash-fp8-tp2.env .env      # 또는 step37-flash-nvfp4-tp2.env
```

`.env`의 `MODEL_PATH`를 실제 모델 경로로 수정:

```
MODEL_PATH=/home/bjk110/Documents/Models/stepfun-ai/Step-3.7-Flash-FP8
```

### 1.2. 양 노드 동기화

`.env`는 spark01/spark02 양쪽 working tree에 직접 동기화 (양쪽 모두 적용해야 head가 신 설정으로 기동):

```bash
scp .env spark01:/home/bjk110/docker/vllm-spark/.env
scp .env spark02:/home/bjk110/docker/vllm-spark/.env
```

모델 가중치도 양 노드에 전체 복사 필요 (TP=2라도 각 노드가 자기 shard를
디스크에서 직접 읽음).

### 1.3. 기동 순서

worker(spark02) 먼저, head(spark01) 나중 — Ray rendezvous 순서:

```bash
# spark02 (worker)
ssh spark02 "cd /home/bjk110/docker/vllm-spark && docker compose --profile worker up -d"

sleep 15

# spark01 (head)
ssh spark01 "cd /home/bjk110/docker/vllm-spark && docker compose --profile head up -d"
```

`entrypoint.sh`가 head에서 `ray start --head` → worker join 대기 →
`vllm serve --distributed-executor-backend ray` 진행. 총 부팅 시간은
가중치 크기에 비례 (FP8 ~6분, NVFP4 ~3-4분 추정).

### 1.4. 부팅 검증

```bash
ssh spark01 "docker logs --tail 100 vllm-spark-head" | grep -E "GPU KV cache size|Maximum concurrency|Application startup complete"
```

NaN 여부 확인 (`logprobs`가 finite한지):

```bash
curl -s http://192.168.0.200:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stepfun-ai/Step-3.7-Flash-FP8","prompt":"The capital of France is","max_tokens":20,"temperature":0,"logprobs":1}'
```

## 2. GB10 UMA 메모리 함정 (양 변형 공통)

- `nvidia-smi`는 GB10 unified memory에서 `Memory-Usage: Not Supported` 표시 —
  실제 가용량은 vLLM pre-init 로그(`Free memory on device cuda:0 (X/121.63 GiB)`)
  또는 `free -h`로 확인.
- 컨테이너가 크래시 후 `docker compose down` 해도 GPU 메모리(~97-110GiB)가
  드라이버에 잔류 — `systemctl reboot`만이 확실한 회수 방법(~10-12분 소요).
  재부팅 시 `pkill` 과 한 ssh 명령으로 묶지 말 것 (자기 셸이 죽어 reboot이
  무산될 수 있음) — 순수 `ssh <host> "sudo systemctl reboot"`만 사용.
- vLLM 0.22.1 (이 이미지엔 `VLLM_SKIP_INIT_MEMORY_CHECK` 패치 없음) 의 2단계
  메모리 체크:
  1. **Pre-init** (`request_memory()`): `util * 121.63 > 시작 시점 free` 면
     `ValueError: Free memory on device cuda:0 (.../121.63 GiB) on startup is
     less than desired GPU memory utilization` 으로 즉시 실패.
  2. **Post-profile** (`_check_enough_kv_cache_memory`): `Available KV cache
     memory = util*total - (weights+activation)` 가 음수면 `ValueError: No
     available memory for the cache blocks.`

## 3. FP8: MAX_NUM_SEQS=1 인 이유

FP8 가중치(~97GB/GPU)는 idle free 한도(106.41/121.63 GiB) 대비 여유가 거의
없음. 시도 이력:

| 설정 | rank0 (spark01) | rank1 (spark02, RDMA worker) | 결과 |
|---|---:|---:|---|
| util=0.85, batched=16384, seqs=4 | -1.2 GiB | -10.83 GiB | **실패** (post-profile) |
| util=0.87, batched=2048, seqs=1 | +6.6 GiB | +2.27 GiB | **성공** |

- `rank1`(spark02, cross-node RDMA Ray worker)이 `rank0`보다 일관되게
  4-10GiB 더 큰 activation/comm 오버헤드를 가짐 — 추정 원인은 NCCL/RDMA
  cross-node 통신 버퍼 비대칭 (미확정).
- `MAX_NUM_BATCHED_TOKENS`/`MAX_NUM_SEQS`를 줄이면 profiling 시점의 더미
  배치(activation 메모리)가 줄어 "Available KV cache memory"가 직접
  증가함 — 이 레버로 두 rank 모두 양수 전환.
- **MAX_NUM_SEQS=1의 의미**: KV cache 80,308 tokens / "max concurrency
  2.45x @ 32768"은 메모리 여유의 산출값일 뿐, `MAX_NUM_SEQS=1`이 실제 동시
  요청 처리를 1개로 강제 — 동시성 목적으로는 이 2.45x를 활용할 수 없음
  (긴 단일 시퀀스용 여유로만 의미 있음).
- **결론**: GB10 121GiB UMA에서 FP8 TP=2는 동시성 확보가 구조적으로 빠듯함.
  동시성이 중요하면 NVFP4(~50GB/GPU, MAX_NUM_SEQS=4 검증됨)가 더 나은 선택.

## 4. NVFP4 NaN 버그 (해결됨)

NVFP4 변형은 2026-06-10 이전 모든 출력이 NaN logits로 깨지는 버그가 있었음.

- **원인**: 체크포인트의 NVFP4 per-expert input scale
  (`.moe.{gate,up,down}_proj.input_scale`, shape `[288]`)이
  `Step3p5Model.load_weights()`의 `expert_params_mapping`에 매핑되지 않아
  `w13_input_scale`/`w2_input_scale`이 `torch.empty()` garbage로 남음 →
  NVFP4 MoE GEMM 전체 NaN. MoE backend(CUTLASS/Marlin) 선택과는 무관.
- **수정**: `patches/patch_step3p7_nvfp4_input_scale.py` (2-file):
  1. `step3p5.py`의 `expert_params_mapping`에 `.input_scale` 매핑 3종 추가
  2. `fused_moe/layer.py`의 input scale 로드에 ModelOpt dual-shard 분기
     (`w13_input_scale`이 `[num_experts, 2]` shape — w1/w3 각각 슬롯에 기록)
- **검증**: `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7` 재빌드 후
  `/v1/completions` → `"Paris."` + finite token_logprobs (-0.03~-1.9 범위).
- 비슷한 NVFP4 ModelOpt MoE 모델(old packed 3D checkpoint format) 추가 시
  `expert_params_mapping`의 `.input_scale` 누락 여부 우선 점검.

## 5. 벤치마크 결과

도구: [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.7,
엔드포인트 `http://192.168.0.200:8000/v1`, depth 스윕 포맷 (pp2048, tg32,
runs=3, latency-mode=generation — DSV4/Step-3.7 벤치마크 표준 포맷).

### 5.1. FP8 (MAX_MODEL_LEN=32768, MAX_NUM_SEQS=1, c=1)

depth는 `pp2048 + depth + tg32 ≤ 32768` 제약으로 d0/4096/8192/16384/28672
(NVFP4의 d32768/d65536은 32768 한도 초과로 제외).

결과 파일: [`benchmarks/llama-benchy/results_step37-flash-fp8-tp2-DEPTH.md`](../benchmarks/llama-benchy/results_step37-flash-fp8-tp2-DEPTH.md)

| depth | pp2048 t/s | tg32 t/s | peak tg t/s |
|---|---:|---:|---:|
| 0 | 1084.4 ± 51.7 | 13.32 ± 0.12 | 14.00 |
| 4096 | 1099.2 ± 5.7 | 13.12 ± 0.05 | 14.00 |
| 8192 | 1055.4 ± 11.6 | 13.12 ± 0.12 | 14.00 |
| 16384 | 1053.8 ± 1.0 | 13.18 ± 0.05 | 14.00 |
| 28672 | 1031.9 ± 0.3 | 13.25 ± 0.01 | 14.00 |

- prefill: ~1030-1100 t/s, depth 증가에 따라 완만히 감소 (d28672에서도 -5%)
- decode: ~13.1-13.3 t/s로 전 구간 평탄, peak 14.0 t/s 동일

### 5.2. NVFP4 (c=1, depth 스윕)

결과 파일: [`benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-DEPTH.md`](../benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-DEPTH.md)

| depth | pp2048 t/s | tg32 t/s | peak tg t/s |
|---|---:|---:|---:|
| 0 | 1251.42 ± 3.22 | 13.35 ± 0.71 | 14.00 ± 0.82 |
| 4096 | 1299.69 ± 1.11 | 12.84 ± 0.34 | 14.00 ± 0.82 |
| 8192 | 1289.83 ± 3.11 | 11.90 ± 0.20 | 12.67 ± 0.47 |
| 16384 | 1267.43 ± 1.14 | 12.11 ± 0.36 | 12.67 ± 0.47 |
| 32768 | 1235.27 ± 16.01 | 12.37 ± 0.55 | 13.33 ± 0.47 |
| 65536 | 1148.67 ± 1.10 | 12.03 ± 0.09 | 13.00 ± 0.00 |

### 5.3. NVFP4 동시성 스윕 (c=1/2/4)

결과 파일: [`benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-c1to4.md`](../benchmarks/llama-benchy/results_step37-flash-nvfp4-tp2-c1to4.md)

| concurrency | pp2048 t/s (total) | tg32 t/s (total) | peak tg t/s |
|---|---:|---:|---:|
| c1 | 1247.40 ± 5.60 | 13.23 ± 0.27 | 14.33 ± 0.47 |
| c2 | 1196.81 ± 42.81 | 22.85 ± 5.05 | 28.00 ± 0.00 |
| c4 | 1230.07 ± 2.61 | 21.18 ± 0.28 | 45.33 ± 1.89 |

`MAX_NUM_SEQS=4`로 c=4까지 동시 처리 확인 (FP8은 `MAX_NUM_SEQS=1`이라 동시성
스윕 불가, §3 참고).

### 5.4. FP8 vs NVFP4 비교 (c=1, d0 기준)

| 지표 | FP8 | NVFP4 | 비고 |
|---|---:|---:|---|
| prefill (pp2048) | 1084.4 | 1251.4 | NVFP4가 +15% |
| decode (tg32) | 13.32 | 13.35 | 거의 동급 |
| 최대 동시성 | 1 (구조적 한계) | 4 (검증됨) | NVFP4 우위 |
| 가중치/GPU | ~97GB | ~50GB | NVFP4가 메모리 여유 큼 |

c=1 단일 스트림 기준으로는 두 변형의 성능 차이가 크지 않음. 동시 사용자
처리(c≥2)가 필요하면 NVFP4가 명확히 유리.

## 6. 운영 권장

| 시나리오 | 권장 |
|---|---|
| 단일 사용자, 긴 컨텍스트(최대 32K) 우선 | FP8 (`MAX_NUM_SEQS=1`, KV pool 80,308 tokens) |
| 다중 동시 사용자 (2-4) | **NVFP4** (`MAX_NUM_SEQS=4`, KV cache fp8) |
| 메모리 여유 확보 우선 | NVFP4 (~50GB/GPU vs FP8 ~97GB/GPU) |

## 7. 참고

- [`presets/step37-flash-fp8-tp2.env`](../presets/step37-flash-fp8-tp2.env)
- [`presets/step37-flash-nvfp4-tp2.env`](../presets/step37-flash-nvfp4-tp2.env)
- [`patches/patch_step3p7_nvfp4_input_scale.py`](../patches/patch_step3p7_nvfp4_input_scale.py)
- [`docs/dsv4-flash-tp2.md`](dsv4-flash-tp2.md) — 동일 GB10 UMA 메모리 함정/대응 패턴 (DeepSeek-V4-Flash)
