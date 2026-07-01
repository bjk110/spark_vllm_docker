# 홈 인프라 & LLM 서빙 아키텍처

| Field | Value |
|---|---|
| Status | `Stack reference` |
| Scope | Repository home-infrastructure and distributed-serving architecture |
| Current replacement | None |
| Last validated | Not recorded (living reference; live-state snapshot 2026-05-26) |
| Runtime or image identity | Not applicable (architecture reference) |
| Historical relevance | Living architecture reference. Where a current production example is needed, the authoritative serving path is the SM121 indexer baseline — see [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md). |

> 최종 갱신: 2026-05-26 · 라이브 상태 기준 (spark01 head + spark02 worker, `dsv4-d568-relax` 가동 중)

---

## 1. 하드웨어 & 네트워크 토폴로지

```mermaid
graph TB
    subgraph LAN["LAN — 192.168.0.0/24"]
        HS["<b>homeserver</b> · 192.168.0.8<br/>x86_64 · Ryzen 7 5825U 16c<br/>29 GiB · Ubuntu 24.04<br/><i>중앙관리/서비스</i><br/>traefik · harbor · plex<br/>prometheus · openclaw · portainer"]
        SP1["<b>spark01</b> — vLLM HEAD<br/>aarch64 · GB10 Blackwell sm_121<br/>121 GiB UMA · drv 595.71.05<br/>Ubuntu 24.04"]
        SP2["<b>spark02</b> — vLLM WORKER<br/>aarch64 · GB10 Blackwell sm_121<br/>121 GiB UMA · drv 595.71.05<br/>Ubuntu 24.04"]
    end

    HS --- SP1
    HS --- SP2

    SP1 <==>|"RoCE v2 / RDMA · 200 Gbps<br/>10.10.10.0/24 · enp1s0f0np0<br/>HCA rocep1s0f0 · NCCL_IB_DISABLE=0"| SP2

    SP1 -. "10.10.10.1" .- SP1
    SP2 -. "10.10.10.2" .- SP2

    classDef server fill:#1f2937,stroke:#60a5fa,stroke-width:2px,color:#e5e7eb;
    classDef gpu fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#e5e7eb;
    class HS server;
    class SP1,SP2 gpu;
```

| 호스트 | Arch | CPU/GPU | RAM | LAN IP | RoCE IP | 역할 |
|---|---|---|---|---|---|---|
| `homeserver` | x86_64 | Ryzen 7 5825U (16c) | 29 GiB | 192.168.0.8 | — | 중앙 관리·서비스 (**빌드 금지**) |
| `spark01` | aarch64 | GB10 (sm_121) | 121 GiB UMA | LAN | **10.10.10.1** | vLLM **head** + Ray GCS |
| `spark02` | aarch64 | GB10 (sm_121) | 121 GiB UMA | LAN | **10.10.10.2** | vLLM **worker** |

> GB10은 **UMA**(host RAM + GPU 메모리 통합 121.63 GiB) — 별도 VRAM 풀 없음. 컨테이너 stop 시 드라이버에 RAM 고착, **reboot가 유일한 클린 회수책**.
> **빌드 규칙**: 모든 Docker/vLLM 빌드는 spark01 또는 spark02에서만. homeserver(29 GiB)는 OOM cascade로 freeze 발생 → 절대 금지.

---

## 2. LLM 서빙 컨테이너 아키텍처 (dual-RDMA TP=2)

```mermaid
graph TB
    CLI(["클라이언트"]) -->|"HTTP · OpenAI-compatible<br/>:8000 · deepseek-v4-flash"| API

    subgraph ENGINE["vLLM TP=2 엔진 · backend=Ray"]
        direction LR
        subgraph N1["spark01 (HEAD)"]
            API["container: <b>vllm-spark-head</b><br/>image: ...:dsv4-d568-relax<br/>network_mode: host · ipc: host<br/>profiles:[head] · restart:no<br/>—<br/>Ray GCS + API server<br/>TP rank 0 → GB10 #0"]
            M1[("/models/DeepSeek-V4-Flash<br/>FP8 safetensors ~149GB")]
        end
        subgraph N2["spark02 (WORKER)"]
            W["container: <b>vllm-spark-worker</b><br/>image: ...:dsv4-d568-relax<br/>network_mode: host · ipc: host<br/>profiles:[worker] · restart:no<br/>—<br/>Ray worker node<br/>TP rank 1 → GB10 #1"]
            M2[("/models/DeepSeek-V4-Flash<br/>FP8 safetensors ~149GB")]
        end
    end

    API <==>|"Ray GCS :6379<br/>+ NCCL all-reduce<br/>over RoCE 10.10.10.0/24"| W
    M1 -.-> API
    M2 -.-> W

    classDef head fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#e5e7eb;
    classDef worker fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#e5e7eb;
    classDef model fill:#374151,stroke:#9ca3af,stroke-width:1px,color:#e5e7eb;
    class API head;
    class W worker;
    class M1,M2 model;
```

### 모델 & 분산 설정

| 항목 | 값 |
|---|---|
| 모델 | **DeepSeek-V4-Flash** (FP8, ~149 GB), served name `deepseek-v4-flash` |
| 병렬 | **TP=2** dual-node, backend = **Ray** (GCS port 6379 over RoCE) |
| 컨텍스트 | `MAX_MODEL_LEN=200000` (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) |
| 배치 | `MAX_NUM_SEQS=8`, `MAX_NUM_BATCHED_TOKENS=8192` |
| 메모리 | `GPU_MEMORY_UTILIZATION=0.85`, `VLLM_SKIP_INIT_MEMORY_CHECK=1` (relax patch) |
| KV cache | `--kv-cache-dtype fp8`, `--block-size 256` |
| MoE | `--enable-expert-parallel`, `VLLM_TRITON_MLA_SPARSE=1` |
| Spec decode | **MTP** `deepseek_mtp`, `num_speculative_tokens=2` |
| CUDA graph | `cudagraph_mode=FULL_AND_PIECEWISE`, `custom_ops=["all"]` |
| 추론/툴 | `deepseek_v4` reasoning + tool-call parser, `<think>` 추론 활성 |
| Arch | `TORCH_CUDA_ARCH_LIST=12.1a` |

---

## 3. 이미지 빌드 스택

```mermaid
graph LR
    NGC["NGC 26.04<br/>PyTorch 2.12.0a0<br/>FlashInfer 0.6.11.post3"]
      --> BASE["dockerfiles/active/Dockerfile.v022-d568<br/>→ <b>vllm-spark:v022-d568</b>"]
    BASE --> DSV4["dockerfiles/active/Dockerfile.dsv4-d568<br/>STAGE1: jasl@edc82b614f51 source wheel<br/>STAGE2: wheel --no-deps + 3 patches"]
    DSV4 --> PROD["<b>:dsv4-d568-relax</b> ✅ 운영"]
    DSV4 -.-> BAD[":dsv4-d568-5d64798 ⚠️<br/>runtime thrash · 미참조 보존"]

    classDef ok fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#e5e7eb;
    classDef bad fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#e5e7eb;
    classDef norm fill:#1f2937,stroke:#9ca3af,stroke-width:1px,color:#e5e7eb;
    class PROD ok;
    class BAD bad;
    class NGC,BASE,DSV4 norm;
```

### 적용 패치 (STAGE 2, 3건)

1. **relax-profile-assertion** (`gpu_worker.py`) — post-profile `determine_available_memory()` assertion 우회
2. **G8 skip-init** (`utils.py`) — pre-init `request_memory()` free<requested 체크 우회
3. **envs-register** (`envs.py`) — `VLLM_USE_SPINLOOP_EXT` 등록

> `VLLM_SKIP_INIT_MEMORY_CHECK=1` 가 위 (1)·(2) assertion을 모두 우회 → UMA 환경의 init+profile memcheck 통과.

---

## 부록: 운영 노트

- **현재 운영 이미지**: `ghcr.io/bjk110/vllm-spark:dsv4-d568-relax` (jasl@edc82b614f51, 2026-05-19).
- **bump 보류**: `5d64798` (HEAD, 2026-05-25)는 빌드는 성공했으나 dual-GB10 load-time thrash로 양 노드 잠김 → power-cycle 후 relax 롤백. 원인 격리 전 재투입 금지.
- **clean shutdown**: `docker compose --profile head down` / `--profile worker down` (profiles 때문에 `--profile` 없으면 silently skip). stop 후 UMA RAM 회수 위해 reboot 필요.
