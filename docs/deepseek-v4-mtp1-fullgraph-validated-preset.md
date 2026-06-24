# DeepSeek-V4-Flash MTP n=1 + FULL_DECODE_ONLY — Validated Preset Provenance

Immutable provenance and validation record for the DeepSeek-V4-Flash MTP n=1 +
FULL_DECODE_ONLY decode-graph configuration on dual DGX Spark (GB10 / SM12.1).

**Status: `VALIDATED_PRESET_CANDIDATE` — NOT a `PRODUCTION_BASELINE`.** Promotion to the
normal-operation baseline requires a separate, explicit user authorization. "Validated"
here means safety + performance + long-soak + independent cold-start reproduction all
PASSED; it is **not** a synonym for "production-ready".

## 1. Immutable identity

| Field | Value |
|---|---|
| Model ID | `deepseek-ai/DeepSeek-V4-Flash` (official FP8) |
| Architecture | `DeepseekV4ForCausalLM` (MoE + sparse MLA + MTP heads) |
| Checkpoint | 46 shards, ~148.7 GiB total |
| Image manifest digest | `ghcr.io/bjk110/vllm-spark@sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44` |
| Image config ID | `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105` |
| Local image tag | `vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019` |
| Pinned source | `72261a7af149fa5d3fe2ed2b9956e92590731012` (vLLM PR #41834) |
| vLLM build string | `v0.24.0.dev0+dsv4.pr41834.72261a7` |
| Repository baseline | `main@546c1d8827e14d126e05e88657c503d8b3bd651f` |

## 2. Presets and SHA-256

**Operational (repository-tracked) presets — the promotion package:**

| Role | Path | SHA-256 |
|---|---|---|
| Validated preset (primary) | `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env` | `d116f132ca0087a6773ba9769134c46ace3e707fbe313456cca6ab34b46969b1` |
| Rollback L1 — graph-only (NET/IB) | `presets/deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env` | `87ad3920b3e681ef48e3d4a183f2508157e3fc7a36719320f2f6a50c0b98a903` |
| Rollback L2 — eager U0-RDMA (NET/IB) | `presets/deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env` | `bffea15863867f5e7d4dfe698b761167e25033cd4a4846f2e9ee32b6c623a71b` |

**Tested source artifacts (disposable / untracked — NOT operational rollback targets):**

| Role | Path | SHA-256 |
|---|---|---|
| Tested MTP+graph candidate (source of primary) | `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-safety-tp2.env` | `f0bb73814dd600c18393599fee6ce40181de737957af6c11f9316117266db88a` |
| Tested graph-only safety (source of L1) | `presets/deepseek-v4-v023-stack-pr41834-fullgraph-safety-f2-tp2.env` | `61468aeff7faf8f4acf30f764d3d6a09ca0d7affd371d0fb83845a111b4896b6` |

Each operational preset's runtime (env) section is **byte-identical** to its tested
source artifact (normalized runtime diff empty; only header comments differ). The
disposable source presets are recorded ONLY as validation provenance — operators must
use the repository-tracked operational presets above, never the disposable sources.

## 3. Node topology and transport

| Node | Role | Management IP | RoCE IP |
|---|---|---|---|
| spark01 | head / rank 0 | 192.168.0.200 | 10.10.10.1 |
| spark02 | worker / rank 1 | 192.168.0.201 | 10.10.10.2 |

- TP=2 multiprocessing (mp), one rank per physical node, EP off, Ray off.
- Transport: native NET/IB over RoCE. HCA `rocep1s0f0`, port 1, GID index 3, no
  NET/Socket fallback (`NCCL_NET` empty, `NCCL_IB_DISABLE=0`, `NCCL_NVLS_ENABLE=0`).
- **Single-node TP=1 is impossible** for the official 46-shard checkpoint on one GB10.

## 4. Runtime arguments (validated)

`vllm serve` effective non-default args: `--tensor-parallel-size 2
--distributed-executor-backend mp --nnodes 2 --node-rank {0,1} --master-addr 10.10.10.1
--master-port 29500 --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.87
--max-num-batched-tokens 2048 --no-enable-prefix-caching --kv-cache-memory-bytes
2147483648 --kv-cache-dtype fp8 --skip-mm-profiling --speculative-config
{"method":"deepseek_mtp","num_speculative_tokens":1} --compilation-config
{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2],"cudagraph_num_of_warmups":2}`.

Key semantics: `deepseek_mtp` is normalized to method `mtp` (drafter = EagleProposer
loading DeepSeekV4MTPModel); `uniform_decode_query_len = 1 + 1 = 2`; capture size MUST be
`[2]` (`[1]` is hard-rejected at engine init); exactly ONE target/verify FULL decode
graph captured per rank; draft (MTP head) path is EAGER; rejection sampling runs outside
the captured graph. `VLLM_USE_BREAKABLE_CUDAGRAPH=0`; init-memory-check bypass disabled.

## 5. Validation evidence

| Gate | Result | Date | Artifacts (`benchmarks/results/`) |
|---|---|---|---|
| Safety + rejection | `MTP1_FULLGRAPH_SAFETY_PASS` | 2026-06-23 | `dsv4-mtp1-fullgraph-safety-20260623-121435/` |
| Combined performance | `MTP1_FULLGRAPH_PERF_MAJOR_GAIN` | 2026-06-23 | `dsv4-mtp1-fullgraph-perf-20260623-125104/` |
| 4-hour long soak | `MTP1_FULLGRAPH_LONG_SOAK_PASS` | 2026-06-23/24 | `dsv4-mtp1-fullgraph-longsoak-20260623-133120/` |
| Independent cold-start | `MTP1_FULLGRAPH_COLD_REPRO_PASS` | 2026-06-24 | `dsv4-mtp1-fullgraph-coldrepro-20260624/` |

API-visible decode (llama-benchy 0.3.7 generation, pp2048 tg128 runs3 c1):

| Run | d0 (t/s) | d4096 (t/s) |
|---|---|---|
| Primary combined performance | 38.92 ± 0.90 | 38.81 ± 1.22 |
| Repeat pass | ~39.82 | ~39.79 |
| Independent cold-start | 40.15 ± 1.03 | 37.11 ± 1.64 |

**Conservatively recorded operational range: d0 ~38.9–40.2 t/s, d4096 ~37.1–39.8 t/s.**
Do not promise a fixed 39 t/s floor. Speculative acceptance (normal workloads) ~80–90%
(4h global 86.89%, cold-repro normal-workload 87.63%).

Long soak: 4.004 h, 2,040/2,040 requests, 102 cycles, zero accounting/corruption/fallback/
recapture/rank failure, global acceptance 86.89%. Cold-start: independent reboot both
nodes + fresh cache + fresh graph capture + fresh MTP init, R0–R4 all PASS, R4 60.1 min,
507/507 requests, normal-workload acceptance 87.63%.

(Raw logs are NOT copied here — see the artifact directories above.)

## 6. Memory envelope (validated)

- Clean boot: ~117–118 GiB MemAvailable per node.
- Weight load: ~73.82 GiB/rank (combined-mode load ~75.5 GiB incl. MTP head 39 params).
- Init swap: ~2–3 GiB, then flat (no sustained paging during measured inference).
- Steady-state minimum: ~30–32 GiB MemAvailable.
- Graph capture: ~2 s, ~0 GiB (one target/verify graph per rank). Negative graph-memory
  log values such as `-0.01 GiB` are allocator accounting noise, not real negative use.
- Post-stop: GB10 UMA retains ~37–40 GiB MemAvailable. **A clean reboot is required
  before the next full model load.** Automatic reboot is NOT part of any preset.

## 7. Required startup gate (operator procedure)

Before every full model load: (1) stop any existing workload; (2) confirm no stale vLLM /
Ray / profiler process; (3) inspect MemAvailable; (4) if MemAvailable < 110 GiB due to UVM
retention, reboot BOTH nodes exactly once; (5) confirm swap zero; (6) verify ports free;
(7) verify RoCE and RDMA; (8) verify numeric RoCE bidirectional connectivity; (9) verify
image config ID on both nodes; (10) verify preset SHA on both nodes; (11) clear ONLY the
dedicated vLLM compilation cache (`./.cache/vllm`) on both nodes; (12) start worker then
head per the validated procedure; (13) prove NET/IB; (14) prove graph capture size `[2]`;
(15) prove MTP activation; (16) run bounded correctness checks; (17) begin serving only
after all gates pass. **Automatic reboot is not part of the preset.**

## 8. Operational stop conditions

Immediate stop: malformed/garbled output; token-accounting violation; duplicate/dropped
token; graph fallback; graph recapture; MTP inactivity; rank exit; NCCL error; CUDA graph
error; sustained paging; MemAvailable < 12 GiB; persistent HTTP health failure; host
responsiveness degradation; disk free < 20 GiB on spark01.

Warning: MemAvailable < 16 GiB; swap growth > 512 MiB after stabilization; normal-workload
acceptance < 75% over repeated windows; d0/d4096 throughput outside the validated range by
> 10%; material change in graph capture duration/memory; image or preset hash mismatch.

## 9. Rollback hierarchy

| Level | Preset | Mode | Decode | When to use |
|---|---|---|---|---|
| Validated | `...-mtp1-fullgraph-validated-tp2.env` | MTP n=1 + FULL graph | ~38.9–40.2 / 37.1–39.8 | primary |
| L1 | `...-fullgraph-validated-rollback-tp2.env` | MTP off, FULL graph | ~27.2 | speculative/MTP fault, graph stable |
| L2 | `...-eager-u0-rollback-tp2.env` | MTP off, enforce-eager | ~7.4 | graph-capture fault; maximal stability |

Bounded rollback procedure (any level): stop serving → preserve logs → controlled
shutdown → clean reboot if required by UVM retention → clear dedicated vLLM cache →
select rollback preset → verify image + preset hashes → start cluster → run correctness
gate → verify transport (NET/IB) → verify expected graph (`[2]` / `[1]`) or eager mode →
record rollback reason. Deletion and reboot are NEVER automated inside a preset.

## 10. Known limitations

- Concurrency validated **only at 1**. Higher concurrency is unvalidated.
- MTP token count validated only at n=1. Capture size locked at `[2]`.
- Parser / tool-parser / B12X / Ray are OFF and unvalidated in this configuration.
- Single-node TP=1 is impossible for the official 46-shard checkpoint.
- The historical opaque **34.59 t/s** equivalence remains **OPEN** (not used for any
  classification; the 1.13× ratio is informational only).
- Promotion to `PRODUCTION_BASELINE` is NOT performed by this record and requires a
  separate explicit authorization. The current production/promoted DSV4 path
  (`dsv4-d568`) remains untouched.
