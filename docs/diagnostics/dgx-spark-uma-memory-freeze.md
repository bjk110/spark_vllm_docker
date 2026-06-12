# DGX Spark UMA host-memory freeze (dual-node vLLM/Ray/EP startup)

**Status: diagnostic runbook. No fix is claimed.** This document describes a
reproducible host-memory failure observed when launching
`stepfun-ai/Step-3.7-Flash-NVFP4` on dual DGX Spark / GB10 with
`TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`, and `--enable-expert-parallel`, and
provides a controlled-blast-radius framework for narrowing down the cause.
The same failure shape may apply to other EP/MoE + Ray + dual-Spark
configurations.

## 1. Observed failure

Hardware: 2x NVIDIA DGX Spark (GB10, Blackwell sm_121), each with **121.63
GiB unified host/GPU memory** — CPU and GPU allocations share one physical
RAM pool, there is no discrete VRAM. Connected via 200 Gbps RoCE/RDMA.

Software: vLLM 0.22.1 (NGC 26.05 base), `stepfun-ai/Step-3.7-Flash-NVFP4`
(288-expert MoE, modelopt NVFP4 weights + FP8 KV cache), `TP_SIZE=2`,
`DISTRIBUTED_BACKEND=ray`, `--enable-expert-parallel` (144 experts/rank).
Node0 ("head") runs Ray head + GCS + dashboard + the OpenAI-compatible API
server, in addition to the vLLM worker. Node1 ("worker") runs only the Ray
raylet + vLLM worker.

Timeline (from repeated attempts):

1. All 14/14 model weight shards load normally on both ranks.
2. Both ranks log from `expert_map_manager.py`:
   ```
   [EP Rank N/2] Expert parallelism is enabled. ... Local/global number of
   experts: 144/288. ...
   ```
3. **~6 minutes after** that log line, `free -m` "available" on the
   **head node only** begins a steep, accelerating decline — observed
   going from ~30 GB available to a low value in 15-20 seconds (e.g.
   30163 MB -> 142 MB in 17 seconds in one run).
4. This is **not** a clean Linux OOM-kill of the vLLM process. Once
   available memory gets very low, the **entire host** becomes
   unresponsive: ICMP ping still replies (network stack alive), but SSH
   fails with "Connection timed out during banner exchange" (sshd cannot
   be scheduled). The host stays frozen until a hard reboot
   (`systemctl reboot` if reachable, otherwise physical power cycle).
5. The **worker node** experiences a smaller dip during the same window,
   but it reliably **self-stabilizes around a 5.4-5.6 GB available
   floor** and does not freeze — independent of what happens on the head.
6. At the moment the head enters its dip, it already has roughly **20 GB
   less available memory** than the worker at the equivalent point in
   startup (e.g. ~30 GB vs ~53 GB available).

### Attempt history (GPU_MEMORY_UTILIZATION sweep)

| Attempt | util | Head dip floor observed | Worker dip floor | Outcome |
|---|---|---|---|---|
| 4 | 0.88 | ~6920 MB | — | watchdog kill |
| 5 | 0.88 | 1739 MB | ~5440 MB | watchdog kill |
| 6 | 0.85 | 1599 MB | — | watchdog kill |
| 7 | 0.85 (THRESH=500MB) | 142 MB | ~5576 MB | watchdog kill (caught at 142MB) |
| 8 | 0.70 | not captured | — | **full host freeze**, watchdog could not react |

Lowering `GPU_MEMORY_UTILIZATION` from 0.88 to 0.70 (≈+18 GB extra
headroom at 0.70 vs 0.85) did **not** stop the head-node decline, and the
most aggressive reduction produced the *worst* outcome. The decline's
floor trends toward 0 regardless of `util`, just reached at different
absolute `avail` values depending on when the watchdog catches it.

## 2. Why `expert_map_manager.py` is likely a phase marker, not the cause

The `[EP Rank N/2] Expert parallelism is enabled... Local/global number of
experts: 144/288` log line appears immediately after weight loading
completes — it is emitted once, synchronously, as part of EP setup
bookkeeping. The steep memory decline begins **~6 minutes later**, not at
that log line. A static expert-map data structure for 288 experts (index
arrays, rank assignment tables) is on the order of kilobytes to low
megabytes — nowhere near the tens of GB observed. The log line is much
more useful as a **timestamp marking the start of the EP-setup phase**
(after which some other initialization — warmup, profiling, all-to-all
buffer allocation, CUDA graph capture prep, etc. — runs for ~6 minutes
before the decline begins) than as the allocation site itself.

## 3. Why the GB10 UMA architecture can freeze instead of OOM-killing

On a normal (discrete VRAM) system, a process that over-allocates host RAM
gets killed by the kernel OOM killer, and a process that over-allocates
GPU memory gets a `CUDA error: out of memory` from the driver — both are
clean, attributable failures.

On GB10, CPU and GPU memory are the same physical pool
(`MemAvailable` in `/proc/meminfo` reflects both). When available memory
drops toward zero very quickly:

- The kernel may enter aggressive page-cache reclaim / writeback, which
  competes for the same CPU cores and memory bus bandwidth that the
  allocating process needs.
- Pinned/non-reclaimable allocations (NVIDIA UVM pages, `mlock`ed pages,
  `SHM`/Ray object-store segments, `Unevictable` pages) do not give memory
  back under pressure — they shrink the reclaimable pool further.
- If the rate of allocation outpaces reclaim, **every** process on the
  host — including sshd — can be starved of schedulable memory/CPU at
  once, producing the "ping works, SSH banner-exchange times out"
  signature. This is a host-wide stall, not a per-process OOM event, so
  there may be no OOM-killer log line at all.

This is why `scripts/diag/trace-memory.sh` captures `Cached`,
`SReclaimable`, `Shmem`, `Unevictable`, and `Mlocked` specifically — these
distinguish "reclaimable page cache being consumed" from "non-reclaimable
allocations growing" from "Ray object store / shm growing".

## 4. Why the head/rank0 node has less headroom

Node0 ("head") runs several processes that node1 ("worker") does not:

- Ray head process + GCS (cluster metadata, actor tables)
- Ray dashboard (event cache, metrics aggregation)
- The OpenAI-compatible API server (uvicorn/FastAPI)
- vLLM rank0 ("driver") bookkeeping — rank0 in a TP/EP deployment often
  holds additional coordination state (e.g. aggregating results across
  ranks) that rank>0 does not.

The observed ~20 GB headroom gap between head and worker at the same point
in startup is consistent with this extra process set, but has **not** been
isolated to a specific one of these. Section 6 (role-swap experiment)
is the way to test whether the gap follows the **Ray head role**, the
**vLLM rank0 role**, or the **physical node** — these three normally
coincide and have never been tested independently.

## 5. Why lowering `GPU_MEMORY_UTILIZATION` alone is not sufficient evidence about KV cache

`GPU_MEMORY_UTILIZATION` controls how much of the 121.63 GiB UMA pool vLLM
reserves for **weights + activations + KV cache** (the GPU-side budget).
It does **not** directly bound:

- Ray's object store (defaults to ~30% of host memory, configurable via
  `RAY_OBJECT_STORE_MEMORY_BYTES`)
- Ray GCS/dashboard/API server CPU-side heap
- EP all-to-all communication buffers, which may be sized by
  `MAX_NUM_BATCHED_TOKENS` / expert count rather than by
  `GPU_MEMORY_UTILIZATION`
- Linux page cache built up while reading ~100 GB of model shards from disk
- NVIDIA UVM / pinned-memory bookkeeping

The fact that lowering `util` from 0.88 to 0.70 did not prevent (and
arguably worsened) the freeze is evidence that **the dominant growth is in
one of these CPU-side / Ray-side / EP-side pools, not in the
GPU-memory-utilization-bounded KV cache**. It does not, by itself, identify
*which* pool. `.env.step37-fi-aot-tp2-low-kv-debug` and
`.env.step37-fi-aot-tp2-ray-tuned-debug` target two of these candidates
independently of `util`.

## 6. The role-swap isolation experiment (most important)

Normally, "Ray head", "vLLM rank0/driver", and "physical node spark01" are
the same node in every run. The single most informative experiment is to
**swap which physical node plays which role** and see what the freeze
follows:

- **Experiment A (baseline)**: spark01 = head (Ray head + rank0 + API
  server), spark02 = worker.
- **Experiment B (role swap)**: spark02 = head (Ray head + rank0 + API
  server), spark01 = worker. This requires swapping `HEAD_ROCE_IP`/
  `WORKER_ROCE_IP` (and which physical host runs the `head`/`worker`
  compose profile) — everything else identical.

### 6.1 Exact commands and env-var swap

Use the same `.env.step37-fi-aot-tp2-ep-debug` (or whichever debug env is
under test) for both experiments — only the **profile-to-host mapping** and
the two RoCE IP vars change. Nothing else in the env file needs editing if
`HEAD_ROCE_IP`/`WORKER_ROCE_IP` are passed as overrides on the command line.

| | Experiment A (baseline) | Experiment B (role swap) |
|---|---|---|
| spark01 runs | `--profile head` | `--profile worker` |
| spark02 runs | `--profile worker` | `--profile head` |
| `HEAD_ROCE_IP` | `10.10.10.1` (spark01) | `10.10.10.2` (spark02) |
| `WORKER_ROCE_IP` | `10.10.10.2` (spark02) | `10.10.10.1` (spark01) |

Env vars that **must be swapped** between the two experiments: `HEAD_ROCE_IP`
and `WORKER_ROCE_IP`. Everything else (`MODEL_PATH`, `TP_SIZE`,
`VLLM_EXTRA_ARGS`, Ray tuning knobs, etc.) stays identical.

**Experiment A** (as in §7 Runbook, spark01=head):

```bash
# spark02 (worker)
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile worker up -d
# spark01 (head), after the worker is up
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile head up -d
```

**Experiment B** (role swap, spark02=head): on **both** nodes, override
`HEAD_ROCE_IP=10.10.10.2` and `WORKER_ROCE_IP=10.10.10.1` (either by editing
a copy of the env file or via `-e`/shell env override before `docker compose
up`), then:

```bash
# spark01 now runs the worker profile
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile worker up -d
# spark02 now runs the head profile, after the worker is up
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile head up -d
```

Run `trace-memory.sh` and `memory-guard.sh` on both nodes for both
experiments, exactly as in §7.

### Quick interpretation

| Observation | Implication |
|---|---|
| Freeze follows the **Ray head / API node** (whichever physical node runs the head profile freezes) | Ray head/object store/API overhead likely |
| Freeze follows **vLLM rank0** (independent of which node runs the Ray head) | vLLM rank0/EP/driver path likely |
| Freeze follows a **specific physical node** regardless of role | Hardware/firmware/kernel/NIC/driver asymmetry likely |

### Extended decision table (EP on/off branches)

| Observation | Implication |
|---|---|
| Freeze follows the **Ray head / API role** (whichever physical node runs it freezes) | Ray head/GCS/dashboard/API-server overhead is the primary driver. Try `.env.step37-fi-aot-tp2-ray-tuned-debug`. |
| Freeze follows **vLLM rank0/driver**, independent of which node runs Ray head | vLLM rank0/driver-side EP/MoE setup path is implicated. Try `.env.step37-fi-aot-tp2-ep-off-debug` to test the EP angle specifically. |
| Freeze follows a **specific physical node** regardless of role | Hardware/firmware/kernel/NIC/driver asymmetry between the two Sparks (check `uname -r`, driver versions, `dmesg` for hardware errors on that node specifically). |
| Freeze **disappears** with `--enable-expert-parallel` removed (`ep-off-debug`) | EP all-to-all / MoE workspace / expert-routing path is implicated — but note the memory-footprint trade-off documented in that env file (each rank loads all 288 experts). |
| Freeze **still happens** with EP off | A UMA / page-cache / Ray-object-store / KV-cache / UVM path independent of EP is implicated — focus on `ray-tuned-debug` and `low-kv-debug`. |

## 7. Runbook

All commands run from the repo root (`/home/bjk110/docker/vllm-spark`) on
**both** spark01 and spark02 unless noted. Per
[`docs/step3.7-flash-tp2.md`](../step3.7-flash-tp2.md) §2: after any crash,
only `sudo systemctl reboot` reliably reclaims GB10 UMA memory — do not
combine `pkill` and `reboot` in the same ssh invocation.

### Before the run (both nodes)

```bash
# 1. Check current sysctl headroom settings (no changes yet)
./scripts/diag/prepare-uma-memory.sh

# 2. Optionally apply more conservative headroom (requires sudo)
sudo ./scripts/diag/prepare-uma-memory.sh --apply

# 3. Start the memory trace (foreground or backgrounded with nohup/tmux —
#    Ctrl-C to stop and finalize). Prints its output directory.
./scripts/diag/trace-memory.sh &

# 4. Start the emergency memory guard, conservative threshold
#    (4096 MB default — well above the 500MB used by the old watchdog.sh,
#    to give the kill command a better chance of being scheduled).
./scripts/diag/memory-guard.sh --threshold-mb 4096 &
```

### During the run

```bash
# spark02 (worker) first, then wait, then spark01 (head) — per existing
# repo convention for DISTRIBUTED_BACKEND=ray.
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile worker up -d
sleep 15
docker compose --env-file .env.step37-fi-aot-tp2-ep-debug --profile head up -d
```

Watch for the `expert_map_manager.py` "Expert parallelism is enabled" log
line on both ranks, then watch `meminfo.log` / `free.log` in the trace
output directory for the head-node decline starting ~6 minutes later.

### After a failed/frozen/rebooted run

Collect (from each node, before or immediately after reboot if SSH is
still reachable):

```bash
# Memory trace output (printed by trace-memory.sh at start/exit)
ls .local/diag/memtrace-*/

# Ray session logs (also snapshotted into the trace dir every 30s)
ls /tmp/ray/session_latest/logs/ 2>/dev/null

# Kernel log from the previous boot (after reboot)
journalctl -k -b -1 -e | tail -200

# Container logs (if containers still exist)
docker compose logs head 2>/dev/null | tail -200
docker compose logs worker 2>/dev/null | tail -200

# memory-guard / watchdog log
ls .local/diag/memory-guard-*.log
```

Then compare node0 vs node1:

- `MemAvailable` floor and decline rate (from `meminfo.log`)
- `Cached` / `SReclaimable` collapse — page cache being reclaimed vs.
  something else growing
- `Shmem` growth — Ray object store / `/dev/shm` usage
- `Unevictable` / `Mlocked` growth — pinned/UVM memory that reclaim cannot
  touch
- Top-RSS process growth (`ps_topRSS.log`) — which process's RSS grows
  during the ~6-minute window before the decline
- Ray GCS/raylet/dashboard log memory-related messages

### Cleanup between attempts

```bash
docker compose --profile worker down -t 5
docker compose --profile head down -t 5
# Reboot is the only confirmed clean reclaim of GB10 UMA memory after a
# vLLM container has run (~10-12 min). See docs/step3.7-flash-tp2.md §2.
sudo systemctl reboot
```

## 8. Experiment env files

| File | Purpose |
|---|---|
| [`.env.step37-fi-aot-tp2-ep-debug`](../../.env.step37-fi-aot-tp2-ep-debug) | EP-enabled baseline reproduction (util=0.85) |
| [`.env.step37-fi-aot-tp2-ep-off-debug`](../../.env.step37-fi-aot-tp2-ep-off-debug) | `--enable-expert-parallel` removed — EP isolation |
| [`.env.step37-fi-aot-tp2-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-ray-tuned-debug) | Ray dashboard off + object-store bound + memory monitor re-enabled |
| [`.env.step37-fi-aot-tp2-low-kv-debug`](../../.env.step37-fi-aot-tp2-low-kv-debug) | Reduced `MAX_MODEL_LEN`/`MAX_NUM_SEQS`/`MAX_NUM_BATCHED_TOKENS`, same util |

All four are **disposable/debug** — not promoted/stable presets. None of
them are claimed to fix the underlying issue; they exist to narrow down
*where* the ~20GB+ growth comes from.
