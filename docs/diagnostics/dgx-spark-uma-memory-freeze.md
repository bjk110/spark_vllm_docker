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
| 9 | 0.85 (ep-debug, memory-guard THRESH=4096MB, 0.1s) | 3810 MB | ~8290 MB (no trip) | **first non-freezing capture** — `docker kill vllm-spark-head`, SSH stayed responsive, avail stabilized ~5.3-5.5GB. See §9. |

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
| [`.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug) | Attempt 10b: low-kv-debug's reduced context/batch knobs + ray-tuned-debug's Ray object-store/dashboard/memory-monitor knobs, combined |
| [`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug) | Attempt 11A: ep-off-debug + ray-tuned-debug's Ray memory-monitor safety net, combined |

All six are **disposable/debug** — not promoted/stable presets. None of
them are claimed to fix the underlying issue; they exist to narrow down
*where* the ~20GB+ growth comes from.

See §9 (Attempt 09 results), §10 (Attempt 10 plan), §11 (Attempt 10
results) and §12 (Attempt 11 plan) below for the most recent findings and
the rationale for the combined `low-kv-ray-tuned-debug` and
`ep-off-ray-tuned-debug` variants.

## 9. Attempt 09 (2026-06-12) — first non-freezing capture

Run: `.env.step37-fi-aot-tp2-ep-debug`, Experiment A (spark01=head,
spark02=worker), `scripts/diag/trace-memory.sh` and
`scripts/diag/memory-guard.sh --threshold-mb 4096 --interval 0.1` active on
both nodes. Raw traces preserved as
`.local/diag/diag-spark01-attempt09-*.tar.gz` and
`.local/diag/diag-spark02-attempt09-*.tar.gz` on each node.

### 9.1 Outcome: memory-guard worked, host did not freeze

For the first time in 9 attempts, `memory-guard.sh` killed
`vllm-spark-head` **before** the host became unresponsive. SSH stayed up
throughout; `MemAvailable` dropped to ~3.81GB at the trip, then recovered
and stabilized around **5.3-5.5GB** after the kill. No reboot was strictly
required to keep SSH alive, but see §10 for why a reboot is still
recommended before the next attempt.

### 9.2 ~6-minute timing reconfirmed

- `expert_map_manager.py` "Expert parallelism is enabled... Local/global
  number of experts: 144/288" logged at **13:01:08 UTC**.
- The steep `MemAvailable` decline began at **13:06:54 UTC** — **5m46s**
  later, consistent with the "~6 minutes" pattern from prior attempts.

### 9.3 New finding: the decline is two-phase, not one smooth slope

High-frequency (`0.2s`) `meminfo.log` sampling (host-local times, KST):

| Phase | Window | `MemAvailable` | Rate |
|---|---|---|---|
| Phase 1 | 22:06:54 - 22:06:58 (~4s) | 52.9 GB -> 30 GB | ~5.7 GB/s |
| Plateau | 22:06:58 - 22:07:03 (~5s) | ~30 GB (flat) | ~0 |
| Phase 2 | 22:07:03 - 22:07:15 (~12s) | 30.3 GB -> 3.0 GB | accelerating, up to ~3.6 GB/s near the end |
| Trip | 22:07:15.322 | 3810 MB | `memory-guard.sh` kills `vllm-spark-head` |
| Post-kill | 22:07:16.4 onward | ~5.5 GB (stable) | — |

The ~5s plateau around 30GB between the two phases was not visible in
earlier lower-frequency captures. It suggests two distinct allocation
events ~9 seconds apart, not one continuous ramp.

### 9.4 CORRECTION: Shmem / SUnreclaim / swap deltas are teardown artifacts, not root-cause indicators

A previous pass over this same run's `memory-guard.sh` pre-kill/post-kill
`/proc/meminfo` snapshots noted large deltas in `Shmem` (-418MB),
`SUnreclaim` (-2GB), and `SwapFree` (+6.5-6.8GB freed) and floated these as
candidate growth sources (Ray object store / `/dev/shm`, kernel
slab/UVM/RDMA buffers, swap usage).

The full `meminfo.log` trace shows this was **wrong**: all three values are
essentially flat through Phase 1, the plateau, and Phase 2 (e.g. `shmem`
stays at 487744 -> ~418512 the whole decline, only dropping to 40 at
22:07:16.194 -- i.e. **after** the kill at 22:07:15.322). The large deltas
previously reported are **container-teardown effects** (the killed
container's `/dev/shm` segments and slab caches being released, plus
`swapoff`-like swap-in reversal as the killed process's swapped pages are
freed), not signals present during the actual decline. **Do not use
Shmem/SUnreclaim/swap as the primary signal for "what's growing" in future
attempts** — see §9.5 for what the data actually shows.

### 9.5 MOST IMPORTANT FINDING: the ~41GB growth is outside the `vllm-spark-head` container's cgroup

`docker_stats.log` (5s snapshots) for `vllm-spark-head` during the same
window:

| Time | Container MEM USAGE |
|---|---|
| 22:06:46 | 11.3 GiB |
| 22:06:53 | 8.49 GiB |
| 22:07:00 | 4.01 GiB |
| 22:07:07 | 1.54 GiB |
| 22:07:15 | (killed, gone) |

The container's **own cgroup memory usage was *decreasing*** throughout
Phase 1 and Phase 2 -- the opposite of what you'd expect if a process
inside the container were the thing ballooning.

Meanwhile, host-wide `free + cached` (a reasonable proxy for
`MemAvailable` on this kernel) dropped from ~50.5GB (22:06:50) to ~9.2GB
(22:07:13) -- a **~41GB** drop. Since the container's own accounted memory
*fell* by ~10GB over the same interval, the ~41GB of "disappearing"
host memory is **not** accounted to the `vllm-spark-head` cgroup at all.

This is strong evidence against:

- Ray object store / `/dev/shm` (would show in Shmem, see §9.4 -- flat)
- Python heap / Ray actor heap inside the container (would show in the
  container's own cgroup memory -- it *decreased*)
- normal process RSS growth inside the container (same reasoning)

...and points toward allocations that live **outside any container
cgroup**, most plausibly:

- **NVIDIA UVM / driver-level unified-memory allocation** -- on GB10's
  UMA architecture, GPU-visible unified memory pages can be accounted to
  the host kernel / `nvidia-uvm` driver rather than to the requesting
  process's cgroup.
- **KV cache reservation / CPU-side block-table bookkeeping** sized by
  `MAX_MODEL_LEN` / `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS`, if that
  bookkeeping is allocated via a path (e.g. pinned/mapped host memory for
  GPU DMA) that bypasses normal cgroup accounting.
- **NCCL/RDMA buffer registration** for EP all-to-all setup -- RDMA memory
  registration (`ibv_reg_mr`) pins physical pages and is typically not
  cgroup-accounted the same way as regular anonymous memory.

### 9.6 `--enforce-eager` was already active -- CUDA graph capture ruled out

`ep-debug`'s `VLLM_EXTRA_ARGS` already includes `--enforce-eager`. CUDA
graph capture (a leading suspect in earlier write-ups of this issue) was
therefore **not** running during this attempt, yet the same two-phase
decline occurred. CUDA graph capture is ruled out as the (sole) cause.

### 9.7 Bug found: `ps_topRSS.log` was empty for the entire run

`scripts/diag/trace-memory.sh`'s top-RSS sampler used
`ps -eo pid,ppid,comm,rss,shr,stat,wchan:32 --sort=-rss 2>/dev/null`. On
this host's `procps-ng 4.0.4`, `shr` is not a valid `-o` field
(`error: unknown user-defined format specifier "shr"`), so every
invocation failed and `2>/dev/null` silently discarded the error -- all 174
snapshot headers in `ps_topRSS.log` were followed by zero process lines.
**Per-process RSS growth during the ~6-minute pre-decline window and the
two-phase decline itself was not captured for Attempt 09.**

Fixed (this commit): the sampler now uses
`ps -eo pid,ppid,user,comm,rss,vsz,stat,wchan:32 --sort=-rss` (no `shr`,
which doesn't exist on this `procps-ng`) and no longer suppresses stderr
(`2>&1` into `ps_topRSS.log`), so a future field error is visible in the
log instead of producing silent empty snapshots.

Separately, `trace-memory.sh`'s header and the auto-stop-timer comment both
referred to a `trap cleanup INT TERM` that was never actually registered --
`SIGTERM`/`SIGINT` terminated the script via bash's default disposition
without running `cleanup()`, leaving background loggers as orphaned
processes and skipping `final_snapshot.log`. This is also fixed (the `trap
cleanup INT TERM` call is now present after the `cleanup()` definition).
Attempt 09's loggers were stopped and finalized manually (see the
`diag-*-attempt09-*.tar.gz` tarballs).

## 10. Attempt 10 plan

Based on §9:

1. **Reboot both nodes first.** After the Attempt 09 emergency kill,
   `MemAvailable` settled around ~5.2-5.5GB on both spark01 and spark02 --
   tight enough that starting a new attempt from this state risks tripping
   `memory-guard.sh` almost immediately for unrelated reasons. Per
   [`docs/step3.7-flash-tp2.md`](../step3.7-flash-tp2.md) §2, `sudo
   systemctl reboot` (not combined with `pkill` in the same ssh
   invocation) is the only confirmed clean GB10 UMA reclaim.

2. **Use `.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug`** (new, see §8) --
   combines low-kv-debug's reduced `MAX_MODEL_LEN`/`MAX_NUM_SEQS`/
   `MAX_NUM_BATCHED_TOKENS` (targets the KV-cache-bookkeeping hypothesis
   from §9.5) with ray-tuned-debug's Ray object-store/dashboard/
   memory-monitor knobs (a safety net / additional signal, even though
   §9.5 makes Ray object store a less likely root cause than before).

3. **Raise the memory-guard threshold to 8192MB** (was 4096MB):

   ```bash
   ./scripts/diag/memory-guard.sh --threshold-mb 8192 --interval 0.1 &
   ```

   4096MB gave the kill command only a narrow window before the host-wide
   stall in Attempt 09 (trip at 3810MB, Phase 2's last ~3 seconds dropped
   from ~9.5GB to ~3.0GB). 8192MB trips earlier in the decline, giving more
   margin if `low-kv-ray-tuned-debug` doesn't fully prevent the decline
   but only slows it.

4. Re-run `trace-memory.sh` (now fixed -- see §9.7) on both nodes and check
   `ps_topRSS.log` for per-process RSS growth during the ~6-minute window
   and the decline itself -- this is the one diagnostic signal Attempt 09
   could not provide.

## 11. Attempt 10 results (2026-06-12)

### 11.1 Attempt 10a -- INVALID (`vm.min_free_kbytes` set too high)

Before the §10 run, `vm.min_free_kbytes` was raised to roughly 6GiB on both
nodes as an extra safety margin (intent: give the kernel more reclaim
headroom before a collapse). This made the kernel's own free-memory
accounting reject `init_device()`'s pre-init memory check immediately --
the run failed before reaching the EP marker or weight loading, i.e.
**before the phenomenon under investigation could even start**. Attempt 10a
produced no signal about the collapse and is **invalid** -- do not cite it
as evidence either way.

Fix: `vm.min_free_kbytes` was reverted back to its default (`45156`,
confirmed via `sysctl vm.min_free_kbytes` on both nodes after Attempt 10b --
see §11.2) before re-running. **Do not set `vm.min_free_kbytes` to ~6GiB on
these hosts** -- it breaks `init_device()`'s pre-init check independent of
this bug.

### 11.2 Attempt 10b -- non-freezing capture with `low-kv-ray-tuned-debug`

With `vm.min_free_kbytes` reverted (`45156`, `vm.swappiness=1` on both
nodes) and `.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug` (§8), the run
reached its intended path: weight loading completed, the EP marker
appeared on both ranks, and the run ended in a **clean, logged Ray-level
OOM error** rather than a host freeze.

**Timeline (UTC):**

| Event | Time |
|---|---|
| EP marker ("Expert parallelism is enabled"), both ranks | 14:15:45 / 14:15:46 |
| Weight loading window | 14:15:55 -> ~14:21:35 |
| Ray memory monitor kills `RayWorkerWrapper` (node0) | 14:21:41 |
| `vllm-spark-head` exits (EngineCore init failure) | shortly after 14:21:41 |

**EP-marker -> crash timing: 5m56s** (14:15:45 -> 14:21:41), vs Attempt 09's
**5m46s**. The two are within 10 seconds of each other despite very
different `MAX_MODEL_LEN`/`MAX_NUM_SEQS`/`MAX_NUM_BATCHED_TOKENS` (8192/4/8192
in Attempt 09 vs 4096/2/2048 here) -- **strongly suggests a roughly
deterministic, time-based (or weight-loading-completion-based) trigger,
largely independent of the low-KV knobs.**

**Crash signature** (`vllm-spark-head` log tail, ~14:21:41 UTC):

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node
running low on memory. Memory on the node (IP: 10.10.10.1) was 110.41GB /
121.63GB (0.907749), which exceeds the memory usage threshold of 0.900000.

RuntimeError: Engine core initialization failed. See root cause above.
Failed core proc(s): {}
```

i.e. **Ray's own memory monitor** (`RAY_memory_usage_threshold=0.90`,
`RAY_memory_monitor_refresh_ms=100`) killed the `RayWorkerWrapper` actor on
node0 once node0's total memory crossed 90% of 121.63GiB, which then
propagated as an `EngineCore` initialization failure and a clean
`Exited(1)` for `vllm-spark-head`. **The host did not freeze.**

**Fine-grained collapse trajectory (node0 / spark01, `meminfo.log`, KST =
UTC+9, so 23:21:xx KST = 14:21:xx UTC):**

| Time (KST) | MemAvailable |
|---|---|
| 23:21:34.674 | 50,786,168 KB (~48.4 GiB) |
| 23:21:36.716 | 45,623,876 KB (~43.5 GiB) |
| 23:21:38.777 | 33,956,664 KB (~32.4 GiB) |
| 23:21:40.834 | 19,981,540 KB (~19.1 GiB) -- peak decline rate ~7 GB/s |
| 23:21:42.893 | 14,336,072 KB (~13.7 GiB) |
| 23:21:43 -- 23:21:51 | oscillates 13.8 -- 14.4 GiB |
| 23:21:53 onward | stabilizes ~15.15 -- 15.2 GiB (flat for 75+ s) |

Total: **~37GB drop in ~17 seconds**, bottoming out around 13.7GiB before
recovering slightly and holding at ~15.2GiB once Ray's kill completed and
the container exited. The post-run `free -m` snapshot taken in this
postmortem (well after the run) still shows spark01 at
`available=14805MB` -- consistent with this ~15.2GiB stabilization point
persisting until reboot.

**cgroup-external pattern -- 2nd independent confirmation.** Exactly as in
Attempt 09 (§9.5), `docker stats` for `vllm-spark-head` *decreased* during
the collapse while host `MemAvailable` cratered:

| Time (KST) | `vllm-spark-head` MEM USAGE |
|---|---|
| 23:21:27.670 | 10.24 GiB |
| 23:21:34.694 | 3.5 GiB |
| 23:21:41.734 | 1.173 GiB |
| 23:21:48.773 | 1.546 GiB |

i.e. the container's own cgroup memory accounting was *falling* throughout
the same window where host `MemAvailable` dropped ~37GB. This is the
**second independent confirmation** (Attempt 09 + Attempt 10b) that the
growth driving the collapse is **outside** `vllm-spark-head`'s cgroup --
most likely NVIDIA UVM/driver-level unified memory or a similar
host-global allocation not charged to the container's cgroup.

**Low-KV result.** Reducing `MAX_MODEL_LEN`/`MAX_NUM_SEQS`/
`MAX_NUM_BATCHED_TOKENS` from 8192/4/8192 (Attempt 09) to 4096/2/2048
(Attempt 10b):
  - did **not** prevent the collapse,
  - did **not** meaningfully change the ~6-minute EP-marker-to-crash delay
    (5m46s vs 5m56s), and
  - therefore makes **KV-cache reservation / CPU-side bookkeeping sized by
    these knobs an unlikely *primary* trigger** for the collapse. It may
    still be a secondary/contributing factor, but it is not the dominant
    one.

**Ray memory-monitor result.** `RAY_memory_usage_threshold=0.90` +
`RAY_memory_monitor_refresh_ms=100` (the "ray-tuned" knobs) converted what
was an unresponsive-host freeze in Attempt 09 into a **controlled,
logged Ray-level OOM crash** in Attempt 10b -- at a much higher
`MemAvailable` floor (~13.8GiB here vs Attempt 09's ~3.8-5.5GiB). **These
two Ray knobs should remain enabled (`RAY_memory_usage_threshold=0.90`,
`RAY_memory_monitor_refresh_ms=100`) for all future dangerous diagnostic
runs on this stack** -- they cost nothing when the collapse doesn't happen,
and turn a freeze into a diagnosable crash when it does.

**`ps_topRSS.log` result (first successful capture, post-§9.7 fix).** At
23:21:39.425 KST (just before the Ray kill), the top-RSS snapshot on
spark01 showed:

| Process | host PID | container PID | RSS | VSZ |
|---|---|---|---|---|
| `ray::RayWorkerWrapper` | 105727 | 1880 | ~1.02 GiB (1,071,660 KB) | **~790.6 GiB (829,302,508 KB)** |
| `vllm` (APIServer, pid 102478) | -- | -- | 194,452 KB -> 547,836 KB (rising during teardown) | -- |
| `VLLM::EngineCore` (pid 104324) | -- | -- | 112,240 KB -> 403,460 KB (rising during teardown) | -- |

By 23:21:54.474 KST, all `vllm`/Ray processes were gone (container exited).

`RayWorkerWrapper`'s **VSZ=~790.6GiB is ~6.5x the entire 121.63GiB UMA
pool**, while its RSS was only ~1.02GiB. **VSZ is virtual address-space
reservation, not physical memory** -- it does not by itself explain the
~37GB *physical* `MemAvailable` drop. But a reservation this large (likely
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` virtual-address
over-commit, and/or NCCL/RDMA registered-memory virtual mappings) is
consistent with a process that *could* trigger a large physical-page
commit if a significant fraction of that virtual space gets touched/backed
around the EP-marker-to-crash window. **Next run must capture
`/proc/<pid>/smaps_rollup` (Rss/Pss/Anonymous/Locked/Swap, etc.) and a
categorized `/proc/<pid>/maps` summary for `RayWorkerWrapper` (and
`EngineCore`) BEFORE Ray kills the worker**, to distinguish "huge VSZ,
small RSS, irrelevant" from "huge VSZ, RSS ramping fast in the final
seconds, this is the mechanism." This is implemented in
`scripts/diag/trace-memory.sh`'s new proc_maps sampler (§13).

**node1 (spark02/worker) stability.** `meminfo.log` on spark02 oscillated
53.8-58.0GB `MemAvailable` throughout, then **jumped to 61.03GB at
23:21:43.675 KST and stayed flat** -- i.e. node1 *released* resources at
almost exactly the moment node0's corresponding worker was killed, rather
than itself collapsing. `docker stats` for `vllm-spark-worker` stayed flat
at ~1.23GiB (1.01%) throughout. The postmortem `free -m` snapshot in this
session shows spark02 at `available=60115MB`, consistent with this
~61GB stabilization point. **Node1/the worker rank was unaffected** -- the
collapse is node0/head/rank0-specific in this run, as in Attempt 09.

**Diagnostics housekeeping note.** In this run's postmortem, `docker
compose down` was executed on both containers *before* `docker logs
vllm-spark-head`/`vllm-spark-worker` were saved to a file, so the full
container logs are not recoverable for Attempt 10b (only the excerpts
above, captured in-session before removal, and `trace-memory.sh`'s
independent meminfo/free/ps/docker-stats/kernel logs, which are unaffected
and are the primary data source above). **For Attempt 11, save full
`docker compose ... logs --no-color > ...` to a file BEFORE running
`down`.** See `.local/diag/docker-logs-NOTE-attempt10b.txt` in each node's
tarball for details.

## 12. Attempt 11 plan

Two isolation experiments, in priority order. Both **keep the Ray
memory-monitor safety net** (`RAY_INCLUDE_DASHBOARD=false`,
`RAY_OBJECT_STORE_MEMORY_BYTES=4294967296`,
`RAY_memory_usage_threshold=0.90`, `RAY_memory_monitor_refresh_ms=100`,
`RAY_DASHBOARD_MAX_EVENTS_TO_CACHE=1000`) and `memory-guard.sh
--threshold-mb 8192 --interval 0.1` per §11.2's finding that these
knobs convert a freeze into a diagnosable crash at no cost. **Do not set
`vm.min_free_kbytes=6GiB`** (§11.1). **Reboot both nodes before each run**
(GB10 UMA reclaim, see §10 step 1).

### 12A. EP-off + Ray-tuned isolation (`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`)

**Goal:** determine whether `--enable-expert-parallel` (EP all-to-all / MoE
expert-routing workspace) is the trigger for the ~6-minute-post-EP-marker
collapse, independent of the Ray safety net.

Based on `.env.step37-fi-aot-tp2-ep-off-debug` (§8) +
the Attempt 10b Ray-tuned knobs above, see
[`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug)
for the full file and rationale. `--enable-expert-parallel` is removed (as
in `ep-off-debug`), `--enforce-eager` is retained, and
`MAX_MODEL_LEN`/`MAX_NUM_SEQS`/`MAX_NUM_BATCHED_TOKENS` are left at
`ep-off-debug`'s existing 8192/4/8192 (not Attempt 10b's low-kv
4096/2/2048).

**`GPU_MEMORY_UTILIZATION`: primary Attempt 11A run uses 0.85** (Attempt
10b's value), **not** `ep-off-debug`'s 0.80 -- so that EP-off is the
*only* new variable vs. Attempt 10b / the EP-enabled baseline. `0.80` was
`ep-off-debug`'s choice for a *different* reason (headroom for EP-off's ~2x
per-rank MoE weight footprint, 288 experts/rank instead of 144) and mixing
it in here would confound "did EP-off change anything?" with "did lowering
util change anything?".

- If `init_device()`'s pre-init check **fails at 0.85 with EP off**, that is
  itself a **valid Attempt 11A result** -- do not lower
  `GPU_MEMORY_UTILIZATION` in the primary env to "fix" it.
- **Only then**, retry as **Attempt 11A-fallback** with a separate env at
  `GPU_MEMORY_UTILIZATION=0.80` (e.g.
  `.env.step37-fi-aot-tp2-ep-off-ray-tuned-gpu080-debug`).
- **Results from the 0.80 fallback must NOT be interpreted as a pure
  EP-only isolation** -- both EP (off) and `GPU_MEMORY_UTILIZATION`
  (0.85->0.80) changed relative to Attempt 10b, so a collapse/no-collapse
  result there cannot distinguish "EP-off fixed it" from "lower util fixed
  it".

**Interpretation (primary, 0.85):**

| Result | Implication |
|---|---|
| EP-off removes the collapse (no EP marker phase / no ~6min decline / clean steady state) | EP / all-to-all / MoE expert-routing workspace is the key trigger |
| EP-off still shows the same ~6min collapse | Issue is more likely in the rank0 / `RayWorkerWrapper` / UVM / CUDA virtual-address path independent of EP (consistent with §11.2's VSZ=790.6GiB finding) |
| `init_device()` pre-init memory check fails at 0.85 | EP-off's ~2x per-rank MoE weight footprint (288 experts/rank instead of 144) pushed memory requirements over `GPU_MEMORY_UTILIZATION=0.85` -- a clean, fast, diagnosable failure; report as-is, then run Attempt 11A-fallback at 0.80 (see above) |
| node0-only Ray OOM again | Head/rank0 asymmetry persists independent of EP |
| node1 also becomes unstable | Points to a common UVM/weight-loading path, not head-specific |

### 12B. Head/worker role-swap with low-kv-ray-tuned

**Goal:** determine whether the failure follows the Ray **head/API node**
role, the vLLM **rank0** role, or the **physical node** (spark01), by
swapping which physical node runs the head/rank0 role while keeping
`.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug`'s settings (the
proven-non-freezing Attempt 10b config). See §6 for the exact
head/worker role-swap command pattern (swap `HEAD_ROCE_IP` /
`WORKER_ROCE_IP` and which physical node runs `--profile head` vs
`--profile worker`).

| Result | Implication |
|---|---|
| Collapse now happens on spark02 (the new head/rank0) | Role-bound (head/rank0), not spark01-hardware-specific |
| Collapse still happens on spark01 (now the worker) | spark01-hardware-specific (e.g. a marginal DIMM, firmware, or driver-state difference between the two GB10 boards) |
| Collapse happens on both | Not a simple head-vs-worker or hardware asymmetry; revisit §11.2's UVM/VSZ hypothesis as the primary path regardless of role |

## 13. `trace-memory.sh` proc_maps sampler (added 2026-06-12)

To capture the data §11.2 identified as missing -- per-process memory-map
detail for `RayWorkerWrapper`/`EngineCore`/vLLM processes *before* Ray
kills them during a collapse -- `scripts/diag/trace-memory.sh` now runs an
additional background sampler (enabled by default, `--no-proc-maps` to
disable):

- Every `--proc-maps-interval` seconds (default 2s; `--proc-maps-burst-interval`,
  default 0.5s, once `MemAvailable` < `--proc-maps-threshold-mb`, default
  32768MB), it identifies up to `--proc-maps-max-pids` (default 8)
  highest-RSS host-namespace processes matching `--proc-maps-pattern`
  (default `RayWorkerWrapper|EngineCore|vllm|python`).
- For each candidate PID, writes timestamped snapshots under
  `<out-dir>/proc_maps/pid_<pid>_<comm>/`:
  - `status-<ts>.log` -- raw `/proc/<pid>/status`
  - `smaps_rollup-<ts>.log` -- `Rss`, `Pss`, `Shared_Clean`, `Shared_Dirty`,
    `Private_Clean`, `Private_Dirty`, `Anonymous`, `Locked`, `Swap` fields
    extracted from `/proc/<pid>/smaps_rollup`
  - `limits-<ts>.log` -- raw `/proc/<pid>/limits`
  - `numa_maps-<ts>.log` -- `/proc/<pid>/numa_maps`, capped at 1MB
  - `maps_summary-<ts>.log` -- `/proc/<pid>/maps` summarized by category
    (`anonymous`, `deleted`, `nvidia_dev`, `infiniband_dev`, `dev_shm`,
    `memfd`, `cuda_nccl_lib`, `other`) with per-category mapping count and
    total size in KB, instead of a raw (potentially huge) dump
- Rotation: keeps only the newest `--proc-maps-max-snapshots` (default 20)
  snapshots per PID per file kind.
- Permission-denied / unreadable files are logged to `notes.log` and
  skipped; the sampler continues.
- Runs in the host PID namespace (the script is not containerized), so
  recorded PIDs are host-namespace PIDs directly usable with `/proc/<pid>`.
- Designed to be lightweight (no `cp -r`, no raw `maps`/full `numa_maps`
  dumps) so it does not itself meaningfully add to memory pressure during a
  collapse.
