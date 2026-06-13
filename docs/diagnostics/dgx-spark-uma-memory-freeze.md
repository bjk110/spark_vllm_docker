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

## 14. Attempt 11A results (2026-06-12)

**Purpose:** §12A's EP-off + Ray-tuned isolation at
`GPU_MEMORY_UTILIZATION=0.85`
([`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug)),
to determine whether `--enable-expert-parallel` (EP all-to-all / MoE
expert-routing workspace) is required to trigger the ~6-minute delayed
host-memory decline seen in Attempts 09 and 10b. `vm.min_free_kbytes` was
kept at its normal default (`45156`, not the invalid ~6GiB from §11.1), and
the Ray safety knobs (`RAY_memory_usage_threshold=0.90`,
`RAY_memory_monitor_refresh_ms=100`, etc.) remained enabled.

**Timeline (UTC):**

| Event | Time |
|---|---|
| `init_device()` pre-init memory check passes at 0.85, EngineCore initialized | 15:00:26 |
| Weight loading started | 15:00:53 |
| Weight loading 14/14 completed (2m17s) | 15:03:10 / 15:03:11 |
| `Available KV cache memory: 37.8 GiB` | 15:05:08 |
| `GPU KV cache size: 790,438 tokens` (padding-layer warning) | 15:06:07 |
| `Application startup complete` (API server ready) | 15:09:00 |
| Inference test (`/v1/completions`) succeeded | 15:14:42 |
| Trace ended (clean shutdown) | 15:15:08 |

**EP marker absent, as expected.** Logs show normal EP-rank bookkeeping
(`parallel_state.py:1735`) but no "Expert parallelism is enabled" line on
either rank, confirming `--enable-expert-parallel` was correctly removed.

**`init_device()` passed at 0.85** -- no pre-init "Free memory ... is less
than desired GPU memory utilization" failure. The §12A
init_device-fails-at-0.85 branch did not occur; no fallback env was needed.

**Weight loading 14/14 completed in 2m17s** (15:00:53 -> 15:03:10/11),
notably faster than Attempt 10b's 5m40s despite EP-off loading ~2x the
per-rank MoE weights (288 experts/rank vs 144). This timing difference is
not yet explained but is orthogonal to the memory-decline question below.

**Memory behavior -- the ~6-minute decline persists despite EP-off.**

| Time (UTC) | spark01/head MemAvailable | spark02/worker MemAvailable |
|---|---|---|
| 15:03:19 (post weight-load) | ~51.9 GiB | ~54.3 GiB |
| 15:03:50 | ~48.7 GiB | -- |
| 15:07:18 | ~19.1 GiB | ~17 GiB |
| 15:09:00 (`Application startup complete`) | ~15.45 GiB | ~17.8 GiB |
| 15:14:07 (steady) | ~15.31 GiB | ~17.75 GiB |
| 15:14:42 (post-inference) | ~14.52 GiB | ~17.58 GiB |
| 15:15:08 (final) | **13.86 GiB** | 16.75 GiB |

spark01/head dropped **~51.9 GiB -> ~13.86 GiB, ~38 GB total** -- almost the
same magnitude as Attempt 10b's ~37GB collapse. spark02/worker stayed
comparatively flat (~54 -> ~17 GiB initial settle, then essentially flat
~17.0-17.8 GiB).

Final memory usage on spark01: `(127,535,272 - 14,525,992) / 127,535,272 ≈
0.886`, i.e. **just under** Ray's `memory_usage_threshold=0.90` (Attempt
10b's crash occurred at 0.9077). **No Ray OOM occurred and no host freeze
occurred.** The API server reached `Application startup complete` and
served a real `/v1/completions` request successfully (10 tokens generated,
`finish_reason: length`; output text was garbled due to the separately
tracked tokenizer_class issue, unrelated to this memory investigation).

**Interpretation.**

- **EP-off did NOT eliminate the underlying ~38GB memory decline** -- it
  occurred over the same ~6-minute window (weight-loading completion ->
  `Application startup complete`, 15:03:10 -> 15:09:00, ~5m50s, matching
  Attempts 09/10b's 5m46s/5m56s EP-marker-to-crash delay) and at almost the
  same magnitude as Attempt 10b's ~37GB.
- **EP / all-to-all / MoE expert-routing workspace is unlikely to be the
  sole root cause.** The only measurable difference vs. Attempt 10b is that
  the final usage (0.886) landed just under the 0.90 Ray threshold instead
  of just over it (0.9077) -- a margin of only ~1.8GB. EP-off may have
  **lowered or smoothed the peak just enough to avoid crossing the Ray
  threshold**, without removing the underlying growth.
- **The suspect window now shifts to KV cache profiling / GPU KV cache
  allocation / engine startup** (15:05:08-15:09:00, overlapping
  `Available KV cache memory` / `GPU KV cache size` / padding-layer log
  lines), and/or CUDA UVM and rank0/`RayWorkerWrapper`-specific
  virtual-memory behavior (§11.2's VSZ=790.6GiB finding) -- **not EP setup
  alone**.
- **node0/head-only asymmetry persists independent of EP**: spark01 lost
  ~38GB while spark02 stayed essentially flat, consistent with Attempts
  09/10b.
- **node1/worker remained stable** throughout (~54 -> ~17 GiB initial
  settle, then flat).
- Ray memory monitor result is **N/A this run** -- usage stayed under
  threshold, so no Ray-level intervention was needed (but the margin was
  only ~1.8GB, i.e. very close).

**proc_maps sampler result (first live run of §13's sampler).** The sampler
correctly entered burst mode (`MemAvailable` < 32768MB threshold) and
identified the relevant candidate PIDs by name/pattern match, e.g. on
spark01: `VLLM::EngineCor` (EngineCore), `ray::RayWorkerW` (RayWorkerWrapper),
`vllm`, `raylet`, `gcs_server`, and supporting `python`/`ray` processes (8
PIDs tracked, matching `--proc-maps-max-pids`). `status-<ts>.log` and
`limits-<ts>.log` were captured correctly for these PIDs.

However, **`smaps_rollup-<ts>.log`, `numa_maps-<ts>.log`, and
`maps_summary-<ts>.log` were all 0 bytes** for these PIDs. All of the
relevant container processes run as `Uid=0` (root) inside the container,
which maps to host UID 0; `trace-memory.sh` was run as the non-root host
user (`bjk110`, UID 1000), which lacks permission to read
`/proc/<pid>/{smaps_rollup,numa_maps,maps}` for UID-0 processes. This
permission failure was **not logged to `notes.log`** for these PIDs (silent
0-byte files) -- see §15 for the fix. **No mapping-category data
(anonymous/`/dev/nvidia*`/`/dev/infiniband*`/`/dev/shm`/memfd/CUDA-NCCL-lib)
could be obtained from this run.** The next diagnostic run that needs this
data **must run `trace-memory.sh` with `sudo`**.

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt11a-20260613-001531.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt11a-20260613-001531.tar.gz`

Both containers were stopped cleanly (`docker compose ... down`, exit code
0) and `trace-memory.sh`/`memory-guard.sh` were stopped with `SIGTERM`
(`final_snapshot.log` written on both nodes).

## 15. `trace-memory.sh` sudo requirement for proc maps / smaps (added 2026-06-12, post-Attempt-11A)

Attempt 11A (§14) showed that `smaps_rollup`/`numa_maps`/`maps_summary` are
silently written as 0-byte files when `trace-memory.sh` runs as a non-root
user against root-owned container processes (`/proc/<pid>/{smaps_rollup,
numa_maps,maps}` require the same UID or `CAP_SYS_PTRACE`). The script now:

- documents in its `--help` output and header comments that **proc_maps /
  smaps capture for root-owned container processes requires running the
  script with `sudo`** -- `meminfo`/`free`/`docker stats`/`ps_topRSS`
  sampling continues to work fine without `sudo`.
- explicitly checks readability of `/proc/<pid>/{smaps_rollup,maps,
  numa_maps}` before extracting, and on failure writes a
  `permission denied or unreadable: <path> (PID <pid>, uid mismatch? try
  sudo)`-style line to `notes.log` instead of leaving a silent 0-byte file.

For any future attempt where proc_maps/smaps data is required (e.g. to
resolve §14's open question about mapping categories during the
15:05:08-15:09:00 window), run `sudo ./scripts/diag/trace-memory.sh ...`.

## 16. Attempt 12 results (2026-06-12) -- fixed KV cache memory isolation

**Purpose:** §14 narrowed the suspect window for the ~38GB head-node-only
decline to KV cache profiling / GPU KV cache allocation / engine startup
(15:05:08-15:09:00, overlapping `Available KV cache memory` and `GPU KV
cache size`). Attempt 12 tests this directly by replacing vLLM's
automatic KV-cache-size derivation (driven by `GPU_MEMORY_UTILIZATION`)
with a fixed, explicit KV cache size via `--kv-cache-memory-bytes`, while
keeping everything else identical to Attempt 11A (EP disabled,
`GPU_MEMORY_UTILIZATION=0.85`, Ray safety knobs, `--enforce-eager`,
`MAX_MODEL_LEN=8192`/`MAX_NUM_SEQS=4`/`MAX_NUM_BATCHED_TOKENS=8192`, sudo
`trace-memory.sh`/`memory-guard.sh` on both nodes, clean reboot
beforehand).

**Env used:**
[`.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug)
-- identical to
[`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug)
(Attempt 11A) plus one added arg:

```
--kv-cache-memory-bytes 8589934592   (= 8 GiB)
```

`--kv-cache-memory-bytes` support was confirmed via `vllm serve
--help=CacheConfig` (not listed in the default top-level `--help`; vLLM
0.22 groups CLI args by config class and requires `--help=<GroupName>`).

**Timeline (UTC):**

| Event | Time |
|---|---|
| Weight loading started | 15:39:21 |
| Weight loading 14/14 completed, head/rank0 (141.5s) | 15:41:44 |
| Weight loading 14/14 completed, worker/rank1 (165.4s) | 15:42:10 |
| `Initial free memory ..., reserved 8.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling` (rank0) | 15:43:39 |
| Same log line (rank1) | 15:44:38 |
| `GPU KV cache size: 174,504 tokens` (3 padding layers, ~9.09% waste) | 15:44:39 |
| `Application startup complete` (API server ready) | 15:46:21 |
| Inference test (`/v1/completions`) succeeded | ~15:48 |

Weight loading (2m21s on rank0) was comparable to Attempt 11A's 2m17s.
`Application startup complete` was reached **4m37s after weight-loading
completion** (15:41:44 -> 15:46:21), notably faster than Attempt 11A's
~5m50s (15:03:10 -> 15:09:00).

**Key log lines confirming the fixed-KV / skip-profiling path:**

```
INFO 06-12 15:43:39 [gpu_worker.py:387] Initial free memory 111.93 GiB,
reserved 8.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes
config and skipped memory profiling. This does not respect the
gpu_memory_utilization config. ...

INFO 06-12 15:44:39 [kv_cache_utils.py:1733] GPU KV cache size: 174,504 tokens
INFO 06-12 15:44:39 [kv_cache_utils.py:1734] Maximum concurrency for 8,192
tokens per request: 21.30x
```

No `Available KV cache memory: X GiB` line was printed at all in this run
-- the `kv_cache_memory_bytes` path **skips the memory-profiling step
entirely**, which is the step that produced that line in Attempt 11A.

**Memory behavior -- decline shrank substantially and became more
symmetric.**

| Time (UTC) | spark01/head MemAvailable | spark02/worker MemAvailable |
|---|---|---|
| 15:41:13 (mid weight-load, 9/14) | ~54.85 GiB | ~50.34 GiB |
| 15:45:05 (post KV-size determination) | ~42.30 GiB | ~42.40 GiB |
| 15:48:05 (post `Application startup complete`) | ~40.78 GiB | ~43.43 GiB |
| 15:49:43 | ~40.17 GiB | ~42.87 GiB |
| 15:54:05 (final, stable) | **40.13 GiB** | **42.85 GiB** |

spark01/head: ~54.85 GiB -> ~40.13 GiB, **~14.7 GB total** -- vs. Attempt
11A's ~38 GB. spark02/worker: ~50.34 GiB -> ~42.85 GiB, **~7.5 GB total**
-- vs. Attempt 11A's essentially-flat behavior. Both nodes plateaued by
15:54:05 (< 50 MB drift over the prior 4+ minutes).

**Comparison with Attempt 11A:**

| | Attempt 11A (auto KV) | Attempt 12 (fixed KV 8GiB) |
|---|---|---|
| KV cache memory determination | profiled (`Available KV cache memory: 37.8 GiB`) | skipped (fixed, no profiling log) |
| `GPU KV cache size` | 790,438 tokens | 174,504 tokens |
| head (spark01) decline | ~38 GB (51.9 -> 13.86 GiB) | ~14.7 GB (54.85 -> 40.13 GiB) |
| worker (spark02) decline | ~flat (~17-17.8 GiB range) | ~7.5 GB (50.34 -> 42.85 GiB) |
| weight-load -> startup-complete | ~5m50s | ~4m37s |
| Ray OOM / guard trip / host freeze | none | none |

**Interpretation.**

- **KV cache memory profiling / automatic KV allocation is now the
  strongest root-cause candidate** for the head-node-only ~38GB decline
  identified in Attempts 09/10b/11A. Replacing the profiled/derived KV
  size (37.8 GiB, 790,438 tokens) with a small fixed value (8 GiB,
  174,504 tokens) and skipping profiling entirely cut the head-node
  decline by more than half (~38GB -> ~14.7GB).
- **EP/all-to-all is confirmed not to be the sole cause** (already shown
  in §14; reaffirmed here since EP remains off).
- **Reducing `max_model_len`/`max_num_seqs`/`max_num_batched_tokens` alone
  (Attempt 10b's low-kv variant) was insufficient** to prevent the
  decline/collapse, but **directly fixing
  `--kv-cache-memory-bytes`** -- which changes *how* vLLM sizes and
  allocates the KV cache, not just the logical token budget -- was
  effective. This points at the KV-cache *allocation/profiling
  mechanism* itself (not merely the resulting cache size) as a major
  contributor.
- **The decline pattern changed from head-only to more symmetric
  head/worker**: head still declines somewhat more (~14.7GB vs ~7.5GB),
  but worker is no longer flat. This suggests the profiled-KV-allocation
  path had a head/rank0-specific extra cost on top of a smaller
  baseline cost shared by both ranks; with profiling skipped, mostly the
  shared baseline cost remains.
- **The remaining ~14.7GB head decline is not fully explained by
  RayWorkerWrapper + EngineCore RSS alone** (combined ~4.3 GiB, see
  below). It should be treated as baseline engine/Ray/CUDA/model-load
  overhead (e.g. CUDA context/driver allocations, Ray object store,
  other Ray subprocess RSS, page-cache effects from the 120GB checkpoint
  read) rather than a single identifiable allocation -- further isolation
  would require per-process accounting across *all* container processes,
  not just EngineCore/RayWorkerWrapper.

**proc_maps / smaps results (sudo capture succeeded).** Unlike Attempt
11A (§14, all 0 bytes), running `trace-memory.sh` with `sudo` produced
non-empty `smaps_rollup`/`maps_summary`/`numa_maps` for root-owned
container processes on both nodes:

| Process | Rss | Pss | Swap |
|---|---|---|---|
| `VLLM::EngineCore` (spark01/head) | 664.6 MB | 595.5 MB | 246 MB |
| `ray::RayWorkerWrapper` rank0 (spark01/head) | 3.66 GB | 3.59 GB | 676 MB |
| `ray::RayWorkerWrapper` rank1 (spark02/worker) | 3.86 GB | 3.82 GB | 509 MB |

`maps_summary` mapping-category breakdown for `RayWorkerWrapper`:

| Category | rank0 (head) | rank1 (worker) |
|---|---|---|
| `anonymous` | count=855, ~1.8 TB | count=851, ~839 GB |
| `cuda_nccl_lib` | ~645 MB | ~645 MB |
| `deleted` (memfd/unlinked-lib-like) | ~4.92 GB | ~4.92 GB |
| `nvidia_dev` | ~200 MB | ~200 MB |
| `infiniband_dev` | -- | 24 KB |
| `dev_shm` | 80 KB | 8 KB |
| `other` | ~5.75 GB | ~5.74 GB |

**The dominant `anonymous` category (~1.8 TB rank0 / ~839 GB rank1) is
virtual address space only, not physical memory** -- `smaps_rollup`'s
`Rss`/`Pss` for the same processes are only ~3.7-3.9 GB. This is
consistent with §11.2's earlier VSZ=790.6GiB finding for
`RayWorkerWrapper` and is most likely CUDA UVM / `expandable_segments`
virtual-address-space over-reservation, which does not by itself explain
the GB-scale `MemAvailable` decline. The actually-resident contributors
(EngineCore + RayWorkerWrapper RSS, ~4.3 GiB combined on head) account for
only a fraction of the ~14.7GB head decline; the rest remains
unattributed to a single process from this data and likely reflects
distributed overhead across the full container process tree plus
page-cache/driver-level effects.

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt12-20260613-005641.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt12-20260613-005641.tar.gz`

Both containers were stopped cleanly (`docker compose ... down`, exit code
0) and `trace-memory.sh`/`memory-guard.sh` were stopped with `SIGTERM` on
both nodes (no orphaned processes).

## 17. Attempt 13 results (2026-06-12) -- EP-on + fixed KV cache re-isolation

**Purpose:** Attempt 12 (§16) showed that `--kv-cache-memory-bytes` (fixed
KV, profiling skipped) cuts the head-node decline from ~38GB to ~14.7GB
with EP **disabled**. Attempt 13 re-enables EP while keeping fixed KV
8GiB, to determine whether fixed KV alone is sufficient to prevent the
original EP-on host-freeze/Ray-OOM failure (§1), or whether EP-on
introduces an additional, independent memory-pressure path.

**Env used:**
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug)
-- identical to Attempt 12's
[`.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug)
plus `--enable-expert-parallel` re-enabled. Configuration:

- `--enable-expert-parallel` (EP on)
- `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `RAY_memory_usage_threshold=0.90`
- `RAY_memory_monitor_refresh_ms=100`
- `memory-guard.sh --threshold-mb 8192`
- `trace-memory.sh` run with `sudo` (proc_maps/smaps capture enabled, per
  §15) on both nodes

**Timeline (UTC):**

| Event | Time |
|---|---|
| Weight loading started | 16:14:04 |
| `[EP Rank 0/2] ... Local/global number of experts: 144/288` (spark01) | during startup, before weight loading |
| `[EP Rank 1/2] ... Local/global number of experts: 144/288` (spark02) | during startup, before weight loading |
| Weight loading 14/14 completed (`Loading weights took 343.02 seconds`) | 16:19:49 |
| `Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['VLLM_CUTLASS', 'MARLIN', 'EMULATION']` | 16:19:51 |
| `Using MoEPrepareAndFinalizeNoDPEPModular` | 16:19:51 |
| Ray OOM: RayWorkerWrapper rank0 killed | 16:19:58 |
| `EngineCore failed to start` / `Engine core initialization failed` | 16:19:58 |

EP markers were correct and identical in shape to Attempt 13's pre-crash
state: `[EP Rank 0/2] Expert parallelism is enabled. Expert placement
strategy: linear. Local/global number of experts: 144/288` (rank0,
experts 0-143) and `[EP Rank 1/2] ... 144/288` (rank1, experts 144-287).
`init_device` passed on both ranks. Weight loading reached 14/14 on both
ranks (343.02s on rank0/head).

**The run crashed before reaching any KV-cache-related log line.** Neither
`Initial free memory ..., reserved 8.0 GiB memory for KV Cache ... skipped
memory profiling` (the fixed-KV/skip-profiling log from Attempt 12) nor
`GPU KV cache size: ...` ever appeared. The failure occurred entirely
within the **9-second window between weight-loading completion (16:19:49)
and Ray OOM (16:19:58)**, during MoE backend setup.

**Ray OOM detail:**

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 10.10.10.1, ID: d47bda435b751d12bfe0a3f4509bd80e7e17b0f1b15bfad1397704f5) was
109.73GB / 121.63GB (0.902205), which exceeds the memory usage threshold of 0.900000.
... Actor(6d86283d43fe50bdad0bbab101000000) pid=1879, actual memory used=0.44GB ...
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

spark02 (worker) lost its Ray GCS connection immediately after:
`[rank1]:[W612 16:19:58...] TCPStore.cpp:125 [c10d] recvValue failed ...
Connection was likely closed. Did the remote server shutdown or crash?`

**Memory behavior -- sharp, head-only collapse within ~10 seconds:**

| Time (UTC) | spark01/head MemAvailable | spark02/worker MemAvailable |
|---|---|---|
| 16:14:00 (weight loading start) | ~113.2 GiB (115884 MB) | ~116.7 GiB (119541 MB) |
| 16:19:40 (mid-MoE-setup) | ~48.6 GiB (49755 MB) | ~54.2 GiB (55500 MB) |
| 16:19:50 (just after weight-load-complete) | ~49.1 GiB (50288 MB) | ~52.7 GiB (54013 MB) |
| 16:20:00 (post-OOM) | **~13.6 GiB (13915 MB)** | ~58.2 GiB (59561 MB) |
| 16:20:30 (settled) | ~14.0 GiB (14379 MB) | ~58.1 GiB (59525 MB) |

spark01/head dropped from ~49.1 GiB to ~13.6 GiB available **within 10
seconds** (16:19:50 -> 16:20:00) -- a ~35.5 GB collapse, comparable in
both speed and shape to the original §1 head-only decline pattern, but
occurring during MoE backend init rather than KV-cache profiling.
spark02/worker stayed flat-to-improving throughout (~52.7 -> ~58.2 GiB),
i.e. **no symmetric decline this time** -- unlike Attempt 12 where both
nodes declined.

`memory-guard.sh` (threshold 8192 MB) did **not** trip on either node:
spark01's available recovered to ~14.0-14.4 GB after Ray's kill, never
crossing the 8192 MB guard threshold. **Ray's memory monitor intervened
first**, at 90.22% usage -- almost identical to Attempts 09/10b's ~90.77%.

No host freeze occurred; SSH remained fully responsive on both nodes
throughout.

**smaps evidence (RayWorkerWrapper rank0/rank1, sudo `trace-memory.sh`
high-frequency burst capture 16:19:15-16:19:58):**

| Process | Time (UTC) | Rss | Anonymous | Private_Dirty | Swap |
|---|---|---|---|---|---|
| rank0 (spark01, pid 44471) | 16:19:15.996 (pre weight-load-complete) | ~5.77 GiB | ~4.69 GiB | ~5.20 GiB | 0 |
| rank0 (spark01, pid 44471) | 16:19:58.039 (kill in progress) | ~0.49 GiB | ~0.31 GiB | ~0.41 GiB | ~1.67 GiB |
| rank1 (spark02, pid 11340) | 16:19:49.053 | ~4.74 GiB | ~3.96 GiB | ~4.47 GiB | 0 |
| rank1 (spark02, pid 11340) | 16:19:58.581 (last sample, survivor) | ~6.17 GiB | ~5.32 GiB | ~5.83 GiB | 0 |

Both `RayWorkerWrapper` processes (rank0 on spark01, rank1 on spark02)
ballooned to **~5-6.5 GiB RSS, predominantly `Anonymous`/`Private_Dirty`**
(i.e. real, resident physical memory -- not virtual-only) within the
~9-43 second window spanning weight-load-complete to Ray OOM. rank0's
final sample shows the process mid-kill (RSS collapsing, partially
swapped out); rank1's last sample (the surviving rank) shows the peak,
~6.17 GiB.

**maps_summary mapping-category comparison (rank0 vs rank1, near OOM):**

| Category | rank0 (spark01) @ 16:19:58.039 | rank1 (spark02) @ 16:19:58.581 |
|---|---|---|
| `nvidia_dev` | 67 mappings, ~108 MB | 66 mappings, ~106 MB |
| `infiniband_dev` | 3 mappings, 12 KB | 3 mappings, 12 KB |
| `cuda_nccl_lib` | 32 mappings, ~645 MB | 32 mappings, ~645 MB |
| `anonymous` (VSZ) | ~819 GB | ~602 GB |
| `deleted` | 61, ~4.75 GB | 61, ~4.75 GB |

No meaningful rank0/rank1 asymmetry in `/dev/nvidia*`, `/dev/infiniband*`,
or CUDA/NCCL library mappings. The `anonymous` VSZ figures (~600-819 GB)
remain far larger than actual RSS (~0.5-6.5 GiB) and should **not** be
treated as physical memory usage -- consistent with §16's
`expandable_segments`/CUDA-UVM virtual-reservation finding.

**Interpretation.**

- **Fixed KV cache (Attempt 12's fix) is necessary but not sufficient**
  with EP enabled. It still solves/reduces the *automatic KV-cache
  profiling* memory-pressure path (the run never even reached that stage
  here, so that path cannot have contributed to this failure), but
  **EP-on introduces a separate, independent memory spike during MoE
  backend initialization** that fixed KV does not address.
- The observed log context at the moment of the spike --
  `Using 'MARLIN' NvFp4 MoE backend ...` /
  `Using MoEPrepareAndFinalizeNoDPEPModular` -- points at **MARLIN NVFP4
  MoE weight preparation/repacking or backend workspace initialization**
  as the proximate trigger. This is phrased deliberately as "weight
  preparation/repacking or backend workspace initialization" rather than
  asserting that expert-routing workspace allocation alone is the proven
  cause; the exact sub-step within MARLIN NVFP4 MoE setup that allocates
  ~5-6 GiB per rank has not been isolated further.
- **The memory spike occurs on both ranks** (rank0 ~5.77 GiB, rank1 ~6.17
  GiB peak RSS, both predominantly Anonymous/Private_Dirty) -- this is not
  a head/rank0-only phenomenon at the process level.
- **spark01 (head) crosses the Ray 0.90 threshold first and alone**
  because it carries lower baseline headroom: in addition to its
  RayWorkerWrapper (rank0), it also hosts the Ray head/GCS/dashboard, the
  EngineCore, and the OpenAI-compatible API server. spark02 (worker) has
  only its RayWorkerWrapper (rank1) and raylet, so the same ~6 GiB spike
  there does not push it over 90% of 121.63 GiB.
- **Ray's memory monitor (threshold 0.90) is confirmed as the operative
  host-freeze safety guard** for this failure mode: it intervened (killed
  rank0's RayWorkerWrapper, 90.22%) well before `memory-guard.sh`'s 8192
  MB threshold was reached (spark01 available bottomed at ~13.6 GiB =
  ~13,915 MB, recovering to ~14.0-14.4 GB) and well before any host
  freeze. This is consistent with -- and reaffirms -- Attempts 09/10b's
  ~90.77% Ray-OOM ratio as the same underlying failure mode.
- **This separates two previously-conflated memory-pressure paths**:
  (1) automatic KV-cache profiling/allocation (addressed by Attempt 12's
  fixed KV), and (2) EP-on MoE backend initialization (not addressed by
  fixed KV, and not yet isolated to a specific backend/code path beyond
  "MARLIN NVFP4 MoE setup"). The next isolation step should vary the MoE
  backend alone (EP-on, fixed KV 8GiB, all else unchanged) to test whether
  path (2) is specific to the MARLIN NVFP4 MoE backend.

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt13-20260613-082329.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt13-20260613-082331.tar.gz`

The head container exited with an error (`EngineCore failed to start`,
Ray OOM); the worker container remained "Up" but disconnected from the
head's GCS (TCPStore recv errors) and was torn down cleanly afterward
(`docker compose ... down`, exit code 0). `trace-memory.sh`/
`memory-guard.sh` were stopped with `SIGTERM` on both nodes (no orphaned
processes).

## 18. Attempt 14A results (2026-06-13) -- MoE backend isolation (MARLIN -> VLLM_CUTLASS)

**Purpose:** Attempt 13 (§17) crashed via Ray OOM ~9 seconds after weight
loading completed, during MoE backend initialization with the
auto-selected `MARLIN` NVFP4 MoE backend (`Using 'MARLIN' NvFp4 MoE
backend out of potential backends: ['VLLM_CUTLASS', 'MARLIN',
'EMULATION']`). Attempt 14A changes **only** the MoE backend selection,
from auto-selected `MARLIN` to explicitly-requested `VLLM_CUTLASS` (the
backend listed first in Attempt 13's "potential backends", which MARLIN
was auto-selected over), to determine whether this single-variable
change alters the EP-on post-weight-load RSS spike. All other Attempt 13
conditions are kept unchanged.

**Env used:**
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-cutlass-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-cutlass-debug)
-- identical to Attempt 13's
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug)
plus `--moe-backend cutlass` added to `VLLM_EXTRA_ARGS`. Configuration:

- `--enable-expert-parallel` (EP on)
- `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- 144/288 experts per rank (rank0, rank1)
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`
- `--kv-cache-dtype fp8`
- `--quantization modelopt`
- `RAY_memory_usage_threshold=0.90`
- `RAY_memory_monitor_refresh_ms=100`
- `memory-guard.sh --threshold-mb 8192`
- `trace-memory.sh` run with `sudo` (proc_maps/smaps capture enabled, per
  §15) on both nodes
- Only intended change vs. Attempt 13: `--moe-backend cutlass` added to
  `VLLM_EXTRA_ARGS` (maps to `NvFp4MoeBackend.VLLM_CUTLASS` per
  `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:
  map_nvfp4_backend["cutlass"]`)

**Result:**

- CLI parsing succeeded; the request was mapped to
  `NvFp4MoeBackend.VLLM_CUTLASS`.
- EP initialized correctly and identically to Attempt 13 on both ranks:
  `[EP Rank 0/2] ... Local/global number of experts: 144/288` (spark01)
  and `[EP Rank 1/2] ... Local/global number of experts: 144/288`
  (spark02).
- The `VLLM_CUTLASS` backend was rejected by
  `select_nvfp4_moe_backend()` **before weight loading started** -- the
  run never reached weight loading, MoE weight preparation/repacking,
  fixed-KV reservation, GPU KV cache creation, or application startup.
- No silent fallback to `MARLIN` occurred; both ranks raised a
  `ValueError` and exited cleanly with code 1.
- Failure occurred ~26 seconds after container start.
- No Ray OOM, no `memory-guard.sh` trip, no host freeze, no SSH
  interruption. Available memory remained ~115 GB on both nodes
  throughout.

**Root-cause excerpt (verbatim from `vllm-spark-head` log):**

```
ray.exceptions.RayTaskError(ValueError): ray::RayWorkerWrapper.execute_method() (pid=1880, ip=10.10.10.1, ...)
  File ".../fused_moe/oracle/nvfp4.py", line 256, in select_nvfp4_moe_backend
    raise ValueError(_make_log_unsupported(backend, reason))
ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support the deployment configuration since kernel does not support parallel config FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=1, ep_size=2, tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=True, all2all_backend='allgather_reducescatter', enable_eplb=False).
```

**Interpretation.**

- This result does **not** establish anything about memory. CUTLASS did
  not "fail due to memory", and it did not "worsen" or "improve" the
  Attempt 13 MoE initialization RSS spike -- the run never reached the
  initialization stage where that spike occurred in Attempt 13.
- The result only establishes that, in the current vLLM build,
  `VLLM_CUTLASS`'s static `is_supported_config()` check rejects the
  deployment configuration produced by this EP setup
  (`ep_size=2, use_ep=True, all2all_backend='allgather_reducescatter'`),
  independent of memory pressure.
- **Attempt 13 remains the valid MARLIN EP-on memory result** -- it is
  the only attempt so far to reach the post-weight-load MoE
  initialization stage with EP enabled.
- A different MoE backend must first pass this static
  deployment-configuration validation before a meaningful memory
  comparison against Attempt 13 is possible. Backend candidates must be
  checked against `is_supported_config()` requirements *before* a
  memory-isolation run is attempted (see §19).

**Attempt 13 vs. Attempt 14A comparison:**

| | Attempt 13 (MARLIN) | Attempt 14A (VLLM_CUTLASS) |
|---|---|---|
| Backend | `MARLIN` (auto-selected) | `VLLM_CUTLASS` (explicitly requested via `--moe-backend cutlass`) |
| EP markers (144/288, both ranks) | reached | reached |
| Weight loading | reached, 14/14 complete (343.02s) | not reached |
| Failure point | ~9s after weight-load completion, during MoE backend init (`Using 'MARLIN' NvFp4 MoE backend ...` / `Using MoEPrepareAndFinalizeNoDPEPModular`) | during MoE backend *selection*, before weight loading (`select_nvfp4_moe_backend` raises `ValueError`) |
| Failure mode | Ray OOM (memory monitor kill, ratio 0.902205) | clean `ValueError`, both ranks exit code 1 |
| Per-rank RSS spike | ~5.77-6.17 GiB (Anonymous/Private_Dirty) | none observed |
| Available memory | dropped to ~13.6 GiB (spark01) | remained ~115 GB (both nodes) |
| memory-guard / host freeze | not reached (Ray monitor intervened first) | no trip, no freeze |
| Valid for memory comparison | yes (only attempt to reach this stage with EP on) | no -- never reached the comparable stage |

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt14a-20260613-093427.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt14a-20260613-093428.tar.gz`

The head container exited with an error (`ValueError` /
`Engine core initialization failed`, code 1); the worker container
exited cleanly after the head's failure. `trace-memory.sh`/
`memory-guard.sh` were stopped with `SIGTERM` on both nodes (no orphaned
processes). No reboot was required before or after this attempt.

## 19. Attempt 14B results (2026-06-13) -- MoE backend isolation (MARLIN -> FLASHINFER_CUTLASS)

**Purpose:** Attempt 14A (§18) showed that `VLLM_CUTLASS` is rejected by
`select_nvfp4_moe_backend()` **before** weight loading because its
`is_supported_config()` requires `ep_size == 1`. Based on static source
inspection of vLLM 0.22.1 / FlashInfer 0.6.12
(`vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`,
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`), `FlashInferExperts`
(`NvFp4MoeBackend.FLASHINFER_CUTLASS`) appeared to pass all seven
`is_supported_config()` checks for this deployment (device family 120,
NVFP4 static/dynamic quant scheme, SILU activation, unconditional parallel-
config support, `allgather_reducescatter` activation format). Attempt 14B
replaces `MARLIN` with `FLASHINFER_CUTLASS` while keeping every other
Attempt 13 setting unchanged, to determine whether `FLASHINFER_CUTLASS`
can support EP=2 and reduce (or otherwise change) the post-weight-load MoE
initialization RSS spike that caused Attempt 13's Ray OOM.

**Env used:**
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-flashinfer-cutlass-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-flashinfer-cutlass-debug)
-- identical to Attempt 13's
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug)
plus `--moe-backend flashinfer_cutlass` added to `VLLM_EXTRA_ARGS`.
Configuration:

- `--enable-expert-parallel` (EP on), `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`, `--kv-cache-dtype fp8`, `--quantization modelopt`
- `RAY_memory_usage_threshold=0.90`, `RAY_memory_monitor_refresh_ms=100`
- `memory-guard.sh --threshold-mb 8192 --interval 0.1`
- `trace-memory.sh --duration 1800` run with `sudo` (proc_maps/smaps
  capture, per §15) on both nodes
- Only intended change vs. Attempt 13: `--moe-backend flashinfer_cutlass`
  added to `VLLM_EXTRA_ARGS` (maps to `NvFp4MoeBackend.FLASHINFER_CUTLASS`)

**Timeline (UTC):**

| Time | Event |
|---|---|
| 02:02:50 | worker (spark02) container started |
| 02:03:05 | head (spark01) container started (+15s) |
| 02:03:22 | Ray cluster connected (`Ray runtime started`) |
| 02:03:45 | EngineCore initialized (v0.22.1) |
| 02:04:12 | EP rank0/rank1 expert maps completed -- 144/288 experts per rank, linear placement (both ranks); `FLASHINFER` attention backend selected; `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION']` logged on **both** ranks |
| 02:04:14 | FusedMoE layer construction rejected `MoEActivation.SWIGLUSTEP`; both ranks raised `ValueError` and exited |

**Result:**

- `FLASHINFER_CUTLASS` **passed** the initial backend/device/parallel-config
  selection -- both ranks logged `Using 'FLASHINFER_CUTLASS' NvFp4 MoE
  backend out of potential backends: [...]` at 02:04:12.
- EP initialized correctly and identically to Attempt 13 on both ranks:
  `[EP Rank 0/2] ... Local/global number of experts: 144/288` (spark01) and
  `[EP Rank 1/2] ... Local/global number of experts: 144/288` (spark02),
  linear expert placement.
- No silent fallback to `MARLIN` occurred.
- During **actual `FusedMoE` layer construction** (`modelopt.py:1402`,
  `FusedMoEMethodCls.__init__`),
  `select_nvfp4_moe_backend()` was called **a second time**, this time with
  the model's resolved MoE activation. Step-3.7's actual activation is
  `MoEActivation.SWIGLUSTEP` (a Step3-specific SwiGLU variant), **not**
  `MoEActivation.SILU`. `FlashInferExperts._supports_activation()` does not
  include `SWIGLUSTEP`, so this second call raised `ValueError` on both
  ranks and the engine core crashed.
- Failure occurred ~84 seconds after container start, **before weight
  loading started** -- the run never reached `model_loader.load_model`,
  fixed-KV reservation, GPU KV cache creation, or application startup.
- No Ray OOM, no `memory-guard.sh` trip (threshold 8192MB, MemAvailable
  never dropped below ~61 GB), no host freeze, no SSH interruption.

**Root-cause excerpt (verbatim from `vllm-spark-head` log):**

```
(EngineCore pid=755) [RayWorkerWrapper pid=1880] INFO 06-13 02:04:12 [nvfp4.py:231] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION'].
...
(EngineCore pid=755) ERROR 06-13 02:04:14 [core.py:1165]     self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
(EngineCore pid=755) ERROR 06-13 02:04:14 [core.py:1165]   File ".../fused_moe/oracle/nvfp4.py", line 256, in select_nvfp4_moe_backend
(EngineCore pid=755) ERROR 06-13 02:04:14 [core.py:1165] ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment configuration since kernel does not support MoEActivation.SWIGLUSTEP activation.
```

**Two-stage NVFP4 MoE backend selection.**

`select_nvfp4_moe_backend()` is called twice in this code path:

1. **First call** (logged at `nvfp4.py:231`, 02:04:12) -- evaluated with the
   device, quantization scheme, parallel-config, and activation-format
   information available at that stage. `FLASHINFER_CUTLASS` passed and was
   logged as selected for both ranks.
2. **Second call** (from `modelopt.py:1402`, inside `FusedMoEMethodCls.__init__`
   during `FusedMoE` layer construction, 02:04:14) -- evaluated with the
   model's actual resolved `MoEActivation`. This call rejected
   `FLASHINFER_CUTLASS` because `MoEActivation.SWIGLUSTEP` is not in its
   supported-activation list.

**Interpretation.**

- Attempt 14B is classified as **backend runtime/configuration
  incompatibility**, the same broad category as Attempt 14A (§18), but for
  a different reason: Attempt 14A failed the *parallel-config* check
  (`ep_size == 1` required); Attempt 14B failed the *activation* check
  (`SWIGLUSTEP` unsupported).
- This is **not** a memory-comparison result. `FLASHINFER_CUTLASS` supports
  this device (SM121/family 120), this EP=2 configuration, and the
  `allgather_reducescatter` activation format -- but not this model's MoE
  activation function. It cannot replace `MARLIN` for Step-3.7-Flash in the
  current vLLM/FlashInfer build.
- The Attempt 14B env header's static preflight assumption -- that Step-3.7's
  `hidden_act="silu"` implies `MoEActivation.SILU` -- was **incorrect**.
  Step-3.7 resolves to `MoEActivation.SWIGLUSTEP`, a distinct enum value
  that the static `is_supported_config()` inspection (which only checked the
  first-call code path) did not surface.
- No silent fallback occurred; both ranks raised an explicit `ValueError`
  and exited cleanly with code 1.
- The run never reached weight loading or the post-load MoE preparation
  path, so **Attempt 13 remains the only valid MARLIN EP-on memory result**.
- **Separate observation (not a memory-comparison result):** despite failing
  before weight loading, host `MemAvailable` fell substantially during the
  ~84s the run was up -- spark01 from ~123.5 GB to ~62.6 GB (-56.6 GiB),
  spark02 from ~123.8 GB to ~73.8 GB (-45.9 GiB) -- while all host process
  RSS values remained below ~300 MB. This is consistent with CUDA
  context / driver / UMA-style allocation from TP=2 initialization
  (FlashInfer attention backend init, EP expert-map setup) occurring before
  the MoE activation check failed, but driver-level memory accounting was
  not captured in this run, so this should **not** be stated as a
  conclusively proven NVIDIA UVM allocation. This memory loss is unrelated
  to, and must not be confused with, Attempt 13's per-rank
  `RayWorkerWrapper` Anonymous/Private_Dirty RSS spike (~5.77-6.17 GiB),
  which was a *process*-RSS phenomenon occurring *after* weight loading.
  Because container teardown may not reclaim this allocation, **both nodes
  should be rebooted before any subsequent full run.**

**Attempt 13 vs. Attempt 14B comparison:**

| | Attempt 13 (MARLIN) | Attempt 14B (FLASHINFER_CUTLASS) |
|---|---|---|
| Backend | `MARLIN` (auto-selected) | `FLASHINFER_CUTLASS` (explicit, `--moe-backend flashinfer_cutlass`) |
| EP markers (144/288, both ranks) | reached | reached |
| First-stage backend selection | `Using 'MARLIN' NvFp4 MoE backend ...` (after weight load) | `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend ...` (before weight load) -- **passed** |
| `MoEActivation.SWIGLUSTEP` | accepted | **rejected** (second-stage check) |
| Weight loading | reached, 14/14 complete (343.02s) | not reached |
| Failure point | ~9s after weight-load completion, during MoE backend init (`Using MoEPrepareAndFinalizeNoDPEPModular`) | during `FusedMoE` construction's second-stage activation check, ~84s after start, before weight loading |
| Failure mode | Ray OOM (memory monitor kill, ratio 0.902205) | clean `ValueError`, both ranks exit code 1 |
| Per-rank process RSS spike | ~5.77-6.17 GiB (Anonymous/Private_Dirty) | none observed (all processes <300 MB) |
| Host MemAvailable | dropped to ~13.6 GiB (spark01) | spark01 -56.6 GiB, spark02 -45.9 GiB (non-RSS, UMA/driver-side) |
| memory-guard / host freeze | not reached (Ray monitor intervened first) | no trip, no freeze |
| Valid for post-weight-load MoE memory comparison | yes (only attempt to reach this stage with EP on) | no -- different failure stage, non-comparable memory signature |

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt14b-20260613T021158Z.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt14b-20260613T021159Z.tar.gz`

The head container exited with an error (`ValueError` / `Engine core
initialization failed`, code 1); the worker container remained `Up` until
`docker compose down` (Ray `--block` did not detect the head's crash).
`trace-memory.sh`/`memory-guard.sh` were stopped with `SIGTERM` on both
nodes (no orphaned processes). **Reboot recommended before the next run**
due to the ~56.6 GiB / ~45.9 GiB MemAvailable loss described above.

## 20. Attempt 15A results (2026-06-13) -- Ray object-store margin isolation (4GiB -> 1GiB)

**Purpose:** Attempt 13 (§17) crashed via Ray OOM at usage ratio 0.902205,
exceeding the `RAY_memory_usage_threshold=0.90` by only ~0.002205
(~0.27 GiB-equivalent of 121.63 GiB). Backend substitution (Attempts 14A/14B,
§18-19) showed `MARLIN` is the only NvFp4 MoE backend compatible with this
deployment's device family, quantization scheme, EP=2 parallel config, and
`MoEActivation.SWIGLUSTEP` activation -- so the Attempt 13 failure must be
addressed via memory accounting/margin, not backend choice. Attempt 15A
targets the Ray object store as the first margin candidate: Ray's
node-memory-usage ratio is computed against total node memory and may
include the object store's reserved share of the head node's baseline
footprint. Attempt 15A reduces `RAY_OBJECT_STORE_MEMORY_BYTES` from 4 GiB to
1 GiB to determine whether this reservation was responsible for Attempt 13
crossing the 0.90 threshold.

**Env used:**
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug)
-- identical to Attempt 13's
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug)
except for a single changed variable:

| Variable | Attempt 13 | Attempt 15A |
|---|---|---|
| `RAY_OBJECT_STORE_MEMORY_BYTES` | `4294967296` (4 GiB) | `1073741824` (1 GiB) |

All other relevant settings unchanged:

- `--enable-expert-parallel` (EP on), `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- `MARLIN` NvFp4 MoE backend (auto-selected, no explicit `--moe-backend`)
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`, `--kv-cache-dtype fp8`, `--quantization modelopt`
- `RAY_memory_usage_threshold=0.90` (kept at the proven host-freeze safety
  boundary -- not raised)
- `RAY_memory_monitor_refresh_ms=100`, `RAY_INCLUDE_DASHBOARD=false`
- `memory-guard.sh --threshold-mb 8192`
- `trace-memory.sh` run with `sudo` (proc_maps/smaps capture) on both nodes
- both nodes rebooted to a clean baseline before this run (per Attempt 14B's
  recommendation), confirmed `MemAvailable` ~123.5 GB (spark01) / ~123.8 GB
  (spark02) pre-run

**Object-store verification (Phase 6):** the entrypoint's `ray start`
command on both head and worker included `--object-store-memory=1073741824`,
and `raylet.out` confirmed the requested value was actually applied:

```
Allowing the Plasma store to use up to 1.07374GB of memory.
create_and_mmap_buffer(1073741832, /dev/shm/plasmaXXXXXX)
```

This rules out an invalid/silently-rejected configuration -- the 4GiB -> 1GiB
change was correctly applied on both nodes.

**Timeline (UTC):**

| Event | Time |
|---|---|
| `[EP Rank 0/2] ... Local/global number of experts: 144/288` (spark01, rank0) | 02:40:35 |
| `Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['VLLM_CUTLASS', 'MARLIN', 'EMULATION']` (rank0) | 02:40:35 |
| `[EP Rank 1/2] ... Local/global number of experts: 144/288` (spark02, rank1) | 02:40:36 |
| `Using 'MARLIN' NvFp4 MoE backend ...` (rank1) | 02:40:36 |
| Weight loading 14/14 completed (`Loading weights took 343.74 seconds`) | 02:46:20 |
| `Using MoEPrepareAndFinalizeNoDPEPModular` (rank0) | 02:46:22 |
| Ray OOM (rank0 RayWorkerWrapper killed) | 02:46:33 |

EP markers, expert counts (144/288 per rank), and `MARLIN` selection were
identical in shape to Attempt 13. Weight loading reached 14/14 on both ranks
(343.74s on rank0/head, vs. 343.02s in Attempt 13). **The run again crashed
before reaching any KV-cache-related log line** -- the fixed-KV
skip-profiling log and `GPU KV cache size: ...` never appeared. The failure
occurred within the **13-second window between weight-loading completion
(02:46:20) and Ray OOM (02:46:33)**, ~11 seconds after
`MoEPrepareAndFinalizeNoDPEPModular` (02:46:22) -- i.e. during the same MoE
backend setup stage as Attempt 13.

**Ray OOM detail:**

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 10.10.10.1, ID: 56fdab6863b688fa2307e8b5538f334c0e9051cc4537817e99d5ba68) was
109.48GB / 121.63GB (0.900134), which exceeds the memory usage threshold of 0.900000.
... RayWorkerWrapper.__init__ pid=1880, actual memory used=0.37GB ...
Object store memory usage: [...] objects in use: 0; bytes in use: 0 [...]
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

spark02 (worker) immediately lost its TCPStore connection after rank0 was
killed:
`[rank1]:[W613 02:46:34...] TCPStore.cpp:125 [c10d] recvValue failed ...
Connection was likely closed. Did the remote server shutdown or crash?`

**Memory behavior:**

| Time (UTC) | spark01/head MemAvailable | spark02/worker MemAvailable |
|---|---|---|
| 11:38:58 (baseline, pre-run) | 123501820 kB (~117.78 GiB) | 123787144 kB (~118.03 GiB) |
| 11:43:59 (spark02's own weight-load RSS peak) | -- | **53358444 kB (~50.88 GiB)** |
| 11:46:33 (Ray OOM moment, spark01) | **12752732 kB (~12.16 GiB)** | -- |
| post-crash (settled) | ~15.28 GB | ~60.95 GB (~58.13 GiB) |

spark01/head dropped from ~117.8 GiB to ~12.16 GiB available by the moment of
Ray's OOM kill -- a collapse of comparable magnitude to Attempt 13's
~49.1 GiB -> ~13.6 GiB (Attempt 13's pre-collapse baseline was already lower
at the same relative point because Attempt 13 was not preceded by a fresh
reboot to the same ~123 GB baseline). spark02/worker's minimum
(~50.88 GiB, at 11:43:59) occurred *during its own weight-loading RSS spike*,
well before rank0's OOM at 11:46:33, and spark02 subsequently recovered to
~58.1 GiB -- matching Attempt 13's pattern of an asymmetric, head-only
collapse.

**proc_maps evidence (RayWorkerWrapper, sudo `trace-memory.sh`):**

| Process | Time (UTC) | RSS |
|---|---|---|
| rank0 (spark01, pid 1880) | 11:46:32.700 (OOM moment) | ~505220 KB (~0.48 GiB) |
| rank1 (spark02, pid 341) | 11:44:01.073 (own weight-load peak) | ~2792496 KB (~2.66 GiB) |

Unlike Attempt 13 (rank0 ~5.77 GiB, rank1 ~6.17 GiB peak RSS,
predominantly Anonymous/Private_Dirty), Attempt 15A's `RayWorkerWrapper`
process RSS values were substantially smaller (rank0 ~0.48 GiB at the OOM
moment, rank1 ~2.66 GiB at its own peak). Ray's own accounting of the killed
actor (`actual memory used=0.37GB`) is consistent with rank0's small
proc-level RSS. **The ~105 GiB difference between spark01's ~117.8 GiB
baseline and ~12.16 GiB at OOM is therefore not explained by
`RayWorkerWrapper` process RSS** -- consistent with §19's observation that
large non-RSS `MemAvailable` losses occur on this platform (UMA-resident
GPU weight/CUDA-context allocation), though driver-level attribution remains
unproven.

`memory-guard.sh` (threshold 8192 MB) did **not** trip on either node:
spark01's available bottomed at ~12.16 GiB (12752732 kB), never crossing the
8192 MB guard threshold. **Ray's memory monitor intervened first**, at
90.0134% -- essentially identical to Attempt 13's 90.2205% and
Attempts 09/10b's ~90.77%.

No host freeze occurred; SSH remained fully responsive on both nodes
throughout.

**Attempt 13 vs. Attempt 15A comparison:**

| | Attempt 13 (object store 4GiB) | Attempt 15A (object store 1GiB) |
|---|---|---|
| `RAY_OBJECT_STORE_MEMORY_BYTES` requested | 4294967296 | 1073741824 |
| Object store actual (raylet.out) | not measured | confirmed: 1.07374 GB |
| EP markers (144/288, both ranks) | reached | reached |
| `MARLIN` selection (both ranks) | reached | reached |
| Weight loading 14/14 | 343.02s | 343.74s |
| `MoEPrepareAndFinalizeNoDPEPModular` | reached (16:19:51) | reached (02:46:22) |
| Fixed-KV reservation | not reached | not reached |
| Time: weight-load-complete -> Ray OOM | ~9s | ~13s |
| **Ray node-memory ratio at OOM** | **0.902205** (109.73/121.63 GB) | **0.900134** (109.48/121.63 GB) |
| Object-store bytes-in-use at OOM | not measured | **0** |
| rank0 `RayWorkerWrapper` RSS (near OOM) | ~5.77 GiB (pre-kill) / ~0.49 GiB (mid-kill) | ~0.48 GiB (at OOM) |
| rank1 `RayWorkerWrapper` RSS (own peak) | ~6.17 GiB | ~2.66 GiB |
| spark01 minimum MemAvailable | ~13.6 GiB (13915 MB) | ~12.16 GiB (12752732 kB) |
| Ray OOM | yes, rank0 killed | yes, rank0 killed |
| rank1 TCPStore disconnect | yes | yes |
| memory-guard trip | no | no |
| Host freeze / SSH | none / responsive | none / responsive |

**Interpretation.**

- The 4GiB -> 1GiB object-store reduction was **correctly applied and
  verified** (raylet.out confirms 1.07374 GB allocated).
- The Ray node-memory-usage ratio decreased by only **~0.0021**
  (0.902205 -> 0.900134) -- roughly 8% of the 3 GiB reduction's
  theoretical proportional effect on a 121.63 GiB node
  (3 GiB / 121.63 GiB ≈ 0.0247).
- **Object-store bytes-in-use was zero at the moment of the OOM** in
  Attempt 15A. The object store's *reserved capacity* therefore does not
  appear to be counted toward Ray's "node memory usage" ratio in a way that
  tracks its configured size -- reducing the reservation did not free a
  proportional amount of "used" headroom.
- **Ray object-store allocation was not the dominant source of the MARLIN
  EP-on initialization pressure.** Attempt 15A failed via the same
  fundamental mechanism as Attempt 13: weight loading completes (14/14),
  `MoEPrepareAndFinalizeNoDPEPModular` is reached, and within ~10-15 seconds
  the head node's memory-usage ratio crosses 0.90, triggering Ray's memory
  monitor to kill rank0's `RayWorkerWrapper`, which breaks rank1's TCPStore
  connection and fails `EngineCore` initialization.
- This is classified as **insufficient margin** (not an invalid
  configuration, and not a different failure mode from Attempt 13).
- The remaining pressure that pushes the ratio from its pre-weight-load
  baseline (~117.8 GiB available) down to ~12.16 GiB available at the OOM
  point is, as in Attempt 13/§19, **primarily outside ordinary process RSS
  accounting** (rank0's `RayWorkerWrapper` RSS was only ~0.48 GiB at the OOM
  moment) and consistent with CUDA/UMA-side allocation accompanying MARLIN
  NVFP4 MoE weight preparation/repacking -- but this remains a
  correlation, not a conclusively proven driver-level attribution.

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt15a-20260613-025408.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt15a-20260613-025409.tar.gz`

The head container exited with an error (`EngineCore failed to start`,
Ray OOM, code 1); the worker container remained `Up` (disconnected from the
head's GCS, TCPStore recv errors) and was torn down cleanly afterward
(`docker compose ... down`, exit code 0). `trace-memory.sh`/
`memory-guard.sh` were stopped with `SIGTERM` on both nodes (no orphaned
processes).

## 21. Attempt 15B results (2026-06-13) -- Ray memory-threshold margin isolation (0.90 -> 0.905)

**Purpose:** Attempt 15A (§20) reduced `RAY_OBJECT_STORE_MEMORY_BYTES` from
4 GiB to 1 GiB on top of Attempt 13, but the Ray OOM recurred at 0.900134
(109.48GB / 121.63GB) -- still above the 0.90 threshold by only 0.000134,
with object-store bytes-in-use at 0. Object-store size was therefore ruled
out as the dominant lever. Attempt 15B isolates the remaining candidate
lever named in the Attempt 15A interpretation: the `RAY_memory_usage_threshold`
itself. This experiment raises the threshold by a small, fixed amount (0.90
-> 0.905, ~0.61 GiB-equivalent of 121.63 GiB) while keeping the 8192MB
`memory-guard.sh` hard safety boundary unchanged, to determine whether the
MARLIN MoE-init memory pressure is a brief transient peak that a small extra
margin absorbs, or a sustained growth that simply consumes the extra margin
and recurs.

**Env used:**
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-threshold0905-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-threshold0905-debug)
-- identical to Attempt 15A's
[`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug`](../../.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug)
except for a single changed variable:

| Variable | Attempt 15A | Attempt 15B |
|---|---|---|
| `RAY_memory_usage_threshold` | `0.90` | `0.905` |

All other relevant settings unchanged from Attempt 15A:

- `--enable-expert-parallel` (EP on), `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- `MARLIN` NvFp4 MoE backend (auto-selected, no explicit `--moe-backend`)
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`, `--kv-cache-dtype fp8`, `--quantization modelopt`
- `RAY_OBJECT_STORE_MEMORY_BYTES=1073741824` (1 GiB, verified actually
  applied in Attempt 15A)
- `RAY_memory_monitor_refresh_ms=100`, `RAY_INCLUDE_DASHBOARD=false`,
  `RAY_DASHBOARD_MAX_EVENTS_TO_CACHE=1000`
- `memory-guard.sh --threshold-mb 8192` (kept active as the hard safety
  boundary, did not trip)
- `trace-memory.sh` run with `sudo` (proc_maps/smaps capture) on both nodes
- both nodes rebooted to a clean baseline before this run, confirmed
  `MemAvailable` ~117.77 GiB (spark01, 123489232 kB) / ~118.00 GiB (spark02,
  123768420 kB) pre-run -- essentially identical to Attempt 15A's ~123.5 GB /
  ~123.8 GB baselines

**Threshold verification (Phase 6):** `raylet.out` on both nodes confirmed
the object-store size was unchanged from Attempt 15A and the memory-monitor
threshold was actually raised to 0.905:

```
[raylet] store_runner.cc:50: Allowing the Plasma store to use up to 1.07374GB of memory.
[raylet] threshold_memory_monitor.cc:52: MemoryMonitor initialized with usage threshold at 118189481984 bytes (90.500000% of system memory), total system memory bytes: 130596118528, monitor interval: 100ms
```

(spark02's raylet.out showed the equivalent line at 90.500000% against its
own total system memory of 130596839424 bytes.) This rules out a
silently-ignored env var -- the 0.90 -> 0.905 change was correctly applied on
both nodes.

**Timeline (UTC):**

| Event | Time |
|---|---|
| `[EP Rank 0/2] ... Local/global number of experts: 144/288` + `Using 'MARLIN' NvFp4 MoE backend ...` (spark01, rank0) | 07:30:52 |
| `[EP Rank 1/2] ... Local/global number of experts: 144/288` + `Using 'MARLIN' NvFp4 MoE backend ...` (spark02, rank1) | 07:30:53 |
| Weight loading 14/14 completed (`Loading weights took 344.10 seconds`) | 07:36:38 |
| `Using MoEPrepareAndFinalizeNoDPEPModular` (rank0) | 07:36:39 |
| Ray OOM (rank0 `RayWorkerWrapper` killed) | 07:37:02 |

EP markers, expert counts (144/288 per rank), and `MARLIN` selection were
identical in shape to Attempts 13 and 15A. Weight loading reached 14/14 on
both ranks (344.10s on rank0/head, vs. 343.74s in 15A and 343.02s in
Attempt 13 -- consistently ~343-344s). **The run again crashed before
reaching any KV-cache-related log line.** The failure occurred within the
**24-second window between weight-loading completion (07:36:38) and Ray OOM
(07:37:02)**, ~23 seconds after `MoEPrepareAndFinalizeNoDPEPModular`
(07:36:39) -- the same MoE backend setup stage as Attempts 13 and 15A, but
the window before OOM nearly doubled (13s -> 24s) compared to 15A.

**Ray OOM detail:**

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 10.10.10.1, ID: 68d3ec3e7fade9f61ebcca34d3acb3d566f9871887dafe3a97f59f10) was
110.53GB / 121.63GB (0.908750), which exceeds the memory usage threshold of 0.905000.
... RayWorkerWrapper.__init__ pid=1903, actual memory used=0.34GB ...
Object store memory usage: [...] objects in use: 0; bytes in use: 0 [...]
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

spark02 (worker) immediately lost its TCPStore connection after rank0 was
killed:
`[rank1]:[W613 07:37:02.402317227 TCPStore.cpp:125] [c10d] recvValue failed
... Connection was likely closed. Did the remote server shutdown or crash?`

**Memory behavior:**

| Time (UTC) | spark01/head MemAvailable | spark02/worker MemAvailable |
|---|---|---|
| 07:26:50 / 07:27:00 (baseline, pre-run) | 123489232 kB (~117.77 GiB) | 123768420 kB (~118.00 GiB) |
| 07:34:16 (spark02's own weight-load RSS peak) | -- | **53375464 kB (~50.90 GiB)** |
| 07:36:09 (spark01, ~29s before weight-load-complete) | 49918152 kB (~47.61 GiB) | -- |
| 07:36:55 (spark01, ~17s after weight-load-complete) | 19295624 kB (~18.40 GiB) | -- |
| 07:37:02.109 (Ray OOM moment, spark01) | **11632304 kB (~11.09 GiB)** | -- |
| post-crash (settled, ~07:37:55-59 / ~07:39:59) | ~14140008 kB (~13.49 GiB) | ~60945000 kB (~58.13 GiB) |

spark01/head dropped from ~117.77 GiB to ~11.09 GiB available by the moment
of Ray's OOM kill -- comparable in magnitude to Attempt 15A's
~117.78 GiB -> ~12.16 GiB and Attempt 13's ~49.1 GiB -> ~13.6 GiB. The bulk of
the drop (~47.6 GiB -> ~18.4 GiB) occurred in the ~17 seconds *after* weight
loading completed, i.e. during `MoEPrepareAndFinalizeNoDPEPModular` setup,
continuing to decline for a further ~7 seconds down to the 11.09 GiB minimum
at the OOM instant. spark02/worker's minimum (~50.90 GiB, at 07:34:16)
occurred *during its own weight-loading RSS spike*, well before rank0's OOM
at 07:37:02, and spark02 subsequently recovered to ~58.13 GiB -- matching
both Attempt 13's and Attempt 15A's asymmetric, head-only collapse pattern.

**New in 15B -- swap usage:** at the OOM instant, spark01's `swapfree`
dropped from 67108860 kB (0 used, at the 07:26:50 baseline) to 62120516 kB,
i.e. **~4.76 GiB of swap in use** -- the first attempt in this series where
swap was observed to be touched during the critical window. Of that, the
rank0 `RayWorkerWrapper` process itself accounted for 1944268 kB
(~1.85 GiB) of swap per its `smaps_rollup` at 07:37:01.906.

**proc_maps evidence (`RayWorkerWrapper`, sudo `trace-memory.sh`):**

| Process | Time (UTC) | RSS | VmHWM (peak-ever RSS) |
|---|---|---|---|
| rank0 (spark01, host pid 62224 / container pid 1903) | 07:36:47.492 (~9s after weight-load-complete) | ~1090808 kB (~1.04 GiB) | ~8641136 kB (~8.24 GiB) |
| rank0 (spark01, host pid 62224 / container pid 1903) | 07:37:01.906 (OOM moment) | ~403040 kB (~0.38 GiB) | ~8641136 kB (~8.24 GiB) |
| rank1 (spark02, host pid 14825 / container pid 339) | 07:37:01.614 (just before OOM) | ~4738648 kB (~4.52 GiB) | ~8604932 kB (~8.21 GiB) |

Ray's own accounting of the killed actor (`actual memory used=0.34GB`) is
consistent with rank0's small proc-level RSS at the OOM instant
(~0.38 GiB). Notably, rank0's `RayWorkerWrapper` RSS was *decreasing* over
the final ~15 seconds before the OOM (1.04 GiB at 07:36:47 -> 0.38 GiB at
07:37:01.906), while spark01's `MemAvailable` fell by ~36.5 GiB
(~47.6 GiB -> ~11.09 GiB) over roughly the same interval. **The dominant
memory growth is therefore unambiguously not attributable to the traced
process RSS** -- both ranks' `RayWorkerWrapper` VmHWM (~8.2-8.24 GiB, reached
earlier during weight loading) is far smaller than the ~36.5 GiB host-level
drop observed after weight loading completed.

`memory-guard.sh` (threshold 8192 MB) did **not** trip on either node:
spark01's available bottomed at ~11.09 GiB (11632304 kB), never crossing the
8192 MB guard threshold. **Ray's memory monitor intervened first**, at
90.8750% -- higher than Attempt 15A's 90.0134%, Attempt 13's 90.2205%, and
Attempts 09/10b's ~90.77%, as expected given the raised 0.905 threshold.

No host freeze occurred; SSH remained fully responsive on both nodes
throughout.

**Attempt 15A vs. Attempt 15B comparison:**

| | Attempt 15A (threshold 0.90) | Attempt 15B (threshold 0.905) |
|---|---|---|
| `RAY_memory_usage_threshold` | 0.90 | 0.905 |
| `RAY_OBJECT_STORE_MEMORY_BYTES` | 1073741824 (1 GiB) | 1073741824 (1 GiB, unchanged) |
| Object store actual (raylet.out) | confirmed: 1.07374 GB | confirmed: 1.07374 GB |
| MemoryMonitor threshold (raylet.out) | not quoted | confirmed: 90.500000% (118189481984 B) |
| EP markers (144/288, both ranks) | reached | reached |
| `MARLIN` selection (both ranks) | reached | reached |
| Weight loading 14/14 | 343.74s | 344.10s |
| `MoEPrepareAndFinalizeNoDPEPModular` | reached (02:46:22) | reached (07:36:39) |
| Fixed-KV reservation | not reached | not reached |
| Time: weight-load-complete -> Ray OOM | ~13s | ~24s |
| **Ray node-memory ratio at OOM** | **0.900134** (109.48/121.63 GB) | **0.908750** (110.53/121.63 GB) |
| Object-store bytes-in-use at OOM | 0 | 0 |
| rank0 `RayWorkerWrapper` RSS at OOM | ~0.48 GiB | ~0.38 GiB (declining from ~1.04 GiB ~15s earlier) |
| rank0 `RayWorkerWrapper` VmHWM (peak) | not measured | ~8.24 GiB |
| rank1 `RayWorkerWrapper` RSS near OOM | ~2.66 GiB (own peak) | ~4.52 GiB |
| rank1 `RayWorkerWrapper` VmHWM (peak) | not measured | ~8.21 GiB |
| spark01 minimum MemAvailable | ~12.16 GiB (12752732 kB) | ~11.09 GiB (11632304 kB) |
| spark02 minimum MemAvailable (own weight-load peak) | ~50.88 GiB | ~50.90 GiB |
| Swap in use at OOM (spark01) | not reported | **~4.76 GiB** (rank0 itself: ~1.85 GiB) |
| Ray OOM | yes, rank0 killed | yes, rank0 killed |
| rank1 TCPStore disconnect | yes | yes |
| memory-guard trip | no | no |
| Host freeze / SSH | none / responsive | none / responsive |

**Interpretation.**

- The `RAY_memory_usage_threshold` 0.90 -> 0.905 increase was **correctly
  applied and verified** (raylet.out confirms 90.500000% /
  118189481984 bytes on both nodes).
- The Ray OOM ratio **rose from 0.900134 to 0.908750** -- an increase of
  ~0.0086, slightly *more* than the 0.005 threshold increase itself. The
  extra ~0.61 GiB-equivalent of headroom was not merely consumed; the
  underlying allocation grew into and past it. This is the signature of
  **sustained growth, not a fixed-size transient peak** -- a transient peak
  bounded below the new threshold would have allowed the run to pass.
- The time-to-OOM after weight-loading completion **increased from ~13s
  (15A) to ~24s (15B)**. Raising the threshold gave the underlying growth
  more time to continue before crossing the (now higher) ceiling, but did
  not stop or bound it -- consistent with an ongoing allocation process
  rather than a one-time spike.
- **Object-store bytes-in-use was again 0** at the OOM moment, with the
  object-store size unchanged at 1 GiB (verified) -- reconfirming Attempt
  15A's finding that the object-store reservation is not the causal lever.
- **Process-level RSS does not explain the growth.** rank0's
  `RayWorkerWrapper` RSS was *decreasing* (1.04 GiB -> 0.38 GiB) during the
  final ~15 seconds before OOM, while spark01's host `MemAvailable` fell by
  ~36.5 GiB over the same window. Both ranks' peak-ever RSS (`VmHWM`,
  ~8.2-8.24 GiB) reached during weight loading is also far smaller than the
  ~36.5 GiB post-weight-load drop. The growth is **outside ordinary process
  RSS accounting**, consistent with §19/§20's hypothesis of CUDA/UMA-resident
  allocation accompanying MARLIN NVFP4 MoE weight preparation/repacking --
  this remains a correlation, not a conclusively proven driver-level
  attribution.
- The new observation of **~4.76 GiB of swap in use at the OOM instant**
  (zero at baseline) is additional evidence of real, sustained memory
  pressure on spark01 -- not merely an accounting artifact of Ray's ratio
  calculation.
- Per the experiment's pre-registered interpretation and the explicit
  constraint not to raise the threshold further: **a small fixed-threshold
  increase does not resolve the MARLIN EP-on post-weight-load memory
  pressure**, because the pressure itself scales to consume added headroom.
  Further threshold increases are not pursued -- the next lever (if any)
  must reduce the underlying allocation (e.g. `GPU_MEMORY_UTILIZATION`), not
  the accounting ceiling.

**Classification: B** (reproducible margin-exhaustion failure under a
controlled, single-variable change; same fundamental mechanism as Attempts
13 and 15A; not a host freeze, not an invalid configuration, and not a
different failure mode -- the 0.905 recurrence is itself the valid
experiment result, as pre-registered).

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt15b-20260613-074625.tar.gz`
- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt15b-20260613-074731.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt15b-20260613-074625.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt15b-20260613-074731.tar.gz`

The head container exited with an error (`EngineCore failed to start`,
Ray OOM, code 1); the worker container disconnected from the head's GCS
(TCPStore recv errors) and was torn down afterward via
`docker compose ... down`. `trace-memory.sh`/`memory-guard.sh` were stopped
with `SIGTERM` on both nodes (no orphaned processes); both nodes show no
active containers post-cleanup.
