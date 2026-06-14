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
| `.env.step37-fi-aot-tp2-ep-debug` | EP-enabled baseline reproduction (util=0.85) |
| `.env.step37-fi-aot-tp2-ep-off-debug` | `--enable-expert-parallel` removed — EP isolation |
| `.env.step37-fi-aot-tp2-ray-tuned-debug` | Ray dashboard off + object-store bound + memory monitor re-enabled |
| `.env.step37-fi-aot-tp2-low-kv-debug` | Reduced `MAX_MODEL_LEN`/`MAX_NUM_SEQS`/`MAX_NUM_BATCHED_TOKENS`, same util |
| `.env.step37-fi-aot-tp2-low-kv-ray-tuned-debug` | Attempt 10b: low-kv-debug's reduced context/batch knobs + ray-tuned-debug's Ray object-store/dashboard/memory-monitor knobs, combined |
| `.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug` | Attempt 11A: ep-off-debug + ray-tuned-debug's Ray memory-monitor safety net, combined |

All six are **disposable/debug** — not promoted/stable presets. None of
them are claimed to fix the underlying issue; they exist to narrow down
*where* the ~20GB+ growth comes from.

> **Note:** These env files are host-specific diagnostic artifacts and are not
> tracked in the repository. Local copies are preserved under
> `.local/env/step37/` on the original host. To reproduce, derive from
> `presets/step37-flash-nvfp4-tp2.env` and apply the configuration delta
> documented for each Attempt below.

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
`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`
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
(`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`),
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
`.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug`
-- identical to
`.env.step37-fi-aot-tp2-ep-off-ray-tuned-debug`
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
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`
-- identical to Attempt 12's
`.env.step37-fi-aot-tp2-ep-off-ray-tuned-kv8g-debug`
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
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-cutlass-debug`
-- identical to Attempt 13's
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`
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
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-flashinfer-cutlass-debug`
-- identical to Attempt 13's
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`
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
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug`
-- identical to Attempt 13's
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-debug`
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
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-threshold0905-debug`
-- identical to Attempt 15A's
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug`
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

## 22. Attempt 16A results (2026-06-13) -- head/worker role-swap isolation and critical-window correction

**Purpose:** Attempts 13, 15A, and 15B (§17, §20, §21) all reproduced the
same failure: shortly after `MoEPrepareAndFinalizeNoDPEPModular` is reached,
spark01's `MemAvailable` collapses from its ~117.7-117.8 GiB baseline to
~11-13 GiB, crosses `RAY_memory_usage_threshold`, and Ray kills a
`RayWorkerWrapper` on spark01. In all three of those attempts, spark01 was
the head/rank0/API node (HEAD_ROCE_IP=10.10.10.1) and spark02 was the
worker/rank1 node (WORKER_ROCE_IP=10.10.10.2). Attempt 16A tests whether the
collapse follows the **head/rank0/API role** or the **physical node
spark01**, by swapping which physical node plays which role:

- spark02 -> head / rank0 / API server (was worker/rank1 in 13/15A/15B)
- spark01 -> worker / rank1 (was head/rank0/API in 13/15A/15B)

This is a single-variable change relative to Attempt 15A: only
`HEAD_ROCE_IP`/`WORKER_ROCE_IP` are swapped
(10.10.10.1 <-> 10.10.10.2); the docker-compose service definitions and
entrypoint are unmodified -- `ROLE` is set per compose profile, and
`VLLM_HOST_IP`/`RAY_NODE_IP_ADDRESS`/`RAY_OVERRIDE_NODE_IP_ADDRESS` are
derived from `HEAD_ROCE_IP` (head service) / `WORKER_ROCE_IP` (worker
service). Deployment order is also swapped: the new worker (spark01) is
started first, then the new head (spark02) ~15s later.

**Env used:**
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-debug`
-- identical to Attempt 15A's
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-debug`
except for the role-swap variable:

| Variable | Attempt 15A | Attempt 16A |
|---|---|---|
| `HEAD_ROCE_IP` | `10.10.10.1` (spark01) | `10.10.10.2` (spark02) |
| `WORKER_ROCE_IP` | `10.10.10.2` (spark02) | `10.10.10.1` (spark01) |

All other settings unchanged from Attempt 15A:

- `--enable-expert-parallel` (EP on), `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- `MARLIN` NvFp4 MoE backend (auto-selected, no explicit `--moe-backend`)
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`, `--kv-cache-dtype fp8`, `--quantization modelopt`
- `RAY_OBJECT_STORE_MEMORY_BYTES=1073741824` (1 GiB)
- `RAY_memory_usage_threshold=0.90` (NOT raised -- pre-registered constraint)
- `RAY_memory_monitor_refresh_ms=100`, `RAY_INCLUDE_DASHBOARD=false`,
  `RAY_DASHBOARD_MAX_EVENTS_TO_CACHE=1000`
- `memory-guard.sh --threshold-mb 8192`
- `trace-memory.sh` run with `sudo` (proc_maps/smaps capture) on both nodes
- both nodes rebooted to a clean baseline before this run, confirmed
  `MemAvailable` ~117.74 GiB (spark01, 123457408 kB) / ~118.00 GiB (spark02,
  123731152 kB) pre-run -- essentially identical to Attempts 15A/15B
  baselines

**Resulting role/rank/expert assignment (confirmed in logs):**

| | New role | RoCE IP | EP rank | Experts | `RayWorkerWrapper` pid |
|---|---|---|---|---|---|
| spark02 | head / rank0 / API (was worker/rank1) | 10.10.10.2 | 0/2 | 0-143 | 1888 |
| spark01 | worker / rank1 (was head/rank0/API) | 10.10.10.1 | 1/2 | 144-287 | 340 |

Both ranks reached `[EP Rank N/2] ... Local/global number of experts:
144/288` and `Using 'MARLIN' NvFp4 MoE backend out of potential backends:
['VLLM_CUTLASS', 'MARLIN', 'EMULATION']` at 08:33:33 -- identical in shape to
Attempts 13/15A/15B.

**Timeline (UTC) -- CORRECTED:**

| Event | Time |
|---|---|
| EP Rank 0/2 (spark02) + EP Rank 1/2 (spark01) + `MARLIN` selection (both ranks) | 08:33:33 |
| spark02/rank0 shard progress: 11/14 (79%) -- last progress line observed | 08:39:03 |
| **spark01/rank1 weight loading completed -- `default_loader.py:397 Loading weights took 340.99 seconds`** | **08:39:15** |
| spark01/rank1 `Using MoEPrepareAndFinalizeNoDPEPModular` (`nvfp4.py:537`) | 08:39:16 |
| spark01 `MemAvailable` collapse begins (49.69 GiB) | 08:39:16.388 |
| **Ray OOM: spark01 (10.10.10.1) at 109.69GB/121.63GB (0.901823), threshold 0.900000 -- `RayWorkerWrapper pid=340` (rank1) killed** | **08:39:24.011** |
| `EngineCore failed to start` / `RuntimeError: Engine core initialization failed` | 08:39:24-26 |

An earlier read of this run mistakenly described **both** ranks as having
failed to complete weight loading at 11/14. That is incorrect for rank1: the
**11/14 stall belongs only to spark02/rank0** (the new head). **spark01/rank1
completed weight loading normally** (340.99s, in line with Attempts
13/15A/15B's ~343-344s on whichever rank/node finished first) and proceeded
into `MoEPrepareAndFinalizeNoDPEPModular` one second later. spark02/rank0's
11/14 stall is a **downstream consequence** of spark01/rank1's Ray OOM kill
and the resulting distributed-initialization abort (see §22.4), not an
independent failure on spark02.

**Ray OOM detail:**

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 10.10.10.1, ID: 2d5471b545d1e069f0305d6e5032ba241cfdd5a58a7859fa7e7b35b1) was
109.69GB / 121.63GB (0.901823), which exceeds the memory usage threshold of 0.900000.
... RayWorkerWrapper.__init__ pid=340, actual memory used=0.79GB ...
Object store memory usage: [...] objects in use: 0; bytes in use: 0 [...]
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

No kernel-level OOM-killer activity was observed on either node
(`journalctl -k -b 0` clean) -- as in 13/15A/15B, Ray's own memory monitor
intervened before any kernel OOM kill.

### 22.1 Critical-window memory trajectory (08:39:10 - 08:39:45 UTC)

| Time (UTC) | spark01 (worker/rank1) `MemAvailable` | spark02 (head/rank0) `MemAvailable` | Note |
|---|---|---|---|
| 08:39:10 | 51.81 GiB | 50.74 GiB | |
| 08:39:11 | 51.66 GiB | 50.59 GiB | |
| 08:39:12 | 51.49 GiB | 50.45 GiB | |
| 08:39:13 | 51.34 GiB | 50.29 GiB | |
| 08:39:14 | 51.20 GiB | 50.15 GiB | |
| **08:39:15** | 51.04 -> 50.98 GiB | 50.00 GiB | **rank1 weight-load complete (340.99s)** |
| 08:39:15.6-16.0 | brief rebound: 52.99 -> 55.16 -> 55.00 GiB | (gentle) | loader-cleanup page-cache transient |
| **08:39:16** | 51.43 GiB | 49.85 GiB | **rank1 enters `MoEPrepareAndFinalizeNoDPEPModular`** |
| 08:39:16.388 | 49.69 GiB | (gentle) | **collapse onset (+0.2s after MoE-prepare marker)** |
| 08:39:17 | 45.92 GiB | 49.71 GiB | |
| 08:39:18 | 41.95 GiB | 49.55 GiB | |
| 08:39:19 | 38.39 GiB | 49.43 GiB | |
| 08:39:20 | 35.19 GiB | 49.27 GiB | |
| 08:39:21 | 29.01 GiB | 49.13 GiB | steepest 1s window starts (21.5-22.6) |
| 08:39:22 | 20.24 GiB | 48.99 GiB | |
| 08:39:23 | 14.49 GiB | 48.83 GiB | |
| **08:39:24.011** | **11.94 GiB (minimum)** | 48.71 GiB (minimum, normal range) | **Ray OOM, ratio 0.901823** |
| 08:39:25.657 | ~14.35 GiB | 53.70 GiB | spark02 recovery begins (+1.6s after spark01 OOM) |
| 08:39:26-27 | ~14.31-14.33 GiB | 55.32 -> 55.35 GiB | |
| 08:39:28-29 | ~14.29 GiB | 57.70 -> 58.55 GiB | |
| 08:39:29-45 | 14.28-14.29 GiB plateau | 58.50-58.55 GiB plateau | both nodes settled |

Post-run `free -h`: spark01 `used=107Gi avail=14Gi swap=323Mi`; spark02
`used=63Gi avail=58Gi swap≈0` -- consistent with the plateau values above.

### 22.2 spark01 collapse-rate analysis

- **Collapse window:** 08:39:16.388 (49.69 GiB) -> 08:39:24.011 (11.94 GiB)
- **Total drop:** ~37.75 GiB over **~7.62 s** -> average **~4.95 GiB/s**
- **Steepest 1-second window:** 08:39:21.5 -> 22.6, 26.33 -> 18.55 GiB,
  **~7.55 GiB/s**
- **1 second before Ray OOM:** ~5.55 GiB drop (22.97 GiB -> 11.94 GiB... see
  table; precisely 17.49 -> 11.94 GiB)
- **5 seconds before Ray OOM:** ~27.69 GiB drop (39.63 -> 11.94 GiB)
- **MoE-prepare marker (08:39:16) -> collapse onset (08:39:16.388):** ~0.2 s
- **Weight-load complete (08:39:15) -> collapse onset (08:39:16.388):** ~1.4 s

The collapse begins essentially the instant `MoEPrepareAndFinalizeNoDPEPModular`
is entered, on whichever node is executing that stage -- in this attempt,
spark01 (rank1).

### 22.3 spark02 -- no collapse, healthy trajectory

Over the same 35-second window, spark02 declined gently and linearly from
50.74 GiB to 48.71 GiB (**~0.14 GiB/s**, consistent with ordinary
weight-loading RSS growth on rank0, not a collapse). Starting ~1.6 s after
spark01's Ray OOM (08:39:25.657), spark02's `MemAvailable` recovered sharply
to 53.70 -> 55.32 -> 58.50-58.55 GiB and remained stable through the end of
the window.

### 22.4 Interpretation of spark02's 11/14 stall

spark02/rank0's weight-loading progress log was last seen at 11/14 (79%,
08:39:03) and never advanced further. Given (a) spark02's own
`MemAvailable` stayed healthy (48.7-50.7 GiB) throughout the critical window
with no collapse, and (b) the Ray OOM that killed spark01/rank1's
`RayWorkerWrapper` (08:39:24.011) immediately preceded `EngineCore failed to
start` / `RuntimeError: Engine core initialization failed`, **spark02's 11/14
stall is a downstream effect of spark01/rank1's termination and the resulting
distributed-initialization abort** -- not evidence of an independent memory
problem on spark02.

### 22.5 Attempt 13 / 15A / 15B / 16A comparison

| | Attempt 13 | Attempt 15A | Attempt 15B | Attempt 16A |
|---|---|---|---|---|
| `RAY_memory_usage_threshold` | 0.90 | 0.90 | 0.905 | 0.90 |
| `RAY_OBJECT_STORE_MEMORY_BYTES` | 4 GiB | 1 GiB | 1 GiB | 1 GiB |
| spark01 role | head/rank0/API | head/rank0/API | head/rank0/API | **worker/rank1** |
| spark02 role | worker/rank1 | worker/rank1 | worker/rank1 | **head/rank0/API** |
| Node that finished weight loading first / entered MoE-prepare first | spark01 | spark01 | spark01 | **spark01** |
| Ray OOM ratio | 0.902205 | 0.900134 | 0.908750 | **0.901823** |
| Time: weight-load-complete -> Ray OOM | ~9s | ~13s | ~24s | **~9s** (~8s after MoE-prepare marker) |
| Node killed by Ray OOM | spark01 | spark01 | spark01 | **spark01** |
| Other node's minimum `MemAvailable` | ~50.9 GiB (own weight-load dip) | ~50.88 GiB | ~50.90 GiB | **48.71 GiB** |
| Other node recovers after | yes | yes | yes | **yes (~58.5 GiB)** |

### 22.6 Conclusions

**Confirmed:**

1. The Ray OOM did **not** move to spark02 when spark02 became the
   head/rank0/API node.
2. The Ray OOM recurred on **spark01** after the role swap, at a ratio
   (0.901823) and timing (~8-9s after `MoEPrepareAndFinalizeNoDPEPModular`)
   consistent with Attempts 13/15A.
3. spark01's `MemAvailable` collapse began ~0.2s after spark01's rank
   entered `MoEPrepareAndFinalizeNoDPEPModular`, and dropped ~37.75 GiB in
   ~7.62s (avg ~4.95 GiB/s, peak ~7.55 GiB/s).
4. spark02 showed a healthy, gentle memory trajectory throughout the
   critical window (50.74 -> 48.71 GiB) and recovered to ~58.5 GiB shortly
   after spark01's OOM -- its 11/14 shard stall is a downstream effect, not
   an independent memory issue.
5. The hypothesis that the head/API/rank0 control-plane role itself
   determines the failure location is **weakened**.

**Still confounded (not yet resolved):**

1. A physical-node-specific factor on spark01 (driver/kernel/page-cache/NVMe
   asymmetry).
2. A code-path effect tied to whichever rank finishes weight loading and
   enters `MoEPrepareAndFinalizeNoDPEPModular` **first** ("first-arriver
   rank").
3. A combination of both.

Attempts 13, 15A, 15B, and 16A all have spark01 finishing weight loading and
entering MoE-prepare first. Therefore, role-swap weakens the head/rank role
hypothesis, but the physical-node hypothesis and the first-arriver-rank
hypothesis remain confounded. Disentangling them requires an experiment in
which spark02 finishes weight loading and enters MoE-prepare first.

### 22.7 Kernel/driver difference (recorded as a lead, not a conclusion)

| | spark01 | spark02 |
|---|---|---|
| Kernel | `6.17.0-1021-nvidia` | `6.17.0-1008-nvidia` |
| Driver | `610.43.02` | `610.43.02` |

This kernel-version difference is recorded as a lead worth investigating. It
is **not** established as the cause by Attempt 16A alone -- the
first-arriver confound (§22.6) must be removed, or the kernel versions
aligned, in a dedicated follow-up before attributing the collapse to either
factor.

**Classification: B** (reproducible margin-exhaustion failure under a
controlled, single-variable role-swap; same fundamental mechanism and
magnitude as Attempts 13/15A/15B; not a host freeze, not an invalid
configuration, and not a different failure mode).

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt16a-20260613-085135.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt16a-20260613-085135.tar.gz`

The head container (spark02, `vllm-spark-head`) exited with an error
(`EngineCore failed to start`, Ray OOM, code 1); the worker container
(spark01, `vllm-spark-worker`) remained `Up` (blocked Ray runtime,
disconnected from GCS) and was torn down cleanly afterward via
`docker compose ... down`. `trace-memory.sh`/`memory-guard.sh` were stopped
with `SIGTERM` on both nodes (no orphaned processes); both nodes show no
active containers post-cleanup.

## 23. Host-level parity audit and same-kernel alignment investigation (2026-06-13)

**Scope:** read-only investigation only, performed on both nodes after
Attempt 16A (§22). No reboots, kernel installs/removals, GRUB changes,
driver reinstalls, sysctl/swap changes, container runs, or vLLM runs were
performed. This section records findings and a plan; no remediation was
executed.

### 23.1 Parity matrix

| Item | spark01 | spark02 | Match |
|---|---|---|---|
| Running kernel | `6.17.0-1021-nvidia` (#21, built 2026-05-27) | `6.17.0-1008-nvidia` (#8, built 2026-01-21) | **NO** |
| `/proc/cmdline` extras | `init_on_alloc=0 iommu.passthrough=0 ... earlycon=... crashkernel=1G-:0M initcall_blacklist=tegra234_cbb_init pci=pcie_bus_safe` | `crashkernel=1G-:0M quiet splash vt.handoff=7` | **NO** -- spark01 has several extra boot params not present on spark02 |
| GRUB top-level menuentry | `'Ubuntu'` (`gnulinux-simple-bb17f2ca-...`) | `'DGX OS GNU/Linux'` (`gnulinux-simple-b68f2c57-...`) | **NO** (cosmetic, but reflects divergent OS-image lineage) |
| `GRUB_DEFAULT` / timeout | `0` / hidden, 0s | `0` / hidden, 0s | YES |
| NVIDIA driver / KMD | `610.43.02` | `610.43.02` | YES |
| CUDA UMD | `13.3` | `13.3` | YES |
| `nvidia.ko` vermagic | matches running kernel (1021) | matches running kernel (1008) | YES (each matches its own kernel) |
| BIOS/firmware | `GX10DGX.0105.2026.0505.1153` (2026-05-05) | identical | YES |
| ConnectX NIC FW (`rocep1s0f0`) | `28.45.4028` | identical | YES |
| RoCE link (`enp1s0f0np0`) | 200 Gb/s, MTU 9000, `PORT_ACTIVE` | identical | YES |
| `sysctl vm.*` (swappiness, overcommit_memory/ratio, min_free_kbytes, zone_reclaim_mode, vfs_cache_pressure) | `1 / 0 / 50 / 45156 / 0 / 100` | identical | YES |
| Swap | `/swap.img`, 64 GiB | identical | YES |
| NUMA / CPU | 1 node, 20 CPUs, ~124546 MB | identical | YES |
| NVMe model | `ESL01TBTLCZ-27J2-TYN`, 1 TB, FW `ERFM12.0`, readahead 256 | identical | YES |
| Root FS | ext4, `/dev/nvme0n1p2`, 917G (84% used) | ext4, `/dev/nvme0n1p2`, 916G (81% used) | YES |
| Docker server version | `29.2.1` | `29.5.3` | minor diff |
| `nvidia-container-runtime` `runtimes` list | `["runc","crun"]` | `["docker-runc","runc","crun"]` | minor diff |
| Operational vLLM image `...step3p7-fi-aot` (`f61c922c9a21`) | present | present | YES |
| Boot-time dmesg (mlx5/nvidia/rdma init sequence) | identical pattern | identical pattern | YES |

### 23.2 Root cause of the kernel-version divergence

spark01 is running `6.17.0-1021-nvidia`; spark02 is running `6.17.0-1008-nvidia`
(the kernel/driver difference recorded as a lead in §22.7). The divergence is
**not** an upstream availability gap -- `linux-image-6.17.0-1021-nvidia`
(6.17.0-1021.21) is available to spark02 directly from
`noble-updates`/`noble-security` (confirmed via `apt-cache policy`). The
divergence is caused by spark02-local apt state:

1. **`/etc/apt/sources.list.d/` is missing the entire DGX repo set on
   spark02.** spark01 has `dgx.sources`, `spark.sources`,
   `cuda-compute-repo.sources`, `ai-workbench-desktop.sources`,
   `canonical-nvidia-ubuntu-nvidia-desktop-edge-noble.sources`,
   `canonical-nvidia-ubuntu-linux-firmware-mbssid-patches-noble.sources`, and
   `nvhpc.sources`. spark02 has none of these -- only
   `cuda-sbsa-ubuntu2404.list`, `docker.list`,
   `nvidia-container-toolkit.list`, `nv-vulkan-desktop-ppa.sources`, and the
   stock `ubuntu*.sources`.

2. **As a downstream consequence, spark02 never received the DGX OTA
   metapackage chain.** spark01 has `dgx-release` (7.5.0), `dgx-repo`
   (25.10-2), `dgx-spark-ota-update-meta` (26.04.1), `nvidia-dgx-telemetry`
   (5.22), `dgx-spark-oobe-customize` (0.17.26), `dgx-spark-mlnx-hotplug`
   (26.01-1), and `nvidia-firmware-580-580.95.05` installed. **None of these
   packages are installed on spark02.** `dgx-spark-ota-update-meta` is the
   package whose dependency chain pulled `linux-image-nvidia-hwe-24.04` to
   1021.21 on spark01.

3. **`linux-image-nvidia-hwe-24.04` / `linux-headers-nvidia-hwe-24.04` are
   explicitly held on spark02** (`apt-mark showhold` lists both, pinned at
   `6.17.0-1008.8`, candidate `6.17.0-1021.21`). On spark01 these packages are
   unheld and installed at 1021.21.

4. **spark02's apt history (`/var/log/apt/history.log`) shows a 2026-06-11
   driver-version detour that is absent on spark01:**
   ```
   2026-06-11 12:47:48  apt-get install -y nvidia-driver-580-open
   2026-06-11 12:50:10  apt-get install -y --allow-downgrades \
       nvidia-persistenced=580.159.04-1ubuntu1 nvidia-modprobe=580.159.04-1ubuntu1 \
       nvidia-settings=580.159.04-1ubuntu1
   2026-06-11 15:47:03  apt-get install -y --allow-downgrades \
       nvidia-dkms-open=610.43.02-1ubuntu1 nvidia-kernel-source-open=610.43.02-1ubuntu1 \
       ... (full 610.43.02 restore)
   ```
   This lines up with `[[qwen35moe_arch_driver580_hang]]` (the driver-580
   hang investigation on 2026-06-11). spark01's apt history for the same
   period shows only a direct, incremental 610.43.02 component-by-component
   install with no 580 detour. The apt holds on the HWE kernel meta-package
   were most likely applied during this driver-580 detour (to prevent a
   kernel change from compounding the driver-version juggling) and were never
   released afterward. The missing `dgx-repo` source set is a separate,
   currently unexplained gap -- it may predate the 06-11 detour (no apt
   history entry removes these sources, so they were likely either never
   present on this node's image or removed outside apt, e.g. manually or by
   an image-customization step before this investigation's history window).

### 23.3 Caveat: current memory-state asymmetry is a transient Attempt-16A artifact, not a host-config difference

While gathering the parity matrix, spark01's current `MemAvailable` was
observed at **~14-15 GiB** (`free -h`: 107Gi used, only 2.7Gi buff/cache) vs
spark02's **~58-61 GiB** (63Gi used, 49Gi buff/cache, mostly reclaimable
page-cache). Both nodes show `system boot` at ~17:18-17:19 (uptime ~1h21m at
observation time), and **no Ray/vLLM/python processes and no
RSS-significant processes are running on either node** (top RSS on spark01
is `dockerd` at 74 MB; `docker ps -a` shows only `portainer_agent`).

spark01's ~107 GiB of unaccounted, non-reclaimable "used" memory with zero
corresponding process RSS matches
[[feedback_gb10_uma_memory_recovery]] exactly: spark01 was the
**worker/rank1** in Attempt 16A and completed weight loading (~75 GB into
UMA) before its `RayWorkerWrapper` was Ray-OOM-killed and its container was
torn down via `docker compose down`. The GB10 driver does not release this
UMA reservation on container stop; only a reboot clears it. spark02 (the
head/rank0 in Attempt 16A) only reached shard 11/14 before stalling, loaded
substantially less into UMA, and shows a normal/reclaimable memory profile.

**This asymmetry is a leftover side-effect of Attempt 16A's teardown, not a
pre-existing or steady-state host-level difference between the two nodes.**
It does not change the §23.1/23.2 findings (which are static configuration
facts), but it means **spark01 needs a reboot before any new Attempt is run**
-- consistent with the existing runbook precedent (reboot both nodes for a
clean baseline before each attempt).

### 23.4 Same-kernel alignment plan (NOT executed)

**Plan A -- bring spark02 up to `6.17.0-1021-nvidia` (recommended direction):**

1. `sudo apt-mark unhold linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04`
2. Restore the missing DGX repo source files
   (`dgx.sources`, `spark.sources`, `cuda-compute-repo.sources`,
   `ai-workbench-desktop.sources`,
   `canonical-nvidia-ubuntu-nvidia-desktop-edge-noble.sources`,
   `canonical-nvidia-ubuntu-linux-firmware-mbssid-patches-noble.sources`,
   `nvhpc.sources`) -- e.g. copy from spark01's `/etc/apt/sources.list.d/`,
   checking each file for host-specific tokens/keys before copying.
3. `sudo apt update`, then review `sudo apt full-upgrade --dry-run` (this
   will likely pull in `dgx-spark-ota-update-meta` 26.04.1 and its
   dependents -- `dgx-dashboard` 0.23.3->0.29.0, `dgx-oobe` 0.19.4->0.25.1,
   `nvidia-dgx-telemetry`, `nvidia-firmware-580-580.95.05`,
   `dgx-spark-oobe-customize`, `dgx-spark-mlnx-hotplug`, and
   `linux-image/headers-nvidia-hwe-24.04` -> 1021.21).
4. Apply the upgrade. The kernel postinst regenerates `grub.cfg` and updates
   the `/boot/vmlinuz` symlink automatically -- verify the `'DGX OS
   GNU/Linux'` / `gnulinux-simple-...` menuentry resolves to
   `vmlinuz-6.17.0-1021-nvidia` afterward (read-only check, no `update-grub`
   needed manually).
5. **Reboot spark02** as a standalone maintenance step (not bundled with a
   vLLM attempt).
6. Verify: `uname -r` = `6.17.0-1021-nvidia`, `nvidia-smi` driver still
   `610.43.02`, RoCE link `PORT_ACTIVE` at 200 Gb/s, `journalctl -k -b0`
   clean (same nvidia/mlx5 init pattern as today's audit).

**Plan B -- roll spark01 back toward spark02's kernel (not recommended):**

- spark01 has `6.17.0-1018-nvidia` installed and bootable (could be selected
  via a one-time `grub-reboot`), but `6.17.0-1008-nvidia` itself is **not**
  installed on spark01 (only leftover `linux-modules-nvidia-fs-6.17.0-1008-nvidia`
  in `rc` state) -- true 1008 parity would mean installing a kernel version
  that `dgx-spark-ota-update-meta` 26.04.1 has already superseded on this
  node, likely fighting its dependency constraints. 1018 would only be a
  partial control (closer to 1008, not identical), and downgrading risks
  breaking spark01's currently-consistent `dgx-release`/driver/firmware
  state. Not recommended.

**Recommendation:** Plan A, executed as an independent host-maintenance task
(its own reboot/validation cycle), separate from the next memory-collapse
experiment.

### 23.5 Next-experiment recommendation

§22.6 leaves two hypotheses confounded: physical-node-specific
(spark01-specific driver/kernel/page-cache/NVMe asymmetry) vs.
first-arriver-rank (whichever rank reaches `MoEPrepareAndFinalizeNoDPEPModular`
first triggers the collapse on its own node). Three options, in increasing
order of scope:

- **Option 1 -- Attempt 17, spark02-first ordering, kernels unchanged.**
  Design a run where spark02 finishes weight loading and enters
  `MoEPrepareAndFinalizeNoDPEPModular` first (e.g. by staggering container
  startup order, or investigating whether weight-load order is
  deterministic/controllable via the existing entrypoint). If the collapse
  follows to spark02, first-arriver-rank is confirmed and the kernel
  difference is ruled out as primary cause; if the collapse stays on spark01
  regardless, the physical-node hypothesis strengthens. This sidesteps the
  kernel question entirely and requires only a reboot + a new attempt under
  the current (mismatched) kernels.

- **Option 2 -- Plan A (spark02 -> 1021) first, then re-run Attempt 13/15A's
  original role assignment (spark01=head) a 5th time under matched kernels.**
  If the OOM ratio/timing changes materially from Attempts 13/15A
  (0.902205 / 0.900134, ~9-13s after MoE-prepare), the kernel difference is
  implicated; if essentially unchanged, kernel version is ruled out and
  first-arriver-rank becomes the leading hypothesis. Requires the Plan A
  maintenance window (apt upgrade + reboot + validation) before the attempt.

- **Option 3 -- both:** Plan A, then a same-kernel Attempt with spark02-first
  ordering. Most informative single experiment (controls for both
  variables at once), but highest total time cost (maintenance window +
  full attempt cycle).

No option was executed. This section is investigation and planning only.

### 23.6 Interpretation

§23.2 identifies the root cause of the **kernel-version drift** between
spark01 and spark02 (incomplete APT/DGX OTA state on spark02, combined with
held NVIDIA HWE kernel meta-packages). This is a separate, established
finding from the still-open question in §22.6.

The audit identifies the cause of the kernel-version drift, not the cause of
the MARLIN memory collapse. The APT/DGX OTA state explains why spark02
remained on kernel `6.17.0-1008-nvidia`, but it does not yet establish that
the kernel difference caused the inference failure observed in Attempts
13/15A/15B/16A. A same-kernel repeat is required before assigning causal
significance to the kernel difference.

## 24. Attempt 16B — Same-kernel role-swap repeat (2026-06-13)

**Purpose:**

- Repeat Attempt 16A with the same role-swap topology.
- Keep the existing Docker-image asymmetry between spark01 and spark02
  unchanged.
- Change only the spark02 running kernel from `6.17.0-1008-nvidia` to
  `6.17.0-1021-nvidia` (Plan A from §23.4, applied as a standalone
  maintenance step before this attempt).
- Determine whether the kernel-version drift recorded in §22.7/§23 materially
  contributed to the MARLIN/UMA collapse.

### 24.1 Controlled configuration

**Attempt 16A:**

| | Role | Kernel | Image ID |
|---|---|---|---|
| spark01 | worker/rank1 | `6.17.0-1021-nvidia` | `sha256:3928025a0c8d...` |
| spark02 | head/rank0/API | `6.17.0-1008-nvidia` | `sha256:1282550c82a1...` |

**Attempt 16B:**

| | Role | Kernel | Image ID |
|---|---|---|---|
| spark01 | worker/rank1 | `6.17.0-1021-nvidia` (unchanged) | `sha256:3928025a0c8d...` (unchanged) |
| spark02 | head/rank0/API | `6.17.0-1021-nvidia` (aligned via §23.4 Plan A) | `sha256:1282550c82a1...` (unchanged) |

Both nodes now run driver `610.43.02` on kernel `6.17.0-1021-nvidia`. The
Docker image asymmetry (different image IDs on spark01 vs spark02) is an
existing, pre-Attempt-16A condition and was deliberately left unchanged in
this attempt.

**Env used:** the same
`.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-debug`
as Attempt 16A, unmodified. All other settings unchanged from Attempt 16A:

- `--enable-expert-parallel` (EP on), `TP_SIZE=2`, `DISTRIBUTED_BACKEND=ray`
- spark02 = head/rank0/API, EP rank 0, experts 0-143, `HEAD_ROCE_IP=10.10.10.2`
- spark01 = worker/rank1, EP rank 1, experts 144-287, `WORKER_ROCE_IP=10.10.10.1`
- `MARLIN` NvFp4 MoE backend (auto-selected, no explicit `--moe-backend`)
- `GPU_MEMORY_UTILIZATION=0.85`
- `--kv-cache-memory-bytes 8589934592` (fixed KV, 8 GiB)
- `--enforce-eager`, `--kv-cache-dtype fp8`, `--quantization modelopt`
- `RAY_OBJECT_STORE_MEMORY_BYTES=1073741824` (1 GiB)
- `RAY_memory_usage_threshold=0.90` (not raised)
- `RAY_memory_monitor_refresh_ms=100`
- `memory-guard.sh --threshold-mb 8192`, `trace-memory.sh --duration 1800`
  (with `sudo`) active on both nodes
- both nodes rebooted to a clean baseline before this run; pre-run
  `MemAvailable` ~123.5 GiB (spark01) / ~123.8 GiB (spark02)

### 24.2 Result

- spark01/rank1 **completed** weight loading:
  `default_loader.py:397 Loading weights took 338.30 seconds`, at
  **10:19:33 UTC**.
- spark02/rank0 **did not complete** weight loading: last recorded progress
  was **12/14 (86%)** when the run was aborted.
- spark01/rank1 entered `Using MoEPrepareAndFinalizeNoDPEPModular`
  (`nvfp4.py:537`) at **10:19:33 UTC** -- the same second as the weight-load
  completion line.
- The `MemAvailable` collapse on spark01 began within approximately one
  second of the `MoEPrepareAndFinalizeNoDPEPModular` entry.
- Ray OOM occurred on **spark01**:
  - **10:19:42 UTC**
  - `109.70GB / 121.63GB` (ratio **0.901933**), threshold `0.900000`
  - `RayWorkerWrapper pid=341` (rank1) killed
- The fixed 8 GiB KV-cache reservation was **not** reached.
- Application startup was **not** reached; `RuntimeError: Engine core
  initialization failed` was raised and the `vllm-spark-head` container
  exited with code 1 (~10:19:48 UTC).
- Inference was not performed.

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 10.10.10.1, ID: 250eb58dfea01ec82ffdcbbe1a954c85332552dc3a2d230502c953da) was
109.70GB / 121.63GB (0.901933), which exceeds the memory usage threshold of 0.900000.
... RayWorkerWrapper.__init__ pid=341, actual memory used=0.85GB ...
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### 24.3 Memory trajectory

The following values were captured via periodic `/proc/meminfo` polling
(roughly every 12-20 seconds), not a continuous high-frequency trace. The
higher-frequency `trace-memory.sh` data for this window is preserved in the
diagnostic tarballs (§24.5) and was not separately analyzed for this section.

- Broad interval: `MemAvailable` on spark01 fell from approximately
  **55.68 GiB at 10:19:18 UTC** to approximately **15.03 GiB at 10:19:46
  UTC** -- a decrease of approximately **40.65 GiB over 28 seconds**.
- Observed collapse interval: from approximately **42.35 GiB at 10:19:34
  UTC** to approximately **15.03 GiB at 10:19:46 UTC** -- a decrease of
  approximately **27.32 GiB over 12 seconds**, an observed average of
  approximately **2.28 GiB/s**.
- Minimum `MemAvailable` observed on spark01: approximately **15.00 GiB**
  (15,004,720 kB at 10:19:59 UTC, post-OOM).
- spark02's `MemAvailable` remained stable in the ~50-53 GiB range
  throughout (no collapse observed on spark02).

The 2.28 GiB/s coarse average above is **not directly comparable** to
Attempt 16A's high-resolution figures (§22.2: ~4.95 GiB/s average over
~7.62s, ~7.55 GiB/s peak over the steepest 1-second window) -- the two were
measured at different sampling resolutions, and the 16B figure should not be
read as evidence that the 16B collapse was slower than 16A's.

### 24.4 Attempt 16A versus 16B

| Item | Attempt 16A | Attempt 16B |
|---|---|---|
| spark01 kernel | `6.17.0-1021-nvidia` | `6.17.0-1021-nvidia` |
| spark02 kernel | `6.17.0-1008-nvidia` | `6.17.0-1021-nvidia` |
| First arriver | spark01/rank1 | spark01/rank1 |
| Weight-loading time (first arriver) | 340.99s | 338.30s |
| `MoEPrepareAndFinalizeNoDPEPModular` entry -> Ray OOM | ~8s | ~9s |
| OOM node | spark01 | spark01 |
| OOM role | worker/rank1 | worker/rank1 |
| OOM ratio | 0.901823 | 0.901933 |
| Fixed KV (8 GiB) reached | no | no |
| Result | Ray OOM | Ray OOM |

### 24.5 Interpretation

Aligning spark02 from kernel `6.17.0-1008-nvidia` to `6.17.0-1021-nvidia` did
not materially change the failure location, timing, or Ray memory ratio. The
1008/1021 kernel-version drift is therefore unlikely to be the primary cause
of the MARLIN initialization collapse.

The physical-node and first-arriver hypotheses remain confounded because
spark01 completed weight loading and entered MoE preparation first in both
Attempt 16A and Attempt 16B.

This does not establish that kernel versions have no effect whatsoever, that
spark01 hardware is conclusively faulty, or that first-arriver behavior is
conclusively the root cause -- it only narrows the 1008/1021 kernel
difference specifically as an unlikely primary cause, under the existing
Docker-image asymmetry which remained unchanged in this attempt.

**Classification: B** (same fundamental mechanism, magnitude, and node as
Attempts 13/15A/15B/16A; not a host freeze, not an invalid configuration, and
not a different failure mode).

**Preserved artifacts:**

- `spark01:~/docker/vllm-spark/.local/diag/diag-spark01-attempt16b-20260613-102231.tar.gz`
- `spark02:~/docker/vllm-spark/.local/diag/diag-spark02-attempt16b-20260613-102230.tar.gz`

---

## 25. Attempt 17 — Controlled first-arriver inversion (2026-06-13)

### 25.1 Purpose

Every prior run (Attempts 13, 15A, 15B, 16A, 16B) showed spark01 completing
weight loading and entering `MoEPrepareAndFinalizeNoDPEPModular` first, and
the UMA memory collapse always occurring on spark01's host.  Attempt 16A
swapped head/worker roles so spark01 became the worker/rank1 and spark02
became head/rank0, but spark01 still arrived at MoE preparation first.
This left two hypotheses for why the collapse always occurred on spark01
confounded:

1. **Physical-node hypothesis** — something specific to spark01's hardware,
   host runtime, NVMe caching, or driver state causes the collapse
   regardless of role.

2. **First-arriver hypothesis** — whichever rank enters MARLIN MoE
   preparation first triggers a local UMA allocation peak on *its own node*,
   large enough to cross the Ray memory threshold.

Attempt 17 breaks the correlation by inserting a 90-second pre-load delay
on rank1 (spark01), guaranteeing that spark02/rank0 completes weight loading
and enters MoE preparation first.

Goals:

- Confirm whether the memory collapse location moves with the first-arriving
  rank or remains anchored to spark01.
- Establish MoE preparation entry as the proximate trigger for the host
  memory event, independently of physical-node or role-based factors.

### 25.2 Controlled configuration

All settings carried forward from Attempt 16A/16B (role-swap baseline):

| Setting | Value |
|---|---|
| spark02 | head / rank0 / API / EP rank 0 / experts 0–143 / RoCE 10.10.10.2 |
| spark01 | worker / rank1 / EP rank 1 / experts 144–287 / RoCE 10.10.10.1 |
| kernel (both nodes) | 6.17.0-1021-nvidia |
| driver (both nodes) | 610.43.02 |
| GPU_MEMORY_UTILIZATION | 0.85 |
| RAY_memory_usage_threshold | 0.90 |
| RAY_OBJECT_STORE_MEMORY_BYTES | 1 073 741 824 (1 GiB) |
| --kv-cache-memory-bytes | 8 589 934 592 (8 GiB fixed) |
| MoE backend | MARLIN (auto-selected, same as all prior MARLIN attempts) |
| --enable-expert-parallel | yes |
| --enforce-eager | yes |
| MAX_NUM_SEQS | 4 |
| MAX_MODEL_LEN | 8192 |

Single controlled variable versus Attempt 16B: diagnostic overlay image with
rank1 90-second pre-load delay hook.

### 25.3 Diagnostic hook

A minimal diagnostic-only patch was applied as a Docker overlay layer on top
of `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7`:

| Item | Value |
|---|---|
| Patched source (inside image) | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py` |
| Patched method | `BaseModelLoader.load_model()` |
| Hook position | Immediately before `self.load_weights(model, model_config)` |
| Activation env var | `VLLM_DIAG_PRE_LOAD_DELAY_RANK=1` |
| Delay env var | `VLLM_DIAG_PRE_LOAD_DELAY_SECONDS=90` |
| Guard | Module-level `_DIAG_PRE_LOAD_DELAY_APPLIED` flag; fires at most once per process |
| Default behaviour | Complete no-op unless both env vars are set; invalid values produce a warning and no-op |
| Diagnostic image | `sha256:213429d55447075c7bd0eb0e63e1df0a44d551a2f454941372f1d601ccab7ed1` |
| Image parity | Image ID confirmed identical on both nodes; 119-byte size difference is a known docker-save/load metadata artifact (Classification I) |

The hook logs `[diag-first-arriver] rank=N pre-load delay start/complete/skipped`
at INFO level via the existing vLLM logger, with UTC timestamp, hostname,
global rank, TP rank, and elapsed time.

### 25.4 Timeline

**spark01 / rank1 (delayed):**

| Event | Timestamp (UTC) |
|---|---|
| Delay start | 2026-06-13T11:05:06.848609Z |
| Delay complete | 2026-06-13T11:06:36.848805Z |
| Actual delay | 90.00 s |
| Weight loading completion | Did not complete — cluster failed before rank1 finished loading |
| MoE preparation | Not reached |

**spark02 / rank0 (no delay):**

| Event | Timestamp (UTC) |
|---|---|
| Delay skipped logged | 11:05:09 UTC |
| Weight loading completed | 11:11:23 UTC (373.98 s) |
| `MoEPrepareAndFinalizeNoDPEPModular` | 11:11:26 UTC |
| Ray OOM termination | 11:11:37 UTC |

### 25.5 Memory collapse on spark02

All measurements from `trace-memory.sh` (0.2 s `meminfo` sampling) on spark02.
Host time shown in KST (UTC+9); UTC = KST − 9 h.

| KST time | MemAvailable (kB) | MemAvailable (GiB) |
|---|---|---|
| 20:11:23.062 | 55 060 004 | ~52.5 |
| 20:11:25.348 | 53 083 584 | ~50.6 (immediately before MoEPrepare) |
| 20:11:26.800 | 50 588 404 | ~48.2 |
| 20:11:28.045 | 47 195 060 | ~45.0 |
| 20:11:30.332 | 41 312 680 | ~39.4 |
| 20:11:33.452 | 31 319 276 | ~29.9 |
| 20:11:35.324 | 21 665 548 | ~20.7 |
| 20:11:36.579 | 14 175 384 | ~13.5 |

Summary:

- Peak-to-trough decrease: approximately 39 GiB (~52.5 → ~13.5 GiB)
- Duration: approximately 11.3 seconds
- Observed average rate: approximately 3.45 GiB/s
- Observed peak interval (20:11:33–36): approximately 5–6 GiB/s
- Ray usage at OOM: 109.51 GB / 121.63 GB = 0.900359 (threshold 0.900000)
- Worker killed: pid=1887, RayWorkerWrapper on spark02 (IP 10.10.10.2)

Note: the 0.2 s sampling resolution means instantaneous-rate figures are
approximations; direct comparisons with the ~100 ms traces from some earlier
attempts should be treated cautiously.

### 25.6 Comparison with Attempt 16B

| Dimension | Attempt 16B | Attempt 17 |
|---|---|---|
| spark01 role | worker / rank1 | worker / rank1 |
| spark02 role | head / rank0 | head / rank0 |
| Pre-load delay on rank1 | none | 90 s |
| First arriver (weight load + MoE prepare) | spark01 / rank1 | spark02 / rank0 |
| Collapse node | spark01 (IP 10.10.10.1) | spark02 (IP 10.10.10.2) |
| Weight loading time (collapsing rank) | 338.30 s | 373.98 s |
| Ray OOM ratio | 0.901933 | 0.900359 |
| Fixed KV reached | no | no |
| Application startup | no | no |

### 25.7 Interpretation

The OOM location moved from spark01 to spark02 in direct correspondence with
the change in which rank entered MARLIN MoE preparation first.  Spark01
remained in weight loading when spark02 failed; the collapse did not occur
on the node that had not yet reached MoE preparation.

Attempt 17 strongly links the failure location to the rank that enters
MARLIN MoE preparation first.  It does not yet distinguish an
ordering-dependent race from a deterministic per-rank preparation peak that
would affect each rank when it reaches the same stage.

The experiment substantially weakens a spark01-specific hardware or
host-runtime explanation, but one inversion run is not sufficient to claim
an exclusive root cause.  In particular, it remains open whether:

- Both ranks independently trigger a preparation-time allocation peak large
  enough to cross the threshold (if so, a fixed-enough threshold would block
  any two-node run, not only the first arriver).
- An ordering or coordination dependency causes the first-arriving rank to
  allocate more than it would if both ranks progressed in step.
- The approximately 39 GiB event is attributable to PyTorch-managed GPU
  tensors, raw CUDA or MARLIN-extension workspace, host-pinned buffers,
  page-cache retention from weight loading, or some combination.

Resolving these questions requires per-marker memory attribution during the
MoE preparation transition (Attempt 18).

### 25.8 Artifacts

- spark01: `.local/diag/diag-spark01-attempt17-full-20260613T112927Z.tar.gz`
- spark02: `.local/diag/diag-spark02-attempt17-full-20260613T112927Z.tar.gz`
- build context: `.local/diag/diag-homeserver-attempt17-build-20260613T112927Z.tar.gz`

---

## 26. Attempt 18 — Initial MARLIN allocation attribution (2026-06-13)

### 26.1 Purpose

Attempt 18 extends the Attempt 17 setup by adding lightweight memory
snapshots at key points across the weight-loading and post-load weight-
processing path.  The goals were:

1. Reproduce the Attempt 17 first-arriver inversion (spark02/rank0 first).
2. Capture `/proc` and PyTorch CUDA counters immediately before and after
   `load_weights` and `process_weights_after_loading`.
3. Capture sub-markers within the first one or two MARLIN MoE preparation
   calls to attribute the approximately 39–41 GiB pressure observed in
   Attempts 13 through 17 to a specific allocator class.

### 26.2 Configuration

| Parameter | Value |
|---|---|
| spark02 role | head / rank0 / API server / EP experts 0–143 |
| spark01 role | worker / rank1 / EP experts 144–287 |
| rank1 pre-load delay | 90 seconds (same as Attempt 17) |
| kernel both nodes | 6.17.0-1021-nvidia |
| driver both nodes | 610.43.02 |
| Ray memory threshold | 0.90 |
| Ray object store | 1 GiB |
| fixed KV cache | 8 GiB (`--kv-cache-memory-bytes 8589934592`) |
| GPU memory utilization | 0.85 |
| diagnostic image | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt18-marlin-memtrace-rank1-delay90` (`sha256:d6bc4d0e68fe2eb025623ee8cabf1f96c644eb7c758467e37289acffe18250c0`) |
| intended markers | 11 (M01–M11); M01–M04 global, M05–M07 per first two MoE layers in `GptOssMxfp4MoEMethod._setup_kernel`, M08–M11 per first two calls to `prepare_moe_mxfp4_layer_for_marlin` |
| marker delivery | Python `logging` → Ray worker relay → Docker logging driver (not durable under SIGKILL) |

### 26.3 Corrected marker timeline

The following table is derived from raw log lines.  UTC timestamps are
as reported by the containers.  All entries are confirmed in the raw log;
no entry is inferred.

| UTC (06-13) | Event | Source | Host | Rank |
|---|---|---|---|---|
| 12:22:35 | `[diag-first-arriver]` rank=1 pre-load delay **start** (90.0 s) | `base_loader.py:122` | spark01 | 1 |
| 12:22:37 | `[diag-first-arriver]` rank=0 delay **skipped** | `base_loader.py:109` | spark02 | 0 |
| 12:22:38 | **M01** `m01_before_load_weights` | `diag_marlin_memory.py` | spark02 | 0 |
| 12:24:05 | `[diag-first-arriver]` rank=1 delay **complete** (elapsed 90.00 s) | `base_loader.py:135` | spark01 | 1 |
| 12:24:05 | **M01** `m01_before_load_weights` | `diag_marlin_memory.py` | spark01 | 1 |
| 12:28:49 | **M02** `m02_after_load_weights` | `diag_marlin_memory.py` | spark02 | 0 |
| 12:28:49 | **M03** `m03_before_process_weights_after_loading` | `diag_marlin_memory.py` | spark02 | 0 |
| 12:28:49 | KV cache FP8 scale warnings (normal, non-fatal) | `kv_cache.py` | spark02 | 0 |
| 12:28:49 | "Your GPU does not have native support for FP4" warning | `marlin_utils_fp4.py` | spark02 | 0 |
| 12:28:53 | **"Using MoEPrepareAndFinalizeNoDPEPModular"** | `oracle/nvfp4.py:537` | spark02 | 0 |
| 12:29:03 | **Ray OOM** — 110.28 GiB / 121.63 GiB = 0.9067 on spark02 | `core.py:1165` | spark02 | 0 |
| 12:29:03 | `EngineCore failed to start` | `core.py` | spark02 | — |
| — | **M04 through M11 absent** (SIGKILL before log delivery) | — | — | — |

**Note on M01 firing pattern.**  M01 fires exactly once per rank process
via a per-process one-shot guard.  It does not fire before the
`_diag_pre_load_delay_hook()` call; the delay hook fires first, then M01
fires immediately after it returns (rank0: delay skipped immediately;
rank1: delay sleeps 90 seconds).  The two M01 log lines at 12:22:38 and
12:24:05 are from two separate OS processes on two separate hosts — not
a double-fire on a single rank.

**Note on "Using MoEPrepareAndFinalizeNoDPEPModular" at 12:28:53.**  This
log line appears inside `make_mxfp4_moe_kernel()` at `oracle/nvfp4.py:537`,
which is called at the end of `GptOssMxfp4MoEMethod._setup_kernel()` after
`replace_parameter()` completes.  Its appearance confirms that at least one
MoE layer reached this point.  It does not confirm that all post-load
cleanup for that layer completed, nor does it identify which layer index
fired first.

### 26.4 Observed memory deltas (rank=0, spark02 only)

All values from raw log.  Host counters in kB converted to GiB.  Byte
counters (torch, CUDA, cgroup) converted to GiB.

#### 26.4.1 Snapshot values

| Counter | M01 (12:22:38) | M02 (12:28:49) | M03 (12:28:49) |
|---|---:|---:|---:|
| MemAvailable (GiB) | 51.699 | 52.394 | 52.394 |
| MemFree (GiB) | 50.499 | 4.995 | 4.995 |
| Cached (GiB) | 2.360 | 48.682 | 48.683 |
| SReclaimable (GiB) | 0.210 | 0.208 | 0.208 |
| SUnreclaim (GiB) | 1.395 | 1.401 | 1.401 |
| VmRSS (GiB) | 2.518 | 2.148 | 2.149 |
| RssAnon (GiB) | 1.472 | 1.528 | 1.528 |
| smaps PSS (GiB) | 2.274 | 2.105 | 2.105 |
| smaps Anonymous (GiB) | 1.472 | 1.528 | 1.528 |
| cgroup memory.current (GiB) | 6.991 | 6.743 | 6.744 |
| cgroup anon (GiB) | 5.143 | 4.571 | 4.571 |
| cgroup file (GiB) | 1.617 | 1.943 | 1.943 |
| torch allocated (GiB) | 58.466 | 58.466 | 58.466 |
| torch reserved (GiB) | 58.951 | 58.951 | 58.951 |
| CUDA free (GiB) | 50.499 | 4.995 | 4.995 |
| CUDA total (GiB) | 121.627 | 121.627 | 121.627 |

#### 26.4.2 Weight-loading interval (M01 → M02, ~6 min 11 s)

| Counter | Delta | Observation |
|---|---:|---|
| MemFree | −45.5 GiB | Large decline in unconditionally free pages |
| Cached | +46.3 GiB | File cache growth of similar magnitude |
| MemAvailable | +0.7 GiB | Near-zero net change (reclaimable cache offsets free loss) |
| torch allocated | 0 | Unchanged; model structure pre-allocated before M01 |
| torch reserved | 0 | Unchanged |
| CUDA free | −45.5 GiB | Mirrors MemFree decline |
| Process RssAnon | +0.06 GiB | Negligible process-side growth |
| cgroup anon | −0.57 GiB | Small decline |
| cgroup file | +0.33 GiB | Small growth |

During weight loading, CUDA-reported free memory decreased substantially
while file cache increased by a similar order of magnitude.  On GB10
unified memory these counters describe overlapping pressure on a shared
physical pool and must not be interpreted as two independent allocations
without further evidence.  The torch allocator counters did not change,
which is consistent with the instanttensor load format writing into UMA
pages that were already reserved by the pre-loading model structure, but
this observation alone is insufficient to attribute the exact mechanism.

#### 26.4.3 Post-load interval (M02 → M03, same second)

M02 and M03 share the same log second (12:28:49).  The monotonic
timestamps differ by 0.139 s.  All memory counters are effectively
identical.  This pair is a near-zero-duration boundary marker: the only
activity between them is the call to `_has_online_quant()` and the
conditional `finalize_layerwise_processing()` branch (not taken for this
model).

#### 26.4.4 Failing interval (M03 → Ray OOM, ~14 s)

| Observation | Value | Source |
|---|---:|---|
| M03 MemAvailable | 52.394 GiB | raw log |
| Ray OOM node memory | 110.28 / 121.63 GiB = 0.9067 | Ray OOM message |
| Implied MemAvailable at OOM | ~11.35 GiB | 121.63 − 110.28 |
| Implied MemAvailable decline | ~41 GiB | 52.394 − 11.35 |
| M03 CUDA free | 4.995 GiB | raw log |
| M04 through M11 | **absent** | SIGKILL before delivery |

Attempt 18 narrows the collapse to post-load processing after M03 and
after at least one MARLIN MoE preparation marker.  The loss of M04
through M11 prevents allocator-level attribution within the failing
interval.

The increase in file cache during weight loading and the later decline in
MemAvailable are separate observations.  Cache reclaim may provide memory
under pressure, but the available data does not establish cache-reclaim
latency as the cause of the collapse.

### 26.5 Logging failure analysis

Python `logging` writes to the worker's stderr.  In Ray's remote-actor
model, worker stderr is relayed through a subprocess pipe to the Ray head
process, which in turn writes to Docker's logging driver.  This creates a
three-stage buffering chain: Python buffer → OS pipe (worker → head) →
Docker log capture.

When the Ray memory monitor issues SIGKILL to the worker process
(pid=1886 on spark02), the OS closes the pipe without draining it.  Any
data not yet transferred across the pipe is discarded.  The `flush()`
calls in `snapshot()` guarantee delivery to the OS pipe from Python's
perspective but cannot guarantee that the receiving side (the Ray head)
has consumed the bytes before SIGKILL fires.

Evidence that M05–M11 executed but were not captured:

- The "Using MoEPrepareAndFinalizeNoDPEPModular" log appeared at
  12:28:53.  This line is emitted inside `make_mxfp4_moe_kernel()`, which
  is the last call in `GptOssMxfp4MoEMethod._setup_kernel()`.  M07
  (`m07_after_make_mxfp4_kernel_L0`) is placed immediately after this
  call; if the MoE kernel construction log appeared, M07 ran but its log
  line was not delivered.
- M05 and M06 are placed before `convert_gpt_oss_weight_to_mxfp4_moe_
  kernel_format()` and before `replace_parameter()` respectively; they
  must also have executed.
- M08–M11 (in `prepare_moe_mxfp4_layer_for_marlin`) similarly ran but
  were not captured.

The fix for Attempt 19 is to bypass the Python logging pipeline entirely
and write JSONL directly to a file inside the container using `os.open`,
`os.write`, and `os.fdatasync`.  This ensures that each line is durable
on disk before execution continues.

### 26.6 Preliminary classification

**Attempt 18 classification: E — Instrumentation inconclusive for
allocator attribution.**

The experiment narrows the failing phase to `process_weights_after_
loading` on spark02/rank0 (the first-arriving rank), and confirms that at
least one MARLIN MoE layer preparation started and that the "Using
MoEPrepareAndFinalizeNoDPEPModular" log appeared before the collapse.
However, the loss of all internal markers (M04–M11) means that the
specific allocator class, tensor operation, and layer index responsible
for the ~41 GiB pressure cannot be determined from Attempt 18 data alone.

### 26.7 Interpretation

Attempt 18 narrows the collapse to post-load processing after M03, but
the missing internal markers prevent identification of the first
allocation operation responsible for the pressure.

Cache growth during loading and cache reclaim during later pressure are
relevant context, not yet a demonstrated causal mechanism.

Attempt 18 is still useful: it confirms that the failure is in
`process_weights_after_loading`, that rank0's failure again tracks the
first-arriver role (consistent with Attempt 17), and that at least one
complete MARLIN MoE preparation occurred before the collapse.  These
observations constrain but do not resolve the root cause.

### 26.8 Artifacts

Note: a spark01-specific tarball was not produced for Attempt 18.
spark01 worker logs were included in the combined head-log capture.
No trace-memory, memory-guard, or Ray log tarballs were collected;
these are added as requirements for Attempt 19.

- Combined head+worker logs + build files:
  `/tmp/diag-spark02-attempt18-20260613_123342.tar.gz`
- Build context + original source files:
  `/tmp/diag-homeserver-attempt18-build-20260613_123342.tar.gz`
- Build directory: `/tmp/attempt18-marlin-allocation-attribution/`

## 28. Attempt 20 — Sub-function phase tracing inside `ModelOptNvFp4FusedMoE.process_weights_after_loading` (2026-06-13)

### 28.1 Purpose

Attempt 19 established that the cumulative UMA memory growth is bounded to
`ModelOptNvFp4FusedMoE.process_weights_after_loading`, but that function
contains three distinct phases:

1. `convert_to_nvfp4_moe_kernel_format(...)` — weight conversion
2. Eight `replace_parameter(...)` calls — parameter substitution
3. `make_nvfp4_moe_kernel(...)` + `fused_experts.process_weights_after_loading(layer)` —
   kernel setup

Attempt 20 adds seven `MO_`-prefixed sub-function markers at the boundaries of
each phase inside `ModelOptNvFp4FusedMoE.process_weights_after_loading` in
`modelopt.py`.  This determines which of the three phases is responsible for
the MemAvailable decline.

### 28.2 Architecture change from Attempt 19

The only code change relative to Attempt 19 is the addition of `modelopt_patch.py`
in the Dockerfile and the corresponding `diag20_durable_writer.py`.  All
global (G01–G04) and per-layer (L_before, L_after) markers from Attempt 19
are retained.  The new markers are:

| Marker | Position |
|---|---|
| `MO_entry` | Function entry; records input tensor shapes and byte counts |
| `MO_before_convert` | Immediately before `convert_to_nvfp4_moe_kernel_format()` call |
| `MO_after_convert` | Immediately after `convert_to_nvfp4_moe_kernel_format()` returns |
| `MO_before_replace_params` | Before the first `replace_parameter()` call |
| `MO_after_replace_params` | After the eighth `replace_parameter()` call |
| `MO_before_make_kernel` | Before `make_nvfp4_moe_kernel()` call |
| `MO_exit` | Function exit |

The durable writer for Attempt 20 is `diag20_durable_writer.py` (schema
version `attempt20-v1`), writing to
`/tmp/attempt20-modelopt-memory-rank-{N}.jsonl`.  The write mechanism is
identical to Attempt 19: `os.open(O_DSYNC)` + `os.fdatasync()` per record,
surviving SIGKILL.

### 28.3 Build

| Property | Value |
|---|---|
| Base image | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt19-durable-marlin-trace-rank1-delay90` |
| New image | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt20-modelopt-internal-trace-rank1-delay90` |
| Image ID | `sha256:795483adb6fd7ebe805319239123971fc939cb428d7d4e5600b7ff98f9d448ce` |
| Build node | spark01 (aarch64); transferred to spark02 via `docker save | ssh … docker load` |
| Build directory | `/tmp/attempt20-modelopt-internal-trace/` on homeserver |

The same image ID (`sha256:795483ad`) was confirmed on both spark01 and spark02
before the run.

### 28.4 Patching mechanism

`modelopt_patch.py` runs once during `docker build`.  It reads
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py`,
locates the exact text of `ModelOptNvFp4FusedMoE.process_weights_after_loading`
using a unique anchor substring (the function docstring), verifies the anchor
appears exactly once, and replaces the function body in-place.  Any mismatch
— zero or more-than-one occurrences — causes the build to fail immediately.

### 28.5 Run configuration

Configuration is identical to Attempt 19 except for the image tag.  Key
parameters:

| Parameter | Value |
|---|---|
| Env file | `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-attempt20-modelopt-internal-trace-debug` |
| EP roles | spark02=rank0/head/experts 0–143; spark01=rank1/worker/experts 144–287 |
| Rank-1 pre-load delay | 90 seconds |
| `MAX_NUM_SEQS` | 4 |
| `GPU_MEMORY_UTILIZATION` | 0.85 |
| `MAX_MODEL_LEN` | 8192 |
| `RAY_memory_usage_threshold` | 0.90 |
| `VLLM_EXTRA_ARGS` | `--enforce-eager --quantization modelopt --kv-cache-dtype fp8 --enable-expert-parallel --kv-cache-memory-bytes 8589934592` (abridged) |

### 28.6 Output and durable records

SIGKILL was issued by the Ray memory monitor during the 27th call to
`convert_to_nvfp4_moe_kernel_format()` (transformer layer 29), after
MemAvailable dropped below the 0.90-threshold kill boundary.

| Node | Rank | Role | JSONL records | Last durable marker |
|---|---|---|---|---|
| spark02 | 0 | head / rank0 | 1034 | `MO_before_convert` (layer 29, `layers.29.moe.experts`) |
| spark01 | 1 | worker / rank1 | 1 | `G01_before_load_weights` |

Rank1 recorded only the G01 global marker.  The 90-second pre-load delay
caused rank1 to begin weight loading approximately 90 seconds after rank0.
Rank0 exhausted available UMA before rank1 reached
`ModelOptNvFp4FusedMoE.process_weights_after_loading`, so no MoE expert
MO_ or L_ markers appear in rank1's output.

### 28.7 Marker counts (rank0)

| Marker | Expected | Observed | Match |
|---|---:|---:|---|
| G01_before_load_weights | 1 | 1 | ✓ |
| G02_after_load_weights | 1 | 1 | ✓ |
| G03_before_process_weights_after_loading | 1 | 1 | ✓ |
| G04_after_process_weights_after_loading | 0 | 0 | ✓ (OOM before G04) |
| L_before | 424 | 424 | ✓ |
| L_after | 423 | 423 | ✓ |
| MO_entry | 27 | 27 | ✓ |
| MO_before_convert | 27 | 27 | ✓ |
| MO_after_convert | 26 | 26 | ✓ (27th call killed mid-flight) |
| MO_before_replace_params | 26 | 26 | ✓ |
| MO_after_replace_params | 26 | 26 | ✓ |
| MO_before_make_kernel | 26 | 26 | ✓ |
| MO_exit | 26 | 26 | ✓ |

All 1034 records parsed without error.  The 27th `MO_entry` and
`MO_before_convert` were written durably; the 27th `MO_after_convert` was not.
This confirms the SIGKILL occurred during the 27th invocation of
`convert_to_nvfp4_moe_kernel_format()`.

### 28.8 Completed modules

Transformer layers 0–2 are dense.  Rank0 processes one
`ModelOptNvFp4FusedMoE` module per MoE transformer layer (experts 0–143 per
layer).  Twenty-six modules completed.

Transformer layer 29 is the first module at which SIGKILL fired.  The
`MO_entry` and `MO_before_convert` markers were durably written (MemAvailable
≈ 13.557 GiB), and `MO_after_convert` was not.  Layer 29 is counted as
incomplete and excluded from cumulative phase totals.

### 28.9 Per-phase MemAvailable delta table (26 complete modules)

Deltas are measured between the flanking MO_ marker pairs.  Phase boundaries:

- **Δ convert**: `MO_before_convert` → `MO_after_convert`
- **Δ replace×8**: `MO_after_convert` → `MO_after_replace_params`
- **Δ make_kernel**: `MO_after_replace_params` → `MO_exit`
- **Δ total**: `MO_entry` → `MO_exit`

| TX layer | Δ convert (GiB) | Δ replace×8 (GiB) | Δ make_kernel (GiB) | Δ total (GiB) |
|---:|---:|---:|---:|---:|
| 3 | −5.2177 | −0.0174 | −0.0000 | −5.2879 |
| 4 | −0.4758 | −0.0001 | +0.0000 | −0.4758 |
| 5 | −2.2726 | −0.0004 | +0.0000 | −2.2729 |
| 6 | −0.7819 | −0.0003 | −0.0012 | −0.7834 |
| 7 | −2.2840 | −0.0035 | +0.0000 | −2.2879 |
| 8 | −0.8259 | −0.0008 | +0.0000 | −0.8268 |
| 9 | −2.2845 | −0.0004 | +0.0000 | −2.2849 |
| 10 | −0.8336 | −0.0001 | −0.0008 | −0.8346 |
| 11 | −2.2721 | −0.0004 | +0.0000 | −2.2711 |
| 12 | −0.8350 | −0.0002 | −0.0015 | −0.8367 |
| 13 | −1.6841 | −0.0015 | +0.0000 | −1.6867 |
| 14 | **+0.3567** | −0.0045 | −0.0102 | **+0.3419** |
| 15 | **+0.1787** | −0.0152 | −0.0002 | **+0.1613** |
| 16 | −0.8029 | −0.0002 | +0.0000 | −0.8102 |
| 17 | −2.1354 | −0.0043 | −0.0167 | −2.1565 |
| 18 | −0.5698 | −0.0173 | +0.0000 | −0.5947 |
| 19 | −2.0973 | −0.0010 | +0.0000 | −2.0987 |
| 20 | −0.6127 | −0.0028 | −0.0075 | −0.6237 |
| 21 | −2.1296 | −0.0003 | +0.0000 | −2.1299 |
| 22 | −0.8265 | −0.0002 | −0.0010 | −0.8274 |
| 23 | −2.2829 | +0.0000 | −0.0013 | −2.2842 |
| 24 | −0.8344 | −0.0002 | +0.0000 | −0.8345 |
| 25 | −2.2818 | −0.0002 | +0.0000 | −2.2821 |
| 26 | −0.8356 | −0.0001 | +0.0000 | −0.8386 |
| 27 | −2.2729 | −0.0065 | −0.0002 | −2.2796 |
| 28 | −0.8235 | −0.0002 | −0.0002 | −0.8240 |
| **29** | **n/a** (SIGKILL mid-flight) | — | — | — |
| **Sum (layers 3–28)** | **−37.737** | **−0.078** | **−0.041** | **−37.856** |
| **Mean (layers 3–28)** | **−1.451** | **−0.003** | **−0.002** | **−1.456** |

Phase attribution over the 26 complete modules: `convert_to_nvfp4_moe_kernel_format`
accounts for −37.737 of the total −37.856 GiB (99.7%).  `replace_parameter`
and `make_nvfp4_moe_kernel` together account for −0.119 GiB (0.3%).

The cumulative memory growth occurs within calls to
`convert_to_nvfp4_moe_kernel_format()`.  `replace_parameter` and
`make_nvfp4_moe_kernel` are negligible contributors.

### 28.10 Tensor inventory

Tensor shapes and byte counts are captured in `MO_entry` and `MO_after_convert`
`tensor_meta` fields.  Values are uniform across all 26 completed modules;
only the first module (transformer layer 3) is shown.

**Input tensors to `convert_to_nvfp4_moe_kernel_format()` (from `MO_entry`):**

| Tensor | Shape | dtype | Bytes | GiB |
|---|---|---|---|---:|
| `layer.w13_weight` | [144, 2560, 2048] | uint8 | 754,974,720 | 0.703 |
| `layer.w13_weight_scale` | [144, 2560, 256] | uint8 | 94,371,840 | 0.088 |
| `layer.w2_weight` | [144, 4096, 640] | uint8 | 377,487,360 | 0.352 |
| `layer.w2_weight_scale` | [144, 4096, 80] | uint8 | 47,185,920 | 0.044 |
| `layer.w13_weight_scale_2` (sliced) | — | — | not captured | — |
| `layer.w13_input_scale` | — | — | not captured | — |
| `layer.w2_weight_scale_2` | — | — | not captured | — |
| `layer.w2_input_scale` | — | — | not captured | — |
| **Captured subtotal** | | | **1,274,019,840** | **1.187** |

**Output tensors from `convert_to_nvfp4_moe_kernel_format()` (from `MO_after_convert`):**

| Tensor | Shape | dtype | Bytes | GiB |
|---|---|---|---|---:|
| `w13` (repacked) | [144, 256, 5120] | int32 | 754,974,720 | 0.703 |
| `w13_scale` (permuted) | [144, 256, 2560] | uint8 | 94,371,840 | 0.088 |
| `w2` (repacked) | [144, 80, 8192] | int32 | 377,487,360 | 0.352 |
| `w2_scale` (permuted) | [144, 80, 4096] | uint8 | 47,185,920 | 0.044 |
| `w13_scale_2`, `a13_scale`, `w2_scale_2`, `a2_scale` | — | — | not captured | — |
| **Captured subtotal** | | | **1,274,019,840** | **1.187** |

The captured output tensors have the same total byte count as the captured
input tensors.  The dtype changes (uint8 → int32 for weights; uint8 → uint8
for scales) with corresponding shape changes, consistent with the Marlin
repack operation repermuting data within the same byte footprint.

The MARLIN backend in `prepare_nvfp4_moe_layer_for_marlin` sets `a13_scale`
and `a2_scale` to `None` on entry, so the corresponding `replace_parameter`
calls for `w13_input_scale` and `w2_input_scale` release those parameters.

### 28.11 Static analysis of `convert_to_nvfp4_moe_kernel_format()`

The function is defined in
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`.

For `NvFp4MoeBackend.MARLIN` (confirmed from runtime `MO_before_convert`
`tensor_meta.nvfp4_backend`), the function dispatches to
`prepare_nvfp4_moe_layer_for_marlin()` in the same file.

**Call graph (MARLIN path):**

```
convert_to_nvfp4_moe_kernel_format()
  └─ prepare_nvfp4_moe_layer_for_marlin()
       ├─ marlin_make_workspace_new(device, 4)           [torch.zeros, small, PyTorch allocator]
       │    └─ stores result in layer.workspace
       ├─ repack_weight(w13, "w13")
       │    └─ for i in range(E=144):
       │         ├─ weight[i].view(torch.int32).T.contiguous()   [new temporary, PyTorch]
       │         └─ ops.gptq_marlin_repack(b_q_weight, perm, ...)
       │              └─ torch.ops._C.gptq_marlin_repack(...)    [_C.abi3.so C extension]
       │    └─ torch.cat(tensor_list)                            [new tensor, PyTorch]
       ├─ repack_weight(w2, "w2")                        [same pattern]
       ├─ premute_scales(w13_scale, w13_scale_2, "w13")
       │    └─ for i in range(E=144):
       │         ├─ scales[i].T
       │         ├─ marlin_permute_scales(...)           [marlin_utils.py]
       │         └─ nvfp4_marlin_process_scales(...)     [nvfp4.py, pure Python/PyTorch]
       │    └─ torch.cat(tensor_list)
       │    └─ nvfp4_marlin_process_global_scale(...)
       └─ premute_scales(w2_scale, w2_scale_2, "w2")    [same pattern]
```

**Extension:** `torch.ops._C.gptq_marlin_repack` is implemented in
`/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so`.  All returned
tensors are standard `torch.Tensor` objects.

**Backend:** `NvFp4MoeBackend.MARLIN`; `experts_cls`:
`vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts`.

**Workspace:** `marlin_make_workspace_new` allocates
`sms × 4` integers via `torch.zeros` on the CUDA device and assigns the
result to `layer.workspace`.  This attribute persists on the layer object
for the duration of model lifetime; its size is proportional to the number
of compute units (tens of kilobytes, not GiB-scale).

**Allocator attribution:** All Python-visible allocations in
`prepare_nvfp4_moe_layer_for_marlin` use standard PyTorch APIs
(`torch.zeros`, `.contiguous()`, `torch.cat`).  The only allocation path not
directly visible from Python is the internal implementation of
`torch.ops._C.gptq_marlin_repack` in `_C.abi3.so`.

The exact allocator, returned-storage ownership, workspace lifetime, and
release behavior remain unresolved at the time of Attempt 20's initial
write-up.  Subsequent re-analysis (see §28.16) confirmed via `nm -D`
analysis of `_C.abi3.so` that `gptq_marlin_repack` allocates its output
tensor through the PyTorch `CUDACachingAllocator` via
`at::TensorMaker::make_tensor`.

### 28.12 Memory accounting analysis

**`torch.cuda.memory_allocated()` is quasi-constant across modules.**
At `MO_before_convert`, `memory_allocated` is approximately 58.465 GiB for
all 27 calls (layers 3–29).  At `MO_after_convert`, it is approximately
59.652 GiB for all 26 completed calls — a consistent +1.187 GiB, matching the
captured output tensor subtotal.  Following `replace_parameter`, the value
returns to approximately 58.465 GiB for the next module's `MO_before_convert`,
indicating that `replace_parameter` frees the old input parameters and the
PyTorch caching allocator recycles the allocation bookkeeping.

**MemAvailable declines monotonically despite the quasi-constant `memory_allocated`.**
The PyTorch caching allocator returns freed pages to its own internal pool
after `replace_parameter`; whether and when those pages are returned to the
OS page allocator (and thus reflected in MemAvailable) depends on the CUDA
driver's behavior on GB10 UMA.  The observed pattern is consistent with pages
being retained in the driver's memory pool across module transitions.  This is
not confirmed at the allocator or driver level at the time of initial write-up.
*(A subsequent re-analysis of `memory_reserved` clarified this picture; see
§28.16.  The reserved pool grew by +42.197 GiB across 26 modules, with zero
reduction after each `replace_parameter`.  This directly accounts for the
observed MemAvailable decline.)*

**CUDA free (`mem_get_info`) is not monotonically correlated with MemAvailable.**
The CUDA free counter increases for some modules (e.g., layer 3: +9.72 GiB)
and decreases for others, in no consistent pattern relative to the MemAvailable
decline.  On UMA systems, `mem_get_info` and `/proc/meminfo MemAvailable`
reflect different accounting views of the same physical pool; their divergence
is consistent with the PyTorch caching allocator returning previously-reserved
CUDA pages to the driver at a different rate than new allocations are made,
but sub-allocator-level tracing is required to confirm this.

**Anomalies at transformer layers 14 and 15.**  These are the only two modules
with a positive Δ convert (+0.357 GiB and +0.179 GiB respectively), meaning
MemAvailable increased slightly during their conversion.  The cause of this
anomaly is not determined from available data.  Possible causes include
OS page-cache reclaim, the PyTorch caching allocator returning pages to the
driver during computation, or an interaction with the per-expert loop
structure for these specific layer configurations.

**Per-loop intermediate tensors.**  `repack_weight` processes E=144 experts
in a loop.  Each iteration creates a `.contiguous()` temporary and passes it
to `gptq_marlin_repack`.  These per-iteration tensors are freed within each
loop iteration; only the final `torch.cat` result persists.  The loop
structure means that peak memory within convert exceeds the final output
size, with a transient spike proportional to the temporary tensor size per
expert.  The magnitude of this transient is not directly observable from
module-boundary markers.

### 28.13 Classification

**Attempt 20 classification: A — memory growth is bounded to the
`convert_to_nvfp4_moe_kernel_format()` call interval.  The responsible
internal operation or allocator is not yet identified.**

Attempt 20 identifies the `convert_to_nvfp4_moe_kernel_format()` call
boundary as the source interval of the cumulative external memory growth.
The three flanking phases — entry overhead, `replace_parameter`, and
`make_nvfp4_moe_kernel` — together account for 0.3% of the measured decline.

The open question for Attempt 21 is why MemAvailable decreases monotonically
across module boundaries despite the PyTorch caching allocator maintaining a
quasi-constant `memory_allocated`.  Candidate explanations involve the CUDA
driver's page-return behavior on UMA and the internal implementation of
`torch.ops._C.gptq_marlin_repack`.  *(Re-analysis of `memory_reserved` in
§28.16 resolved this question: the caching allocator's reserved pool grows
cumulatively by +42.197 GiB, directly tracking the MemAvailable decline.
Attempt 21P therefore tests `expandable_segments:False` as an
allocator-policy A/B variable.)*

### 28.14 Interpretation

The per-phase tracing in Attempt 20 narrows the allocation site from
"within `ModelOptNvFp4FusedMoE.process_weights_after_loading`" (Attempt 19)
to "within `convert_to_nvfp4_moe_kernel_format()`" specifically.

The static call graph shows that for `NvFp4MoeBackend.MARLIN`, the active
sub-function is `prepare_nvfp4_moe_layer_for_marlin()`.  That function calls
`ops.gptq_marlin_repack` 144 times per module (once per expert) and aggregates
results with `torch.cat`.  All Python-visible allocations use the PyTorch
caching allocator.  The one sub-Python-layer call path —
`torch.ops._C.gptq_marlin_repack` in `_C.abi3.so` — is a candidate for
investigation in Attempt 21, but its internal allocation behavior is not
observable from Python.  *(Static `nm -D` analysis of `_C.abi3.so`,
documented in §28.16, confirmed that `gptq_marlin_repack` uses the standard
`CUDACachingAllocator` path for its output tensor via
`at::TensorMaker::make_tensor`.  No persistent workspace or external
allocator is involved.)*

### 28.15 Artifacts

- Rank0 durable JSONL (spark02): `/tmp/attempt20-modelopt-memory-rank-0.jsonl`
  (1034 records, all valid JSON)
- Rank1 durable JSONL (spark01): `/tmp/attempt20-modelopt-memory-rank-1.jsonl`
  (1 record, G01 only)
- Per-module phase delta CSV: `/tmp/attempt20-analysis/attempt20-mo-deltas.csv`
- Combined tarball (homeserver):
  `~/.local/diag/attempt20-modelopt-internal-trace-20260613-142610.tar.gz`
  (SHA256: `8aa598ae32b1a14eb979884a8fc20651f136f86a127c5a436483bd244953fab3`)
- Build directory: `/tmp/attempt20-modelopt-internal-trace/`
- Env file: `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-attempt20-modelopt-internal-trace-debug`

### 28.16 Follow-up allocator-statistics correction (post-Attempt 20 re-analysis)

This section records a correction to the allocator attribution made in §27.8,
§27.9, and §27.10 (Attempt 19 write-up), and to the open questions in
§28.12–14.  All measurements below come from the already-captured Attempt 20
JSONL (`/tmp/attempt20-modelopt-memory-rank-0.jsonl`, 1034 records).

#### Prior interpretation (Attempt 19)

The Attempt 19 analysis observed that `torch_alloc_delta` (i.e.,
`memory_allocated` delta) was approximately zero across 26 module boundaries.
It concluded:

> *"The dominant additional allocation is likely external to that
> [PyTorch caching] allocator."*

This was the most consistent interpretation available from `memory_allocated`
data alone.  The Attempt 19 counters did not include `memory_reserved`.

#### Corrected interpretation

A subsequent re-analysis of the Attempt 20 JSONL included the
`memory_reserved` counter.  Key measurements across 26 completed module
conversions (`MO_before_convert` → `MO_exit`):

| Metric | Value |
|---|---|
| `memory_allocated` at `MO_before_convert` | ≈ 58.465 GiB (constant all 27 calls) |
| `memory_allocated` at `MO_after_convert` | ≈ 59.652 GiB (+1.187 GiB) |
| `memory_allocated` at `MO_after_replace_params` | ≈ 58.465 GiB (returns to baseline) |
| `memory_reserved` at first module entry (layer 3) | ≈ 58.951 GiB |
| `memory_reserved` after last completed module (layer 28) | ≈ 101.148 GiB |
| Total reserved pool growth (26 modules) | **+42.197 GiB** |
| `d_reserved_replace` per module | **0.000 GiB for ALL 26 modules** |
| Total MemAvailable decline (26 modules) | −37.737 GiB |
| Pearson r (reserved growth vs MemAvail decline) | −0.884 |
| `inactive_split_bytes_all_current` | 0 throughout (105 MO_ marker records) |
| `num_alloc_retries` / `num_ooms` | 0 / 0 |

Counter equivalences confirmed exhaustively across all 105 MO_ marker records:

- `cuda.memory_allocated == cuda.active_bytes_all_current == cuda.allocated_bytes_all_current`
- `cuda.memory_reserved == cuda.reserved_bytes_all_current`

(An earlier report section labelled `active_bytes` as tracking with
`reserved_bytes_all_current`; this was a labelling error.  Both
`active_bytes_all_current` and `allocated_bytes_all_current` track active
allocations, identical to `memory_allocated`.)

#### Revised conclusion

The dominant persistent UMA pressure is strongly associated with growth of
the PyTorch CUDA caching allocator's reserved pool during repeated NVFP4
MARLIN conversion.  Active allocations return to the pre-conversion level
after parameter replacement, while the reserved pool grows cumulatively
by +42.197 GiB.

The freed blocks from each module's per-expert `tensor_list`, `torch.cat`,
and `replace_parameter` cleanup are retained in the allocator's reserved pool
(`d_reserved_replace = 0` for all 26 modules).  On GB10 UMA, these reserved
pages remain physically mapped, reducing MemAvailable.

The statement from §27.8/§27.10 —

> *"The dominant additional allocation is likely external to that [PyTorch
> caching] allocator."*

— is superseded by this re-analysis.  The corrected statement is:

> *"The dominant persistent UMA pressure is strongly associated with growth of
> the PyTorch CUDA caching allocator's reserved pool.  Active allocations
> return to their pre-conversion baseline; the reserved pool does not.  This
> does not prove that every retained page is controlled exclusively by PyTorch:
> CUDA-driver and GB10 unified-memory page-management details remain below the
> available instrumentation boundary."*

#### `gptq_marlin_repack` allocator confirmation

Static `nm -D` analysis of `_C.abi3.so` (161 MiB, vLLM 0.22.1,
`__commit_id__ = None`) confirmed:

- Symbol `_ZN2at11TensorMaker11make_tensorEv` (U, undefined/external):
  output tensor allocated via `CUDACachingAllocator`, not raw `cudaMalloc`.
- Op schema `-> Tensor` (non-aliased, non-in-place): output is a fresh
  PyTorch tensor allocated by the C++ wrapper before kernel launch.
- CUDA kernels (`gptq_marlin_repack_kernel<256,4,...>`) receive a
  pre-allocated output pointer; no allocation occurs within the kernel.
- No persistent workspace or static cache attributable to
  `gptq_marlin_repack` from binary analysis.
- `cudaMalloc` is present as a `U` symbol in `_C.abi3.so` but cannot be
  attributed to `gptq_marlin_repack` specifically; the `.so` contains 1460
  exported symbols across many ops.

The 144-expert per-module loop (`tensor_list` accumulation + `torch.cat`)
forces the `expandable_segments:True` allocator to expand its reserved pool
in steps.  Freed blocks from each iteration are not returned to the OS.

#### Odd/even pattern correction

The alternating reserved pool growth (+2.2461 GiB odd / +0.8203 GiB even,
exact to 4 decimal places) is an allocator segment-reuse artifact, not a
model-architecture or processing-path difference.  All MoE layers 3–28 have
identical expert shapes (w13_weight: [144,2560,2048] uint8; w2_weight:
[144,4096,640] uint8).  The pattern arises because odd-layer loops exhaust
the previous module's freed blocks and force new segment allocation, while
even-layer loops partially reuse the previous odd-layer's released segments.

#### Expandable-segment disablement as an allocator-policy test (Attempt 21P)

Disabling expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`)
is proposed as the single-variable Attempt 21P test.  The traditional
block-pool allocator also caches freed blocks, so this experiment tests
segment-growth and reuse behavior rather than guaranteeing immediate
system-memory release.

#### Analysis artifacts (non-Git, `/tmp/attempt21-analysis-only/`)

- `attempt20-pytorch-counter-validation.csv` — exhaustive field-equivalence
  check (105 records × 8 counters)
- `attempt20-reserved-by-module.csv` — per-module reserved/allocated/MemAvail
  deltas for all 26 complete modules
- `attempt20-odd-even-analysis.csv` — odd/even layer reserved growth breakdown
- `attempt20-reserved-analysis.md` — full English analysis narrative
- `attempt20-reserved-summary.json` — structured JSON of all findings
- `gptq-marlin-allocation-analysis.md` — `nm`/`strings`/`ldd` symbol analysis
- `gptq-marlin-callgraph.md` — Python-to-CUDA call graph

---

## 27. Attempt 19 — Durable per-layer ModelOpt NVFP4 tracing (2026-06-13)

### 27.1 Purpose

Attempt 19 addresses the logging failure that caused loss of all internal
markers in Attempt 18.  The core change is replacing Python `logging` with
a durable JSONL writer that calls `os.open` with `O_DSYNC`, `os.write`,
and `os.fdatasync` for each record, ensuring each line is on disk before
execution continues and survives a subsequent SIGKILL.

A secondary purpose is to extend per-layer markers to all
`QuantizeMethodBase` modules in `process_weights_after_loading`, not just
the first two layers used in Attempt 18's intended design.

Additionally, Attempt 19 patches `GptOssMxfp4MoEMethod._setup_kernel` in
`mxfp4.py` and `prepare_moe_mxfp4_layer_for_marlin` in
`marlin_utils_fp4.py` with internal A_-prefixed markers.  These were the
functions identified during Attempt 18's design phase as the suspected
allocation site, based on the "Your GPU does not have native support for
FP4" warning and the "Using MoEPrepareAndFinalizeNoDPEPModular" log.  As
described in §27.3, these patches were applied to the wrong code path.

### 27.2 Configuration

| Parameter | Value |
|---|---|
| spark02 role | head / rank0 / API server / EP experts 0–143 |
| spark01 role | worker / rank1 / EP experts 144–287 |
| rank1 pre-load delay | 90 seconds |
| kernel both nodes | 6.17.0-1021-nvidia |
| driver both nodes | 610.43.02 |
| Ray memory threshold | 0.90 |
| Ray object store | 1 GiB |
| fixed KV cache | 8 GiB (`--kv-cache-memory-bytes 8589934592`) |
| GPU memory utilization | 0.85 |
| diagnostic image | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt19-durable-marlin-trace-rank1-delay90` (`sha256:38a0b0cb`) |
| JSONL writer | `diag19_durable_writer.py`: O_DSYNC, per-rank file, fdatasync per record |
| output paths | `/tmp/attempt19-marlin-memory-rank-0.jsonl` (rank0), `/tmp/attempt19-marlin-memory-rank-1.jsonl` (rank1) |
| markers intended | G01–G04 global; L_before/L_after per `QuantizeMethodBase` module; A_before/A_after internal in `GptOssMxfp4MoEMethod._setup_kernel` and `prepare_moe_mxfp4_layer_for_marlin` |
| markers delivered | G01–G04 and L_before/L_after only (A_ markers not reached — see §27.3) |

### 27.3 Actual quantization class discovery

The A_-prefixed markers were placed in `GptOssMxfp4MoEMethod._setup_kernel`
(in `mxfp4.py`) and `prepare_moe_mxfp4_layer_for_marlin` (in
`marlin_utils_fp4.py`).  These are the classes used for models loaded via
the `gpt_oss` MARLIN-MXFP4 path.  Step-3.7-Flash-NVFP4 uses a different
quantization path.

Post-run analysis of the Attempt 19 JSONL revealed that the
`quant_method_class` field in the L_before markers for all MoE expert
modules consistently reads `ModelOptNvFp4FusedMoE`, not
`GptOssMxfp4MoEMethod`.  The class `ModelOptNvFp4FusedMoE` is defined in
`vllm/model_executor/layers/quantization/modelopt.py` and registered via
`ModelOptNvFp4Config.FusedMoEMethodCls = ModelOptNvFp4FusedMoE`.  Its
`process_weights_after_loading` calls `convert_to_nvfp4_moe_kernel_format`
from `vllm.model_executor.layers.fused_moe.oracle.nvfp4`, a different
conversion path from the MARLIN
`convert_gpt_oss_weight_to_mxfp4_moe_kernel_format`.

Consequence: all A_-prefixed markers in `mxfp4.py` and
`marlin_utils_fp4.py` were never reached.  The L_before/L_after layer
boundary markers, which iterate all `QuantizeMethodBase` subclasses without
filtering by class name, captured the correct module instances.

### 27.4 Marker hierarchy

```
G01_before_load_weights
  [model.load_weights() — instanttensor format, ~64 s]
G02_after_load_weights
G03_before_process_weights_after_loading
  [for each QuantizeMethodBase module in named_modules():]
    L_before  (all classes: ModelOptFp8LinearMethod, ModelOptNvFp4LinearMethod,
                             ModelOptKVCacheMethod, ModelOptNvFp4FusedMoE, …)
    [quant_method.process_weights_after_loading(module)]
    L_after   (if SIGKILL does not interrupt)
G04_after_process_weights_after_loading
```

A_-prefixed sub-markers (in `mxfp4.py`, `marlin_utils_fp4.py`) were
present in the image but never fired because the model uses
`ModelOptNvFp4FusedMoE` rather than `GptOssMxfp4MoEMethod`.

### 27.5 Rank output summary

| Node | Rank | Role | JSONL records | Last marker |
|---|---|---|---|---|
| spark02 | 0 | head / rank0 | 850 | L_before, layer_idx=423 (`layers.29.moe.experts`) |
| spark01 | 1 | worker / rank1 | 1 | G01_before_load_weights |

Rank1 (spark01) recorded only the G01 global marker.  The 90-second
pre-load delay caused rank1 to begin weight loading approximately 90 seconds
after rank0.  Rank0 exhausted available UMA before rank1 reached
`process_weights_after_loading`, so no MoE expert L_before markers appeared
in rank1's output.

The 850 rank0 records cover G01–G04 and L_before/L_after pairs for all
`QuantizeMethodBase` modules in the model.  The total includes non-MoE
linear and attention modules that complete without significant memory
movement, plus the 26 `ModelOptNvFp4FusedMoE` pairs described in §27.6
and one incomplete L_before for transformer layer 29.

### 27.6 Completed ModelOptNvFp4FusedMoE modules and per-layer MemAvailable deltas

Transformer layers 0–2 are dense; layers 3–60 contain MoE experts
(RoutedExperts).  Rank0 (EP rank 0, experts 0–143) processes one
`ModelOptNvFp4FusedMoE` module per MoE transformer layer.  Twenty-six
modules completed before SIGKILL.

| TX layer | layer_idx | MemAvail before (GiB) | MemAvail after (GiB) | Δ MemAvail (GiB) | Δ CUDA free (GiB) | Δ torch_alloc |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 215 | 51.725 | 48.154 | −3.571 | +12.869 | ≈ 0 |
| 4 | 223 | 48.168 | 47.772 | −0.396 | −0.109 | ≈ 0 |
| 5 | 231 | 47.748 | 45.976 | −1.772 | +2.563 | ≈ 0 |
| 6 | 239 | 45.974 | 45.194 | −0.780 | +0.089 | ≈ 0 |
| 7 | 247 | 45.191 | 43.207 | −1.983 | +1.101 | ≈ 0 |
| 8 | 255 | 43.203 | 42.418 | −0.785 | +0.043 | ≈ 0 |
| 9 | 263 | 42.417 | 40.395 | −2.022 | +0.408 | ≈ 0 |
| 10 | 271 | 40.388 | 39.634 | −0.754 | −0.155 | ≈ 0 |
| 11 | 279 | 39.626 | 37.406 | −2.220 | −1.715 | ≈ 0 |
| 12 | 287 | 37.399 | 36.758 | −0.641 | +0.688 | ≈ 0 |
| 13 | 295 | 36.746 | 35.746 | −1.000 | +0.834 | ≈ 0 |
| 14 | 303 | 35.723 | 34.878 | −0.845 | +0.141 | ≈ 0 |
| 15 | 311 | 34.869 | 32.585 | −2.284 | −1.815 | ≈ 0 |
| 16 | 319 | 32.581 | 31.745 | −0.836 | −0.388 | ≈ 0 |
| 17 | 327 | 31.745 | 29.505 | −2.240 | +2.653 | ≈ 0 |
| 18 | 335 | 29.498 | 28.667 | −0.831 | +0.112 | ≈ 0 |
| 19 | 343 | 28.662 | 26.395 | −2.266 | −0.192 | ≈ 0 |
| 20 | 351 | 26.389 | 25.555 | −0.834 | −0.554 | ≈ 0 |
| 21 | 359 | 25.552 | 23.266 | −2.286 | −1.434 | ≈ 0 |
| 22 | 367 | 23.266 | 22.434 | −0.833 | −0.423 | ≈ 0 |
| 23 | 375 | 22.432 | 20.145 | −2.286 | −0.873 | ≈ 0 |
| 24 | 383 | 20.145 | 19.313 | −0.832 | −0.612 | ≈ 0 |
| 25 | 391 | 19.302 | 17.028 | −2.275 | −1.619 | ≈ 0 |
| 26 | 399 | 17.027 | 16.210 | −0.818 | −0.582 | ≈ 0 |
| 27 | 407 | 16.197 | 14.016 | −2.181 | −1.643 | ≈ 0 |
| 28 | 415 | 14.015 | 13.548 | −0.467 | −0.466 | ≈ 0 |
| **29** | **423** | **13.533** | — | — | — | ≈ 0 |
| | | | **Cumulative (layers 3–28)** | **−38.038** | **+8.919** | **< 0.001** |

Layer 29 (layer_idx=423) is the module at which SIGKILL fired.  The
L_before record was durably written (13.533 GiB remaining) and the
L_after was not.  This layer is counted as incomplete and excluded from
the cumulative totals.

### 27.7 Odd/even processing pattern

A consistent odd/even alternation is visible in the per-layer MemAvailable
deltas.  Odd transformer layers (3, 5, 7, …, 27) consume substantially
more memory during processing than even layers (4, 6, 8, …, 28).

| Parity | Layer count | Δ MemAvail mean (GiB) | Δ MemAvail range (GiB) |
|---|---:|---:|---|
| Odd (3, 5, 7, …, 27) | 13 | −2.184 | −1.000 to −3.571 |
| Even (4, 6, 8, …, 28) | 13 | −0.742 | −0.396 to −0.845 |

The mean for odd layers (−2.184 GiB) is approximately 2.9× that for even
layers (−0.742 GiB).  Transformer layer 3 shows the largest single-layer
drop (−3.571 GiB) and transformer layer 13 shows an anomalously small
odd-layer drop (−1.000 GiB).  The remaining odd layers (5, 7, 9, 11,
15–27) cluster near −2.2 to −2.3 GiB.  All even layers cluster near −0.7
to −0.8 GiB.

The CUDA-free counter shows a complementary but unsystematic pattern —
it increases for many odd layers and decreases or stays flat for most even
layers — consistent with temporary PyTorch allocator activity during
`convert_to_nvfp4_moe_kernel_format`.  This counter oscillation does not
correspond to the MemAvailable loss, which is monotonically decreasing.

The cause of the odd/even alternation is not determined from Attempt 19
data.  Possible causes include differences in routing-table alignment,
expert-mapping shapes, or backend selection for alternating layers.

### 27.8 Memory attribution analysis

**Allocator attribution.**  The durable layer markers strongly associate
the cumulative UMA growth with `ModelOptNvFp4FusedMoE` post-load
processing.  The nearly unchanged PyTorch caching-allocator counters
(`torch_alloc_delta` ≈ 0 throughout, total drift < 1 KiB across all 26
completed layers) indicate that the dominant additional allocation is
likely external to that allocator.  *(Allocator attribution revised in
§28.16 following Attempt 20 `memory_reserved` re-analysis.)*

Comparing the cumulative MemAvailable decline (−38.038 GiB) against the
cumulative cgroup `memory.current` change (approximately −7.354 GiB net)
reveals a discrepancy of roughly 31 GiB.  This gap is not accounted for by
process-mapped memory (smaps PSS and Anonymous show negligible drift across
the 26 layers).  On GB10 UMA, allocations made through the CUDA driver
directly — for example, `cuMemAlloc`-family calls, kernel workspace
buffers, or proprietary ModelOpt / TensorRT-LLM internal data — are
reflected in MemAvailable but are not reported through cgroup v2 or
through torch's caching-allocator counters.  This was the most consistent
interpretation available from `memory_allocated` data alone at the time of
Attempt 19.  A subsequent analysis including `memory_reserved` revised this
conclusion; see §28.16.

The CUDA-free counter increased by a net +8.919 GiB over the 26 layers.
On discrete GPUs, an increase in `CUDA free` concurrent with a decrease in
`MemAvailable` would be contradictory; on GB10 UMA both draw from the same
physical pool under different accounting views.  The observed divergence is
consistent with the PyTorch caching allocator returning pages to the CUDA
driver at a different rate than new external allocations claim them, but
sub-function-level tracing is required to confirm this.

**Failing layer.**  SIGKILL was issued by the Ray memory monitor during
processing of transformer layer 29 (layer_idx=423), after MemAvailable had
dropped to 13.533 GiB.  The trigger condition is
`(MemTotal − MemAvailable) / MemTotal > 0.90`: with MemTotal=121.63 GiB
and threshold 0.90, the kill threshold corresponds to MemAvailable below
approximately 12.16 GiB.  Layer 29 was the first layer to cross this
boundary during active processing.

**Remaining unknown.**  Attempt 19 identifies the failing module boundary
but does not yet identify the exact internal operation or allocator
responsible for the growth.

The L_before and L_after markers bound the memory consumption to
`ModelOptNvFp4FusedMoE.process_weights_after_loading`, but that function
contains three distinct phases:

1. `convert_to_nvfp4_moe_kernel_format(...)` — the main weight-conversion
   call.  Input: 8 tensor groups (w13_weight, w13_weight_scale,
   w13_weight_scale_2, w13_input_scale, w2_weight, w2_weight_scale,
   w2_weight_scale_2, w2_input_scale) totalling approximately 3.3 GiB per
   rank-local expert set.
2. Eight `replace_parameter(...)` calls — parameter substitution with the
   converted tensors.
3. `make_nvfp4_moe_kernel(...)` and
   `self.moe_kernel.fused_experts.process_weights_after_loading(layer)` —
   kernel and workspace setup.

Attempt 20 will place sub-function MO_-prefixed markers at the boundaries
of each phase within `ModelOptNvFp4FusedMoE.process_weights_after_loading`
in `modelopt.py`.

### 27.9 Classification

**Attempt 19 classification: B — allocation strongly associated with
`ModelOptNvFp4FusedMoE.process_weights_after_loading`, likely external to
PyTorch caching allocator.  Internal operation not yet identified.**
*(Classification label B retained as the historical finding at the time of
Attempt 19.  The allocator attribution was revised following Attempt 20
`memory_reserved` re-analysis; see §28.16.)*

The durable JSONL writer succeeded: 850 records were written on rank0 and
survived the SIGKILL, confirming that all markers through the last
L_before were captured without loss.  The per-layer boundary markers
isolated the allocation to the correct class.  The A_-prefixed internal
markers were absent because they targeted the wrong class.

### 27.10 Interpretation

The per-layer boundary tracing in Attempt 19 narrows the allocation site
from "somewhere in `process_weights_after_loading` across all module types"
(Attempt 18) to "within `ModelOptNvFp4FusedMoE.process_weights_after_
loading` specifically."  This is the correct class for the
Step-3.7-Flash-NVFP4 model.

The odd/even alternation and the PyTorch-allocator neutrality together
suggest that the dominant pressure comes from a source external to the
standard CUDA caching allocator — likely internal ModelOpt or
TensorRT-LLM kernel buffers allocated through the CUDA driver.  The
alternating pattern may reflect different processing paths or kernel-format
sizes for alternating MoE layers.  *(Interpretation revised in §28.16:
subsequent re-analysis of `memory_reserved` showed that the reserved pool
grew monotonically by +42.197 GiB across 26 modules, and that the odd/even
alternation is a caching-allocator segment-reuse artifact rather than a
model-architecture or processing-path difference.)*

### 27.11 Artifacts

- Rank0 durable JSONL (spark02): `/tmp/attempt19-marlin-memory-rank-0.jsonl`
  (850 records, all valid JSON)
- Rank1 durable JSONL (spark01): `/tmp/attempt19-marlin-memory-rank-1.jsonl`
  (1 record, G01 only)
- Per-layer delta table (CSV): `/tmp/attempt20-modelopt-analysis/attempt19-layer-deltas.csv`
- Per-layer delta table (JSON): `/tmp/attempt20-modelopt-analysis/attempt19-layer-deltas.json`
- Combined tarball (homeserver):
  `~/.local/diag/attempt19-durable-marlin-trace-20260613-131832.tar.gz`
  (SHA256: `fb9734026074c6f83e8c83f4bc16a96a7b3bf84ea880206915a5ccbf5eb3df99`)
- Build directory: `/tmp/attempt19-marlin-durable-trace/`
- Env file: `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-attempt19-durable-marlin-trace-debug`

## 29. Attempt 21P — Disabling expandable CUDA allocator segments (2026-06-13)

### 29.1 Purpose

Attempt 20 established that `memory_reserved` grows monotonically during
`ModelOptNvFp4FusedMoE.process_weights_after_loading`, with parameter
replacement returning `memory_allocated` to its baseline but leaving the
caching-allocator reserved pool unreduced at each module boundary.  The
alternating +2.2461 GiB / +0.8203 GiB staircase was identified as a
caching-allocator segment-reuse artifact rather than a model-architecture
difference (§28.16).

Attempt 21P tests whether the traditional block-pool allocator
(`expandable_segments:False`) exhibits different segment-growth and reuse
behaviour compared with the expandable-segments allocator used in
Attempt 20.  The experiment keeps all serving, model, role, image, Ray,
and diagnostic settings unchanged so that the allocator policy is the
only runtime variable.

### 29.2 Controlled variable

```
Attempt 20:  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Attempt 21P: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

All other settings are identical:

- Same diagnostic image
  (`vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt20-modelopt-internal-trace-rank1-delay90`)
- Same model (Step-3.7-Flash-NVFP4, MARLIN backend, NVFP4)
- Same cluster topology: spark02=head/rank0 (experts 0–143),
  spark01=worker/rank1 (experts 144–287)
- Same Ray thresholds, object-store size, fixed KV cache (8 GiB),
  GPU memory utilisation (0.85), and rank-1 pre-load delay (90 s)
- Same MO_ diagnostic markers from Attempt 20

### 29.3 Result summary

| Metric | Attempt 20 | Attempt 21P |
|---|---|---|
| Completed MoE modules | 26 / 58 | **35 / 58** |
| Failure point | 27th module conversion | 36th module conversion |
| Additional modules | — | **+9 (+34.6%)** |
| OOM cause | Ray memory monitor | Ray memory monitor |
| Ray memory ratio at OOM | — | **109.64 / 121.63 GiB (0.9014 > 0.90)** |
| Fixed KV reached | No | No |
| Startup reached | No | No |

Attempt 21P completed 9 additional MoE modules before Ray killed the
RayWorkerWrapper on spark02 (10.10.10.2).  The experiment did not reach
the fixed-KV reservation or vLLM startup phase.

### 29.4 Reserved-memory behaviour

**Attempt 20 (`expandable_segments:True`):**

| Measurement | Value |
|---|---|
| Initial reserved (layer-3 entry) | ~58.951 GiB |
| Reserved after 26th completed module | ~101.148 GiB |
| Cumulative reserved growth | ~42.197 GiB |
| Growth pattern | alternating +2.2461 GiB (odd) / +0.8203 GiB (even) |
| `d_reserved_replace` (parameter-replacement phase) | 0.000 GiB for all 26 modules |

**Attempt 21P (`expandable_segments:False`):**

| Measurement | Value |
|---|---|
| Initial reserved (MO_entry, first module) | ~58.812 GiB |
| Reserved after first completed module (MO_exit) | ~65.494 GiB |
| Reserved after 35th completed module | ~102.682 GiB |
| First-module reserved increase (entry → exit) | ~6.682 GiB |
| Subsequent per-module reserved increase (modules 2–35) | **+1.0938 GiB (stddev ≈ 0)** |
| `d_reserved` at parameter-replacement phase | 0.000 GiB for all 35 modules |

The traditional allocator replaced the alternating staircase with a
perfectly uniform +1.0938 GiB step per completed module.  The per-module
steady-state growth rate fell from approximately 1.533 GiB (Attempt 20
weighted average) to 1.0938 GiB (−28.6%).

`memory_allocated` remained fixed at approximately 58.640 GiB across all
35 completed modules.  `inactive_split_bytes` was approximately 48.3 MB
throughout (negligible).

### 29.5 Interpretation

Disabling expandable segments changed the allocator's segment-growth and
reuse behaviour, eliminating the alternating staircase and reducing the
steady-state per-module reserved increase.  It did not eliminate
caching-allocator retention.

The traditional block-pool allocator continued to cache released blocks.
Parameter replacement returned `memory_allocated` to its baseline after
each module, but `memory_reserved` did not decrease at module boundaries.
All reserved growth occurred during the `MO_after_convert` phase; neither
the replace-params phase nor the make-kernel phase released any reserved
memory.

The experiment tested allocator segment-growth and reuse behaviour.  Both
allocator modes retained cached blocks.  The result does not indicate that
`expandable_segments:False` returns freed memory to the operating system;
the reduction in growth rate reflects different internal segment sizing and
reuse patterns within the CUDA caching allocator.

**Mathematical constraint:**  With 42 total `ModelOptNvFp4FusedMoE` modules
to convert (transformer layers 3–44; see Attempt 22 §30.2 for model
structure), the projected reserved pool at completion is approximately
65.494 + (41 × 1.0938) ≈ 110.3 GiB — still exceeds the 121.63 GiB GB10
UMA total when accounting for the pre-existing weight footprint (~58.6 GiB
`memory_allocated`).  Allocator-policy adjustment alone cannot enable full
model loading within the current hardware envelope without explicit cache
release or buffer-reuse changes.

**Note (corrected 2026-06-14):**  An earlier version of this text stated
"58 total MoE modules".  The actual `moe_layers_enum` in
`text_config` covers layers 3–44 (42 layers).  The figure 58 was
erroneous; it conflated a different model's layer count.  The corrected
projection still demonstrates that monotonic growth is unsustainable at 42
modules.

### 29.6 Classification

**Partial improvement.**  The allocator policy change reduced per-module
reserved growth and allowed 9 additional modules to complete, but the
reserved pool continued to grow monotonically and the run terminated before
fixed KV or startup were reached.

### 29.7 Safety and cleanup

Both nodes were rebooted after all artifacts were preserved.

| Node | MemAvailable before reboot | MemAvailable after reboot |
|---|---|---|
| spark01 | ~61.2 GiB (partially recovered after run) | ~117.8 GiB |
| spark02 | ~14.4 GiB (UMA retained by NVIDIA driver) | ~118.1 GiB |

Kernel (`6.17.0-1021-nvidia`) and driver (`610.43.02`) were unchanged on
both nodes.  No experiment containers or orphan processes remained after
cleanup.

### 29.8 Artifacts

- **spark02 tarball:**
  `.local/diag/diag-spark02-attempt21p-20260613-162500.tar.gz`
  (SHA256: `2a7a92824e309e0f50396632acfb8f444b665849ae5e4033357540bee397899c`)
  Contains: rank0 JSONL (1241 records), spark02 head Docker log,
  trace-memory log, memory-guard log, memtrace directory.

- **spark01 tarball:**
  `.local/diag/diag-spark01-attempt21p-20260613-162500.tar.gz`
  (SHA256: `9ac0946ed65e279086817d027626800eea8adc20a501dfa72123aef876427d66`)
  Contains: rank1 JSONL (1 record, delay-90 image — no MO_ markers on
  rank1), spark01 worker Docker log, trace-memory log, memory-guard log.

- **homeserver analysis tarball:**
  `/tmp/diag-homeserver-attempt21p-analysis-20260613-162500.tar.gz`
  (SHA256: `48d98bc1d3f38ae7863c12343152d78136e765535df471a78a325de82f4b955c`)
  Contains: Attempt 21P rank0 JSONL, Attempt 20 reserved-by-module CSV,
  Attempt 20 reserved summary JSON, odd/even analysis CSV, PyTorch counter
  validation CSV.

- **Env file (untracked):**
  `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-attempt21p-no-expandable-segments-debug`

- **Allocator config evidence:**
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` confirmed in both
  `vllm-spark-head` and `vllm-spark-worker` container `docker inspect`
  output at runtime.

---

## 30. Attempt 22 — Per-module CUDA cache release (2026-06-14)

### 30.1 Purpose

Test whether releasing unused caching-allocator blocks after each completed
`ModelOptNvFp4FusedMoE.process_weights_after_loading()` call prevents
cumulative reserved-memory growth that terminated Attempt 21P at module 35.

Attempt 22 retained the `expandable_segments:False` allocator mode from
Attempt 21P.  The only new variable was an explicit
`torch.cuda.empty_cache()` call inserted at the outer traversal boundary
after each completed module, controlled by a feature flag
(`VLLM_DIAG_EMPTY_CACHE_AFTER_MODELOPT_MOE=1`).  Instrumentation
(`EC_before` / `EC_after` JSONL markers) recorded the allocator state
around each call.

The aim was to return cached but unreferenced caching-allocator blocks to
the CUDA runtime before the next module began, thereby preventing the
module-by-module reserved-pool staircase observed in Attempts 18–21P.

### 30.2 Corrected model structure

Earlier sections of this document and interim session reports referred to
"58 MoE modules".  Attempt 22 instrumentation and direct inspection of the
model configuration establish the correct figures:

| Item | Count |
|---|---|
| Transformer hidden layers (`text_config.num_hidden_layers`) | 45 |
| Dense transformer layers | 3 (layers 0–2) |
| MoE transformer layers (`text_config.moe_layers_enum` = `3…44`) | 42 |
| Separate MTP layers (`text_config.num_nextn_predict_layers`) | 3 |
| `ModelOptNvFp4FusedMoE` quant modules | **42** |

Module names confirmed by diag20 instrumentation:
`language_model.model.layers.3.moe.experts` through
`language_model.model.layers.44.moe.experts`.

Config path: `stepfun-ai/Step-3.7-Flash-NVFP4/config.json`
Config SHA256: `e09a8654f5c894c50378db85fc950fa127cff800fe77e81caa4692cdd41beab8`

The figure 58 was erroneous: it conflated a different model's layer count.
MTP layers are a separate dense path and must not be counted as MoE
conversion modules.  Attempt 21P therefore failed at module 35 of 42
(83%), not 35 of 58.

### 30.3 Hook location

The `empty_cache()` call was placed at the outer traversal boundary in
`vllm/model_executor/model_loader/utils.py:process_weights_after_loading()`,
immediately after the existing `L_after` snapshot and before
`_layer_idx += 1`:

```
L_after  (diag20 snapshot — Attempt 22 retained)
EC_before (Attempt 22 diag snapshot)
torch.cuda.empty_cache()
EC_after  (Attempt 22 diag snapshot)
_layer_idx += 1  →  next module
```

This location guarantees that:

- `process_weights_after_loading()` has returned: parameter replacement
  and kernel object construction are complete.
- Local conversion tensors (NVFP4 input buffers, intermediate MARLIN
  conversion outputs) are out of scope.
- The call applies only to `ModelOptNvFp4FusedMoE` modules (guarded by
  `type(quant_method).__name__ == "ModelOptNvFp4FusedMoE"`).
- Exactly one call is made per completed module.
- No `gc.collect()`, `torch.cuda.synchronize()`, or explicit parameter
  deletion was used.

### 30.4 Single variable

All settings from Attempt 21P were preserved:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
- `GPU_MEMORY_UTILIZATION=0.85`, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=8192`
- Fixed KV 8 GiB (`--kv-cache-memory-bytes 8589934592`)
- Ray thresholds and object-store unchanged
- Same cluster topology, roles, network config

The only functional change was the addition of the per-module
`empty_cache()` hook (enabled by `VLLM_DIAG_EMPTY_CACHE_AFTER_MODELOPT_MOE=1`
baked into the Attempt 22 image).

An artificial 90-second rank-1 pre-load delay was also baked into the
image to serialize weight loading between ranks.  This delay is a
diagnostic aid, not part of the tested mechanism.

### 30.5 Results

| Metric | Attempt 21P | Attempt 22 |
|---|---|---|
| Completed modules | 35 / 42 | **42 / 42** |
| Ray OOM | yes | **no** |
| Fixed 8 GiB KV allocation | no | **yes** |
| KV token capacity | — | **174,504** |
| Application startup | no | **yes** |
| HTTP inference (200 OK) | not available | **yes** |

Experiment validity confirmed: spark02 (rank0) was the first-arriver in
the Ray cluster join phase.

### 30.6 JSONL marker validation

Both ranks produced identical marker counts with zero parse errors:

| Marker | rank0 | rank1 |
|---|---|---|
| `MO_entry` | 42 | 42 |
| `MO_exit` | 42 | 42 |
| `EC_before` | 42 | 42 |
| `EC_after` | 42 | 42 |
| Incomplete EC pairs | 0 | 0 |
| Last marker | `EC_after` @ layer 543 | `EC_after` @ layer 543 |

### 30.7 Reserved-pool behaviour

`memory_reserved` at `EC_before` (immediately before each `empty_cache()` call):

- First module (layer 215): approximately 65.4941 GiB
- Modules 2–42: approximately 65.3691 GiB on every module (flat;
  max − min = 0.125 GiB, attributable to the first-module cache flush)

**No cumulative staircase was observed.**  Compare Attempt 21P, where
`memory_reserved` grew by approximately 1.094 GiB per module: at module 35
the projected reserved value was approximately 96.9 GiB, with total host
usage reaching approximately 109.6 GiB (90.1% of 121.63 GiB), triggering
the Ray OOM kill.

Effect of each `empty_cache()` call:

| Metric | First module | Modules 2–42 |
|---|---|---|
| `memory_reserved` decrease | ≈ 6.81 GiB | ≈ 6.68 GiB (all equal) |
| `cudaMemGetInfo` free increase | ≈ 88 MiB | ≈ 91–96 MiB |
| `MemAvailable` increase | ≈ 88 MiB | ≈ 91–96 MiB |

### 30.8 Lower-level accounting caveat

`memory_reserved` decreased by approximately 6.68 GiB per completed
module, while the immediately observed changes in `cudaMemGetInfo` free
memory and host `MemAvailable` were much smaller (≈ 91–96 MiB).  These
counters represent different accounting layers and may reflect delayed page
reclamation within the GB10 Unified Memory subsystem.  The experiment
establishes that allocator-level cache release prevented cumulative
reserved-pool growth, but it does not resolve the exact driver-level page
lifecycle or confirm when freed pages became available to other subsystems.

The available instrumentation therefore proves the allocator-level release
and the elimination of cumulative growth, but not the exact lower-level
GB10 page-release timing.

### 30.9 Timing

Instrumentation overhead is included in these figures.  Production-path
overhead (without JSONL writers) will be lower.

| Metric | rank0 (spark02) | rank1 (spark01) |
|---|---|---|
| `empty_cache()` calls | 42 | 42 |
| Average duration | ≈ 10.1 ms/call | ≈ 7.0 ms/call |
| Cumulative duration | ≈ 422 ms | ≈ 292 ms |

The rank1 calls were faster, likely because spark01 had lower concurrent
memory pressure during the serialized weight-loading window (rank-1 delay
was active).

Model loading completed in approximately 421.7 s (rank0) and 546.3 s
(rank1, including the 90-second artificial delay).

### 30.10 Interpretation

Per-module cache release at the outer post-processing boundary eliminated
cumulative CUDA caching-allocator reserved-pool growth and allowed all 42
`ModelOptNvFp4FusedMoE` modules to complete.

This strongly confirms caching-allocator retention as the direct
operational cause of the previous startup OOM.

The mechanism is straightforward: NVFP4-to-MARLIN conversion temporarily
allocates approximately 6.68 GiB of scratch and intermediate tensors per
module.  Without `empty_cache()`, the caching allocator retains those
blocks in the reserved pool after the tensors go out of scope.  Successive
modules accumulate these retained blocks without release.  With
`empty_cache()`, the cached blocks are returned to the CUDA runtime before
the next module begins, bounding the reserved pool to a stable plateau
(≈ 65.37 GiB) rather than allowing monotonic growth.

### 30.11 Classification

**Strong diagnostic success.**

The single added variable (per-module `empty_cache()`) was sufficient to
transform a repeatable OOM failure into a complete, successful startup with
inference available.

### 30.12 Production assessment

The Attempt 22 image includes instrumentation (durable JSONL writers,
`EC_before`/`EC_after` markers, pre-load rank delay) that is not
appropriate for production use.  The diagnostic result strongly motivates
a production-candidate image that applies only the minimal `empty_cache()`
hook without any tracing overhead.

Key open items before promotion:

1. Validate that the hook works correctly in the absence of instrumentation
   (timing interactions with the JSONL writers are not expected but
   unverified).
2. Measure startup-time overhead of the hook alone (expected < 500 ms over
   42 modules based on instrumented timing).
3. Assess shutdown UMA recovery behaviour (whether reboot is required after
   normal container stop when the hook is active).
4. Determine whether the pattern generalises to other EP/MoE + ModelOpt
   deployments.

Upstreaming consideration: the hook is a workaround for a GB10 UMA memory
accounting interaction.  Upstream generalization would require verification
on non-UMA CUDA hardware and confirmation that the allocator-level
mechanism does not introduce regressions.

### 30.13 Artifacts

- **homeserver analysis tarball:**
  `.local/diag/diag-homeserver-attempt22-analysis-20260614T010639Z.tar.gz`
  (SHA256: `80477494dce78bb26330b1e87a8a3e5d11d7de1231c22a07e2e84194ea7fb2f9`,
  size: 233 KiB)
  Contains: rank0/rank1 JSONL (attempt22 + diag20), analysis CSV, full
  and tail Docker logs, inference test output, key log excerpts, patched
  source files, Dockerfile, image inspect, post-startup memory snapshot,
  pre-reboot state.

- **spark02 tarball** (lost on reboot; homeserver copy is authoritative):
  `/tmp/diag-spark02-attempt22-final.tar.gz`
  (SHA256: `c70cd5abc2d765d7a0fcf561f653a1400b32a0dbe0d9bbcc9bdcb35557c0866c`)

- **spark01 tarball** (lost on reboot; homeserver copy is authoritative):
  `/tmp/diag-spark01-attempt22-final.tar.gz`
  (SHA256: `15ec4228625937fcd73cf9627c58b2dda2c9c8372dd462a9e5ef70d54972896e`)

- **Diagnostic image:**
  `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-attempt22-empty-cache-per-modelopt-moe-rank1-delay90`
  Image ID: `sha256:fb53f798e57ce5df0fec78b20ab370dcae5da9bf715be9d36e1ce8f9918fb248`
  (identical on spark01 and spark02)

- **Env file (untracked):**
  `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-role-swap-attempt22-empty-cache-debug`

### 30.14 Safety and cleanup

Both nodes were rebooted after all artifacts were preserved.

| Node | MemAvailable before reboot | MemAvailable after reboot |
|---|---|---|
| spark02 | ≈ 38 GiB | ≈ 118.0 GiB |
| spark01 | ≈ 40 GiB | ≈ 117.7 GiB |

Kernel (`6.17.0-1021-nvidia`) and driver (`610.43.02`) were confirmed
unchanged on both nodes.  No experiment containers or orphan processes
remained after cleanup.  Image parity of the diagnostic image was
confirmed on both nodes post-reboot.

---

## 31. Production-candidate validation without diagnostic instrumentation (2026-06-14)

### 31.1 Purpose

Attempt 22 (§ 30) confirmed that per-module `torch.cuda.empty_cache()` after
each `ModelOptNvFp4FusedMoE.process_weights_after_loading()` call eliminates
the cumulative reserved-pool staircase and allows the 42-module NVFP4→MARLIN
conversion to complete on dual GB10 without a Ray OOM kill.

Section 31 documents the non-instrumented production-candidate validation: the
same fix applied to the clean non-debug base image (`-step3p7`, no artificial
rank delay, no durable JSONL writers, no memory snapshots) and validated
end-to-end from startup to HTTP inference on the real serving stack.

### 31.2 Build differences from Attempt 22

| Dimension | Attempt 22 diagnostic image | Production-candidate image |
|---|---|---|
| Base image | `-step3p7` | `-step3p7` (identical) |
| Artificial rank delay | 90 s baked in | None |
| Durable JSONL writers | `diag20` + `diag22` | None |
| Memory-guard integration | Yes (`VLLM_DIAG_*`) | None |
| EC_before / EC_after markers | Yes | None |
| G/L/MO_ markers | Yes | None |
| Feature flag | Always-on | `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=1` (opt-in) |
| Per-module INFO log | Yes (every module) | No (DEBUG only) |
| Rank-level enable log | No | Yes (1 × per rank) |
| Rank-level summary log | Yes | Yes (1 × per rank) |

The only functional change from the validated Attempt 22 behaviour is the
removal of all diagnostic instrumentation.  The hook location, timing, and
class condition are identical.

### 31.3 Hook specification

The cache-release hook is inserted in
`vllm/model_executor/model_loader/utils.py`,
function `process_weights_after_loading()`, immediately after the inner call:

```python
with device_loading_context(module, target_device):
    quant_method.process_weights_after_loading(module)
# ← hook fires here, after method returned, after parameter replacement,
#   after kernel construction, before the next module iteration
```

Activation conditions (both must hold):

```python
type(quant_method).__name__ == "ModelOptNvFp4FusedMoE"
and "modelopt" in type(quant_method).__module__
```

The dual condition guards against name-substring collision with other quant
methods.  Direct import of `ModelOptNvFp4FusedMoE` is avoided because
`utils.py` is loaded early in the vLLM import chain; a cross-package import
would risk circular dependency.

### 31.4 Feature flag

```
VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE
```

| Value | Behaviour |
|---|---|
| Unset or `"0"` | Disabled (default; upstream-equivalent) |
| `"1"` | Enabled |
| Any other value | Warning printed to stderr, feature disabled |

The flag is evaluated once at module import time.  Default is disabled so the
image is upstream-equivalent when the flag is absent.

### 31.5 Image details

| Property | Value |
|---|---|
| Tag | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` |
| Image ID | `sha256:6e62e76c35a0fa57f669f4d5e0cc9fdcdd817414af97d720020963fd07f58777` |
| Base | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7` (`sha256:3928025a0c8d`) |
| vLLM version | 0.22.1 |
| torch | 2.12.0a0+5aff3928d8.nv26.05 |
| FlashInfer | 0.6.12 |
| modelopt | 0.43.0 |
| Parity | Image ID identical on spark01 and spark02 |

Patch applied by `patches/patch_step3p7_modelopt_cache_release.py` (SHA256:
`1748834ac9749e6c57029321ad3ccdd8a6512ba1df460e740371d2766acac62e`).

Source before patching SHA256 (upstream utils.py):
`55b03cc8443e66f482340790a42b16fa3be4df692eb9db42e0103f52e266dc80`

Source after patching SHA256:
`9bab6e58dee16dfe7491aaacd5b973ff79b837789408101cd226ff6895016ba2`

### 31.6 Startup results

| Metric | Rank 0 (spark02) | Rank 1 (spark01) |
|---|---|---|
| Workaround enabled log | `02:19:14 UTC` | `02:19:57 UTC` |
| Modules flushed | **42 / 42** | **42 / 42** |
| Cumulative cache-release time | **402 ms** | **279 ms** |
| KV cache allocated | 8.0 GiB (fixed) | 8.0 GiB (fixed) |
| Initial free memory before KV | 112.11 GiB | 115.2 GiB |
| GPU KV token capacity | 174,504 tokens | 174,504 tokens |
| Ray OOM during conversion | None | None |
| Application startup complete | **02:24:32 UTC** | — (worker) |

Container start time: spark01 worker `02:11:17 UTC`, spark02 head `02:11:24 UTC`.
Total time from container start to `Application startup complete`: ≈ 13 min 8 s.

### 31.7 API verification

All requests issued from `127.0.0.1:8000` on spark02 (host-network mode).
Served model: `stepfun-ai/Step-3.7-Flash-NVFP4`.

| Endpoint | HTTP status | Notes |
|---|---|---|
| `GET /health` | **200** | Empty body, as expected |
| `GET /v1/models` | **200** | Model list returned correctly |
| `POST /v1/completions` | **200** | `prompt_tokens=5, completion_tokens=8` |
| `POST /v1/chat/completions` | **200** | `prompt_tokens=22, completion_tokens=32` |

No engine death, no worker loss, no Ray OOM on any request.

### 31.8 Five-minute stability observation

Observation window: `03:12:19 UTC` to `03:17:09 UTC` (10 checks, 30 s interval).

| Check | spark02 MemAvail | spark01 MemAvail | `/health` | Head RC | Worker RC |
|---|---|---|---|---|---|
| 1/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 2/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 3/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 4/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 5/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 6/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 7/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 8/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 9/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |
| 10/10 | 38.8 GiB | 40.8 GiB | 200 | 0 | 0 |

MemAvailable stable (no monotonic decrease).  OOMKilled=false throughout.
Delayed Ray OOM: none.

### 31.9 Classification

**Production-candidate success with separate tokenizer and output-format
limitations** (see § 31.10).

All eleven success criteria satisfied:

1. `Application startup complete` ✓
2. Both ranks: 42/42 cache-release summary confirmed ✓
3. Fixed KV 8 GiB allocation ✓
4. KV capacity 174,504 tokens confirmed ✓
5. `/health` HTTP 200 ✓
6. `/v1/completions` HTTP 200 ✓
7. `/v1/chat/completions` HTTP 200 ✓
8. Five minutes: zero container restarts ✓
9. Five minutes: Ray/vLLM processes survived ✓
10. Delayed Ray OOM: none ✓
11. MemAvailable: no dangerous monotonic decrease ✓

The chat completion final-answer field (`content`) was `null` because
`max_tokens=32` was exhausted during the reasoning phase; this is a budget and
output-format limitation, not a cache-release failure.

### 31.10 Separate limitations (not cache-release failures)

The following issues were observed but are independent of the cache-release
workaround:

1. **Completion Unicode/fullwidth corruption**: `/v1/completions` output
   `"4ï¼Į4+2=6ï¼Į"` contains mojibake of fullwidth comma characters.
   Root cause not yet confirmed; likely a tokenizer decoding path issue.

2. **Reasoning BPE markers in chat output**: The `reasoning` field in
   `/v1/chat/completions` responses contains raw BPE space markers (`Ġ`, `Ċ`)
   that should be decoded to regular spaces and newlines.

3. **Chat completion `content=null`**: With `max_tokens=32`, the step3p5
   reasoning parser exhausted the token budget during the thinking phase
   before the model emitted the final answer.  Increasing `max_tokens`
   resolves this.  Engine and workers were healthy throughout.

These three issues are present in the non-instrumented image and are not
introduced by the cache-release patch.  They require a separate investigation
track.

### 31.11 Shutdown UMA recovery

Pre-shutdown MemAvailable: spark02 ≈ 38.8 GiB, spark01 ≈ 40.8 GiB.

Post-`docker stop` timeline (T0 = `03:31:55 UTC`):

| Checkpoint | spark02 MemAvail | spark01 MemAvail | vLLM/Ray procs |
|---|---|---|---|
| T0 (`03:32:21`) | 46.4 GiB | 46.0 GiB | 0 |
| T+15 s (`03:32:53`) | 46.4 GiB | 46.0 GiB | 0 |
| T+60 s (`03:32:55`) | 46.4 GiB | 46.0 GiB | 0 |
| T+120 s (`03:33:55`) | 46.4 GiB | 46.0 GiB | 0 |

Recovery classification: **Residual UMA state** (< 90 GiB at T+120 s).
Consistent with the known GB10 driver behaviour documented in § 19.

Both nodes were rebooted (`sudo sync && sudo systemctl reboot`).

Post-reboot state (`03:39:32 UTC`):

| Node | Kernel | Driver | MemAvailable | RDMA |
|---|---|---|---|---|
| spark02 | 6.17.0-1021-nvidia | 610.43.02 | 117.9 GiB | rocep1s0f0 ACTIVE |
| spark01 | 6.17.0-1021-nvidia | 610.43.02 | 117.7 GiB | rocep1s0f0 ACTIVE |

### 31.12 Repository integration

The cache-release hook was promoted to the repository production build path:

- **Patch script**: `patches/patch_step3p7_modelopt_cache_release.py`
- **Dockerfile**: `dockerfiles/active/Dockerfile.step3p7` — patch applied as
  step 3 in the chain (after registry and input_scale patches)
- **Preset env**: `presets/step37-flash-nvfp4-tp2.env` — updated to reference
  the new image tag and include both required env vars
- **docker-compose.yml**: `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE`
  passthrough added to both `head` (line 36) and `worker` (line 215) service
  environment sections

New formal image tag:
`vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release`

Image ID (spark01 = spark02):
`sha256:6e62e76c35a0fa57f669f4d5e0cc9fdcdd817414af97d720020963fd07f58777`

### 31.13 Artifacts

- **Runtime artifacts**: `/tmp/step37-production-candidate-final/`
  (homeserver; full Docker logs, container inspect, API responses,
  five-minute stability logs, image inspect, process lists, meminfo)

- **Production-candidate env file (untracked)**:
  `.env.step37-fi-aot-tp2-ep-ray-tuned-kv8g-objectstore1g-modelopt-cache-release-candidate`

- **Candidate image** (preserved, not promoted to production tag):
  `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release-candidate`
  Image ID: `sha256:8721ee5b68914b96e30f95d54b2f98983407c398a43962d94adad11b859c1708`

---

## Section 32 — Formal repository image acceptance (2026-06-14)

### 32.1 Purpose

Final acceptance test of the image produced by the formal repository build
path (`dockerfiles/active/Dockerfile.step3p7`, git commit `15b1895`).
This run uses the committed preset (`presets/step37-flash-nvfp4-tp2.env`)
with no ad-hoc modifications. The goal is to confirm that the patch-script
approach produces a functionally equivalent image to the production-candidate
validated in §31.

### 32.2 Image and patch verification

| Check | Result |
|---|---|
| Image tag | `vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release` |
| Image ID (spark01 = spark02) | `sha256:6e62e76c35a0fa57f669f4d5e0cc9fdcdd817414af97d720020963fd07f58777` |
| Layer count vs base | 108 layers (base 106 + COPY + RUN = +2 expected) |
| Git commit | `15b1895 fix(step37): release ModelOpt MoE conversion cache` |
| Patched `utils.py` SHA256 (formal image) | `411674b6705aa439f3fc6a8b9c9391f8c532eec1d0ae9f6f7ec4feef84fd9b3d` |
| Functional gate `_SPARK_EC_ENABLED` | PASS |
| Env var `VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE` | PASS |
| `empty_cache()` call | PASS |
| Class guard `ModelOptNvFp4FusedMoE` | PASS |
| Module guard `"modelopt" in __module__` | PASS |
| Enable log present | PASS |
| Summary log present | PASS |
| No `gc.collect`, no `cuda.synchronize`, no `time.sleep` | PASS |

**Note on SHA256 difference from §31**: The production-candidate image used a
directly-copied pre-patched file; the formal image uses the patch script
(`patch_step3p7_modelopt_cache_release.py`) applied to the base image's
`utils.py`.  The resulting bytes differ (different approach, different
intermediate whitespace), but all 12 functional checks pass.

### 32.3 Baseline

Both nodes freshly rebooted after §31 shutdown.

| Node | Role | kernel | driver | MemAvailable |
|---|---|---|---|---|
| spark02 | head / rank0 | 6.17.0-1021-nvidia | 610.43.02 | 117 GiB |
| spark01 | worker / rank1 | 6.17.0-1021-nvidia | 610.43.02 | 117 GiB |

Swap used: 0 on both. No experiment containers running.

### 32.4 Acceptance env

File: `.env.step37-modelopt-cache-release-acceptance`
Generated from `presets/step37-flash-nvfp4-tp2.env` with host-specific values:

```
VLLM_IMAGE=vllm-spark:v022-d568-ngc2605-tx5102-vllm022-step3p7-modelopt-cache-release
MODEL_PATH=/home/bjk110/Documents/Models/stepfun-ai/Step-3.7-Flash-NVFP4
HEAD_ROCE_IP=10.10.10.2  WORKER_ROCE_IP=10.10.10.1
ROCE_IF_NAME=enp1s0f0np0  IB_HCA_NAME=rocep1s0f0  RAY_PORT=6379
GPU_MEMORY_UTILIZATION=0.88  MAX_MODEL_LEN=8192  MAX_NUM_SEQS=4
VLLM_SPARK_EMPTY_CACHE_AFTER_MODELOPT_MOE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

### 32.5 Startup results

Worker (spark01) started first, then head (spark02).

**Cache-release hook (both ranks confirmed)**:

| Rank | Node | Modules flushed | Cumulative time |
|---|---|---|---|
| rank0 | spark02 (head) | 42 / 42 | 427 ms |
| rank1 | spark01 (worker) | 42 / 42 | 297 ms |

Both well under the 500 ms per-rank limit documented in the preset.

**Memory and KV cache**:

| Metric | Value |
|---|---|
| Model loading peak (both ranks) | 58.58 GiB |
| Available KV cache (rank0/spark02) | 37.33 GiB |
| Available KV cache (rank1/spark01) | 35.83 GiB |
| GPU KV cache capacity | 781,611 tokens |
| Application startup complete | yes |
| RestartCount (head) | 0 |
| RestartCount (worker) | 0 |
| OOMKilled | false (both) |

### 32.6 API acceptance

All tests performed from spark02 localhost immediately after startup complete.

| Endpoint | Method | Input | HTTP status | Outcome |
|---|---|---|---|---|
| `/health` | GET | — | 200 | PASS |
| `/v1/models` | GET | — | 200 | model ID `stepfun-ai/Step-3.7-Flash-NVFP4` |
| `/v1/completions` | POST | `"2+2="`, max_tokens=8 | 200 | text returned; fullwidth encoding issue (known pre-existing, §31 note) |
| `/v1/chat/completions` | POST | short English prompt, max_tokens=512 | 200 | finish_reason=stop, content returned; BPE markers in output (known pre-existing, §31 note) |

### 32.7 5-minute stability

Interval: 30 s, 10 checks.

| Check | head RC | head OOMKilled | worker RC | worker OOMKilled | /health | head MemAvail | worker MemAvail |
|---|---|---|---|---|---|---|---|
| T+30s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+60s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+90s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+120s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+150s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+180s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+210s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+240s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+270s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |
| T+300s | 0 | false | 0 | false | 200 | 14 GiB | 17 GiB |

Result: **10/10 PASS**

**Note on MemAvail during run**: The formal preset uses `GPU_MEMORY_UTILIZATION=0.88`
without an explicit KV memory cap, so vLLM claims ~107 GiB of the 121.63 GiB UMA
pool.  The remaining 14–17 GiB available during the run is expected for this
configuration and does not indicate memory pressure (no OOM, no restart, no swap
use, health 200 throughout).

### 32.8 Acceptance judgment

**PASS — Success with pre-existing tokenizer limitation**

The formal repository image is functionally equivalent to the
production-candidate validated in §31:

- Cache-release hook fires correctly: 42/42 modules per rank, overhead <0.5 s
- No OOM, no restart, health 200 for the full 5-minute stability window
- All API endpoints respond HTTP 200 with model output

The tokenizer/BPE-marker issues observed in §31 persist and are unchanged by
this release.  They are independent of the ModelOpt cache-release workaround
and documented separately in `/tmp/step37-tokenizer-followup.md`.

### 32.9 Shutdown and UMA recovery

Graceful stop: head first, then worker (`docker compose stop`), then `rm -f`.

| Time | spark02 MemAvailable | spark01 MemAvailable |
|---|---|---|
| Before shutdown | 14 GiB | 17 GiB |
| T+15s | 19 GiB | 19 GiB |
| T+60s | 19 GiB | 19 GiB |
| T+120s | 19 GiB | 19 GiB |
| T+300s | 19 GiB | 19 GiB |
| After reboot | 118 GiB | 117 GiB |

**Classification: Residual** (< 90 GiB threshold).  No spontaneous recovery
observed after graceful shutdown.  Recovery method: `sudo systemctl reboot` on
both nodes (uptime < 300 s before confirming memory recovery).

Full UMA recovery confirmed after reboot.

### 32.10 Operational note

The formal preset (`presets/step37-flash-nvfp4-tp2.env`) targets
`GPU_MEMORY_UTILIZATION=0.88`, consuming ~107 GiB of the 121.63 GiB GB10
UMA pool.  At this utilization level:

- Runtime MemAvail is ~14–17 GiB (OS + CPU use only)
- Post-shutdown UMA residue is ~102 GiB (Residual class)
- **Reboot is always required after each Step-3.7-Flash-NVFP4 serving session**
  to recover the GB10 UMA pool before the next workload

This is consistent with the documented GB10 UMA residue behaviour (§6, §14).
The cache-release workaround itself does not affect the shutdown residue —
the residue is an NVIDIA driver characteristic, not a vLLM allocator artefact.
