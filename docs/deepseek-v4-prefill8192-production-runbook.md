# DeepSeek-V4-Flash prefill8192 — Production Activation Runbook

| Field | Value |
|---|---|
| Status | `Rollback baseline` |
| Scope | Activation / shutdown / rollback operations for the prefill8192 (`4c41950c`) baseline |
| Current replacement | [`deepseek-v4-sm121-indexer-production.md`](deepseek-v4-sm121-indexer-production.md) (current production) |
| Last validated | 2026-06-25 (prefill8192 production activation) |
| Runtime or image identity | image config `4c41950c`; preset `../presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env` (SHA `593ba898`) |
| Historical relevance | The immediate rollback target for the current SM121-indexer production baseline; superseded as the active serving path on 2026-07-01 |

> **Scope note:** the operational commands in this runbook are for **rollback
> activation or historical reproduction of the prefill8192 baseline**, not the
> normal current serving path (see the current-production document above).

> **This document prepares production operations only. It does not activate the
> runtime.** The repository production package exists; live activation is a
> separate, explicitly-approved maintenance-window procedure described below.

## 1. Status and scope

- **Repository production package**: exists (production preset + this runbook +
  preset index entry). Repository production status does **not** start the runtime.
- **Runtime activation**: remains a separate, explicitly-approved step. No host is
  rebooted and no model is loaded by creating this package.
- **Approved envelope**: concurrency 1 only; prompts up to 131,072 tokens; typical
  output allowance 128 tokens; fixed 4 GiB fp8 KV; prefix cache disabled; MTP n=1;
  FULL_DECODE_ONLY; cudagraph capture `[2]`; TP=2 multiprocessing, one rank per
  physical node; transport NET/IB over RoCE; EP off, B12X off, Ray off.
- **Candidate provenance**: promoted from
  `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-validated-candidate-tp2.env`
  (candidate SHA-256 `ed13ee5e17668509f8e08072bcc206981bf43f497121380dc0000d780aa1f88a`).
- **Production preset path**:
  `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-prefill8192-production-tp2.env`
  (SHA-256 `fdf765ec5f6c8abaea4f4a302ef62f28add80d1fe906e5d0417ae277acb5deda`).
- **Image identity**: `vllm-spark:v023-stack-dsv4-sm12x-pr41834-exp-72261a7-tvmfam019`;
  image config `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`;
  image manifest `sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44`;
  source pin `72261a7af149fa5d3fe2ed2b9956e92590731012`.
- **Model identity**: `deepseek-ai/DeepSeek-V4-Flash`, official checkpoint, 46 shards.
- **M1 status**: `OPENWEBUI_TIMEOUT_CONFIGURED` — OpenWebUI v0.9.6,
  `AIOHTTP_CLIENT_TIMEOUT=240`, endpoint `http://192.168.0.200:8000/v1`, healthy.
- **M2 status**: `VLLM_MONITORING_CONFIGURED_WITH_LOG_GAPS` — Prometheus and both
  Spark node-exporters live; `vllm-spark01` scrape job present and expected DOWN
  while vLLM is offline; activation gate `vllm_serving_activation_gate = 0`.
- **Accepted monitoring log gaps**: graph fallback, graph recapture, per-rank
  health, NCCL errors, and CUDA errors are operator log checks (no confirmed
  exported Prometheus metric). Spark cAdvisor, Spark DCGM exporter, and homeserver
  cAdvisor are intentionally deferred and are **not** active.

## 2. Topology

| Component | Value |
|---|---|
| homeserver | `192.168.0.8` |
| spark01 management | `192.168.0.200` |
| spark01 RoCE | `10.10.10.1` |
| spark02 management | `192.168.0.201` |
| spark02 RoCE | `10.10.10.2` |
| SSH user | `bjk110` |
| HCA | `rocep1s0f0` |
| RDMA port | `1` |
| GID index | `3` |
| vLLM API | `192.168.0.200:8000` |
| distributed init port | `29500` |
| OpenWebUI endpoint | `http://192.168.0.200:8000/v1` |

## 3. Preconditions

Verify before any activation:

- repository `HEAD` is the intended commit and tracked tree is clean
- production preset SHA-256 matches
  `fdf765ec5f6c8abaea4f4a302ef62f28add80d1fe906e5d0417ae277acb5deda`
- candidate SHA-256 matches
  `ed13ee5e17668509f8e08072bcc206981bf43f497121380dc0000d780aa1f88a`
- validated baseline SHA-256
  `d116f132ca0087a6773ba9769134c46ace3e707fbe313456cca6ab34b46969b1`
- rollback L1 SHA-256
  `87ad3920b3e681ef48e3d4a183f2508157e3fc7a36719320f2f6a50c0b98a903`
- rollback L2 SHA-256
  `bffea15863867f5e7d4dfe698b761167e25033cd4a4846f2e9ee32b6c623a71b`
- image config equal on both nodes
  (`sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`)
- image manifest verified where available
  (`sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44`)
- source pin verified (`72261a7af149fa5d3fe2ed2b9956e92590731012`)
- official model identity confirmed (`deepseek-ai/DeepSeek-V4-Flash`)
- 46 shard count on both nodes
- identical relative model paths on both nodes
- maintenance-window approval obtained
- current vLLM requests drained
- rollback readiness confirmed
- OpenWebUI `AIOHTTP_CLIENT_TIMEOUT=240`
- Prometheus healthy; targets `node-local`, `node-spark01`, `node-spark02`,
  `prometheus` UP
- vLLM alert gate `vllm_serving_activation_gate = 0` before activation
- spark01 disk free: **warning below 22 GiB**, **stop below 20 GiB**
- preserve previous runtime artifacts (do not delete prior logs/results/backups)

## 4. Clean-start procedure

1. Confirm no model container is running on either node.
2. Record existing boot IDs on both nodes (`cat /proc/sys/kernel/random/boot_id`).
3. Reboot spark01 exactly once.
4. Reboot spark02 exactly once.
5. Verify both boot IDs changed.
6. Require `MemAvailable >= 110 GiB` per node.
7. Require swap usage zero; otherwise explain and stop.
8. Require no stale vLLM or Ray processes.
9. Require ports `8000` and `29500` free on the relevant node(s).
10. Verify the dedicated cache path (`./.cache/vllm`) is not a symlink and does not
    escape its project root.
11. Clear only the dedicated vLLM cache on both nodes.
12. Do not delete model, tokenizer, AOT, raw results, or unrelated caches.

## 5. Network and deployment verification

- RoCE interfaces (`enp1s0f0np0`) UP on both nodes
- RDMA link `ACTIVE` / `LINK_UP` (`rdma link`, `ibstat rocep1s0f0`)
- numeric RoCE ping both directions (`10.10.10.1` <-> `10.10.10.2`)
- correct HCA `rocep1s0f0`, RDMA port `1`, GID index `3`
- image identity equal on both nodes
- model identity (46 shards) equal on both nodes
- production preset synchronized byte-for-byte to both nodes
- no Ray processes; exactly one rank per node

## 6. Startup order

1. Start the worker (spark02 rank).
2. Verify worker startup and rendezvous readiness.
3. Start the head (spark01 rank).
4. Verify exactly one engine initialization.
5. Verify both ranks remain alive.
6. Verify API health returns HTTP 200.

No automatic retry. On failure, go to Section 10.

## 7. Runtime activation proof

Require log or runtime evidence for:

- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_MODEL_LEN=135168`
- fixed 4 GiB KV (`--kv-cache-memory-bytes 4294967296`, `--kv-cache-dtype fp8`)
- runtime KV capacity compatible with MML (expected ~159,445 tokens, ~1.18x)
- MTP n=1 (draft head present on both ranks)
- both-rank cudagraph capture
- FULL_DECODE_ONLY
- cudagraph capture `[2]`
- NET/IB selected
- no NET/Socket fallback
- graph fallback zero
- graph recapture zero
- no NCCL error
- no CUDA error

> Graph fallback, graph recapture, per-rank health, NCCL, and CUDA are **operator
> log checks** because they are not fully exported as confirmed Prometheus metrics.

## 8. Acceptance tests

Bounded tests (concurrency 1):

- API health
- arithmetic correctness
- English generation
- Korean generation
- Unicode handling
- speculative rejection-heavy request
- token accounting
- short generation
- one 32K or 64K retrieval test
- direct API streaming
- OpenWebUI streaming
- timeout behavior (no premature cut within the 240 s budget)
- no prompt truncation
- no malformed output

Do not require another 131K soak unless a material runtime difference is detected.

## 9. Monitoring activation

After vLLM health and acceptance pass:

- confirm Prometheus target `vllm-spark01` becomes UP
- inspect the available vLLM metric surface
- reclassify TTFT, request, KV-utilization, and speculative-decode metrics from
  live evidence
- verify both Spark node-exporter targets remain UP
- verify resource rules evaluate without error
- set `vllm_serving_activation_gate` from `0` to `1` **only after explicit
  production acceptance**
- validate `VLLMSpark01Down` behavior (fires only when UP-then-down while gated on)
- record the monitoring configuration SHA after gate activation

> The activation-gate change must be separately approved and recorded. This
> repository-package task does **not** change the gate; the gate remains `0`.

## 10. Stop conditions

Stop immediately on any of:

- health failure
- rank exit
- graph fallback
- graph recapture
- NCCL error
- CUDA error
- HTTP 400
- request timeout
- malformed output
- retrieval failure
- token-accounting violation
- prompt truncation
- `MemAvailable` below 12 GiB
- sustained swap-out
- PSI memory pressure
- spark01 disk below 20 GiB
- unexpected NET/Socket fallback
- incorrect preset or image identity

Preserve evidence before rollback. No automatic retry.

## 11. Rollback

1. Stop accepting new requests.
2. Preserve logs and monitoring evidence.
3. Stop the head gracefully.
4. Stop the worker.
5. Verify containers and processes are gone.
6. Select the rollback preset (see order below).
7. Reboot both Spark nodes to reclaim UVM.
8. Re-run the clean-memory and cache gates (Section 4).
9. Start the rollback configuration.
10. Run bounded acceptance checks (Section 8 subset).
11. Keep the vLLM monitoring gate disabled until rollback acceptance passes.

Rollback order (no image rebuild required):

1. committed 2048 validated baseline
   `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env`
2. graph-only L1
   `presets/deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env`
3. eager U0-RDMA L2
   `presets/deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env`

## 12. Controlled shutdown

- stop the head first
- stop the worker second
- verify processes and ports (`8000`, `29500`) are released
- record UVM retention (GB10 retains UVM after stop)
- require a reboot before another full model load if `MemAvailable` remains low
- leave node-exporter and Prometheus running

## 13. Evidence checklist

- [ ] repository `HEAD` SHA
- [ ] production preset SHA-256
- [ ] image config / manifest IDs (both nodes)
- [ ] model shard count (46, both nodes)
- [ ] boot IDs (pre/post, both nodes)
- [ ] clean-memory values (`MemAvailable >= 110 GiB`)
- [ ] network state (RoCE up, RDMA active, GID index 3)
- [ ] graph state (FULL_DECODE_ONLY, capture `[2]`, fallback/recapture 0)
- [ ] MTP state (n=1, both ranks)
- [ ] transport state (NET/IB, no Socket fallback)
- [ ] API results (health + correctness)
- [ ] retrieval result (one 32K/64K)
- [ ] monitoring target state (`vllm-spark01`, node-exporters)
- [ ] activation-gate state (record before/after if changed)
- [ ] minimum `MemAvailable` observed
- [ ] swap and PSI
- [ ] rollback readiness
- [ ] final classification
