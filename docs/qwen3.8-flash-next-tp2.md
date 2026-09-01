# Qwen3.8-Flash-Next-FP8 — Dual DGX Spark TP=2 (PRODUCTION-QUALIFIED — c1/c2)

| Field | Value |
|---|---|
| Status | `PRODUCTION-QUALIFIED -- c1/c2` -- Gate 0/1/2 passed at `MAX_NUM_SEQS=1` (c1, 2026-08-29); `MAX_NUM_SEQS=2` CUDA-graph production requalification PASSED (2026-08-31 -> 2026-09-01, §5.8); `MAX_NUM_SEQS=4` unbounded (c4) remains BLOCKED |
| Current state | Runtime-qualified at c1 (2026-08-29, §5.5) and, separately, production-requalified at `MAX_NUM_SEQS=2` with CUDA-graph capture `[1,2]` (2026-08-31 -> 2026-09-01, §5.8): bounded c2 Gate PASS, full 12-row/135-request llama-benchy suite (12/12 rows, 135/135 requests, c8 protocol/syntax/semantic 8/8, 0 errors/fatals), and a 4-hour c1/c2 soak (target 14400s, actual 14400.0001s; 1120/1120 records -- 640 model requests + 480 health; post-c1/post-c2 both PASS; 0 fatal/restart/OOM). `MAX_NUM_SEQS=4` unbounded concurrency is still BLOCKED (§5.5/§5.6, unchanged). Not running as a persistent service. |
| Scope | Production runtime recipe for `Qwen/Qwen3.8-Flash-Next-FP8` on 2x DGX Spark (GB10, SM121), TP=2, over 200 Gbps RoCE, at server-side `MAX_NUM_SEQS=2` |
| Production status | **Production-qualified runtime config at `MAX_NUM_SEQS=2` (§5.8). This is a validated runtime configuration, not an auto-start service** -- a launch still requires the worker-first procedure in §6 under this repository's normal operational authorization. Server-side `MAX_NUM_SEQS=2` means client concurrency above 2 (c4/c8) is accepted and queued by vLLM, not rejected -- it exercises queue pressure on a 2-slot server, not an independently qualified c4/c8 concurrency level. Do not raise `MAX_NUM_SEQS` above 2 or extend `cudagraph_capture_sizes` beyond `[1,2]` without separately re-qualifying. |
| Preset | [`presets/qwen3.8-flash-next-fp8-tp2-candidate.env`](../presets/qwen3.8-flash-next-fp8-tp2-candidate.env) |
| Compose overlay (required) | [`compose/qwen3.8-flash-next/docker-compose.candidate.yml`](../compose/qwen3.8-flash-next/docker-compose.candidate.yml) |
| Static verifier | [`scripts/diag/verify_qwen38_flash_next_recipe.py`](../scripts/diag/verify_qwen38_flash_next_recipe.py) |

> **Status note:** Gate 0/1/2 were executed at c1 (`MAX_NUM_SEQS=1`) on 2026-08-29 -- see the ledger
> in §5.5 for the raw evidence pointer. The `MAX_NUM_SEQS=2` CUDA-graph production requalification
> (§5.8) is a separate, later, bounded runtime result (2026-08-31 -> 2026-09-01) with its own full
> evidence trail -- it is what promotes this document's status to production-qualified, not the
> c1-only §5.5 ledger or the narrower 32K/c2 follow-up in §5.6. `MAX_NUM_SEQS=4` unbounded (c4) is
> still **BLOCKED**, not qualified -- do not cite this document as evidence of working concurrent
> inference beyond the bounded `MAX_NUM_SEQS=2` envelope recorded in §5.8.

This follows the same validation/Git policy used for DeepSeek-V4 in this repository (see
[`deepseek-v4-production.md`](deepseek-v4-production.md) and the static-only candidate docs it
links, e.g. [`h1z-b1s-indexer-route-candidate.md`](h1z-b1s-indexer-route-candidate.md)): prepare and
statically verify a candidate first, gate real hardware runtime behind explicit staged approval,
and never claim a status this document has not itself earned.

## 0. Model identity

- **Model:** `Qwen/Qwen3.8-Flash-Next-FP8` (official FP8 checkpoint). **Not** the BF16 checkpoint --
  BF16 is 360000192888 bytes (335.28 GiB) and does not fit safely across 2x121.63 GiB GB10 unified
  memory once framework/KV/activation overhead is included. See §8 for the full exclusion rationale.
- **Pinned HF revision (exact, immutable commit):** `970c569adaca6b35532111fd6b27351b2baefe50`
- **Weight size:** 185523317458 bytes (172.78 GiB)
- **Image (immutable, digest-pinned):**
  `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`
  (arm64 manifest). Per the official vLLM recipe (§10), this model is **not supported via a PyPI
  vLLM install** and requires vLLM >=0.28 or a nightly build -- this dedicated image, not the
  general-purpose `vllm-spark` images used elsewhere in this repository, satisfies that requirement.

## 1. Prerequisites

- Both `spark01` and `spark02` reachable over management LAN (`192.168.0.0/24`) and RoCE
  (`10.10.10.0/24`, interface `enp1s0f0np0`, HCA `rocep1s0f0`, 200 Gbps).
- Both nodes on the same clean tracked Git revision of this repository (`git status --porcelain`
  clean for tracked files; untracked historical artifacts are expected and unrelated -- see
  `CLAUDE.md`). Verify:
  ```bash
  git -C /home/bjk110/docker/vllm-spark rev-parse HEAD
  git -C /home/bjk110/docker/vllm-spark status --porcelain --untracked-files=no
  ```
  Run on both nodes; the revision must match and the second command must print nothing.
- `docker compose` v2 available on both nodes (`docker compose version`).
- `/dev/infiniband` and `/sys/class/infiniband` present on both nodes (bind-mounted by
  `docker-compose.yml`); verify with `ip -br addr`, `rdma link`, `ibdev2netdev`.
- >=200 GiB free disk per node for the 172.78 GiB checkpoint plus image layers and cache headroom.
- No conflicting workload already bound to `HOST_PORT=8000` or `MASTER_PORT=50000` on either node
  (`ss -ltnp | grep -E ':8000|:50000'`).
- Read-only inspect current container/GPU state before anything else:
  ```bash
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
  ```

## 2. Model acquisition (pinned revision, both nodes)

Download to the host path this preset expects, using the exact pinned revision -- do not download
`main` or omit `--revision`:

```bash
# Both spark01 and spark02 (each node needs the full weights for TP=2 -- see docs/dsv4-flash-tp2.md
# §2.3 for why TP does not reduce per-node disk footprint on this cluster's launch pattern).
huggingface-cli download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision 970c569adaca6b35532111fd6b27351b2baefe50 \
  --local-dir /home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8
```

If your `huggingface_hub` version uses the newer CLI name, the equivalent is:

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision 970c569adaca6b35532111fd6b27351b2baefe50 \
  --local-dir /home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8
```

For node-to-node sync instead of two independent downloads, prefer RoCE per this repository's
convention (`docs/architecture.md`, `README.md` deployment rules): download once, then
```bash
rsync -avP /home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8/ \
  spark02:/home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8/
```

### 2.1. Identity and checksum verification (both nodes, before first launch)

- Confirm the downloaded snapshot's resolved commit matches the pin exactly:
  ```bash
  cat /home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8/.cache/huggingface/download/*.metadata 2>/dev/null \
    || huggingface-cli scan-cache | grep -A2 Qwen3.8-Flash-Next-FP8
  ```
  or, if the snapshot was fetched into a `snapshots/<revision>/` layout, confirm the directory name
  under `snapshots/` is exactly `970c569adaca6b35532111fd6b27351b2baefe50`.
- Confirm total on-disk weight size is consistent with 185523317458 bytes (172.78 GiB) before
  trusting the checkpoint is complete:
  ```bash
  du -sb /home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8/*.safetensors | \
    awk '{sum+=$1} END {print sum, sum/1024/1024/1024 " GiB"}'
  ```
- Verify per-file integrity against the repo's published `*.safetensors` SHA256 manifest (HF model
  pages publish per-file hashes via the `?blob` API / `huggingface-cli`'s local metadata; do not
  hand-copy hashes from anywhere other than the official repo at the pinned revision):
  ```bash
  huggingface-cli scan-cache -v   # or: sha256sum path/to/each *.safetensors, compare against the
                                   # repo's published manifest for revision 970c569a...
  ```
  Treat any mismatch as a hard stop -- do not proceed to Gate0 with an unverified checkpoint.
- Confirm the same total byte count and (if computed) the same per-file hashes on **both** nodes.

## 3. Image acquisition (both nodes)

```bash
docker pull vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
docker inspect --format '{{.Id}}' \
  vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
```

The pulled digest is immutable by construction (Docker refuses to resolve a `@sha256:...` reference
to different content); confirm the local image ID printed above is identical on both nodes before
proceeding.

## 4. Static verification (no containers, no weights required)

Run before any live launch attempt:

```bash
python3 scripts/diag/verify_qwen38_flash_next_recipe.py
```

This checks the preset's pinned identities and conservative safety contract by parsing the `.env`
file directly (no `source`/shell execution), and -- if `docker compose` is available locally -- also
renders `docker compose ... config` for both the `head` and `worker` profiles and checks the
rendered role/env wiring. It does not start any container and does not require the model weights to
be present. See the script's own header for the exact check list.

## 5. Staged runtime gates

These gates are **not executed by this document** -- they define the procedure a separately
authorized runtime session must follow, in the spirit of the DeepSeek-V4 static probes in this
repository (e.g. `scripts/diag/dsv4_fullgraph_safety_probe.py`, `scripts/diag/dsv4_mtp1_fullgraph_safety_probe.py`).
**Stop immediately on the first anomaly at any gate -- no retry, no silent continuation to the next
gate.** Persist raw evidence (logs, full response bodies, checksums) for every gate that runs.

### Gate 0 -- Static / provenance / network / image / model identity

- Repository revision identical on both nodes (§1).
- `docker inspect` image ID identical on both nodes and matches the pinned digest (§3).
- Model snapshot revision, byte count, and (if computed) per-file checksums identical on both nodes
  and match §2.1.
- RoCE link up on both nodes at expected line rate (`ibstat` / `ethtool enp1s0f0np0`), both
  `10.10.10.1` and `10.10.10.2` pingable over the RoCE interface.
- `scripts/diag/verify_qwen38_flash_next_recipe.py` exits 0.
- **No engine process is started in this gate.**

### Gate 1 -- Startup / health and one canonical greedy request

- Worker-first launch (§6), then head. Wait for `head`'s Docker healthcheck to report `healthy`
  and for `GET /health` to return HTTP 200.
- `GET /v1/models` lists `Qwen/Qwen3.8-Flash-Next-FP8` with `max_model_len: 262144`.
- Exactly one canonical greedy request (temperature 0, fixed short prompt, small `max_tokens`),
  HTTP 200, well-formed JSON, non-empty `choices[0].message.content`, `finish_reason` sane.
- Persist: full startup log from both nodes, the exact request/response pair.

### Gate 2 -- Canonical 6 repeats

- Repeat the exact Gate 1 canonical request 6 times sequentially (same prompt, same sampling
  params, temperature 0), with a health check before and after each repeat.
- All 6 must be HTTP 200, well-formed, free of the replacement character (`�`) and of
  degenerate repetition.
- Persist all 6 full raw responses.

### Gate 3 -- Deterministic fixed-byte payload repeats (only after Gate 2 passes)

- Define one or more fixed-byte request payloads (exact bytes, not regenerated per run).
- **Concurrency 1 (c1) first.** Repeat each fixed payload N times at c1 (temperature 0). Persist
  every full raw response body and its SHA256 checksum. All N checksums for a given payload must be
  identical (byte-for-byte determinism) -- any divergence is a hard stop, not a retry.
- **Only after c1 passes fully**, repeat at c2 and then c4. Same persistence and checksum
  requirement at each concurrency level.
- **Stop immediately** on any HTTP error, malformed response, checksum divergence within a
  concurrency level, or any corruption signal (replacement characters, degenerate repetition,
  truncated JSON). Do not proceed to the next concurrency level or payload after a stop.
- This gate is the determinism precondition for ever considering the MTP profile in §7 -- MTP must
  not be evaluated on a config that has not itself passed Gate 3 without speculation.

Gate 0/1/2 have been executed at c1 (`MAX_NUM_SEQS=1`) -- see §5.5 for the raw ledger. This
document's Gate 3 procedure describes the fixed-byte c1->c2->c4 determinism sweep in the abstract;
the concrete result that promotes this document to production status is the separate, bounded
`MAX_NUM_SEQS=2` CUDA-graph requalification in §5.8 (2026-08-31 -> 2026-09-01), not a completion of
this abstract Gate 3 sweep at c4. `MAX_NUM_SEQS=4` unbounded concurrency remains BLOCKED per §5.5/
§5.6; do not treat it as qualified. Further gate execution beyond what §5.5/§5.8 record requires
separate explicit authorization per this repository's confirmation policy (`CLAUDE.md`).

## 5.5. Runtime ledger -- 2026-08-29 (Gate 1/2 qualification result)

- **Identity/sync verified.** Both nodes matched 144 top-level files, 185563783486 total bytes, and all 133 official LFS SHA-256 entries. Repair evidence: `/home/bjk110/docker-build/qwen38-flash-next-fp8-sync-repair-20260829/`.
- **Gate 0/1/2 QUALIFIED at c1 (`MAX_NUM_SEQS=1`).** Readiness took ~18m27s in the qualifying run,
  consistent with a measured ~17-19 min readiness range; the test harness's outer timeout was 45
  min. All 6/6 canonical Gate 2 repeats returned HTTP 200, non-empty content, and the identical
  semantic-content SHA256 `218b829d8f5d3b0531392550884c4fcef65a343976d96ce54983bfb1c068415c`. No
  fatal CUDA/NCCL/Xid signal; cleanup on both nodes completed fully. Evidence:
  `/home/bjk110/docker-build/qwen38-fp8-gate12-20260829/`.
- **`MAX_NUM_SEQS=4` (c2/c4) is BLOCKED -- do not use.** With thinking disabled, c1 was internally
  consistent (4/4 identical hashes) but c2/c4 diverged: 5 distinct hashes across 18 HTTP 200
  responses, including some responses ending at different lengths/finish reasons. Evidence:
  `/home/bjk110/docker-build/qwen38-fp8-c4-thinking-diagnostic-20260829/`.
- **Thinking-enabled diagnostics also failed.** The 256-token run and 1024-token diagnostic both
  exhausted the full budget as reasoning, returned null content, and ended with `finish_reason=length`.
  Evidence: `/home/bjk110/docker-build/qwen38-fp8-c4-gate3-20260829/` and
  `/home/bjk110/docker-build/qwen38-fp8-c4-thinking-diagnostic-20260829/`.
- No fatal CUDA/NCCL/Xid signal was observed in the qualifying c1 run or in either blocked c4
  investigation; cleanup completed on both nodes in all cases.
- **Client bounded-output guidance:** until `MAX_NUM_SEQS=4` is separately requalified, callers must
  pass `chat_template_kwargs: {"enable_thinking": false}` explicitly. Thinking budgets above 1024
  tokens remain entirely unqualified (not only at c4) -- do not assume a larger budget resolves the
  null-content failure above without separately measuring it.
- **MTP:** off in this preset and untested -- investigation stopped at the `MAX_NUM_SEQS=4` blocker
  before MTP was ever reached.
- **Historical note (superseded by §5.8):** at the time this ledger was recorded (2026-08-29),
  `MAX_NUM_SEQS=1` (c1) was the only qualified value and nothing here authorized a persistent service
  or a promotion. That has since changed -- §5.8 (2026-08-31 -> 2026-09-01) separately production-
  requalified a bounded `MAX_NUM_SEQS=2` CUDA-graph envelope. This ledger's own c1 findings and the
  `MAX_NUM_SEQS=4` (c2/c4) BLOCKED finding above are unchanged and remain historically accurate.

## 5.6. Follow-up ledger -- 2026-08-29 (32K/c2 exact-output stability probe; narrower scope than Gate 3, still NOT a concurrent-serving qualification)

> This is a bounded protocol-stability observation at `MAX_MODEL_LEN=32768` and `MAX_NUM_SEQS=2`,
> run as a single-variable diagnostic series distinct from -- and narrower than -- the Gate 3
> concurrency procedure in §5. It does **not** requalify `MAX_NUM_SEQS>1` at this candidate's
> `MAX_MODEL_LEN=262144` default, and it does not authorize a c2 preset. See "Supervisor stopping
> point" below.

- **(1) Baseline probe -- 32768 context, `MAX_NUM_SEQS=2`, prefix caching ON.** Exact-output
  fixed-byte payload: c1 4/4 identical, c2 10/10 identical. Code-generation payload: c1 4/4
  identical; c2 diverged into 3 distinct semantic hashes across 6 responses. All responses HTTP 200,
  `finish_reason: stop`, and (for the code payload) AST-valid. Evidence:
  `/home/bjk110/docker-build/qwen38-fp8-32k-c2-prefixon-20260829/`.
- **(2) Prefix caching OFF, single-variable.** Exact-output payload remained deterministic. Code
  payload: c1 4/4 identical; c2 diverged into 3 distinct hashes (the harness observed 4 distinct hashes total
  when the c1 hash is included). No fatal CUDA/NCCL/Xid signal. Evidence:
  `/home/bjk110/docker-build/qwen38-fp8-32k-c2-prefixoff-20260829/`. Prefix caching is **excluded**
  as the root cause of the c2 code-generation divergence.
- **(3) Expert parallel OFF, single-variable.** Exact-output payload remained deterministic. Code
  payload: c1 4/4 identical; c2 diverged into 3 distinct hashes (the harness observed 4 distinct hashes total
  when the c1 hash is included). Evidence: `/home/bjk110/docker-build/qwen38-fp8-32k-c2-noep-20260829/`.
  Expert-parallel / all-to-all collective path is **excluded** as the root cause.
- **(4) CUTLASS FP8 MoE backend trial -- failed before readiness.** Exact `ValueError`:
  `vLLM CUTLASS FP8 MoE backend is disabled for this configuration.` Evidence:
  `/home/bjk110/docker-build/qwen38-fp8-32k-c2-cutlass-20260829/`.
- All four runs cleaned up containers and released `HOST_PORT`/`MASTER_PORT` on both nodes; nothing
  left running.

**Supervisor stopping point (2026-08-29):** the 32K/c2 exact-output determinism above is a bounded
protocol-stability observation, not a general concurrent-serving qualification -- open-ended
generation (the code-generation payload) remains nondeterministic under `MAX_NUM_SEQS=2` at this
context length. Per this stopping point (as it stood on 2026-08-29): **do not** create or promote a
c2 preset from this result, **do not** run MTP, and **retain** the `MAX_MODEL_LEN=262144` /
`MAX_NUM_SEQS=1` default candidate above as-is. DeepGEMM / batch-dependent MoE numeric behavior
remains the leading inference for the c2 code-generation divergence but is **unproven** -- prefix
caching and expert-parallel/all-to-all have both been excluded by the single-variable probes above.

**Historical note (superseded by §5.8):** this 32K/c2 follow-up predates and is distinct from the
`MAX_NUM_SEQS=2` CUDA-graph production requalification in §5.8 (2026-08-31 -> 2026-09-01), which ran
at this candidate's actual `MAX_MODEL_LEN=262144` default (not the narrower 32768 probed here) and is
the result that actually promotes `MAX_NUM_SEQS=2` to production. The findings above (prefix caching
and expert-parallel excluded as root cause of 32K code-generation divergence; DeepGEMM/batch-
dependent MoE numerics unproven as leading hypothesis) remain historically accurate for the narrower
32K scope and are not retracted, but "do not create or promote a c2 preset" no longer describes this
document's current status -- see §5.8.

## 5.7. Follow-up -- 2026-08-29 (batch-invariant support preflight; static check, no runtime launch)

> Static inspection only -- an ephemeral container was used to read the pinned image's installed
> source, but **no model-serving container or engine was launched**. This is a source-level preflight
> support check, not a runtime gate, and does not itself constitute an observed runtime failure.

- **Scope:** whether `VLLM_BATCH_INVARIANT=1` is usable with this candidate's pinned image and
  backend, checked before considering it as a possible route past the c2/c4 concurrency blocker in
  §5.5/§5.6.
- **Pinned identities re-confirmed (unchanged from §0/§3):** image digest
  `sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`; vLLM
  `0.1.dev20073+g8e685d198`.
- **Finding.** Static inspection of the pinned image's vLLM source showed
  `GDNAttentionBackend.supports_batch_invariance` resolves to the inherited
  `AttentionBackend.supports_batch_invariance` (i.e. `GDNAttentionBackend` defines no override) and
  that inherited method returns `False`. Qwen3.8-Flash-Next uses the GDN (Gated DeltaNet) attention
  backend, so `VLLM_BATCH_INVARIANT=1` is **unsupported** for this model on this pinned official
  image.
- **Consequence -- fail-closed, no launch attempted.** Per this repository's fail-closed policy, the
  32K/c2 runtime gate was **deliberately not launched** with `VLLM_BATCH_INVARIANT=1`, because doing
  so would be rejected by vLLM's own startup check with
  `VLLM batch_invariant mode is not supported for GDN_ATTN`. This is a **static preflight stop**, not
  an observed runtime failure -- no engine process was started and the rejection above was never
  actually triggered live; the classification is derived from source inspection alone.
- **No workaround attempted.** No image rebuild, no backport of a `GDNAttentionBackend` override, and
  no nonofficial patch was attempted -- this stays within the "official pinned image only" posture of
  this candidate (§0).
- **Does not change anything else in this document.** This finding is independent of the
  `MAX_NUM_SEQS=2` CUDA-graph production requalification in §5.8 -- `VLLM_BATCH_INVARIANT` plays no
  role in that recipe either.
  `presets/qwen3.8-flash-next-fp8-tp2-candidate.env` does not set `VLLM_BATCH_INVARIANT`.
  `MAX_NUM_SEQS=4` unbounded concurrency remains **BLOCKED** per §5.5/§5.6 -- `VLLM_BATCH_INVARIANT=1`
  was never a candidate for resolving that blocker on this backend, and this finding forecloses it as
  a future option on this image without an unofficial patch (not attempted, not authorized). The
  bounded `MAX_NUM_SEQS=2` envelope is separately production-qualified via CUDA-graph capture, not
  batch-invariant mode -- see §5.8.
- Evidence: `/home/bjk110/docker-build/qwen38-fp8-bic-support-check-20260829/`.

## 5.8. Production requalification -- 2026-08-31 -> 2026-09-01 (`MAX_NUM_SEQS=2`, CUDA-graph `FULL_DECODE_ONLY` capture `[1,2]`)

> This is the runtime result that promotes this document and
> `presets/qwen3.8-flash-next-fp8-tp2-candidate.env` to production status. It is distinct from, and
> later than, both the c1-only §5.5 ledger and the narrower 32K/c2 protocol-stability follow-up in
> §5.6. It runs at this candidate's actual `MAX_MODEL_LEN=262144` default, replaces `--enforce-eager`
> with `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}`,
> and raises the server-side `MAX_NUM_SEQS` pin from `1` to `2`.

- **Bounded c2 Gate PASS.** A dedicated c1/c2 gate (canonical, coding, tool-call, tool-continuation,
  long-context sentinel probes) passed at `MAX_NUM_SEQS=2` with the CUDA-graph capture config above.
- **Full 12-row/135-request llama-benchy suite -- 12/12 rows, 135/135 requests, 0 errors.** Standard
  `{pp512, pp2048, pp8192} x {c1, c2, c4, c8}` llama-benchy matrix (12 rows; each row's own configured
  repeats sum to 135 total HTTP requests across the run, all HTTP 200/`request_end`, 0 tracked
  errors). The c8 row additionally ran an explicit 8-request protocol/syntax/semantic gate:
  8/8 `protocol_valid`, 8/8 `syntax_valid`, 8/8 `semantic_valid`, 0 fail. No fatal CUDA/NCCL/Xid
  signal; both nodes torn down cleanly (`exit_code` 0). Evidence:
  `/home/bjk110/docker-build/qwen38-graphonly-c2-capture12-fullbench-k1031-20260831/` (see
  `llama-benchy-results.json` for the 12 rows, `llama-benchy-progress.jsonl` for the 135
  `request_end` records, and `gate-c8-summary.json` for the 8/8 c8 protocol/syntax/semantic result).
- **4-hour c1/c2 soak -- target 14400s, actual 14400.0001s; 1120/1120 records; 0 fatal/restart/OOM.**
  A continuous c1/c2 soak (`stability4h-summary.json`: `stop_reason: "completed_duration"`,
  `target_duration_s: 14400.0`, `actual_duration_s: 14400.000115...`) recorded 1120/1120 total
  records with 0 failures in every category: 640 model-request records (480 canonical + 48 coding +
  48 tool-call + 48 tool-continuation + 16 long-context sentinel) plus 480 health-check records.
  `post-c1-summary.json` and `post-c2-summary.json` both report `"status": "PASS"`. No restart, no
  OOM, and no fatal/Xid signal was found in either node's dmesg or the run's heartbeat log (the only
  `oom` string match in either dmesg log is the unrelated `systemd-oomd.socket` listener line, not an
  actual OOM event). Current top-level outputs at
  `/home/bjk110/docker-build/qwen38-graphonly-c2-capture12-soak4h-k1031-20260901/` are this
  **successful retry** -- an earlier same-day attempt hit a writer-race condition and was moved into
  that same directory's own `archive/pre-run-20260901T022341Z/` and
  `archive/pre-run-20260901T030512Z/` subdirectories rather than being deleted; the current top-level
  `stability4h-summary.json`, `heartbeat.jsonl`, `requests.jsonl`, `node-monitor.jsonl`, and gate
  outputs are all from the successful retry, not the archived attempt.
- **Server-cap limitation -- c4/c8 client load is queue pressure, not a separately qualified
  concurrency level.** `MAX_NUM_SEQS=2` is a server-side scheduling cap; the fullbench suite's c4/c8
  rows and the soak's own health/canonical polling exercised the server under client concurrency
  above 2, but that only demonstrates the 2-slot server queues and drains correctly under load -- it
  is not evidence that `MAX_NUM_SEQS=4` (unbounded, §5.5/§5.6) is itself qualified. Do not cite this
  section as resolving the `MAX_NUM_SEQS=4` blocker.
- **Memory headroom -- minimum ~9.9 GB (spark01) / ~14.2 GB (spark02) `MemAvailable` mid-run.** The
  soak's own `min_mem_available_bytes` are `9891164160` (spark01, ~9.9 GB) and `14246248448`
  (spark02, ~14.2 GB); both are decimal-GB (1e9-byte) figures. This is narrower headroom than the
  original `MAX_NUM_SEQS=1` envelope and should be treated as this recipe's practical floor, not a
  large margin.
- **Teardown / operational note -- ~19 GB `MemAvailable` after stop; reboot before next fresh
  launch.** Both `final-state.txt` snapshots (fullbench and soak4h runs) show `MemAvailable` in the
  ~18.3-19.4 GB range immediately after `docker compose ... down` on both nodes -- consistent with
  this repository's known GB10 UMA-recovery lesson that vLLM teardown does not release the full
  ~100 GiB back to the host and only a reboot does. Reboot spark01/spark02 before the next fresh
  launch of this or any other memory-heavy runtime if that low-headroom state is still current at
  launch time.
- **This is a validated runtime configuration, not an auto-start service.** Nothing in this section
  starts, restarts, or leaves running any container -- promotion to production status means this
  preset's values are qualified for a future launch via the worker-first procedure in §6 under this
  repository's normal operational authorization, not that a service is currently running.
- Evidence:
  `/home/bjk110/docker-build/qwen38-graphonly-c2-capture12-fullbench-k1031-20260831/`,
  `/home/bjk110/docker-build/qwen38-graphonly-c2-capture12-soak4h-k1031-20260901/`.

## 6. Worker-first launch (exact commands)

Run from the repository root. **Start the worker before the head** -- this is TP=2 startup, not two
independent servers. `--no-deps` is used so Compose does not try to (re)create or depend-order any
other service defined by the base file for a profile that only needs `worker` or `head`:

```bash
# spark02 — worker first
docker compose \
  --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  --profile worker up -d --no-deps

# spark01 — head
docker compose \
  --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  --profile head up -d --no-deps
```

To validate the rendered configuration without starting anything (safe, repeatable, no weights
required):

```bash
docker compose --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  --profile head config
docker compose --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  --profile worker config
```

## 7. Health / readiness checks

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
docker inspect --format '{{.State.Health.Status}}' vllm-spark-head
```

`/health` means the engine process is alive, not that any warmup has completed -- this candidate
has no automatic prewarm (§ compose overlay header). Measured readiness (Gate 1/2 qualification,
2026-08-29, c1): ~17-19 min to healthy/ready, with the qualifying run reaching readiness at
~18m27s; the test harness used a 45 min outer timeout as its safety bound. The healthcheck's 900s
(15 min) `start_period` is close to this measured range rather than generously above it -- treat it
as tight, and prefer a ~45 min outer wait bound for any automated readiness check rather than
relying on `start_period` alone.

## 8. Stop / cleanup

```bash
# Both nodes, in either order once you intend to fully stop:
docker compose \
  --env-file presets/qwen3.8-flash-next-fp8-tp2-candidate.env \
  -f docker-compose.yml \
  -f compose/qwen3.8-flash-next/docker-compose.candidate.yml \
  --profile head --profile worker down
```

`restart: "no"` is inherited from the base `docker-compose.yml` (see that file's own comment on why
automatic restart is intentionally disabled on this cluster) -- a crashed container will not
auto-restart; a manual `docker compose ... up -d` is required to relaunch.

## 9. MTP -- explicitly experimental, later profile (not part of this default)

Multi-token prediction / speculative decoding is **not** part of this conservative candidate.
`VLLM_EXTRA_ARGS` in `presets/qwen3.8-flash-next-fp8-tp2-candidate.env` deliberately has no
`--speculative-config`. A future MTP profile is a **separate preset and a separate document
section**, and must not be evaluated until this conservative candidate has itself passed Gate 3
(§5) without speculation -- mirroring this repository's own DeepSeek-V4 experience, where MTP
measurably reduced decode throughput in one measured configuration
(`dsv4-flash-tp2.md` §7) despite normal-looking acceptance rates. Do not assume MTP is a strict
improvement for this model without measuring it the same deliberate way.

## 10. Known limitations

- **Production-qualified at `MAX_NUM_SEQS=2` with CUDA-graph capture `[1,2]`; `MAX_NUM_SEQS=4`
  unbounded remains BLOCKED.** Gate 0/1/2 passed at `MAX_NUM_SEQS=1` (c1) on 2026-08-29 (§5.5); the
  `MAX_NUM_SEQS=2` production requalification (§5.8, 2026-08-31 -> 2026-09-01) separately passed a
  bounded c2 gate, a full 12-row/135-request llama-benchy suite, and a 4-hour c1/c2 soak. `MAX_NUM_SEQS=4`
  (unbounded c4) is still BLOCKED -- content diverged across repeats in the 2026-08-29 ledger, and
  thinking mode returned null content at both a 256- and a 1024-token budget. Do not raise
  `MAX_NUM_SEQS` above 2, or extend `cudagraph_capture_sizes` beyond `[1,2]`, without separately
  requalifying.
- **Server-side cap, not a client concurrency ceiling.** `MAX_NUM_SEQS=2` bounds how many sequences
  vLLM schedules concurrently; client requests at c4/c8 are accepted and queued, not rejected. The
  §5.8 fullbench suite's own c4/c8 rows exercise this queuing behavior, not an independently
  qualified c4/c8 serving concurrency -- do not present c4/c8 results from that evidence as a
  qualified higher-concurrency recipe.
- **KV/activation memory floor measured tighter than the original conservative estimate.** The §5.8
  4-hour soak measured minimum `MemAvailable` of ~9.9 GB (spark01) / ~14.2 GB (spark02) mid-run, and
  ~18.3-19.4 GB on both nodes immediately after teardown -- narrower than headroom was assumed to be
  under the original `MAX_NUM_SEQS=1`/`--enforce-eager` envelope. Per this repository's GB10
  UMA-recovery lesson, teardown does not release the full memory back to the host; reboot spark01/
  spark02 before the next fresh launch if that low-headroom post-teardown state is still current.
- **Production default is a validated runtime configuration, not an auto-start service.** Nothing in
  this promotion starts, restarts, or schedules a persistent service -- a launch still requires the
  explicit worker-first procedure in §6 under this repository's normal operational authorization.
- **`GPU_MEMORY_UTILIZATION=0.83` and `MAX_NUM_BATCHED_TOKENS=8192` remain untuned.** These two values
  come from the original known dual-Spark community command, not from this cluster's own
  measurement, and were not changed by the §5.8 requalification. `MAX_NUM_SEQS` and the CUDA-graph
  capture config, by contrast, are now this cluster's own measured values. Expect
  `GPU_MEMORY_UTILIZATION`/`MAX_NUM_BATCHED_TOKENS` to be conservative rather than optimal.
- **KV/activation headroom is tight by construction.** 172.78 GiB of FP8 weights split across 2
  nodes is roughly 86.4 GiB/node before KV cache and activations; `GPU_MEMORY_UTILIZATION=0.83` of
  121.63 GiB is ~100.9 GiB/node. This leaves a narrow (unmeasured) margin for KV cache and
  activations at `MAX_MODEL_LEN=262144` -- if Gate 1 fails with an out-of-memory error, the
  conservative next step is lowering `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION`, not raising
  `MAX_NUM_SEQS`.
- **`mp` backend command-line shape is comparatively less battle-tested** in this repository than
  `ray` (see `dsv4-flash-tp2.md` §5) -- it is used here because it is what the known dual-Spark
  community command and the current entrypoint's `--nnodes`/`--node-rank`/`--master-addr`/
  `--headless` support target.
- **PyPI install of this model's vLLM support is unsupported** per the official recipe (§0); the
  dedicated pinned image is required. Do not substitute a general-purpose `vllm-spark` image.
- **vLLM PR #53896 was open at investigation time** (see §11) -- re-check its status before
  treating anything it touches as settled upstream behavior.
- Full multimodal support is retained (no `--language-model-only`); this candidate has not been
  exercised with any multimodal input, only the text-only Gate 1/2/3 procedure in §5.
- **32K/c2 follow-up (2026-08-29, §5.6) -- historical, superseded by §5.8 for current status.** At
  `MAX_MODEL_LEN=32768`, fixed-byte exact-output payloads were deterministic at c1 and c2 with
  prefix caching on and off; open-ended code generation still diverged into distinct outputs at c2
  with prefix caching on, off, and with expert-parallel off, excluding those three variables as root
  cause. A CUTLASS FP8 MoE backend trial failed before reaching readiness (`ValueError: vLLM CUTLASS
  FP8 MoE backend is disabled for this configuration.`). DeepGEMM / batch-dependent MoE numeric
  behavior remains the leading unproven hypothesis for that narrower 32K scope. This follow-up by
  itself did not requalify `MAX_NUM_SEQS>1` at this candidate's actual `MAX_MODEL_LEN=262144`
  default -- the later §5.8 CUDA-graph requalification (2026-08-31 -> 2026-09-01), run at
  `MAX_MODEL_LEN=262144`, is what actually qualifies `MAX_NUM_SEQS=2` for production.
- **`VLLM_BATCH_INVARIANT=1` is unsupported on this pinned image (2026-08-29 static preflight, §5.7).**
  `GDNAttentionBackend.supports_batch_invariance` resolves to the inherited (always-`False`)
  `AttentionBackend.supports_batch_invariance` -- vLLM would reject batch-invariant mode for this
  GDN backend. This was a static source-inspection finding, not an observed runtime failure; the
  32K/c2 runtime gate was deliberately not launched with this flag. Not a route past the c2/c4
  concurrency blocker above without an unofficial patch (not attempted, not authorized).

## 11. Citations

- Official model card: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next`
- Official FP8 card: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8`
- Official vLLM recipe: `https://qwen.readthedocs.io/en/latest/deployment/vllm.html`
- vLLM PR #53896 (open at investigation time): `https://github.com/vllm-project/vllm/pull/53896`
- vLLM issue #53960 (PLE-offload deadlock): `https://github.com/vllm-project/vllm/issues/53960`

## 12. Excluded alternatives

- **BF16 (`Qwen/Qwen3.8-Flash-Next`, full precision):** 360000192888 bytes (335.28 GiB) exceeds
  2x121.63 GiB = 243.26 GiB of combined GB10 UMA even before accounting for KV cache, activations,
  CUDA/driver reserve, and framework overhead on both nodes. Excluded from the default candidate for
  this reason alone; not re-evaluated here.
- **PLE-offloaded NVFP4:** excluded per the vLLM issue #53960 PLE-offload deadlock citation above
  (§11) and per explicit supervisor direction for this conservative candidate. Not used anywhere in
  this preset or its overlay.

## 13. Status (repeat, for anyone skimming to the bottom)

**PRODUCTION-QUALIFIED -- c1/c2.** Gate 0/1/2 passed at `MAX_NUM_SEQS=1` (c1) on 2026-08-29 (§5.5).
The `MAX_NUM_SEQS=2` CUDA-graph production requalification (§5.8, 2026-08-31 -> 2026-09-01) --
`--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}`
in place of `--enforce-eager` -- separately passed a bounded c2 gate, a full 12-row/135-request
llama-benchy suite (12/12 rows, 135/135 requests, c8 protocol/syntax/semantic 8/8, 0 errors/fatals),
and a 4-hour c1/c2 soak (target 14400s, actual 14400.0001s; 1120/1120 records -- 640 model requests +
480 health; post-c1/post-c2 both PASS; 0 fatal/restart/OOM). `MAX_NUM_SEQS=4` unbounded concurrency
remains BLOCKED (§5.5/§5.6) -- do not raise `MAX_NUM_SEQS` above 2 or extend
`cudagraph_capture_sizes` beyond `[1,2]` without separately requalifying. `MAX_NUM_SEQS=2` is a
server-side cap, not a client ceiling -- c4/c8 client load is accepted and queued, not an
independently qualified concurrency level. Minimum measured `MemAvailable` during the §5.8
qualification was ~9.9 GB (spark01) / ~14.2 GB (spark02); teardown leaves ~19 GB on both nodes --
reboot before the next fresh launch if that low-headroom state is still current. This is a validated
production **runtime configuration**, not an auto-start service -- a launch still requires the
explicit worker-first procedure in §6 under this repository's normal operational authorization. No
MTP in this preset (§9, unchanged).
