# DeepSeek-V4-Flash-Vision-Exp: UNVALIDATED dual DGX Spark TP2

This is an attended, bounded correctness experiment, **UNVALIDATED on GB10**.
The supplied NVIDIA validation scope is **GB200 TP4+EP, NOT GB10**. Dual GB10
TP2 is experimental; static checks establish neither kernel support nor memory
fit, multimodal correctness, or RoCE transport. No autostart or promotion.
The pins below are user-supplied facts, not independently acquired runtime evidence.

Use [the preset](../presets/deepseek-v4-flash-vision-exp-tp2.env) with the
[required overlay](../compose/deepseek-v4/docker-compose.vision-exp.yml).
Initial envelope: 32768 context, one sequence, FP8 KV, block 256, TP2 MP/RoCE,
expert parallel, prefix cache disabled, eager execution, at most one image and
no video. No DSpark, draft model, MTP, or speculative decoding. Do not reuse
text-only shims, skip multimodal profiling, or bypass memory checks to obtain PASS.
The tool/reasoning parser names are experimental CLI assumptions and must be
confirmed against this exact image before launch; unsupported flags are a blocker.

## Pinned acquisition

On 2026-10-03 the operator explicitly authorized removal of the previously live
Qwen service, old model snapshots, and unused serving images to make room for this
experiment. Acquisition is authorized; launch and promotion remain separate gates.
Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, HF revision
`6821d6ad3681a4b137b066b76094fa82ebd0a380`, **48 shards / 167819404368 bytes**
(shard total, not whole snapshot disk usage). Allow space for metadata, caches,
and transfers in addition to this total. Download the complete snapshot including
processor, tokenizer, configuration and template files, without filtering it to weights.
The 48 official LFS object hashes and sizes queried at that immutable revision are
recorded in [`provenance/deepseek-v4-flash-vision-exp-6821d6ad-weights.json`](../provenance/deepseek-v4-flash-vision-exp-6821d6ad-weights.json).

```bash
# homeserver source of truth; do not replace the revision with a moving ref.
export HF_XET_HIGH_PERFORMANCE=1
/home/bjk110/.venvs/hf/bin/hf download \
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --local-dir /mnt/data/llm-models/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash-Vision-Exp

# After homeserver verification, deploy homeserver -> spark01 over management LAN.
rsync -a --delete --exclude=.cache/ \
  /mnt/data/llm-models/deepseek-ai/deepseek-ai_DeepSeek-V4-Flash-Vision-Exp/ \
  spark01:/home/bjk110/Documents/Models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/

# On spark01, deploy spark01 -> spark02 over RoCE.
rsync -a --delete --exclude=.cache/ \
  /home/bjk110/Documents/Models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/ \
  10.10.10.2:/home/bjk110/Documents/Models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/

# On homeserver, acquire the immutable arm64 image and export it once.
IMAGE_REF=vllm/vllm-openai@sha256:0075fd82e3b6d943b0aa91e35da8dbca63d88516c607745131055e9d81f37ebb
IMAGE_TAG=vllm/vllm-openai:deepseekv4-flash-vision-arm64-0075fd82
IMAGE_TAR=/mnt/data/llm-images/vllm/vllm-openai-deepseekv4-flash-vision-arm64-0075fd82.tar
docker pull --platform linux/arm64 "$IMAGE_REF"
docker tag "$IMAGE_REF" "$IMAGE_TAG"
test "$(docker image inspect --format '{{.Architecture}}' "$IMAGE_TAG")" = arm64
docker save -o "$IMAGE_TAR.part" "$IMAGE_TAG"
mv "$IMAGE_TAR.part" "$IMAGE_TAR"
(cd "$(dirname "$IMAGE_TAR")" && \
  sha256sum "$(basename "$IMAGE_TAR")" > "$(basename "$IMAGE_TAR").sha256")

# Deploy the verified archive homeserver -> spark01, then spark01 -> spark02
# over RoCE. Verify SHA256 before loading on each node.
ssh spark01 mkdir -p /home/bjk110/docker-images
rsync -a "$IMAGE_TAR" "$IMAGE_TAR.sha256" spark01:/home/bjk110/docker-images/
# Run the following on spark01:
(cd /home/bjk110/docker-images && sha256sum -c vllm-openai-deepseekv4-flash-vision-arm64-0075fd82.tar.sha256)
docker load -i /home/bjk110/docker-images/vllm-openai-deepseekv4-flash-vision-arm64-0075fd82.tar
rsync -a /home/bjk110/docker-images/ 10.10.10.2:/home/bjk110/docker-images/
# Run the same exact sha256sum -c and docker load commands on spark02, then
# compare image ID and Architecture on both nodes before Gate 0.
```

The multiarch digest is `sha256:0075fd82e3b6d943b0aa91e35da8dbca63d88516c607745131055e9d81f37ebb`;
the expected arm64 manifest is
`sha256:8568b4bbc821903d93a0a9c17dd80382fdc0ba78eaa128e3eb5cb71c3bf06b79`.
Record registry manifest resolution, local image/config IDs, architecture, vLLM,
PyTorch, CUDA, NCCL and driver versions on both nodes. Require matching images.
Read the image's `vllm serve --help` and parser registry before starting: verify
MP `--nnodes`/`--headless`, EP, FP8/block 256, multimodal and parser support.
Do not assume the official image contains this repository's GB10 patches or
HPC-X plugin. The overlay uses the official image library path convention;
verify NCCL libraries/plugin availability and SM_121 kernels in the live image.
Missing SM_121 support, memory fit, parser support or actual IB transport blocks
validation; do not fall back silently to TCP or a different image.

Preserve the exact snapshot under the preset's `MODEL_PATH` on both nodes. Use the
safetensors index to assert 48 unique referenced shard files, all present, and
sum their sizes to 167819404368 bytes. Generate SHA256 for every snapshot file
on the source and compare on both Sparks; retain that ledger and acquisition
revision evidence. Do not treat `--revision` on a local directory as proof of
provenance. Both containers mount the verified snapshot read-only and run offline.

## Exact attended launch order (future, separately authorized work)

Use a maintenance window with both GPUs idle. Different container names and
port 8001 do **not** permit simultaneous production use. If production is running,
abort this recipe and arrange a separately authorized maintenance/restore plan
using the [production runbook](deepseek-v4-production.md). Record the actually
running family, exact image, preset, overlays, health, model ID and restart policy
before any maintenance; the catalog is not proof of live state.

1. On both nodes inspect `docker ps`, `ip -br addr`, `rdma link`, `ibdev2netdev`,
   free memory, swap, disk space and driver state. Reverify the preset's inspected
   `enp1s0f0np0` / `rocep1s0f0` mapping and 10.10.10.1/2 addresses. Check ports
   29601 and 8001 are free. Require at least
   110 GiB MemAvailable per idle node, no increasing swap activity and responsive
   management access. Otherwise abort; do not clear caches or reboot automatically.
2. On each node from repository root, in a clean shell without serving-variable
   overrides, define the same command (keep the verified per-node network exports):

   ```bash
   dc=(docker compose -p dsv4-vision-exp \
     --env-file presets/deepseek-v4-flash-vision-exp-tp2.env \
     -f docker-compose.yml -f compose/deepseek-v4/docker-compose.vision-exp.yml)
   "${dc[@]}" --profile head --profile worker config --quiet
   ```

   Inspect rendered config and actual argv: exact pins, c1/32K, prefix off,
   EP on, no speculation, offline mode, restart `no`, names and ports above.
   Explicit `-f` excludes any untracked automatic Compose override.
3. **spark02 worker first**, then **spark01 head within 30 seconds**:

   ```bash
   # spark02
   "${dc[@]}" --profile worker up -d --pull never worker
   # spark01
   "${dc[@]}" --profile head up -d --pull never head
   ```

   Worker readiness here means its process is alive and attempting MP rendezvous;
   it has no HTTP health endpoint and cannot finish initialization alone. Preserve
   both logs. Require rank 0/1, TP2/EP and NCCL IB/RoCE transport evidence. Do not
   wait for a Ray cluster or worker API before starting the head.
4. Observe both nodes throughout startup; allow **30 minutes maximum** to reach
   health. Poll `curl --max-time 5 -fsS http://127.0.0.1:8001/health` on spark01.
   Abort on either process exit, OOM, unsupported kernel, NCCL error/hang,
   restart, swap growth >2 GiB, MemAvailable <8 GiB, or management probe timeout
   >10 seconds. No retry loop. Keep requests serial with 120-second timeouts,
   temperature 0, maximum 512 output tokens and small fixed inputs.

## Predeclared gates (all pending / UNVALIDATED)

Save timestamped requests, responses, status codes, latency, exact argv, both
rank logs, image/snapshot identities and per-node memory/swap samples every
10 seconds. Never infer functional PASS from `/health` alone. A missing or
skipped gate means overall UNVALIDATED, and any failure requires teardown.

| Gate | Required PASS evidence |
|---|---|
| Health | HTTP 200 on `/health` within 30 minutes, then 10 consecutive polls 5 seconds apart; both ranks alive, zero restarts, TP2+EP and IB/RoCE proven in logs. |
| Models | `/v1/models` HTTP 200; expected ID `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, advertised context 32768; reject another model or missing context evidence. |
| Text | Five serial repeats each of `Return only the integer: 2+2` (4) and `Return only the capital of France` (Paris); HTTP 200, valid chat schema, nonempty correct final content, clean finish, no NaN/garbage. |
| Image | Fixed local PNG with a red square and blue circle, fixture SHA256 recorded before launch; send a base64 data URI via `image_url` chat content. Five serial repeats must identify both shapes/colors correctly. A second fixture with colors swapped must change answers accordingly. Reject ignored image, text-only fallback or processor failure. |
| Tool/reasoning | Five serial requests with a declared `add(a:integer,b:integer)` tool, asking to add 2 and 3; auto tool choice must return exactly one correctly named call with parseable JSON arguments 2 and 3. Supply tool result 5 and require final answer 5. Separately repeat a two-step arithmetic prompt five times; final answer correct, reasoning parsed into its designated field, no leaked reasoning/tool markup in final content. Store the exact pinned template/mode used; unavailable parsing is FAIL, not text PASS. |
| Stability | 30-minute attended soak after functional gates: at least 30 serial mixed text/image/tool cycles, all correct and HTTP 200; health every 5 seconds, no rank loss/restart/OOM/NCCL error, swap growth ≤2 GiB, MemAvailable ≥8 GiB, management response ≤10 seconds. Abort at first breach. |

This gate set qualifies only the recorded small-input c1 envelope. It does not
validate 32K-length requests, concurrency, video, throughput, speculative decoding
or unattended operation. Even all gates passing requires a separately reviewed
GB10 validation report before changing status; it never authorizes promotion.

## Stop and rollback (future operations only)

Capture logs before teardown if management remains responsive:

```bash
# spark01
"${dc[@]}" logs --no-color head > /tmp/dsv4-vision-exp-head.log
"${dc[@]}" stop -t 30 head
"${dc[@]}" rm -f head
# spark02
"${dc[@]}" logs --no-color worker > /tmp/dsv4-vision-exp-worker.log
"${dc[@]}" stop -t 30 worker
"${dc[@]}" rm -f worker
```

Stop head then worker, including after partial startup. Never use global Docker
prune, Ray stop, image deletion or generic production container stops. If a node
is unresponsive, end the experiment and use the separately authorized host
recovery procedure; do not launch replacements. Verify both experiment containers
are absent, ports released, GPU processes gone and memory stable. GB10 UMA may
require a separately authorized reboot before reuse. Restore only the exact
previously recorded production family with its canonical runbook and digest,
then verify health/models and its smoke gate. No previous service means leave
the slot idle. Keep experiment evidence; do not edit `.env`, defaults or autostart.
