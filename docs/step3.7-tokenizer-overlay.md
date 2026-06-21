# Step-3.7 FP8 — non-mutating runtime tokenizer overlay

**Status: EXPERIMENTAL.** Not a production or promoted image. The overlay is
opt-in and disabled by default.

## Problem statement

Serving `stepfun-ai/Step-3.7-Flash-FP8` produced corrupted Korean output: Korean
text was dropped on encode and whitespace/newlines were collapsed. The previous
mitigation edited `tokenizer_config.json` **in place** inside each Spark model
directory, mutating a synchronized, otherwise-upstream checkpoint. This overlay
removes that in-place mutation: the model directory stays read-only and
byte-identical to upstream, and the fix is generated into an ephemeral location
at container start.

## Affected model and revision

- Model: `stepfun-ai/Step-3.7-Flash-FP8` (198B sparse MoE VLM, `quant_method=fp8`).
- Revision: `b3d7916fccac844cca050d7520f2aaa513f9a84f`.

## Affected Transformers behavior / confirmed root cause

The checkpoint ships `"tokenizer_class": "LlamaTokenizerFast"`. Under
Transformers 5.10.2 that class resolves to the SentencePiece-backed **slow**
`LlamaTokenizer`, which drops non-ASCII (Korean) text and collapses whitespace.
Setting `"tokenizer_class": "PreTrainedTokenizerFast"` makes Transformers use the
Rust fast tokenizer (reads `tokenizer.json` directly), which is correct. This is
the single field that differs between the upstream and the fixed config.

### Upstream-like tokenizer behavior (`LlamaTokenizerFast`)

Resolves to `LlamaTokenizer`. For `"대한민국의 수도는 서울입니다."` it produces
2 ids decoding to `"."` — Korean lost. `"a    b\tc"` → `"abc"` (whitespace
collapsed). No Korean survives round-trip.

### Overlay tokenizer behavior (`PreTrainedTokenizerFast`)

Resolves to the fast `TokenizersBackend`. Korean, mixed Korean/English,
multiline, repeated whitespace, and Korean punctuation all round-trip exactly,
with no `Ġ`/`Ċ` artifacts. `/tokenize` counts: korean=10, mixed=10, multiline=11,
repeated-spaces=5, punct=16 — byte-for-byte identical to the previously validated
in-place edit.

## Why permanent model-directory editing was rejected

The Spark model directories are synchronized, read-only-mounted copies of an
upstream checkpoint. Editing `tokenizer_config.json` in place mutates that
checkpoint, diverges it from the homeserver source of truth, requires per-node
backups, and risks silent drift. A non-mutating overlay keeps every model
directory byte-identical to upstream and confines the fix to an ephemeral
container-local path.

## Selected architecture — Option A (runtime overlay + explicit `--tokenizer`)

At container start a wrapper entrypoint generates an overlay directory under
`/run` containing only the small tokenizer metadata, with `tokenizer_class`
changed, and injects `--tokenizer <overlay>` into the vLLM command. The model
weights are never copied and the model mount is never written.

```
ENTRYPOINT_FILE=./entrypoints/entrypoint-tokenizer-overlay.sh   (bind-mounted to /entrypoint.sh)
TOKENIZER_OVERLAY_ENABLE=1
  -> gen_tokenizer_overlay.py  : build /run/.../overlay (78202af4 -> e4bec1b1)
  -> inject  --tokenizer /run/vllm-tokenizer-overlay/step37-fp8
  -> exec    /opt/vllm-spark/entrypoint-base.sh   (baked standard entrypoint)
```

### Serving-tokenizer resolution path (vLLM 0.23.0)

`--tokenizer` → `EngineArgs.tokenizer` (`engine/arg_utils.py:788`) →
`ModelConfig.tokenizer` (`config/model.py:495`; defaults to the model path only
when unset) → `cached_tokenizer_from_config()` →
`cached_get_tokenizer(model_config.tokenizer, ...)`
(`tokenizers/registry.py:268`). The serving tokenizer therefore loads from the
overlay path.

### `Step3VLProcessor` resolution path

`BaseProcessingInfo.get_tokenizer()` (`multimodal/processing/context.py:308` →
`:103`) returns the context tokenizer built from `model_config.tokenizer`.
`Step3VLForConditionalGeneration.get_hf_processor()` constructs
`Step3VLProcessor(tokenizer=self.get_tokenizer())` (`models/step3_vl.py:109`).
`Step3p7ForConditionalGeneration(Step3VLForConditionalGeneration)`
(`models/step3p7.py:21`) inherits this, so the multimodal processor uses the same
overlay tokenizer. No processor path reads tokenizer metadata directly from the
model directory.

## Generator safety guards (`patches/common/gen_tokenizer_overlay.py`)

- Requires explicit `--model` and `--out`; resolves symlinks before enforcing
  allowed roots; rejects output inside any model root.
- Verifies the full source `tokenizer_config.json` SHA256 and the expected
  original class; verifies each verbatim file by hash.
- Copies only an explicit allowlist (`tokenizer.json`, `special_tokens_map.json`,
  `chat_template.jinja`); never copies weights; never writes the source.
- Builds in a temp dir then `os.replace` (atomic); changes only `tokenizer_class`;
  verifies the regenerated postimage SHA256 and all copied-file hashes.
- Idempotent for an already-valid overlay; replaces only a stale overlay it owns
  (has a manifest); **rejects an unexpected existing output directory** rather
  than clobbering it; fails non-zero on any mismatch; emits a JSON manifest.

## Wrapper behavior (`entrypoints/entrypoint-tokenizer-overlay.sh`)

- `set -euo pipefail`; no `eval`; disabled unless `TOKENIZER_OVERLAY_ENABLE=1`
  (pass-through otherwise); ends with `exec` so signals reach the vLLM process
  (verified PID-preserving).
- Generates the overlay only in an ephemeral path; never writes the model mount;
  runs in both head and worker roles (each builds its own overlay).
- **Duplicate/conflicting `--tokenizer` handling**: if `VLLM_EXTRA_ARGS` already
  contains `--tokenizer <p>` or `--tokenizer=<p>`, the wrapper fails fast instead
  of injecting a second, ambiguous tokenizer argument.

## Compose opt-in variables

`docker-compose.yml` passes two variables into the head and worker containers:

```yaml
- TOKENIZER_OVERLAY_ENABLE=${TOKENIZER_OVERLAY_ENABLE:-0}
- TOKENIZER_OVERLAY_DIR=${TOKENIZER_OVERLAY_DIR:-}
```

Default off / unset → no effect on any existing workload. The wrapper itself is
selected only when the env sets `ENTRYPOINT_FILE` to the overlay wrapper, so a
default deployment is unchanged.

## Read-only model-mount behavior

The model is mounted `:ro` (`docker-compose.yml`). The overlay is written to
`/run` (writable, ephemeral). Runtime write attempts to the model mount fail with
`Read-only file system`; the source `tokenizer_config.json` SHA256 is unchanged
before, during, and after serving.

## Hashes and provenance

| Item | SHA256 |
|---|---|
| source/upstream `tokenizer_config.json` | `78202af487f4d4360e8d15cb0506d60718f8599770599c6f28f7f1fa045a591f` |
| generated overlay `tokenizer_config.json` | `e4bec1b1841cdb9da779f34c2260604ed27800252a445e7ad2811e3b37acc4ea` |
| `tokenizer.json` (verbatim) | `b564c620eb77fa11d0926011c2202347d6cfc358d79724ee04ae7007e13636f0` |
| `special_tokens_map.json` (verbatim) | `d47424bda11df4cedc3f9458915c465a28e601d3b7df0e78f6dff4d7727006c4` |
| `chat_template.jinja` (verbatim) | `f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633` |
| immutable Prometheus base image | `ghcr.io/bjk110/vllm-spark@sha256:81653ff7e16ca29afc9c6fa057c5216b8c9e14fa9a09430d6ed58b88d9115b0a` |
| Prometheus patched `routing.py` | `a3addfd90d1132a5ab5dca54c788f4743fe180b9607a662bf34ef0453750848c` |
| validated local image ID | `sha256:f195d6e15041743c6b8bb95dfbf47305fa3c11b60d1272c4001baf877ab6e1fa` (`linux/arm64`) |

## Validation results

- **Static (CPU-only, in-image)**: `py_compile` OK; shell syntax OK; generator
  unit tests 14/14 (incl. duplicate- and foreign-directory rejection); behavior
  OK; wrapper argument tests 20/20; read-only fixture deterministic; Prometheus
  patch `--inspect` = `a3addfd9` patched; vLLM 0.23.0 and Transformers 5.10.2
  import.
- **Dual-node correctness (FP8 TP=2, mp, EP off, enforce-eager)**: 6/6 PASS,
  byte-identical, completion tokens 61 / 85 / 126 / 171 / 288 / 315, all
  `finish=stop`, no `Ġ`/`Ċ`.
- **Tokenizer-focused API**: Korean-only, mixed, multiline, repeated-spaces,
  Korean-punctuation, and chat-template/special-token cases all correct.
- **Performance (within ±15% of the validated FP8 baseline)**: prefill
  ~1025.97 t/s, decode (warm) ~10.25 t/s (range 9.7–10.9), prefill TTFT ~1.172 s.
- Backend selection identical to the FP8 baseline: TRITON Fp8 MoE + TRITON_ATTN
  (ViT FLASH_ATTN), `quantization=fp8`.

## Rollback procedure

Set `TOKENIZER_OVERLAY_ENABLE=0` (or unset it) and set `ENTRYPOINT_FILE` back to
`./entrypoints/entrypoint.sh`. The wrapper becomes a no-op pass-through and vLLM
uses the model directory's tokenizer directly. No image rebuild is required to
disable the overlay. The Compose passthrough has no effect while disabled.

## Known limitations

- The overlay tokenizer fix depends on the validated source/overlay SHA256 pair;
  the generator refuses to run on any other `tokenizer_config.json`.
- The overlay directory must be an ephemeral writable path (`/run`, `/tmp`,
  `/dev/shm`); a writable model mount is not required and is not used.
- FP8 MoE decode runs on the untuned default TRITON config (unchanged from the
  baseline); MoE tuning is out of scope.
- GB10 unified memory: a large FP8 load requires a clean memory state (reboot if
  UVM is retained from a prior run).

## Published image provenance (immutable)

The exact dual-node-validated local image was published to GHCR without rebuild
(the local image ID was tagged directly). The remote manifest's config digest
equals the validated local image ID, proving content identity (matching tags
alone were not relied upon).

| Item | Value |
|---|---|
| implementation commit | `07a2722b093b1e9e8750b6f664826f0e39a25218` |
| immutable GHCR tag | `ghcr.io/bjk110/vllm-spark:v023-step37-tokenizer-overlay-exp-07a2722` |
| remote manifest digest | `sha256:1c987173177a69d58d2ce61babf874f1a7c6c9a2830dcd33b180f2d81c9fde1e` |
| config / image ID | `sha256:f195d6e15041743c6b8bb95dfbf47305fa3c11b60d1272c4001baf877ab6e1fa` |
| platform | `linux/arm64` |

Verification: pulling `ghcr.io/bjk110/vllm-spark@sha256:1c987173…` on a Spark
node resolved to image ID `f195d6e1…`; embedded generator/validator/wrapper-base
hashes, Prometheus patched-source hash `a3addfd9…`, and the tokenizer
source/overlay label hashes all matched, with no model load. Pull by immutable
digest (or the immutable tag above), never `latest`.

## Local-only evidence (not part of this repository)

Full validation artifacts (logs, manifests, correctness/perf JSON, memory gates)
are preserved locally at
`/home/bjk110/tokenizer-overlay-validation-20260620-224239Z/`. This directory is
not committed and is not part of the public repository.
