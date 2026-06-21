#!/bin/bash
# =============================================================================
# Tokenizer-overlay wrapper entrypoint (EXPERIMENTAL).
#
# Selected via ENTRYPOINT_FILE in the disposable env. Generates a non-mutating
# tokenizer overlay (gen_tokenizer_overlay.py, baked in the image) from the
# read-only model mount, injects `--tokenizer <overlay>` into VLLM_EXTRA_ARGS,
# then hands off to the baked copy of the standard entrypoint. The mounted
# model directory is never modified.
#
# Gated by TOKENIZER_OVERLAY_ENABLE=1 so this wrapper is a no-op pass-through
# otherwise. Runs in both head and worker containers (each builds its own
# ephemeral overlay); only metadata is copied, so the cost is negligible.
# =============================================================================
set -euo pipefail

BASE_ENTRYPOINT="/opt/vllm-spark/entrypoint-base.sh"
GEN="/usr/local/bin/gen_tokenizer_overlay.py"
OVERLAY_DIR="${TOKENIZER_OVERLAY_DIR:-/run/vllm-tokenizer-overlay/step37-fp8}"

if [ "${TOKENIZER_OVERLAY_ENABLE:-0}" = "1" ]; then
    : "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set for tokenizer overlay}"
    # Refuse to inject a duplicate/conflicting tokenizer. The caller controls the
    # tokenizer when the overlay is enabled; a pre-existing --tokenizer (either
    # "--tokenizer <p>" or "--tokenizer=<p>") in VLLM_EXTRA_ARGS would otherwise
    # become a second, ambiguous tokenizer argument. Fail fast instead.
    case " ${VLLM_EXTRA_ARGS:-} " in
        *" --tokenizer "*|*" --tokenizer="*)
            echo "[overlay-entrypoint] ERROR: VLLM_EXTRA_ARGS already supplies --tokenizer; refusing to inject a conflicting overlay tokenizer. Unset it or disable TOKENIZER_OVERLAY_ENABLE." >&2
            exit 1
            ;;
    esac
    echo "[overlay-entrypoint] source=${MODEL_CONTAINER_PATH} overlay=${OVERLAY_DIR} manifest=${OVERLAY_DIR}/overlay_manifest.json"
    echo "[overlay-entrypoint] generating tokenizer overlay (source mount is read-only; overlay is ephemeral)"
    python3 "${GEN}" --model "${MODEL_CONTAINER_PATH}" --out "${OVERLAY_DIR}" --print-manifest
    echo "[overlay-entrypoint] overlay generation OK (validation: build-time --unit + live Stage checks)"
    export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-} --tokenizer ${OVERLAY_DIR}"
    echo "[overlay-entrypoint] injected --tokenizer ${OVERLAY_DIR}"
else
    echo "[overlay-entrypoint] TOKENIZER_OVERLAY_ENABLE != 1; pass-through"
fi

exec bash "${BASE_ENTRYPOINT}"
