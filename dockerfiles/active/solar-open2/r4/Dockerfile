# =============================================================================
# Solar-Open2 v0.25.1 r4 (EXPERIMENTAL): r3 lineage + VLLM_SPARK_B12X_SHARED_
# WORKSPACE env gate for the FlashInfer b12x scratch workspaces (default OFF).
#
# Base (immutable, both nodes ID 001dcd2fb66d):
#   vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-r3-exp
#   = NGC 26.05 + vLLM 0.25.1@752a3a504485 + FlashInfer 0.6.15 + Upstage
#     Solar overlay (r2) + C3 raw-g1 fix + VLLM_SPARK_ST_PREAD gate (r3).
#
# r4 delta (the ONLY change; no package installs, no version changes):
#   flashinfer/fused_moe/cute_dsl/b12x_moe.py
#     VLLM_SPARK_B12X_SHARED_WORKSPACE opt-in bounded shared scratch-
#     workspace manager (patches/solar/vllm-flashinfer-b12x-shared-
#     workspace-env-gate.patch). Disabled -> original per-wrapper path.
#
# Cumulative patch order of this lineage (recorded):
#   1. Solar Open 2 support (rebased 00907fc)          -> r2 (built-in)
#   2. C3 raw-g1 contract fix                          -> r3
#   3. Safetensors pread env gate                      -> r3
#   4. b12x shared-workspace env gate                  -> r4 (this image)
#
# Applied as SHA-verified COPY of the externally patched file (r3 lineage
# ships no `patch` binary; canonical .patch kept in-image for provenance).
# EXPERIMENTAL — do not promote, do not publish to GHCR.
# =============================================================================
FROM vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rawg1-pread-r3-exp

COPY b12x_moe_r4.py /tmp/r4/b12x_moe.py.new
COPY vllm-flashinfer-b12x-shared-workspace-env-gate.patch /opt/spark-patches/r4/

RUN set -eux; \
    cd /usr/local/lib/python3.12/dist-packages; \
    echo "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4  flashinfer/fused_moe/cute_dsl/b12x_moe.py" | sha256sum -c -; \
    cp /tmp/r4/b12x_moe.py.new flashinfer/fused_moe/cute_dsl/b12x_moe.py; \
    echo "38c253a37c131b38713b92cea038bb6ff07f7b38a1c91fdd68f38567cb8180bc  flashinfer/fused_moe/cute_dsl/b12x_moe.py" | sha256sum -c -; \
    rm -f flashinfer/fused_moe/cute_dsl/__pycache__/b12x_moe.cpython-*.pyc; \
    python3 -c "import ast; ast.parse(open('flashinfer/fused_moe/cute_dsl/b12x_moe.py').read()); print('AST_OK')"; \
    rm -rf /tmp/r4
