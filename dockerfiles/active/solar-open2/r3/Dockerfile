# =============================================================================
# Solar-Open2 v0.25.1 r3 (EXPERIMENTAL): r2 lineage + baked C3 raw-g1 contract
# fix + VLLM_SPARK_ST_PREAD safetensors pread env gate (default OFF).
#
# Base (immutable, built 2026-07-25, both nodes ID 7e282758328c):
#   vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rebased-ecfix-upstream-r2-exp
#   = NGC 26.05 + vLLM v0.25.1@752a3a504485 + FlashInfer 0.6.15@8eccd0c1 +
#     Upstage Solar-Open2 overlay rebased from 00907fc (r2 KDA warmup fix).
#
# r3 delta (the ONLY changes; no package installs, no version changes):
#   1) vllm/model_executor/models/solar_open2.py
#        C3 raw-g1 contract fix (patches/solar/solar-open2-rawg1-contract-v025.patch)
#   2) vllm/model_executor/model_loader/weight_utils.py
#        VLLM_SPARK_ST_PREAD opt-in env gate for the lazy safetensors loader
#        (patches/solar/vllm-safetensors-pread-env-gate.patch)
#
# Patch application: the two canonical .patch files were applied to the
# pristine r2 sources OUTSIDE the build (patch -p1 --fuzz=0, verified), and
# the resulting files are COPYed here with strict SHA-256 verification of
# BOTH the pre-patch originals and the post-patch results. This avoids
# installing a `patch` binary into the runtime image (r2 ships none) while
# keeping byte-exact provenance. Patch order (recorded): rawg1 -> pread gate
# (independent files; order has no functional effect).
#
# Build context: staging dir with the patched sources + canonical patches.
# EXPERIMENTAL — do not promote, do not publish to GHCR.
# =============================================================================
FROM vllm-spark:solar-open2-nvfp4-v0251-upstage00907fc-rebased-ecfix-upstream-r2-exp

COPY solar_open2-rawg1.py /tmp/r3/solar_open2.py.new
COPY weight_utils-pread-gate.py /tmp/r3/weight_utils.py.new
COPY solar-open2-rawg1-contract-v025.patch \
     vllm-safetensors-pread-env-gate.patch \
     /opt/spark-patches/r3/

RUN set -eux; \
    cd /usr/local/lib/python3.12/dist-packages; \
    echo "4e5c907155a9400ab24d68eaceef965bbc84e2fc54a9e1e5c095c716732c8015  vllm/model_executor/models/solar_open2.py" | sha256sum -c -; \
    echo "ad98e4040aa78fe1803ad47d8e2caf0a67445eb5aeffa00b0bc5525ec7eff198  vllm/model_executor/model_loader/weight_utils.py" | sha256sum -c -; \
    cp /tmp/r3/solar_open2.py.new vllm/model_executor/models/solar_open2.py; \
    cp /tmp/r3/weight_utils.py.new vllm/model_executor/model_loader/weight_utils.py; \
    echo "56b487479c06a4c23d19276c624cde2d462da0ac07e90e394bed580984761645  vllm/model_executor/models/solar_open2.py" | sha256sum -c -; \
    echo "de48cbefb218b5a0c7160df0253c116bf5b84cf4bf4cd90aecdd4396bb5de7aa  vllm/model_executor/model_loader/weight_utils.py" | sha256sum -c -; \
    rm -f vllm/model_executor/models/__pycache__/solar_open2.cpython-*.pyc \
          vllm/model_executor/model_loader/__pycache__/weight_utils.cpython-*.pyc; \
    python3 -c "import ast; [ast.parse(open(p).read()) for p in ('vllm/model_executor/models/solar_open2.py','vllm/model_executor/model_loader/weight_utils.py')]; print('AST_OK')"; \
    rm -rf /tmp/r3
