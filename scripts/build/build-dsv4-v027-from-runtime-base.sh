#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="${ROOT_DIR}/dockerfiles/active/Dockerfile.v027-ngc2607-dsv4-from-runtime-base"
OUTPUT_TAG="${1:-vllm-spark:v027-ngc2607-dsv4-from-runtime-base-local}"
RECIPE_REVISION="${RECIPE_REVISION:-$(git -C "${ROOT_DIR}" rev-parse --verify HEAD)}"

# No project build-context files are required. Use a one-file temporary context so unrelated local
# experiment artifacts are never uploaded to BuildKit.
CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT
cp "${DOCKERFILE}" "${CONTEXT_DIR}/Dockerfile"
docker build \
  --file "${CONTEXT_DIR}/Dockerfile" \
  --build-arg "RECIPE_REVISION=${RECIPE_REVISION}" \
  --tag "${OUTPUT_TAG}" \
  "${CONTEXT_DIR}"

# Validate in a disposable container so imports cannot add cache files to the derivative image layer.
docker run --rm -i --entrypoint python3 "${OUTPUT_TAG}" - <<'PY'
import importlib.metadata as metadata
import vllm.entrypoints.cli.main
import vllm.entrypoints.openai.api_server
import flashinfer.mla._sparse_mla_sm120 as sparse_mla

expected = {
    "vllm": "0.27.1.dev0+g4bdc8a788.d20260813.cu133",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-cubin": "0.6.16.post3",
    "torch": "2.13.0a0+9186a08b2c.nv26.7.59513937",
}
for package, version in expected.items():
    actual = metadata.version(package)
    assert actual == version, (package, actual, version)
assert (32, 256) in sparse_mla._DECODE_DSV4_DISPATCH_BY_PBS[64]
assert (32, 128) in sparse_mla._DECODE_DSV4_DISPATCH_BY_PBS[256]
print("validated DSV4 v0.27 runtime-base contract: PASS")
PY

docker image inspect "${OUTPUT_TAG}" --format \
  'id={{.Id}} base_digest={{index .Config.Labels "org.vllm-spark.runtime-base.digest"}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
