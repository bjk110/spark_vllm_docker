#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_ENTRYPOINT = ROOT / "entrypoints/qwen38-ngc2609-mtp-k2-c2.sh"
RAY_ENTRYPOINT = ROOT / "entrypoints/qwen38-ngc2609-mtp-k2-c2-ray.sh"
DOCKERFILE = ROOT / "dockerfiles/experimental/Dockerfile.qwen38-ngc2609-ray259-candidate"
PRESET = ROOT / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
ROLLBACK = ROOT / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2-mp-rollback.env"
RUNBOOK = ROOT / "docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md"
MANIFEST = ROOT / "patches/qwen/ngc2609-e77-prefault/MANIFEST.json"

PARENT = "ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6"
RAY_VERSION = "2.59.0"


def test_ray_recipe_is_an_exact_parent_child_with_pinned_ray_cgraph():
    text = DOCKERFILE.read_text()
    assert f"FROM {PARENT}" in text
    assert f'"ray[cgraph]=={RAY_VERSION}"' in text
    assert "--no-cache-dir" in text
    assert "org.bjk110.parent.manifest" in text
    assert "org.bjk110.ray.version" in text


def test_ray_recipe_preserves_the_mp_serving_contract():
    base = BASE_ENTRYPOINT.read_text()
    ray = RAY_ENTRYPOINT.read_text()
    common_flags = [
        "--served-model-name",
        "--tensor-parallel-size",
        "--trust-remote-code",
        "--dtype auto",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--gpu-memory-utilization",
        "--kv-cache-memory-bytes",
        "--kv-cache-dtype",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-expert-parallel",
        "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'",
        "--no-enable-flashinfer-autotune",
        "--reasoning-parser qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser qwen3_coder",
    ]
    for flag in common_flags:
        assert flag in base
        assert flag in ray
    assert "--distributed-executor-backend mp" in base
    assert "--distributed-executor-backend ray" in ray
    for mp_only in ("--nnodes", "--node-rank", "--master-addr", "--master-port", "--headless"):
        assert mp_only not in ray


def test_ray_recipe_has_bounded_two_node_control_plane():
    text = RAY_ENTRYPOINT.read_text()
    assert '${RAY_PORT:?RAY_PORT required}' in text
    assert '${RAY_JOIN_TIMEOUT_SEC:-600}' in text
    assert "ray start --head" in text
    assert 'ray start --address="${HEAD_ROCE_IP}:${RAY_PORT}"' in text
    assert "--block" in text
    assert "alive >= expected" in text
    assert "Ray cluster join timeout" in text
    assert '[[ "${ROLE}" == worker ]]' in text


def test_ray_recipe_matches_the_qualified_model_and_cache_guards():
    text = RAY_ENTRYPOINT.read_text()
    assert '${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH required}' in text
    assert '${MODEL_CONTAINER_PATH}/config.json' in text
    assert '${MODEL_REVISION:?MODEL_REVISION required}' in text
    assert '${MODEL_CONTAINER_PATH}/.hf-revision' in text
    assert 'model revision mismatch' in text
    assert 'mkdir -p "${HF_HOME}"' in text
    assert "ray stop --force" in text
    assert "rm -rf /tmp/ray" in text


def parse_env(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def test_canonical_preset_selects_ray_and_keeps_mp_as_rollback():
    ray = parse_env(PRESET)
    mp = parse_env(ROLLBACK)
    assert ray["VLLM_IMAGE"] == "ghcr.io/bjk110/vllm-spark@sha256:1308dca36201a271adbb928b7b2927fb0c049e5a6d6748900991e03484a09de5"
    assert ray["ENTRYPOINT_FILE"] == "./entrypoints/qwen38-ngc2609-mtp-k2-c2-ray.sh"
    assert ray["DISTRIBUTED_BACKEND"] == "ray"
    assert mp["VLLM_IMAGE"] == PARENT
    assert mp["ENTRYPOINT_FILE"] == "./entrypoints/qwen38-ngc2609-mtp-k2-c2.sh"
    assert mp["DISTRIBUTED_BACKEND"] == "mp"
    for key in ("MODEL_PATH", "SERVED_MODEL_NAME", "MODEL_REVISION", "TP_SIZE", "NNODES", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "KV_CACHE_MEMORY_BYTES", "KV_CACHE_DTYPE"):
        assert ray[key] == mp[key]


def test_runbook_separates_ray_gates_from_mp_full_matrix():
    text = RUNBOOK.read_text()
    assert "### Ray canonical arm — 2026-10-18" in text
    assert "### MP rollback arm — 2026-10-16" in text
    assert "### MP rollback full-matrix performance reference" in text
    assert "not the Ray image" in text


def test_mp_rollback_commands_use_a_clean_environment():
    text = RUNBOOK.read_text()
    rollback = text.split("After an authorized clean reboot", 1)[1]
    clean = 'env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"'
    assert rollback.count(clean) == 2
    assert rollback.index("--profile worker up -d") < rollback.index("--profile head up -d")


def test_manifest_attributes_long_context_pair_only_to_mp():
    import json

    measured = json.loads(MANIFEST.read_text())["measured_contract"]
    assert measured["ray"]["actual_262k_single"] == "PASS"
    assert "actual_262k_pair" not in measured["ray"]
    assert measured["mp_rollback"]["actual_262k_pair"] == "PASS"
