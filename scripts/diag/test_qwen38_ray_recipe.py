#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_ENTRYPOINT = ROOT / "entrypoints/qwen38-ngc2609-mtp-k2-c2.sh"
RAY_ENTRYPOINT = ROOT / "entrypoints/qwen38-ngc2609-mtp-k2-c2-ray.sh"
DOCKERFILE = ROOT / "dockerfiles/experimental/Dockerfile.qwen38-ngc2609-ray259-candidate"
PRESET = ROOT / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env"
ROLLBACK = ROOT / "presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2-mp-rollback.env"
RUNBOOK = ROOT / "docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md"
MANIFEST = ROOT / "patches/qwen/ngc2609-e77-prefault/MANIFEST.json"
COMPOSE = ROOT / "docker-compose.yml"
OVERLAY = ROOT / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml"
ROLLBACK_OVERLAY = ROOT / "compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml"

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


def test_canonical_ray_render_uses_only_the_qualified_runtime_environment():
    trusted_env = {
        "PATH": os.environ.get("PATH", os.defpath),
        **{key: os.environ[key] for key in ("HOME", "DOCKER_CONFIG") if key in os.environ},
    }
    proc = subprocess.run(
        [
            "docker", "compose", "--env-file", str(PRESET),
            "-f", str(COMPOSE), "-f", str(OVERLAY),
            "--profile", "*", "config", "--format", "json",
        ],
        cwd=ROOT,
        env=trusted_env,
        text=True,
        capture_output=True,
        check=True,
    )
    services = json.loads(proc.stdout)["services"]
    common = {
        "MODEL_CONTAINER_PATH": "/model",
        "SERVED_MODEL_NAME": "orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8",
        "MODEL_REVISION": "8bebbc05cbe65099f37c520865463c1377d8050d",
        "HOST_PORT": "8000",
        "TP_SIZE": "2",
        "NNODES": "2",
        "HEAD_ROCE_IP": "10.10.10.1",
        "WORKER_ROCE_IP": "10.10.10.2",
        "RAY_PORT": "6379",
        "RAY_JOIN_TIMEOUT_SEC": "600",
        "RAY_INCLUDE_DASHBOARD": "false",
        "RAY_OBJECT_STORE_MEMORY_BYTES": "4294967296",
        "RAY_memory_monitor_refresh_ms": "0",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": "auto",
        "MAX_MODEL_LEN": "262144",
        "MAX_NUM_SEQS": "2",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "GPU_MEMORY_UTILIZATION": "0.83",
        "KV_CACHE_MEMORY_BYTES": "8589934592",
        "KV_CACHE_DTYPE": "auto",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "rocep1s0f0",
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
        "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
        "UCX_NET_DEVICES": "enp1s0f0np0",
        "NCCL_NVLS_ENABLE": "0",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "7200",
        "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter",
        "VLLM_MOE_USE_DEEP_GEMM": "1",
        "VLLM_USE_DEEP_GEMM_E8M0": "1",
        "HF_HOME": "/cache/huggingface",
        "XDG_CACHE_HOME": "/cache/xdg",
        "VLLM_CACHE_ROOT": "/cache/vllm",
        "TRITON_CACHE_DIR": "/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert services["head"]["environment"] == {
        **common, "ROLE": "head", "VLLM_HOST_IP": "10.10.10.1",
    }
    assert services["worker"]["environment"] == {
        **common, "ROLE": "worker", "VLLM_HOST_IP": "10.10.10.2",
    }


def test_mp_rollback_render_preserves_the_qualified_mp_launch_contract():
    assert ROLLBACK_OVERLAY.is_file()
    trusted_env = {
        "PATH": os.environ.get("PATH", os.defpath),
        **{key: os.environ[key] for key in ("HOME", "DOCKER_CONFIG") if key in os.environ},
    }
    proc = subprocess.run(
        [
            "docker", "compose", "--env-file", str(ROLLBACK),
            "-f", str(COMPOSE), "-f", str(ROLLBACK_OVERLAY),
            "--profile", "*", "config", "--format", "json",
        ],
        cwd=ROOT,
        env=trusted_env,
        text=True,
        capture_output=True,
        check=True,
    )
    services = json.loads(proc.stdout)["services"]
    assert services["head"]["entrypoint"] == ["bash", "/entrypoint.sh"]
    assert services["worker"]["entrypoint"] == ["bash", "/entrypoint.sh"]
    assert services["head"]["environment"]["DISTRIBUTED_BACKEND"] == "mp"
    assert services["worker"]["environment"]["DISTRIBUTED_BACKEND"] == "mp"
    assert services["head"]["environment"]["MASTER_PORT"] == "29538"
    assert services["worker"]["environment"]["MASTER_PORT"] == "29538"
    assert services["worker"]["environment"]["NODE_RANK"] == "1"
    assert services["head"]["environment"]["CLUSTER_MODE"] == "dual-rdma"
    assert services["worker"]["environment"]["CLUSTER_MODE"] == "dual-rdma"
    assert services["head"]["restart"] == "no"
    assert services["worker"]["restart"] == "no"


def test_runbook_separates_ray_gates_from_mp_full_matrix():
    text = RUNBOOK.read_text()
    assert "### Ray canonical arm — 2026-10-18" in text
    assert "### MP rollback arm — 2026-10-16" in text
    assert "### MP rollback full-matrix performance reference" in text
    assert "not the Ray image" in text


def test_runbook_pins_compose_override_requirement():
    text = RUNBOOK.read_text()
    assert "Docker Compose 2.24.4 or newer" in text
    assert "`environment: !override`" in text
    assert "present-but-empty" in text


def test_mp_rollback_commands_use_a_clean_environment():
    text = RUNBOOK.read_text()
    rollback = text.split("After an authorized clean reboot", 1)[1]
    clean = 'env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"'
    rollback_overlay = "docker-compose.ngc2609-mtp-k2-c2-mp-rollback.yml"
    assert rollback.count(clean) == 2
    assert rollback.count(rollback_overlay) == 2
    assert "docker-compose.ngc2609-mtp-k2-c2.yml" not in rollback.replace(
        rollback_overlay, ""
    )
    assert rollback.index("--profile worker up -d") < rollback.index("--profile head up -d")
    assert rollback_overlay in ROLLBACK.read_text()


def test_manifest_attributes_long_context_pair_only_to_mp():
    import json

    measured = json.loads(MANIFEST.read_text())["measured_contract"]
    assert measured["ray"]["actual_262k_single"] == "PASS"
    assert "actual_262k_pair" not in measured["ray"]
    assert measured["mp_rollback"]["actual_262k_pair"] == "PASS"
