#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PRESET = Path("presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env")
OVERLAY = Path("compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml")
ENTRYPOINT = Path("entrypoints/qwen38-ngc2609-mtp-k2-c2.sh")
PATCH_ROOT = Path("patches/qwen/ngc2609-e77-prefault")
MANIFEST = PATCH_ROOT / "MANIFEST.json"
DOC = Path("docs/qwen3.8-flash-next-uncensored-ngc2609-mtp-k2-c2.md")
IMAGE = "ghcr.io/bjk110/vllm-spark@sha256:928d86903dfce3b11203f1f6d3118aa47998e55fee36fc73423e3bef8ab04cd6"
PUBLISHED_IMAGE_ID = "sha256:8bb958ffb833c407cd642d72c3938c0f8f0ed31136851c1edcd868bf135651c4"
QUALIFIED_ROOTFS_IMAGE_ID = "sha256:8bb958ffb833c407cd642d72c3938c0f8f0ed31136851c1edcd868bf135651c4"
MODEL = "orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8"
REVISION = "8bebbc05cbe65099f37c520865463c1377d8050d"
VLLM_COMMIT = "e77daef89e18e08321ae7b8b24827eedd5fe8673"

EXPECTED_ENV = {
    "VLLM_IMAGE": IMAGE,
    "MODEL_PATH": "/mnt/data/llm-models/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8",
    "MODEL_CONTAINER_PATH": "/model",
    "SERVED_MODEL_NAME": MODEL,
    "MODEL_REVISION": REVISION,
    "ENTRYPOINT_FILE": "./entrypoints/qwen38-ngc2609-mtp-k2-c2.sh",
    "CLUSTER_MODE": "dual-rdma",
    "TP_SIZE": "2",
    "DISTRIBUTED_BACKEND": "mp",
    "NNODES": "2",
    "WORKER_NODE_RANK": "1",
    "HEAD_ROCE_IP": "10.10.10.1",
    "WORKER_ROCE_IP": "10.10.10.2",
    "ROCE_IF_NAME": "enp1s0f0np0",
    "IB_HCA_NAME": "rocep1s0f0",
    "MASTER_PORT": "29538",
    "HOST_PORT": "8000",
    "MAX_MODEL_LEN": "262144",
    "MAX_NUM_SEQS": "2",
    "MAX_NUM_BATCHED_TOKENS": "8192",
    "GPU_MEMORY_UTILIZATION": "0.83",
    "KV_CACHE_MEMORY_BYTES": "8589934592",
    "KV_CACHE_DTYPE": "auto",
    "NCCL_DEBUG": "WARN",
    "NCCL_IB_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "7200",
    "VLLM_ALL2ALL_BACKEND": "allgather_reducescatter",
    "VLLM_MOE_USE_DEEP_GEMM": "1",
    "VLLM_USE_DEEP_GEMM_E8M0": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "QWEN38_NGC2609_OVERLAY_ROOT": "./patches/qwen/ngc2609-e77-prefault",
    "VLLM_EXTRA_ARGS": "",
}

SOURCE_FILES = [
    "model_executor/utils.py",
    "model_executor/parameter.py",
    "model_executor/layers/linear.py",
    "model_executor/layers/vocab_parallel_embedding.py",
    "model_executor/layers/fused_moe/routed_experts.py",
    "model_executor/model_loader/weight_utils.py",
]
TARGET_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def parse_env(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def expected_mounts() -> list[str]:
    prefix = "${QWEN38_NGC2609_OVERLAY_ROOT:-./patches/qwen/ngc2609-e77-prefault}"
    return [f"{prefix}/{name}:{TARGET_ROOT}/{name}:ro" for name in SOURCE_FILES] + [
        "./cache/qwen38-ngc2609-e77:/cache:rw"
    ]


def expected_overlay_environment() -> list[str]:
    return [
        "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-8589934592}",
        "KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}",
        "MODEL_REVISION=${MODEL_REVISION}",
        "NNODES=2",
        "VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-allgather_reducescatter}",
        "UCX_NET_DEVICES=${ROCE_IF_NAME}",
        "HF_HOME=/cache/huggingface",
        "XDG_CACHE_HOME=/cache/xdg",
        "VLLM_CACHE_ROOT=/cache/vllm",
        "TRITON_CACHE_DIR=/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
        "TOKENIZERS_PARALLELISM=false",
    ]


def verify(repo_root: Path, render_compose: bool = True):
    failures: list[str] = []
    warnings: list[str] = []
    paths = [PRESET, OVERLAY, ENTRYPOINT, MANIFEST, DOC]
    for rel in paths:
        if not (repo_root / rel).is_file():
            failures.append(f"missing tracked recipe artifact: {rel}")
    if failures:
        return failures, warnings

    env = parse_env(repo_root / PRESET)
    for key, expected in EXPECTED_ENV.items():
        if env.get(key) != expected:
            failures.append(f"{key} drift: {env.get(key)!r} != {expected!r}")

    try:
        manifest = json.loads((repo_root / MANIFEST).read_text())
    except Exception as exc:
        failures.append(f"manifest parse failed: {exc}")
        manifest = {}
    runtime = manifest.get("source_runtime", {})
    model = manifest.get("model", {})
    if manifest.get("status") != "VALIDATED_MANUAL_RECIPE":
        failures.append("manifest recipe status drift")
    if runtime.get("registry_manifest") != IMAGE:
        failures.append("manifest registry identity drift")
    if runtime.get("published_image_id") != PUBLISHED_IMAGE_ID:
        failures.append("manifest published image ID drift")
    if runtime.get("qualified_rootfs_source_image_id") != QUALIFIED_ROOTFS_IMAGE_ID:
        failures.append("manifest qualified rootfs identity drift")
    if runtime.get("vllm_commit") != VLLM_COMMIT:
        failures.append("manifest vLLM commit drift")
    if model != {"repo": MODEL, "revision": REVISION}:
        failures.append("manifest model identity drift")
    declared = manifest.get("files", {})
    if set(declared) != set(SOURCE_FILES):
        failures.append("manifest source file set is not exact")
    for name in SOURCE_FILES:
        source = repo_root / PATCH_ROOT / name
        if not source.is_file():
            failures.append(f"missing source: {name}")
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if declared.get(name) != actual:
            failures.append(f"source SHA-256 mismatch: {name}")
        try:
            compile(source.read_text(), str(source), "exec")
        except SyntaxError as exc:
            failures.append(f"source syntax error: {name}: {exc}")

    try:
        data = yaml.load((repo_root / OVERLAY).read_text(), Loader=UniqueKeyLoader)
        if not isinstance(data, dict) or set(data) != {"services"}:
            raise ValueError("top-level keys must be exactly services")
        services = data.get("services")
        if not isinstance(services, dict) or set(services) != {"head", "worker"}:
            raise ValueError("services must be exactly head and worker")
        allowed = {
            "head": {"environment", "volumes", "cap_add", "security_opt", "oom_score_adj", "shm_size", "ulimits", "healthcheck"},
            "worker": {"environment", "volumes", "cap_add", "security_opt", "oom_score_adj", "shm_size", "ulimits"},
        }
        for role in ("head", "worker"):
            service = services[role]
            if not isinstance(service, dict) or set(service) != allowed[role]:
                raise ValueError(f"{role} keys must be exactly {sorted(allowed[role])}")
            if service.get("volumes") != expected_mounts():
                raise ValueError(f"{role} volume list is not exact")
            if service.get("environment") != expected_overlay_environment():
                raise ValueError(f"{role} environment list is not exact")
            if service.get("cap_add") != ["SYS_PTRACE", "IPC_LOCK"]:
                raise ValueError(f"{role} capability list drift")
            if service.get("security_opt") != ["label=disable"]:
                raise ValueError(f"{role} security option drift")
            if service.get("oom_score_adj") != 1000:
                raise ValueError(f"{role} oom_score_adj drift")
            if service.get("shm_size") != "32gb":
                raise ValueError(f"{role} shm_size drift")
            if service.get("ulimits") != {"stack": 67108864}:
                raise ValueError(f"{role} ulimits drift")
    except Exception as exc:
        failures.append(f"compose shape invalid: {exc}")

    script = (repo_root / ENTRYPOINT).read_text()
    required = [
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-expert-parallel",
        "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'",
        "--kv-cache-memory-bytes",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        'case "${ROLE}:${NODE_RANK}"',
        "head:0|worker:1",
        'mkdir -p "${HF_HOME}"',
        '${MODEL_REVISION:?MODEL_REVISION required}',
        '${MODEL_CONTAINER_PATH}/.hf-revision',
        "model revision mismatch",
    ]
    for token in required:
        if token not in script:
            failures.append(f"entrypoint missing required contract: {token}")
    if "--language-model-only" in script:
        failures.append("entrypoint disables multimodal support with --language-model-only")

    doc = (repo_root / DOC).read_text()
    for phrase in (
        "Validated (non-production)",
        "manual activation only",
        "immutable GHCR manifest digest",
        "not auto-started",
    ):
        if phrase not in doc:
            failures.append(f"runbook lost required qualification limit: {phrase}")

    if render_compose:
        if not shutil.which("docker"):
            failures.append("docker unavailable; rendered Compose contract cannot be verified")
        else:
            for key, expected in EXPECTED_ENV.items():
                if key in os.environ and os.environ[key] != expected:
                    failures.append(
                        f"rendered Compose contract blocked by ambient {key}="
                        f"{os.environ[key]!r}; expected {expected!r}"
                    )
            trusted_env = {
                key: os.environ[key]
                for key in ("HOME", "DOCKER_CONFIG") if key in os.environ
            }
            trusted_env["PATH"] = os.environ.get("PATH", os.defpath)
            cmd = [
                "docker", "compose", "--env-file", str(PRESET),
                "-f", "docker-compose.yml", "-f", str(OVERLAY),
                "--profile", "*", "config", "--format", "json",
            ]
            proc = subprocess.run(
                cmd, cwd=repo_root, env=trusted_env, text=True, capture_output=True
            )
            if proc.returncode:
                failures.append(f"Compose render failed: {proc.stderr.strip()}")
            else:
                try:
                    rendered_services = json.loads(proc.stdout)["services"]
                    render_env = {
                        key: value for key, value in EXPECTED_ENV.items()
                        if key not in {
                            "VLLM_IMAGE", "MODEL_PATH", "ENTRYPOINT_FILE",
                            "QWEN38_NGC2609_OVERLAY_ROOT", "WORKER_NODE_RANK",
                        }
                    }
                    for role in ("head", "worker"):
                        service = rendered_services[role]
                        if service.get("image") != IMAGE:
                            raise ValueError(f"{role} image drift")
                        actual_env = service.get("environment", {})
                        for key, expected in render_env.items():
                            if actual_env.get(key) != expected:
                                raise ValueError(
                                    f"{role} {key} drift: {actual_env.get(key)!r}"
                                )
                        if service.get("restart") != "no":
                            raise ValueError(f"{role} restart policy drift")
                        if service.get("cap_add") != ["SYS_PTRACE", "IPC_LOCK"]:
                            raise ValueError(f"{role} rendered capability drift")
                        if service.get("security_opt") != ["label=disable"]:
                            raise ValueError(f"{role} rendered security option drift")
                        if service.get("oom_score_adj") != 1000:
                            raise ValueError(f"{role} rendered oom_score_adj drift")
                        if service.get("shm_size") != "34359738368":
                            raise ValueError(f"{role} rendered shm_size drift")
                        if role == "worker" and actual_env.get("NODE_RANK") != "1":
                            raise ValueError("worker rendered NODE_RANK drift")
                        volumes = {
                            item["target"]: item for item in service.get("volumes", [])
                        }
                        for name in SOURCE_FILES:
                            target = f"{TARGET_ROOT}/{name}"
                            item = volumes.get(target, {})
                            expected_source = str((repo_root / PATCH_ROOT / name).resolve())
                            if (item.get("source") != expected_source
                                    or not item.get("read_only")):
                                raise ValueError(f"{role} overlay mount drift: {name}")
                        model_mount = volumes.get("/model", {})
                        if (model_mount.get("source") != EXPECTED_ENV["MODEL_PATH"]
                                or not model_mount.get("read_only")):
                            raise ValueError(f"{role} model mount drift")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    failures.append(f"rendered Compose contract invalid: {exc}")
    return failures, warnings


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    failures, warnings = verify(root)
    for item in warnings:
        print(f"WARN: {item}")
    for item in failures:
        print(f"FAIL: {item}")
    if failures:
        return 1
    print("PASS: Qwen3.8 NGC 26.09 MTP k=2/C2 recipe is internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
