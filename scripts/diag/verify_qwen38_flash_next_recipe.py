#!/usr/bin/env python3
# =============================================================================
# Static recipe verifier for the Qwen/Qwen3.8-Flash-Next-FP8
# PRODUCTION-QUALIFIED DEFAULT recipe (MAX_NUM_SEQS=2, CUDA-graph
# FULL_DECODE_ONLY capture [1,2])
# (presets/qwen3.8-flash-next-fp8-tp2-candidate.env +
#  compose/qwen3.8-flash-next/docker-compose.candidate.yml).
#
# Fails CLOSED: any missing file, any identity mismatch, or any violation of
# the conservative safety contract exits non-zero. Prints every check result
# (not just the first failure) so a single run gives the full picture.
#
# What this script does NOT do:
#   - It does not start any container (docker compose ... up is never called).
#   - It does not require the model weights to be present on disk.
#   - It does not `source` or otherwise shell-execute the .env file -- the
#     preset is parsed as plain text (KEY=VALUE lines only).
#   - It does not depend on any pre-existing untracked file in this
#     repository -- only on the two tracked artifacts named above and the
#     tracked base docker-compose.yml.
#
# `docker compose ... config` is exercised for both the head and worker
# profiles ONLY if a working `docker compose` is available on this host; if
# not, that portion is skipped with a clear WARN (not a silent pass) and the
# purely-static checks still run and still gate the exit code.
#
# Usage:
#   python3 scripts/diag/verify_qwen38_flash_next_recipe.py
# =============================================================================

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_PATH = REPO_ROOT / "presets" / "qwen3.8-flash-next-fp8-tp2-candidate.env"
OVERLAY_PATH = REPO_ROOT / "compose" / "qwen3.8-flash-next" / "docker-compose.candidate.yml"
BASE_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
DOC_PATH = REPO_ROOT / "docs" / "qwen3.8-flash-next-tp2.md"

EXPECTED_IMAGE = (
    "vllm/vllm-openai:qwen38-flash-next"
    "@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
)
EXPECTED_DIGEST = (
    "sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
)
EXPECTED_HF_REVISION = "970c569adaca6b35532111fd6b27351b2baefe50"
EXPECTED_WEIGHT_BYTES = "185523317458"
EXPECTED_MODEL_PATH = "/home/bjk110/Documents/Models/Qwen/Qwen3.8-Flash-Next-FP8"
EXPECTED_MODEL_CONTAINER_PATH = "/models/Qwen/Qwen3.8-Flash-Next-FP8"
EXPECTED_SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-FP8"
EXPECTED_HEAD_ROCE_IP = "10.10.10.1"
EXPECTED_WORKER_ROCE_IP = "10.10.10.2"
EXPECTED_ROCE_IF_NAME = "enp1s0f0np0"
EXPECTED_IB_HCA_NAME = "rocep1s0f0"
EXPECTED_MASTER_PORT = "50000"
EXPECTED_ALL2ALL_BACKEND = "allgather_reducescatter"
EXPECTED_COMPILATION_CONFIG = (
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
)
# Structural (JSON-parsed) form of the same contract, used for the argv
# check below so equivalent-but-differently-formatted JSON (key order,
# whitespace) is not rejected -- only the semantic fields matter.
EXPECTED_COMPILATION_CONFIG_MODE = 0
EXPECTED_COMPILATION_CONFIG_CUDAGRAPH_MODE = "FULL_DECODE_ONLY"
EXPECTED_COMPILATION_CONFIG_CAPTURE_SIZES = [1, 2]

REQUIRED_EXTRA_ARGS_TOKENS = [
    "--enable-expert-parallel",
    "--no-enable-flashinfer-autotune",
    "--enable-chunked-prefill",
    "--reasoning-parser",
    "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_coder",
    "--compilation-config",
]

# Substrings that must NOT appear anywhere in VLLM_EXTRA_ARGS for this
# recipe to keep its stated production safety contract.
FORBIDDEN_EXTRA_ARGS_SUBSTRINGS = [
    "--speculative-config",   # MTP must stay off in this production recipe
    "--language-model-only",  # multimodal support must be preserved
    "--load-format",          # default lazy/mmap safetensors loading must not be overridden
    "offload",                # case-normalized check below covers PLE / cpu-offload flags
    "nvfp4",                  # this recipe is FP8-only; NVFP4 belongs to the excluded PLE profile
    "--enforce-eager",        # superseded by the exact --compilation-config contract below
]

MAX_SAFE_GPU_MEMORY_UTILIZATION = 0.90


class Results:
    def __init__(self):
        self.failures = []
        self.passes = []
        self.warnings = []

    def ok(self, msg):
        self.passes.append(msg)
        print(f"[PASS] {msg}")

    def fail(self, msg):
        self.failures.append(msg)
        print(f"[FAIL] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"[WARN] {msg}")


def parse_env_file(path):
    """Minimal KEY=VALUE .env parser. No shell execution, no variable
    expansion, no command substitution -- exactly what is needed to read
    back the literal values docker compose --env-file would load, without
    trusting the file as executable code."""
    env = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Compose env-file syntax does not honor shell quoting; strip a
        # single layer of matching quotes only if present, mirroring what
        # `docker compose --env-file` itself does.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        env[key] = value
    return env


def check_files_exist(r):
    for label, path in (
        ("preset", PRESET_PATH),
        ("compose overlay", OVERLAY_PATH),
        ("base docker-compose.yml", BASE_COMPOSE_PATH),
        ("doc", DOC_PATH),
    ):
        if path.is_file():
            r.ok(f"{label} exists: {path.relative_to(REPO_ROOT)}")
        else:
            r.fail(f"{label} MISSING: {path.relative_to(REPO_ROOT)}")


def check_preset_identities(r, env):
    if not env:
        r.fail("preset parsed to zero KEY=VALUE lines -- cannot check identities")
        return

    def expect(key, expected):
        got = env.get(key)
        if got == expected:
            r.ok(f"{key}={got!r} matches pinned value")
        else:
            r.fail(f"{key}={got!r} does NOT match required pinned value {expected!r}")

    expect("VLLM_IMAGE", EXPECTED_IMAGE)
    expect("MODEL_PATH", EXPECTED_MODEL_PATH)
    expect("MODEL_CONTAINER_PATH", EXPECTED_MODEL_CONTAINER_PATH)
    expect("SERVED_MODEL_NAME", EXPECTED_SERVED_MODEL_NAME)
    expect("CLUSTER_MODE", "dual-rdma")
    expect("TP_SIZE", "2")
    expect("DISTRIBUTED_BACKEND", "mp")
    expect("HEAD_ROCE_IP", EXPECTED_HEAD_ROCE_IP)
    expect("WORKER_ROCE_IP", EXPECTED_WORKER_ROCE_IP)
    expect("ROCE_IF_NAME", EXPECTED_ROCE_IF_NAME)
    expect("IB_HCA_NAME", EXPECTED_IB_HCA_NAME)
    expect("MASTER_PORT", EXPECTED_MASTER_PORT)
    expect("MAX_MODEL_LEN", "262144")
    expect("MAX_NUM_SEQS", "2")
    expect("MAX_NUM_BATCHED_TOKENS", "8192")
    expect("VLLM_ALL2ALL_BACKEND", EXPECTED_ALL2ALL_BACKEND)
    expect("MAX_JOBS", "4")
    expect("FLASHINFER_NVCC_THREADS", "1")

    if "PYTORCH_CUDA_ALLOC_CONF" in env:
        r.fail(
            "PYTORCH_CUDA_ALLOC_CONF is set in the preset "
            f"({env['PYTORCH_CUDA_ALLOC_CONF']!r}) -- it must stay absent here; "
            "only the compose overlay's ${PYTORCH_CUDA_ALLOC_CONF-} may force it empty"
        )
    else:
        r.ok("PYTORCH_CUDA_ALLOC_CONF is absent from the preset (overlay is the enforcement point)")

    gmu_raw = env.get("GPU_MEMORY_UTILIZATION")
    try:
        gmu = float(gmu_raw)
    except (TypeError, ValueError):
        r.fail(f"GPU_MEMORY_UTILIZATION={gmu_raw!r} is not a parseable float")
    else:
        if gmu == 0.83:
            r.ok(f"GPU_MEMORY_UTILIZATION={gmu} matches the conservative envelope")
        else:
            r.fail(f"GPU_MEMORY_UTILIZATION={gmu} does not match required 0.83")
        if not (0.0 < gmu <= MAX_SAFE_GPU_MEMORY_UTILIZATION):
            r.fail(
                f"GPU_MEMORY_UTILIZATION={gmu} outside safe bound "
                f"(0, {MAX_SAFE_GPU_MEMORY_UTILIZATION}]"
            )


def check_extra_args_contract(r, env):
    extra = env.get("VLLM_EXTRA_ARGS")
    if extra is None:
        r.fail("VLLM_EXTRA_ARGS is not set in the preset")
        return
    tokens = extra.split()
    missing = [t for t in REQUIRED_EXTRA_ARGS_TOKENS if t not in tokens]
    if missing:
        r.fail(f"VLLM_EXTRA_ARGS missing required token(s): {missing}")
    else:
        r.ok("VLLM_EXTRA_ARGS contains every required conservative-envelope flag")

    extra_lower = extra.lower()
    hit = [s for s in FORBIDDEN_EXTRA_ARGS_SUBSTRINGS if s.lower() in extra_lower]
    if hit:
        r.fail(f"VLLM_EXTRA_ARGS contains forbidden substring(s) for this recipe: {hit}")
    else:
        r.ok(
            "VLLM_EXTRA_ARGS contains none of the forbidden substrings "
            "(MTP/PLE/BF16/NVFP4/load-format/--enforce-eager)"
        )

    # --compilation-config must carry exactly this production recipe's
    # measured contract -- not merely be present. A wrong or looser value
    # here (e.g. a larger cudagraph_capture_sizes set, or mode>0) would
    # silently exceed what §5.8 actually re-qualified.
    if "--compilation-config" in tokens:
        idx = tokens.index("--compilation-config")
        got = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if got == EXPECTED_COMPILATION_CONFIG:
            r.ok(f"--compilation-config value matches the exact production contract ({EXPECTED_COMPILATION_CONFIG})")
        else:
            r.fail(
                f"--compilation-config value {got!r} does NOT match the required exact "
                f"production contract {EXPECTED_COMPILATION_CONFIG!r}"
            )
    else:
        r.fail("VLLM_EXTRA_ARGS missing required --compilation-config token")

    # VLLM_EXTRA_ARGS is space-split by entrypoint.sh -- any JSON-valued
    # token must survive that split as one argv element. None of the
    # required flags above take a JSON value today; this check simply
    # verifies no stray unmatched brace/space combination is present that
    # would indicate a future JSON flag was added without collapsing its
    # internal spaces.
    if re.search(r"\{[^{}]*\s[^{}]*\}", extra):
        r.fail(
            "VLLM_EXTRA_ARGS contains a brace-delimited value with an internal "
            "space -- this will be split into multiple argv tokens by "
            "entrypoint.sh's word-splitting. Use compact JSON with no internal spaces."
        )
    else:
        r.ok("VLLM_EXTRA_ARGS has no brace-delimited value with an internal space")


def check_preset_header_identities(r):
    text = PRESET_PATH.read_text() if PRESET_PATH.is_file() else ""
    for label, needle in (
        ("HF revision pin", EXPECTED_HF_REVISION),
        ("weight byte count", EXPECTED_WEIGHT_BYTES),
        ("image digest", EXPECTED_DIGEST),
    ):
        if needle in text:
            r.ok(f"preset header documents {label} ({needle})")
        else:
            r.fail(f"preset header MISSING {label} ({needle})")


def check_doc_identities_and_status(r):
    if not DOC_PATH.is_file():
        return
    text = DOC_PATH.read_text()
    for label, needle in (
        ("HF revision pin", EXPECTED_HF_REVISION),
        ("weight byte count", EXPECTED_WEIGHT_BYTES),
        ("image digest", EXPECTED_DIGEST),
        ("status banner", "PRODUCTION-QUALIFIED -- c1/c2"),
        ("not-an-auto-start-service disclaimer", "not an auto-start service"),
        ("MAX_NUM_SEQS=4 blocked disclaimer", "remains BLOCKED"),
        ("compilation-config exact contract", EXPECTED_COMPILATION_CONFIG),
    ):
        if needle in text:
            r.ok(f"doc contains {label} ({needle!r})")
        else:
            r.fail(f"doc MISSING {label} ({needle!r})")

    # Fail closed on the doc accidentally claiming a runtime result it
    # cannot have, since no gate beyond Gate 0 has run. These checks are
    # deliberately narrow (standalone uppercase PASS token; explicit
    # negation-aware) to avoid false-firing on legitimate prose like
    # "only after Gate 2 passes" or "NOT runtime-validated".
    NEGATION_WINDOW = 20
    claim_hit = False

    for m in re.finditer(r"Gate\s*[0-3]", text):
        window = text[m.end(): m.end() + NEGATION_WINDOW]
        if re.search(r"\bPASS\b", window):  # case-sensitive: literal "PASS", not "passes"/"passing"
            claim_hit = True
            r.fail(f"doc appears to claim an executed gate result near: {m.group(0)!r} + {window!r}")

    for m in re.finditer(r"runtime[- ]validated", text, re.IGNORECASE):
        before = text[max(0, m.start() - NEGATION_WINDOW): m.start()]
        if not re.search(r"\bnot\b", before, re.IGNORECASE):
            claim_hit = True
            r.fail(f"doc claims 'runtime-validated' without a preceding negation: ...{before!r}{m.group(0)!r}")

    if re.search(r"measured\s+t/s", text, re.IGNORECASE):
        claim_hit = True
        r.fail("doc contains an unsupported measured t/s benchmark claim")

    if not claim_hit:
        r.ok("doc contains no unexpected runtime-result claim pattern")


FOLLOWUP_EVIDENCE_PATHS = [
    "/home/bjk110/docker-build/qwen38-fp8-32k-c2-prefixon-20260829/",
    "/home/bjk110/docker-build/qwen38-fp8-32k-c2-prefixoff-20260829/",
    "/home/bjk110/docker-build/qwen38-fp8-32k-c2-noep-20260829/",
    "/home/bjk110/docker-build/qwen38-fp8-32k-c2-cutlass-20260829/",
]

BIC_PREFLIGHT_EVIDENCE_PATH = (
    "/home/bjk110/docker-build/qwen38-fp8-bic-support-check-20260829/"
)


def check_followup_ledger(r):
    """Fail-closed checks for the 2026-08-29 32K/c2 follow-up (doc §5.6):
    the doc must cite every evidence path and the explicit stopping-point
    disclaimer, and no c2-oriented preset may exist as a result of it."""
    if not DOC_PATH.is_file():
        return
    text = DOC_PATH.read_text()

    for path_str in FOLLOWUP_EVIDENCE_PATHS:
        if path_str in text:
            r.ok(f"doc contains 32K/c2 follow-up evidence path ({path_str})")
        else:
            r.fail(f"doc MISSING 32K/c2 follow-up evidence path ({path_str})")

    if "Supervisor stopping point" in text:
        r.ok("doc contains the 32K/c2 follow-up supervisor stopping point")
    else:
        r.fail("doc MISSING the 32K/c2 follow-up supervisor stopping point")

    # Fail closed if the doc doesn't explicitly disclaim the obvious
    # misreading of this follow-up -- that c2 exact-output determinism
    # authorizes a c2 preset or general concurrent serving. It does not.
    if re.search(r"do not.{0,20}(create|promote).{0,20}c2 preset", text, re.IGNORECASE | re.DOTALL):
        r.ok("doc explicitly states the 32K/c2 result does not authorize a c2 preset")
    else:
        r.fail("doc does not explicitly disclaim c2-preset promotion from the 32K/c2 follow-up")

    # No c2-oriented preset file should exist as a result of this follow-up --
    # promotion requires separate explicit authorization, not this ledger.
    presets_dir = REPO_ROOT / "presets"
    c2_candidates = sorted(
        p.name for p in presets_dir.glob("qwen3.8-flash-next*c2*")
    ) if presets_dir.is_dir() else []
    if c2_candidates:
        r.fail(f"found unexpected c2-oriented Qwen3.8-Flash-Next preset file(s): {c2_candidates}")
    else:
        r.ok("no c2-oriented Qwen3.8-Flash-Next preset file exists (no premature promotion)")

    # This preset's MAX_NUM_SEQS pin is 2 in production, but it must be
    # justified by the separate, later §5.8 CUDA-graph production
    # requalification -- NOT by this 32K/c2 follow-up alone. Checked again
    # here (in addition to check_preset_identities) specifically in the
    # follow-up's context so a future edit that raises MAX_NUM_SEQS "because
    # the follow-up looked clean" without the doc's §5.8 distinction is
    # caught under this section too.
    env = parse_env_file(PRESET_PATH)
    if env.get("MAX_NUM_SEQS") == "2":
        r.ok("preset MAX_NUM_SEQS is 2, per the separate later §5.8 production requalification")
    else:
        r.fail(
            f"preset MAX_NUM_SEQS={env.get('MAX_NUM_SEQS')!r} -- expected 2, matching the §5.8 "
            "production requalification pin (this 32K/c2 follow-up alone does not authorize it)"
        )

    if "did not requalify" in text:
        r.ok("doc explicitly states the 32K/c2 follow-up alone did not requalify MAX_NUM_SEQS>1")
    else:
        r.fail(
            "doc does not explicitly distinguish the 32K/c2 follow-up from the later §5.8 "
            "production requalification that actually justifies MAX_NUM_SEQS=2"
        )

    # The pinned image's Qwen GDN backend explicitly reports no batch-invariant
    # support. Preserve both the classification and the no-launch decision.
    if BIC_PREFLIGHT_EVIDENCE_PATH in text:
        r.ok("doc contains the batch-invariant static-preflight evidence path")
    else:
        r.fail("doc MISSING the batch-invariant static-preflight evidence path")

    required_bic_phrases = (
        "GDNAttentionBackend.supports_batch_invariance",
        "returns `False`",
        "static preflight stop",
        "deliberately not launched",
    )
    for phrase in required_bic_phrases:
        if phrase in text:
            r.ok(f"doc contains batch-invariant classification phrase ({phrase!r})")
        else:
            r.fail(f"doc MISSING batch-invariant classification phrase ({phrase!r})")

    if "VLLM_BATCH_INVARIANT" not in env:
        r.ok("preset does not enable unsupported VLLM_BATCH_INVARIANT")
    else:
        r.fail(
            "preset unexpectedly sets VLLM_BATCH_INVARIANT; the pinned Qwen GDN backend "
            "does not support batch-invariant mode"
        )


def check_overlay_structure(r):
    if not OVERLAY_PATH.is_file():
        return
    text = OVERLAY_PATH.read_text()

    # Lightweight structural checks (line/keyword based, not a full YAML
    # parse) -- intentionally dependency-free. The authoritative structural
    # check is the `docker compose ... config` run below when available.
    if "healthcheck" in text and "head" in text:
        r.ok("overlay defines a healthcheck block")
    else:
        r.fail("overlay does not appear to define a healthcheck block")

    if "VLLM_ALL2ALL_BACKEND" in text:
        r.ok("overlay forwards VLLM_ALL2ALL_BACKEND")
    else:
        r.fail("overlay does not forward VLLM_ALL2ALL_BACKEND")

    alloc_conf_hits = text.count("PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-}")
    if alloc_conf_hits == 2:
        r.ok("overlay contains exactly two literal PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-} entries (head + worker)")
    else:
        r.fail(
            "overlay contains "
            f"{alloc_conf_hits} literal PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF-}} entries "
            "-- expected exactly 2 (one for head, one for worker, unset-only fallback)"
        )

    # Structural check only -- strip comment lines first so this doesn't
    # false-fire on prose that explains *why* prewarm/depends_on are absent
    # (which necessarily mentions those words).
    code_lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("#")
    ]
    code_text = "\n".join(code_lines)
    if re.search(r"^\s*depends_on\s*:", code_text, re.MULTILINE):
        r.fail("overlay contains an actual `depends_on:` key -- spec requires healthcheck-only, no auto-deps")
    else:
        r.ok("overlay contains no actual `depends_on:` key (healthcheck-only, as required)")
    if re.search(r"^\s{2}prewarm\s*:", code_text, re.MULTILINE):
        r.fail("overlay defines an actual `prewarm:` service -- spec requires healthcheck-only, no auto-prewarm")
    else:
        r.ok("overlay defines no `prewarm:` service (healthcheck-only, as required)")

    if re.search(r"^\s*image:\s*\S", text, re.MULTILINE):
        r.fail("overlay sets an `image:` override -- VLLM_IMAGE must keep coming from the preset")
    else:
        r.ok("overlay does not override `image:`")


def docker_compose_available():
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def render_compose_config(profile):
    cmd = [
        "docker", "compose",
        "--env-file", str(PRESET_PATH),
        "-f", str(BASE_COMPOSE_PATH),
        "-f", str(OVERLAY_PATH),
        "--profile", profile,
        "config",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    return proc


def check_compose_config(r):
    if not docker_compose_available():
        r.warn("docker compose not available on this host -- skipping `config` render checks "
               "(all other checks still gate the exit code)")
        return

    for profile, expected_role in (("head", "head"), ("worker", "worker")):
        proc = render_compose_config(profile)
        if proc.returncode != 0:
            r.fail(
                f"`docker compose --profile {profile} config` exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
            continue
        rendered = proc.stdout
        r.ok(f"`docker compose --profile {profile} config` rendered successfully")

        if EXPECTED_DIGEST in rendered:
            r.ok(f"[{profile}] rendered config contains the pinned image digest")
        else:
            r.fail(f"[{profile}] rendered config MISSING the pinned image digest")

        if re.search(rf"ROLE:\s*{expected_role}\b", rendered):
            r.ok(f"[{profile}] rendered config sets ROLE={expected_role}")
        else:
            r.fail(f"[{profile}] rendered config does not set ROLE={expected_role}")

        other_role = "worker" if expected_role == "head" else "head"
        if re.search(rf"ROLE:\s*{other_role}\b", rendered):
            r.fail(f"[{profile}] rendered config unexpectedly also sets ROLE={other_role}")
        else:
            r.ok(f"[{profile}] rendered config does not set the opposite ROLE={other_role}")

        if f"VLLM_ALL2ALL_BACKEND: {EXPECTED_ALL2ALL_BACKEND}" in rendered:
            r.ok(f"[{profile}] rendered config forwards VLLM_ALL2ALL_BACKEND={EXPECTED_ALL2ALL_BACKEND}")
        else:
            r.fail(f"[{profile}] rendered config does not forward VLLM_ALL2ALL_BACKEND correctly")

        if "MASTER_PORT: \"50000\"" in rendered or f"MASTER_PORT: {EXPECTED_MASTER_PORT}" in rendered:
            r.ok(f"[{profile}] rendered config carries MASTER_PORT={EXPECTED_MASTER_PORT}")
        else:
            r.warn(f"[{profile}] could not confirm MASTER_PORT in rendered config (formatting-dependent)")


def main():
    r = Results()

    check_files_exist(r)
    env = parse_env_file(PRESET_PATH)
    check_preset_identities(r, env)
    check_extra_args_contract(r, env)
    check_preset_header_identities(r)
    check_doc_identities_and_status(r)
    check_followup_ledger(r)
    check_overlay_structure(r)
    check_compose_config(r)

    print()
    print(f"Summary: {len(r.passes)} pass, {len(r.warnings)} warn, {len(r.failures)} fail")
    if r.failures:
        print("FATAL: recipe verification failed closed. See [FAIL] lines above.")
        return 1
    print("Recipe verification PASSED (static checks only -- see docs/qwen3.8-flash-next-tp2.md "
          "for the production-qualified c1/c2 ledger with MAX_NUM_SEQS=4 unbounded still blocked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
