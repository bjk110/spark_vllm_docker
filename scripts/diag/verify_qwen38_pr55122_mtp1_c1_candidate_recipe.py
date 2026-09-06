#!/usr/bin/env python3
"""Fail-closed static verifier for the locally promoted PR55122/MTP1/c1 recipe."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "dockerfiles/active/Dockerfile.qwen38-pr55122-mtp1-c1-candidate"
KERNEL_PATH = REPO_ROOT / "patches/qwen/pr55122/_C_det.so"
QSA_PATH = REPO_ROOT / "patches/qwen/pr55122/qsa.det.py"
UPSTREAM_KERNEL_PATH = REPO_ROOT / "patches/qwen/pr55122/persistent_topk.cuh"
PRESET_PATH = REPO_ROOT / "presets/qwen3.8-flash-next-fp8-tp2-pr55122-mtp1-c1-candidate.env"
PRODUCTION_PRESET_PATH = REPO_ROOT / "presets/qwen3.8-flash-next-fp8-tp2-candidate.env"
OVERLAY_PATH = REPO_ROOT / "compose/qwen3.8-flash-next/docker-compose.pr55122-mtp1-c1-candidate.yml"
DOC_PATH = REPO_ROOT / "docs/qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md"
BASE_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
EXPECTED_BASE = "vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
EXPECTED_IMAGE_ID = "sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610"
EXPECTED_KERNEL_SHA = "803037813e307a7c5fa05efee5e874695a0620a336dd798800ccd51d566377b3"
EXPECTED_QSA_SHA = "7e6aff79a37b866b130de65bd483a6526d9e8ae14cc45733c2deb080ff2f2c47"
EXPECTED_UPSTREAM_KERNEL_SHA = "db6f9c2b2c580ddb97b7026e01100ae23b2e2bfa5dbaa6396ba61189cc33fe6c"
EXPECTED_DOC_SHA = "85faa958b427fe695637599c5a945f0dc5ae2bd90f0c567aadbfd704964f8233"
EXPECTED_PRESET_SHA = "75544abe918284fd1d027901d0007e5809da4ed177c5331bd44e41e0d6603452"
KERNEL_SOURCE = "patches/qwen/pr55122/_C_det.so"
QSA_SOURCE = "patches/qwen/pr55122/qsa.det.py"
KERNEL_DEST = "/opt/llm/kernel-det/_C_det.so"
QSA_DEST = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
COPY_FLAGS = ["--chmod=0644"]
EXPECTED_LABELS = {
    "org.vllm-spark.candidate.purpose": "LOCAL NON-PRODUCTION experimental deterministic persistent_topk evaluation; not promoted or pushed",
    "org.vllm-spark.candidate.base-image": EXPECTED_BASE,
    "org.vllm-spark.candidate.base-image-id": "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a",
    "org.vllm-spark.candidate.vllm-version": "0.1.dev20073+g8e685d198",
    "org.vllm-spark.candidate.upstream-pr": "https://github.com/vllm-project/vllm/pull/55122",
    "org.vllm-spark.candidate.upstream-pr-head": "1c4caa3dbe1c3c616f63810fd6fa60877105132d",
    "org.vllm-spark.candidate.community-source": "https://github.com/jschmied/qwen38-flash-next-gb10@0c5598782b33bbfc9acb46acd57b495ca0eb01b7",
    "org.vllm-spark.candidate.kernel-sha256": EXPECTED_KERNEL_SHA,
    "org.vllm-spark.candidate.qsa-original-sha256": "c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f",
    "org.vllm-spark.candidate.qsa-patched-sha256": EXPECTED_QSA_SHA,
    "org.vllm-spark.candidate.activation": "VLLM_QSA_DET_TOPK=1 and VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so",
}
EXPECTED_PROJECT_NAME = "qwen38-pr55122-mtp1-c1-candidate"
EXPECTED_SERVICES = {"head", "worker"}
EXPECTED_ENV = {"VLLM_QSA_DET_TOPK":"1", "VLLM_QSA_DET_LIB":KERNEL_DEST,
                "VLLM_ALL2ALL_BACKEND":"allgather_reducescatter", "PYTORCH_CUDA_ALLOC_CONF":""}
EXPECTED_ENV_ENTRIES = [f"{key}={value}" for key,value in EXPECTED_ENV.items()]
EXPECTED_SERVICE_KEYS = {"head": {"environment", "healthcheck"}, "worker": {"environment"}}
LOCAL_IMAGE_GATE = "QWEN38_PR55122_REQUIRE_LOCAL_IMAGE_ID"
IMAGE_HOSTS_ENV = "QWEN38_PR55122_IMAGE_ID_HOSTS"
APPROVED_EXTRA_ARGS_SUFFIX = ' --no-enable-prefix-caching --speculative-config={"method":"mtp","num_speculative_tokens":1}'

PROMOTION_INDEX_PATHS = {
    "README.md": REPO_ROOT / "README.md",
    "presets/README.md": REPO_ROOT / "presets/README.md",
    "docs/README.md": REPO_ROOT / "docs/README.md",
    "docs/images.md": REPO_ROOT / "docs/images.md",
    "dockerfiles/README.md": REPO_ROOT / "dockerfiles/README.md",
    "patches/README.md": REPO_ROOT / "patches/README.md",
}
PROMOTION_INDEX_EXPECTED_SHAS = {
    "README.md": "2191e767b0297e11a05f5fce5b4f993f1fb6b76f4cbe067ddb3ead64ef77791a",
    "presets/README.md": "f2e2381b49fe479591dcc0a47925032c013bd39dc42c3e29791e7faeeb936347",
    "docs/README.md": "b48136c403e34e4884bf27eaa9906e294d36b84ca34d466304c8b156f08bf7be",
    "docs/images.md": "9be679646d090730630fe0361525999feadf7aacfe2f33d9a3de6205fea8b10a",
    "dockerfiles/README.md": "30a29a01a7ca4137bbb7054f06bc9a943d00cbd98da0dfcfd728986b39102ce5",
    "patches/README.md": "be4af154f2aad4c001f0fa4368df204dd47210bad9a24594d91fec6687f1d586",
}
PROMOTION_INDEX_REQUIRED_LINES = {
    "README.md": (
        "Three independent model-family repository paths are documented: DeepSeek-V4-Flash, Solar-Open2-250B,",
        "port 8000), are not served simultaneously, and no model is inferred active from this index.",
        "| `qwen3.8-flash-next-c1-c2` | **Qwen3.8-Flash-Next production-qualified baseline** — manual c1/c2, not auto-start | `mp` | Authoritative existing TP2 baseline: `MAX_NUM_SEQS=2`, prefix cache ON, FULL_DECODE_ONLY `[1,2]`; MTP off. |",
        "| `qwen3.8-flash-next-pr55122-mtp1-c1` | `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` — opt-in interactive c1 only; NOT the production default; no auto-start | `mp` | Additional TP2 MTP depth-1 profile: `MAX_NUM_SEQS=2`, prefix cache OFF, FULL_DECODE_ONLY `[1,2]`, local image ID `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610`; MTP1 c2/c8, prefix-cache ON exact, and `MAX_NUM_SEQS=8` performance excluded. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
    "presets/README.md": (
        "Qwen3.8-Flash-Next-FP8 has two indexed presets: the existing production-qualified c1/c2 baseline and",
        "| `qwen3.8-flash-next-fp8-tp2-candidate.env` | Qwen/Qwen3.8-Flash-Next-FP8 | **Production-qualified** (manual activation only, not auto-start) | dual-rdma TP2 (mp), production-qualified at c1/c2, `MAX_NUM_SEQS=2`, `FULL_DECODE_ONLY` capture sizes `[1,2]`, `MAX_MODEL_LEN=262144`, `GPU_MEMORY_UTILIZATION=0.83`. Requires `compose/qwen3.8-flash-next/docker-compose.candidate.yml` overlay. Reboot before next fresh launch after teardown. `MAX_NUM_SEQS=4` remains BLOCKED (content diverges across identical repeats at c2/c4) and must not be raised. MTP and PLE-offloaded NVFP4 excluded from the default; MTP untested/off. |",
        "| `qwen3.8-flash-next-fp8-tp2-pr55122-mtp1-c1-candidate.env` | Qwen/Qwen3.8-Flash-Next-FP8 | `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` — opt-in interactive c1 only; NOT the production default; no auto-start | dual-rdma TP2 (mp), MTP depth 1, `MAX_NUM_SEQS=2`, prefix cache OFF, FULL_DECODE_ONLY `[1,2]`, exact local image ID `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610`; requires `compose/qwen3.8-flash-next/docker-compose.pr55122-mtp1-c1-candidate.yml`. MTP1 c2/c8, prefix-cache ON exact, and `MAX_NUM_SEQS=8` performance excluded. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
    "docs/README.md": (
        "| [qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md](qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md) | Detailed PR55122 deterministic-kernel MTP1 interactive-c1 opt-in recipe and Gate0–Gate5 evidence | `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` — NOT the production default; no auto-start | Local product/config authority only: TP2, MTP depth 1, `MAX_NUM_SEQS=2`, prefix cache OFF, FULL_DECODE_ONLY `[1,2]`; MTP1 c2/c8, prefix-cache ON exact, and `MAX_NUM_SEQS=8` performance excluded. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
    "docs/images.md": (
        "## Qwen3.8 PR55122 MTP1 c1 registry-free local image authority",
        "| `qwen3.8-flash-next-pr55122-mtp1-c1` | `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` — opt-in interactive c1 only; NOT the production default; no auto-start | `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610` (registry-free local image authority on spark01/spark02) | TP2, MTP depth 1, `MAX_NUM_SEQS=2`, prefix cache OFF, FULL_DECODE_ONLY `[1,2]`; MTP1 c2/c8, prefix-cache ON exact, and `MAX_NUM_SEQS=8` performance excluded. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
    "dockerfiles/README.md": (
        "| `Dockerfile.qwen38-pr55122-mtp1-c1-candidate` | no tag authorized | `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` reconstruction-only Dockerfile for the opt-in Qwen3.8 PR55122 MTP1 interactive-c1 recipe. Preserves immutable build-time provenance labels and documents COPY closure; it does not reproduce image ID `sha256:5c957f7cc93f1944a310d9b4858a6fd33718fc2beb7bedfa7eb7da499d8b2610`. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
    "patches/README.md": (
        "| `qwen/pr55122/_C_det.so` | Qwen3.8 deterministic top-k | Active for `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` local opt-in recipe only | SHA256 `803037813e307a7c5fa05efee5e874695a0620a336dd798800ccd51d566377b3`; used only by the Qwen3.8 PR55122 MTP1 interactive-c1 reconstruction recipe. No broader model/profile use. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
        "| `qwen/pr55122/qsa.det.py` | Qwen3.8 QSA integration | Active for `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` local opt-in recipe only | SHA256 `7e6aff79a37b866b130de65bd483a6526d9e8ae14cc45733c2deb080ff2f2c47`; used only by the Qwen3.8 PR55122 MTP1 interactive-c1 reconstruction recipe. No broader model/profile use. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
        "| `qwen/pr55122/persistent_topk.cuh` | Qwen3.8 upstream source provenance | Active for `PROMOTED_LOCAL_CONFIG_NOT_RELEASED` local opt-in recipe only | SHA256 `db6f9c2b2c580ddb97b7026e01100ae23b2e2bfa5dbaa6396ba61189cc33fe6c`; pinned reconstruction/provenance source only for the Qwen3.8 PR55122 MTP1 interactive-c1 recipe. No broader model/profile use. Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized. |",
    ),
}

class Results:
    def __init__(self): self.failures=[]; self.passes=[]; self.warnings=[]
    def ok(self,msg): self.passes.append(msg); print(f"[PASS] {msg}")
    def fail(self,msg): self.failures.append(msg); print(f"[FAIL] {msg}")
    def warn(self,msg): self.warnings.append(msg); print(f"[WARN] {msg}")

def parse_env(path):
    result={}
    if not path.is_file(): return result
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line=raw.strip()
        if not line or line.startswith("#"): continue
        match=re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", raw)
        if not match:
            raise ValueError(f"{path}:{line_number}: invalid preset syntax: {raw!r}")
        key, value=match.groups()
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate preset key {key!r}")
        result[key]=value
    return result

def check_files(r):
    for label,path in (("Dockerfile",DOCKERFILE_PATH),("kernel",KERNEL_PATH),("QSA patch",QSA_PATH),("upstream kernel source",UPSTREAM_KERNEL_PATH),("preset",PRESET_PATH),("production preset",PRODUCTION_PRESET_PATH),("overlay",OVERLAY_PATH),("doc",DOC_PATH),("base compose",BASE_COMPOSE_PATH)):
        (r.ok if path.is_file() else r.fail)(f"{label} {'exists' if path.is_file() else 'MISSING'}: {path}")

def check_hashes(r):
    for label,path,want in (("kernel",KERNEL_PATH,EXPECTED_KERNEL_SHA),("QSA patch",QSA_PATH,EXPECTED_QSA_SHA),("upstream kernel source",UPSTREAM_KERNEL_PATH,EXPECTED_UPSTREAM_KERNEL_SHA)):
        if not path.is_file(): r.fail(f"cannot hash missing {label}"); continue
        got=hashlib.sha256(path.read_bytes()).hexdigest()
        (r.ok if got==want else r.fail)(f"{label} SHA256 {got} {'matches' if got==want else 'does not match'} {want}")

def check_dockerfile(r):
    if not DOCKERFILE_PATH.is_file(): r.fail("Dockerfile missing"); return
    text=DOCKERFILE_PATH.read_text()
    parser_directives=[line for line in text.splitlines() if re.match(r"^\s*#\s*(?:syntax|escape|check)\s*=",line,re.I)]
    (r.fail if parser_directives else r.ok)(f"Dockerfile has no build-active parser directive; found={parser_directives!r}")
    lines=[x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    bad=[x for x in lines if x.split()[0].upper() not in {"FROM","LABEL","COPY"}]
    (r.fail if bad else r.ok)(f"Dockerfile closed instruction set; unexpected={bad}")
    froms=[x for x in lines if x.upper().startswith("FROM ")]
    (r.ok if froms==[f"FROM {EXPECTED_BASE}"] else r.fail)(f"Dockerfile exact FROM line; found={froms}")
    label_lines=[x for x in lines if x.upper().startswith("LABEL ")]
    labels={}
    malformed=[]
    for line in label_lines:
        match=re.fullmatch(r'LABEL ([A-Za-z0-9_.-]+)="([^"]*)"',line,re.I)
        if not match: malformed.append(line)
        else: labels[match.group(1)]=match.group(2)
    if not malformed and labels==EXPECTED_LABELS and len(label_lines)==len(EXPECTED_LABELS): r.ok("Dockerfile exact closed LABEL key/value set")
    else: r.fail(f"Dockerfile LABEL key/value set mismatch: labels={labels!r}, malformed={malformed!r}, count={len(label_lines)}")
    copies=[x for x in lines if x.upper().startswith("COPY ")]
    counts={instruction:sum(1 for x in lines if x.split()[0].upper()==instruction) for instruction in ("FROM","LABEL","COPY")}
    expected_counts={"FROM":1,"LABEL":len(EXPECTED_LABELS),"COPY":2}
    (r.ok if counts==expected_counts and len(lines)==sum(expected_counts.values()) else r.fail)(f"Dockerfile exact instruction counts: found={counts}, total={len(lines)}, required={expected_counts}")
    expected={(KERNEL_SOURCE,KERNEL_DEST),(QSA_SOURCE,QSA_DEST)}; found=set()
    for line in copies:
        tokens=line.split(); flags=[]; args=[]; positional_seen=False
        for token in tokens[1:]:
            if token.startswith("--") and not positional_seen:
                flags.append(token)
            else:
                positional_seen=True
                args.append(token)
        if any(token.startswith("--") for token in args):
            r.fail(f"COPY flags must be an exact contiguous prefix after COPY: {tokens[1:]!r}")
        if flags != COPY_FLAGS:
            r.fail(f"COPY flag contract violated: {flags!r}; required {COPY_FLAGS!r}")
        if len(args)!=2:
            r.fail(f"COPY must have one source and one destination: {args!r}")
        else: found.add(tuple(args))
    if found==expected and len(copies)==2: r.ok("Dockerfile COPY sources and destinations match exactly")
    else: r.fail(f"Dockerfile COPY source/destination contract mismatch: {found!r}")

def _service_blocks(text):
    blocks={}; current=None
    for line in text.splitlines():
        m=re.match(r'^  ["\']?([A-Za-z0-9_.-]+)["\']?\s*:\s*$',line)
        if m: current=m.group(1); blocks[current]=[]; continue
        if current and (not line.strip() or len(line)-len(line.lstrip())>=4): blocks[current].append(line)
        elif line and not line.startswith(" "): current=None
    return blocks

def check_overlay_structure(r):
    if not OVERLAY_PATH.is_file(): r.fail("overlay missing"); return
    text=OVERLAY_PATH.read_text(); code="\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    if re.search(r'(?<![A-Za-z0-9_.-])[&*][A-Za-z0-9_][A-Za-z0-9_.-]*|<<\s*:',code): r.fail("YAML anchor/alias/merge syntax is forbidden")
    else: r.ok("overlay has no YAML anchor, alias, or merge syntax")
    top={m.group(1) for m in re.finditer(r'^(?:["\']?)([A-Za-z0-9_.-]+)(?:["\']?)\s*:',code,re.M)}
    if top=={"name","services"}: r.ok("overlay top-level keys are closed")
    else: r.fail(f"overlay top-level key set mismatch: {sorted(top)}")
    try:
        parsed_overlay=yaml.safe_load(code)
    except yaml.YAMLError as exc:
        r.fail(f"overlay YAML parse failed before name check: {exc}")
        parsed_overlay={}
    overlay_name=parsed_overlay.get("name") if isinstance(parsed_overlay,dict) else None
    if type(overlay_name) is str and overlay_name==EXPECTED_PROJECT_NAME: r.ok("overlay name is exact candidate project name")
    else: r.fail(f"overlay name must be exact string {EXPECTED_PROJECT_NAME!r}: found {overlay_name!r} ({type(overlay_name).__name__})")
    blocks=_service_blocks(code); services=set(blocks)
    if services==EXPECTED_SERVICES: r.ok("overlay service set is exactly head/worker")
    else: r.fail(f"overlay service set contains extra service or is incomplete: {sorted(services)}")
    for role in EXPECTED_SERVICES:
        body="\n".join(blocks.get(role,[]))
        service={}
        try:
            parsed=yaml.safe_load(code)
            service=parsed.get("services",{}).get(role,{})
            direct_keys=set(service) if isinstance(service,dict) else set()
        except (yaml.YAMLError, TypeError, AttributeError) as exc:
            r.fail(f"overlay YAML parse failed: {exc}")
            direct_keys=set()
        want_keys=EXPECTED_SERVICE_KEYS[role]
        if direct_keys==want_keys: r.ok(f"{role} direct-child key set is exactly {sorted(want_keys)}")
        else: r.fail(f"{role} direct-child key set mismatch: found {sorted(direct_keys)}, required {sorted(want_keys)}")
        environment=service.get("environment") if isinstance(service,dict) else None
        if environment==EXPECTED_ENV_ENTRIES: r.ok(f"{role} environment entries are exact")
        else: r.fail(f"{role} environment entries mismatch: found {environment!r}, required {EXPECTED_ENV_ENTRIES!r}")
        if role=="head":
            healthcheck=service.get("healthcheck") if isinstance(service,dict) else None
            if isinstance(healthcheck,dict) and "disable" not in healthcheck: r.ok("head healthcheck disable is absent")
            else: r.fail(f"head healthcheck disable is forbidden: found {healthcheck!r}")
            expected_health_keys={"test","interval","timeout","retries","start_period"}
            health_keys=set(healthcheck) if isinstance(healthcheck,dict) else set()
            if health_keys == expected_health_keys: r.ok("head healthcheck key set is exact")
            else: r.fail(f"head healthcheck key set mismatch: found {sorted(health_keys)}, required {sorted(expected_health_keys)}")
            test=healthcheck.get("test") if isinstance(healthcheck,dict) else None
            if isinstance(test,list) and test and test[0]=="CMD": r.ok("head healthcheck test uses exec-form CMD")
            else: r.fail(f"head healthcheck test must use exec-form CMD: found {test!r}")
            if isinstance(test,list) and len(test)>3 and test[3]=="http://127.0.0.1:8000/health": r.ok("head healthcheck URL is exact loopback health endpoint")
            else: r.fail(f"head healthcheck URL must be exact loopback health endpoint: found {test!r}")
            expected_test=["CMD","curl","-f","http://127.0.0.1:8000/health"]
            if test==expected_test: r.ok("head healthcheck test list is exact")
            else: r.fail(f"head healthcheck test list mismatch: found {test!r}, required {expected_test!r}")
            retries=healthcheck.get("retries") if isinstance(healthcheck,dict) else None
            if type(retries) is int and retries==40: r.ok("head healthcheck retries is exact integer 40")
            else: r.fail(f"head healthcheck retries must be integer 40 with bool rejected: found {retries!r} ({type(retries).__name__})")
            for key,want in (("interval","15s"),("timeout","5s"),("start_period","900s")):
                got=healthcheck.get(key) if isinstance(healthcheck,dict) else None
                if type(got) is str and got==want: r.ok(f"head healthcheck {key} is exact string {want!r}")
                else: r.fail(f"head healthcheck {key} must be exact string {want!r}: found {got!r} ({type(got).__name__})")
        for key,value in EXPECTED_ENV.items():
            literal=f"      - {key}={value}"
            count=sum(1 for line in blocks.get(role,[]) if line==literal)
            if count==1: r.ok(f"{role} has exactly one literal {key}={value}")
            else: r.fail(f"{role} VLLM_QSA_DET_TOPK/environment contract violation for {key}: count={count}")

def check_preset(r):
    if PRESET_PATH.is_file():
        got_sha = hashlib.sha256(PRESET_PATH.read_bytes()).hexdigest()
        (r.ok if got_sha == EXPECTED_PRESET_SHA else r.fail)(
            f"preset full-file SHA256 {got_sha} {'matches' if got_sha == EXPECTED_PRESET_SHA else 'does not match'} {EXPECTED_PRESET_SHA}"
        )
    else:
        r.fail("cannot hash missing preset full file")
    try:
        env=parse_env(PRESET_PATH)
        production=parse_env(PRODUCTION_PRESET_PATH)
    except ValueError as exc:
        r.fail(f"preset syntax invalid: {exc}")
        return
    if set(env)==set(production): r.ok("candidate and production preset key sets are identical")
    else: r.fail(f"candidate preset key set differs from production: added={sorted(set(env)-set(production))}, missing={sorted(set(production)-set(env))}")
    for key in sorted(set(env) & set(production) - {"VLLM_IMAGE","VLLM_EXTRA_ARGS"}):
        if env[key]==production[key]: r.ok(f"preset {key} is identical to production")
        else: r.fail(f"preset {key} differs from production: candidate={env[key]!r}, production={production[key]!r}")
    exact={"VLLM_IMAGE":EXPECTED_IMAGE_ID,"TP_SIZE":"2","DISTRIBUTED_BACKEND":"mp","MAX_MODEL_LEN":"262144","MAX_NUM_SEQS":"2","GPU_MEMORY_UTILIZATION":"0.83","MAX_NUM_BATCHED_TOKENS":"8192","VLLM_ALL2ALL_BACKEND":"allgather_reducescatter"}
    for key,want in exact.items():
        got=env.get(key); (r.ok if got==want else r.fail)(f"preset {key}={got!r}; required {want!r}")
    extra=env.get("VLLM_EXTRA_ARGS","")
    expected_extra=production.get("VLLM_EXTRA_ARGS","")+APPROVED_EXTRA_ARGS_SUFFIX
    if extra==expected_extra: r.ok("preset VLLM_EXTRA_ARGS is exact production string plus approved suffix")
    else: r.fail(f"preset VLLM_EXTRA_ARGS must be exact production string plus approved suffix: got {extra!r}, required {expected_extra!r}")
    for forbidden in ("VLLM_QSA_EXACT_TOPK","VLLM_BATCH_INVARIANT"):
        if forbidden in env or forbidden in extra: r.fail(f"preset unexpectedly enables {forbidden}")

def check_ambient_environment(r):
    try:
        preset = parse_env(PRESET_PATH)
    except ValueError as exc:
        r.fail(f"cannot validate ambient environment against invalid preset: {exc}")
        return
    initial_failures = len(r.failures)
    for key, want in preset.items():
        if key in os.environ and os.environ[key] != want:
            r.fail(f"ambient preset drift {key}={os.environ[key]!r}; required absent or exact {want!r}")
    project = os.environ.get("COMPOSE_PROJECT_NAME")
    if project is not None and project != EXPECTED_PROJECT_NAME:
        r.fail(f"ambient COMPOSE_PROJECT_NAME={project!r}; required absent or exact {EXPECTED_PROJECT_NAME!r}")
    if len(r.failures) == initial_failures:
        r.ok("ambient preset/project variables are absent or exact candidate values")

def docker_compose_available():
    if shutil.which("docker") is None: return False
    try: return subprocess.run(["docker","compose","version"],capture_output=True,text=True,timeout=10).returncode==0
    except (OSError,subprocess.TimeoutExpired): return False

def check_compose_config(r):
    if not docker_compose_available(): r.fail("docker compose unavailable; rendered verification is mandatory"); return
    try:
        preset = parse_env(PRESET_PATH)
        expected_entrypoint_source = str((REPO_ROOT / preset["ENTRYPOINT_FILE"]).resolve())
    except (ValueError, KeyError) as exc:
        r.fail(f"cannot derive rendered entrypoint contract from preset: {exc}")
        return
    poison=os.environ.copy(); poison.update({"VLLM_QSA_DET_TOPK":"0","VLLM_QSA_DET_LIB":"/tmp/poison.so","VLLM_ALL2ALL_BACKEND":"poison","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
    for role in ("head","worker"):
        cmd=["docker","compose","--env-file",str(PRESET_PATH),"-f",str(BASE_COMPOSE_PATH),"-f",str(OVERLAY_PATH),"--profile",role,"config","--format","json"]
        cp=subprocess.run(cmd,capture_output=True,text=True,env=poison,timeout=30)
        if cp.returncode: r.fail(f"{role} compose render failed: {cp.stderr.strip()}"); continue
        try: data=json.loads(cp.stdout)
        except json.JSONDecodeError as exc: r.fail(f"{role} compose JSON invalid: {exc}"); continue
        project=data.get("name")
        (r.ok if project==EXPECTED_PROJECT_NAME else r.fail)(f"{role} rendered project name={project!r}; required {EXPECTED_PROJECT_NAME!r}")
        service=data.get("services",{}).get(role)
        if not service: r.fail(f"{role} missing from rendered config"); continue
        image=service.get("image")
        (r.ok if image==EXPECTED_IMAGE_ID else r.fail)(f"{role} rendered image={image!r}; required {EXPECTED_IMAGE_ID!r}")
        entrypoints=[volume for volume in service.get("volumes",[]) if isinstance(volume,dict) and volume.get("target")=="/entrypoint.sh"]
        entrypoint_ok=(len(entrypoints)==1 and entrypoints[0].get("type")=="bind" and entrypoints[0].get("source")==expected_entrypoint_source and entrypoints[0].get("read_only") is True)
        (r.ok if entrypoint_ok else r.fail)(f"{role} rendered /entrypoint.sh bind={entrypoints!r}; required one read-only bind from {expected_entrypoint_source!r}")
        env=service.get("environment",{})
        for key,want in EXPECTED_ENV.items():
            got=env.get(key); (r.ok if got==want else r.fail)(f"{role} rendered {key}={got!r}; required {want!r} despite poison")

def check_doc(r):
    if DOC_PATH.is_file():
        got_sha = hashlib.sha256(DOC_PATH.read_bytes()).hexdigest()
        (r.ok if got_sha == EXPECTED_DOC_SHA else r.fail)(
            f"doc full-file SHA256 {got_sha} {'matches' if got_sha == EXPECTED_DOC_SHA else 'does not match'} {EXPECTED_DOC_SHA}"
        )
    else:
        r.fail("cannot hash missing doc full file")
    text=DOC_PATH.read_text() if DOC_PATH.is_file() else ""
    expected_status="**Status: PROMOTED_LOCAL_CONFIG_NOT_RELEASED — opt-in interactive c1 only; NOT the production default; no auto-start.**"
    lines=text.splitlines()
    if len(lines)>2 and lines[2]==expected_status: r.ok("doc exact status line location is line 3 immediately after heading separator")
    else: r.fail(f"doc exact status line location mismatch: line 3 is {lines[2] if len(lines)>2 else None!r}, required {expected_status!r}")
    positive_patterns=(
        r"(?<!not )(?<!non-)\bPRODUCTION-PROMOTED\b",
        r"\bproduction promotion\s*:\s*(?:PASS|APPROVED)\b",
        r"\bc2/c8\s+(?:authorized|qualified)\b",
        r"\bprefix-cache\s+ON\s+(?:qualified|authorized)\b",
        r"\bMTP1\b[^\n]{0,80}\b(?:RELEASED|PUBLISHED)\b",
    )
    contradictions=[match.group(0) for pattern in positive_patterns for match in re.finditer(pattern,text,re.I)]
    if contradictions: r.fail(f"doc contradictory positive claim(s) forbidden: {contradictions!r}")
    else: r.ok("doc has no contradictory positive promotion or scope claim")
    for needle in ("PROMOTED_LOCAL_CONFIG_NOT_RELEASED","NOT the production default","no auto-start","c1 only","MTP1 c2/c8","prefix-cache ON","MAX_NUM_SEQS=8","rollback","zero-start","qwen38-pr55122-mtp1-c1-soak4h-20260905-r1","coldstart-e3-20260906-r1",EXPECTED_IMAGE_ID,EXPECTED_KERNEL_SHA,EXPECTED_QSA_SHA,EXPECTED_UPSTREAM_KERNEL_SHA,"separately observed embedded _C_det.so SHA256","separately observed embedded QSA SHA256","c056c2d","open and unmerged","no maintainer approval","pre-run-check failure","8e685d198","NV_ERR_NO_MEMORY","Decision-integrity correction history",'Local commit was subsequently explicitly user-authorized and applied; push, release, image tag/publication, build, launch, service activation, auto-start, and production-default change remain HOLD/unauthorized.',"/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate4-20260906-r1/gate4-decision.json","19520f6abaff5fb4a0bcad29e3afd78b131a2f732e397cc67eff5d22f8f275d5","/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate4-20260906-r1/gate4-independent-review.json","b126037c2be0300c3fb62b2d6b3b5dca5fbfc6a1f141bdb3dce3274b8007f928","/home/bjk110/docker-build/qwen38-pr55122-mtp1-c1-gate5-20260906-r1/gate5-independent-review.json","05feba33f0f68f0dd3802c442be56ed04986623640889ba949a286cfe2355680","LOCAL NON-PRODUCTION experimental deterministic persistent_topk evaluation; not promoted or pushed","preserved as build-time provenance","does not mean the image was rebuilt or retagged","does not reproduce the image ID"):
        present=needle.lower() in text.lower()
        state="contains" if present else "missing"
        (r.ok if present else r.fail)(f"doc {state} {needle!r}")


def check_promotion_indexes(r):
    expected_names = set(PROMOTION_INDEX_REQUIRED_LINES)
    if set(PROMOTION_INDEX_PATHS) != expected_names:
        r.fail(f"promotion index path set mismatch: found={sorted(PROMOTION_INDEX_PATHS)}, required={sorted(expected_names)}")
        return
    for name in sorted(expected_names):
        path = PROMOTION_INDEX_PATHS[name]
        if not path.is_file():
            r.fail(f"promotion index {name} missing: {path}")
            continue
        got_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        want_sha = PROMOTION_INDEX_EXPECTED_SHAS[name]
        (r.ok if got_sha == want_sha else r.fail)(
            f"promotion index {name} full-file SHA256 {got_sha} {'matches' if got_sha == want_sha else 'does not match'} {want_sha}"
        )
        lines = path.read_text().splitlines()
        for required in PROMOTION_INDEX_REQUIRED_LINES[name]:
            count = lines.count(required)
            if count == 1:
                r.ok(f"promotion index {name} contains required entry exactly once")
            else:
                r.fail(f"promotion index {name} required entry must appear exactly once; count={count}: {required!r}")
        expected_promoted = {line for line in PROMOTION_INDEX_REQUIRED_LINES[name] if "PROMOTED_LOCAL_CONFIG_NOT_RELEASED" in line}
        actual_promoted = {line for line in lines if "PROMOTED_LOCAL_CONFIG_NOT_RELEASED" in line}
        if actual_promoted == expected_promoted:
            r.ok(f"promotion index {name} has no unexpected promoted-status entry")
        else:
            r.fail(f"promotion index {name} promoted-status scope creep: unexpected={sorted(actual_promoted-expected_promoted)}, missing={sorted(expected_promoted-actual_promoted)}")
        text = "\n".join(lines)
        scope_patterns = (
            r"\bMTP1\b[^\n]{0,120}\b(?:c2/c8|c2|c8)\b(?:(?!\b(?:excluded|not)\b)[^\n]){0,80}\b(?:authorized|qualified)\b",
            r"\bprefix-cache\s+ON\b(?:(?!\b(?:excluded|not)\b)[^\n]){0,80}\b(?:authorized|qualified)\b",
            r"\bMAX_NUM_SEQS=8\b(?:(?!\b(?:excluded|not)\b)[^\n]){0,80}\b(?:authorized|qualified)\b",
            r"\b(?:PR55122|MTP1)\b[^\n]{0,120}\b(?:is|was|are|has been)\s+(?:RELEASED|PUBLISHED)\b",
        )
        bad = [m.group(0) for pattern in scope_patterns for m in re.finditer(pattern, text, re.I)]
        if bad:
            r.fail(f"promotion index {name} scope creep or false release claim: {bad!r}")
        else:
            r.ok(f"promotion index {name} has no forbidden scope/release claim")

def check_local_image_identity(r):
    if os.environ.get(LOCAL_IMAGE_GATE)!="1":
        r.warn(f"image ID not checked; set {LOCAL_IMAGE_GATE}=1 and optionally {IMAGE_HOSTS_ENV}=spark01,spark02")
        return
    if IMAGE_HOSTS_ENV not in os.environ:
        targets=[None]
    else:
        hosts_raw=os.environ[IMAGE_HOSTS_ENV]
        if hosts_raw != "spark01,spark02":
            r.fail(f"forced remote hosts must be exact ordered spark01,spark02; got {hosts_raw!r}")
            return
        targets=["spark01", "spark02"]
    for host in targets:
        inspect=["docker","image","inspect",EXPECTED_IMAGE_ID]
        cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8",host,*inspect] if host else inspect
        cp=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        label=host or "local host"
        try:
            data=json.loads(cp.stdout) if cp.returncode==0 else {}
            if isinstance(data,list): data=data[0] if len(data)==1 else {}
        except json.JSONDecodeError:
            data={}
        got=data.get("Id")
        labels=data.get("Config",{}).get("Labels")
        provenance={key:value for key,value in labels.items() if key.startswith("org.vllm-spark.candidate.")} if isinstance(labels,dict) else None
        if got==EXPECTED_IMAGE_ID: r.ok(f"{label} forced image ID matches {EXPECTED_IMAGE_ID}")
        else: r.fail(f"{label} forced image ID mismatch: got {got!r}, required {EXPECTED_IMAGE_ID!r}; stderr={cp.stderr.strip()!r}")
        if provenance==EXPECTED_LABELS: r.ok(f"{label} forced provenance labels match exact expected set")
        else: r.fail(f"{label} forced provenance labels mismatch: got {provenance!r}, required {EXPECTED_LABELS!r}")

def main():
    r=Results()
    for check in (check_files,check_hashes,check_dockerfile,check_overlay_structure,check_preset,check_ambient_environment,check_compose_config,check_doc,check_promotion_indexes,check_local_image_identity): check(r)
    print(f"Summary: {len(r.passes)} pass, {len(r.warnings)} warn, {len(r.failures)} fail")
    if r.failures: print("Local-promoted recipe verification FAILED"); return 1
    print("Local-promoted recipe verification PASSED (not released; zero-start static closure)"); return 0
if __name__=="__main__": sys.exit(main())
