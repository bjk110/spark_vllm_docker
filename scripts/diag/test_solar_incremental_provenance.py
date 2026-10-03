"""Byte identity of the available Solar incremental build evidence (no imports/builds)."""
import hashlib
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('path,digest', [
 ('dockerfiles/active/solar-open2/r3/Dockerfile', '4cebb9c2effb5565e1d4de478e630c28f337596b221634532837761bf92de627'),
 ('dockerfiles/active/solar-open2/r3/solar_open2-rawg1.py', '56b487479c06a4c23d19276c624cde2d462da0ac07e90e394bed580984761645'),
 ('dockerfiles/active/solar-open2/r3/weight_utils-pread-gate.py', 'de48cbefb218b5a0c7160df0253c116bf5b84cf4bf4cd90aecdd4396bb5de7aa'),
 ('dockerfiles/active/solar-open2/r3/solar-open2-rawg1-contract-v025.patch', 'f252383c949de06cf14d2fadf6ff6c0a5ae76fa7408e7beb5b30c8db823dd6ca'),
 ('dockerfiles/active/solar-open2/r3/vllm-safetensors-pread-env-gate.patch', '39b26822f2bfb85dba1fb75e8e90a62ee2aabb32014d4b26ce963b6e2b1371ea'),
 ('dockerfiles/active/solar-open2/r4/Dockerfile', 'c2df5e9dcd5409cf26ec2f48d7fa2fd367a4dd4d3665b3f3e155b38076755b0b'),
 ('dockerfiles/active/solar-open2/r4/b12x_moe_r4.py', '38c253a37c131b38713b92cea038bb6ff07f7b38a1c91fdd68f38567cb8180bc'),
 ('dockerfiles/active/solar-open2/r4/vllm-flashinfer-b12x-shared-workspace-env-gate.patch', '2b31f3a873e7a29c991cefc39599b01b10f41f4dbdef096fc55f23c175382c83'),
])
def test_incremental_build_bytes(path, digest):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
