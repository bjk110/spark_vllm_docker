"""Public wording policy; immutable artifact spellings remain valid and pinned."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESS_DOCS = {
    'CHANGELOG.md',
    'benchmarks/evidence/retained/solar-open2-v0251-b12xsw-r4-20260726T053457Z/report.md',
    'benchmarks/evidence/retained/solar-open2-v0251-multi-instance-lowconc-20260727T004456Z/09_final/report.md',
    'benchmarks/evidence/retained/solar-open2-v0251-pread-complete-20260725T225149Z/report.md',
    'benchmarks/evidence/retained/solar-open2-v0251-pread-r3-build-20260726T002411Z/report.md',
    'benchmarks/evidence/retained/solar-open2-v0251-promotion-readiness-20260727T065646Z/01_provenance/provenance.md',
    'benchmarks/evidence/retained/solar-open2-v0251-promotion-readiness-20260727T065646Z/09_final/report.md',
    'benchmarks/evidence/retained/solar-open2-v0251-r4-acceptance-20260726T121058Z/report.md',
    'benchmarks/llama-benchy/results_dsv4-b3i-sparse-mla-prefill-vs-marlin-64k-c1.md',
    'dockerfiles/active/solar-open2/r3/PROVENANCE.md',
    'dockerfiles/active/solar-open2/r4/PROVENANCE.md',
    'docs/benchmarks/bt-matrix-step37-nvfp4-v023.md',
    'docs/benchmarks/step37-v022-long-context-validation.md',
    'docs/deepseek-v4-external-runtime-qualification-2026-08-28.md',
    'docs/deepseek-v4-sm121-rowwise-mqa-cudagraph-fix.md',
    'docs/deepseek-v4-sm121-rowwise-mqa-patch-playbook.md',
    'docs/deepseek-v4-v027-b43s-promotion-candidate.md',
    'docs/diagnostics/dgx-spark-uma-memory-freeze.md',
    'docs/dsv4-dspark-speculative-decoding-72261a7-closure.md',
    'docs/dsv4-sparse-mla-b3-investigation-closure.md',
    'docs/qwen3.8-flash-next-pr55122-mtp1-c1-candidate.md',
    'docs/unholy-fusion-benchmark.md',
}
OVERLAYS = {
    'compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml',
    'compose/qwen3.8-flash-next/docker-compose.candidate.yml',
    'compose/solar-open2/docker-compose.b12x-cache.yml',
    'compose/solar-open2/docker-compose.r4-production.yml',
}
# Only literal file paths, image identities, evidence IDs and state IDs are exempt.
ARTIFACT = re.compile(
    r'[\w@./:-]+\.(?:md|env|ya?ml)(?:#[\w-]+)?'
    r'|[\w.-]+/[\w./-]+:[\w.:-]+'
    r'|[A-Z][A-Z0-9_]*_(?:PRODUCTION|PROMOTED|PROMOTION)_[A-Z0-9_]+'
    r'|[\w-]*(?:production|promotion)[\w-]*-\d[\w-]*'
    r'|(?<![#\w-])[\w-]*\d[\w-]*-(?:production|promotion)[\w-]*'
)
MATURITY = re.compile(r'(?<![A-Za-z0-9])(?:production|promot[A-Za-z0-9_]*)(?![A-Za-z0-9])', re.I)
PINS = json.loads(Path(__file__).with_name('documentation_policy_pins.json').read_text())


def prose(path, text):
    """Remove only the exact, count-pinned identifiers approved for this path."""
    for identifier in PINS.get(path, {}).get('identifiers', []):
        assert identifier in text, (path, identifier)
        text = text.replace(identifier, '', 1)
    return text


def surfaces():
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    return sorted(p for p in tracked if (
        (p.endswith('.md') and p not in PROCESS_DOCS)
        or p == '.env.example'
        or p == 'docker-compose.yml'
        or (p.startswith('presets/') and p.endswith('.env'))
        or (p.startswith('compose/') and p.endswith(('.yml', '.yaml')))
        or (p.startswith('dockerfiles/active/') and Path(p).name.startswith('Dockerfile'))
        or (p.startswith('scripts/prewarm') and p.endswith('.py'))
        or p == 'scripts/diag/verify_qwen38_flash_next_recipe.py'
    ))


@pytest.mark.parametrize('path', surfaces())
def test_public_surfaces_have_neutral_prose(path):
    text = (ROOT / path).read_text()
    if path.endswith(('.env', '.yml', '.yaml')):
        text = '\n'.join(line for line in text.splitlines() if line.lstrip().startswith('#'))
    assert not MATURITY.search(prose(path, text)), path


@pytest.mark.parametrize('text', [
    'Production-qualified', 'promoted/current production.', 'promotion/readiness',
    'promoting this profile', '`production`', 'not production', 'PRODUCTION.',
    'PRODUCTION_QUALIFIED', 'B4_4C_FORMALLY_PROMOTED_AND_PUBLISHED',
])
def test_artifact_exception_cannot_hide_maturity_prose(text):
    assert MATURITY.search(prose('__synthetic__', text))


def test_unpinned_identifier_shaped_text_is_not_exempt():
    assert MATURITY.search(prose('__synthetic__', 'fake-production-2026'))


def test_policy_files_and_scanned_benchmark_readmes_trigger_ci():
    workflow = (ROOT / '.github/workflows/preset-integrity.yml').read_text()
    for path in (
        'scripts/diag/test_documentation_policy.py',
        'scripts/diag/documentation_policy_pins.json',
        'benchmarks/README.md',
        'benchmarks/llama-benchy/README.md',
        '.env.example',
        'scripts/prewarm*.py',
        '**/*.md',
        '*.md',
    ):
        assert workflow.count(f"      - '{path}'") == 2, path


def test_ngc2609_qwen_recipe_tests_and_verifier_run_in_ci():
    workflow = (ROOT / '.github/workflows/preset-integrity.yml').read_text()
    assert 'scripts/diag/test_verify_qwen38_ngc2609_mtp_k2_c2_recipe.py' in workflow
    assert 'scripts/diag/test_qwen38_ray_recipe.py' in workflow
    assert 'python scripts/diag/verify_qwen38_ngc2609_mtp_k2_c2_recipe.py' in workflow


def test_every_tracked_markdown_is_scanned_or_explicit_process_record():
    tracked = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '*.md', '**/*.md'], text=True
    ).splitlines())
    scanned = {path for path in surfaces() if path.endswith('.md')}
    assert PROCESS_DOCS <= tracked
    assert scanned | PROCESS_DOCS == tracked


def test_compose_overlay_runtime_bytes_are_pinned():
    for path in OVERLAYS:
        assert PINS.get(path, {}).get('runtime_sha256'), path


def test_every_tracked_preset_has_a_runtime_pin():
    tracked = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', 'presets/*.env'], text=True
    ).splitlines())
    pinned = {path for path, data in PINS.items()
              if path.startswith('presets/') and path.endswith('.env')
              and data.get('runtime_sha256')}
    assert pinned == tracked


def test_deepseek_rollback_headers_match_the_catalog_hierarchy():
    primary = (ROOT / 'presets/deepseek-v4-flash-dspark-k7-64k-production-tp2.env').read_text()
    legacy = (ROOT / 'presets/deepseek-v4-flash-mtp1-production-tp2.env').read_text()
    assert '# Status: Primary rollback (stopped)' in primary
    assert '# Status: Legacy rollback (stopped)' in legacy
    assert 'second-tier' in primary.lower()
    assert 'second-tier' in legacy.lower()
    assert not re.search(r'\b(?:primary|authoritative) rollback\b', legacy, re.I)


def test_root_readme_uses_a_generic_model_serving_workflow():
    root_readme = (ROOT / 'README.md').read_text()
    preset_readme = (ROOT / 'presets/README.md').read_text()
    for heading in (
        '### 1. Select a preset',
        '### 2. Prepare and render',
        '### 3. Start by topology and backend',
        '### 4. Verify readiness',
        '### 5. Stop, switch, or roll back',
    ):
        assert heading in root_readme
    for contract in (
        'Required overlay',
        '`DISTRIBUTED_BACKEND=ray`',
        '`DISTRIBUTED_BACKEND=mp`',
        '`/health`',
        '`/v1/models`',
        '`OOMKilled=false`',
        '`RestartCount=0`',
    ):
        assert contract in root_readme
    assert root_readme.lower().count('deepseek-v4') <= 4
    assert 'For DeepSeek-V4 and Solar-Open2' in preset_readme
    assert 'There is no other operational choice for a validated model family' not in preset_readme


def test_readme_qwen_uncensored_manifest_matches_preset():
    readme = (ROOT / 'README.md').read_text()
    assert 'presets/qwen3.8-flash-next-uncensored-fp8-ngc2609-mtp-k2-c2.env' in readme
    assert 'compose/qwen3.8-flash-next/docker-compose.ngc2609-mtp-k2-c2.yml' in readme


def test_root_readme_commands_preserve_the_rendered_environment_contract():
    readme = (ROOT / 'README.md').read_text()
    quick_start = readme.split('## Quick Start', 1)[1].split('## Repository layout', 1)[0]
    clean = 'env -i PATH="$PATH" HOME="$HOME" DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"'
    commands = [
        re.sub(r'\s+', ' ', line.strip())
        for line in quick_start.replace('\\\n', ' ').splitlines()
        if 'docker compose --env-file "$PRESET"' in line
    ]
    assert len(commands) == 8
    for command in commands:
        assert command.startswith(f'{clean} docker compose --env-file "$PRESET"')
        assert '-f docker-compose.yml' in command
        assert '-f "$OVERLAY"' in command
    assert 'PRESET=presets/<' not in quick_start
    assert 'OVERLAY=compose/<' not in quick_start
    assert 'config --quiet' not in quick_start
    assert "--profile '*' config" in quick_start

    single = quick_start.split('#### Single Spark', 1)[1].split(
        '#### Dual Spark with `DISTRIBUTED_BACKEND=ray`', 1
    )[0]
    ray = quick_start.split('#### Dual Spark with `DISTRIBUTED_BACKEND=ray`', 1)[1].split(
        '#### Dual Spark with `DISTRIBUTED_BACKEND=mp`', 1
    )[0]
    mp = quick_start.split('#### Dual Spark with `DISTRIBUTED_BACKEND=mp`', 1)[1].split(
        '### 4. Verify readiness', 1
    )[0]
    stop = quick_start.split('### 5. Stop, switch, or roll back', 1)[1]
    profile_pattern = r'--profile (head|worker) (?:up -d|down)'
    assert re.findall(profile_pattern, single) == ['head']
    assert re.findall(profile_pattern, ray) == ['head', 'worker']
    assert re.findall(profile_pattern, mp) == ['worker', 'head']
    assert re.findall(profile_pattern, stop) == ['head', 'worker']


def test_stack_references_identify_v027_as_current_deepseek_route():
    architecture = (ROOT / 'docs/architecture.md').read_text()
    software_stack = (ROOT / 'docs/software-stack.md').read_text()
    images = (ROOT / 'docs/images.md').read_text()
    dockerfiles = (ROOT / 'dockerfiles/README.md').read_text()
    troubleshooting = (ROOT / 'docs/troubleshooting.md').read_text()
    stack_v022 = (ROOT / 'docs/stack-v022.md').read_text()
    assert 'vLLM 0.27 native-DSpark route' in architecture
    assert '`dsv4-d568`: frozen historical DeepSeek-V4-Flash baseline' in software_stack
    assert '`dsv4-d568` (historical, DSV4-specific)' in images
    assert '`dsv4-d568` | Frozen historical/reproduction DeepSeek-V4-Flash image path.' in dockerfiles
    assert '## Historical dsv4-d568 issues' in troubleshooting
    assert '../README.md#4-verify-readiness' in troubleshooting
    assert '../README.md#3-verify' not in troubleshooting
    assert '`v022-d568`, is the stable general base' in stack_v022
    assert '../README.md#representative-serving-paths' in stack_v022
    assert 'current main' not in stack_v022
    assert 'dsv4-d568` as the primary path' not in software_stack


def test_literal_artifacts_and_runtime_bytes_remain_pinned():
    for path, expected in PINS.items():
        raw = (ROOT / path).read_bytes()
        identifiers = [m[0] for m in ARTIFACT.finditer(raw.decode()) if MATURITY.search(m[0])]
        assert identifiers == expected['identifiers'], path
        if 'runtime_sha256' in expected:
            runtime = b''.join(line for line in raw.splitlines(keepends=True)
                               if line.strip() and not line.lstrip().startswith(b'#'))
            assert hashlib.sha256(runtime).hexdigest() == expected['runtime_sha256'], path
