"""Catalog contracts tested in isolated Git repositories."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('catalog', Path(__file__).with_name('verify_preset_catalog.py'))
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)

@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'presets').mkdir()
    (tmp_path / 'presets/a.env').write_text('''# Status: General supported
# Validated date: Not recorded
# Topology: single; TP=1; backend=ray
# Image identity: VLLM_IMAGE=example:tag
# Required overlay: None documented; use docker-compose.yml
# Known limits: Not recorded
VLLM_IMAGE=example:tag
CLUSTER_MODE=single
TP_SIZE=1
DISTRIBUTED_BACKEND=ray
''')
    (tmp_path / 'docker-compose.yml').write_text('services: {}\n')
    (tmp_path / 'presets/README.md').write_text('''# Catalog
## 3. General supported presets
| Preset | Topology | Launch / docs |
|---|---|---|
| [a.env](a.env) | single TP1 | [Generic TP1](#generic-tp1-launch) |
### Generic TP1 launch
''')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    return tmp_path

def check(repo):
    return catalog.verify(repo, skip_compose=True, expected_count=1)[0]

def test_pass_and_ignore_untracked(repo):
    (repo / 'presets/untracked.env').write_text('bad')
    assert check(repo) == []

@pytest.mark.parametrize('old,new,needle', [
    ('# Status: General supported', '# Status: Experimental', 'status'),
    ('# Known limits: Not recorded', '# Known limits: Not recorded\n# Known limits: duplicate', 'Known limits'),
    ('TP_SIZE=1', 'TP_SIZE=2', 'topology'),
    ('# Image identity: VLLM_IMAGE=example:tag', '# Image identity: wrong', 'image'),
    ('# Required overlay: None documented; use docker-compose.yml', '# Required overlay: compose/missing.yml', 'tracked path'),
    ('DISTRIBUTED_BACKEND=ray', 'DISTRIBUTED_BACKEND=mp', 'backend'),
    ('TP_SIZE=1', 'TP_SIZE=1\nAPI_TOKEN=secret123', 'credential'),
    ('TP_SIZE=1', 'TP_SIZE=1\nENTRYPOINT_FILE=./missing.sh', 'tracked path'),
])
def test_preset_mutations(repo, old, new, needle):
    path = repo / 'presets/a.env'
    path.write_text(path.read_text().replace(old, new))
    assert any(needle in error for error in check(repo))

@pytest.mark.parametrize('mutation,needle', [
    ('duplicate', 'catalog row'), ('missing', 'catalog row'),
    ('anchor', 'anchor'), ('generic', 'generic'), ('link', 'link'),
])
def test_catalog_mutations(repo, mutation, needle):
    path = repo / 'presets/README.md'
    text = path.read_text()
    row = next(line for line in text.splitlines() if '[a.env]' in line)
    if mutation == 'duplicate': text += row + '\n'
    if mutation == 'missing': text = text.replace(row, '')
    if mutation == 'anchor': text = text.replace('](#generic-tp1-launch)', '](#missing)')
    if mutation == 'generic': text = text.replace('Generic TP1](#generic-tp1-launch)', 'Generic TP2](#generic-tp1-launch)')
    if mutation == 'link': text += '\n[broken](missing.md)\n'
    path.write_text(text)
    assert any(needle in error for error in check(repo))

def test_compose_render_is_config_only_and_isolated(repo, monkeypatch):
    calls = []
    real_run = catalog.subprocess.run
    def run(args, **kwargs):
        if args[:2] == ['docker', 'compose']:
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, '{}', '')
        return real_run(args, **kwargs)
    monkeypatch.setattr(catalog.subprocess, 'run', run)
    monkeypatch.setenv('VLLM_IMAGE', 'poison')
    errors, rendered = catalog.verify(repo, expected_count=1)
    assert errors == []
    assert rendered == 1
    args, kwargs = calls[-1]
    assert args[-2:] == ['config', '--quiet']
    assert kwargs['env']['VLLM_IMAGE'] == 'example:tag'
    assert kwargs['env']['MODEL_PATH'].startswith('/tmp/')

def test_token_counts_are_not_credentials(repo):
    path = repo / 'presets/a.env'
    path.write_text(path.read_text() + 'MAX_NUM_BATCHED_TOKENS=8192\n')
    assert check(repo) == []

def test_validated_general_configuration_can_use_generic_launch(repo):
    path = repo / 'presets/a.env'
    path.write_text(path.read_text().replace('Status: General supported', 'Status: Validated (non-production)'))
    index = repo / 'presets/README.md'
    index.write_text(index.read_text().replace('3. General supported presets', '2. Validated presets (non-production)'))
    assert check(repo) == []

@pytest.mark.parametrize('old,new,needle', [
    ('| single TP1 |', '| dual-rdma TP2 |', 'catalog topology'),
    ('Launch / docs', 'Other column', 'Launch/docs column'),
])
def test_row_contract_drift(repo, old, new, needle):
    path = repo / 'presets/README.md'
    path.write_text(path.read_text().replace(old, new))
    assert any(needle in error for error in check(repo))

def test_image_identity_requires_exact_value(repo):
    path = repo / 'presets/a.env'
    path.write_text(path.read_text().replace('# Image identity: VLLM_IMAGE=example:tag', '# Image identity: VLLM_IMAGE=example:tagwrong'))
    assert any('image' in error for error in check(repo))

def test_model_specific_entrypoint_cannot_use_generic_launch(repo):
    entrypoint = repo / 'wrapper.sh'
    entrypoint.write_text('#!/bin/sh\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'wrapper.sh'], check=True)
    path = repo / 'presets/a.env'
    path.write_text(path.read_text() + 'ENTRYPOINT_FILE=./wrapper.sh\n')
    assert any('generic' in error for error in check(repo))

def test_production_overlay_is_declared_and_rendered(repo, monkeypatch):
    (repo / 'compose').mkdir()
    (repo / 'compose/a.yml').write_text('services: {}\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'compose/a.yml'], check=True)
    path = repo / 'presets/a.env'
    path.write_text(path.read_text().replace('General supported', 'Active production').replace('None documented; use docker-compose.yml', 'compose/a.yml (with docker-compose.yml)'))
    (repo / 'docs.md').write_text('# Runbook\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'docs.md'], check=True)
    index = repo / 'presets/README.md'
    index.write_text(index.read_text().replace('3. General supported presets', '1. Production presets').replace('[Generic TP1](#generic-tp1-launch)', 'Active production [Runbook](../docs.md)'))
    calls = []
    real_run = catalog.subprocess.run
    def run(args, **kwargs):
        if args[:2] == ['docker', 'compose']:
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, '', '')
        return real_run(args, **kwargs)
    monkeypatch.setattr(catalog.subprocess, 'run', run)
    errors, rendered = catalog.verify(repo, expected_count=1)
    assert errors == []
    assert rendered == 1
    assert 'compose/a.yml' in calls[-1]

def test_unknown_catalog_section_is_rejected(repo):
    path = repo / 'presets/README.md'
    path.write_text(path.read_text().replace('3. General supported presets', 'Unclassified'))
    assert any('status' in error for error in check(repo))

def test_generic_link_target_matches_label(repo):
    path = repo / 'presets/README.md'
    path.write_text(path.read_text().replace('](#generic-tp1-launch)', '](#generic-tp2-launch)') + '\n### Generic TP2 launch\n')
    assert any('generic' in error for error in check(repo))

def test_production_section_cannot_bless_experimental_status(repo):
    path = repo / 'presets/a.env'
    path.write_text(path.read_text().replace('Status: General supported', 'Status: Experimental'))
    index = repo / 'presets/README.md'
    index.write_text(index.read_text().replace('3. General supported presets', '1. Production presets').replace('| single TP1 |', '| Experimental single TP1 |'))
    assert any('status' in error for error in check(repo))
