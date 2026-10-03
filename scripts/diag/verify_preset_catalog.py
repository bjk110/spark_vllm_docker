#!/usr/bin/env python3
"""Read-only integrity/preflight check of Git-indexed presets; never starts Docker."""
import argparse
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

LABELS = ('Status', 'Validated date', 'Topology', 'Image identity', 'Required overlay', 'Known limits')

def tracked_files(root):
    return set(subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], check=True,
                              capture_output=True, text=True).stdout.split('\0')) - {''}

def parse_env(text):
    return dict(re.findall(r'^([A-Z][A-Z0-9_]*)=(.*)$', text, re.M))

def anchors(text):
    result = set()
    counts = {}
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f'-{count}' if count else ''))
    result.update(re.findall(r'(?:id|name)=["\']([^"\']+)', text))
    return result

def verify(root, skip_compose=False, expected_count=42):
    root = Path(root).resolve()
    tracked = tracked_files(root)
    presets = sorted(p for p in tracked if p.startswith('presets/') and p.endswith('.env'))
    errors = []
    def fail(message): errors.append(message)
    if len(presets) != expected_count: fail(f'preset count: {len(presets)} != {expected_count}')
    readme = root / 'presets/README.md'
    text = readme.read_text()
    rows = {}
    section = ''
    launch_column = False
    for line in text.splitlines():
        if line.startswith('## '): section = line; launch_column = False
        if line.startswith('|') and 'Preset' in line and 'Launch / docs' in line: launch_column = True
        if not line.startswith('|'): continue
        match = re.search(r'\]\(([^)]+\.env)\)', line)
        if match and not launch_column: fail(f'{match[1]}: missing Launch/docs column')
        if match: rows.setdefault('presets/' + match[1], []).append((section, line))
    for path in rows:
        if path not in presets: fail(f'catalog row has no tracked preset: {path}')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if re.match(r'\w+://|mailto:', target): continue
        filename, _, anchor = unquote(target).partition('#')
        dest = (readme.parent / filename).resolve() if filename else readme
        try: rel = dest.relative_to(root).as_posix()
        except ValueError: fail(f'link outside repository: {target}'); continue
        if rel not in tracked or not dest.is_file(): fail(f'unresolved tracked README link: {target}')
        elif anchor and anchor not in anchors(dest.read_text()): fail(f'unresolved README anchor: {target}')
    available = False
    if not skip_compose:
        try:
            available = subprocess.run(['docker', 'compose', 'version'], capture_output=True,
                                       timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired): pass
        if not available: fail('docker compose unavailable; explicitly use --skip-compose for static checks')
    rendered = 0
    for path in presets:
        source = (root / path).read_text()
        env = parse_env(source)
        metadata = {}
        for label in LABELS:
            values = re.findall(r'^#\s*' + re.escape(label) + r':\s*(.*)$', source, re.M)
            if len(values) != 1: fail(f'{path}: {label} must occur exactly once')
            metadata[label] = values[0] if values else ''
        entries = rows.get(path, [])
        if len(entries) != 1: fail(f'{path}: exactly one catalog row required'); continue
        section, row = entries[0]
        status = metadata['Status'].lower()
        section_number = re.search(r'## (\d)\.', section)
        number = section_number.group(1) if section_number else ''
        if number not in ('1', '2', '3', '4', '5'): fail(f'{path}: status in unrecognized catalog section')
        expected = {'2': 'validated', '3': 'general supported', '4': 'experimental', '5': 'historical'}
        if number in expected and not status.startswith(expected[number]): fail(f'{path}: status disagrees with catalog section')
        if number == '1' and not status.startswith(('active production', 'optional validated profile', 'primary rollback', 'legacy rollback', 'production rollback', 'production-qualified', 'main_merge_approved_opt_in')):
            fail(f'{path}: status not operationally qualified for production catalog section')
        if number == '1' and status.split(' (')[0] not in re.sub(r'[*`]', '', row).lower(): fail(f'{path}: status disagrees with catalog row')
        mode, tp = env.get('CLUSTER_MODE'), env.get('TP_SIZE')
        topology = metadata['Topology']
        if mode not in ('single', 'dual-rdma') or tp != {'single':'1', 'dual-rdma':'2'}.get(mode) or not re.search(r'\b' + str(mode) + r'\b', topology) or f'TP={tp}' not in topology:
            fail(f'{path}: topology differs from CLUSTER_MODE/TP_SIZE')
        row_topologies = re.findall(r'\b(single|dual-rdma)\s+TP=?([12])\b', re.sub(r'[`*]', '', row))
        if row_topologies and any(pair != (mode, tp) for pair in row_topologies): fail(f'{path}: catalog topology differs from preset')
        backend = env.get('DISTRIBUTED_BACKEND', 'ray')
        if backend not in ('ray', 'mp') or f'backend={backend}' not in topology: fail(f'{path}: backend missing or differs from preset/Compose default')
        identity = re.search(r'VLLM_IMAGE=([^;\s]+)', metadata['Image identity'])
        if not env.get('VLLM_IMAGE') or not identity or identity.group(1) != env['VLLM_IMAGE']: fail(f'{path}: image identity differs from VLLM_IMAGE')
        overlays = list(dict.fromkeys(file for file in re.findall(r'[\w./-]+\.ya?ml', metadata['Required overlay']) if file != 'docker-compose.yml'))
        dependencies = ['docker-compose.yml', *overlays]
        if env.get('ENTRYPOINT_FILE'): dependencies.append(env['ENTRYPOINT_FILE'].removeprefix('./'))
        for dependency in dependencies:
            if dependency not in tracked or not (root / dependency).is_file(): fail(f'{path}: required tracked path missing: {dependency}')
        generic = re.findall(r'Generic TP([12])', row)
        if not re.search(r'\]\([^)]+\)', row.split('|')[-2]): fail(f'{path}: missing Launch/docs')
        if generic and (number not in ('2', '3') or overlays or env.get('ENTRYPOINT_FILE', './entrypoints/entrypoint.sh') != './entrypoints/entrypoint.sh' or 'tokenizer overlay' in metadata['Required overlay'].lower()): fail(f'{path}: generic launch forbidden for production/overlay/evidence row')
        for label_size, target_size in re.findall(r'\[Generic TP([12])\]\(#generic-tp([12])-launch\)', row):
            if label_size != target_size: fail(f'{path}: generic launch link target disagrees with label')
        if generic and any(size != tp for size in generic): fail(f'{path}: generic launch topology mismatch')
        if number in ('4', '5') and 'validation/reproduction only' not in row: fail(f'{path}: experimental/historical row must be evidence-only')
        for key, value in env.items():
            if re.search(r'(?:^|_)(?:TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY)(?:_|$)', key) and value and not value.startswith(('$', '<', '[', 'placeholder', '\"\"', "''")):
                fail(f'{path}: obvious credential assignment ({key})')
        if available:
            # Whitelist process necessities; do not inherit preset/Compose overrides or credentials.
            render_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'DOCKER_CONFIG') if key in os.environ}
            render_env.update(env)
            render_env.update(MODEL_PATH='/tmp/preset-integrity-model', HEAD_ROCE_IP='192.0.2.1',
                              WORKER_ROCE_IP='192.0.2.2', ROCE_IF_NAME='eth0', IB_HCA_NAME='mlx5_0',
                              B12X_CACHE_DIR='/tmp/preset-integrity-cache', COMPOSE_PROJECT_NAME='preset-integrity')
            cmd = ['docker', 'compose', '--env-file', path]
            for file in ['docker-compose.yml', *overlays]: cmd.extend(['-f', file])
            cmd.extend(['--profile', 'head', '--profile', 'worker', 'config', '--quiet'])
            try:
                cp = subprocess.run(cmd, cwd=root, env=render_env, capture_output=True, text=True, timeout=30)
                if cp.returncode: fail(f'{path}: compose render failed: {cp.stderr.strip()}')
                else: rendered += 1
            except (OSError, subprocess.TimeoutExpired) as exc: fail(f'{path}: compose render failed: {exc}')
    return errors, rendered

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-compose', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors, rendered = verify(args.root, args.skip_compose)
    for error in errors: print(f'[FAIL] {error}')
    print(f'{"FAIL" if errors else "PASS"}: 42 tracked presets; compose rendered {rendered}/42' + (' (explicitly skipped)' if args.skip_compose else ''))
    return bool(errors)

if __name__ == '__main__': raise SystemExit(main())
