"""Restore a bounded set of historical analysis inputs without changing checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path('/data/coding/robot-whisper-0909')
OUT = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')
COMMIT = '5da830a37230f4ecd761f70be4999f4b3a01687e'
RAW = 'moe-v7-legacy16x32-0906/results/raw_features'
ENCODED = 'safe&vlaconf/moe_trainfree/results/routing_dynamics_20260908'
FILES = [
    *[f'{RAW}/{cohort}_{suffix}'
      for cohort in ('development_main', 'external_8b', 'legacy_16x32')
      for suffix in ('route_features.npz', 'extraction_audit.json')],
    'moe-v7-legacy16x32-0906/results/legacy16x32/episode_alarms.csv',
    'moe-v7-legacy16x32-0906/results/manifest.json',
    'moe-v7-legacy16x32-0906/experiments/raw_route_features.py',
    f'{ENCODED}/index.csv', f'{ENCODED}/s05_features.csv',
    f'{ENCODED}/plot_task_map.csv', f'{ENCODED}/input_verification.json',
    'analysis_moe_execution_signals/rich_event_functional_32d/records.jsonl',
    'analysis_moe_execution_signals/rich_event_functional_32d/summary.json',
    'analysis_moe_execution_signals/rich_event_analysis_summary.json',
    'analysis_moe_execution_signals/rich_event_analysis_report.md',
    'moe-trap-control/design/frozen_alarm_comparison_20260908/first_alarms.csv',
    'moe-trap-control/design/frozen_alarm_comparison_20260908/contract.json',
    'moe-trap-control/design/frozen_alarm_comparison_20260908/profiles/parameters.json',
]


def restore():
    records = []
    for source in FILES:
        item = subprocess.check_output(
            ['git', 'ls-tree', COMMIT, '--', source], cwd=REPO, text=True).strip()
        expected = item.split()[2]
        target = OUT / 'inputs' / source
        if target.exists():
            content = target.read_bytes()
        else:
            content = subprocess.check_output(['git', 'show', f'{COMMIT}:{source}'], cwd=REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if actual != expected:
            raise ValueError(f'Git object mismatch: {source}')
        records.append(dict(source=source, git_blob=actual, bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest()))
        print(f'verified {source} ({len(content)} bytes)', flush=True)
    for cohort in ('development_main', 'external_8b', 'legacy_16x32'):
        audit = json.loads((OUT / 'inputs' / RAW / f'{cohort}_extraction_audit.json').read_text())
        data = next(row for row in records if row['source'] == f'{RAW}/{cohort}_route_features.npz')
        if data['sha256'] != audit['output_sha256']:
            raise ValueError(f'Published extraction hash mismatch: {cohort}')
    (OUT / 'input-manifest.json').write_text(json.dumps(
        dict(commit=COMMIT, records=records, published_extraction_hashes_verified=True), indent=2) + '\n')


if __name__ == '__main__':
    restore()
