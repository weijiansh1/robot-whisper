"""Restore recorded kNN evidence without modifying the original checkout."""

import hashlib
import argparse
import json
from pathlib import Path
import subprocess

REPO = Path('/data/coding/robot-whisper-0909')
ROOT = Path('/data/libero-runtime/samples/v82-knn-mechanism-20260915')
COMMIT = '5da830a37230f4ecd761f70be4999f4b3a01687e'
FROZEN = 'moe-trap-control/design/frozen_alarm_comparison_20260908'
ROUND7 = 'safe&vlaconf/moe_trainfree/results/round7_temporal_fusion/false_alarm_diagnosis'
FILES = [
    *[f'{FROZEN}/{name}' for name in ('contract.json', 'index.csv', 'first_alarms.csv',
                                      'verification.json', 'input_verification.json',
                                      'profiles/manifest.json', 'profiles/parameters.json')],
]
OPTIONAL_FILES = [
    *[f'{ROUND7}/{name}' for name in ('alarm_feature_attribution.csv', 'feature_summary.csv',
                                     'task_feature_summary.csv', 'same_episode_seen_unseen.csv',
                                     'bank_addition_summary.csv', 'verification.json')],
    'safe&vlaconf/moe_trainfree/results/round9_full_corpus/trajectory_results.csv',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-prior-details', action='store_true')
    args = parser.parse_args()
    records = []
    for source in FILES + (OPTIONAL_FILES if args.include_prior_details else []):
        item = subprocess.check_output(['git', '-c', 'gc.auto=0', 'ls-tree', COMMIT, '--', source],
                                       cwd=REPO, text=True).strip()
        expected = item.split()[2]
        path = ROOT / 'inputs' / source
        if path.exists():
            content = path.read_bytes()
        else:
            content = subprocess.check_output(['timeout', '45s', 'git', '-c', 'gc.auto=0',
                                               'show', f'{COMMIT}:{source}'], cwd=REPO)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if actual != expected:
            raise ValueError(f'Git content mismatch: {source}')
        records.append(dict(source=source, git_blob=actual, bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest()))
        print(f'Restored {source}: {len(content)} bytes', flush=True)
    missing = [f'{FROZEN}/profiles/global_reference.npz', f'{FROZEN}/scores_and_alarms.npz',
               'safe&vlaconf/moe_trainfree/results/round3_safe/v7/v7_inputs.npz']
    for path in missing:
        entry = subprocess.check_output(['git', '-c', 'gc.auto=0', 'ls-tree', COMMIT, '--', path],
                                        cwd=REPO, text=True)
        if entry.strip() or (REPO / path).exists():
            raise ValueError(f'Previously missing original input is now available: {path}')
    (ROOT / 'input-manifest.json').write_text(json.dumps(dict(commit=COMMIT, files=records,
                unavailable_original_arrays=missing, optional_prior_details_requested=args.include_prior_details), indent=2) + '\n')


if __name__ == '__main__':
    main()
