"""Seal a completed crossover rollout without touching other GPU services."""

import argparse
from pathlib import Path
import subprocess
import sys

from gate_runtime import BASE
from run_crossover_rollout import check_config
from audit_online_experiment import read, write, digest
from himoe_libero_bridge.client import PolicyClient


def main(root):
    config = check_config(root)
    assert read(root / 'collection.json')['passed'] and read(root / 'audit.json')['passed']
    assert not list(root.rglob('*failure.json'))
    for name in ('REPORT.zh.md', 'EXECUTION_NOTES.zh.md', 'summary.json', 'crossover-rollout.png', 'branches.csv', 'decisions.csv'):
        assert (root / name).is_file()
    for parent in config['parents']:
        for name in ('final-frames.png', 'moe-trajectories.png'):
            assert (root / parent['name'] / name).is_file()
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == config['shared_metadata'][key]
    write(root / 'shared-service-check.json', dict(passed=True, inference_requests_sent=0, metadata=metadata))
    old_runs = {}
    names = ('p1a-20260914-my_4lkxd', 'p1b-gate-20260914-1nvvkzw2', 'p2a-coverage-20260914-4i18r353',
             'p3-online-20260915-tc0mhlig', 'p3b-head-control-20260915-s39ow9qi',
             'p3c-response-matrix-20260915-463ir6kj', 'p3d-gated-rollout-20260915-t9k1v766',
             'p3e-mechanism-20260915-9u_9j9b8', 'p3f-crossover-v2-20260915-ihpn5n13')
    for name in names:
        old = BASE / 'runs' / name
        verification = read(old / 'verification.json')
        files = verification.get('files', verification.get('file_sha256'))
        for path, expected in files.items():
            assert digest(old / path) == expected, path
        for path, expected in read(old / 'config.json')['source_hashes'].items():
            assert digest(path) == expected, path
        for field in ('post_collection_implementation_hashes', 'post_collection_sources'):
            for path, expected in verification.get(field, {}).items():
                assert digest(path) == expected, path
        old_runs[name] = dict(unchanged_artifacts=len(files))
    blocked = BASE / 'runs/p3g-crossover-rollout-20260915-lq1_j641'
    assert read(blocked / 'failure.json')['model_completed'] == 0
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    result = dict(passed=True, source_hashes_unchanged=True, old_runs_unchanged=old_runs,
        shared_service_identity_unchanged=True, original_parameter_content_hash_unchanged=True,
        initial_resource_failure={str(p): digest(p) for p in sorted(blocked.rglob('*')) if p.is_file()},
        post_collection_sources={str(BASE / name): digest(BASE / name) for name in
            ('audit_crossover_rollout.py', 'plot_crossover_rollout.py', 'finalize_crossover_rollout.py')},
        files={str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    write(root / 'verification.json', result)
    print(dict(passed=True, files=len(result['files']), old_artifacts=sum(r['unchanged_artifacts'] for r in old_runs.values()), gpu=gpu), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
