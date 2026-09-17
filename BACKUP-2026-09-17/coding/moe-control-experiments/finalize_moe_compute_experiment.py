"""Seal MoE replay results without changing any previous experiment."""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

from audit_replan_window import read
from gate_runtime import BASE
from run_moe_compute_experiment import check_config
from run_online_experiment import now, save_json
from run_replan_window import shared_metadata
from himoe_libero_bridge.episode_trace import sha256_file


def main(root):
    assert not (root / 'verification.json').exists()
    config = check_config(root, protected=True)
    for name in ('preflight.json', 'collection.json', 'analysis-summary.json', 'prediction-summary.json',
                 'audit.json', 'port-amplitudes.json', 'visual-check.json', 'report-summary.json'):
        assert read(root / name)['passed'], name
    assert not (root / 'failure.json').exists() and not (root / 'analysis-failure.json').exists()
    assert (root / 'REPORT.zh.md').is_file()
    for record, name in (('analysis-started.json', 'analyze_moe_compute_experiment.py'),
                         ('audit-started.json', 'audit_moe_compute_experiment.py')):
        assert read(root / record)['source_sha256'] == sha256_file(BASE / name)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    assert tests.returncode == 0, tests.stdout
    service = shared_metadata()
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert service[key] == config['shared_metadata'][key]
    save_json(root / 'shared-service-check.json', dict(passed=True, metadata=service, inference_requests_sent=0))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    post_sources = {str(BASE / name): sha256_file(BASE / name) for name in (
        'analyze_moe_compute_experiment.py', 'audit_moe_compute_experiment.py', 'test_moe_compute_audit.py',
        'report_moe_compute_experiment.py', 'finalize_moe_compute_experiment.py')}
    files = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob('*')) if path.is_file()}
    collection = read(root / 'collection.json')
    save_json(root / 'verification.json', dict(passed=True, utc=now(), model_calls=collection['calls'],
        new_environment_actions=0, new_success_parent_rollouts=0, new_controller_test=False,
        parameter_content_hash_unchanged=True, shared_service_identity_unchanged=True,
        versions={name: metadata.version(name) for name in ('numpy', 'scipy', 'networkx', 'torch')},
        gudhi_version='3.13.0', python_version=sys.version,
        source_hashes_unchanged=len(config['source_hashes']), previous_sealed_files_unchanged=len(config['protected_files']),
        dependency_files_unchanged=len(config['dependency_hashes']), post_collection_sources=post_sources, files=files))
    print('Sealed %d artifacts; %d protected files and %d frozen sources unchanged' %
          (len(files), len(config['protected_files']), len(config['source_hashes'])), flush=True)
    print(tests.stdout.split('----------------------------------------------------------------------')[-1].strip(), flush=True)
    print(gpu, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
