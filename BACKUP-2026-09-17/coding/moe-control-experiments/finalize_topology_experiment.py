"""Seal CPU-only topology results and verify untouched archived experiments."""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

from audit_replan_window import read
from gate_runtime import BASE
from run_online_experiment import now, save_json
from run_replan_window import shared_metadata
from run_topology_experiment import check_config
from himoe_libero_bridge.episode_trace import sha256_file


def main(root):
    assert not (root / 'verification.json').exists()
    config = check_config(root, protected=True)
    for name in ('summary.json', 'audit.json', 'post-analysis.json', 'visual-check.json'):
        assert read(root / name)['passed']
    assert not (root / 'failure.json').exists()
    assert (root / 'REPORT.zh.md').is_file()
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
    result = dict(passed=True, utc=now(), model_inference_calls=0, new_environment_actions=0,
                  new_success_parent_rollouts=0, new_controller_test=False,
                  source_hashes_unchanged=len(config['source_hashes']),
                  previous_sealed_files_unchanged=len(config['protected_files']),
                  dependency_files_unchanged=len(config['dependency_hashes']),
                  shared_service_identity_unchanged=True, networkx_version=metadata.version('networkx'),
                  post_collection_sources={str(BASE / name): sha256_file(BASE / name) for name in
                                           ('audit_topology_experiment.py', 'report_topology_experiment.py',
                                            'finalize_topology_experiment.py')},
                  files={str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob('*')) if path.is_file()})
    save_json(root / 'verification.json', result)
    print('Sealed %d topology artifacts; %d previous files unchanged' %
          (len(result['files']), result['previous_sealed_files_unchanged']), flush=True)
    print(tests.stdout.split('----------------------------------------------------------------------')[-1].strip(), flush=True)
    print(gpu, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
