"""Seal local artifacts after collection, independent audit and visual checks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from gate_runtime import BASE
from run_replan_window import check_config, shared_metadata
from run_online_experiment import now, save_json
from himoe_libero_bridge.episode_trace import sha256_file


def main(root):
    config = check_config(root, check_old=True)
    for name in ('collection.json', 'audit.json', 'visual-check.json', 'secondary-analysis.json'):
        assert json.loads((root / name).read_text())['passed']
    assert not list(root.rglob('*failure.json'))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    metadata = shared_metadata()
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == config['shared_metadata'][key]
    save_json(root / 'shared-service-check.json', dict(passed=True, metadata=metadata, inference_requests_sent=0))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    preflight_files = {}
    for name in ('p3h-env-preflight-bckhxgu6', 'p3h-cleanup-diagnostic-o3ukf6q4',
                 'p3h-cleanup-diagnostic-hpv0mvk1', 'p3h-env-preflight-7bdwhzfj'):
        for path in sorted((BASE / 'runs' / name).rglob('*')):
            if path.is_file():
                preflight_files[str(path)] = sha256_file(path)
    result = dict(passed=True, utc=now(), source_hashes_unchanged=True,
                  previous_sealed_artifacts_unchanged=len(config['previous_sealed_files']),
                  shared_service_identity_unchanged=True, initial_preflight_and_diagnostics=preflight_files,
                  post_collection_sources={str(BASE / name): sha256_file(BASE / name) for name in
                                           ('plot_replan_window.py', 'analyze_replan_window.py', 'finalize_replan_window.py')},
                  files={str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob('*')) if path.is_file()})
    save_json(root / 'verification.json', result)
    print(json.dumps(dict(passed=True, artifacts=len(result['files']), gpu=gpu)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
