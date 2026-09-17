"""Freeze completed online artifacts and verify the untouched shared service."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from gate_runtime import BASE
from audit_online_experiment import digest, read, write
from himoe_libero_bridge.client import PolicyClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    config = read(root / 'config.json')
    assert read(root / 'audit.json')['passed']
    assert (root / 'REPORT.zh.md').is_file()
    assert not (root / 'verification.json').exists()
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE),
                           '-p', 'test_*.py', '-v'], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=True)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(test.stdout)
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        service = client.metadata
    manifest = read(Path(config['parents'][0]['source']) / 'episode-trace.json')
    for actual, expected in (('server_instance_id', 'instance_id'), ('server_pid', 'pid'),
                             ('server_started_unix_ns', 'started_unix_ns')):
        assert service[actual] == manifest['source_server'][expected]
    for key, expected in manifest['policy_identity'].items():
        assert service[key] == expected
    write(root / 'shared-service-check.json', dict(passed=True, metadata=service,
                                                  inference_requests_sent=0))
    old_runs = {}
    for name in ('p1a-20260914-my_4lkxd', 'p1b-gate-20260914-1nvvkzw2', 'p2a-coverage-20260914-4i18r353'):
        old = BASE / 'runs' / name
        previous = read(old / 'verification.json')
        artifacts = previous.get('files', previous.get('file_sha256'))
        for path, expected in artifacts.items():
            assert digest(old / path) == expected, path
        sources = read(old / 'config.json')['source_hashes']
        for path, expected in sources.items():
            assert digest(path) == expected, path
        old_runs[name] = dict(unchanged_artifacts=len(artifacts), unchanged_sources=len(sources))
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    files = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()}
    result = dict(passed=True, files=files, unchanged_old_runs=old_runs,
                  source_hashes_unchanged=True, shared_service_same_source_instance=True,
                  model_calls=read(root / 'summary.json')['model_calls'],
                  post_collection_implementation_hashes={str(BASE / name): digest(BASE / name) for name in
                    ('audit_online_experiment.py', 'inspect_online_frames.py', 'finalize_online_experiment.py')},
                  environment_preflight=dict(run=str(BASE / 'runs/online-env-preflight-nu9cjdcj'),
                                              actual_action_steps=40, actual_settle_steps=20, model_calls=0))
    write(root / 'verification.json', result)
    print(json.dumps(dict(passed=True, files=len(files), old_runs=old_runs, gpu_after=gpu)), flush=True)


if __name__ == '__main__':
    main()
