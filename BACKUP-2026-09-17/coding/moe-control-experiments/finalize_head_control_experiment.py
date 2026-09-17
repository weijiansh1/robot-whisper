"""Preserve the complete controlled-variable experiment and older frozen runs."""

import argparse
from pathlib import Path
import subprocess
import sys

from gate_runtime import BASE
from audit_online_experiment import digest, read, write
from himoe_libero_bridge.client import PolicyClient


def main(root):
    config = read(root / 'config.json')
    assert read(root / 'audit.json')['passed'] and (root / 'REPORT.zh.md').is_file()
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                          check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(test.stdout)
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    manifest = read(Path(config['parents'][0]['source']) / 'episode-trace.json')
    for live_key, source_key in (('server_instance_id', 'instance_id'), ('server_pid', 'pid'),
                                  ('server_started_unix_ns', 'started_unix_ns')):
        assert metadata[live_key] == manifest['source_server'][source_key]
    for key, expected in manifest['policy_identity'].items():
        assert metadata[key] == expected
    write(root / 'shared-service-check.json', dict(passed=True, metadata=metadata, inference_requests_sent=0))
    old_runs = {}
    for name in ('p1a-20260914-my_4lkxd', 'p1b-gate-20260914-1nvvkzw2',
                 'p2a-coverage-20260914-4i18r353', 'p3-online-20260915-tc0mhlig'):
        old = BASE / 'runs' / name
        verification = read(old / 'verification.json')
        files = verification.get('files', verification.get('file_sha256'))
        for path, expected in files.items():
            assert digest(old / path) == expected, path
        sources = read(old / 'config.json')['source_hashes']
        for path, expected in sources.items():
            assert digest(path) == expected, path
        for path, expected in verification.get('post_collection_implementation_hashes', {}).items():
            assert digest(path) == expected, path
        old_runs[name] = dict(unchanged_artifacts=len(files), unchanged_sources=len(sources))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    result = dict(passed=True, source_hashes_unchanged=True, old_runs_unchanged=old_runs,
                  shared_service_source_identity_unchanged=True,
                  files={str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()},
                  post_collection_sources={str(BASE / name): digest(BASE / name) for name in
                    ('audit_head_control_experiment.py', 'plot_head_control_results.py', 'finalize_head_control_experiment.py')})
    write(root / 'verification.json', result)
    print(dict(passed=True, files=len(result['files']), old_runs=old_runs, gpu_after=gpu), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve())
