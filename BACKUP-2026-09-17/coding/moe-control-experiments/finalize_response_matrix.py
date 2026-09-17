"""Seal the response matrix and verify unchanged earlier experiments."""

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
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    original = config['shared_metadata']
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == original[key]
    for parent in config['parents']:
        manifest = read(Path(parent['source']) / 'episode-trace.json')
        for key, expected in parent['policy_identity'].items():
            assert metadata[key] == expected
        for live_key, source_key in (('server_instance_id', 'instance_id'), ('server_pid', 'pid'),
                                     ('server_started_unix_ns', 'started_unix_ns')):
            assert metadata[live_key] == manifest['source_server'][source_key]
    write(root / 'shared-service-check.json', dict(passed=True, inference_requests_sent=0, metadata=metadata))
    old_runs = {}
    for name in ('p1a-20260914-my_4lkxd', 'p1b-gate-20260914-1nvvkzw2', 'p2a-coverage-20260914-4i18r353',
                 'p3-online-20260915-tc0mhlig', 'p3b-head-control-20260915-s39ow9qi'):
        old = BASE / 'runs' / name
        verification = read(old / 'verification.json')
        files = verification.get('files', verification.get('file_sha256'))
        for path, expected in files.items():
            assert digest(old / path) == expected, path
        sources = read(old / 'config.json')['source_hashes']
        for path, expected in sources.items():
            assert digest(path) == expected, path
        for field in ('post_collection_implementation_hashes', 'post_collection_sources'):
            for path, expected in verification.get(field, {}).items():
                assert digest(path) == expected, path
        old_runs[name] = dict(unchanged_artifacts=len(files), unchanged_sources=len(sources))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    result = dict(passed=True, source_hashes_unchanged=True, old_runs_unchanged=old_runs,
                  shared_service_identity_unchanged=True,
                  post_collection_sources={str(BASE / name): digest(BASE / name) for name in
                      ('audit_response_matrix.py', 'plot_response_matrix.py', 'summarize_response_matrix.py',
                       'finalize_response_matrix.py')},
                  files={str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()})
    write(root / 'verification.json', result)
    print(dict(passed=True, sealed_files=len(result['files']), old_runs=old_runs, gpu_after=gpu), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
