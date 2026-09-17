"""Seal P3d after collection, audit, media and protection of six old runs."""

import argparse
from pathlib import Path
import subprocess
import sys

from gate_runtime import BASE
from audit_online_experiment import digest, read, write
from himoe_libero_bridge.client import PolicyClient


def main(root):
    config = read(root / 'config.json')
    assert read(root / 'collection.json')['passed'] and read(root / 'audit.json')['passed']
    assert (root / 'REPORT.zh.md').is_file() and (root / 'rollout-overview.png').is_file()
    assert not (root / 'failure.json').exists()
    for parent in config['parents']:
        for name in ('endpoint-comparison.png', 'component-trajectories.png', 'component-trajectories.pdf'):
            assert (root / parent['name'] / name).is_file()
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == config['shared_metadata'][key]
    for parent in config['parents']:
        manifest = read(Path(parent['source']) / 'episode-trace.json')
        for key, expected in parent['policy_identity'].items():
            assert metadata[key] == expected
        for live, original in (('server_instance_id', 'instance_id'), ('server_pid', 'pid'),
                               ('server_started_unix_ns', 'started_unix_ns')):
            assert metadata[live] == manifest['source_server'][original]
    write(root / 'shared-service-check.json', dict(passed=True, metadata=metadata, inference_requests_sent=0))
    old_runs = {}
    for name in ('p1a-20260914-my_4lkxd', 'p1b-gate-20260914-1nvvkzw2', 'p2a-coverage-20260914-4i18r353',
                 'p3-online-20260915-tc0mhlig', 'p3b-head-control-20260915-s39ow9qi',
                 'p3c-response-matrix-20260915-463ir6kj'):
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
    result = dict(passed=True, source_hashes_unchanged=True, shared_service_identity_unchanged=True,
                  old_runs_unchanged=old_runs,
                  post_collection_sources={str(BASE / name): digest(BASE / name) for name in
                    ('audit_gated_rollout_experiment.py', 'plot_gated_rollout_results.py', 'finalize_gated_rollout_experiment.py')},
                  files={str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()})
    write(root / 'verification.json', result)
    print(dict(passed=True, sealed_files=len(result['files']), old_runs=old_runs, gpu_after=gpu), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
