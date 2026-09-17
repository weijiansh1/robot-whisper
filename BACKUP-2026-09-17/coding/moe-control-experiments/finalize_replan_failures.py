"""Seal audited failed-parent results, retaining the aborted out-of-scope branch."""

import argparse
from pathlib import Path
import subprocess
import sys

from audit_replan_window import read
from gate_runtime import BASE
from run_replan_window import check_config, shared_metadata
from run_online_experiment import now, save_json
from himoe_libero_bridge.episode_trace import sha256_file


def main(root):
    assert not (root / 'verification.json').exists()
    config = check_config(root, check_old=True)
    audit = read(root / 'failure-only-audit.json')
    visual = read(root / 'failure-only-visual-check.json')
    assert audit['passed'] and visual['passed']
    assert audit['actual_completed_rollouts'] == len(visual['videos']) == 48
    assert not audit['original_collection_completed'] and not (root / 'collection.json').exists()
    assert (root / 'FAILURE_ONLY_REPORT.zh.md').is_file()
    assert (root / 'failure.json').is_file()
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (root / 'failure-only-tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    assert tests.returncode == 0, tests.stdout
    metadata = shared_metadata()
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == config['shared_metadata'][key]
    save_json(root / 'failure-only-shared-service-check.json',
              dict(passed=True, metadata=metadata, inference_requests_sent=0))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    with (root / 'failure-only-gpu-after.txt').open('x') as stream:
        stream.write(gpu)
    preflight_files = {}
    for name in ('p3h-env-preflight-bckhxgu6', 'p3h-cleanup-diagnostic-o3ukf6q4',
                 'p3h-cleanup-diagnostic-hpv0mvk1', 'p3h-env-preflight-7bdwhzfj'):
        for path in sorted((BASE / 'runs' / name).rglob('*')):
            if path.is_file():
                preflight_files[str(path)] = sha256_file(path)
    result = dict(passed=True, utc=now(), scope='artifact verification for completed original-failure subset only',
                  original_collection_completed=False, completed_rollouts=48, paired_units=24,
                  partial_success_parent_preserved_and_excluded=True, terminal_parameter_digest_available=False,
                  source_hashes_unchanged=len(config['source_hashes']),
                  previous_sealed_artifacts_unchanged=len(config['previous_sealed_files']),
                  shared_service_identity_unchanged=True, initial_preflight_and_diagnostics=preflight_files,
                  post_collection_sources={str(BASE / name): sha256_file(BASE / name) for name in
                                           ('audit_replan_failures.py', 'test_replan_failure_audit.py',
                                            'plot_replan_failures.py', 'finalize_replan_failures.py')},
                  files={str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob('*')) if path.is_file()})
    save_json(root / 'verification.json', result)
    print('Failure-only artifacts sealed: %d files; %d prior files unchanged' %
          (len(result['files']), result['previous_sealed_artifacts_unchanged']), flush=True)
    print(tests.stdout.split('----------------------------------------------------------------------')[-1].strip(), flush=True)
    print(gpu, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
