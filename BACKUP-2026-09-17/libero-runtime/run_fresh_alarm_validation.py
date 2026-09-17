"""Execute the twenty preselected episodes against the isolated local policy."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from alarm_trajectory_study import ROOT, load_bank
from v82_knn_research import digest, save_json

BASE = Path('/data/libero-runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9521)
    args = parser.parse_args()
    load_bank()
    frozen = json.loads((ROOT / 'fresh-plans.json').read_text())
    started = time.monotonic()
    while True:
        try:
            with socket.create_connection(('127.0.0.1', args.port), timeout=1):
                break
        except OSError:
            if time.monotonic()-started > 900:
                raise RuntimeError('Isolated model did not become ready')
            time.sleep(2)
    sys.path.insert(0, '/data/srv/src')
    from himoe_libero_bridge.client import PolicyClient
    with PolicyClient('127.0.0.1', args.port, connect_timeout=10) as client:
        assert client.metadata.get('fresh_alarm_validation') is True
    records = []
    for variant in ('plus', 'pro'):
        root = ROOT / 'fresh' / variant
        prior_batches = list(root.glob('batch-*')) if root.exists() else []
        if len(prior_batches) > 1:
            raise RuntimeError('Ambiguous fresh batch')
        command = [str(BASE / 'envs/libero/bin/python'), '-u', str(BASE / 'run_benchmark_sample.py'),
                   '--benchmark', variant, '--sample-seed', '2026091502', '--episodes-per-task', '1',
                   '--port', str(args.port), '--output-root', str(ROOT / 'fresh')]
        if prior_batches:
            command += ['--resume', str(prior_batches[0])]
        print('Starting frozen fresh %s sample: ten tasks, init 11' % variant, flush=True)
        with (ROOT / ('fresh-'+variant+'.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(event, dict) and event.get('event') == 'sample_episode_finished':
                    print('%s task %02d: success=%s, actions=%d' % (
                        variant, event['base_task_id'], event['success'], event['action_steps']), flush=True)
            if process.wait() != 0:
                raise RuntimeError('Fresh %s runner failed; inspect its log' % variant)
        plans = frozen['plans'][variant]['jobs']
        summaries = list(root.glob('batch-*/episodes/*/episode-*/summary.json'))
        for job in plans:
            matches = []
            for path in summaries:
                summary = json.loads(path.read_text())
                if summary.get('task_name') == job['task_name'] and summary['init_state_id'] == job['init_state_id']:
                    matches.append((path, summary))
            if len(matches) != 1:
                raise RuntimeError('Missing or ambiguous complete fresh episode')
            path, summary = matches[0]
            assert summary['status'] == 'completed' and summary['flow_noise_seed'] == job['flow_noise_seed']
            trace = json.loads((path.parent / 'episode-trace.json').read_text())
            assert trace['result']['trace_complete']
            assert digest(path.parent / trace['array_file']) == trace['array_file_sha256']
            records.append(dict(name='fresh-%s-task%02d-init%03d' % (variant, job['base_task_id'], job['init_state_id']),
                benchmark=variant, base_task_id=job['base_task_id'], init_state_id=job['init_state_id'],
                source_artifact_dir=str(path.parent), source_trace_sha256=trace['array_file_sha256'],
                task_suite=summary['task_suite'], task_id=summary['task_id'], prompt=summary['prompt'],
                queries=summary['inference_calls'], action_steps=summary['action_steps'],
                task_name=summary['task_name'], flow_noise_seed=job['flow_noise_seed'],
                source_result=dict(success=summary['success'], status=summary['status'],
                                   action_steps=summary['action_steps'], trace_complete=True),
                policy_identity=trace['policy_identity']))
        save_json(ROOT / 'fresh-manifest.json', dict(episodes=records, plans_sha256=digest(ROOT / 'fresh-plans.json')))
    assert len(records) == 20 and len({r['source_trace_sha256'] for r in records}) == 20
    save_json(ROOT / 'fresh-complete.json', dict(episodes=20, success=sum(r['source_result']['success'] for r in records),
              environment_actions=sum(r['action_steps'] for r in records), model_forwards=sum(r['queries'] for r in records),
              all_completed=True, plans_sha256=digest(ROOT / 'fresh-plans.json'),
              elapsed_seconds=time.monotonic()-started, shared_service_used=False))
    print((ROOT / 'fresh-complete.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
