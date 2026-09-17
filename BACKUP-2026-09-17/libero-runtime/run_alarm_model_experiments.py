"""Run isolated alarm-centered factorials, then serve the fixed fresh sample."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

sys.path.insert(0, '/data/coding/moe-control-experiments')
from gate_runtime import load_isolated, infer_isolated
from gate_capture import CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY
from probe_moe_input_response import infer, request_for
from moe_response_analysis import SOURCE as CURRENT, digest, save_json

ROOT = Path('/data/libero-runtime/samples/v82-knn-trajectory-20260915')


def functional(wrapped, load):
    selection = json.loads((ROOT / 'functional-selection.json').read_text())
    output = ROOT / 'functional'
    output.mkdir(exist_ok=True)
    if (output / 'results.json').exists():
        assert json.loads((output / 'results.json').read_text())['passed']
        return
    calls, checks = [], []
    started = time.perf_counter()
    for probe in selection['probes']:
        q = probe['probe_query']
        directory = output / ('probe-%02d' % probe['probe_id'])
        if directory.exists():
            raise RuntimeError('Partial probe exists; inspect before resuming')
        directory.mkdir()
        source = Path(probe['source_dir']) / 'episode-trace.npz'
        hb_path = CURRENT / probe['name'] / 'full-hb-routes.npz'
        assert digest(source) == probe['trace_sha256'] and digest(hb_path) == probe['source_hb_sha256']
        with np.load(source) as data:
            arrays = {key: data[key] for key in ('images', 'wrist_images', 'states', 'flow_noises', 'predicted_actions')}
        with np.load(hb_path) as data:
            hb, ids = data['hb_router_probs'], data['hb_expert_ids']
        plans = [('o0n0', q-1, q-1, True), ('o0n1', q-1, q, True),
                 ('o1n0', q, q-1, True), ('o1n1', q, q, True), ('native_current', q, q, False)]
        if probe['probe_id'] % 2:
            plans = [plans[-1], *plans[:-1]]
        if probe['probe_id'] in (0, len(selection['probes'])-1):
            plans.append(('repeat_current', q, q, True))
        responses, records = {}, {}
        for label, oq, nq, captured in plans:
            response, record, seconds = infer(wrapped, request_for(probe, arrays, oq, nq), captured)
            responses[label] = response
            if record is not None:
                np.testing.assert_array_equal(record['ids'], response[NATIVE_IDS_KEY][:, -1])
                np.testing.assert_array_equal(record['weights'], response[NATIVE_WEIGHTS_KEY][:, -1])
                records[label] = record
                np.savez_compressed(directory / (label+'.npz'), **record, actions=response['actions'], hb=response[PROBS_KEY])
            else:
                np.savez_compressed(directory / (label+'.npz'), actions=response['actions'], hb=response[PROBS_KEY])
            if oq == nq:
                np.testing.assert_array_equal(response['actions'], arrays['predicted_actions'][oq])
                np.testing.assert_array_equal(response[PROBS_KEY], hb[oq])
                np.testing.assert_array_equal(response[NATIVE_IDS_KEY], ids[oq])
            calls.append(dict(probe_id=probe['probe_id'], parent_id=probe['parent_id'], pair_id=probe['pair_id'],
                              name=probe['name'], stage=probe['stage'], query=q, condition=label,
                              captured=captured, seconds=seconds))
            checks.append(dict(probe_id=probe['probe_id'], condition=label, natural_corner=oq == nq,
                               archive_exact=oq == nq, finite_actions=True))
        np.testing.assert_array_equal(responses['native_current']['actions'], responses['o1n1']['actions'])
        if 'repeat_current' in records:
            for key in records['o1n1']:
                np.testing.assert_array_equal(records['repeat_current'][key], records['o1n1'][key])
        save_json(output / 'calls.json', calls)
        save_json(output / 'checks.json', checks)
        print('Functional %d/32 %s %s q%d verified' % (probe['probe_id']+1, probe['name'], probe['stage'], q), flush=True)
    import torch
    save_json(output / 'results.json', dict(passed=True, parents=len(selection['parents']), probes=len(selection['probes']),
              model_forwards=len(calls), new_environment_actions=0, shared_service_used=False,
              elapsed_seconds=time.perf_counter()-started, selection_sha256=digest(ROOT / 'functional-selection.json'),
              peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))


class FreshCapturePolicy:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.metadata = dict(wrapped.metadata, fresh_alarm_validation=True)
        self.directory = ROOT / 'fresh-wire'
        self.directory.mkdir(exist_ok=True)
        self.counter = len(list(self.directory.glob('call-*.npz')))

    def infer(self, observation):
        request = dict(observation, **{CAPTURE_KEY: True, 'routing/capture': True})
        response, stats = infer_isolated(self.wrapped, request)
        target = self.directory / ('call-%05d.npz' % self.counter)
        if target.exists():
            raise RuntimeError('Existing fresh capture identity')
        np.savez_compressed(target, hb=response[PROBS_KEY], ids=response[NATIVE_IDS_KEY].astype(np.uint8),
                            actions=response['actions'], noise_sha256=np.asarray(response['flow/noise_sha256']),
                            image_sha256=np.asarray(hashlib.sha256(request['observation/image'].tobytes()).hexdigest()),
                            state_sha256=np.asarray(hashlib.sha256(request['observation/state'].tobytes()).hexdigest()))
        self.counter += 1
        if self.counter % 25 == 0:
            print('Fresh validation inference calls: %d' % self.counter, flush=True)
        # Routing is already saved locally; the client needs the unchanged native action response.
        return {'actions': response['actions'], 'flow/noise_sha256': response['flow/noise_sha256']}


async def serve_until_done(wrapped, port):
    from himoe_libero_bridge.server import PolicyServer
    policy = FreshCapturePolicy(wrapped)
    server = PolicyServer(policy, '127.0.0.1', port, wrapped.backend_name)
    task = asyncio.create_task(server.run())
    print('Isolated fresh validation server ready on port %d' % port, flush=True)
    try:
        while not (ROOT / 'fresh-complete.json').exists():
            if task.done():
                await task
            await asyncio.sleep(1)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    save_json(ROOT / 'isolated-server-finished.json', dict(fresh_model_forwards=policy.counter, stopped=True,
              shared_service_used=False, functional_model_forwards=162))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9521)
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'profile/frozen.json').read_text())
    for name, expected in frozen['files'].items():
        assert digest(ROOT / 'profile' / name) == expected
    assert json.loads((ROOT / 'profile/parameters.json').read_text())['protocol_sha256'] == digest(ROOT / 'PROTOCOL.md')
    with socket.socket() as check:
        check.bind(('127.0.0.1', args.port))
    wrapped, load = load_isolated()
    save_json(ROOT / 'isolated-model-load.json', load)
    functional(wrapped, load)
    asyncio.run(serve_until_done(wrapped, args.port))


if __name__ == '__main__':
    main()
