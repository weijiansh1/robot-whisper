"""Bounded, isolated observation/noise factorial from existing trajectory frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, '/data/coding/moe-control-experiments')
from gate_runtime import load_isolated
from gate_capture import CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY
from moe_final_response_recorder import FinalResponseRecorder
from moe_response_analysis import ROOT, SOURCE, digest, save_json


def request_for(parent, arrays, observation_query, noise_query):
    return {'observation/image': arrays['images'][observation_query],
            'observation/wrist_image': arrays['wrist_images'][observation_query],
            'observation/state': arrays['states'][observation_query], 'prompt': parent['prompt'],
            'flow/noise': arrays['flow_noises'][noise_query],
            'episode_id': -1526091500-parent['parent_id'], 'routing/capture': True, CAPTURE_KEY: True}


def infer(wrapped, request, captured):
    import torch
    recorder = FinalResponseRecorder(wrapped.policy._policy.model).attach() if captured else None
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=False):
            response = wrapped.infer(dict(request))
        torch.cuda.synchronize()
        record = recorder.end() if recorder else None
        inference_seconds = time.perf_counter()-start
    finally:
        if recorder:
            recorder.close()
    if response.get('flow/noise_sha256') != hashlib.sha256(request['flow/noise'].tobytes()).hexdigest():
        raise ValueError('flow noise acknowledgement mismatch')
    if any(m._forward_hooks or m._forward_pre_hooks for m in wrapped.policy._policy.model.modules()):
        raise ValueError('capture hooks leaked into another request')
    return response, record, inference_seconds


def main():
    import torch
    selection = json.loads((ROOT / 'selection.json').read_text())
    output = ROOT / 'controlled-probe'
    if output.exists():
        raise SystemExit('Probe output exists; inspect it before another run')
    # The existing loader checks current free memory before allocating anything.
    wrapped, load = load_isolated()
    output.mkdir()
    save_json(output / 'model-load.json', load)
    start = time.perf_counter()
    calls, checks = [], []
    parents = selection['parents']
    for parent in parents:
        q = parent['query']
        manifest = json.loads((Path(parent['source_dir']) / 'episode-trace.json').read_text())
        trace_path = Path(parent['source_dir']) / manifest['array_file']
        hb_path = SOURCE / parent['name'] / 'full-hb-routes.npz'
        if digest(trace_path) != parent['trace_sha256'] or digest(hb_path) != parent['source_hb_sha256']:
            raise ValueError('source changed since selection')
        with np.load(trace_path) as z:
            arrays = {key: z[key] for key in ('images', 'wrist_images', 'states', 'flow_noises', 'predicted_actions')}
        with np.load(hb_path) as z:
            hb, native_ids = z['hb_router_probs'], z['hb_expert_ids']
        directory = output / parent['name']
        directory.mkdir()
        responses, records, times = {}, {}, {}
        plans = [('o0n0', q-1, q-1, True), ('o0n1', q-1, q, True),
                 ('o1n0', q, q-1, True), ('o1n1', q, q, True), ('native_current', q, q, False)]
        if parent['parent_id'] % 2:
            plans = [plans[-1], *plans[:-1]]
        if parent['parent_id'] in (0, len(parents)-1):
            plans.append(('repeat_current', q, q, True))
        for label, oq, nq, captured in plans:
            request = request_for(parent, arrays, oq, nq)
            response, record, seconds = infer(wrapped, request, captured)
            responses[label], times[label] = response, seconds
            if record is not None:
                records[label] = record
                if not np.array_equal(record['ids'], response[NATIVE_IDS_KEY][:, -1]):
                    raise ValueError('captured dispatch IDs disagree')
                if not np.array_equal(record['weights'], response[NATIVE_WEIGHTS_KEY][:, -1]):
                    raise ValueError('captured dispatch weights disagree')
                np.savez_compressed(directory / (label + '.npz'), **record,
                                    actions=response['actions'], hb=response[PROBS_KEY])
            else:
                np.savez_compressed(directory / (label + '.npz'), actions=response['actions'], hb=response[PROBS_KEY])
            check = dict(parent=parent['name'], condition=label, observation_query=oq, noise_query=nq,
                         captured=captured, finite_actions=bool(np.isfinite(response['actions']).all()))
            if oq == nq:
                check.update(archived_actions_exact=bool(np.array_equal(response['actions'], arrays['predicted_actions'][oq])),
                             archived_hb_exact=bool(np.array_equal(response[PROBS_KEY], hb[oq])),
                             archived_ids_exact=bool(np.array_equal(response[NATIVE_IDS_KEY], native_ids[oq])))
            if not all(v for k, v in check.items() if k.endswith('_exact') or k == 'finite_actions'):
                save_json(output / 'failed-check.json', check)
                raise ValueError('natural corner failed archived replay')
            checks.append(check)
            calls.append(dict(parent_id=parent['parent_id'], pair_id=parent['pair_id'], parent=parent['name'],
                              condition=label, captured=captured, seconds=seconds,
                              flow_noise_sha256=response['flow/noise_sha256']))
            save_json(output / 'calls.json', calls)
            save_json(output / 'checks.json', checks)
        if not np.array_equal(responses['native_current']['actions'], responses['o1n1']['actions']):
            raise ValueError('recorder changed native action')
        if 'repeat_current' in records:
            if not all(np.array_equal(records['repeat_current'][k], records['o1n1'][k]) for k in records['o1n1']):
                raise ValueError('full functional tensors are not deterministic on repeated input')
        print(f"Verified {parent['name']} q{q}: observation/noise factorial, capture delta "
              f"{times['o1n1']-times['native_current']:.3f}s", flush=True)
    save_json(output / 'results.json', dict(passed=True, parents=len(parents), task_pairs=len(selection['pairs']),
                                           model_forwards=len(calls), captured_forwards=sum(c['captured'] for c in calls),
                                           new_environment_actions=0, shared_service_used=False,
                                           experimental_seconds=time.perf_counter()-start,
                                           peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                                           selection_sha256=digest(ROOT / 'selection.json')))
    print((output / 'results.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
