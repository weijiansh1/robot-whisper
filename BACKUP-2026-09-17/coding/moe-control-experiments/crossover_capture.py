"""Scoped total-output replay, preserving a separate pre-replay compute trace."""

import hashlib

import numpy as np

from gate_runtime import infer_isolated
from collection_routes import HB_LAYERS
from crossover_protocol import validate_donor
from mechanism_capture import MechanismCapture


def replace_action_tokens(output, donor, slot, step):
    import torch

    if tuple(output.shape) != (1, 11, 1024) or output.dtype != torch.float32:
        raise RuntimeError('Unexpected MoE output shape or dtype')
    if slot not in range(4, 8) or step not in range(10):
        raise RuntimeError('Replay index outside frozen scope')
    patched = output.clone()
    patched[:, 1:, :] = torch.from_numpy(donor[slot, step, 1:, :].copy()).to(output.device)
    return patched


class OutputReplayScope:
    def __init__(self, model, donor=None):
        self.layers = model.paligemma_with_expert.gemma_expert.layers
        self.donor = None if donor is None else validate_donor(donor)
        self.handles, self.raw, self.patches = [], [], []
        self.steps = {slot: 0 for slot in range(4, 8)}

    def raw_hook(self, index):
        def capture(module, inputs, output):
            self.raw.append((index, output.detach().clone()))
        return capture

    def patch_hook(self, slot):
        def patch(module, inputs, output):
            step = self.steps[slot]
            value = replace_action_tokens(output, self.donor, slot, step)
            self.steps[slot] += 1
            self.patches.append((HB_LAYERS[slot], step))
            return value
        return patch

    def __enter__(self):
        try:
            for slot, index in enumerate(HB_LAYERS):
                module = self.layers[index].mlp
                if type(module).__name__ != 'HBMoE':
                    raise RuntimeError('HB module identity changed')
                self.handles.append(module.register_forward_hook(self.raw_hook(index)))
                if self.donor is not None and slot >= 4:
                    self.handles.append(module.register_forward_hook(self.patch_hook(slot)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def response(self):
        import torch

        if [i for i, _ in self.raw] != list(HB_LAYERS) * 10:
            raise RuntimeError('Incomplete raw-total capture')
        expected = [] if self.donor is None else [(i, t) for t in range(10) for i in HB_LAYERS[4:]]
        if self.patches != expected:
            raise RuntimeError('Output replay order or coverage mismatch')
        values = torch.stack([v for _, v in self.raw]).float().cpu().numpy()
        value = np.ascontiguousarray(values.reshape(10, 8, 11, 1024).transpose(1, 0, 2, 3))
        validate_donor(value)
        return {'crossover/raw_total': value,
                'crossover/patch_sites': np.asarray(self.patches, np.int16).reshape(-1, 2)}


def crossover_infer(wrapped, request, donor=None):
    model = wrapped.policy._policy.model
    modules = list(model.modules())
    before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]
    try:
        # Registration order is raw capture, optional replay, effective trace.
        with OutputReplayScope(model, donor) as replay, MechanismCapture(model) as capture:
            response, resource = infer_isolated(wrapped, request)
    finally:
        after = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]
        if before != after:
            raise RuntimeError('Crossover hook leaked')
    response.update(capture.response())
    response.update(replay.response())
    return response, resource


def parameter_versions(model):
    return [(name, tuple(value.shape), str(value.dtype), value.data_ptr(), value._version)
            for name, value in model.named_parameters()]


def parameter_digest(model):
    import torch

    result = hashlib.sha256()
    for name, value in model.named_parameters():
        result.update(repr((name, tuple(value.shape), str(value.dtype))).encode('ascii'))
        result.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()
