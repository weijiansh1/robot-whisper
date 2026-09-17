"""Read-only capture of three HB MoE ports, with no other activations."""

import numpy as np

from gate_runtime import infer_isolated
from collection_routes import HB_LAYERS

FIELDS = ('input', 'shared', 'total')


class MoEPortCapture:
    def __init__(self, model):
        self.layers = model.paligemma_with_expert.gemma_expert.layers
        self.handles = []
        self.records = {field: [] for field in FIELDS}

    def hook(self, field, layer):
        def capture(module, inputs, output):
            value = inputs[0] if field == 'input' else output
            if tuple(value.shape) != (1, 11, 1024):
                raise RuntimeError('Unexpected HB MoE port shape: ' + field)
            self.records[field].append((layer, value.detach().clone()))
        return capture

    def __enter__(self):
        try:
            for index in HB_LAYERS:
                module = self.layers[index].mlp
                if type(module).__name__ != 'HBMoE':
                    raise RuntimeError('HB MoE module identity changed')
                for field in FIELDS:
                    port = module.shared_experts if field == 'shared' else module
                    self.handles.append(port.register_forward_hook(self.hook(field, index)))
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
        result = {}
        for field, rows in self.records.items():
            if [i for i, _ in rows] != list(HB_LAYERS) * 10:
                raise RuntimeError('Incomplete MoE port capture: ' + field)
            values = torch.stack([value for _, value in rows]).float().cpu().numpy()
            values = np.ascontiguousarray(values.reshape(10, 8, 11, 1024).transpose(1, 0, 2, 3))
            if not np.isfinite(values).all():
                raise RuntimeError('Nonfinite MoE port')
            result['mechanism/' + field] = values
        return result


def moe_infer(wrapped, request):
    model = wrapped.policy._policy.model
    modules = list(model.modules())
    before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]
    try:
        with MoEPortCapture(model) as capture:
            response, resources = infer_isolated(wrapped, request)
    finally:
        after = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]
        if before != after:
            raise RuntimeError('MoE capture leaked a hook')
    response.update(capture.response())
    return response, resources
