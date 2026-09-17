"""Read-only request hooks; returned model tensors are never replaced."""

import numpy as np

from collection_routes import HB_LAYERS
from mechanism_protocol import HB_FIELDS


class MechanismCapture:
    def __init__(self, model):
        self.model = model
        self.layers = model.paligemma_with_expert.gemma_expert.layers
        self.handles = []
        self.records = {field: [] for field in HB_FIELDS + ('projection_input', 'velocity')}

    def hook(self, field, layer, input_value=False, tuple_output=False):
        def capture(module, inputs, output):
            value = inputs[0] if input_value else output[0] if tuple_output else output
            expected = (1, 10) if field in ('projection_input', 'velocity') else (1, 11)
            if tuple(value.shape[:2]) != expected:
                raise RuntimeError('Unexpected activation shape: ' + field)
            self.records[field].append((layer, value.detach().clone()))
        return capture

    def __enter__(self):
        try:
            for index in HB_LAYERS:
                block = self.layers[index]
                if type(block.mlp).__name__ != 'HBMoE':
                    raise RuntimeError('HB module identity changed')
                for module, field, inp, tup in (
                    (block.mlp, 'input', True, False),
                    (block.mlp.shared_experts, 'shared', False, False),
                    (block.mlp, 'total', False, False),
                    (block.post_attention_layernorm, 'residual', True, False),
                    (block, 'block', False, True),
                ):
                    self.handles.append(module.register_forward_hook(self.hook(field, index, inp, tup)))
            for field, inp in (('projection_input', True), ('velocity', False)):
                self.handles.append(self.model.action_out_proj.register_forward_hook(self.hook(field, -1, inp)))
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
            expected = list(HB_LAYERS) * 10 if field in HB_FIELDS else [-1] * 10
            if [index for index, _ in rows] != expected:
                raise RuntimeError('Incomplete activation capture: ' + field)
            stacked = torch.stack([value for _, value in rows]).float().cpu().numpy()
            value = stacked.reshape(10, 8, 11, 1024).transpose(1, 0, 2, 3) if field in HB_FIELDS else stacked[:, 0]
            if not np.isfinite(value).all():
                raise RuntimeError('Nonfinite activation')
            result['mechanism/' + field] = np.ascontiguousarray(value)
        if not np.array_equal(result['mechanism/residual'] + result['mechanism/total'], result['mechanism/block']):
            raise RuntimeError('Captured residual addition does not reproduce block output')
        return result


def traced_infer(wrapped, request):
    from gate_runtime import infer_isolated
    model = wrapped.policy._policy.model
    modules = list(model.modules())
    hooks_before = [tuple(module._forward_hooks) for module in modules]
    with MechanismCapture(model) as capture:
        response, resource = infer_isolated(wrapped, request)
    if hooks_before != [tuple(module._forward_hooks) for module in modules]:
        raise RuntimeError('Activation hook leaked')
    response.update(capture.response())
    return response, resource
