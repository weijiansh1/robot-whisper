"""Whole-HB-output replay to isolate the complete finite-precision boundary."""

import numpy as np

from crossover_capture import OutputReplayScope, crossover_infer
from gate_runtime import infer_isolated
from mechanism_capture import MechanismCapture
from collection_routes import HB_LAYERS

OUTPUT_MASK = np.zeros((8, 10, 11), bool)
OUTPUT_MASK[4:] = True


class WholeOutputScope(OutputReplayScope):
    def patch_hook(self, slot):
        def patch(module, inputs, output):
            import torch
            step = self.steps[slot]
            if tuple(output.shape) != (1, 11, 1024) or output.dtype != torch.float32 or step not in range(10):
                raise RuntimeError('Unexpected whole-output replay shape, dtype, or step')
            patched = output.clone()
            patched.copy_(torch.from_numpy(self.donor[slot, step].copy()).to(output.device)[None])
            self.steps[slot] += 1
            self.patches.append((HB_LAYERS[slot], step))
            return patched
        return patch


def whole_output_infer(wrapped, request, donor=None):
    model = wrapped.policy._policy.model
    modules = list(model.modules())
    before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]
    try:
        with WholeOutputScope(model, donor) as replay, MechanismCapture(model) as capture:
            response, resource = infer_isolated(wrapped, request)
    finally:
        if before != [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in modules]:
            raise RuntimeError('Whole-output replay hook leaked')
    response.update(capture.response())
    response.update(replay.response())
    response['crossover/replay_token_mask'] = OUTPUT_MASK.copy() if donor is not None else np.zeros_like(OUTPUT_MASK)
    return response, resource
