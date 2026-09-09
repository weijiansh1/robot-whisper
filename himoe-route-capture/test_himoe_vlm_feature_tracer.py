from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from himoe_vlm_feature_tracer import VLMFeatureTracer


class _Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)

    def forward(self, *, hidden_states, **_kwargs):
        return (2.0 * hidden_states,)


class _Model:
    def __init__(self):
        language = SimpleNamespace(layers=nn.ModuleList([_Layer(4)]))
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(model=language))
        )

    def embed_prefix(self, embeddings, mask):
        attention = torch.zeros_like(mask, dtype=torch.bool)
        return embeddings, mask, attention


def test_vlm_tracer_pools_only_valid_final_layer_prefix_tokens():
    model = _Model()
    tracer = VLMFeatureTracer(model)
    embeddings = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [100.0] * 4]]
    )
    mask = torch.tensor([[True, True, False]])
    tracer.begin()
    prefix, _mask, _attention = model.embed_prefix(embeddings, mask)
    model.paligemma_with_expert.paligemma.language_model.model.layers[-1](
        hidden_states=prefix
    )
    feature = tracer.finish()
    np.testing.assert_allclose(feature.astype(np.float32), [[4.0, 6.0, 8.0, 10.0]])
    with pytest.raises(RuntimeError, match="exactly one"):
        tracer.finish()
    tracer.close()
