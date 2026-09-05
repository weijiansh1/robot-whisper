from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from himoe_functional_recorder import (  # noqa: E402
    HBFunctionalSnapshotRecorder,
    save_functional_record,
)


class ToyGate(nn.Module):
    n_routed_experts = 2
    top_k = 2
    norm_topk_prob = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))

    def forward(self, hidden):
        count = hidden.shape[0] * hidden.shape[1]
        ids = torch.tensor([0, 1], device=hidden.device).expand(count, 2)
        weights = torch.full((count, 2), 0.5, device=hidden.device)
        return ids, weights, None


class Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return self.value * x


class HBMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ToyGate()
        self.experts = nn.ModuleList((Scale(2.0), Scale(-2.0)))
        self.shared_experts = Scale(0.25)
        self.num_experts_per_tok = 2

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        result = torch.zeros_like(x).float()
        order = flat_expert_indices.argsort()
        ends = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token = order // self.num_experts_per_tok
        for expert_id, end in enumerate(ends):
            start = 0 if expert_id == 0 else ends[expert_id - 1]
            if start == end:
                continue
            positions = order[start:end]
            selected = token[start:end]
            expert_out = self.experts[expert_id](x[selected]).float()
            expert_out.mul_(flat_expert_weights[positions])
            result.scatter_reduce_(
                0,
                selected[:, None].repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return result

    def forward(self, hidden, _mask=None):
        shape = hidden.shape
        ids, weights, _ = self.gate(hidden)
        routed = self.moe_infer(
            hidden.reshape(-1, shape[-1]), ids.reshape(-1), weights.reshape(-1, 1)
        ).reshape(shape)
        return routed + self.shared_experts(hidden)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = HBMoE()


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList((Layer(), Layer()))

    def forward(self, hidden):
        return [layer.mlp(hidden, None) for layer in self.layers]


def test_request_gated_capture_is_noop_and_reduces_one_dispatch(tmp_path):
    core = Core().eval()
    hidden = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [2.0, 4.0]]])
    reference = core(hidden)
    recorder = HBFunctionalSnapshotRecorder(
        core,
        expected_denoise=2,
        sketch_dim=4,
        expect_hb_blocks=2,
    ).attach()
    try:
        disabled = core(hidden)
        for got, expected in zip(disabled, reference, strict=True):
            assert torch.equal(got, expected)

        recorder.begin(episode_id=7, control_step=11)
        captured = []
        for _ in range(2):
            captured = core(hidden)
        record = recorder.end()
        for got, expected in zip(captured, reference, strict=True):
            assert torch.equal(got, expected)

        assert record.router_logits_centered.shape == (1, 2, 2, 3, 2)
        assert record.expert_contrib_norm.shape == (1, 2, 2, 3, 2)
        assert record.expert_contrib_sketch.shape == (1, 2, 2, 3, 2, 4)
        np.testing.assert_allclose(record.topk_exec_weight, 0.5)
        np.testing.assert_allclose(record.top4_top5_logit_margin, np.inf)
        np.testing.assert_allclose(record.tail_mass, 0.0)
        np.testing.assert_allclose(record.exec_entropy, 1.0)
        np.testing.assert_allclose(record.routed_output_norm, 0.0)
        np.testing.assert_allclose(record.routed_authority, 0.0)
        np.testing.assert_allclose(record.expert_cancellation, 1.0)
        expected_disagreement = np.array([648.0, 40.0, 80.0])
        np.testing.assert_allclose(
            record.expert_disagreement[0, 0],
            np.broadcast_to(expected_disagreement, (2, 3)),
        )
        np.testing.assert_allclose(record.expert_disagreement_ratio, 1.0)
        np.testing.assert_allclose(record.routed_output_sketch, 0.0)
        assert not core.layers[0].mlp.experts[0]._forward_hooks

        path = tmp_path / "snapshot.npz"
        save_functional_record(record, path)
        with np.load(path) as data:
            assert int(data["episode_id"]) == 7
            np.testing.assert_array_equal(data["hb_layers"], [0, 1])
    finally:
        recorder.close()

    assert "moe_infer" not in core.layers[0].mlp.__dict__
