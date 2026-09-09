"""Correctness checks for true HB-MoE amplitude and joint flow storage."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from himoe_activation_recorder import HBActivationRMSRecorder
from himoe_activation_store import ZarrActivationFlowReader, ZarrActivationFlowWriter


class ToyGate(nn.Module):
    n_routed_experts = 2
    top_k = 2
    norm_topk_prob = True

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, hidden_size))

    def forward(self, hidden):
        batch_tokens = hidden.shape[0] * hidden.shape[1]
        ids = torch.tensor([[0, 1], [1, 0]], device=hidden.device).repeat(
            (batch_tokens + 1) // 2, 1
        )[:batch_tokens]
        weights = torch.full((batch_tokens, 2), 0.5, device=hidden.device)
        return ids, weights, None


class ScaleExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return self.scale * x


class InconsistentToyGate(ToyGate):
    def forward(self, hidden):
        ids, _weights, aux = super().forward(hidden)
        weights = torch.tensor([0.75, 0.25], device=hidden.device).expand_as(ids)
        return ids, weights, aux


class HBMoE(nn.Module):
    """Small inference-only block with the upstream HBMoE dispatch contract."""

    def __init__(self):
        super().__init__()
        self.gate = ToyGate(hidden_size=2)
        self.experts = nn.ModuleList([ScaleExpert(2.0), ScaleExpert(-4.0)])
        self.shared_experts = ScaleExpert(0.25)
        self.num_experts_per_tok = 2

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        result = torch.zeros_like(x).float()
        order = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount(minlength=len(self.experts)).cumsum(0)
        token = order // self.num_experts_per_tok
        for expert_id, end in enumerate(counts):
            start = 0 if expert_id == 0 else counts[expert_id - 1]
            if start == end:
                continue
            positions = order[start:end]
            selected_token = token[start:end]
            expert_out = self.experts[expert_id](x[selected_token]).float()
            # Match upstream exactly: a forward-hook alias would be corrupted.
            expert_out.mul_(flat_expert_weights[positions])
            result.scatter_reduce_(
                0,
                selected_token[:, None].expand(-1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return result

    def forward(self, hidden, _data_mask=None):
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


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, hidden):
        # Feed the same input to both layers so the expected amplitude is simple.
        return [layer.mlp(hidden, None) for layer in self.layers]


def test_true_weighted_routed_rms_after_expert_cancellation():
    core = ToyCore().eval()
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=2,
        expected_denoise=3,
        expect_hb_blocks=2,
        probe_layer=0,
    ).attach()
    original_class_method = HBMoE.moe_infer
    try:
        recorder.begin(episode_id=7, control_step=11)
        inputs = []
        for denoise in range(3):
            hidden = torch.tensor(
                [[[99.0, 99.0], [1.0 + denoise, 3.0], [2.0, 4.0 + denoise]]]
            )
            inputs.append(hidden[:, 1:])
            core(hidden)
        record = recorder.end()

        routed_scale = torch.tensor([-1.0, -1.0]).reshape(1, 2, 1)
        expected = torch.stack(
            [
                (routed_scale * action).square().mean(-1).sqrt()
                for action in inputs
            ],
            dim=1,
        ).numpy()
        shared = torch.stack(
            [(0.25 * action).square().mean(-1).sqrt() for action in inputs],
            dim=1,
        ).numpy()
        total_scale = routed_scale + 0.25
        total = torch.stack(
            [
                (total_scale * action).square().mean(-1).sqrt()
                for action in inputs
            ],
            dim=1,
        ).numpy()
        assert record.hb_routed_rms.shape == (1, 2, 3, 2)
        np.testing.assert_allclose(record.hb_routed_rms[:, 0], expected, atol=1e-7)
        np.testing.assert_allclose(record.hb_routed_rms[:, 1], expected, atol=1e-7)
        np.testing.assert_allclose(record.hb_shared_rms[:, 0], shared, atol=1e-7)
        np.testing.assert_allclose(record.hb_total_mlp_rms[:, 0], total, atol=1e-7)
        np.testing.assert_allclose(record.hb_topk_weight_sum_error, 0.0, atol=0.0)
        raw_scale = torch.tensor([[4.0, 2.0], [2.0, 4.0]])
        raw = torch.stack(
            [
                action.square().mean(-1).sqrt().unsqueeze(-1) * raw_scale
                for action in inputs
            ],
            dim=1,
        ).numpy()
        np.testing.assert_allclose(record.hb_selected_expert_raw_rms[:, 0], raw)
        np.testing.assert_array_equal(
            record.hb_selected_expert_id[:, 0],
            np.broadcast_to(np.array([[1, 0], [0, 1]]), (1, 3, 2, 2)),
        )
        np.testing.assert_allclose(
            record.hb_selected_expert_weight[:, 0],
            np.broadcast_to(np.array([0.5, 0.5]), (1, 3, 2, 2)),
        )

        assert record.probe_hb_layer == 0
        assert record.probe_denoise == 0
        assert record.hb_probe_input_hidden.dtype == np.float16
        assert record.hb_probe_shared_output.dtype == np.float16
        assert record.hb_probe_selected_expert_raw.dtype == np.float16
        np.testing.assert_allclose(record.hb_probe_input_hidden, inputs[0])
        np.testing.assert_allclose(record.hb_probe_shared_output, inputs[0] * 0.25)
        ids_d0 = record.hb_selected_expert_id[:, 0, 0]
        raw_vector = record.hb_probe_selected_expert_raw.astype(np.float32)
        for token in range(2):
            for slot in range(2):
                expert_scale = 2.0 if ids_d0[0, token, slot] == 0 else -4.0
                np.testing.assert_allclose(
                    raw_vector[0, token, slot],
                    inputs[0][0, token].numpy() * expert_scale,
                )
        np.testing.assert_allclose(record.hb_probe_router_probs, 0.5)
        np.testing.assert_allclose(
            record.hb_probe_routed_reconstruction_max_abs_error, 0.0, atol=1e-7
        )

        # Summing individual magnitudes cannot reproduce cancellation.
        wrong_no_cancellation = 3.0 * inputs[0].square().mean(-1).sqrt().numpy()
        assert not np.allclose(record.hb_routed_rms[:, 0, 0], wrong_no_cancellation)
        # The huge state token is absent rather than averaged into the actions.
        assert float(record.hb_routed_rms.max()) < 20.0
    finally:
        recorder.close()

    # close removes the temporary instance wrapper and restores normal dispatch.
    assert "moe_infer" not in core.layers[0].mlp.__dict__
    assert HBMoE.moe_infer is original_class_method


def test_probe_rejects_router_probability_weight_mismatch():
    core = ToyCore().eval()
    core.layers[0].mlp.gate = InconsistentToyGate(hidden_size=2)
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=2,
        expected_denoise=1,
        expect_hb_blocks=2,
        probe_layer=0,
    ).attach()
    try:
        recorder.begin(episode_id=0, control_step=0)
        with pytest.raises(
            RuntimeError, match="full softmax does not reproduce top-k weights"
        ):
            core(torch.ones(1, 3, 2))
    finally:
        recorder.close()


def test_joint_activation_flow_store_roundtrip(tmp_path):
    core = ToyCore().eval()
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=2,
        expected_denoise=3,
        expect_hb_blocks=2,
        probe_layer=0,
    ).attach()
    try:
        recorder.begin(
            episode_id=5,
            control_step=9,
            query_id=41,
            candidate_id=3,
        )
        for denoise in range(3):
            core(torch.full((2, 3, 2), float(denoise + 1)))
        record = recorder.end()
    finally:
        recorder.close()

    trajectory = np.arange(2 * 4 * 2 * 5, dtype=np.float32).reshape(2, 4, 2, 5)
    path = tmp_path / "activation_flow.zarr"
    with ZarrActivationFlowWriter(
        str(path),
        hb_layers=[0, 1],
        n_denoise=3,
        n_action_steps=2,
        max_action_dim=5,
        action_std=np.ones(7),
        top_k=2,
        chunk_queries=8,
        probe_hb_layer=0,
        probe_hidden_size=2,
        n_routed_experts=2,
    ) as writer:
        invalid = copy.deepcopy(record)
        invalid.hb_probe_router_probs = invalid.hb_probe_router_probs[..., :1]
        with pytest.raises(ValueError, match="hb_probe_router_probs must have shape"):
            writer.append(invalid, trajectory)
        writer.append(record, trajectory)

    reader = ZarrActivationFlowReader(str(path))
    assert len(reader) == 2
    assert reader.meta["n_flow_states"] == 4
    assert reader.meta["suffix_state_token_stored"] is False
    assert reader.meta["probe_hb_layer"] == 0
    assert reader.meta["probe_denoise"] == 0
    assert reader.meta["probe_hidden_size"] == 2
    assert reader.meta["n_routed_experts"] == 2
    np.testing.assert_array_equal(reader.meta["normalization_action_std"], np.ones(7))
    np.testing.assert_array_equal(reader["x_traj"][:], trajectory)
    np.testing.assert_allclose(reader["hb_routed_rms"][:], record.hb_routed_rms)
    np.testing.assert_allclose(
        reader["hb_selected_expert_raw_rms"][:], record.hb_selected_expert_raw_rms
    )
    assert reader["hb_probe_input_hidden"].dtype == np.dtype("float16")
    assert reader["hb_probe_shared_output"].dtype == np.dtype("float16")
    assert reader["hb_probe_selected_expert_raw"].dtype == np.dtype("float16")
    np.testing.assert_array_equal(
        reader["hb_probe_input_hidden"][:], record.hb_probe_input_hidden
    )
    np.testing.assert_array_equal(
        reader["hb_probe_shared_output"][:], record.hb_probe_shared_output
    )
    np.testing.assert_array_equal(
        reader["hb_probe_selected_expert_raw"][:],
        record.hb_probe_selected_expert_raw,
    )
    np.testing.assert_array_equal(
        reader["hb_probe_router_probs"][:], record.hb_probe_router_probs
    )
    np.testing.assert_allclose(
        reader["hb_probe_routed_reconstruction_max_abs_error"][:],
        record.hb_probe_routed_reconstruction_max_abs_error,
    )
    np.testing.assert_array_equal(reader["episode_id"][:], [5, 5])
    np.testing.assert_array_equal(reader["control_step"][:], [9, 9])
    np.testing.assert_array_equal(reader["query_id"][:], [41, 41])
    np.testing.assert_array_equal(reader["candidate_id"][:], [3, 3])


def test_recorder_matches_upstream_hbmoe_dispatch():
    upstream = (
        Path(__file__).parents[1]
        / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA/src"
    )
    sys.path.insert(0, str(upstream))
    from moevla.models.himoe import HBMoEConfig
    from moevla.models.modeling_moe import HBMoE as UpstreamHBMoE

    config = HBMoEConfig()
    config.hidden_size = 4
    config.moe_intermediate_size = 8
    config.n_routed_experts = 3
    config.num_experts_per_tok = 2
    block = UpstreamHBMoE(config).eval()

    class RealLayer(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

    class RealCore(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.layers = nn.ModuleList(
                [RealLayer(nn.Identity()) for _ in range(5)] + [RealLayer(mlp)]
            )

        def forward(self, hidden):
            return self.layers[5].mlp(hidden, None)

    core = RealCore(block)
    generator = torch.Generator().manual_seed(17)
    inputs = [torch.randn(1, 3, 4, generator=generator) + i for i in range(2)]
    references = []
    with torch.no_grad():
        for hidden in inputs:
            ids, actual_weights, _ = block.gate(hidden)
            flat = hidden.reshape(-1, hidden.shape[-1])
            all_outputs = torch.stack(
                [expert(flat) for expert in block.experts], dim=1
            )
            selected = all_outputs.gather(
                1, ids.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            routed = (selected * actual_weights.unsqueeze(-1)).sum(dim=1)
            router_probs = torch.nn.functional.linear(
                flat, block.gate.weight, None
            ).softmax(dim=-1)
            references.append(
                {
                    "hidden": hidden[:, 1:],
                    "ids": ids.reshape(1, 3, 2)[:, 1:],
                    "weights": actual_weights.reshape(1, 3, 2)[:, 1:],
                    "raw": selected.reshape(1, 3, 2, 4)[:, 1:],
                    "routed": routed.reshape_as(hidden)[:, 1:],
                    "shared": block.shared_experts(hidden)[:, 1:],
                    "router_probs": router_probs.reshape(1, 3, 3)[:, 1:],
                    "total": (
                        routed.reshape_as(hidden) + block.shared_experts(hidden)
                    ),
                }
            )

    recorder = HBActivationRMSRecorder(
        core, n_action_steps=2, expected_denoise=2, expect_hb_blocks=1
    ).attach()
    captured_outputs = []
    try:
        recorder.begin(episode_id=1, control_step=2)
        for hidden in inputs:
            captured_outputs.append(core(hidden))
        record = recorder.end()
    finally:
        recorder.close()

    expected = torch.stack(
        [reference["routed"].square().mean(-1).sqrt() for reference in references],
        dim=1,
    ).numpy()
    np.testing.assert_allclose(
        record.hb_routed_rms[:, 0], expected, rtol=1e-6, atol=1e-7
    )
    assert float(record.hb_topk_weight_sum_error.max()) < 1e-6
    np.testing.assert_array_equal(
        record.hb_selected_expert_id[:, 0, 0], references[0]["ids"].numpy()
    )
    np.testing.assert_allclose(
        record.hb_selected_expert_weight[:, 0, 0],
        references[0]["weights"].numpy(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record.hb_selected_expert_raw_rms[:, 0, 0],
        references[0]["raw"].square().mean(-1).sqrt().numpy(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record.hb_probe_input_hidden,
        references[0]["hidden"].numpy(),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        record.hb_probe_shared_output,
        references[0]["shared"].numpy(),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        record.hb_probe_selected_expert_raw,
        references[0]["raw"].numpy(),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        record.hb_probe_router_probs,
        references[0]["router_probs"].numpy(),
        rtol=5e-4,
        atol=5e-4,
    )
    weighted_raw = (
        references[0]["raw"] * references[0]["weights"].unsqueeze(-1)
    )
    assert not np.allclose(
        record.hb_probe_selected_expert_raw.astype(np.float32),
        weighted_raw.numpy(),
    )
    np.testing.assert_allclose(
        record.hb_probe_routed_reconstruction_max_abs_error, 0.0, atol=1e-6
    )
    for output, reference in zip(captured_outputs, references, strict=True):
        np.testing.assert_allclose(
            output.detach().numpy(), reference["total"].numpy(), rtol=1e-6, atol=1e-7
        )
