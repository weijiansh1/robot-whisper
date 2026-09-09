from __future__ import annotations

import copy
import math
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import zarr

from himoe_first_output_recorder import (
    FirstControlIdentity,
    FirstControlOutputRecorder,
    flow_noise_digest,
    seeded_flow_noise_digest,
)
from himoe_first_output_store import EpisodePlan, ZarrFirstOutputWriter


class MoEGate_load_bal(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.n_routed_experts = 32
        self.top_k = 4
        self.weight = nn.Parameter(torch.zeros(32, hidden))

    def forward(self, hidden):
        n = hidden.shape[0] * hidden.shape[1]
        ids = torch.arange(4, device=hidden.device).repeat(n, 1)
        weights = torch.full((n, 4), 0.25, device=hidden.device)
        return ids, weights, None


class MoEGate_mutual_info(nn.Module):
    def __init__(self, selected: int):
        super().__init__()
        self.n_routed_experts = 3
        self.top_k = 1
        self.selected = selected
        # The AS gate really reads a 24-dim data_mask, not the block hidden state.
        self.weight = nn.Parameter(torch.zeros(3, 24))

    def forward(self, data_mask):
        n = data_mask.shape[0] * data_mask.shape[1]
        ids = torch.full((n, 1), self.selected, device=data_mask.device)
        weights = torch.ones((n, 1), device=data_mask.device)
        return ids, weights, None


class Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden):
        return hidden * self.value


class HBMoE(nn.Module):
    def __init__(self, layer: int, hidden: int):
        super().__init__()
        self.gate = MoEGate_load_bal(hidden)
        self.shared_experts = Scale(0.5)
        self.routed_scale = 0.1 + layer / 100.0

    def forward(self, hidden, _data_mask):
        self.gate(hidden)
        routed = hidden * self.routed_scale
        return routed + self.shared_experts(hidden)


class ASMoE(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.gate = MoEGate_mutual_info(selected=layer % 3)
        self.scale = 1.0 + layer / 10.0

    def forward(self, hidden, data_mask):
        self.gate(data_mask)
        return hidden * self.scale


class Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class FakeCore(nn.Module):
    def __init__(self, hidden: int = 6):
        super().__init__()
        moe = {0: ASMoE(0), 1: ASMoE(1), 16: ASMoE(16), 17: ASMoE(17)}
        moe.update({i: HBMoE(i, hidden) for i in (2, 3, 4, 5, 12, 13, 14, 15)})
        self.layers = nn.ModuleList(
            [Layer(moe.get(i, nn.Identity())) for i in range(18)]
        )


def identity(episode: int, flow_seed: int, init_state: int = 7):
    return FirstControlIdentity(
        episode_id=episode,
        task_id=5,
        init_state_id=init_state,
        flow_seed=flow_seed,
        control_step=0,
        flow_noise_sha256=bytes([episode + 1]) * 32,
    )


class FirstOutputTest(unittest.TestCase):
    def setUp(self):
        self.core = FakeCore()
        self.recorder = FirstControlOutputRecorder(
            self.core, n_action_steps=2, expected_denoise=2
        ).attach()

    def tearDown(self):
        self.recorder.close()

    def capture(self, episode: int = 0):
        self.recorder.begin(identity(episode, 1000 + episode))
        for denoise in range(2):
            hidden = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
            hidden = hidden + 1 + denoise
            data_mask = torch.ones(1, 3, 24)
            for layer in self.core.layers:
                if isinstance(layer.mlp, (ASMoE, HBMoE)):
                    layer.mlp(hidden, data_mask)
        return self.recorder.end()

    def test_as_gate_width_and_block_output_are_distinct(self):
        record = self.capture()
        self.assertEqual(record.as_output.shape, (1, 2, 2, 2, 6))
        self.assertEqual(record.as_expert_ids.shape, (1, 2, 2, 2))
        np.testing.assert_array_equal(record.as_expert_ids[:, 0], 0)
        np.testing.assert_array_equal(record.as_expert_ids[:, 1], 1)
        # Action token 1 from AS0 is the actual weighted selected-expert output.
        expected = np.arange(6, 12, dtype=np.float32) + 1
        np.testing.assert_allclose(record.as_output[0, 0, 0, 0], expected)

    def test_hb_branch_decomposition_and_angle(self):
        record = self.capture()
        self.assertEqual(record.hb_post_output.shape, (1, 8, 2, 2, 6))
        np.testing.assert_allclose(record.hb_branch_cosine, 1.0, atol=1e-6)
        # acos amplifies float32 cosine rounding near exactly parallel vectors.
        np.testing.assert_allclose(record.hb_branch_angle_deg, 0.0, atol=0.03)
        first_action = np.arange(6, 12, dtype=np.float32) + 1
        routed = first_action * 0.12  # HB layer 2
        shared = first_action * 0.5
        np.testing.assert_allclose(
            record.hb_post_output[0, 0, 0, 0], routed + shared, rtol=2e-3
        )

    def test_exact_k_group_reduction(self):
        template = self.capture()
        records = []
        for candidate, value in enumerate((1.0, 2.0, 3.0)):
            record = copy.deepcopy(template)
            record.identity = identity(candidate, 1000 + candidate)
            record.hb_post_output.fill(0)
            record.hb_post_output[..., 0] = value
            records.append(record)

        with tempfile.TemporaryDirectory() as temp:
            path = temp + "/outputs.zarr"
            writer = ZarrFirstOutputWriter(
                path,
                expected_k=3,
                task_id=5,
                task_name="synthetic",
                suite="goal",
                benchmark="libero_goal",
                hb_layers=template.hb_layers.tolist(),
                as_layers=template.as_layers.tolist(),
                n_denoise=2,
                n_action_steps=2,
                hidden_size=6,
            )
            for record in records:
                writer.append(record)
            writer.close()
            root = zarr.open_group(path, mode="r")
            self.assertEqual(root["episode_id"].shape, (3,))
            self.assertEqual(root["group_size"][0], 3)
            self.assertEqual(root["group_complete"][0], 1)
            expected_dispersion = math.sqrt(2.0 / 3.0)
            expected_relative = math.sqrt(2.0 / 14.0)
            np.testing.assert_allclose(
                root["hb_post_rms_dispersion"][0], expected_dispersion, rtol=1e-6
            )
            np.testing.assert_allclose(
                root["hb_post_relative_dispersion"][0], expected_relative, rtol=1e-6
            )
            np.testing.assert_allclose(root["hb_post_mean_pair_cosine"][0], 1.0)
            np.testing.assert_allclose(root["hb_post_contraction_from_d0"][0], 0.0)
            np.testing.assert_allclose(
                root["hb_post_distance_to_group_mean"][:, 0, 0, 0], [1, 0, 1]
            )
            np.testing.assert_array_equal(root["action_token_id"][:], [1, 2])

    def test_episode_plan_matches_right16_order(self):
        plan = EpisodePlan((0, 3), draws=32, noise_seed_base=1000)
        self.assertEqual(plan.decode(0), (0, 1000, 0))
        self.assertEqual(plan.decode(31), (0, 1031, 31))
        self.assertEqual(plan.decode(32), (3, 1000, 0))

    def test_seeded_flow_noise_digest_detects_plan_mismatch(self):
        shape = (2, 4)
        actual = np.random.default_rng(1000).standard_normal(shape).astype(np.float32)
        self.assertEqual(flow_noise_digest(actual), seeded_flow_noise_digest(1000, shape))
        self.assertNotEqual(flow_noise_digest(actual), seeded_flow_noise_digest(1001, shape))


if __name__ == "__main__":
    unittest.main()
