from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from himoe_activation_recorder import HBActivationRMSRecorder
from himoe_hb5_intervention import (
    ARMS,
    HB5D0Intervention,
    matched_random_delta,
    select_drop_slots,
)
from himoe_intervention_protocol import OnlinePairAudit
from himoe_intervention_store import (
    STORE_NAME,
    ZarrHB5InterventionReader,
    ZarrHB5InterventionWriter,
    paired_mechanism_metrics,
    validate_paired_store,
)
from rollout_hb5_intervention import build_paired_requests
from test_activation_recorder import HBMoE


class PlainLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class InterventionToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [PlainLayer(nn.Identity()) for _ in range(5)] + [PlainLayer(HBMoE())]
        )

    def forward(self, hidden):
        return self.layers[5].mlp(hidden, None)


def _capture_arm(core, arm: str, rounds: int = 2):
    intervention = HB5D0Intervention()
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=2,
        expected_denoise=rounds,
        expect_hb_blocks=1,
        probe_layer=5,
        intervention=intervention,
    ).attach()
    outputs = []
    try:
        recorder.begin(
            episode_id=0,
            control_step=0,
            query_id=9,
            candidate_id=3,
            intervention_pair_id=17,
            intervention_draw_id=4,
            intervention_arm=arm,
        )
        for denoise in range(rounds):
            hidden = torch.tensor(
                [[[99.0, 101.0], [1.0 + denoise, 3.0], [2.0, 4.0 + denoise]]]
            )
            outputs.append(core(hidden).detach().clone())
        record = recorder.end()
    finally:
        recorder.close()
    return record, outputs


def test_baseline_is_strict_noop_and_intervention_is_hb5_d0_action_only():
    reference_core = InterventionToyCore().eval()
    intervention_core = copy.deepcopy(reference_core).eval()
    inputs = [
        torch.tensor([[[99.0, 101.0], [1.0 + d, 3.0], [2.0, 4.0 + d]]])
        for d in range(2)
    ]
    reference = [reference_core(value).detach().clone() for value in inputs]
    baseline, baseline_outputs = _capture_arm(intervention_core, "baseline")
    for actual, expected in zip(baseline_outputs, reference, strict=True):
        assert torch.equal(actual, expected)
    assert np.array_equal(
        baseline.hb_probe_original_routed,
        baseline.hb_probe_executed_routed,
    )
    assert np.count_nonzero(baseline.hb_probe_intervention_delta) == 0

    paired = {}
    for arm in ARMS:
        paired[arm] = _capture_arm(copy.deepcopy(reference_core).eval(), arm)
    for arm in ("drop", "random"):
        _record, outputs = paired[arm]
        # State suffix token is row zero and must never be touched.
        assert torch.equal(outputs[0][:, 0], reference[0][:, 0])
        assert not torch.equal(outputs[0][:, 1:], reference[0][:, 1:])
        # The independently supplied d1 input has no intervention.
        assert torch.equal(outputs[1], reference[1])
    for field in (
        "hb_probe_input_hidden",
        "hb_probe_selected_expert_raw",
        "hb_selected_expert_id",
        "hb_selected_expert_weight",
        "hb_probe_original_routed",
        "hb_probe_dropped_slot",
        "hb_probe_dropped_expert_id",
    ):
        expected = getattr(paired["baseline"][0], field)
        for arm in ("drop", "random"):
            np.testing.assert_array_equal(getattr(paired[arm][0], field), expected)


def test_drop_formula_and_random_control_match_all_three_invariants():
    core = InterventionToyCore().eval()
    records = {arm: _capture_arm(copy.deepcopy(core), arm)[0] for arm in ARMS}
    drop = records["drop"]
    random = records["random"]
    ids = drop.hb_selected_expert_id[:, 0, 0]
    weights = drop.hb_selected_expert_weight[:, 0, 0]
    raw = drop.hb_probe_selected_expert_raw.astype(np.float32)
    slot = drop.hb_probe_dropped_slot.astype(np.int64)
    selected = np.take_along_axis(raw, slot[..., None, None], axis=2).squeeze(2)
    selected_weight = np.take_along_axis(weights, slot[..., None], axis=2).squeeze(2)
    original = drop.hb_probe_original_routed
    expected_drop = (original - selected_weight[..., None] * selected) / (
        1.0 - selected_weight[..., None]
    )
    # fp16 storage of v_e is only used here for a storage-scale cross-check;
    # the runtime intervention itself used fp32 pre-gate outputs.
    np.testing.assert_allclose(drop.hb_probe_executed_routed, expected_drop, atol=2e-3)
    np.testing.assert_array_equal(
        drop.hb_probe_dropped_expert_id,
        np.take_along_axis(ids, slot[..., None], axis=2).squeeze(2),
    )
    drop_delta = drop.hb_probe_intervention_delta
    random_delta = random.hb_probe_intervention_delta
    np.testing.assert_allclose(
        np.linalg.norm(random_delta, axis=-1),
        np.linalg.norm(drop_delta, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.sum(random_delta * original, axis=-1),
        np.sum(drop_delta * original, axis=-1),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.linalg.norm(original + random_delta, axis=-1),
        np.linalg.norm(original + drop_delta, axis=-1),
        rtol=3e-5,
        atol=3e-5,
    )


def test_drop_slot_policy_library_distinguishes_weight_and_output_size():
    weights = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    raw = torch.tensor([[[[9.0], [3.0], [1.0], [6.0]]]])
    assert select_drop_slots(weights, raw, 1, 2, "min_weight").item() == 0
    assert select_drop_slots(weights, raw, 1, 2, "max_weight").item() == 3
    assert select_drop_slots(weights, raw, 1, 2, "min_raw_output_rms").item() == 2
    assert select_drop_slots(weights, raw, 1, 2, "max_raw_output_rms").item() == 0
    first = select_drop_slots(weights, raw, 1, 2, "categorical_weighted")
    second = select_drop_slots(weights, raw, 1, 2, "categorical_weighted")
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_zero_routed_random_control_still_matches_norm_deterministically():
    routed = torch.zeros(1, 2, 8)
    drop = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    one = matched_random_delta(routed, drop, pair_id=3, draw_id=7)
    two = matched_random_delta(routed, drop, pair_id=3, draw_id=7)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(one, dim=-1),
        torch.linalg.vector_norm(drop, dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_request_builder_reuses_exact_explicit_noise_without_aliasing():
    observation = {
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((2, 2, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "test",
    }
    noise = np.arange(240, dtype=np.float32).reshape(10, 24)
    requests = build_paired_requests(observation, noise, pair_id=5, draw_id=6)
    assert [request["intervention/arm"] for request in requests] == list(ARMS)
    assert all(np.array_equal(request["flow/noise"], noise) for request in requests)
    assert (
        len(
            {
                hashlib.sha256(request["flow/noise"].tobytes()).digest()
                for request in requests
            }
        )
        == 1
    )
    assert requests[0]["flow/noise"] is not requests[1]["flow/noise"]


def test_v4_store_roundtrip_and_pair_audits(tmp_path):
    core = InterventionToyCore().eval()
    records = {
        arm: _capture_arm(copy.deepcopy(core), arm, rounds=10)[0] for arm in ARMS
    }
    path = tmp_path / STORE_NAME
    observation_digest = hashlib.sha256(b"observation").digest()
    noise_digest = hashlib.sha256(b"noise").digest()
    trajectories = {}
    online = OnlinePairAudit()
    with ZarrHB5InterventionWriter(
        str(path),
        n_denoise=10,
        n_action_steps=2,
        max_action_dim=7,
        hidden_size=2,
        top_k=2,
        n_routed_experts=2,
        action_std=np.ones(7),
    ) as writer:
        for arm_index, arm in enumerate(ARMS):
            trajectory = np.zeros((1, 11, 2, 7), dtype=np.float32)
            trajectory[:, 1:] = arm_index
            trajectory[:, 10] = 2 * arm_index
            trajectories[arm] = trajectory
            online.prepare(17, 4, arm, observation_digest, noise_digest)
            online.commit(records[arm], trajectory, observation_digest, noise_digest)
            writer.append(
                records[arm],
                trajectory,
                observation_sha256=observation_digest,
                flow_noise_sha256=noise_digest,
            )
    online.require_complete()
    reader = ZarrHB5InterventionReader(str(path))
    audit = validate_paired_store(reader)
    assert audit["rows"] == 3
    assert audit["complete_pairs"] == 1
    assert audit["max_original_pair_difference"] == 0.0
    assert audit["max_x0_pair_difference"] == 0.0
    metrics = paired_mechanism_metrics(reader)
    assert {row["arm"] for row in metrics} == {"drop", "random"}
    by_arm = {row["arm"]: row for row in metrics}
    assert by_arm["drop"]["f_rem_rms_live7"] == pytest.approx(1.0)
    assert by_arm["random"]["f_rem_rms_live7"] == pytest.approx(2.0)


def test_pair_audit_rejects_changed_x0():
    core = InterventionToyCore().eval()
    baseline = _capture_arm(copy.deepcopy(core), "baseline")[0]
    drop = _capture_arm(copy.deepcopy(core), "drop")[0]
    digest = hashlib.sha256(b"same").digest()
    audit = OnlinePairAudit()
    x = np.zeros((1, 3, 2, 7), dtype=np.float32)
    audit.commit(baseline, x, digest, digest)
    changed = x.copy()
    changed[:, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="same x0"):
        audit.commit(drop, changed, digest, digest)
