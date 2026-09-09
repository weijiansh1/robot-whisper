import numpy as np
import torch

from bestofn_critic import (
    DuelingChunkCritic,
    aggregate_metrics,
    critic_loss,
    hierarchical_delta_ci,
    snapshot_metrics,
)


def _batch():
    torch.manual_seed(4)
    return {
        "state_vlm": torch.randn(2, 12),
        "state_proprio": torch.randn(2, 8),
        "action": torch.randn(2, 4, 7),
        "route_candidate": torch.randn(2, 4, 9),
        "route_state": torch.randn(2, 6),
        "hidden_candidate": torch.randn(2, 4, 3, 2, 5),
        "hidden_state": torch.randn(2, 3, 2, 5),
        "mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
    }


def _model(mode):
    return DuelingChunkCritic(
        mode=mode,
        vlm_dim=12,
        action_dim=7,
        route_candidate_dim=9,
        route_state_dim=6,
        hidden_candidate_shape=(3, 2, 5),
        hidden_state_shape=(3, 2, 5),
        width=32,
        latent=16,
        hidden_projection=3,
    )


def test_all_four_primary_critics_preserve_snapshot_axis():
    batch = _batch()
    for mode in (
        "state_action",
        "state_route",
        "state_action_route",
        "state_action_router_hidden",
    ):
        logits = _model(mode)(batch)
        assert logits.shape == (2, 4)
        assert torch.all(logits[1, 2:] == 0)


def test_dueling_advantage_is_centered_within_valid_candidates():
    batch = _batch()
    model = _model("state_action_route")
    logits = model(batch)
    assert torch.isfinite(logits).all()
    # The centered advantage makes each set mean equal to its value branch;
    # duplicating a mask-only padded slot must not alter valid outputs.
    altered = {key: value.clone() for key, value in batch.items()}
    altered["action"][1, 2:] = 1e6
    altered["route_candidate"][1, 2:] = -1e6
    assert torch.allclose(logits[1, :2], model(altered)[1, :2], atol=1e-5)


def test_joint_value_and_pairwise_loss_is_finite():
    logits = _model("state_action")(_batch())
    success = torch.tensor([[0, 1, 3, 4], [2, 0, 0, 0]])
    repeats = torch.full((2, 4), 4)
    loss, parts = critic_loss(
        logits, success, repeats, _batch()["mask"], rank_weight=1.0
    )
    assert torch.isfinite(loss)
    assert parts["pairs"] > 0


def test_snapshot_metrics_and_hierarchical_delta_are_snapshot_based():
    offsets = np.asarray([0, 3, 6])
    success = np.asarray([0, 2, 4, 4, 2, 0])
    repeats = np.full(6, 4)
    task = np.asarray([0, 1])
    good = snapshot_metrics(
        np.asarray([-2, 0, 2, 2, 0, -2]), success, repeats, offsets, task, [0, 1]
    )
    bad = snapshot_metrics(
        np.asarray([2, 0, -2, -2, 0, 2]), success, repeats, offsets, task, [0, 1]
    )
    assert aggregate_metrics(good)["pairwise_accuracy"] == 1.0
    point, interval = hierarchical_delta_ci(
        good,
        bad,
        field="top1_regret",
        draws=200,
        rng=np.random.default_rng(9),
    )
    assert point < 0
    assert interval[1] < 0
