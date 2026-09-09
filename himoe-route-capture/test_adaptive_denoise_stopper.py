import numpy as np
import pytest

from adaptive_denoise_stopper import (
    ConsecutiveRouteStopper,
    StopConfig,
    rms_hellinger_change,
    stopping_step,
)
from serve_adaptive_denoise import (
    deduplicated_count_change,
    deduplicated_jaccard_change,
    storage_compatible_action_route,
)


def test_rms_hellinger_is_zero_after_independent_layer_normalization():
    previous = np.array([[1.0, 3.0], [2.0, 2.0]])
    current = np.array([[2.0, 6.0], [5.0, 5.0]])
    assert rms_hellinger_change(previous, current) == pytest.approx(0.0)


def test_rms_hellinger_known_disjoint_routes():
    previous = np.array([[1.0, 0.0], [0.0, 1.0]])
    current = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert rms_hellinger_change(previous, current) == pytest.approx(1.0)
    assert rms_hellinger_change(previous, current) == pytest.approx(
        rms_hellinger_change(current, previous)
    )


def test_stopper_honors_minimum_and_consecutive_changes():
    config = StopConfig(threshold=0.01, min_steps=5, consecutive=2, max_steps=10)
    rule = ConsecutiveRouteStopper(config)
    assert not rule.observe(2, 0.005)
    assert not rule.observe(3, 0.020)
    assert not rule.observe(4, 0.005)
    assert rule.observe(5, 0.010)


def test_large_change_resets_streak():
    config = StopConfig(threshold=0.01, min_steps=3, consecutive=2, max_steps=4)
    assert stopping_step([0.005, 0.02, 0.005], config) == 4


def test_default_rule_replays_expected_step():
    config = StopConfig()
    changes = [0.0035, 0.0036, 0.0035, 0.0036, 0.0040, 0.0042, 0.0046, 0.008, 0.01]
    assert stopping_step(changes, config) == 8


def test_online_route_reduction_matches_stored_float16_definition():
    torch = pytest.importorskip("torch")
    source = torch.tensor(
        [
            [
                [0.101234, 0.198765, 0.300123, 0.399878],
                [0.401111, 0.299222, 0.199333, 0.100334],
            ]
        ],
        dtype=torch.float32,
    )
    actual = storage_compatible_action_route(source, n_action=2).numpy()
    stored = source.numpy().astype(np.float16).astype(np.float32)
    stored /= stored.sum(axis=-1, keepdims=True)
    expected = stored.mean(axis=1)
    expected /= expected.sum(axis=-1, keepdims=True)
    assert np.array_equal(actual, expected)


def test_deduplicated_jaccard_uses_identity_not_only_count():
    torch = pytest.importorskip("torch")
    previous = torch.tensor([[[True, True, False, False], [True, False, True, False]]])
    current = torch.tensor([[[True, False, True, False], [True, True, False, False]]])
    # Both layers keep two experts, but each swaps one: intersection=1, union=3.
    assert deduplicated_count_change(previous, current) == pytest.approx(0.0)
    assert deduplicated_jaccard_change(previous, current) == pytest.approx(2.0 / 3.0)


def test_deduplicated_jaccard_is_zero_for_identical_sets():
    torch = pytest.importorskip("torch")
    route = torch.tensor([[[True, False, True], [False, True, True]]])
    assert deduplicated_jaccard_change(route, route) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -1.0},
        {"min_steps": 1},
        {"min_steps": 11},
        {"consecutive": 0},
        {"consecutive": 10},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        StopConfig(**kwargs)
