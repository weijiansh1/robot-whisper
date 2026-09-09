import numpy as np

from analyze_rich_event_features import extract_features, sign_flip_max_t


def _snapshot() -> dict[str, np.ndarray]:
    shape = (1, 8, 10, 11)
    arrays = {}
    for key in (
        "routed_authority",
        "expert_cancellation",
        "expert_disagreement_ratio",
        "routed_shared_cosine",
        "top4_top5_logit_margin",
        "tail_mass",
        "exec_entropy",
    ):
        arrays[key] = np.zeros(shape, np.float32)
    # State tokens must not contribute. Action tokens encode the layer index.
    arrays["routed_authority"][..., 0] = 1000.0
    arrays["routed_authority"][..., 1:] = np.arange(8)[None, :, None, None]
    sketch = np.zeros(shape + (32,), np.float32)
    sketch[..., 0] = np.arange(10)[None, None, :, None]
    arrays.update(
        routed_output_sketch=sketch,
        topk_idx=np.broadcast_to(
            np.arange(4, dtype=np.uint8), shape + (4,)
        ).copy(),
        topk_exec_weight=np.full(shape + (4,), 0.25, np.float32),
        hb_layers=np.asarray([2, 3, 4, 5, 12, 13, 14, 15], np.int16),
        n_denoise=np.asarray(10, np.int16),
    )
    return arrays


def test_extract_features_uses_late_layers_and_action_tokens():
    features, maps = extract_features(_snapshot())

    np.testing.assert_allclose(features["routed_authority"], 5.5)
    assert maps["routed_authority"].shape == (8, 10)
    assert maps["functional_flow_delta"].shape == (8, 10)
    assert np.isfinite(features["functional_flow_delta"])


def test_joint_sign_flip_handles_cell_specific_missing_pairs():
    differences = np.asarray(
        [
            [0.8, np.nan],
            [1.2, -0.5],
            [0.7, -0.9],
            [1.4, -0.4],
            [0.9, -1.1],
            [1.1, -0.7],
        ]
    )

    raw_a, max_t_a = sign_flip_max_t(
        differences, n_permutations=5000, seed=17
    )
    raw_b, max_t_b = sign_flip_max_t(
        differences, n_permutations=5000, seed=17
    )

    np.testing.assert_array_equal(raw_a, raw_b)
    np.testing.assert_array_equal(max_t_a, max_t_b)
    assert np.all((raw_a > 0) & (raw_a <= 1))
    assert np.all(max_t_a >= raw_a)
