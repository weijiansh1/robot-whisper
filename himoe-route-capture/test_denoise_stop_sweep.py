from __future__ import annotations

import numpy as np

from analyze_denoise_stop_sweep import (
    STAGES,
    TOTAL_ROUNDS,
    TraceData,
    _selected_error,
    _stops_for_threshold,
    apply_endpoint_coefficients,
    build_signals,
    calibrate_threshold,
    endpoint_error,
    endpoint_predictions,
    fit_endpoint_coefficients,
    oof_endpoint_predictions,
)


def _linear_trace(rows: int = 3) -> TraceData:
    rng = np.random.default_rng(4)
    origin = rng.normal(size=(rows, 10, 7))
    velocity = rng.normal(size=(rows, 10, 7)) * 0.1
    time = np.arange(TOTAL_ROUNDS + 1, dtype=np.float64)
    x = origin[:, None] + time[None, :, None, None] * velocity[:, None]
    probability = np.full((rows, 8, TOTAL_ROUNDS, 10, 32), 1.0 / 32.0)
    ids = np.broadcast_to(
        np.arange(4, dtype=np.int16), (rows, 8, TOTAL_ROUNDS, 10, 4)
    ).copy()
    selected = np.full_like(ids, 1.0 / 32.0, dtype=np.float64)
    return TraceData(
        name="linear",
        run=None,  # type: ignore[arg-type]
        x=x,
        probability=probability,
        state_probability=probability[:, :, :, :1],
        expert_ids=ids,
        state_expert_ids=ids[:, :, :, :1],
        selected_probability=selected,
        query_id=np.arange(rows),
        candidate_id=np.zeros(rows, dtype=np.int64),
        rollout_id=np.arange(rows),
        action_std=np.ones(7),
    )


def test_constant_velocity_is_exact_on_linear_flow() -> None:
    data = _linear_trace()
    predictions = endpoint_predictions(data)
    final = data.x[:, TOTAL_ROUNDS]
    for name in (
        "constant_velocity",
        "mean_last2_velocity",
        "constant_acceleration",
        "linear_fit_3",
        "linear_fit_4",
        "quadratic_fit_3",
    ):
        expected = np.broadcast_to(final[:, None], predictions[name].shape)
        np.testing.assert_allclose(predictions[name], expected, atol=1e-12)
        np.testing.assert_allclose(endpoint_error(predictions[name], final), 0.0, atol=1e-12)


def test_self_supervised_endpoint_heads_recover_linear_flow() -> None:
    data = _linear_trace(rows=4)
    coefficients = fit_endpoint_coefficients(data, np.ones(data.rows, dtype=bool))
    predictions = apply_endpoint_coefficients(
        data, np.ones(data.rows, dtype=bool), coefficients
    )
    final = data.x[:, TOTAL_ROUNDS]
    for prediction in predictions.values():
        expected = np.broadcast_to(final[:, None], prediction.shape)
        np.testing.assert_allclose(prediction, expected, atol=4e-6)
    for prediction in oof_endpoint_predictions(data).values():
        expected = np.broadcast_to(final[:, None], prediction.shape)
        np.testing.assert_allclose(prediction, expected, atol=4e-6)


def test_threshold_stops_at_first_eligible_stage() -> None:
    values = np.asarray(
        [
            [0.9, 0.8, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        ]
    )
    stops = _stops_for_threshold(values, threshold=0.35, direction="le", min_stage=4)
    np.testing.assert_array_equal(stops, [4, 8])
    reverse = _stops_for_threshold(values, threshold=0.75, direction="ge", min_stage=6)
    np.testing.assert_array_equal(reverse, [10, 10])


def test_zero_risk_calibration_can_fall_back_to_full_rounds() -> None:
    values = np.tile(np.linspace(0.0, 1.0, len(STAGES)), (6, 1))
    errors = np.full_like(values, 0.2)
    threshold, stats = calibrate_threshold(
        values,
        errors,
        tolerance=0.05,
        max_risk=0.0,
        direction="le",
        min_stage=4,
    )
    stops = _stops_for_threshold(values, threshold, "le", 4)
    selected = _selected_error(errors, stops)
    np.testing.assert_array_equal(stops, np.full(6, TOTAL_ROUNDS))
    np.testing.assert_allclose(selected, 0.0)
    assert stats["saving"] == 0.0
    assert stats["risk"] == 0.0


def test_static_routes_have_zero_change_signals() -> None:
    data = _linear_trace()
    signals = build_signals(data)
    by_name = {signal.name: signal.values for signal in signals}
    for name in (
        "prob_mean_hellinger",
        "prob_site_tv_mean",
        "top1_switch_mean",
        "top4_token_jaccard_mean",
        "dedup_set_jaccard_mean",
        "dedup_count_delta_mean",
        "state_token_hellinger",
        "state_token_top1_switch",
        "as_route_constant",
    ):
        np.testing.assert_allclose(by_name[name], 0.0, atol=1e-12)
