from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))

from analyze_activation_flow_stop import (
    STAGES,
    ActivationFlowData,
    ControllerErrors,
    Signal,
    apply_frozen_rule,
    build_activation_signals,
    controller_errors,
    endpoint_predictions,
    fit_frozen_rule,
    fixed_stage_rows,
    loeo_sweep,
    load_activation_flow,
    select_champion,
)
from serve_activation_flow_trace import request_control_step


def synthetic_data(rows: int = 4) -> ActivationFlowData:
    rng = np.random.default_rng(3)
    origin = rng.normal(size=(rows, 2, 7))
    velocity = rng.normal(size=(rows, 2, 7)) * 0.02
    time = np.arange(11, dtype=np.float64)
    x = origin[:, None] + time[None, :, None, None] * velocity[:, None]
    rounds = np.linspace(1.0, 0.4, 10)
    activation = np.broadcast_to(rounds[None, None, :, None], (rows, 2, 10, 2)).copy()
    activation[1] *= 10.0
    return ActivationFlowData(
        name="synthetic",
        store=Path("synthetic.zarr"),
        x=x,
        routed=activation,
        shared=activation * 2.0,
        total=activation * 2.5,
        rollout_id=np.arange(rows),
        control_step=np.arange(rows),
        action_std=np.ones(7),
        hb_layers=np.asarray([2, 3], dtype=np.int16),
        topk_weight_sum_error_max=0.0,
    )


def test_linear_flow_endpoint_estimators_are_exact():
    data = synthetic_data()
    final = data.x[:, 10]
    for prediction in endpoint_predictions(data.x).values():
        expected = np.broadcast_to(final[:, None], prediction.shape)
        np.testing.assert_allclose(prediction, expected, atol=1e-12)


def test_query_normalization_removes_scale_and_veto_detects_rebound():
    data = synthetic_data()
    routed = data.routed.copy()
    routed[0, 1, 8] = 1.2
    data = dataclasses_replace(data, routed=routed)
    signals = {signal.name: signal.values for signal in build_activation_signals(data)}
    # Query 1 is exactly ten times query 0 before the injected rebound.
    assert signals["routed_level_mean"][1, 3] > 9 * signals["routed_level_mean"][0, 3]
    np.testing.assert_allclose(
        signals["routed_first_ratio_mean"][0, 3],
        signals["routed_first_ratio_mean"][1, 3],
        atol=1e-12,
    )
    assert signals["routed_all_layer_rebound_max"][0, -1] > 0.5


def dataclasses_replace(data: ActivationFlowData, **changes) -> ActivationFlowData:
    values = {field: getattr(data, field) for field in data.__dataclass_fields__}
    values.update(changes)
    return ActivationFlowData(**values)


def test_controller_metric_treats_gripper_only_by_sign():
    data = synthetic_data(rows=2)
    prediction = np.broadcast_to(data.x[:, None, 10], (2, len(STAGES), 2, 7)).copy()
    prediction[0, 0, 0, 6] *= -1.0
    prediction[1, 0, :, 6] *= 2.0
    prediction[1, 1, 0, 6] = 0.0
    errors = controller_errors(data.x, prediction, data.action_std)
    assert errors.continuous_rms[0, 0] == 0.0
    assert bool(errors.gripper_flip[0, 0])
    assert not bool(errors.gripper_flip[1, 0])
    assert bool(errors.gripper_flip[1, 1])
    assert bool(errors.unsafe(0.01)[0, 0])


def test_loeo_and_frozen_rule_leave_whole_rollouts_out():
    data = synthetic_data(rows=8)
    data = dataclasses_replace(data, rollout_id=np.repeat(np.arange(4), 2))
    safe = np.tile([True, False], 4)
    values = np.where(safe[:, None], 0.1, 0.9) * np.ones((8, len(STAGES)))
    signal = Signal("known_safe", "synthetic", values)
    continuous = np.where(safe[:, None], 0.0, 1.0) * np.ones((8, len(STAGES)))
    errors = ControllerErrors(
        continuous_rms=continuous,
        translation_rms=continuous,
        rotation_rms=continuous,
        gripper_flip=np.zeros_like(continuous, dtype=bool),
    )
    rows = loeo_sweep(
        data,
        [signal],
        {"constant_velocity": errors, "constant_acceleration": errors},
        tolerance=0.1,
        max_train_risk=0.0,
        min_stages=(8,),
    )
    assert len(rows) == 2
    assert all(row["unsafe_rate"] == 0.0 for row in rows)
    assert all(abs(row["saving"] - 0.1) < 1e-12 for row in rows)
    champion = select_champion(rows, accept_risk=0.01)
    rule = fit_frozen_rule(
        data,
        [signal],
        {"constant_velocity": errors, "constant_acceleration": errors},
        champion,
        tolerance=0.1,
        max_train_risk=0.0,
    )
    held = apply_frozen_rule(
        data,
        [signal],
        {"constant_velocity": errors, "constant_acceleration": errors},
        rule,
        tolerance=0.1,
    )
    assert held["unsafe_rate"] == 0.0
    assert abs(held["saving"] - 0.1) < 1e-12


def test_control_step_restarts_inside_each_episode():
    counters = {}
    assert request_control_step(3, None, 20, counters) == 0
    assert request_control_step(3, None, 21, counters) == 1
    assert request_control_step(4, None, 22, counters) == 0
    assert request_control_step(-1, None, 23, counters) == 23
    assert request_control_step(-1, 7, 24, counters) == 7


def test_loader_accepts_v2_rms_store(tmp_path):
    path = tmp_path / "activation_flow.zarr"
    root = zarr.create_group(store=str(path), overwrite=True)
    root.attrs.update(
        {
            "format": "himoe_hb_activation_flow_v2",
            "hb_layers": [2, 5],
            "normalization_action_std": np.ones(7).tolist(),
        }
    )
    arrays = {
        "x_traj": np.zeros((1, 11, 2, 7), dtype=np.float32),
        "hb_routed_rms": np.ones((1, 2, 10, 2), dtype=np.float32),
        "hb_shared_rms": np.ones((1, 2, 10, 2), dtype=np.float32),
        "hb_total_mlp_rms": np.ones((1, 2, 10, 2), dtype=np.float32),
        "hb_topk_weight_sum_error": np.zeros(
            (1, 2, 10, 2), dtype=np.float32
        ),
        "episode_id": np.array([4], dtype=np.int32),
        "control_step": np.array([0], dtype=np.int32),
    }
    for name, value in arrays.items():
        root.create_array(name, data=value)

    data = load_activation_flow(path, require_loeo=False)
    assert data.rows == 1
    np.testing.assert_array_equal(data.hb_layers, [2, 5])


def test_fixed_stage_rows_include_full_ten_round_baseline():
    data = synthetic_data()
    predictions = endpoint_predictions(data.x)
    errors = {
        name: controller_errors(data.x, value, data.action_std)
        for name, value in predictions.items()
    }
    rows = fixed_stage_rows(data, errors, tolerance=0.01)
    full = [row for row in rows if row["stage"] == 10]
    assert len(full) == 2
    assert all(row["saving"] == 0.0 and row["unsafe_rate"] == 0.0 for row in full)
