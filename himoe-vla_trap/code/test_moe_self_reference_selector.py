#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moe_self_reference_selector import (
    SCHEMA,
    SelfReferenceConfig,
    SelfReferenceCouplingCollapseAlarm,
)


def _route(scale: float, state_expert: int, action_expert: int) -> np.ndarray:
    route = np.full((8, 10, 11, 32), 1e-4, dtype=np.float32)
    route[..., 0] = 1.0
    for flow in range(10):
        route[4:, flow, 1:, 1] = scale * (1.0 if flow % 2 else 0.1)
    route[:4, 9, 0, :] = 1e-4
    route[:4, 9, 0, state_expert] = 1.0
    route[:4, 9, 1:, :] = 1e-4
    route[:4, 9, 1:, action_expert] = 1.0
    return route / route.sum(axis=-1, keepdims=True)


def test_config_rejects_hidden_calibration(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "schema": SCHEMA,
        "training": False,
        "learned_parameters": False,
        "runtime_normal_trajectory_reference_used": True,
        "runtime_outcome_labels_used": False,
        "runtime_physical_state_used": False,
        "runtime_action_values_used": False,
        "runtime_task_identity_used": False,
        "warmup_transitions": 2,
        "window": 2,
        "state_response_ratio_max": 0.25,
        "planning_churn_ratio_min": 1.1,
        "state_action_gap_ratio_min": 1.2,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        SelfReferenceConfig.load(path)
    except RuntimeError as error:
        assert "runtime_normal_trajectory_reference_used" in str(error)
    else:
        raise AssertionError("normal-reference config was accepted")


def test_alarm_requires_all_three_effects() -> None:
    config = SelfReferenceConfig(2, 2, 0.25, 1.1, 1.2)
    alarm = SelfReferenceCouplingCollapseAlarm(config)
    sequence = [
        _route(0.3, 0, 0),
        _route(0.4, 1, 1),
        _route(0.5, 2, 2),
        _route(1.2, 2, 3),
        _route(1.2, 2, 3),
    ]
    decisions = [alarm.update(route) for route in sequence]
    assert not any(item.alarm for item in decisions[:-1])
    assert decisions[-1].state_response_ratio <= config.state_response_ratio_max
    assert decisions[-1].planning_churn_ratio >= config.planning_churn_ratio_min
    assert decisions[-1].state_action_gap_ratio >= config.state_action_gap_ratio_min
    assert decisions[-1].alarm
