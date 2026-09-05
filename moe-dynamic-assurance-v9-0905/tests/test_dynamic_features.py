from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dynamic"))

from features import (  # noqa: E402
    QUERY_FEATURE_NAMES,
    WITHIN_FEATURE_NAMES,
    compute_query_dynamics,
    compute_dynamic_history,
    compute_within_flow,
)


def uniform_router(rows: int) -> torch.Tensor:
    return torch.full((rows, 8, 10, 11, 32), 1.0 / 32.0)


def test_full_dynamic_schema_and_static_identities():
    result = compute_within_flow(uniform_router(3))
    assert result["features"].shape == (3, len(WITHIN_FEATURE_NAMES))
    assert result["final_root"].shape == (3, 8, 11, 32)
    names = list(WITHIN_FEATURE_NAMES)
    for prefix in ("flow_speed|", "flow_accel|", "flow_flux|", "expert_endpoint|", "expert_tv|"):
        columns = [index for index, name in enumerate(names) if name.startswith(prefix)]
        np.testing.assert_allclose(result["features"][:, columns], 0.0, atol=1e-7)
    turn = [index for index, name in enumerate(names) if name.startswith("flow_turn|")]
    np.testing.assert_allclose(result["features"][:, turn], 1.0, atol=1e-7)


def test_query_dynamics_are_prefix_local_and_static_zero():
    within = compute_within_flow(uniform_router(6))
    episode = np.zeros(6, dtype=np.int32)
    step = np.arange(6, dtype=np.int32)
    first = compute_query_dynamics(within["final_root"], within["final_graph"], episode, step)
    altered_root = within["final_root"].copy()
    altered_root[5] = 0.0
    altered_root[5, :, :, 0] = 1.0
    second = compute_query_dynamics(altered_root, within["final_graph"], episode, step)
    np.testing.assert_allclose(first[:5], second[:5], equal_nan=True)
    assert not np.allclose(first[5], second[5], equal_nan=True)
    assert first.shape == (6, len(QUERY_FEATURE_NAMES))
    finite = first[np.isfinite(first)]
    assert np.all(finite >= -1e-7)


def test_feature_name_inventory_is_unique():
    names = WITHIN_FEATURE_NAMES + QUERY_FEATURE_NAMES
    assert len(names) == len(set(names))
    assert len(names) > 2000


def test_online_history_wrapper_matches_batch_functions():
    router = uniform_router(4)
    direct = compute_within_flow(router)
    query = compute_query_dynamics(
        direct["final_root"],
        direct["final_graph"],
        np.zeros(4, dtype=np.int32),
        np.arange(4, dtype=np.int32),
    )
    online = compute_dynamic_history(router)
    np.testing.assert_allclose(online["within_features"], direct["features"])
    np.testing.assert_allclose(online["query_features"], query, equal_nan=True)
