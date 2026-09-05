from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score_module = load_module("trainfree_score_heads", HERE / "score_heads.py")
evaluation_module = load_module("trainfree_evaluate_heads", HERE / "evaluate_heads.py")


def test_pool_rank_is_group_local_and_supports_both_directions() -> None:
    value = np.array([[1.0], [3.0], [10.0], [20.0]])
    group = np.array([0, 0, 1, 1])
    valid = np.ones_like(value, bool)
    high = score_module.pool_rank(value, group, valid, high=True)
    low = score_module.pool_rank(value, group, valid, high=False)
    assert np.array_equal(high[:, 0], [0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(low[:, 0], [1.0, 0.0, 1.0, 0.0])


def test_pool_rank_ties_are_average_rank() -> None:
    value = np.array([[1.0], [1.0], [3.0]])
    rank = score_module.pool_rank(
        value, np.zeros(3, np.int64), np.ones_like(value, bool)
    )
    assert np.allclose(rank[:, 0], [0.25, 0.25, 1.0])


def test_auc_handles_order_and_ties() -> None:
    target = np.array([False, False, True, True])
    assert evaluation_module.binary_auc(target, np.array([0, 1, 2, 3])) == 1.0
    assert evaluation_module.binary_auc(target, np.ones(4)) == 0.5


def test_topk_tie_break_is_episode_stable() -> None:
    indices = np.array([2, 0, 1])
    scores = np.array([0.5, 0.5, 0.5])
    episode = np.array([10, 11, 12])
    selected = score_module.stable_topk(indices, scores, episode, 2)
    assert np.array_equal(selected, [0, 1])


def test_frozen_score_manifest_is_label_blind() -> None:
    manifest = json.loads((HERE / "results/score_manifest.json").read_text())
    protocol_hash = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    assert manifest["train_free"] is True
    assert manifest["fit_calls"] == 0
    assert manifest["calibrated_parameters"] == 0
    assert manifest["label_columns_read"] == []
    assert manifest["protocol_sha256"] == protocol_hash
