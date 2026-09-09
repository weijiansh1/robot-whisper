import json

import numpy as np
import torch

from build_bestofn_critic_dataset import DATASET_SCHEMA
from train_bestofn_critics import CriticDataset, _make_model, train_member


def _write(path, name, value):
    np.save(path / f"{name}.npy", np.asarray(value), allow_pickle=False)


def _dataset(tmp_path):
    rng = np.random.default_rng(12)
    offsets = np.asarray([0, 3, 6, 9, 12], dtype=np.int64)
    _write(tmp_path, "state_offsets", offsets)
    _write(tmp_path, "task_id", [0, 0, 1, 1])
    _write(tmp_path, "split", [0, 0, 1, 2])
    _write(tmp_path, "state_vlm", rng.normal(size=(4, 10)).astype(np.float16))
    _write(tmp_path, "state_proprio", rng.normal(size=(4, 8)).astype(np.float32))
    _write(tmp_path, "action", rng.normal(size=(12, 7)).astype(np.float32))
    _write(tmp_path, "success_count", [0, 2, 4] * 4)
    _write(tmp_path, "repeat_count", np.full(12, 4))
    _write(tmp_path, "route_candidate_all", rng.normal(size=(12, 9)).astype(np.float16))
    _write(tmp_path, "route_state_all", rng.normal(size=(4, 6)).astype(np.float16))
    _write(tmp_path, "as_route_candidate", rng.normal(size=(12, 5)).astype(np.float16))
    _write(tmp_path, "hidden_candidate", rng.normal(size=(12, 3, 2, 5)).astype(np.float16))
    _write(tmp_path, "hidden_state", rng.normal(size=(4, 3, 2, 5)).astype(np.float16))
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"schema": DATASET_SCHEMA, "config_sha256": "test"})
    )
    return CriticDataset(tmp_path)


def _config():
    return {
        "critic": {
            "model": {"width": 24, "latent": 12, "hidden_projection": 3},
            "training": {
                "deterministic": True,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "patience": 2,
                "batch_snapshots": 2,
                "rank_weight": 1.0,
                "gradient_clip": 5.0,
                "minimum_delta": 0.0,
            },
        }
    }


def test_dataset_builds_grouped_batches_for_each_modality(tmp_path):
    dataset = _dataset(tmp_path)
    for mode in (
        "state_action",
        "state_route",
        "state_action_route",
        "state_action_router_hidden",
        "state_as_route",
    ):
        norm = dataset.normalization([0, 1], mode=mode, route_variant="all")
        batch = dataset.batch(
            [0, 1],
            mode=mode,
            route_variant="all",
            normalization=norm,
            device=torch.device("cpu"),
        )
        model = _make_model(_config(), dataset, mode, "all")
        assert model(batch).shape == (2, 3)


def test_tiny_member_training_runs_snapshot_grouped(tmp_path):
    dataset = _dataset(tmp_path)
    result = train_member(
        _config(),
        dataset,
        mode="state_action_route",
        variant="all",
        seed=33,
        train_states=[0, 1],
        validation_states=[2],
        device=torch.device("cpu"),
    )
    assert result.best_epoch in {0, 1}
    assert 0.0 <= result.validation["pairwise_accuracy"] <= 1.0
