"""Shared schema and numerical helpers for the MoE Assurance Profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "himoe.moe_assurance_profile.v1"
LAYER_NAMES = ("L12", "L13", "L14", "L15")
EPSILON = 1e-8


@dataclass(frozen=True)
class AssuranceProfile:
    episode_id: np.ndarray
    query: np.ndarray
    scene: np.ndarray
    repeat: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    layer_features: np.ndarray
    layer_feature_names: tuple[str, ...]
    flow_profile: np.ndarray
    recurrence_distance: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "AssuranceProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                raise ValueError(f"unsupported assurance profile schema in {path}")
            return cls(
                episode_id=np.asarray(archive["episode_id"], dtype=np.int32),
                query=np.asarray(archive["query"], dtype=np.int16),
                scene=np.asarray(archive["scene"], dtype=np.int16),
                repeat=np.asarray(archive["repeat"], dtype=np.int16),
                features=np.asarray(archive["features"], dtype=np.float32),
                feature_names=tuple(archive["feature_names"].astype(str)),
                layer_features=np.asarray(archive["layer_features"], dtype=np.float32),
                layer_feature_names=tuple(archive["layer_feature_names"].astype(str)),
                flow_profile=np.asarray(archive["flow_profile"], dtype=np.float32),
                recurrence_distance=np.asarray(
                    archive["recurrence_distance"], dtype=np.float32
                ),
            )

    def feature(self, name: str) -> np.ndarray:
        try:
            index = self.feature_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.features[:, index]


def episode_local_query(episode_id: np.ndarray, control_step: np.ndarray) -> np.ndarray:
    episode_id = np.asarray(episode_id)
    control_step = np.asarray(control_step)
    if episode_id.shape != control_step.shape:
        raise ValueError("episode and control-step arrays do not align")
    query = np.empty(len(episode_id), dtype=np.int16)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        order = rows[np.argsort(control_step[rows], kind="stable")]
        query[order] = np.arange(len(rows), dtype=np.int16)
    return query


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    reference = reference[np.isfinite(reference)]
    if len(reference) < 2:
        raise ValueError("percentile reference is too small")
    value = np.asarray(values, dtype=np.float64)
    rank = np.searchsorted(reference, value, side="right")
    output = (rank + 0.5) / (len(reference) + 1.0)
    output[~np.isfinite(value)] = np.nan
    return np.clip(output, EPSILON, 1.0 - EPSILON)


def auc_pairwise(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).sum(dtype=np.float64)
    ties = (positive[:, None] == negative[None, :]).sum(dtype=np.float64)
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))
