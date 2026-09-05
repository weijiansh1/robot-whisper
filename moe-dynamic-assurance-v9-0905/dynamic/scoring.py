"""Runtime scoring for sealed train-free dynamic MoE profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "himoe.dynamic_score_reference.v1"


@dataclass(frozen=True)
class DynamicScoreReference:
    feature_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    reference_offsets: np.ndarray
    reference_values: np.ndarray
    score_names: tuple[str, ...]
    score_offsets: np.ndarray
    score_reference_indices: np.ndarray
    score_directions: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "DynamicScoreReference":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                raise ValueError(f"unsupported score reference schema: {archive['schema']}")
            return cls(
                feature_names=tuple(archive["feature_names"].astype(str)),
                reference_names=tuple(archive["reference_names"].astype(str)),
                reference_offsets=np.asarray(archive["reference_offsets"], dtype=np.int64),
                reference_values=np.asarray(archive["reference_values"], dtype=np.float32),
                score_names=tuple(archive["score_names"].astype(str)),
                score_offsets=np.asarray(archive["score_offsets"], dtype=np.int64),
                score_reference_indices=np.asarray(archive["score_reference_indices"], dtype=np.int32),
                score_directions=np.asarray(archive["score_directions"], dtype=np.int8),
            )

    def _percentile(self, reference_index: int, values: np.ndarray) -> np.ndarray:
        start, stop = self.reference_offsets[reference_index : reference_index + 2]
        reference = self.reference_values[start:stop]
        output = np.full(len(values), np.nan, dtype=np.float64)
        valid = np.isfinite(values)
        ranks = np.searchsorted(reference, values[valid], side="right")
        output[valid] = (ranks + 0.5) / (len(reference) + 1.0)
        return np.clip(output, 1e-8, 1.0 - 1e-8)

    def score(self, features: np.ndarray) -> dict[str, np.ndarray]:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have shape [N,{len(self.feature_names)}], got {matrix.shape}"
            )
        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        output = {}
        for score_index, score_name in enumerate(self.score_names):
            start, stop = self.score_offsets[score_index : score_index + 2]
            components = []
            for position in range(start, stop):
                reference_index = int(self.score_reference_indices[position])
                axis = self.reference_names[reference_index]
                percentile = self._percentile(reference_index, matrix[:, feature_index[axis]])
                components.append(
                    percentile if self.score_directions[position] > 0 else 1.0 - percentile
                )
            component_matrix = np.column_stack(components)
            score = np.full(len(matrix), np.nan, dtype=np.float32)
            valid = np.isfinite(component_matrix).all(axis=1)
            score[valid] = component_matrix[valid].mean(axis=1).astype(np.float32)
            output[score_name] = score
        return output
