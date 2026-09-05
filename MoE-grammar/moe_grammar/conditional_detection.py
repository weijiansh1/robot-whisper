from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from moe_grammar.statistics import empirical_percentile


@dataclass
class PhaseConditionalCDF:
    """Empirical upper-tail calibration conditional on task and query position."""

    min_support: int = 32

    def fit(
        self,
        values: dict[str, np.ndarray],
        task: np.ndarray,
        position: np.ndarray,
        rows: np.ndarray,
    ) -> "PhaseConditionalCDF":
        selected = np.asarray(rows, dtype=np.int64)
        task = np.asarray(task, dtype=np.int16)
        position = np.asarray(position, dtype=np.int16)
        self.global_references: dict[str, np.ndarray] = {}
        self.task_references: dict[tuple[str, int], np.ndarray] = {}
        self.cell_references: dict[tuple[str, int, int], np.ndarray] = {}
        selected_task = task[selected]
        for name, raw in values.items():
            raw = np.asarray(raw, dtype=np.float64)
            self.global_references[name] = np.sort(raw[selected])
            for task_index in np.unique(selected_task):
                task_rows = selected[selected_task == task_index]
                self.task_references[(name, int(task_index))] = np.sort(raw[task_rows])
                task_positions = position[task_rows]
                for query_position in np.unique(task_positions):
                    cell_rows = task_rows[task_positions == query_position]
                    if len(cell_rows) >= self.min_support:
                        self.cell_references[
                            (name, int(task_index), int(query_position))
                        ] = np.sort(raw[cell_rows])
        return self

    def transform(
        self,
        name: str,
        values: np.ndarray,
        task_index: int,
        positions: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.int16)
        if len(values) != len(positions):
            raise ValueError("value and position axes differ")
        fallback = self.task_references.get(
            (name, int(task_index)), self.global_references[name]
        )
        output = np.empty(len(values), dtype=np.float64)
        for position in np.unique(positions):
            selected = positions == position
            reference = self.cell_references.get(
                (name, int(task_index), int(position)), fallback
            )
            output[selected] = empirical_percentile(reference, values[selected])
        return output


def normal_scores(percentiles: np.ndarray) -> np.ndarray:
    values = np.asarray(percentiles, dtype=np.float64)
    return ndtri(np.clip(values, 1e-5, 1.0 - 1e-5))


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window < 1:
        raise ValueError("window must be positive")
    prefix = np.r_[0.0, np.cumsum(values)]
    output = np.empty(len(values), dtype=np.float64)
    for stop in range(1, len(values) + 1):
        start = max(0, stop - window)
        output[stop - 1] = (prefix[stop] - prefix[start]) / (stop - start)
    return output


def page_cusum(values: np.ndarray, kappa: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(len(values), dtype=np.float64)
    current = 0.0
    for index, value in enumerate(values):
        current = max(0.0, current + float(value) - kappa)
        output[index] = current
    return output


def multiscale_scan(values: np.ndarray, windows: tuple[int, ...] = (1, 2, 4, 8)) -> np.ndarray:
    """Causal one-sided scan over standardized trailing sums."""
    values = np.asarray(values, dtype=np.float64)
    prefix = np.r_[0.0, np.cumsum(values)]
    output = np.empty(len(values), dtype=np.float64)
    for stop in range(1, len(values) + 1):
        candidates = []
        for window in windows:
            width = min(window, stop)
            candidates.append((prefix[stop] - prefix[stop - width]) / np.sqrt(width))
        output[stop - 1] = max(candidates)
    return output


def causal_aggregates(percentiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return running alarm statistics; every value at q uses only positions <= q."""
    percentiles = np.asarray(percentiles, dtype=np.float64)
    z_score = normal_scores(percentiles)
    output = {"point": np.maximum.accumulate(percentiles)}
    for window in (2, 4):
        output[f"window{window}"] = np.maximum.accumulate(
            trailing_mean(z_score, window)
        )
    for kappa in (0.0, 0.25, 0.5, 1.0):
        output[f"page{kappa:g}"] = np.maximum.accumulate(page_cusum(z_score, kappa))
    output["glr"] = np.maximum.accumulate(multiscale_scan(z_score))
    return output
