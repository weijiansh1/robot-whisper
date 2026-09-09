"""Route-only stopping rule for one flow-matching policy query.

The rule consumes one scalar change after each pair of consecutive denoising
passes.  It never reads actions, rewards, simulator state, or rollout outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StopConfig:
    threshold: float = 0.0047
    min_steps: int = 8
    consecutive: int = 2
    max_steps: int = 10

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("threshold must be finite and nonnegative")
        if not 2 <= self.min_steps <= self.max_steps:
            raise ValueError("min_steps must be in [2, max_steps]")
        if not 1 <= self.consecutive < self.max_steps:
            raise ValueError("consecutive must be in [1, max_steps)")


class ConsecutiveRouteStopper:
    """Stop after enough consecutive route changes are below a threshold."""

    def __init__(self, config: StopConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.streak = 0

    def observe(self, completed_steps: int, change: float) -> bool:
        if not 2 <= completed_steps <= self.config.max_steps:
            raise ValueError("completed_steps must be in [2, max_steps]")
        if not np.isfinite(change) or change < 0.0:
            raise ValueError("route change must be finite and nonnegative")
        self.streak = self.streak + 1 if change <= self.config.threshold else 0
        return (
            completed_steps >= self.config.min_steps
            and self.streak >= self.config.consecutive
        )


def rms_hellinger_change(previous: np.ndarray, current: np.ndarray) -> float:
    """RMS Hellinger change over aligned layer probability vectors.

    Inputs have shape ``[layers, experts]``.  Each layer is independently
    renormalized, which makes the number comparable across HB layers.
    """

    left = np.asarray(previous, dtype=np.float64)
    right = np.asarray(current, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("route arrays must be aligned [layers, experts]")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("route arrays must be finite")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("route probabilities must be nonnegative")
    left_mass = left.sum(axis=-1, keepdims=True)
    right_mass = right.sum(axis=-1, keepdims=True)
    if np.any(left_mass <= 0.0) or np.any(right_mass <= 0.0):
        raise ValueError("every route probability vector must have positive mass")
    left = left / left_mass
    right = right / right_mass
    squared = 0.5 * np.square(np.sqrt(left) - np.sqrt(right)).sum(axis=-1)
    return float(np.sqrt(squared.mean()))


def stopping_step(changes: Iterable[float], config: StopConfig) -> int:
    """Replay a query and return the number of completed denoising passes."""

    rule = ConsecutiveRouteStopper(config)
    values = list(changes)
    if len(values) != config.max_steps - 1:
        raise ValueError("expected max_steps - 1 adjacent route changes")
    for completed_steps, change in enumerate(values, start=2):
        if rule.observe(completed_steps, float(change)):
            return completed_steps
    return config.max_steps
