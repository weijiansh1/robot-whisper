"""Causal motion descriptors; phenotype flags are not failure ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import periodogram


@dataclass(frozen=True)
class Config:
    movement_window: int = 6
    recurrence_window: int = 12
    confirmations: int = 2
    baseline_transitions: int = 4
    still_step_m: float = 0.001
    movement_floor_m: float = 0.001
    span_floor_m: float = 0.005
    relative_still: float = 0.25
    reversal_cosine: float = -0.3
    reversal_fraction: float = 0.5
    efficiency_ceiling: float = 0.35
    recurrence_ratio: float = 0.5
    command_rms_floor: float = 0.05
    command_high_frequency: float = 0.3
    command_high_fraction: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()
PHENOTYPES = ("still", "backtracking", "periodic_return", "command_jitter",
              "route_periodic_front", "route_periodic_back")


def confirmed(mask: np.ndarray, count: int = 2) -> np.ndarray:
    if count < 1:
        raise ValueError("confirmation count must be positive")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("expected a single episode")
    result = np.zeros_like(mask)
    run = 0
    for q, hit in enumerate(mask):
        run = run + 1 if hit else 0
        result[q] = run >= count
    return result


def first_hit(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1


def valid_alarms(first: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, lengths = np.asarray(first, dtype=int), np.asarray(lengths, dtype=int)
    if first.shape != lengths.shape or np.any(lengths < 1) or np.any(first < -1):
        raise ValueError("invalid alarm or length array")
    invalid = first >= lengths
    return np.where(invalid, -1, first), invalid


def reversal_rate(vectors: np.ndarray, cutoff: float, floor: float) -> float:
    norms = np.linalg.norm(vectors, axis=-1)
    denominator = norms[1:] * norms[:-1]
    valid = (norms[1:] > floor) & (norms[:-1] > floor)
    cosine = np.divide((vectors[1:] * vectors[:-1]).sum(axis=-1), denominator,
                       out=np.ones_like(denominator), where=valid)
    # Inactive pairs count against repeated reversals, not as evidence for them.
    return float(np.mean(valid & (cosine <= cutoff))) if len(valid) else 0.0


def trajectory_features(values: np.ndarray, config: Config = CONFIG,
                        vector_floor: float = 1e-10) -> dict[str, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected finite [query,dimension] trajectory")
    names = ("step", "speed", "relative_speed", "reversal", "efficiency", "span",
             "return_ratio", "return_valley", "period")
    out = {name: np.full(len(x), np.nan) for name in names}
    for lag in range(1, 7):
        out[f"lag{lag}"] = np.full(len(x), np.nan)
    delta = np.diff(x, axis=0)
    out["step"][1:] = np.linalg.norm(delta, axis=-1)
    baseline = (out["step"][1:config.baseline_transitions + 1].mean()
                if len(x) > config.baseline_transitions else np.nan)
    w = config.movement_window
    for q in range(w, len(x)):
        steps = out["step"][q - w + 1:q + 1]
        path = steps.sum()
        out["speed"][q] = steps.mean()
        if q > config.baseline_transitions and baseline > vector_floor:
            out["relative_speed"][q] = steps.mean() / baseline
        if path > vector_floor:
            out["efficiency"][q] = np.linalg.norm(x[q] - x[q - w]) / path
        out["reversal"][q] = reversal_rate(delta[q - w:q], config.reversal_cosine, vector_floor)
        out["span"][q] = np.linalg.norm(np.ptp(x[q - w:q + 1], axis=0))
    w = config.recurrence_window
    for q in range(w, len(x)):
        window = x[q - w:q + 1]
        distances = np.array([np.sqrt(np.square(window[k:] - window[:-k]).sum(axis=-1).mean())
                              for k in range(1, 7)])
        for lag, value in enumerate(distances, 1):
            out[f"lag{lag}"][q] = value
        if distances[0] > vector_floor:
            candidates = distances[1:5] / distances[0]
            period = int(np.argmin(candidates)) + 2
            neighbor = (distances[period - 2] + distances[period]) / 2
            out["return_ratio"][q] = distances[period - 1] / distances[0]
            out["return_valley"][q] = distances[period - 1] / max(neighbor, vector_floor)
            out["period"][q] = period
    return out


def command_features(actions: np.ndarray, config: Config = CONFIG) -> dict[str, np.ndarray]:
    """Describe all predicted commands, including the possibly unused final suffix."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[1:] != (10, 7) or not np.isfinite(actions).all():
        raise ValueError("expected finite [query,10,7] predicted actions")
    u = actions[:, :, :3]
    centered = u - u.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=1))
    frequencies, power = periodogram(u, fs=1.0, window="hann", detrend="constant", axis=1)
    power = power.sum(axis=-1)
    total = power.sum(axis=1)
    high = power[:, frequencies >= config.command_high_frequency].sum(axis=1)
    fraction = np.divide(high, total, out=np.zeros_like(high), where=total > 1e-15)
    reversal = np.array([reversal_rate(np.diff(chunk, axis=0), config.reversal_cosine, 1e-8)
                         for chunk in u])
    jitter = ((rms >= config.command_rms_floor) & (fraction >= config.command_high_fraction)
              & (reversal >= config.reversal_fraction))
    return {"command_rms": rms, "command_high_fraction": fraction,
            "command_reversal": reversal, "command_mean": u.mean(axis=1),
            "command_jitter": confirmed(jitter, config.confirmations)}


def motion_features(state: np.ndarray, actions: np.ndarray,
                    config: Config = CONFIG) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 8 or len(state) != len(actions):
        raise ValueError("state/actions do not align")
    f = trajectory_features(state[:, :3], config, vector_floor=1e-6)
    moving = (f["speed"] >= config.movement_floor_m) & (f["span"] >= config.span_floor_m)
    still = (f["speed"] <= config.still_step_m) & (f["relative_speed"] <= config.relative_still)
    back = moving & (f["reversal"] >= config.reversal_fraction) & (f["efficiency"] <= config.efficiency_ceiling)
    periodic = (moving & (f["return_ratio"] <= config.recurrence_ratio)
                & (f["return_valley"] <= config.recurrence_ratio))
    out = {f"pose_{name}": value for name, value in f.items()}
    out.update(position=state[:, :3], still=confirmed(still, config.confirmations),
               backtracking=confirmed(back, config.confirmations),
               periodic_return=confirmed(periodic, config.confirmations))
    out.update(command_features(actions, config))
    return out


def route_features(probabilities: np.ndarray, config: Config = CONFIG) -> dict[str, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 4 or p.shape[1:] != (8, 10, 32):
        raise ValueError("expected final-denoising action probabilities [query,8,10,32]")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=-1) <= 0).any():
        raise ValueError("invalid probabilities")
    roots = np.sqrt(p / p.sum(axis=-1, keepdims=True))
    out = {}
    for name, layers in (("front", slice(0, 4)), ("back", slice(4, 8))):
        x = roots[:, layers].reshape(len(roots), -1) / np.sqrt(2 * 4 * 10)
        f = trajectory_features(x, config)
        out.update({f"route_{name}_{key}": value for key, value in f.items()})
        periodic = ((f["speed"] > 1e-9) & (f["relative_speed"] >= config.relative_still)
                    & (f["return_ratio"] <= config.recurrence_ratio)
                    & (f["return_valley"] <= config.recurrence_ratio))
        out[f"route_periodic_{name}"] = confirmed(periodic, config.confirmations)
    return out


def event_window_start(event: str, confirmation: int, config: Config = CONFIG) -> int:
    if confirmation < 0:
        return -1
    width = (0 if event == "command_jitter" else config.recurrence_window
             if "periodic" in event else config.movement_window)
    return max(0, confirmation - (config.confirmations - 1) - width)
