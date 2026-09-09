"""Pure numerical primitives for action--physics--outcome geometry.

The candidate pair is the unit of geometry, but never the unit of statistical
independence.  Aggregation helpers in this module first reduce pairs within a
snapshot and only then resample snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class PairThresholds:
    """Frozen thresholds used to assign candidate pairs to causal quadrants."""

    action_near: float
    action_far: float
    physics_near: float
    physics_far: float
    q_equivalence: float = 0.1
    q_difference: float = 0.1

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("pair thresholds must be finite and non-negative")
        if self.action_near >= self.action_far:
            raise ValueError("action_near must be strictly below action_far")
        if self.physics_near >= self.physics_far:
            raise ValueError("physics_near must be strictly below physics_far")
        if self.q_difference < self.q_equivalence:
            raise ValueError("q_difference must not be below q_equivalence")


def _candidate_tensor(values: np.ndarray, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if array.ndim < 2 or len(array) < 2:
        raise ValueError(f"{name} must contain at least two candidates")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validated_scale(scale: np.ndarray, width: int, name: str) -> np.ndarray:
    value = np.asarray(scale, dtype=np.float64)
    if value.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {value.shape}")
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def action_distance_matrix(
    actions: np.ndarray,
    action_std: np.ndarray,
    gripper_weight: float = 0.25,
) -> np.ndarray:
    """Mean stepwise L2 distance in checkpoint-normalized action space.

    This implements ``H^-1 sum_t ||W_a (a_i,t - a_j,t)||_2``.  It is not the
    flattened RMS distance used by the original RAD bridge, so both should be
    reported when comparing against older analyses.
    """

    value = _candidate_tensor(actions, "actions", ndim=3)
    scale = _validated_scale(action_std, value.shape[-1], "action_std")
    if not np.isfinite(gripper_weight) or gripper_weight < 0.0:
        raise ValueError("gripper_weight must be finite and non-negative")
    weights = np.ones(value.shape[-1], dtype=np.float64)
    weights[-1] = float(gripper_weight)
    delta = (value[:, None] - value[None, :]) / scale
    distance = np.linalg.norm(delta * weights, axis=-1).mean(axis=-1)
    np.fill_diagonal(distance, 0.0)
    return distance


def action_rms_distance_matrix(actions: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """Legacy bridge sensitivity: RMS over all action steps and dimensions."""

    value = _candidate_tensor(actions, "actions", ndim=3)
    scale = _validated_scale(action_std, value.shape[-1], "action_std")
    delta = (value[:, None] - value[None, :]) / scale
    distance = np.sqrt(np.mean(np.square(delta), axis=(-2, -1)))
    np.fill_diagonal(distance, 0.0)
    return distance


def coordinate_scale(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Exploratory coordinate scale estimated from candidate trajectories."""

    value = _candidate_tensor(values, "values")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and positive")
    width = value.shape[-1]
    scale = value.reshape(-1, width).std(axis=0)
    return np.where(scale > floor, scale, 1.0)


def temporal_l2_distance_matrix(values: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Mean aligned-time L2 distance for a ``[candidate,time,feature]`` tensor."""

    value = _candidate_tensor(values, "values", ndim=3)
    if value.shape[-1] == 0:
        return np.zeros((len(value), len(value)), dtype=np.float64)
    if scale is None:
        denominator = np.ones(value.shape[-1], dtype=np.float64)
    else:
        denominator = _validated_scale(scale, value.shape[-1], "scale")
    delta = (value[:, None] - value[None, :]) / denominator
    distance = np.linalg.norm(delta, axis=-1).mean(axis=-1)
    np.fill_diagonal(distance, 0.0)
    return distance


def quaternion_distance_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Mean quaternion geodesic distance in radians, treating ``q`` and ``-q`` alike."""

    value = _candidate_tensor(quaternions, "quaternions", ndim=3)
    if value.shape[-1] != 4:
        raise ValueError("quaternions must have shape [candidate,time,4]")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("quaternions contain a zero-norm value")
    unit = value / norm
    dot = np.einsum("itd,jtd->ijt", unit, unit)
    angle = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    distance = angle.mean(axis=-1)
    np.fill_diagonal(distance, 0.0)
    return distance


def contact_event_distance_matrix(contact_active: np.ndarray) -> np.ndarray:
    """Mean aligned-time Jaccard distance between raw contact-pair sets."""

    active = np.asarray(contact_active, dtype=np.bool_)
    if active.ndim != 3 or len(active) < 2:
        raise ValueError("contact_active must have shape [candidate,time,contact_pair]")
    intersection = np.logical_and(active[:, None], active[None, :]).sum(axis=-1)
    union = np.logical_or(active[:, None], active[None, :]).sum(axis=-1)
    per_time = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    distance = (1.0 - per_time).mean(axis=-1)
    np.fill_diagonal(distance, 0.0)
    return distance


def exact_event_equal_matrix(
    contact_active: np.ndarray,
    chunk_success: np.ndarray | None = None,
) -> np.ndarray:
    """Whether candidates have exactly the same contact and success event tape."""

    active = np.asarray(contact_active, dtype=np.bool_)
    if active.ndim != 3 or len(active) < 2:
        raise ValueError("contact_active must have shape [candidate,time,contact_pair]")
    equal = np.all(active[:, None] == active[None, :], axis=(-2, -1))
    if chunk_success is not None:
        success = np.asarray(chunk_success, dtype=np.bool_)
        if success.shape != active.shape[:2]:
            raise ValueError("chunk_success must have shape [candidate,time]")
        equal &= np.all(success[:, None] == success[None, :], axis=-1)
    np.fill_diagonal(equal, True)
    return equal


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    return value[np.triu_indices(len(value), k=1)]


def component_scale(distance: np.ndarray, floor: float = 1e-12) -> float:
    """Median positive within-snapshot pair distance for one physical component."""

    pairs = np.asarray(upper_triangle(distance), dtype=np.float64)
    positive = pairs[pairs > floor]
    return float(np.median(positive)) if len(positive) else 1.0


def composite_distance_matrix(
    components: Mapping[str, np.ndarray],
    scales: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Equal-component RMS after each physical component receives a fixed scale."""

    if not components:
        raise ValueError("at least one physical component is required")
    shape = None
    used_scales: dict[str, float] = {}
    normalized = []
    for name, raw in components.items():
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"physical component {name!r} is not square")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("physical components have different candidate counts")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError(f"physical component {name!r} is invalid")
        scale = component_scale(matrix) if scales is None else float(scales[name])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"physical scale {name!r} must be finite and positive")
        used_scales[name] = scale
        normalized.append(matrix / scale)
    distance = np.sqrt(np.mean(np.square(np.stack(normalized)), axis=0))
    np.fill_diagonal(distance, 0.0)
    return distance, used_scales


def physical_distance_components(
    sim_states: np.ndarray,
    eef_positions: np.ndarray,
    eef_quaternions: np.ndarray,
    gripper_qpos: np.ndarray,
    nq: int,
    robot_qpos_indices: Sequence[int],
    object_qpos_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Compute separately interpretable physical-trajectory distances."""

    sim = _candidate_tensor(sim_states, "sim_states", ndim=3)
    if nq <= 0 or sim.shape[-1] < 1 + nq:
        raise ValueError("nq is inconsistent with sim_states")
    qpos = sim[..., 1 : 1 + nq]
    robot = np.asarray(robot_qpos_indices, dtype=np.int64)
    objects = np.asarray(object_qpos_indices, dtype=np.int64)
    for name, indices in (("robot_qpos_indices", robot), ("object_qpos_indices", objects)):
        if np.any(indices < 0) or np.any(indices >= nq):
            raise ValueError(f"{name} is outside the qpos vector")
    components = {
        "eef_position_m": temporal_l2_distance_matrix(eef_positions),
        "eef_orientation_rad": quaternion_distance_matrix(eef_quaternions),
        "gripper_qpos_native": temporal_l2_distance_matrix(gripper_qpos),
    }
    if len(robot):
        components["robot_qpos_native"] = temporal_l2_distance_matrix(qpos[..., robot])
    if len(objects):
        components["object_qpos_native"] = temporal_l2_distance_matrix(qpos[..., objects])
    return components


def _clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("invalid binomial count")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def paired_binary_difference_interval(
    left: np.ndarray,
    right: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Conservative CI for a paired binary probability difference.

    Write ``delta = theta * (2*psi - 1)``, where ``theta`` is the probability
    that a CRN pair is discordant and ``psi`` is the conditional probability
    that the left candidate wins.  Clopper--Pearson intervals for both terms,
    combined with Bonferroni coverage, avoid declaring equivalence merely because
    a small number of repeated continuations happened to agree.
    """

    a = np.asarray(left)
    b = np.asarray(right)
    if a.ndim != 1 or b.shape != a.shape or len(a) == 0:
        raise ValueError("paired outcomes must be non-empty aligned vectors")
    if not np.all(np.isin(a, (0, 1, False, True))) or not np.all(
        np.isin(b, (0, 1, False, True))
    ):
        raise ValueError("paired outcomes must be binary")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    a = a.astype(np.int8)
    b = b.astype(np.int8)
    left_only = int(np.sum((a == 1) & (b == 0)))
    right_only = int(np.sum((a == 0) & (b == 1)))
    discordant = left_only + right_only
    n = len(a)
    family_alpha = 1.0 - confidence
    component_alpha = family_alpha / 2.0
    theta_low, theta_high = _clopper_pearson(discordant, n, component_alpha)
    if discordant:
        psi_low, psi_high = _clopper_pearson(left_only, discordant, component_alpha)
    else:
        psi_low, psi_high = 0.0, 1.0
    direction_low = 2.0 * psi_low - 1.0
    direction_high = 2.0 * psi_high - 1.0
    lower = theta_high * direction_low if direction_low < 0.0 else theta_low * direction_low
    upper = theta_high * direction_high if direction_high > 0.0 else theta_low * direction_high
    return {
        "estimate": float((a - b).mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "repeats": int(n),
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
    }


def interval_is_equivalent(interval: Mapping[str, float], epsilon: float) -> bool:
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    return bool(float(interval["lower"]) >= -epsilon and float(interval["upper"]) <= epsilon)


def interval_is_different(interval: Mapping[str, float], delta: float) -> bool:
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")
    return bool(float(interval["lower"]) > delta or float(interval["upper"]) < -delta)


def classify_pair(
    action_distance: float,
    physics_distance: float,
    events_equal: bool,
    q_interval: Mapping[str, float] | None,
    thresholds: PairThresholds,
) -> dict[str, bool]:
    """Assign one pair to the preregistered behavior-geometry regions."""

    q_equivalent = bool(
        q_interval is not None
        and interval_is_equivalent(q_interval, thresholds.q_equivalence)
    )
    q_different = bool(
        q_interval is not None and interval_is_different(q_interval, thresholds.q_difference)
    )
    far_action = action_distance >= thresholds.action_far
    near_action = action_distance <= thresholds.action_near
    far_physics = physics_distance >= thresholds.physics_far
    near_physics = physics_distance <= thresholds.physics_near
    outcome_same = bool(events_equal and q_equivalent)
    outcome_different = bool((not events_equal) or q_different)
    return {
        "command_redundancy": bool(far_action and near_physics and outcome_same),
        "control_equivalence": bool(far_action and far_physics and outcome_same),
        "physics_amplification": bool(near_action and far_physics and outcome_different),
        "critical_microdifference": bool(near_action and near_physics and outcome_different),
        "sensitive_fork": bool(near_action and outcome_different),
        "q_equivalent": q_equivalent,
        "q_different": q_different,
        "outcome_same": outcome_same,
        "outcome_different": outcome_different,
    }


def empirical_thresholds(
    action_pairs: np.ndarray,
    physics_pairs: np.ndarray,
    near_quantile: float = 0.2,
    far_quantile: float = 0.8,
    q_equivalence: float = 0.1,
    q_difference: float = 0.1,
) -> PairThresholds:
    """Fit exploratory thresholds on a designated calibration split only."""

    if not 0.0 <= near_quantile < far_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= near < far <= 1")
    action = np.asarray(action_pairs, dtype=np.float64)
    physics = np.asarray(physics_pairs, dtype=np.float64)
    if action.ndim != 1 or physics.ndim != 1 or not len(action) or not len(physics):
        raise ValueError("calibration pair distances must be non-empty vectors")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(physics)):
        raise ValueError("calibration pair distances must be finite")
    return PairThresholds(
        action_near=float(np.quantile(action, near_quantile)),
        action_far=float(np.quantile(action, far_quantile)),
        physics_near=float(np.quantile(physics, near_quantile)),
        physics_far=float(np.quantile(physics, far_quantile)),
        q_equivalence=q_equivalence,
        q_difference=q_difference,
    )


def snapshot_bootstrap_mean(
    per_snapshot: Mapping[str, Sequence[float] | np.ndarray],
    draws: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | bool | None]:
    """Bootstrap a pair metric after reducing all dependent pairs per snapshot."""

    if draws < 100:
        raise ValueError("draws must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    reduced = []
    for key, raw in per_snapshot.items():
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"snapshot {key!r} has invalid pair values")
        reduced.append(float(value.mean()))
    if not reduced:
        raise ValueError("at least one snapshot is required")
    value = np.asarray(reduced, dtype=np.float64)
    if len(value) == 1:
        return {
            "mean": float(value[0]),
            "lower": None,
            "upper": None,
            "confidence": float(confidence),
            "snapshots": 1,
            "draws": 0,
            "available": False,
        }
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(value), size=(draws, len(value)))
    distribution = value[sample].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return {
        "mean": float(value.mean()),
        "lower": float(low),
        "upper": float(high),
        "confidence": float(confidence),
        "snapshots": int(len(value)),
        "draws": int(draws),
        "available": True,
    }
