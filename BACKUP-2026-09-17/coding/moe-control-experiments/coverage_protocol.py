"""Frozen discrete gate pools and action-space coverage diagnostics."""

import numpy as np

from gate_protocol import SHAPE, scope_mask

SEED = 2026091501
TASKS = (0, 3, 6, 9)
QUERY = 8
INIT = 39
GENERATORS = ("noise", "state_gate", "action_gate", "action_gate_l2")
POOLS = (0, 1)
CANDIDATES = (0, 1, 2, 3)
TARGETS = tuple(range(8))


def specifications():
    return [dict(kind="candidate", generator=generator, pool=pool, candidate=candidate)
            for pool in POOLS for generator in GENERATORS for candidate in CANDIDATES]


def gate_bias(spec):
    result = np.zeros(SHAPE, np.float32)
    if spec is None or spec["generator"] == "noise":
        return result
    if spec["generator"] not in GENERATORS or spec["pool"] not in POOLS or spec["candidate"] not in CANDIDATES:
        raise ValueError("Unregistered candidate")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, spec["pool"], spec["candidate"]]))
    ranks = np.argsort(np.argsort(rng.random(SHAPE), axis=-1), axis=-1)
    direction = np.where(ranks < 16, -1., 1.).astype(np.float32)
    state = spec["generator"] == "state_gate"
    amplitude = .1 / np.sqrt(10) if spec["generator"] == "action_gate_l2" else .1
    return direction * np.float32(amplitude) * scope_mask("front_state_late" if state else "front_action_late")[..., None]


def request_noise(parent, source_noise, spec):
    if spec is None or spec["generator"] != "noise":
        return np.asarray(source_noise, np.float32).copy()
    if spec["kind"] not in ("candidate", "target"):
        raise ValueError("Invalid noise namespace")
    namespace = 1 if spec["kind"] == "candidate" else 2
    benchmark = {"plus": 0, "pro": 1}[parent["benchmark"]]
    seed = [SEED, namespace, benchmark, parent["task"], INIT, QUERY, spec["pool"], spec["candidate"]]
    return np.random.default_rng(np.random.SeedSequence(seed)).standard_normal(source_noise.shape).astype(np.float32)


def action_vectors(actions, std):
    value = np.asarray(actions, np.float64)
    scale = np.asarray(std, np.float64)[:6]
    if value.shape[-2:] != (10, 7) or scale.shape != (6,) or not np.all(scale > 0):
        raise ValueError("Invalid action shape or normalization")
    result = (value[..., :6] / scale).reshape(value.shape[:-2] + (60,))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite action vectors")
    return result


def pool_metrics(default, candidates, targets, std):
    base = action_vectors(default, std)
    points = action_vectors(candidates, std)
    reference = action_vectors(targets, std)
    if points.shape != (4, 60) or reference.ndim != 2 or reference.shape[1] != 60:
        raise ValueError("Expected four candidates and reference action blocks")
    delta = points - base
    radii = np.sqrt(np.mean(delta ** 2, axis=-1))
    pool = np.concatenate((base[None], points))
    pairwise = np.sqrt(np.mean((pool[:, None] - pool[None]) ** 2, axis=-1))
    reference_delta = reference - base
    base_distance = np.linalg.norm(reference_delta, axis=-1)
    nearest = np.linalg.norm(reference[:, None] - pool[None], axis=-1).min(axis=1)
    valid = base_distance > 1e-12
    gains = np.full(len(reference), np.nan)
    gains[valid] = np.clip(1 - nearest[valid] / base_distance[valid], 0., 1.)
    norms = np.linalg.norm(delta, axis=-1)
    denominator = base_distance[:, None] * norms[None]
    cosines = np.divide(reference_delta @ delta.T, denominator, out=np.zeros_like(denominator), where=denominator > 1e-24)
    best_cosine = np.clip(cosines.max(axis=1), 0., 1.)
    energy = np.linalg.svd(delta, compute_uv=False) ** 2
    rank = float(energy.sum() ** 2 / np.square(energy).sum()) if energy.sum() > 1e-24 else 0.
    return {
        "mean_radius_rms": float(radii.mean()), "max_radius_rms": float(radii.max()),
        "mean_pairwise_rms": float(pairwise[np.triu_indices(5, 1)].mean()),
        "participation_rank": rank, "coverage_gain": float(gains[valid].mean()) if valid.any() else None,
        "best_positive_cosine": float(best_cosine[valid].mean()) if valid.any() else None,
        "target_coverage_gains": [float(x) if np.isfinite(x) else None for x in gains],
        "target_default_rms": (base_distance / np.sqrt(60)).tolist(),
        "target_nearest_rms": (nearest / np.sqrt(60)).tolist(),
        "valid_targets": int(valid.sum()), "candidate_radii_rms": radii.tolist(),
    }
