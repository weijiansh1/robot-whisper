"""Frozen, route-only response diagnostics. No task-success labels are used."""

from itertools import combinations

import numpy as np
from scipy.optimize import nnls

HOLDOUT_TASKS = (2, 5, 8)
QUERIES = (8, 20)
SEED = 2026091421
PAIRS = np.array(list(combinations(range(8), 2)), dtype=np.int64)
RIDGE = 0.01


def split_for(task):
    return "holdout" if int(task) in HOLDOUT_TASKS else "train"


def candidate_noise(benchmark, task, query, candidate, source_noise):
    if candidate == 0:
        return np.array(source_noise, dtype=np.float32, copy=True)
    if benchmark not in ("plus", "pro") or not 1 <= candidate < 8:
        raise ValueError("Invalid noise identity")
    seed = np.random.SeedSequence([SEED, int(benchmark == "pro"), task, query, candidate])
    return np.random.default_rng(seed).standard_normal((10, 24)).astype(np.float32)


def normalize_routes(value):
    p = np.asarray(value, dtype=np.float64)
    if p.shape != (8, 8, 10, 11, 32):
        raise ValueError("Expected candidate/layer/denoise/token/expert axes")
    mass = p.sum(-1, keepdims=True)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(mass <= 0):
        raise ValueError("Invalid probability rows")
    return p / mass


def distances(p, ids):
    root = np.sqrt(normalize_routes(p))
    ids = np.asarray(ids)
    if ids.shape != (8, 8, 10, 11, 4) or ids.dtype.kind not in "iu":
        raise ValueError("Invalid actual top-k axes or dtype")
    if np.any(ids < 0) or np.any(ids >= 32):
        raise ValueError("Invalid expert ID")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate expert at one site")
    i, j = PAIRS.T
    h2 = np.square(root[i] - root[j]).sum(-1) / 2
    shared = (ids[i, ..., :, None] == ids[j, ..., None, :]).any(-1).mean(-1)
    disagreement = 1 - shared
    columns, names = [], []
    for layer in range(8):
        for role, tokens in (("state", slice(0, 1)), ("action", slice(1, 11))):
            for phase, denoise in (("early", slice(0, 3)), ("late", slice(7, 10))):
                columns.append(np.sqrt(h2[:, layer, denoise, tokens].mean((1, 2))))
                names.append("L%d_%s_%s_hellinger" % (layer, role, phase))
    for layer in range(8):
        for role, tokens in (("state", slice(0, 1)), ("action", slice(1, 11))):
            for phase, denoise in (("early", slice(0, 3)), ("late", slice(7, 10))):
                columns.append(disagreement[:, layer, denoise, tokens].mean((1, 2)))
                names.append("L%d_%s_%s_top4" % (layer, role, phase))
    return {
        "structured": np.stack(columns, axis=1),
        "full": np.sqrt(h2.mean((1, 2, 3)))[:, None],
        "old_center": np.sqrt(h2[:, 4:, :3, 1:].mean((1, 2, 3)))[:, None],
        "names": names,
    }


def pair_rms(values):
    value = np.asarray(values, dtype=np.float64)
    if value.shape[0] != 8 or not np.isfinite(value).all():
        raise ValueError("Eight finite candidates required")
    i, j = PAIRS.T
    return np.sqrt(np.square(value[i] - value[j]).reshape(28, -1).mean(-1))


def medoid(pair_distances):
    d = np.asarray(pair_distances, dtype=float).reshape(28)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Invalid pair distances")
    scores = np.zeros(8)
    for (i, j), distance in zip(PAIRS, d):
        scores[i] += distance
        scores[j] += distance
    return int(np.argmin(scores)), scores / 7


def parent_weights(parents):
    parents = np.asarray(parents)
    unique, counts = np.unique(parents, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1 / (len(unique) * sizes[p]) for p in parents])


def fit_distance(x, y, parents):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or len(parents) != len(x):
        raise ValueError("Invalid regression shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0):
        raise ValueError("Expected finite nonnegative distances")
    weights = parent_weights(parents)
    scale = np.maximum(np.sqrt((weights[:, None] * x * x).sum(0)), 1e-12)
    design = x / scale * np.sqrt(weights[:, None])
    design = np.vstack((design, np.sqrt(RIDGE) * np.eye(x.shape[1])))
    target = np.concatenate((y * np.sqrt(weights), np.zeros(x.shape[1])))
    coefficient, _ = nnls(design, target, maxiter=10000)
    return {"scale": scale.tolist(), "coefficient": coefficient.tolist()}


def predict_distance(model, x):
    return np.asarray(x) / np.asarray(model["scale"]) @ np.asarray(model["coefficient"])


def metrics(y, prediction, parents):
    error = np.asarray(y) - np.asarray(prediction)
    weights = parent_weights(parents)
    return {"rmse": float(np.sqrt(np.sum(weights * error ** 2))),
            "mae": float(np.sum(weights * np.abs(error))),
            "parents": len(set(parents)), "correlated_pairs": len(error)}


def permute_state_blocks(x, seed):
    x = np.asarray(x)
    if len(x) % 28:
        raise ValueError("Expected intact 28-pair state blocks")
    blocks = x.reshape(-1, 28, x.shape[-1])
    order = np.random.default_rng(seed).permutation(len(blocks))
    return blocks[order].reshape(x.shape)
