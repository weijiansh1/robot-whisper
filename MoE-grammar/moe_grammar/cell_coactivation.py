"""Co-activation structure of the 256 (layer, expert) cells.

This is a descriptive study, not a detector. The unit of analysis is the *cell*
(one expert of one layer), and the sample axis is the query. We ask whether the
256 cells fall into eight per-layer blobs or whether cross-layer modules form.

Two facts drive every design choice here.

1.  Each layer's 32 router probabilities are a composition: they sum to one for
    every query. The closure is *per layer*, not across all 256 cells, so the
    right transform is a centred log-ratio taken inside each layer. The loads
    are strictly positive (the action channel bottoms out near 0.012 and the
    state channel near 1.4e-4), so no zero replacement is needed.

2.  CLR leaves an exact sum-to-zero constraint inside each layer, which forces
    the mean within-layer off-diagonal correlation to roughly -1/31 = -0.032.
    Any statement about "layer blobs" must therefore be read against that
    offset, which is why the headline test runs on |r| coupling strength, where
    the offset *helps* the layer partition rather than hurting it.

Experts are independently permutable per layer, so we never compute a distance
between layer a's expert coordinates and layer b's. Everything here is built
from correlations between individual cells across queries, which is well
defined for a fixed checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
N_LAYERS = 8
N_EXPERTS = 32
N_CELLS = N_LAYERS * N_EXPERTS
FRONT_LAYERS = 4

CELL_LAYER = np.repeat(np.arange(N_LAYERS), N_EXPERTS)
CELL_EXPERT = np.tile(np.arange(N_EXPERTS), N_LAYERS)
CELL_FRONT = (CELL_LAYER < FRONT_LAYERS).astype(np.int64)


def cell_names() -> list[str]:
    return [f"{LAYER_NAMES[layer]}.e{expert:02d}" for layer, expert in zip(CELL_LAYER, CELL_EXPERT)]


@dataclass
class CellLoads:
    """Router load for every (layer, expert) cell on the fitting split."""

    action: np.ndarray  # [N, 8, 32] float32
    state: np.ndarray  # [N, 8, 32] float32
    run_index: np.ndarray  # [N] which run file the query came from
    episode_index: np.ndarray  # [N] globally unique episode id
    episode_step: np.ndarray  # [N]
    run_keys: list[str]

    def channel(self, name: str) -> np.ndarray:
        if name == "action":
            return self.action
        if name == "state":
            return self.state
        raise ValueError(f"unknown channel {name!r}")


def load_cell_loads(
    load_dir: Path,
    features_dir: Path,
    max_init_state_id: int = 30,
    stride: int = 1,
) -> CellLoads:
    """Load success queries with ``init_state_id < max_init_state_id`` (the train split).

    Failure labels are never used downstream; ``success`` only selects the split
    that the rest of this repository fits on.
    """

    load_files = sorted(load_dir.glob("*.npz"))
    if not load_files:
        raise FileNotFoundError(f"no load files under {load_dir}")

    action: list[np.ndarray] = []
    state: list[np.ndarray] = []
    run_index: list[np.ndarray] = []
    episode_index: list[np.ndarray] = []
    episode_step: list[np.ndarray] = []
    run_keys: list[str] = []
    episode_offset = 0

    for index, load_path in enumerate(load_files):
        feature_path = features_dir / load_path.name
        if not feature_path.exists():
            raise FileNotFoundError(f"no feature file matching {load_path.name}")
        loads = np.load(load_path, allow_pickle=True)
        features = np.load(feature_path, allow_pickle=True)
        left = np.asarray(loads["episode_step"], dtype=np.int64)
        right = np.asarray(features["episode_step"], dtype=np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"{load_path.name} is not row aligned with its feature file")

        keep = np.asarray(features["success"], dtype=bool)
        keep &= np.asarray(features["init_state_id"], dtype=np.int64) < max_init_state_id
        rows = np.flatnonzero(keep)
        if stride > 1:
            rows = rows[::stride]
        if rows.size == 0:
            continue

        action.append(np.asarray(loads["action_load"], dtype=np.float32)[rows])
        state.append(np.asarray(loads["state_load"], dtype=np.float32)[rows])
        run_index.append(np.full(rows.size, index, dtype=np.int32))
        episodes = np.asarray(loads["episode_id"], dtype=np.int64)[rows]
        _, compact = np.unique(episodes, return_inverse=True)
        episode_index.append((compact + episode_offset).astype(np.int64))
        episode_offset += int(compact.max()) + 1
        episode_step.append(np.asarray(loads["episode_step"], dtype=np.int32)[rows])
        run_keys.append(load_path.stem)

    return CellLoads(
        action=np.concatenate(action),
        state=np.concatenate(state),
        run_index=np.concatenate(run_index),
        episode_index=np.concatenate(episode_index),
        episode_step=np.concatenate(episode_step),
        run_keys=run_keys,
    )


def clr_matrix(loads: np.ndarray) -> np.ndarray:
    """Per-layer centred log-ratio, flattened to [N, 256]."""

    logged = np.log(np.asarray(loads, dtype=np.float64))
    centred = logged - logged.mean(axis=2, keepdims=True)
    return centred.reshape(len(logged), N_CELLS)


def sqrt_matrix(loads: np.ndarray) -> np.ndarray:
    """Hellinger (square-root) transform, flattened to [N, 256]."""

    return np.sqrt(np.asarray(loads, dtype=np.float64)).reshape(len(loads), N_CELLS)


def standardize(values: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    centred = values - values.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale <= 0] = 1.0
    return np.ascontiguousarray(centred / scale, dtype=dtype)


def correlation(standardized: np.ndarray) -> np.ndarray:
    matrix = (standardized.T @ standardized) / standardized.shape[0]
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def group_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract the mean of each group, so only within-group variation survives."""

    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(groups)]))
    out = np.array(values, dtype=np.float64, copy=True)
    for start, stop in zip(starts, stops):
        rows = order[start:stop]
        out[rows] -= out[rows].mean(axis=0, keepdims=True)
    return out


def episode_differences(values: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    """Query-to-query differences inside each episode; the first query is dropped.

    Front-layer cells drift on a much slower timescale than back-layer cells, and a
    slow timescale inflates correlation on its own. Differencing whitens the drift
    and puts every cell on a comparable footing, so it is the control for reading
    the front/back gap as coupling rather than as timescale. Assumes the rows of
    each episode are contiguous and in step order, which holds at stride 1.
    """

    boundary = np.ones(len(values), dtype=bool)
    boundary[1:] = episode_index[1:] != episode_index[:-1]
    differences = np.empty_like(values)
    differences[1:] = values[1:] - values[:-1]
    differences[0] = 0.0
    return differences[~boundary]


def cross_layer_mask() -> np.ndarray:
    return CELL_LAYER[:, None] != CELL_LAYER[None, :]


def upper_cross_mask() -> np.ndarray:
    return np.triu(cross_layer_mask(), 1)


def block_means(matrix: np.ndarray, absolute: bool) -> np.ndarray:
    """8x8 mean of ``matrix`` over layer-pair blocks, diagonal excluded on the block diagonal."""

    values = np.abs(matrix) if absolute else matrix
    out = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            block = values[i * N_EXPERTS : (i + 1) * N_EXPERTS, j * N_EXPERTS : (j + 1) * N_EXPERTS]
            if i == j:
                keep = ~np.eye(N_EXPERTS, dtype=bool)
                out[i, j] = float(block[keep].mean())
            else:
                out[i, j] = float(block.mean())
    return out


def layer_shift_surrogate(
    standardized: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Circularly shift each layer's block of columns by its own random offset.

    Within-layer structure (compositional closure, serial autocorrelation, task
    blocking) is preserved exactly; cross-layer alignment is destroyed. This is
    the null for "is there any real cross-layer coupling", and it is the reason
    the chance level for |r| is 0.015 rather than 0 - queries are massively
    serially dependent, so the effective sample size is far below the row count.
    """

    n_rows = standardized.shape[0]
    out = np.empty_like(standardized)
    for layer in range(N_LAYERS):
        columns = slice(layer * N_EXPERTS, (layer + 1) * N_EXPERTS)
        offset = int(rng.integers(1, n_rows))
        out[:, columns] = np.roll(standardized[:, columns], offset, axis=0)
    return out


def cell_shift_closed_surrogate(
    loads: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shift every cell independently, then re-close each layer to the simplex.

    All real coupling is destroyed, both within and across layers, but the
    per-layer sum-to-one constraint is restored before the CLR, so the resulting
    null keeps the closure-induced negative offset. It is the only baseline that
    makes within-layer and cross-layer blocks directly comparable.
    """

    n_rows = loads.shape[0]
    flat = loads.reshape(n_rows, N_CELLS)
    offsets = rng.integers(0, n_rows, size=N_CELLS)
    rows = (np.arange(n_rows)[:, None] + offsets[None, :]) % n_rows
    shifted = np.take_along_axis(flat, rows, axis=0).reshape(n_rows, N_LAYERS, N_EXPERTS)
    shifted = shifted / shifted.sum(axis=2, keepdims=True)
    return standardize(clr_matrix(shifted))


def modularity(weights: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of ``labels`` on a weighted undirected graph."""

    graph = np.array(weights, dtype=np.float64, copy=True)
    np.fill_diagonal(graph, 0.0)
    degree = graph.sum(axis=1)
    total = degree.sum()
    if total <= 0:
        return 0.0
    same = labels[:, None] == labels[None, :]
    expected = np.outer(degree, degree) / total
    return float((graph - expected)[same].sum() / total)


def closure_corrected(matrix: np.ndarray) -> np.ndarray:
    """Add back the constant that the per-layer CLR closure subtracts inside a layer.

    CLR forces each layer's 32 components to sum to zero, so the average
    within-layer off-diagonal correlation is pinned near -1/(32-1). Adding that
    offset back to the within-layer blocks removes the mechanical part of the
    within-layer negativity and lets the layer partition be judged on merit.
    """

    corrected = np.array(matrix, dtype=np.float64, copy=True)
    same = CELL_LAYER[:, None] == CELL_LAYER[None, :]
    same &= ~np.eye(len(matrix), dtype=bool)
    corrected[same] += 1.0 / (N_EXPERTS - 1)
    return corrected


def split_half_reliability(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator, repeats: int = 5
) -> dict:
    """Correlate the 256x256 matrices built from two disjoint halves of the episodes.

    This bounds how much of the observed structure is reproducible at all. Splitting
    on episodes rather than rows keeps the serial dependence inside one half.
    """

    unique = np.unique(groups)
    cross = CELL_LAYER[:, None] != CELL_LAYER[None, :]
    upper = np.triu(np.ones_like(cross, dtype=bool), 1)
    all_pairs = []
    cross_pairs = []
    for _ in range(repeats):
        shuffled = rng.permutation(unique)
        left = np.isin(groups, shuffled[: len(shuffled) // 2])
        first = correlation(standardize(values[left]))
        second = correlation(standardize(values[~left]))
        all_pairs.append(np.corrcoef(first[upper], second[upper])[0, 1])
        keep = upper & cross
        cross_pairs.append(np.corrcoef(first[keep], second[keep])[0, 1])
    return {
        "repeats": repeats,
        "all_pairs_r": float(np.mean(all_pairs)),
        "all_pairs_r_std": float(np.std(all_pairs, ddof=1)),
        "cross_layer_pairs_r": float(np.mean(cross_pairs)),
        "cross_layer_pairs_r_std": float(np.std(cross_pairs, ddof=1)),
    }


def top_edge_graph(matrix: np.ndarray, density: float) -> np.ndarray:
    """Binary graph on the strongest ``density`` fraction of |r| edges.

    Newman modularity on a *dense* weighted graph is dominated by the relative
    spread of the weights, so an all-noise surrogate can score higher than the
    real matrix. Matching the edge density and binarising removes that: observed
    and surrogate graphs then have the same number of edges and the same weights,
    and only the placement of the edges differs.
    """

    off = ~np.eye(len(matrix), dtype=bool)
    strength = np.abs(matrix)
    cutoff = np.quantile(strength[off], 1.0 - density)
    graph = ((strength >= cutoff) & off).astype(np.float64)
    return graph


def coupling_distance(matrix: np.ndarray) -> np.ndarray:
    """Distance that treats strong coupling of either sign as proximity."""

    distance = 1.0 - np.abs(matrix)
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, 0.0)


def signed_distance(matrix: np.ndarray) -> np.ndarray:
    """Distance that only counts positive co-activation as proximity."""

    distance = np.sqrt(np.maximum(2.0 * (1.0 - matrix), 0.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def classical_mds(distance: np.ndarray, components: int = 2) -> np.ndarray:
    squared = distance**2
    n = len(squared)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:components]
    scale = np.sqrt(np.maximum(values[order], 0.0))
    return vectors[:, order] * scale


def procrustes_disparity(left: np.ndarray, right: np.ndarray) -> float:
    """Scale/rotation/reflection invariant disagreement between two 2D layouts."""

    from scipy.spatial import procrustes

    _, _, disparity = procrustes(left, right)
    return float(disparity)


def neighbour_overlap(distance: np.ndarray, embedding: np.ndarray, k: int = 10) -> float:
    """Fraction of each cell's k nearest neighbours in the matrix kept in the layout."""

    from scipy.spatial.distance import squareform, pdist

    layout = squareform(pdist(embedding))
    overlap = []
    for row in range(len(distance)):
        left = set(np.argsort(distance[row])[1 : k + 1].tolist())
        right = set(np.argsort(layout[row])[1 : k + 1].tolist())
        overlap.append(len(left & right) / k)
    return float(np.mean(overlap))
