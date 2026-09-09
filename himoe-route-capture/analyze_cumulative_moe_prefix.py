"""Cumulative, label-free analysis of early MoE routes across control calls.

Only inference-time quantities are used: each call's initial flow noise and the
first router evaluations.  Final actions and rollout outcomes are deliberately
excluded.  The 16 sibling candidates at a query are an unordered noise cloud;
only candidate zero continues to the next physical state.  Cloud means are used
to estimate the state-conditioned routing backbone without inventing temporal
identity for the other candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import zarr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


N_EXPERTS = 32
LIVE_DIMS = 7
ROUTE_DEPTHS = (1, 2, 3, 5, 10)
CONTROL_ENDPOINTS = (2, 4, 6, 8, 10)
PRIMARY_DEPTH = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = rankdata(left).astype(np.float64)
    right = rankdata(right).astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / scale) if scale > 1e-20 else float("nan")


def _partial_rank_correlation(
    left: np.ndarray, right: np.ndarray, controls: list[np.ndarray]
) -> float:
    left = rankdata(left).astype(np.float64)
    right = rankdata(right).astype(np.float64)
    design = np.column_stack(
        [np.ones(len(left))]
        + [rankdata(control).astype(np.float64) for control in controls]
    )
    left -= design @ np.linalg.lstsq(design, left, rcond=None)[0]
    right -= design @ np.linalg.lstsq(design, right, rcond=None)[0]
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / scale) if scale > 1e-20 else float("nan")


def _route_embedding(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, None)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-20)
    sites = int(np.prod(probability.shape[-4:-1]))
    return np.sqrt(probability).reshape(*probability.shape[:-4], -1) / math.sqrt(
        2.0 * sites
    )


def _euclidean_embedding(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(*values.shape[:-2], -1)
    return flat / math.sqrt(flat.shape[-1])


def _load(run: pathlib.Path) -> dict:
    paths = sorted(run.glob("flow_traces_part*.npz"))
    if not paths:
        raise FileNotFoundError("no flow traces under %s" % run)
    parts = [np.load(path) for path in paths]
    trajectory = np.concatenate([part["x_traj"] for part in parts])
    query_id = np.concatenate([part["query_id"] for part in parts]).astype(np.int64)
    candidate_id = np.concatenate([part["candidate_id"] for part in parts]).astype(np.int64)
    if trajectory.ndim == 5 and trajectory.shape[2] == 1:
        trajectory = trajectory[:, :, 0]
    if trajectory.shape[1:] != (11, 10, 24):
        raise ValueError("unexpected trajectory shape %s" % (trajectory.shape,))

    records = json.loads((run / "query_records.json").read_text())
    records = sorted(records, key=lambda row: int(row["query_id"]))
    by_query = {int(row["query_id"]): row for row in records}
    rollouts = sorted({int(row["flow_noise_seed"]) for row in records})
    controls = sorted({int(row["control_step"]) for row in records})
    if len(rollouts) != 4 or controls != list(range(11)):
        raise ValueError("expected a complete 4-rollout x 11-control grid")

    row_grid = np.empty((len(rollouts), len(controls), 16), dtype=np.int64)
    state = np.empty((len(rollouts), len(controls), 8), dtype=np.float64)
    query_grid = np.empty((len(rollouts), len(controls)), dtype=np.int64)
    for rollout_axis, rollout in enumerate(rollouts):
        for control in controls:
            matches = [
                row
                for row in records
                if int(row["flow_noise_seed"]) == rollout
                and int(row["control_step"]) == control
            ]
            if len(matches) != 1:
                raise ValueError("missing or duplicate rollout/control cell")
            query = int(matches[0]["query_id"])
            rows = np.flatnonzero(query_id == query)
            rows = rows[np.argsort(candidate_id[rows])]
            if len(rows) != 16 or not np.array_equal(candidate_id[rows], np.arange(16)):
                raise ValueError("query %d is not a complete sibling cloud" % query)
            row_grid[rollout_axis, control] = rows
            state[rollout_axis, control] = np.asarray(by_query[query]["state"])
            query_grid[rollout_axis, control] = query

    route_group = zarr.open_group(str(run / "routes.zarr"), mode="r")
    if not np.array_equal(
        np.asarray(route_group["episode_id"][:], dtype=np.int64), query_id
    ):
        raise ValueError("route and flow rows are not aligned")
    probability = np.asarray(
        route_group["hb_router_probs"][:, :, :, 1:, :], dtype=np.float32
    )
    probability = np.clip(probability, 0.0, None)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-20)
    probability = probability[row_grid]
    if probability.shape != (4, 11, 16, 8, 10, 10, N_EXPERTS):
        raise ValueError("unexpected arranged routing shape %s" % (probability.shape,))

    metadata = json.loads((run / "server_metadata.json").read_text())
    layer_numbers = [int(value) for value in metadata["routing_hb_layer_indices"]]
    as_probability = np.asarray(route_group["as_probs"][:], dtype=np.float32)
    return {
        "trajectory": np.asarray(trajectory[row_grid], dtype=np.float32),
        "probability": probability,
        "rollouts": rollouts,
        "controls": controls,
        "query_grid": query_grid,
        "state": state,
        "layer_numbers": layer_numbers,
        "probability_sum_max_error_before_renormalization": float(
            np.max(
                np.abs(
                    np.asarray(
                        route_group["hb_router_probs"][:, :, :, 1:, :],
                        dtype=np.float32,
                    ).sum(axis=-1)
                    - 1.0
                )
            )
        ),
        "as_probability_std": float(as_probability.std(axis=0).mean()),
    }


def _flat_metadata(data: dict) -> tuple[np.ndarray, np.ndarray]:
    rollout = np.repeat(np.asarray(data["rollouts"], dtype=np.int64), 11)
    control = np.tile(np.asarray(data["controls"], dtype=np.int64), 4)
    return rollout, control


def _phase_geometry(embedding: np.ndarray, data: dict) -> dict:
    flat = embedding.reshape(44, -1)
    distance = pdist(flat)
    rollout, control = _flat_metadata(data)
    upper = np.triu_indices(44, 1)
    cross = rollout[upper[0]] != rollout[upper[1]]
    same = control[upper[0]] == control[upper[1]]
    gap = np.abs(control[upper[0]] - control[upper[1]])
    matrix = squareform(distance)
    nearest = []
    for row in range(44):
        candidates = np.flatnonzero(rollout != rollout[row])
        neighbor = candidates[np.argmin(matrix[row, candidates])]
        nearest.append(neighbor)
    nearest = np.asarray(nearest, dtype=np.int64)
    return {
        "phase_gap_rho": _rank_correlation(distance[cross], gap[cross]),
        "same_control_distance": float(distance[cross & same].mean()),
        "different_control_distance": float(distance[cross & ~same].mean()),
        "same_to_different_ratio": float(
            distance[cross & same].mean() / distance[cross & ~same].mean()
        ),
        "nearest_control_exact": float(np.mean(control == control[nearest])),
        "nearest_control_mae": float(np.mean(np.abs(control - control[nearest]))),
        "distance": distance,
        "nearest": nearest,
    }


def _phase_snr(embedding: np.ndarray, endpoint: int) -> dict:
    values = embedding[:, : endpoint + 1].reshape(4, endpoint + 1, -1)
    phase_mean = values.mean(axis=0)
    grand_mean = phase_mean.mean(axis=0)
    between = float(np.mean(np.sum(np.square(phase_mean - grand_mean), axis=1)))
    within = float(
        np.mean(np.sum(np.square(values - phase_mean[None, :, :]), axis=2))
    )
    ratio = between / max(within, 1e-30)
    return {
        "between_to_within": ratio,
        "icc": ratio / (1.0 + ratio),
        "between": between,
        "within": within,
    }


def _phase_label_permutation(
    geometry: dict,
    data: dict,
    state_distance: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    rollout, control = _flat_metadata(data)
    upper = np.triu_indices(44, 1)
    cross = rollout[upper[0]] != rollout[upper[1]]
    distance = geometry["distance"]
    observed_gap = np.abs(control[upper[0]] - control[upper[1]])
    observed_partial = _partial_rank_correlation(
        distance[cross], observed_gap[cross], [state_distance[cross]]
    )
    null_rho = np.empty(permutations, dtype=np.float64)
    null_partial = np.empty(permutations, dtype=np.float64)
    null_exact = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted_control = control.copy()
        for rollout_id in data["rollouts"]:
            selected = np.flatnonzero(rollout == rollout_id)
            permuted_control[selected] = rng.permutation(permuted_control[selected])
        gap = np.abs(permuted_control[upper[0]] - permuted_control[upper[1]])
        null_rho[draw] = _rank_correlation(distance[cross], gap[cross])
        null_partial[draw] = _partial_rank_correlation(
            distance[cross], gap[cross], [state_distance[cross]]
        )
        null_exact[draw] = np.mean(
            permuted_control == permuted_control[geometry["nearest"]]
        )

    def result(observed: float, null: np.ndarray, upper_tail: bool = False) -> dict:
        if upper_tail:
            p_value = (1.0 + float(np.sum(null >= observed))) / (permutations + 1.0)
        else:
            center = float(null.mean())
            p_value = (
                1.0 + float(np.sum(np.abs(null - center) >= abs(observed - center)))
            ) / (permutations + 1.0)
        return {
            "observed": observed,
            "null_mean": float(null.mean()),
            "null_95": [float(value) for value in np.percentile(null, [2.5, 97.5])],
            "p": p_value,
        }

    return {
        "phase_gap_rho": result(geometry["phase_gap_rho"], null_rho),
        "phase_gap_rho_given_robot_state": result(observed_partial, null_partial),
        "nearest_control_exact": result(
            geometry["nearest_control_exact"], null_exact, upper_tail=True
        ),
        "permutations": permutations,
    }


def _snr_permutation(
    embedding: np.ndarray,
    endpoint: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    observed = _phase_snr(embedding, endpoint)["between_to_within"]
    selected = embedding[:, : endpoint + 1]
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted = np.stack(
            [row[rng.permutation(endpoint + 1)] for row in selected], axis=0
        )
        null[draw] = _phase_snr(permuted, endpoint)["between_to_within"]
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_95": [float(value) for value in np.percentile(null, [2.5, 97.5])],
        "p": (1.0 + float(np.sum(null >= observed))) / (permutations + 1.0),
        "permutations": permutations,
    }


def _candidate_permutation(
    route_distance: np.ndarray,
    noise_distance: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    observed_per_query = np.empty((4, 11), dtype=np.float64)
    for rollout in range(4):
        for control in range(11):
            observed_per_query[rollout, control] = _rank_correlation(
                route_distance[rollout, control], noise_distance[rollout, control]
            )
    observed = float(observed_per_query.mean())
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        values = []
        for rollout in range(4):
            for control in range(11):
                matrix = squareform(route_distance[rollout, control])
                order = rng.permutation(16)
                permuted = squareform(matrix[np.ix_(order, order)], checks=False)
                values.append(
                    _rank_correlation(permuted, noise_distance[rollout, control])
                )
        null[draw] = np.mean(values)
    center = float(null.mean())
    return {
        "mean": observed,
        "per_rollout": observed_per_query.mean(axis=1).tolist(),
        "null_mean": center,
        "null_95": [float(value) for value in np.percentile(null, [2.5, 97.5])],
        "p": (
            1.0 + float(np.sum(np.abs(null - center) >= abs(observed - center)))
        )
        / (permutations + 1.0),
        "permutations": permutations,
    }


def _aligned_peak_test(
    values: np.ndarray,
    control_offset: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    standardized = (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-20
    )
    curve = standardized.mean(axis=0)
    observed = float(curve.max())
    peak = int(curve.argmax()) + control_offset
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted = np.stack(
            [row[rng.permutation(values.shape[1])] for row in standardized], axis=0
        )
        null[draw] = float(permuted.mean(axis=0).max())
    return {
        "peak_control": peak,
        "peak_standardized_mean": observed,
        "standardized_curve": curve.tolist(),
        "raw_mean_curve": values.mean(axis=0).tolist(),
        "max_time_corrected_p": (1.0 + float(np.sum(null >= observed)))
        / (permutations + 1.0),
        "permutations": permutations,
    }


def _pca(embedding: np.ndarray, data: dict, permutations: int, rng: np.random.Generator) -> dict:
    feature = embedding.reshape(44, -1)
    feature -= feature.mean(axis=0, keepdims=True)
    left, singular, _ = np.linalg.svd(feature, full_matrices=False)
    variance = np.square(singular)
    variance /= variance.sum()
    scores = left * singular[None, :]
    rollout, control = _flat_metadata(data)
    components = []
    for component in range(5):
        matched = []
        for left_rollout in range(4):
            for right_rollout in range(left_rollout + 1, 4):
                matched.append(
                    _rank_correlation(
                        scores[rollout == data["rollouts"][left_rollout], component],
                        scores[rollout == data["rollouts"][right_rollout], component],
                    )
                )
        components.append(
            {
                "component": component + 1,
                "explained_variance": float(variance[component]),
                "control_rho": _rank_correlation(scores[:, component], control),
                "matched_rollout_rho_mean": float(np.mean(matched)),
                "matched_rollout_rho_range": [float(min(matched)), float(max(matched))],
            }
        )
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        permuted_control = control.copy()
        for rollout_id in data["rollouts"]:
            selected = np.flatnonzero(rollout == rollout_id)
            permuted_control[selected] = rng.permutation(permuted_control[selected])
        null[draw] = _rank_correlation(scores[:, 0], permuted_control)
    observed = components[0]["control_rho"]
    center = float(null.mean())
    return {
        "components": components,
        "first_five_explained_variance": float(variance[:5].sum()),
        "pc1_control_permutation_p": (
            1.0 + float(np.sum(np.abs(null - center) >= abs(observed - center)))
        )
        / (permutations + 1.0),
    }


def analyze(data: dict, permutations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    probability = data["probability"]
    noise = data["trajectory"][:, :, :, 0]
    cloud_mean = probability.mean(axis=2)
    executed = probability[:, :, 0]
    noise7_embedding = _euclidean_embedding(noise[:, :, 0, :, :LIVE_DIMS])
    state = data["state"]
    state = (state - state.mean(axis=(0, 1), keepdims=True)) / np.maximum(
        state.std(axis=(0, 1), keepdims=True), 1e-20
    )
    state_embedding = state.reshape(4, 11, -1) / math.sqrt(state.shape[-1])
    state_distance = pdist(state_embedding.reshape(44, -1))

    route_distance_by_depth = {}
    noise7_pair = np.empty((4, 11, 120), dtype=np.float64)
    noise24_pair = np.empty_like(noise7_pair)
    for rollout in range(4):
        for control in range(11):
            noise7_pair[rollout, control] = pdist(
                _euclidean_embedding(noise[rollout, control, :, :, :LIVE_DIMS])
            )
            noise24_pair[rollout, control] = pdist(
                _euclidean_embedding(noise[rollout, control])
            )

    depth_results = []
    for depth in ROUTE_DEPTHS:
        center_embedding = _route_embedding(cloud_mean[:, :, :, :depth])
        executed_embedding = _route_embedding(executed[:, :, :, :depth])
        center_geometry = _phase_geometry(center_embedding, data)
        executed_geometry = _phase_geometry(executed_embedding, data)
        route_pair = np.empty((4, 11, 120), dtype=np.float64)
        for rollout in range(4):
            for control in range(11):
                route_pair[rollout, control] = pdist(
                    _route_embedding(probability[rollout, control, :, :, :depth])
                )
        route_distance_by_depth[depth] = route_pair
        coupling7 = np.mean(
            [
                _rank_correlation(route_pair[r, t], noise7_pair[r, t])
                for r in range(4)
                for t in range(11)
            ]
        )
        coupling24 = np.mean(
            [
                _rank_correlation(route_pair[r, t], noise24_pair[r, t])
                for r in range(4)
                for t in range(11)
            ]
        )
        depth_results.append(
            {
                "early_route_count": depth,
                "cloud_center_phase": {
                    key: value
                    for key, value in center_geometry.items()
                    if key not in ("distance", "nearest")
                },
                "executed_phase": {
                    key: value
                    for key, value in executed_geometry.items()
                    if key not in ("distance", "nearest")
                },
                "within_query_noise7_route_rho": float(coupling7),
                "within_query_noise24_route_rho": float(coupling24),
            }
        )

    primary_center = _route_embedding(cloud_mean[:, :, :, :PRIMARY_DEPTH])
    primary_executed = _route_embedding(executed[:, :, :, :PRIMARY_DEPTH])
    center_geometry = _phase_geometry(primary_center, data)
    executed_geometry = _phase_geometry(primary_executed, data)
    noise7_geometry = _phase_geometry(noise7_embedding, data)
    state_geometry = _phase_geometry(state_embedding, data)

    cumulative = []
    rollout, control = _flat_metadata(data)
    upper = np.triu_indices(44, 1)
    cross = rollout[upper[0]] != rollout[upper[1]]
    gap = np.abs(control[upper[0]] - control[upper[1]])
    for endpoint in CONTROL_ENDPOINTS:
        pair_selection = (
            cross
            & (control[upper[0]] <= endpoint)
            & (control[upper[1]] <= endpoint)
        )
        cumulative.append(
            {
                "control_endpoint": endpoint,
                "cloud_center_phase_rho": _rank_correlation(
                    center_geometry["distance"][pair_selection], gap[pair_selection]
                ),
                "executed_phase_rho": _rank_correlation(
                    executed_geometry["distance"][pair_selection], gap[pair_selection]
                ),
                "initial_noise7_phase_rho": _rank_correlation(
                    noise7_geometry["distance"][pair_selection], gap[pair_selection]
                ),
                "cloud_center_phase_snr": _phase_snr(primary_center, endpoint),
                "executed_phase_snr": _phase_snr(primary_executed, endpoint),
                "initial_noise7_phase_snr": _phase_snr(noise7_embedding, endpoint),
            }
        )

    # Noise-conditioned routing information and top-1 stability.
    selected_probability = probability[:, :, :, :, :PRIMARY_DEPTH]
    entropy = -np.sum(
        selected_probability * np.log(np.maximum(selected_probability, 1e-20)), axis=-1
    )
    mean_probability = selected_probability.mean(axis=2)
    mean_entropy = -np.sum(
        mean_probability * np.log(np.maximum(mean_probability, 1e-20)), axis=-1
    )
    js_by_layer = (
        mean_entropy - entropy.mean(axis=2)
    ).mean(axis=(3, 4)) / math.log(N_EXPERTS)
    normalized_entropy_by_layer = entropy.mean(axis=(2, 4, 5)) / math.log(N_EXPERTS)
    ordered = np.sort(selected_probability, axis=-1)
    margin_by_layer = (ordered[..., -1] - ordered[..., -2]).mean(axis=(2, 4, 5))
    top1 = np.argmax(selected_probability, axis=-1)
    consensus_by_layer = np.empty((4, 11, 8), dtype=np.float64)
    for r in range(4):
        for t in range(11):
            for layer in range(8):
                values = []
                for denoise in range(PRIMARY_DEPTH):
                    for token in range(10):
                        counts = np.bincount(
                            top1[r, t, :, layer, denoise, token], minlength=N_EXPERTS
                        )
                        values.append(float(counts.max()) / 16.0)
                consensus_by_layer[r, t, layer] = float(np.mean(values))

    cloud_dispersion = np.empty((4, 11), dtype=np.float64)
    for r in range(4):
        for t in range(11):
            candidate_embedding = _route_embedding(selected_probability[r, t])
            center = _route_embedding(mean_probability[r, t][None])[0]
            cloud_dispersion[r, t] = float(
                np.mean(np.linalg.norm(candidate_embedding - center[None], axis=-1))
            )
    center_drift = np.linalg.norm(
        primary_center[:, 1:] - primary_center[:, :-1], axis=-1
    )

    layer_results = []
    for layer_axis, layer in enumerate(data["layer_numbers"]):
        layer_center = _route_embedding(
            cloud_mean[:, :, layer_axis : layer_axis + 1, :PRIMARY_DEPTH]
        )
        layer_geometry = _phase_geometry(layer_center, data)
        snr = _phase_snr(layer_center, 10)
        layer_results.append(
            {
                "layer": int(layer),
                "phase_gap_rho": layer_geometry["phase_gap_rho"],
                "phase_snr": snr["between_to_within"],
                "phase_icc": snr["icc"],
                "noise_route_js_normalized": float(js_by_layer[:, :, layer_axis].mean()),
                "noise_route_js_bits_per_site": float(
                    js_by_layer[:, :, layer_axis].mean() * math.log2(N_EXPERTS)
                ),
                "normalized_router_entropy": float(
                    normalized_entropy_by_layer[:, :, layer_axis].mean()
                ),
                "top1_margin": float(margin_by_layer[:, :, layer_axis].mean()),
                "top1_consensus_across_noise": float(
                    consensus_by_layer[:, :, layer_axis].mean()
                ),
            }
        )

    permutation_tests = {
        "cloud_center_phase": _phase_label_permutation(
            center_geometry, data, state_distance, permutations, rng
        ),
        "executed_phase": _phase_label_permutation(
            executed_geometry, data, state_distance, permutations, rng
        ),
        "cloud_center_phase_snr": _snr_permutation(
            primary_center, 10, permutations, rng
        ),
        "executed_phase_snr": _snr_permutation(
            primary_executed, 10, permutations, rng
        ),
        "noise7_route_coupling": _candidate_permutation(
            route_distance_by_depth[PRIMARY_DEPTH], noise7_pair, permutations, rng
        ),
        "noise24_route_coupling": _candidate_permutation(
            route_distance_by_depth[PRIMARY_DEPTH], noise24_pair, permutations, rng
        ),
    }

    change_points = {
        "noise_conditioned_route_dispersion": _aligned_peak_test(
            cloud_dispersion, 0, permutations, rng
        ),
        "noise_route_js": _aligned_peak_test(
            js_by_layer.mean(axis=2), 0, permutations, rng
        ),
        "route_centroid_drift": _aligned_peak_test(
            center_drift, 1, permutations, rng
        ),
    }

    return {
        "experiment": "cumulative_early_moe_route_prefix",
        "data": {
            "task": "LIBERO-Goal task 0: open the middle drawer of the cabinet",
            "initial_state_id": 24,
            "rollouts": 4,
            "controls_per_rollout": 11,
            "noise_candidates_per_query": 16,
            "primary_early_routes": [0, 1, 2],
            "hb_layers": data["layer_numbers"],
            "action_tokens": 10,
            "router_experts": N_EXPERTS,
        },
        "normalization": {
            "router": "renormalize every layer x denoise x token 32-way probability to sum 1",
            "cross_layer": "equal-site Hellinger embedding; each layer/token has the same bounded weight",
            "noise": "x^(0) is the model-standard normal flow input; RMS Euclidean distance",
            "layer_comparison": "rank correlations and between/within ratios are scale invariant",
            "stored_probability_sum_max_error_before_renormalization": data[
                "probability_sum_max_error_before_renormalization"
            ],
        },
        "constraints": {
            "uses_final_action": False,
            "uses_outcome": False,
            "trains_supervised_decoder": False,
            "cloud_candidate_temporal_identity": False,
            "candidate_zero_is_the_only_executed_temporal_path": True,
            "as_probability_std": data["as_probability_std"],
        },
        "route_depth_comparison": depth_results,
        "primary_phase_geometry": {
            "cloud_center": {
                key: value
                for key, value in center_geometry.items()
                if key not in ("distance", "nearest")
            },
            "executed": {
                key: value
                for key, value in executed_geometry.items()
                if key not in ("distance", "nearest")
            },
            "initial_noise7": {
                key: value
                for key, value in noise7_geometry.items()
                if key not in ("distance", "nearest")
            },
            "robot_state_control": {
                key: value
                for key, value in state_geometry.items()
                if key not in ("distance", "nearest")
            },
        },
        "cumulative_control_prefix": cumulative,
        "noise_route_information": {
            "mean_normalized_js": float(js_by_layer.mean()),
            "mean_bits_per_layer_token_site": float(
                js_by_layer.mean() * math.log2(N_EXPERTS)
            ),
            "mean_normalized_entropy": float(normalized_entropy_by_layer.mean()),
            "mean_top1_margin": float(margin_by_layer.mean()),
            "mean_top1_consensus_across_noise": float(consensus_by_layer.mean()),
        },
        "layer_structure": layer_results,
        "change_points": change_points,
        "pca": _pca(primary_center, data, permutations, rng),
        "permutation_tests": permutation_tests,
    }


def _fmt(value: float) -> str:
    return "%.3f" % value


def render_report(summary: dict) -> str:
    data = summary["data"]
    phase = summary["primary_phase_geometry"]
    tests = summary["permutation_tests"]
    info = summary["noise_route_information"]
    changes = summary["change_points"]
    pca = summary["pca"]
    lines = [
        "# 累计控制前缀中的 MoE 早期路由规律",
        "",
        "## 实验思路",
        "",
        "- 使用 %d 条 rollout x %d 个连续 control；每次推理只取初始噪声 `x^(0)` 和前三个 router evaluation `tau0-tau2`。"
        % (data["rollouts"], data["controls_per_rollout"]),
        "- 每个 query 的 16 个 sibling noise 不跨 control 串联。candidate 0 是真实连续执行路径；16-noise 路由云中心只用于估计去噪声后的共享路由骨架。",
        "- 每个 layer x denoise x token 的 32-way probability 重新归一到和为 1，再使用等 site 权重的 Hellinger 表示；没有最终动作、成功标签或监督 decoder。",
        "- 对 control `0...C` 做累计方差分解、跨 rollout 阶段对齐、候选级噪声-路由耦合、PCA 和保持结构的置乱检验。",
        "",
        "## 结果",
        "",
        "### 累计 control prefix",
        "",
        "| 截止 control C | 云中心-阶段差 rho | 单次执行路由-阶段差 rho | 初始噪声-阶段差 rho | 云中心 phase ICC | 单次执行 phase ICC | 初始噪声 phase ICC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cumulative_control_prefix"]:
        lines.append(
            "| %d | %s | %s | %s | %s | %s | %s |"
            % (
                row["control_endpoint"],
                _fmt(row["cloud_center_phase_rho"]),
                _fmt(row["executed_phase_rho"]),
                _fmt(row["initial_noise7_phase_rho"]),
                _fmt(row["cloud_center_phase_snr"]["icc"]),
                _fmt(row["executed_phase_snr"]["icc"]),
                _fmt(row["initial_noise7_phase_snr"]["icc"]),
            )
        )

    lines.extend(
        [
            "",
            "完整前缀下，云中心 phase SNR=%s（ICC=%s，置乱 p=%s），单次执行路由 SNR=%s（ICC=%s，p=%s）；初始噪声自身的阶段 rho 只有 %s。"
            % (
                _fmt(tests["cloud_center_phase_snr"]["observed"]),
                _fmt(summary["cumulative_control_prefix"][-1]["cloud_center_phase_snr"]["icc"]),
                _fmt(tests["cloud_center_phase_snr"]["p"]),
                _fmt(tests["executed_phase_snr"]["observed"]),
                _fmt(summary["cumulative_control_prefix"][-1]["executed_phase_snr"]["icc"]),
                _fmt(tests["executed_phase_snr"]["p"]),
                _fmt(phase["initial_noise7"]["phase_gap_rho"]),
            ),
            "",
            "跨 rollout 最近邻控制阶段：云中心为 %d/44，真实 candidate 0 为 %d/44，初始噪声为 %d/44。云中心的同阶段/异阶段 Hellinger 距离比为 %s。"
            % (
                round(44 * phase["cloud_center"]["nearest_control_exact"]),
                round(44 * phase["executed"]["nearest_control_exact"]),
                round(44 * phase["initial_noise7"]["nearest_control_exact"]),
                _fmt(phase["cloud_center"]["same_to_different_ratio"]),
            ),
            "",
            "### 前几个路由与初始噪声",
            "",
            "| 累计 early route 数 | 云中心阶段 rho | 单次执行阶段 rho | query 内 live7 noise-route rho | query 内 full24 noise-route rho |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["route_depth_comparison"]:
        lines.append(
            "| %d | %s | %s | %s | %s |"
            % (
                row["early_route_count"],
                _fmt(row["cloud_center_phase"]["phase_gap_rho"]),
                _fmt(row["executed_phase"]["phase_gap_rho"]),
                _fmt(row["within_query_noise7_route_rho"]),
                _fmt(row["within_query_noise24_route_rho"]),
            )
        )

    lines.extend(
        [
            "",
            "前三个路由下，live7 noise-route rho=%s（候选身份置乱 p=%s），full24 为 %s（p=%s）。第一条路由已经包含几乎全部阶段骨架；增加到三条主要提高单次执行路由的稳健性。"
            % (
                _fmt(tests["noise7_route_coupling"]["mean"]),
                _fmt(tests["noise7_route_coupling"]["p"]),
                _fmt(tests["noise24_route_coupling"]["mean"]),
                _fmt(tests["noise24_route_coupling"]["p"]),
            ),
            "",
            "### 小概率变化的放大",
            "",
            "router 的平均归一化熵=%s，平均 top-1/top-2 margin=%g，说明完整 32-way 分布非常接近均匀。初始噪声带来的 generalized-JS 信息只有 %.6f bit/site，但跨 8 层 x 3 路由 x 10 token 累积后形成稳定结构；top-1 在 16 个噪声间的平均共识率为 %s。"
            % (
                _fmt(info["mean_normalized_entropy"]),
                info["mean_top1_margin"],
                info["mean_bits_per_layer_token_site"],
                _fmt(info["mean_top1_consensus_across_noise"]),
            ),
            "",
            "这些 site 高度相关，不能把 bit/site 直接乘以 240 当作总互信息；这里的放大量由跨 rollout 复现度和置乱检验衡量。",
            "",
            "| HB layer | 阶段差 rho | phase ICC | noise-route JS (bit/site) | top-1 noise 共识率 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["layer_structure"]:
        lines.append(
            "| %d | %s | %s | %.6f | %s |"
            % (
                row["layer"],
                _fmt(row["phase_gap_rho"]),
                _fmt(row["phase_icc"]),
                row["noise_route_js_bits_per_site"],
                _fmt(row["top1_consensus_across_noise"]),
            )
        )

    pc1 = pca["components"][0]
    pc_rho = [entry["matched_rollout_rho_mean"] for entry in pca["components"]]
    lines.extend(
        [
            "",
            "浅层 2-5 的路由距离更像平滑进度轴，深层 12-15 的同阶段复现度更高但不再随时间差单调变化，表现为更离散的阶段 motif。无监督 PCA 的前 5 维解释 %.3f 方差；PC1 与 control 的 rho=%s（置乱 p=%s），前 5 个 PC 的跨 rollout 同步 rho 范围为 %s-%s。PCA 在全部 44 个 query 上联合拟合，因此只作描述；主检验仍是原始 Hellinger 距离的置乱结果。"
            % (
                pca["first_five_explained_variance"],
                _fmt(pc1["control_rho"]),
                _fmt(pca["pc1_control_permutation_p"]),
                _fmt(min(pc_rho)),
                _fmt(max(pc_rho)),
            ),
            "",
            "### 共同变化点",
            "",
            "只看路由，control %d 同时是噪声条件路由离散度峰值（max-time corrected p=%s）、noise-route JS 峰值（p=%s）和路由云中心最大漂移点（p=%s）。离散度与 JS 数学上相关，不算两份独立证据；云中心漂移给出了另一侧验证。累计 phase ICC 也从 C=7 附近的高位在纳入 C=8 后下降，说明这里是可复现的内部计算阶段切换。"
            % (
                changes["noise_conditioned_route_dispersion"]["peak_control"],
                _fmt(
                    changes["noise_conditioned_route_dispersion"][
                        "max_time_corrected_p"
                    ]
                ),
                _fmt(changes["noise_route_js"]["max_time_corrected_p"]),
                _fmt(changes["route_centroid_drift"]["max_time_corrected_p"]),
            ),
            "",
            "## 结论",
            "",
            "早期 MoE 路由可以分成两部分：`route(t, noise) = phase backbone(t) + noise response(t, noise)`。phase backbone 在四条 rollout 上高度复现；noise response 虽然每个 site 只有约 1e-3 bit 量级，却在大量层/token 上一致积累，并在 control 8 被同步放大。这个结构支持把路由解释为“随状态推进的内部计算阶段 + 当前噪声条件下的局部路径选择”，而不是单一专家的固定语义或提前决定的最终动作。",
            "",
            "限制：只有一个任务、一个初始场景和 4 条 rollout；robot-state 距离与阶段差的 rho=%s，控制 robot state 后云中心路由的剩余阶段 rho=%s（p=%s），所以 phase backbone 很可能主要是状态进度编码，不能称为独立内部时钟。当前数据仍只有 router probability，没有真实 expert output norm。"
            % (
                _fmt(phase["robot_state_control"]["phase_gap_rho"]),
                _fmt(
                    tests["cloud_center_phase"][
                        "phase_gap_rho_given_robot_state"
                    ]["observed"]
                ),
                _fmt(
                    tests["cloud_center_phase"][
                        "phase_gap_rho_given_robot_state"
                    ]["p"]
                ),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run = pathlib.Path(args.run).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(_load(run), args.permutations, args.seed)
    summary["run"] = str(run)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / "report.md").write_text(render_report(summary))
    print("wrote %s" % (out_dir / "summary.json"), flush=True)
    print("wrote %s" % (out_dir / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
