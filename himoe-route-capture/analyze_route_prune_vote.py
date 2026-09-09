"""Shadow evaluation of early MoE-route pruning followed by action voting.

This analysis is deliberately retrospective.  Every candidate in the source RAD
trace was fully denoised; the script simulates an action-consensus proxy for keeping
only a route-central subset after the first ``r`` flow steps.  It does not claim
that routing predicts task success and it does not implement online pruning.

The two-stage rule is:

1. Reconstruct the normalized top-k expert distribution at every aligned
   ``[flow, layer, action-token]`` site.
2. Average pairwise total-variation distance over the first ``r`` flow steps and
   keep the ``M`` route-medoid candidates (the lowest row sums).
3. Let those candidates finish, then select their checkpoint-normalized action
   medoid over the complete ``10 x 7`` chunks.

Initial-noise centrality and the exact expectation over all random subsets are
reported as baselines.  Statistics are clustered by task, not by candidate pair or
replan state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "himoe-route-prune-vote-shadow-v1"
EXPERT_COUNT = 32
REGRET_THRESHOLDS = (0.02, 0.05, 0.10)
TIE_ATOL = 1e-12
HERE = Path(__file__).resolve().parent
DEFAULT_BATCH_DIR = (
    HERE.parent
    / "himoe-vla-cache/himoe-libero-bridge/formal-artifacts/rad-goal-10x5-tv-v4"
)
DEFAULT_OUT = HERE / "analysis/route-prune-vote/goal-v4.json"


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    """Return pairwise RMS distance for ``[candidate, ...]`` values."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite [candidate, ...] array")
    flat = array.reshape(array.shape[0], -1)
    delta = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.mean(np.square(delta), axis=-1))


def action_distance(actions: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """Checkpoint-std-normalized RMS distance between complete action chunks."""

    chunks = np.asarray(actions, dtype=np.float64)
    scale = np.asarray(action_std, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[0] < 2:
        raise ValueError("actions must have shape [candidate, chunk-step, action-dim]")
    if scale.shape != (chunks.shape[-1],):
        raise ValueError("action_std does not match the action dimension")
    if not np.all(np.isfinite(chunks)) or not np.all(np.isfinite(scale)):
        raise ValueError("actions and action_std must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("action_std must be strictly positive")
    return pairwise_rms(chunks / scale[None, None, :])


def normalized_route_distributions(
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    expert_count: int = EXPERT_COUNT,
) -> np.ndarray:
    """Scatter sparse top-k routes and L1-normalize each aligned routing site."""

    ids = np.asarray(expert_ids)
    weights = np.asarray(expert_weights, dtype=np.float64)
    if ids.shape != weights.shape or ids.ndim != 5:
        raise ValueError(
            "routes must have shape [candidate, flow, layer, action-token, top-k]"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expert_ids must be integers")
    if np.any(ids < 0) or np.any(ids >= expert_count):
        raise ValueError("expert_ids contains an out-of-range expert")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("expert_ids must be unique within every top-k site")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("expert_weights must be finite and non-negative")

    dense = np.zeros(ids.shape[:-1] + (expert_count,), dtype=np.float64)
    np.put_along_axis(dense, ids.astype(np.int64, copy=False), weights, axis=-1)
    mass = dense.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every routing site must have positive top-k mass")
    return dense / mass


def prefix_route_distances(
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    expert_count: int = EXPERT_COUNT,
) -> np.ndarray:
    """Return mean aligned TV matrices for every flow prefix.

    The result has shape ``[flow-prefix, candidate, candidate]``.  Index zero is
    the distance after one completed denoising step.
    """

    distributions = normalized_route_distributions(
        expert_ids, expert_weights, expert_count=expert_count
    )
    site_tv = 0.5 * np.abs(
        distributions[:, None, ...] - distributions[None, :, ...]
    ).sum(axis=-1)
    # [candidate, candidate, flow, layer, action-token] -> per-flow matrices.
    per_flow = site_tv.mean(axis=(3, 4))
    divisor = np.arange(1, per_flow.shape[-1] + 1, dtype=np.float64)
    cumulative = np.cumsum(per_flow, axis=-1) / divisor[None, None, :]
    return np.moveaxis(np.clip(cumulative, 0.0, 1.0), -1, 0)


def _lowest_tied_index(scores: np.ndarray, candidate_ids: np.ndarray) -> int:
    optimum = np.min(scores)
    tied = candidate_ids[
        np.isclose(scores, optimum, atol=TIE_ATOL, rtol=0.0)
    ]
    return int(np.min(tied))


def _ascending_tie_order(scores: np.ndarray) -> np.ndarray:
    """Order low scores first, resolving each 1e-12 tie toward the lowest ID."""

    values = np.asarray(scores, dtype=np.float64)
    remaining = np.arange(len(values), dtype=np.int64)
    order = []
    while len(remaining):
        local = values[remaining]
        tied = remaining[
            np.isclose(local, np.min(local), atol=TIE_ATOL, rtol=0.0)
        ]
        selected = int(np.min(tied))
        order.append(selected)
        remaining = remaining[remaining != selected]
    return np.asarray(order, dtype=np.int64)


def medoid_index(distance: np.ndarray, candidates: Iterable[int] | None = None) -> int:
    """Select a distance medoid, breaking numerical ties by candidate ID."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance must be finite")
    if candidates is None:
        indices = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        indices = np.unique(np.fromiter(candidates, dtype=np.int64))
    if len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(matrix)):
        raise ValueError("candidates is empty or out of range")
    subset = matrix[np.ix_(indices, indices)]
    return _lowest_tied_index(subset.sum(axis=1), indices)


def central_subset(distance: np.ndarray, survivor_count: int) -> np.ndarray:
    """Keep candidates with the lowest distance-centrality, with stable ties."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance must be finite")
    if not 1 <= survivor_count <= len(matrix):
        raise ValueError("survivor_count must be in [1, candidate_count]")
    return _ascending_tie_order(matrix.sum(axis=1))[:survivor_count]


def flow_compute_fraction(
    candidate_count: int,
    survivor_count: int,
    prefix_steps: int,
    flow_steps: int,
) -> float:
    """Idealized denoising-step fraction for K -> M pruning after r steps."""

    if not 1 <= survivor_count <= candidate_count:
        raise ValueError("invalid survivor_count")
    if not 0 <= prefix_steps <= flow_steps or flow_steps < 1:
        raise ValueError("invalid flow prefix")
    work = candidate_count * prefix_steps + survivor_count * (flow_steps - prefix_steps)
    return float(work / (candidate_count * flow_steps))


def random_retention_probability(
    candidate_count: int, survivor_count: int, target_count: int
) -> float:
    """Probability a uniform subset retains at least one target candidate."""

    if not 0 <= target_count <= candidate_count:
        raise ValueError("invalid target_count")
    if not 0 <= survivor_count <= candidate_count:
        raise ValueError("invalid survivor_count")
    total = math.comb(candidate_count, survivor_count)
    misses = (
        math.comb(candidate_count - target_count, survivor_count)
        if survivor_count <= candidate_count - target_count
        else 0
    )
    return float(1.0 - misses / total)


def low_distance_auc(score_distance: np.ndarray, target_distance: np.ndarray) -> float:
    """AUC for predicting below-median final-action pair distance."""

    score = np.asarray(score_distance, dtype=np.float64)
    target = np.asarray(target_distance, dtype=np.float64)
    if score.shape != target.shape or score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score_distance and target_distance must be equal square matrices")
    upper = np.triu_indices(len(score), 1)
    target_pairs = target[upper]
    labels = target_pairs < np.median(target_pairs)
    positive = score[upper][labels]
    negative = score[upper][~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = (positive[:, None] < negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _metric_key(threshold: float) -> str:
    return "regret_at_most_%s" % format(threshold, ".2f")


def subset_metrics(
    action_matrix: np.ndarray,
    survivors: Sequence[int],
    full_winner: int,
    full_top2: Sequence[int],
    regret_scale: float,
) -> Dict[str, float]:
    """Evaluate final action-medoid voting inside one survivor set."""

    keep = np.unique(np.asarray(survivors, dtype=np.int64))
    selected = medoid_index(action_matrix, keep)
    global_centrality = np.asarray(action_matrix, dtype=np.float64).mean(axis=1)
    regret = float(
        (global_centrality[selected] - global_centrality[full_winner]) / regret_scale
    )
    # The full medoid is the global optimum; clip tiny round-off below zero.
    regret = max(regret, 0.0)
    result = {
        "full_winner_retained": float(full_winner in keep),
        "any_full_top2_retained": float(bool(set(keep) & set(full_top2))),
        "selected_equals_full_winner": float(selected == full_winner),
        "normalized_global_medoid_regret": regret,
    }
    result.update(
        {_metric_key(threshold): float(regret <= threshold) for threshold in REGRET_THRESHOLDS}
    )
    return result


def exact_random_subset_metrics(
    action_matrix: np.ndarray,
    survivor_count: int,
    full_winner: int,
    full_top2: Sequence[int],
    regret_scale: float,
) -> Dict[str, float]:
    """Average subset-vote metrics over every equally likely survivor subset."""

    rows = [
        subset_metrics(action_matrix, subset, full_winner, full_top2, regret_scale)
        for subset in itertools.combinations(range(len(action_matrix)), survivor_count)
    ]
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    expected_winner = survivor_count / len(action_matrix)
    expected_top2 = random_retention_probability(
        len(action_matrix), survivor_count, len(full_top2)
    )
    if not np.isclose(result["full_winner_retained"], expected_winner):
        raise AssertionError("random winner-retention enumeration disagrees with combinatorics")
    if not np.isclose(result["any_full_top2_retained"], expected_top2):
        raise AssertionError("random top2-retention enumeration disagrees with combinatorics")
    return result


def _append_metrics(store: Dict[str, list[float]], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        store.setdefault(key, []).append(float(value))


def _task_means(values: Sequence[float], task_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    tasks = np.asarray(task_ids, dtype=np.int64)
    unique = np.unique(tasks)
    if len(array) != len(tasks) or not len(unique):
        raise ValueError("values and task_ids must be non-empty and aligned")
    means = np.asarray([array[tasks == task].mean() for task in unique], dtype=np.float64)
    return unique, means


def aggregate_metric(
    values: Sequence[float],
    task_ids: Sequence[int],
    bootstrap_indices: np.ndarray,
) -> Dict[str, object]:
    """Pool mean plus an equal-task macro mean and task-cluster bootstrap CI."""

    tasks, means = _task_means(values, task_ids)
    draws = means[bootstrap_indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "pool_weighted_mean": float(np.mean(values)),
        "task_macro_mean": float(means.mean()),
        "task_cluster_bootstrap_ci_95": [float(low), float(high)],
        "per_task_mean": {str(int(task)): float(value) for task, value in zip(tasks, means)},
    }


def aggregate_metrics(
    store: Mapping[str, Sequence[float]],
    task_ids: Sequence[int],
    bootstrap_indices: np.ndarray,
) -> Dict[str, object]:
    return {
        key: aggregate_metric(values, task_ids, bootstrap_indices)
        for key, values in store.items()
    }


def paired_differences(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    task_ids: Sequence[int],
    bootstrap_indices: np.ndarray,
) -> Dict[str, object]:
    if left.keys() != right.keys():
        raise ValueError("paired metric stores must have the same keys")
    return {
        key: aggregate_metric(
            np.asarray(left[key], dtype=np.float64) - np.asarray(right[key], dtype=np.float64),
            task_ids,
            bootstrap_indices,
        )
        for key in left
    }


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in raw.split(",") if item.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def _trace_files(batch_dir: Path) -> list[Path]:
    pattern = "cases/task*-init*/arms/rad-action-medoid-*/rad-trace.json"
    files = sorted(batch_dir.glob(pattern))
    if not files:
        raise FileNotFoundError("no action-medoid RAD traces under %s" % batch_dir)
    return files


def _pool_regret_scale(action_matrix: np.ndarray) -> float:
    upper = action_matrix[np.triu_indices(len(action_matrix), 1)]
    scale = float(np.median(upper))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("action pool has no positive median pair distance")
    return scale


def run_analysis(
    batch_dir: Path,
    survivors: Sequence[int],
    bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    files = _trace_files(batch_dir)
    task_ids: list[int] = []
    case_ids: set[tuple[int, int]] = set()
    cases_by_task: Dict[int, int] = {}
    pools_by_task: Dict[int, int] = {}
    route_values: Dict[tuple[int, int], Dict[str, list[float]]] = {}
    noise_values: Dict[int, Dict[str, list[float]]] = {
        count: {} for count in survivors
    }
    random_values: Dict[int, Dict[str, list[float]]] = {
        count: {} for count in survivors
    }
    route_auc: Dict[int, list[float]] = {}
    noise_auc: list[float] = []
    expected_shape: tuple[int, int, int, int, int] | None = None
    expected_std: np.ndarray | None = None
    stored_winner_mismatches = 0
    selector_score_mismatches = 0
    max_raw_weight_mass_error = 0.0
    candidate_count = flow_steps = layer_count = action_steps = action_dim = top_k = None

    for trace_path in files:
        metadata_doc = json.loads(trace_path.read_text())
        metadata = metadata_doc["metadata"]
        config = metadata["config"]
        if config["selector"] != "action_medoid":
            raise ValueError("unexpected selector in %s" % trace_path)
        if metadata["result"]["status"] != "completed":
            raise ValueError("incomplete trace in %s" % trace_path)
        task_id = int(config["task_id"])
        init_id = int(config["init_state_id"])
        case = (task_id, init_id)
        if case in case_ids:
            raise ValueError("duplicate action-medoid arm for task/init %s" % (case,))
        case_ids.add(case)
        cases_by_task[task_id] = cases_by_task.get(task_id, 0) + 1

        action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
        if expected_std is None:
            expected_std = action_std
        elif not np.array_equal(action_std, expected_std):
            raise ValueError("normalization_action_std changed across traces")

        array_path = trace_path.with_name(metadata_doc["array_file"])
        with np.load(array_path, allow_pickle=False) as arrays:
            actions_all = arrays["candidate_actions"]
            ids_all = arrays["expert_ids"]
            weights_all = arrays["expert_weights"]
            noise_all = arrays["noise_schedule"]
            selected_all = arrays["selected_indices"]
            selector_scores_all = arrays["selector_scores"]
            replan_count = len(actions_all)
            if not (
                replan_count
                == len(ids_all)
                == len(weights_all)
                == len(selected_all)
                == len(selector_scores_all)
                and len(noise_all) >= replan_count
            ):
                raise ValueError("replan arrays are not aligned in %s" % array_path)

            for actions, ids, weights, noise, stored_winner, stored_scores in zip(
                actions_all,
                ids_all,
                weights_all,
                noise_all[:replan_count],
                selected_all,
                selector_scores_all,
            ):
                shape = tuple(int(value) for value in ids.shape)
                if expected_shape is None:
                    expected_shape = shape
                    candidate_count, flow_steps, layer_count, action_steps, top_k = shape
                    action_dim = int(actions.shape[-1])
                    if action_std.shape != (action_dim,):
                        raise ValueError("normalization_action_std has the wrong shape")
                    if any(count < 2 or count >= candidate_count for count in survivors):
                        raise ValueError(
                            "survivor counts must be in [2, candidate_count - 1]"
                        )
                elif shape != expected_shape:
                    raise ValueError("routing shape changed across pools")
                if actions.shape != (candidate_count, action_steps, action_dim):
                    raise ValueError("unexpected action shape")
                if noise.shape != (candidate_count, action_steps, 24):
                    raise ValueError("unexpected noise shape")

                matrix = action_distance(actions, action_std)
                full_winner = medoid_index(matrix)
                action_scores = matrix.sum(axis=1)
                full_top2 = np.argsort(action_scores, kind="stable")[:2]
                regret_scale = _pool_regret_scale(matrix)
                stored_winner_mismatches += int(full_winner != int(stored_winner))
                selector_score_mismatches += int(
                    not np.allclose(action_scores, stored_scores, atol=1e-9, rtol=1e-9)
                )

                raw_mass = np.asarray(weights, dtype=np.float64).sum(axis=-1)
                max_raw_weight_mass_error = max(
                    max_raw_weight_mass_error, float(np.max(np.abs(raw_mass - 1.0)))
                )
                route_prefix = prefix_route_distances(ids, weights)
                # LIBERO's data mask has seven live action dimensions followed by padding.
                noise_matrix = pairwise_rms(noise[..., :action_dim])
                noise_auc.append(low_distance_auc(noise_matrix, matrix))

                task_ids.append(task_id)
                pools_by_task[task_id] = pools_by_task.get(task_id, 0) + 1
                for survivor_count in survivors:
                    noise_keep = central_subset(noise_matrix, survivor_count)
                    _append_metrics(
                        noise_values[survivor_count],
                        subset_metrics(
                            matrix,
                            noise_keep,
                            full_winner,
                            full_top2,
                            regret_scale,
                        ),
                    )
                    _append_metrics(
                        random_values[survivor_count],
                        exact_random_subset_metrics(
                            matrix,
                            survivor_count,
                            full_winner,
                            full_top2,
                            regret_scale,
                        ),
                    )

                for prefix_index, route_matrix in enumerate(route_prefix):
                    prefix_steps = prefix_index + 1
                    route_auc.setdefault(prefix_steps, []).append(
                        low_distance_auc(route_matrix, matrix)
                    )
                    for survivor_count in survivors:
                        keep = central_subset(route_matrix, survivor_count)
                        condition = (prefix_steps, survivor_count)
                        store = route_values.setdefault(condition, {})
                        _append_metrics(
                            store,
                            subset_metrics(
                                matrix,
                                keep,
                                full_winner,
                                full_top2,
                                regret_scale,
                            ),
                        )

    if stored_winner_mismatches or selector_score_mismatches:
        raise RuntimeError(
            "failed to reproduce stored action-medoid selector: winner mismatches=%d, "
            "score mismatches=%d"
            % (stored_winner_mismatches, selector_score_mismatches)
        )
    unique_tasks = sorted(set(task_ids))
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, len(unique_tasks), size=(bootstrap, len(unique_tasks)), endpoint=False
    )

    random_out = {}
    noise_out = {}
    for survivor_count in survivors:
        random_out[str(survivor_count)] = {
            "survivor_count": survivor_count,
            "can_prune_before_flow": True,
            "idealized_flow_compute_fraction": flow_compute_fraction(
                candidate_count, survivor_count, 0, flow_steps
            ),
            "metrics": aggregate_metrics(
                random_values[survivor_count], task_ids, bootstrap_indices
            ),
        }
        noise_out[str(survivor_count)] = {
            "survivor_count": survivor_count,
            "can_prune_before_flow": True,
            "idealized_flow_compute_fraction": flow_compute_fraction(
                candidate_count, survivor_count, 0, flow_steps
            ),
            "metrics": aggregate_metrics(
                noise_values[survivor_count], task_ids, bootstrap_indices
            ),
            "paired_difference_minus_exact_random": paired_differences(
                noise_values[survivor_count],
                random_values[survivor_count],
                task_ids,
                bootstrap_indices,
            ),
        }

    route_out = []
    for prefix_steps, survivor_count in sorted(route_values):
        values = route_values[(prefix_steps, survivor_count)]
        compute = flow_compute_fraction(
            candidate_count, survivor_count, prefix_steps, flow_steps
        )
        route_out.append(
            {
                "prefix_steps": prefix_steps,
                "survivor_count": survivor_count,
                "idealized_flow_compute_fraction": compute,
                "idealized_flow_saving_fraction": 1.0 - compute,
                "metrics": aggregate_metrics(values, task_ids, bootstrap_indices),
                "paired_difference_minus_exact_random": paired_differences(
                    values,
                    random_values[survivor_count],
                    task_ids,
                    bootstrap_indices,
                ),
                "paired_difference_minus_initial_noise": paired_differences(
                    values,
                    noise_values[survivor_count],
                    task_ids,
                    bootstrap_indices,
                ),
            }
        )

    route_signal = []
    for prefix_steps in sorted(route_auc):
        route_signal.append(
            {
                "prefix_steps": prefix_steps,
                "route_pair_auc": aggregate_metric(
                    route_auc[prefix_steps], task_ids, bootstrap_indices
                ),
                "paired_auc_difference_route_minus_initial_noise": aggregate_metric(
                    np.asarray(route_auc[prefix_steps]) - np.asarray(noise_auc),
                    task_ids,
                    bootstrap_indices,
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": (
            "Can an early HB-MoE routing prefix prune candidates while preserving "
            "the final full-K action-medoid consensus?"
        ),
        "data": {
            "batch_dir": str(batch_dir.resolve()),
            "arm_filter": "one action_medoid arm per task/init case",
            "case_count": len(files),
            "pool_count": len(task_ids),
            "task_count": len(unique_tasks),
            "cases_per_task": {str(key): value for key, value in sorted(cases_by_task.items())},
            "pools_per_task": {str(key): value for key, value in sorted(pools_by_task.items())},
            "candidate_count": candidate_count,
            "flow_steps": flow_steps,
            "routing_layers": layer_count,
            "action_chunk_steps": action_steps,
            "action_dim": action_dim,
            "routing_top_k": top_k,
            "routing_expert_count": EXPERT_COUNT,
            "normalization_action_std": expected_std.tolist(),
        },
        "definitions": {
            "route_prefix_distance": (
                "mean aligned total variation after per-site L1 normalization of "
                "top-k weights; axes flow/layer/action-token are preserved"
            ),
            "route_pruner": "keep M candidates with lowest route-distance row sum",
            "noise_baseline": (
                "keep M candidates with lowest RMS-distance row sum in the seven live "
                "dimensions of the known initial flow noise"
            ),
            "final_vote": (
                "checkpoint-std-normalized action medoid among completed survivors; "
                "a candidate ID tie is resolved toward the lowest ID"
            ),
            "normalized_global_medoid_regret": (
                "(full-pool mean action distance of subset winner - that of full winner) "
                "/ median full-pool pair distance"
            ),
            "pair_auc_target": (
                "whether final action distance is below the within-pool pair median; "
                "this is action geometry, not environment success"
            ),
            "bootstrap_unit": (
                "10 LIBERO Goal tasks; replans are averaged within task before tasks "
                "are resampled with replacement"
            ),
            "estimand": (
                "an equal-task, random-replan call under action-medoid-induced states; "
                "longer cases contribute more replans within their task"
            ),
            "difference_sign": (
                "left minus baseline; negative is better only for regret, positive is "
                "better for retention/equality/threshold rates"
            ),
        },
        "validation": {
            "stored_action_medoid_winner_mismatches": stored_winner_mismatches,
            "stored_action_medoid_score_mismatches": selector_score_mismatches,
            "max_raw_topk_weight_mass_error_before_renormalization": max_raw_weight_mass_error,
        },
        "statistics": {
            "bootstrap_draws": bootstrap,
            "bootstrap_seed": seed,
            "task_cluster_count": len(unique_tasks),
            "grid_status": "exploratory; no multiplicity correction",
        },
        "signal": {
            "initial_noise_pair_auc": aggregate_metric(
                noise_auc, task_ids, bootstrap_indices
            ),
            "route_prefix_curve": route_signal,
        },
        "baselines": {
            "exact_uniform_random_subset": random_out,
            "initial_noise_centrality": noise_out,
        },
        "route_prefix_pruning": route_out,
        "decision": {
            "online_route_pruning_supported": False,
            "route_basin_detected": False,
            "reason": (
                "Early routing beats exact random subsets on several action-consensus "
                "metrics, but is dominated by centrality of the already-known initial "
                "flow noise and has not been tested against counterfactual task value."
            ),
        },
        "diagnostic_shadow_points": {
            "conservative": {"prefix_steps": 1, "survivor_count": 6},
            "aggressive": {"prefix_steps": 1, "survivor_count": 4},
            "note": (
                "These points expose the trade-off for further diagnosis; they are not "
                "recommendations and both are dominated by initial-noise pruning on this "
                "development set. M=2 is diagnostic only because a two-candidate medoid "
                "always ties."
            ),
        },
        "limitations": [
            "Retrospective shadow simulation: all source candidates were fully denoised.",
            "No sampler pause/resume, candidate-batch compaction, CUDA latency, or memory was measured.",
            "The target is final action consensus, not counterfactual task value or success.",
            "The full action-medoid selector itself has not shown a significant success gain over K1.",
            "States come only from trajectories induced by the action-medoid arm.",
            "Replans are weighted within task, so this is not an equal-case controller estimate.",
            "Only normalized top-k routing mass is available, not the full router softmax.",
            "Row-sum route centrality is not a clustering or dynamical-basin detector.",
            "All prefix/budget choices were inspected on this development set and need held-out-task confirmation.",
            "Idealized flow savings exclude vision-prefix, routing-capture, selection, and scheduling overhead.",
        ],
    }


def _find_condition(report: Mapping[str, object], prefix: int, survivors: int) -> Mapping[str, object]:
    for row in report["route_prefix_pruning"]:
        if row["prefix_steps"] == prefix and row["survivor_count"] == survivors:
            return row
    raise KeyError((prefix, survivors))


def _print_summary(report: Mapping[str, object]) -> None:
    data = report["data"]
    print(
        "loaded {case_count} cases, {pool_count} pools, {task_count} task clusters; "
        "K={candidate_count}, flow={flow_steps}, action={action_chunk_steps}x{action_dim}".format(
            **data
        )
    )
    noise_auc = report["signal"]["initial_noise_pair_auc"]["task_macro_mean"]
    print("\nprefix  route-AUC  survivors  save    keep-full  same-vote  regret  noise-regret")
    for signal in report["signal"]["route_prefix_curve"]:
        prefix = signal["prefix_steps"]
        route_auc = signal["route_pair_auc"]["task_macro_mean"]
        for survivor_count in sorted(int(key) for key in report["baselines"]["initial_noise_centrality"]):
            row = _find_condition(report, prefix, survivor_count)
            metrics = row["metrics"]
            noise_metrics = report["baselines"]["initial_noise_centrality"][str(survivor_count)][
                "metrics"
            ]
            print(
                "%6d  %9.3f  %9d  %5.1f%%  %9.3f  %9.3f  %6.3f  %12.3f"
                % (
                    prefix,
                    route_auc,
                    survivor_count,
                    100.0 * row["idealized_flow_saving_fraction"],
                    metrics["full_winner_retained"]["task_macro_mean"],
                    metrics["selected_equals_full_winner"]["task_macro_mean"],
                    metrics["normalized_global_medoid_regret"]["task_macro_mean"],
                    noise_metrics["normalized_global_medoid_regret"]["task_macro_mean"],
                )
            )
    print("\ninitial-noise pair AUC: %.3f" % noise_auc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--survivors", type=_parse_int_list, default=(2, 4, 6))
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    report = run_analysis(
        batch_dir=args.batch_dir,
        survivors=args.survivors,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
