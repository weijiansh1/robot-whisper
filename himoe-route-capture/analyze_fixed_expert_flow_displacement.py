"""Fixed-expert audit of HB5 norms against total flow displacement.

The source archive stores selected expert identities, selected gate weights, and
offline-reconstructed pre-gate expert-output norms.  This analysis never treats
the four selected slots as exchangeable expert identities.  Every comparison
fixes task, state, denoise round, action token, and expert ID, then compares only
candidates for which that same expert was selected.

The available endpoint is total flow displacement ``RMS(x0 - x10)``.  The
archive does not contain intermediate ``x_d`` values, so this script cannot test
remaining correction, one-step updates, or action commitment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata


N_EXPERTS = 32
FROZEN_FEATURES = ("raw.d6", "raw.d7", "raw.d8", "weighted.rebound")
FROZEN_ORIENTATION = np.asarray((-1.0, -1.0, -1.0, 1.0))


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    default_root = here / "analysis" / "unweighted-expert-norm"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=default_root / "unweighted_expert_norms.npz",
    )
    parser.add_argument(
        "--source-summary",
        type=pathlib.Path,
        default=default_root / "summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=here / "analysis" / "fixed-expert-flow-displacement",
    )
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--split-perms", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--min-selected", type=int, default=8)
    return parser.parse_args()


def _slope(value: np.ndarray) -> np.ndarray:
    time_axis = np.arange(value.shape[-1], dtype=np.float64)
    time_axis -= time_axis.mean()
    return np.sum(value * time_axis, axis=-1) / np.sum(time_axis * time_axis)


def _pair_index(n_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(n_seed, 1)
    pair_id = np.full((n_seed, n_seed), -1, dtype=np.int64)
    pair_id[pair_i, pair_j] = np.arange(len(pair_i))
    pair_id[pair_j, pair_i] = np.arange(len(pair_i))
    return pair_i, pair_j, pair_id


def _descriptive_pool_spearman(
    predictor: np.ndarray,
    target: np.ndarray,
    states: np.ndarray | None = None,
) -> dict[str, Any]:
    """Equal-pool and equal-task Spearman summary for [task,state,seed]."""
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("predictor and target must share [task,state,seed]")
    if states is None:
        states = np.arange(x.shape[1])
    states = np.asarray(states, dtype=np.int64)
    if states.ndim != 1 or len(states) == 0:
        raise ValueError("states must be a nonempty one-dimensional index")
    x_rank = rankdata(x[:, states], axis=-1)
    y_rank = rankdata(y[:, states], axis=-1)
    x_rank -= x_rank.mean(axis=-1, keepdims=True)
    y_rank -= y_rank.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(
        np.sum(np.square(x_rank), axis=-1)
        * np.sum(np.square(y_rank), axis=-1)
    )
    pool = np.divide(
        np.sum(x_rank * y_rank, axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    task = np.nanmean(pool, axis=-1)
    return {
        "scene_axes": states.tolist(),
        "pool_spearman": pool.tolist(),
        "equal_pool_per_task_spearman": task.tolist(),
        "equal_task_macro_spearman": float(np.nanmean(task)),
    }


def _state_loo_seed_template(target: np.ndarray) -> np.ndarray:
    """For each task/state/seed, average the other states at that exact seed."""
    value = np.asarray(target, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] < 2:
        raise ValueError("target must be [task,state>=2,seed]")
    return (value.sum(axis=1, keepdims=True) - value) / (value.shape[1] - 1)


def _discovery_seed_template_residual(
    target: np.ndarray,
    discovery_states: np.ndarray,
    validation_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a task+seed template on discovery states, then subtract it."""
    value = np.asarray(target, dtype=np.float64)
    discovery = np.asarray(discovery_states, dtype=np.int64)
    validation = np.asarray(validation_states, dtype=np.int64)
    if value.ndim != 3:
        raise ValueError("target must be [task,state,seed]")
    if discovery.ndim != 1 or validation.ndim != 1:
        raise ValueError("state indices must be one-dimensional")
    if len(discovery) == 0 or len(validation) == 0:
        raise ValueError("discovery and validation states must be nonempty")
    if np.intersect1d(discovery, validation).size:
        raise ValueError("discovery and validation states must be disjoint")
    all_states = np.concatenate((discovery, validation))
    if np.any(all_states < 0) or np.any(all_states >= value.shape[1]):
        raise ValueError("state index is out of range")
    template = value[:, discovery].mean(axis=1)
    residual = value - template[:, None, :]
    return template, residual


def _correct_actions_over_std(
    actions_over_std: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Convert the archived ``action/std`` value to ``(action-mean)/std``."""
    actions = np.asarray(actions_over_std, dtype=np.float64)
    mean_value = np.asarray(mean, dtype=np.float64)
    std_value = np.asarray(std, dtype=np.float64)
    if mean_value.shape != std_value.shape or actions.shape[-1] != len(mean_value):
        raise ValueError("action normalization dimensions do not match")
    if np.any(~np.isfinite(std_value)) or np.any(std_value <= 0):
        raise ValueError("action std must be finite and positive")
    reshape = (1,) * (actions.ndim - 1) + (len(mean_value),)
    return actions - (mean_value / std_value).reshape(reshape)


def _resolve_episode(client: pathlib.Path, episode: int) -> pathlib.Path:
    for width in (2, 4):
        path = client / f"episode_{episode:0{width}d}.npz"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no episode file for {episode} below {client}")


def _load_corrected_actions(
    task_names: list[str],
    scene_ids: np.ndarray,
    seed_ids: np.ndarray,
    archived_actions: np.ndarray,
    source_summary_path: pathlib.Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct action means and independently verify task/state/seed alignment."""
    source_summary = json.loads(source_summary_path.read_text())
    corrected = np.empty_like(archived_actions, dtype=np.float64)
    audit: dict[str, Any] = {}
    for task_axis, task in enumerate(task_names):
        run = pathlib.Path(source_summary["sources"][task]["run"])
        client = run / "client"
        metadata = json.loads((client / "server_metadata.json").read_text())
        stats = json.loads(
            pathlib.Path(metadata["normalization_stats_path"]).read_text()
        )["actions"]
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        corrected[task_axis] = _correct_actions_over_std(
            archived_actions[task_axis], mean, std
        )

        summaries = json.loads((client / "summaries.json").read_text())
        episode_by_key = {
            (int(row["init_state_id"]), int(row["flow_noise_seed"])): int(
                row["episode_index"]
            )
            for row in summaries
        }
        raw_actions = np.empty_like(archived_actions[task_axis], dtype=np.float64)
        for state_axis, state in enumerate(scene_ids[task_axis]):
            for seed_axis, noise_seed in enumerate(seed_ids[task_axis]):
                episode = episode_by_key[(int(state), int(noise_seed))]
                with np.load(
                    _resolve_episode(client, episode), allow_pickle=False
                ) as payload:
                    raw_actions[state_axis, seed_axis] = np.asarray(
                        payload["actions"][0], dtype=np.float64
                    )

        expected_over_std = raw_actions / std
        expected_normalized = (raw_actions - mean) / std
        source_seed_grid = sorted({int(row["flow_noise_seed"]) for row in summaries})
        audit[task] = {
            "run": str(run),
            "source_seed_grid_matches_archive": bool(
                np.array_equal(source_seed_grid, seed_ids[task_axis])
            ),
            "archive_action_over_std_max_abs_error": float(
                np.max(np.abs(archived_actions[task_axis] - expected_over_std))
            ),
            "mean_corrected_x10_max_abs_error": float(
                np.max(np.abs(corrected[task_axis] - expected_normalized))
            ),
            "action_mean": mean.tolist(),
            "action_std": std.tolist(),
        }
    return corrected, audit


def _noise_from_seed_ids(seed_ids: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if seed_ids.ndim != 2 or not np.all(seed_ids == seed_ids[0]):
        raise ValueError("all tasks must use the same ordered seed columns")
    noise24 = np.stack(
        [
            np.random.default_rng(int(seed)).standard_normal((10, 24))
            for seed in seed_ids[0]
        ]
    )
    live = noise24[:, :, :7]
    audit = {
        "seed_ids": seed_ids[0].tolist(),
        "noise24_shape": list(noise24.shape),
        "live7_shape": list(live.shape),
        "construction": (
            "default_rng(flow_noise_seed).standard_normal((10,24))[:, :7]"
        ),
        "axis_order": "seed, action_token, live_action_dimension",
        "sha256_float64_bytes": hashlib.sha256(live.tobytes()).hexdigest(),
    }
    return live, audit


def _fixed_expert_pair_coefficients(
    expert_ids: np.ndarray,
    raw_norm: np.ndarray,
    gate_weight: np.ndarray,
    min_selected: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Construct equal-stratum pair-sign coefficients.

    Input axes are ``task, state, seed, denoise, token, selected_slot``.  A
    stratum fixes task/state/denoise/token/expert ID.  Trajectory strata also
    require a candidate/token/expert to remain selected in all denoise rounds.
    """
    ids = np.asarray(expert_ids, dtype=np.int64)
    raw = np.asarray(raw_norm, dtype=np.float64)
    weight = np.asarray(gate_weight, dtype=np.float64)
    if ids.shape != raw.shape or ids.shape != weight.shape or ids.ndim != 6:
        raise ValueError("expert id, norm, and weight arrays must share six axes")
    if min_selected < 2 or min_selected > ids.shape[2]:
        raise ValueError("min_selected is outside the candidate axis")
    n_task, n_state, n_seed, n_denoise, n_token, _ = ids.shape
    if n_denoise != 10:
        raise ValueError("trajectory definitions require exactly ten denoise rounds")
    _, _, pair_id = _pair_index(n_seed)
    n_pair = n_seed * (n_seed - 1) // 2

    cell_names = [
        f"{kind}.d{denoise}"
        for kind in ("raw", "weighted", "gate")
        for denoise in range(n_denoise)
    ]
    trajectory_names = [
        f"{kind}.{shape}"
        for kind in ("raw", "weighted", "gate")
        for shape in ("slope", "rebound", "curvature")
    ]
    names = cell_names + trajectory_names
    coefficients = np.zeros((len(names), n_task, n_state, n_pair), np.float64)
    valid = np.zeros((len(names), n_task, n_state), dtype=bool)
    strata_count = np.zeros((len(names), n_task, n_state), dtype=np.int32)

    for task in range(n_task):
        for state in range(n_state):
            for denoise in range(n_denoise):
                accumulated = np.zeros((3, n_pair), dtype=np.float64)
                count = 0
                for token in range(n_token):
                    token_ids = ids[task, state, :, denoise, token]
                    token_raw = raw[task, state, :, denoise, token]
                    token_weight = weight[task, state, :, denoise, token]
                    for expert in range(N_EXPERTS):
                        matches = token_ids == expert
                        selected = np.any(matches, axis=-1)
                        index = np.flatnonzero(selected)
                        if len(index) < min_selected:
                            continue
                        raw_value = np.sum(token_raw * matches, axis=-1)[index]
                        gate_value = np.sum(token_weight * matches, axis=-1)[index]
                        values = (raw_value, raw_value * gate_value, gate_value)
                        local_i, local_j = np.triu_indices(len(index), 1)
                        global_pair = pair_id[index[local_i], index[local_j]]
                        for predictor, value in enumerate(values):
                            signs = np.sign(value[local_i] - value[local_j])
                            np.add.at(
                                accumulated[predictor],
                                global_pair,
                                signs / len(global_pair),
                            )
                        count += 1
                if count:
                    for predictor in range(3):
                        feature = predictor * n_denoise + denoise
                        coefficients[feature, task, state] = (
                            accumulated[predictor] / count
                        )
                        valid[feature, task, state] = True
                        strata_count[feature, task, state] = count

            accumulated_trajectory = np.zeros((9, n_pair), dtype=np.float64)
            trajectory_count = 0
            for token in range(n_token):
                token_ids = ids[task, state, :, :, token]
                token_raw = raw[task, state, :, :, token]
                token_weight = weight[task, state, :, :, token]
                for expert in range(N_EXPERTS):
                    matches = token_ids == expert
                    selected_every_round = np.all(np.any(matches, axis=-1), axis=-1)
                    index = np.flatnonzero(selected_every_round)
                    if len(index) < min_selected:
                        continue
                    raw_curve = np.sum(token_raw * matches, axis=-1)[index]
                    gate_curve = np.sum(token_weight * matches, axis=-1)[index]
                    curves = (raw_curve, raw_curve * gate_curve, gate_curve)
                    local_i, local_j = np.triu_indices(len(index), 1)
                    global_pair = pair_id[index[local_i], index[local_j]]
                    for predictor, curve in enumerate(curves):
                        summaries = (
                            _slope(curve),
                            curve[:, -1] - curve.min(axis=-1),
                            _slope(curve[:, 5:]) - _slope(curve[:, :5]),
                        )
                        for shape, value in enumerate(summaries):
                            signs = np.sign(value[local_i] - value[local_j])
                            np.add.at(
                                accumulated_trajectory[predictor * 3 + shape],
                                global_pair,
                                signs / len(global_pair),
                            )
                    trajectory_count += 1
            if trajectory_count:
                for local_feature in range(9):
                    feature = len(cell_names) + local_feature
                    coefficients[feature, task, state] = (
                        accumulated_trajectory[local_feature] / trajectory_count
                    )
                    valid[feature, task, state] = True
                    strata_count[feature, task, state] = trajectory_count

    coverage: dict[str, Any] = {}
    for feature, name in enumerate(names):
        coverage[name] = {
            "valid_states_per_task": valid[feature].sum(axis=-1).tolist(),
            "median_strata_per_valid_state": [
                float(np.median(strata_count[feature, task, valid[feature, task]]))
                if np.any(valid[feature, task])
                else 0.0
                for task in range(n_task)
            ],
        }
    return coefficients, valid, names, coverage


def _target_pair_signs(
    target: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    return np.sign(target[..., pair_i] - target[..., pair_j]).astype(np.float64)


def _task_effects(
    coefficients: np.ndarray,
    valid: np.ndarray,
    target_signs: np.ndarray,
    states: np.ndarray | None = None,
) -> np.ndarray:
    if states is None:
        states = np.arange(coefficients.shape[2])
    state_effect = np.einsum(
        "ftsp,tsp->fts",
        coefficients[:, :, states],
        target_signs[:, states],
        optimize=True,
    )
    state_valid = valid[:, :, states]
    count = state_valid.sum(axis=-1)
    return np.divide(
        np.sum(state_effect * state_valid, axis=-1),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _macro_effect(task_effect: np.ndarray, eligible_tasks: np.ndarray) -> np.ndarray:
    return np.nanmean(task_effect[:, eligible_tasks], axis=-1)


def _global_max_t_test(
    coefficients: np.ndarray,
    valid: np.ndarray,
    feature_names: list[str],
    endpoints: dict[str, tuple[np.ndarray, np.ndarray]],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    endpoint_names = list(endpoints)
    observed_task = []
    observed_macro = []
    for target, eligible in endpoints.values():
        task = _task_effects(
            coefficients, valid, _target_pair_signs(target, pair_i, pair_j)
        )
        observed_task.append(task)
        observed_macro.append(_macro_effect(task, eligible))
    observed_task_array = np.stack(observed_task)
    observed_macro_array = np.stack(observed_macro)

    rng = np.random.default_rng(seed)
    null = np.empty(
        (permutations, len(endpoint_names), len(feature_names)), dtype=np.float64
    )
    for permutation in range(permutations):
        order = rng.permutation(endpoints[endpoint_names[0]][0].shape[-1])
        for endpoint_axis, (target, eligible) in enumerate(endpoints.values()):
            task = _task_effects(
                coefficients,
                valid,
                _target_pair_signs(target[..., order], pair_i, pair_j),
            )
            null[permutation, endpoint_axis] = _macro_effect(task, eligible)
        if (permutation + 1) % 500 == 0:
            print(f"global permutations {permutation + 1}/{permutations}", flush=True)

    null_mean = null.mean(axis=0)
    null_sd = null.std(axis=0, ddof=1)
    observed_z = (observed_macro_array - null_mean) / np.maximum(null_sd, 1e-12)
    null_z = (null - null_mean[None]) / np.maximum(null_sd[None], 1e-12)
    null_max = np.max(np.abs(null_z), axis=(1, 2))
    adjusted_p = (
        1 + np.sum(null_max[:, None, None] >= np.abs(observed_z)[None], axis=0)
    ) / (1 + permutations)

    results: dict[str, Any] = {}
    for endpoint_axis, endpoint_name in enumerate(endpoint_names):
        eligible = endpoints[endpoint_name][1]
        task_values = observed_task_array[endpoint_axis]
        direction = np.sign(observed_macro_array[endpoint_axis])
        consistent = np.all(task_values[:, eligible] * direction[:, None] > 0, axis=1)
        rows = []
        for feature, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature_name,
                    "macro_pair_concordance": float(
                        observed_macro_array[endpoint_axis, feature]
                    ),
                    "per_task": task_values[feature].tolist(),
                    "all_eligible_tasks_same_direction": bool(consistent[feature]),
                    "global_117_test_maxT_p": float(adjusted_p[endpoint_axis, feature]),
                    "z_against_common_seed_null": float(
                        observed_z[endpoint_axis, feature]
                    ),
                }
            )
        rows.sort(
            key=lambda row: abs(row["z_against_common_seed_null"]),
            reverse=True,
        )
        consistent_rows = [
            row for row in rows if row["all_eligible_tasks_same_direction"]
        ]
        results[endpoint_name] = {
            "eligible_task_axes": np.flatnonzero(eligible).tolist(),
            "ineligible_task_axes": np.flatnonzero(~eligible).tolist(),
            "maxT_significant_count": int(np.sum(adjusted_p[endpoint_axis] < 0.05)),
            "direction_consistent_count": int(consistent.sum()),
            "significant": [
                row for row in rows if row["global_117_test_maxT_p"] < 0.05
            ],
            "top10_by_abs_z": rows[:10],
            "top10_direction_consistent": consistent_rows[:10],
            "all_features": rows,
        }
    return {
        "permutations": permutations,
        "maxT_family_size": len(endpoint_names) * len(feature_names),
        "null_max_abs_z_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
        "endpoints": results,
    }


def _frozen_split_test(
    coefficients: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    states: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed_task = _task_effects(
        coefficients,
        valid,
        _target_pair_signs(target, pair_i, pair_j),
        states,
    )
    observed = np.nanmean(observed_task, axis=-1)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(FROZEN_FEATURES)), dtype=np.float64)
    for draw in range(permutations):
        order = rng.permutation(target.shape[-1])
        task = _task_effects(
            coefficients,
            valid,
            _target_pair_signs(target[..., order], pair_i, pair_j),
            states,
        )
        null[draw] = np.nanmean(task, axis=-1)

    null_mean = null.mean(axis=0)
    null_sd = null.std(axis=0, ddof=1)
    oriented_observed = FROZEN_ORIENTATION * observed
    oriented_null = FROZEN_ORIENTATION[None] * null
    raw_p = (1 + np.sum(oriented_null >= oriented_observed[None], axis=0)) / (
        1 + permutations
    )
    observed_z = (
        FROZEN_ORIENTATION * (observed - null_mean) / np.maximum(null_sd, 1e-12)
    )
    null_z = (
        FROZEN_ORIENTATION[None]
        * (null - null_mean[None])
        / np.maximum(null_sd[None], 1e-12)
    )
    null_max = np.max(null_z, axis=1)
    max_t_p = (1 + np.sum(null_max[:, None] >= observed_z[None], axis=0)) / (
        1 + permutations
    )

    rows = {}
    for feature, name in enumerate(FROZEN_FEATURES):
        oriented_concordance = FROZEN_ORIENTATION[feature] * observed[feature]
        rows[name] = {
            "expected_direction": (
                "negative" if FROZEN_ORIENTATION[feature] < 0 else "positive"
            ),
            "task_equal_pair_concordance": float(observed[feature]),
            "expected_direction_pair_ranking_accuracy": float(
                0.5 + 0.5 * oriented_concordance
            ),
            "expected_direction_pair_ranking_advantage_percentage_points": float(
                50.0 * oriented_concordance
            ),
            "per_task_pair_concordance": observed_task[feature].tolist(),
            "all_five_tasks_expected_direction": bool(
                np.all(FROZEN_ORIENTATION[feature] * observed_task[feature] > 0)
            ),
            "one_sided_common_seed_permutation_p": float(raw_p[feature]),
            "one_sided_maxT_p_over_four_frozen_features": float(max_t_p[feature]),
            "oriented_z": float(observed_z[feature]),
        }
    return {
        "scene_axes": states.tolist(),
        "n_states_per_task": int(len(states)),
        "permutations": permutations,
        "null_max_oriented_z_q95": float(np.quantile(null_max, 0.95)),
        "features": rows,
    }


def _format_tasks(values: list[float]) -> str:
    return " / ".join(f"{value:+.4f}" for value in values)


def _render_report(summary: dict[str, Any]) -> str:
    task_names = summary["task_names"]
    global_result = summary["global_maxT_117"]["endpoints"]
    splits = summary["frozen_state_splits"]
    baselines = summary["free_baseline_controls"]
    strict = summary["strict_seed_template_validation_residual"]
    lines = [
        "# 固定专家的总 Flow 位移审计",
        "",
        "本分析固定 `task / state / denoise / action token / expert ID`，只在同一",
        "expert 被选中的候选之间比较。`raw` 是乘 gate 前的 `RMS(E_e(h))`，",
        "`weighted` 是 `w_e * RMS(E_e(h))`，`gate` 是 `w_e`。",
        "",
        "## 目标与边界",
        "",
        "目标是标准化 live-7 action 空间中的 `RMS(x0-x10)`。这是从初始 flow",
        "noise 到最终 chunk 的总位移，不是 `RMS(x_d-x10)` 剩余修正，也不是",
        "`RMS(x_(d+1)-x_d)` 单步更新。源 NPZ 没有中间 `x_d` 或专家输出向量。",
        "",
        "## 对齐校正",
        "",
        "源 NPZ 的 `final_actions_standardized` 实际是 `action/std`。本分析从每个",
        "source run 读取 normalization mean，修复为 `(action-mean)/std`，并逐",
        "episode 复核 task/state/seed/action 对齐。",
        "",
        "| task | seed grid | action/std max error | corrected x10 max error |",
        "|---|---:|---:|---:|",
    ]
    for task in task_names:
        row = summary["alignment_audit"][task]
        lines.append(
            "| %s | %s | %.2e | %.2e |"
            % (
                task,
                "yes" if row["source_seed_grid_matches_archive"] else "no",
                row["archive_action_over_std_max_abs_error"],
                row["mean_corrected_x10_max_abs_error"],
            )
        )
    noise = summary["alignment_audit"]["noise_axis"]
    lines += [
        "",
        "x0 按 `default_rng(seed).standard_normal((10,24))[:, :7]` 重建；轴为",
        f"`seed, action_token, live_dim`，shape `{noise['live7_shape']}`。",
        "",
        "## 免费 baseline",
        "",
        "总位移本身可被不读取 MoE 的量强预测。下表在每个 task/state 的 32 个 exact-seed",
        "候选内算 Spearman，再对 state、task 等权平均；这里只报告描述性 rho。",
        "",
        "| baseline | macro rho | per-task rho |",
        "|---|---:|---|",
        "| RMS(x0) | %.3f | %s |"
        % (
            baselines["RMS_x0"]["equal_task_macro_spearman"],
            _format_tasks(baselines["RMS_x0"]["equal_pool_per_task_spearman"]),
        ),
        "| exact-seed task-local state-LOO target template | %.3f | %s |"
        % (
            baselines["exact_seed_task_local_state_LOO_target_template"][
                "equal_task_macro_spearman"
            ],
            _format_tasks(
                baselines["exact_seed_task_local_state_LOO_target_template"][
                    "equal_pool_per_task_spearman"
                ]
            ),
        ),
        "",
        "第二个 baseline 对每个 task/state/seed 使用同任务、同 seed 的其他 15 states",
        "目标均值。它使用其他 state 的 target 标签，是严格的 nuisance control，不是部署时",
        "可免费获得的剪枝信号；约 0.971 的 rho 说明必须先去掉 seed-template 才能谈 MoE 增量。",
        "",
        "## 全搜索族",
        "",
        "机制对齐的搜索族为 39 个 feature × 3 endpoints = 117 项：raw / weighted /",
        "gate 在 10 轮的值，以及三者在十轮持续选中同一 expert 时的 slope / rebound /",
        "curvature。%d 次共同 seed-column 置换对 117 项做全局双侧 maxT。"
        % summary["global_maxT_117"]["permutations"],
        "",
        "| endpoint | maxT-significant |",
        "|---|---:|",
    ]
    for endpoint, row in global_result.items():
        lines.append(f"| {endpoint} | {row['maxT_significant_count']} |")
    lines += [
        "",
        "通过全局 maxT 且五任务同向的总位移关系：",
        "",
        "| feature | macro concordance | per-task concordance | global p |",
        "|---|---:|---|---:|",
    ]
    for row in global_result["noise_to_final_correction"]["significant"]:
        if not row["all_eligible_tasks_same_direction"]:
            continue
        lines.append(
            "| %s | %+.4f | %s | %.4f |"
            % (
                row["feature"],
                row["macro_pair_concordance"],
                _format_tasks(row["per_task"]),
                row["global_117_test_maxT_p"],
            )
        )
    correction_rows = {
        row["feature"]: row
        for row in global_result["noise_to_final_correction"]["all_features"]
    }
    lines += [
        "",
        "同轮 gate-only 对照不复现 raw d6-d8 关系：",
        "",
        "| feature | macro concordance | five-task direction | global p |",
        "|---|---:|---:|---:|",
    ]
    for feature in ("gate.d6", "gate.d7", "gate.d8", "gate.rebound"):
        row = correction_rows[feature]
        lines.append(
            "| %s | %+.4f | %s | %.4f |"
            % (
                feature,
                row["macro_pair_concordance"],
                "yes" if row["all_eligible_tasks_same_direction"] else "no",
                row["global_117_test_maxT_p"],
            )
        )
    lines += [
        "",
        "action eccentricity 与 success 均无全局校正后信号。`goal-middle` 为",
        "512/512 全成功，因此 success 的五任务共性在定义上不可检验。",
        "",
        "## 冻结 State 切分",
        "",
        "四个 feature 是在全 state 事后探索后冻结的。以下切分只检验 state 集中性与",
        "稳定性，不是未见数据上的前瞻确认。每个 split 用 %s 次共同 seed-column"
        % f"{next(iter(splits.values()))['permutations']:,}",
        "置换，并在四个冻结方向内做单侧 maxT。",
        "",
    ]
    for split_name, split in splits.items():
        lines += [
            f"### {split_name}",
            "",
            "| feature | macro | per-task | five-task direction | maxT4 p |",
            "|---|---:|---|---:|---:|",
        ]
        for feature in FROZEN_FEATURES:
            row = split["features"][feature]
            lines.append(
                "| %s | %+.4f | %s | %s | %.5f |"
                % (
                    feature,
                    row["task_equal_pair_concordance"],
                    _format_tasks(row["per_task_pair_concordance"]),
                    "yes" if row["all_five_tasks_expected_direction"] else "no",
                    row["one_sided_maxT_p_over_four_frozen_features"],
                )
            )
        lines.append("")
    lines += [
        "## Discovery-template → Validation-residual 严格控制",
        "",
        "只用 discovery scene axes `0..7` 构造每个 `task + exact seed` 的 target 均值模板；",
        "在 validation axes `8..15` 上令 `residual = target - template`。随后只检验四个",
        "冻结 feature，用 %s 次共同 seed-column permutation 做单侧 maxT4。"
        % f"{strict['permutations']:,}",
        "",
        "| feature | residual macro | per-task | five-task direction | ranking advantage | maxT4 p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for feature in FROZEN_FEATURES:
        row = strict["features"][feature]
        lines.append(
            "| %s | %+.5f | %s | %s | %+.2f pp | %.5f |"
            % (
                feature,
                row["task_equal_pair_concordance"],
                _format_tasks(row["per_task_pair_concordance"]),
                "yes" if row["all_five_tasks_expected_direction"] else "no",
                row["expected_direction_pair_ranking_advantage_percentage_points"],
                row["one_sided_maxT_p_over_four_frozen_features"],
            )
        )
    lines += [
        "",
        "raw d6/d7/d8 在 validation residual 上仍为五任务负向，但 concordance 只有",
        "约 -0.015 到 -0.019：换成直观排序准确率，只比 50% 高约 0.73–0.95 个百分点。",
        "weighted rebound 不再五任务同向，也未通过 maxT4。这个量级不能支持候选剪枝。",
        "",
        "该 residual 控制的 template 构造与 validation state 严格分离；但四个 feature",
        "来自更早的全 state 探索，因此仍不是完全未触碰数据上的前瞻确认。",
        "",
        "## 裁决",
        "",
        "1. 原始总位移高度受 x0/seed template 支配；不控制它时的 MoE 关联会被夸大。",
        "2. 严格 discovery-template → validation-residual 后，raw d6-d8 仍保持五任务负向；",
        "   weighted rebound 不通过，因此撤回它作为稳定冻结信号。",
        "3. raw 的增量只有约 0.7–0.95 pp pairwise ranking advantage，不能用于剪枝、",
        "   early stop 或动作承诺判断。",
        "4. 结果仍是 selected-only、离线 fp16-hidden 重建后的关联；固定选中集合会带来",
        "   selection conditioning，也不能解释为 expert 幅度的因果作用。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.perms <= 0 or args.split_perms <= 0:
        raise ValueError("permutation counts must be positive")
    started = time.perf_counter()
    args.archive = args.archive.resolve()
    args.source_summary = args.source_summary.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.archive, allow_pickle=False) as payload:
        task_names = [str(value) for value in payload["task_names"]]
        scene_ids = np.asarray(payload["scene_ids"], dtype=np.int64)
        seed_ids = np.asarray(payload["seed_ids"], dtype=np.int64)
        expert_ids = np.asarray(payload["selected_expert_ids"], dtype=np.int64)
        gate_weight = np.asarray(payload["selected_expert_weight"], dtype=np.float64)
        raw_norm = np.asarray(payload["selected_expert_raw_rms"], dtype=np.float64)
        archived_actions = np.asarray(
            payload["final_actions_standardized"], dtype=np.float64
        )
        success = np.asarray(payload["success"], dtype=np.int8)
        action_eccentricity = np.asarray(
            payload["action_eccentricity"], dtype=np.float64
        )

    corrected_actions, alignment_audit = _load_corrected_actions(
        task_names,
        scene_ids,
        seed_ids,
        archived_actions,
        args.source_summary,
    )
    initial_noise, noise_audit = _noise_from_seed_ids(seed_ids)
    alignment_audit["noise_axis"] = noise_audit
    correction = np.sqrt(
        np.mean(
            (corrected_actions - initial_noise[None, None]) ** 2,
            axis=(-1, -2),
        )
    )
    noise_rms = np.sqrt(np.mean(np.square(initial_noise), axis=(-1, -2)))
    noise_baseline = np.broadcast_to(
        noise_rms[None, None, :], correction.shape
    ).copy()
    state_loo_template = _state_loo_seed_template(correction)
    free_baselines = {
        "RMS_x0": {
            "definition": "RMS(x0) over the same 10x7 live coordinates",
            **_descriptive_pool_spearman(noise_baseline, correction),
        },
        "exact_seed_task_local_state_LOO_target_template": {
            "definition": (
                "for each task/state/seed, mean target over the other 15 states "
                "at that exact seed"
            ),
            "deployment_boundary": (
                "uses target labels from other states and is a descriptive free "
                "baseline, not an online pruning feature"
            ),
            **_descriptive_pool_spearman(state_loo_template, correction),
        },
    }

    coefficients, valid, feature_names, coverage = _fixed_expert_pair_coefficients(
        expert_ids, raw_norm, gate_weight, args.min_selected
    )
    pair_i, pair_j, _ = _pair_index(expert_ids.shape[2])
    success_eligible = np.asarray(
        [len(np.unique(success[task])) == 2 for task in range(len(task_names))]
    )
    endpoints = {
        "action_eccentricity": (
            action_eccentricity,
            np.ones(len(task_names), dtype=bool),
        ),
        "noise_to_final_correction": (
            correction,
            np.ones(len(task_names), dtype=bool),
        ),
        "success": (success, success_eligible),
    }
    global_result = _global_max_t_test(
        coefficients,
        valid,
        feature_names,
        endpoints,
        pair_i,
        pair_j,
        args.perms,
        args.seed,
    )

    frozen_axes = np.asarray(
        [feature_names.index(name) for name in FROZEN_FEATURES], dtype=np.int64
    )
    frozen_coefficients = coefficients[frozen_axes]
    frozen_valid = valid[frozen_axes]
    split_definitions = {
        "contiguous_discovery_0_7": np.arange(0, 8),
        "contiguous_validation_8_15": np.arange(8, 16),
        "even_scene_axis_sensitivity": np.arange(0, 16, 2),
        "odd_scene_axis_sensitivity": np.arange(1, 16, 2),
    }
    split_results = {}
    for split_axis, (split_name, states) in enumerate(split_definitions.items()):
        row = _frozen_split_test(
            frozen_coefficients,
            frozen_valid,
            correction,
            pair_i,
            pair_j,
            states,
            args.split_perms,
            args.seed + split_axis,
        )
        row["scene_id_values_per_task"] = {
            task_names[task]: scene_ids[task, states].tolist()
            for task in range(len(task_names))
        }
        split_results[split_name] = row
        print(f"finished {split_name}", flush=True)

    discovery_states = np.arange(0, 8)
    validation_states = np.arange(8, 16)
    discovery_template, validation_residual = _discovery_seed_template_residual(
        correction, discovery_states, validation_states
    )
    validation_template = np.broadcast_to(
        discovery_template[:, None, :], correction.shape
    ).copy()
    strict_residual = _frozen_split_test(
        frozen_coefficients,
        frozen_valid,
        validation_residual,
        pair_i,
        pair_j,
        validation_states,
        args.split_perms,
        args.seed + 2,
    )
    strict_residual.update(
        {
            "discovery_scene_axes": discovery_states.tolist(),
            "validation_scene_axes": validation_states.tolist(),
            "template_definition": (
                "task+exact-seed mean target over discovery scene axes 0..7"
            ),
            "residual_definition": (
                "validation target minus the frozen discovery task+seed template"
            ),
            "template_validation_target_spearman": _descriptive_pool_spearman(
                validation_template, correction, validation_states
            ),
            "permutation_seed": args.seed + 2,
            "feature_freeze_boundary": (
                "the four features were selected in the earlier full-state audit; "
                "only template construction and residual evaluation are state-disjoint"
            ),
        }
    )
    print("finished strict discovery-template validation residual", flush=True)

    summary = {
        "experiment": "fixed_expert_flow_displacement_v1",
        "status": "post_selection_state_stability_not_prospective_confirmation",
        "archive": str(args.archive),
        "source_summary": str(args.source_summary),
        "task_names": task_names,
        "fixed_layer": 5,
        "comparison_unit": (
            "task/state/denoise/action_token/expert_id; compare candidate pairs "
            "only when the same expert is selected"
        ),
        "minimum_candidates_per_fixed_stratum": args.min_selected,
        "statistic": (
            "equal-stratum Kendall-style pair concordance; then equal-state and "
            "equal-task means; ties contribute zero"
        ),
        "target": (
            "RMS(x0-x10) over 10x7 live normalized action coordinates after "
            "correcting action/std to (action-mean)/std"
        ),
        "feature_family": {
            "cell": "raw norm, gate*raw norm, gate weight at each of 10 rounds",
            "trajectory": (
                "raw, weighted, gate slope/rebound/curvature for the same "
                "candidate-token-expert selected all 10 rounds"
            ),
            "features_per_endpoint": len(feature_names),
            "endpoints": len(endpoints),
            "global_tests": len(feature_names) * len(endpoints),
        },
        "global_maxT_117": global_result,
        "frozen_features": list(FROZEN_FEATURES),
        "frozen_orientations": dict(
            zip(FROZEN_FEATURES, ("negative", "negative", "negative", "positive"))
        ),
        "frozen_state_splits": split_results,
        "free_baseline_controls": free_baselines,
        "strict_seed_template_validation_residual": strict_residual,
        "alignment_audit": alignment_audit,
        "coverage": coverage,
        "success_counts": {
            task_names[task]: {
                "success": int(success[task].sum()),
                "total": int(success[task].size),
            }
            for task in range(len(task_names))
        },
        "post_selection_warning": (
            "The four frozen features were identified in an earlier full-state "
            "exploratory audit. State splits test stability and concentration, "
            "not untouched prospective confirmation."
        ),
        "selected_only_boundary": (
            "Norms exist only for selected experts. Conditioning on selection can "
            "induce association and does not identify a causal expert effect."
        ),
        "missing_for_commitment": [
            "intermediate x_d and x_(d+1)",
            "runtime-exact expert outputs",
            "expert output vectors and directions",
            "shared/post-MoE directions aligned to the fixed expert stratum",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
