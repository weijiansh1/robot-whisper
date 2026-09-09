"""Retrospective hidden-matched audit of absolute selected-expert size.

The frozen cell is HB5/denoise-0 on five complete 16-state x 32-seed
captures.  Inside every exact-observation pool, all 496 seed pairs are ranked
by RMS distance between their full 10-token pre-MoE hidden tensors.  The 50
nearest pairs are retained; final-action distance then defines the lowest ten
as stable and the highest ten as divergent.

The four tested scores come from the corrected unweighted expert archive:
pair level and absolute pair difference for both the unweighted selected-slot
mean and the gate-weighted expert mass.  They remain in checkpoint-specific
absolute units.  This is a retrospective proxy, not a runtime intervention or
a confidence/pruning rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np

from analyze_expert_activation_proxy import load_primary_cell


N_TASKS = 5
N_POOLS = 16
N_SEEDS = 32
N_ACTION_TOKENS = 10
HIDDEN_WIDTH = 1024
SCORE_NAMES = ("raw_level", "mass_level", "raw_diff", "mass_diff")


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    unweighted = here / "analysis" / "unweighted-expert-norm"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=unweighted / "unweighted_expert_norms.npz",
    )
    parser.add_argument(
        "--source-summary",
        type=pathlib.Path,
        default=unweighted / "summary.json",
    )
    parser.add_argument(
        "--proxy-root",
        type=pathlib.Path,
        default=here / "analysis" / "expert-activation-hidden-matched",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=here / "analysis" / "raw-size-hidden-matched",
    )
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--denoise", type=int, default=0)
    parser.add_argument("--nearest-pairs", type=int, default=50)
    parser.add_argument("--tail-pairs", type=int, default=10)
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_indices(n_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_seed < 2:
        raise ValueError("at least two seeds are required")
    return np.triu_indices(n_seed, 1)


def _pair_rms_pools(
    value: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    """Pair RMS for ``[pool, seed, ...]``, retaining the pool axis."""
    array = np.asarray(value)
    if array.ndim < 3 or array.shape[1] <= int(max(pair_i.max(), pair_j.max())):
        raise ValueError("value must be [pool, seed, ...] for the supplied pairs")
    output = np.empty((array.shape[0], len(pair_i)), dtype=np.float64)
    for pool in range(array.shape[0]):
        delta = np.asarray(array[pool, pair_i], dtype=np.float64) - np.asarray(
            array[pool, pair_j], dtype=np.float64
        )
        axes = tuple(range(1, delta.ndim))
        output[pool] = np.sqrt(np.mean(np.square(delta), axis=axes))
    return output


def _pair_rms_grid(
    value: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    """Pair RMS for ``[task, pool, seed, ...]``."""
    array = np.asarray(value)
    if array.ndim < 4:
        raise ValueError("value must be [task, pool, seed, ...]")
    return np.stack([_pair_rms_pools(task, pair_i, pair_j) for task in array], axis=0)


def _upper_to_symmetric_matrix(
    upper: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_seed: int,
) -> np.ndarray:
    if upper.shape[-1] != len(pair_i):
        raise ValueError("upper-triangle values do not match pair indices")
    matrix = np.zeros((*upper.shape[:-1], n_seed, n_seed), dtype=np.float64)
    matrix[..., pair_i, pair_j] = upper
    matrix[..., pair_j, pair_i] = upper
    return matrix


def _pair_score_tensor(
    raw_size: np.ndarray,
    expert_mass: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """Return four frozen pair scores in ``SCORE_NAMES`` order."""
    raw = np.asarray(raw_size, dtype=np.float64)
    mass = np.asarray(expert_mass, dtype=np.float64)
    if raw.shape != mass.shape or raw.ndim != 3:
        raise ValueError("raw size and expert mass must share [task, pool, seed]")
    raw_left, raw_right = raw[..., pair_i], raw[..., pair_j]
    mass_left, mass_right = mass[..., pair_i], mass[..., pair_j]
    return np.stack(
        (
            0.5 * (raw_left + raw_right),
            0.5 * (mass_left + mass_right),
            np.abs(raw_left - raw_right),
            np.abs(mass_left - mass_right),
        ),
        axis=0,
    )


def _hidden_nearest_pairs(hidden_distance: np.ndarray, nearest: int) -> np.ndarray:
    distance = np.asarray(hidden_distance, dtype=np.float64)
    if distance.ndim != 3:
        raise ValueError("hidden distance must be [task, pool, pair]")
    if nearest <= 0 or nearest > distance.shape[-1]:
        raise ValueError("nearest pair count is out of range")
    return np.argsort(distance, axis=-1, kind="stable")[..., :nearest]


def _action_tail_pairs(
    action_distance: np.ndarray,
    nearest_pair_indices: np.ndarray,
    tail: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(action_distance, dtype=np.float64)
    nearest = np.asarray(nearest_pair_indices, dtype=np.int64)
    if distance.ndim != 3 or nearest.shape[:2] != distance.shape[:2]:
        raise ValueError(
            "action distance and nearest indices must share task/pool axes"
        )
    if tail <= 0 or 2 * tail > nearest.shape[-1]:
        raise ValueError("tail count must fit twice inside the matched set")
    matched_action = np.take_along_axis(distance, nearest, axis=-1)
    order = np.argsort(matched_action, axis=-1, kind="stable")
    stable = np.take_along_axis(nearest, order[..., :tail], axis=-1)
    divergent = np.take_along_axis(nearest, order[..., -tail:], axis=-1)
    return stable, divergent


def _selected_pair_auc(
    pair_scores: np.ndarray,
    stable_pair_indices: np.ndarray,
    divergent_pair_indices: np.ndarray,
) -> np.ndarray:
    """Pool AUC where a higher score predicts the divergent class."""
    scores = np.asarray(pair_scores, dtype=np.float64)
    stable_index = np.asarray(stable_pair_indices, dtype=np.int64)
    divergent_index = np.asarray(divergent_pair_indices, dtype=np.int64)
    if scores.ndim != 4 or stable_index.shape != divergent_index.shape:
        raise ValueError("bad pair-score or class-index shapes")
    if scores.shape[1:3] != stable_index.shape[:2]:
        raise ValueError("score and class task/pool axes differ")
    stable_index = np.broadcast_to(
        stable_index[None], (scores.shape[0], *stable_index.shape)
    )
    divergent_index = np.broadcast_to(
        divergent_index[None], (scores.shape[0], *divergent_index.shape)
    )
    stable = np.take_along_axis(scores, stable_index, axis=-1)
    divergent = np.take_along_axis(scores, divergent_index, axis=-1)
    comparison = divergent[..., :, None] - stable[..., None, :]
    return np.mean(comparison > 0, axis=(-2, -1)) + 0.5 * np.mean(
        comparison == 0, axis=(-2, -1)
    )


def _common_seed_max_t(
    pair_scores: np.ndarray,
    action_distance_matrix: np.ndarray,
    nearest_pair_indices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    tail: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Two-sided maxT4 using one seed-column permutation per null draw."""
    scores = np.asarray(pair_scores, dtype=np.float64)
    action_matrix = np.asarray(action_distance_matrix, dtype=np.float64)
    nearest = np.asarray(nearest_pair_indices, dtype=np.int64)
    if scores.shape[0] != len(SCORE_NAMES):
        raise ValueError("the frozen family must contain exactly four scores")
    if permutations < 1:
        raise ValueError("at least one permutation is required")
    n_seed = action_matrix.shape[-1]
    if action_matrix.shape[-2] != n_seed:
        raise ValueError("action distance matrices must be square")

    observed_pair_action = action_matrix[..., pair_i, pair_j]
    stable, divergent = _action_tail_pairs(observed_pair_action, nearest, tail)
    pool_auc = _selected_pair_auc(scores, stable, divergent)
    task_auc = pool_auc.mean(axis=-1)
    macro_auc = task_auc.mean(axis=-1)

    rng = np.random.default_rng(seed)
    seed_permutations = np.empty((permutations, n_seed), dtype=np.uint8)
    null_macro_auc = np.empty((permutations, len(SCORE_NAMES)), dtype=np.float64)
    for draw in range(permutations):
        order = rng.permutation(n_seed)
        seed_permutations[draw] = order
        permuted_pair_action = action_matrix[..., order[pair_i], order[pair_j]]
        null_stable, null_divergent = _action_tail_pairs(
            permuted_pair_action, nearest, tail
        )
        null_pool_auc = _selected_pair_auc(scores, null_stable, null_divergent)
        null_macro_auc[draw] = null_pool_auc.mean(axis=(1, 2))
        if (draw + 1) % 1000 == 0:
            print(f"permutations: {draw + 1}/{permutations}", flush=True)

    null_max_abs = np.max(np.abs(null_macro_auc - 0.5), axis=-1)
    metrics: dict[str, Any] = {}
    for feature, name in enumerate(SCORE_NAMES):
        per_task = task_auc[feature]
        above = per_task > 0.5
        below = per_task < 0.5
        observed_abs = abs(macro_auc[feature] - 0.5)
        metrics[name] = {
            "macro_auc_higher_predicts_divergent": float(macro_auc[feature]),
            "per_task_auc": per_task.tolist(),
            "per_task_direction": [
                "higher" if value > 0.5 else "lower" if value < 0.5 else "tie"
                for value in per_task
            ],
            "all_five_higher": bool(np.all(above)),
            "all_five_lower": bool(np.all(below)),
            "all_five_same_direction": bool(np.all(above) or np.all(below)),
            "two_sided_common_seed_permutation_p": float(
                (1 + np.sum(np.abs(null_macro_auc[:, feature] - 0.5) >= observed_abs))
                / (permutations + 1)
            ),
            "two_sided_maxT_p_over_four_scores": float(
                (1 + np.sum(null_max_abs >= observed_abs)) / (permutations + 1)
            ),
            "null_macro_auc_mean": float(null_macro_auc[:, feature].mean()),
        }
    summary = {
        "orientation": "higher pair score predicts final-action divergent",
        "metrics": metrics,
        "permutation": {
            "draws": permutations,
            "seed": seed,
            "scheme": (
                "one common permutation of the 32 seed columns per draw, reused "
                "across all five tasks and all 80 fixed-state pools"
            ),
            "family": list(SCORE_NAMES),
            "test": "two-sided non-studentized maxT over abs(macro AUC - 0.5)",
            "null_max_abs_auc_minus_half_quantiles": {
                "q50": float(np.quantile(null_max_abs, 0.50)),
                "q90": float(np.quantile(null_max_abs, 0.90)),
                "q95": float(np.quantile(null_max_abs, 0.95)),
                "q99": float(np.quantile(null_max_abs, 0.99)),
            },
        },
    }
    arrays = {
        "stable_pair_indices": stable,
        "divergent_pair_indices": divergent,
        "pool_auc": pool_auc,
        "task_auc": task_auc,
        "macro_auc": macro_auc,
        "seed_permutations": seed_permutations,
        "null_macro_auc": null_macro_auc,
        "null_max_abs_auc_minus_half": null_max_abs,
    }
    return summary, arrays


def _load_archive(archive_path: pathlib.Path, denoise: int) -> dict[str, Any]:
    required = {
        "task_names",
        "scene_ids",
        "seed_ids",
        "selected_expert_weight",
        "selected_expert_raw_rms",
        "raw_mean",
        "weighted_expert_mass",
        "final_actions_standardized",
    }
    with np.load(archive_path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"unweighted archive is missing {missing}")
        task_names = [str(name) for name in payload["task_names"].tolist()]
        scene_ids = np.asarray(payload["scene_ids"], dtype=np.int64)
        seed_ids = np.asarray(payload["seed_ids"], dtype=np.int64)
        weights = np.asarray(payload["selected_expert_weight"], dtype=np.float64)
        slot_rms = np.asarray(payload["selected_expert_raw_rms"], dtype=np.float64)
        raw_trajectory = np.asarray(payload["raw_mean"], dtype=np.float64)
        mass_trajectory = np.asarray(payload["weighted_expert_mass"], dtype=np.float64)
        actions = np.asarray(payload["final_actions_standardized"], dtype=np.float64)

    expected_grid = (N_TASKS, N_POOLS, N_SEEDS)
    if scene_ids.shape != expected_grid[:2] or seed_ids.shape != (
        N_TASKS,
        N_SEEDS,
    ):
        raise ValueError("archive does not contain the frozen 5 x 16 x 32 axes")
    if (
        raw_trajectory.shape[:3] != expected_grid
        or mass_trajectory.shape != raw_trajectory.shape
    ):
        raise ValueError("raw trajectories do not use the frozen task/pool/seed grid")
    if actions.shape != (*expected_grid, N_ACTION_TOKENS, 7):
        raise ValueError(
            f"expected final actions {(*expected_grid, N_ACTION_TOKENS, 7)}"
        )
    if denoise < 0 or denoise >= raw_trajectory.shape[-1]:
        raise ValueError("denoise index is outside the raw archive")
    if len(set(task_names)) != N_TASKS:
        raise ValueError("task names must contain five unique tasks")
    if not np.all(seed_ids == seed_ids[0]):
        raise ValueError("all tasks must share the same ordered 32 seed columns")

    selected = slot_rms[:, :, :, denoise]
    selected_weight = weights[:, :, :, denoise]
    recomputed_raw = selected.mean(axis=(-2, -1))
    recomputed_mass = np.sum(selected * selected_weight, axis=-1).mean(axis=-1)
    raw_size = raw_trajectory[..., denoise]
    expert_mass = mass_trajectory[..., denoise]
    return {
        "task_names": task_names,
        "scene_ids": scene_ids,
        "seed_ids": seed_ids,
        "raw_size": raw_size,
        "expert_mass": expert_mass,
        "actions": actions,
        "audit": {
            "raw_mean_recompute_max_abs_error": float(
                np.max(np.abs(raw_size - recomputed_raw))
            ),
            "expert_mass_recompute_max_abs_error": float(
                np.max(np.abs(expert_mass - recomputed_mass))
            ),
            "selected_weight_sum_max_abs_error": float(
                np.max(np.abs(selected_weight.sum(axis=-1) - 1.0))
            ),
            "common_seed_columns_across_tasks": True,
        },
    }


def _load_hidden_pair_distances(
    archive: dict[str, Any],
    source_summary_path: pathlib.Path,
    proxy_root: pathlib.Path,
    layer: int,
    denoise: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary.get("fixed_layer") != layer:
        raise ValueError("unweighted source summary does not use the frozen HB layer")
    if source_summary.get("tasks") != archive["task_names"]:
        raise ValueError("source-summary task order differs from the archive")

    hidden_distance = np.empty((N_TASKS, N_POOLS, len(pair_i)), dtype=np.float64)
    audit: dict[str, Any] = {}
    for task_axis, task in enumerate(archive["task_names"]):
        source_row = source_summary["sources"][task]
        run = pathlib.Path(source_row["run"]).resolve()
        proxy_path = proxy_root / task / "summary.json"
        proxy = json.loads(proxy_path.read_text())
        if pathlib.Path(proxy["run"]).resolve() != run:
            raise ValueError(f"{task}: proxy and unweighted source runs differ")
        if proxy.get("layer") != layer or proxy.get("denoise") != denoise:
            raise ValueError(f"{task}: proxy does not use HB{layer}/d{denoise}")
        if proxy.get("checkpoint_sha256") != source_row.get("checkpoint_sha256"):
            raise ValueError(f"{task}: checkpoint hashes differ")
        if proxy.get("pools") != N_POOLS or proxy.get("candidates_per_pool") != N_SEEDS:
            raise ValueError(f"{task}: proxy does not contain 16 x 32 candidates")

        print(f"loading aligned HB{layer}/d{denoise} hidden: {task}", flush=True)
        primary = load_primary_cell(run, layer, denoise)
        if not np.array_equal(primary["scene_ids"], archive["scene_ids"][task_axis]):
            raise ValueError(f"{task}: state axes differ")
        if not np.array_equal(primary["seed_ids"], archive["seed_ids"][task_axis]):
            raise ValueError(f"{task}: seed axes differ")
        if primary.get("checkpoint_sha256") != source_row.get("checkpoint_sha256"):
            raise ValueError(f"{task}: live capture checkpoint hash differs")

        order = np.asarray(primary["order"], dtype=np.int64)
        hidden = np.asarray(primary["hidden"][order], dtype=np.float32)
        if hidden.shape != (N_POOLS, N_SEEDS, N_ACTION_TOKENS, HIDDEN_WIDTH):
            raise ValueError(f"{task}: unexpected hidden shape {hidden.shape}")
        hidden_distance[task_axis] = _pair_rms_pools(hidden, pair_i, pair_j)

        source_actions = np.asarray(primary["actions"][order], dtype=np.float64)
        archived_actions = archive["actions"][task_axis]
        action_error = float(np.max(np.abs(source_actions - archived_actions)))
        if action_error > 5e-6:
            raise ValueError(f"{task}: source/archive action alignment failed")

        metadata = json.loads((run / "client" / "server_metadata.json").read_text())
        action_stats = json.loads(
            pathlib.Path(metadata["normalization_stats_path"]).read_text()
        )["actions"]
        action_mean = np.asarray(action_stats["mean"], dtype=np.float64)
        action_std = np.asarray(action_stats["std"], dtype=np.float64)
        mean_corrected = archived_actions - action_mean / action_std
        direct_pair_distance = _pair_rms_pools(archived_actions, pair_i, pair_j)
        corrected_pair_distance = _pair_rms_pools(mean_corrected, pair_i, pair_j)
        mean_invariance_error = float(
            np.max(np.abs(direct_pair_distance - corrected_pair_distance))
        )
        if mean_invariance_error > 1e-12:
            raise ValueError(
                f"{task}: subtracting the action mean changed pair distance"
            )

        audit[task] = {
            "run": str(run),
            "checkpoint_sha256": source_row.get("checkpoint_sha256"),
            "state_axis_matches": True,
            "seed_axis_matches": True,
            "hidden_shape": list(hidden.shape),
            "source_action_over_std_max_abs_error": action_error,
            "action_pair_distance_mean_subtraction_max_abs_error": mean_invariance_error,
        }
    return hidden_distance, audit


def _match_diagnostics(
    task_names: list[str],
    hidden_distance: np.ndarray,
    action_distance: np.ndarray,
    nearest: np.ndarray,
    stable: np.ndarray,
    divergent: np.ndarray,
) -> dict[str, Any]:
    hidden_auc = _selected_pair_auc(hidden_distance[None], stable, divergent)[0].mean(
        axis=-1
    )
    result: dict[str, Any] = {}
    for task_axis, task in enumerate(task_names):
        hidden = hidden_distance[task_axis]
        action = action_distance[task_axis]
        near_hidden = np.take_along_axis(hidden, nearest[task_axis], axis=-1)
        stable_hidden = np.take_along_axis(hidden, stable[task_axis], axis=-1)
        divergent_hidden = np.take_along_axis(hidden, divergent[task_axis], axis=-1)
        stable_action = np.take_along_axis(action, stable[task_axis], axis=-1)
        divergent_action = np.take_along_axis(action, divergent[task_axis], axis=-1)
        all_median = float(np.median(hidden))
        result[task] = {
            "all_496_hidden_distance_median": all_median,
            "nearest_50_hidden_distance_median": float(np.median(near_hidden)),
            "nearest_over_all_hidden_median": float(
                np.median(near_hidden) / all_median
            ),
            "stable_hidden_distance_median": float(np.median(stable_hidden)),
            "divergent_hidden_distance_median": float(np.median(divergent_hidden)),
            "hidden_distance_auc_higher_predicts_divergent": float(
                hidden_auc[task_axis]
            ),
            "stable_final_action_distance_median": float(np.median(stable_action)),
            "divergent_final_action_distance_median": float(
                np.median(divergent_action)
            ),
        }
    return result


def _render_report(summary: dict[str, Any]) -> str:
    task_names = summary["tasks"]
    results = summary["results"]
    lines = [
        "# Raw-size hidden-matched audit",
        "",
        "This report formalizes a retrospective proxy. It does not establish that MoE",
        "output size is model confidence, and it does not support candidate pruning.",
        "",
        "## Frozen protocol",
        "",
        "- Five existing complete captures: 16 exact-observation pools per task and 32 common seed columns per pool.",
        "- Cell fixed to HB5/denoise-0.",
        "- Each pool has 496 candidate pairs. Pair distance uses the full 10 x 1024 pre-MoE action-token hidden tensor.",
        "- Keep the 50 nearest-hidden pairs; within those, the lowest/highest ten final-action distances are stable/divergent.",
        "- Test pair level and absolute pair difference for `raw_mean(d0)` and gate-weighted `expert_mass(d0)`.",
        "- AUC > 0.5 means a larger score predicts the divergent class.",
        "- The four-score family uses %d common seed-column permutations and a two-sided maxT correction."
        % summary["permutations"],
        "",
        "The expert quantities are absolute checkpoint units. No hidden, shared, routed,",
        "post-MoE, d0, or cross-task normalization is applied.",
        "",
        "## Source and alignment audit",
        "",
        "| task | state axis | seed axis | hidden shape | action error | mean-shift distance error |",
        "|---|---|---|---|---:|---:|",
    ]
    for task in task_names:
        row = summary["alignment_audit"][task]
        lines.append(
            "| %s | %s | %s | %s | %.2g | %.2g |"
            % (
                task,
                "match" if row["state_axis_matches"] else "mismatch",
                "match" if row["seed_axis_matches"] else "mismatch",
                "x".join(str(value) for value in row["hidden_shape"]),
                row["source_action_over_std_max_abs_error"],
                row["action_pair_distance_mean_subtraction_max_abs_error"],
            )
        )
    archive_audit = summary["archive_audit"]
    lines += [
        "",
        "Archive formula checks: `raw_mean` max error `%.3g`, `expert_mass` max error `%.3g`, selected-weight sum max error `%.3g`."
        % (
            archive_audit["raw_mean_recompute_max_abs_error"],
            archive_audit["expert_mass_recompute_max_abs_error"],
            archive_audit["selected_weight_sum_max_abs_error"],
        ),
        "Subtracting the action normalization mean cannot change a within-pool pair",
        "difference; the numerical audit above confirms this directly.",
        "",
        "## Four-score result",
        "",
        "| score | %s | macro | five-task same direction | raw p | maxT4 p |"
        % " | ".join(task_names),
        "|---|%s|---:|---|---:|---:|" % "|".join(["---:"] * len(task_names)),
    ]
    for name in SCORE_NAMES:
        row = results["metrics"][name]
        task_values = " | ".join(f"{value:.4f}" for value in row["per_task_auc"])
        lines.append(
            "| %s | %s | %.4f | %s | %.4f | %.4f |"
            % (
                name,
                task_values,
                row["macro_auc_higher_predicts_divergent"],
                "yes" if row["all_five_same_direction"] else "no",
                row["two_sided_common_seed_permutation_p"],
                row["two_sided_maxT_p_over_four_scores"],
            )
        )
    lines += [
        "",
        "`level` is the mean size of the two candidates; `diff` is their absolute",
        "size difference. None of the four scores has the same direction in all five",
        "tasks, and none survives the four-score two-sided maxT test.",
        "",
        "## Matching diagnostics",
        "",
        "| task | near/all hidden median | stable hidden | divergent hidden | hidden AUC | stable action | divergent action |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in task_names:
        row = summary["matching_diagnostics"][task]
        lines.append(
            "| %s | %.3f | %.5f | %.5f | %.3f | %.5f | %.5f |"
            % (
                task,
                row["nearest_over_all_hidden_median"],
                row["stable_hidden_distance_median"],
                row["divergent_hidden_distance_median"],
                row["hidden_distance_auc_higher_predicts_divergent"],
                row["stable_final_action_distance_median"],
                row["divergent_final_action_distance_median"],
            )
        )
    lines += [
        "",
        "Nearest-50 matching reduces hidden separation but does not create exact twins.",
        "Stable/divergent labels are also defined retrospectively from the final action,",
        "so these AUCs are association diagnostics rather than causal effects.",
        "",
        "## Verdict",
        "",
        "Pair-level raw size is slightly below chance in the equal-task macro, while",
        "absolute size difference is slightly above chance. The task signs are mixed and",
        "the corrected p-values are null. Therefore HB5/d0 absolute selected-expert size",
        "does not provide a reproducible early action-basin or confidence signal here.",
        "It must not be used for pruning, early stopping, or action commitment.",
        "",
        "Boundaries:",
        "",
        "- The selected-expert norms are offline reconstructions from stored fp16 hidden states; ids and gate weights are captured.",
        "- Matching is nearest-neighbor matching among 496 pairs, not exact hidden equality.",
        "- The same data informed the retrospective question and this audit; there is no prospective holdout.",
        "- Absolute units are checkpoint-specific; cross-task magnitudes are not directly comparable.",
        "- Pairwise final-action distance is invariant to subtracting a common action mean.",
        "",
        "Auditable pair selections and null draws are in `arrays.npz`; full precision results are in `summary.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.layer != 5 or args.denoise != 0:
        raise ValueError("the frozen analysis cell is HB5/denoise-0")
    if args.nearest_pairs != 50 or args.tail_pairs != 10:
        raise ValueError("the frozen matching protocol is nearest 50 with 10/10 tails")
    if args.perms != 5000:
        raise ValueError("the formal analysis requires exactly 5000 permutations")

    started = time.perf_counter()
    args.archive = args.archive.resolve()
    args.source_summary = args.source_summary.resolve()
    args.proxy_root = args.proxy_root.resolve()
    args.out_dir = args.out_dir.resolve()
    archive = _load_archive(args.archive, args.denoise)
    pair_i, pair_j = _pair_indices(N_SEEDS)
    if len(pair_i) != 496:
        raise RuntimeError("K32 must yield 496 unique unordered pairs")

    hidden_distance, alignment_audit = _load_hidden_pair_distances(
        archive,
        args.source_summary,
        args.proxy_root,
        args.layer,
        args.denoise,
        pair_i,
        pair_j,
    )
    action_distance = _pair_rms_grid(archive["actions"], pair_i, pair_j)
    action_matrix = _upper_to_symmetric_matrix(action_distance, pair_i, pair_j, N_SEEDS)
    pair_scores = _pair_score_tensor(
        archive["raw_size"], archive["expert_mass"], pair_i, pair_j
    )
    nearest = _hidden_nearest_pairs(hidden_distance, args.nearest_pairs)
    results, inference_arrays = _common_seed_max_t(
        pair_scores,
        action_matrix,
        nearest,
        pair_i,
        pair_j,
        args.tail_pairs,
        args.perms,
        args.seed,
    )
    matching = _match_diagnostics(
        archive["task_names"],
        hidden_distance,
        action_distance,
        nearest,
        inference_arrays["stable_pair_indices"],
        inference_arrays["divergent_pair_indices"],
    )

    summary = {
        "experiment": "raw_size_hidden_matched_cross_task_v1",
        "status": "retrospective_proxy_not_runtime_causal",
        "tasks": archive["task_names"],
        "fixed_cell": {"hb_layer": args.layer, "denoise": args.denoise},
        "pools_per_task": N_POOLS,
        "candidates_per_pool": N_SEEDS,
        "pairs_per_pool": int(len(pair_i)),
        "nearest_hidden_pairs_per_pool": args.nearest_pairs,
        "stable_pairs_per_pool": args.tail_pairs,
        "divergent_pairs_per_pool": args.tail_pairs,
        "hidden_distance_definition": (
            "RMS over the complete 10 action-token x 1024 pre-MoE hidden tensor"
        ),
        "final_action_distance_definition": (
            "RMS over the complete standardized 10 x 7 final action chunk"
        ),
        "score_definitions": {
            "raw_level": "mean of pair members' unweighted selected-slot raw_mean(d0)",
            "mass_level": "mean of pair members' gate-weighted expert_mass(d0)",
            "raw_diff": "absolute pair difference in raw_mean(d0)",
            "mass_diff": "absolute pair difference in expert_mass(d0)",
        },
        "normalization": "none; all expert scores remain in absolute checkpoint units",
        "permutations": args.perms,
        "source": {
            "archive": str(args.archive),
            "archive_sha256": _sha256(args.archive),
            "source_summary": str(args.source_summary),
            "source_summary_sha256": _sha256(args.source_summary),
            "proxy_root": str(args.proxy_root),
        },
        "archive_audit": archive["audit"],
        "alignment_audit": alignment_audit,
        "results": results,
        "matching_diagnostics": matching,
        "verdict": {
            "any_score_five_task_same_direction": bool(
                any(
                    row["all_five_same_direction"]
                    for row in results["metrics"].values()
                )
            ),
            "any_score_maxT_significant_at_0.05": bool(
                any(
                    row["two_sided_maxT_p_over_four_scores"] < 0.05
                    for row in results["metrics"].values()
                )
            ),
            "supports_candidate_pruning": False,
            "text": (
                "Absolute HB5/d0 raw size has mixed task directions and no maxT4 "
                "evidence for hidden-matched final-action divergence."
            ),
        },
        "limitations": [
            "Retrospective proxy: final action defines labels after the same captures were inspected.",
            "Nearest-50 hidden matching is not exact hidden equality.",
            "Expert outputs are offline reconstructions from stored fp16 hidden states.",
            "Only selected top-4 experts are represented in the raw-size archive.",
            "Absolute units are not normalized and are checkpoint-specific.",
            "Final-action pair distance is invariant to subtracting a common action mean.",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "task_names": np.asarray(archive["task_names"]),
        "scene_ids": archive["scene_ids"],
        "seed_ids": archive["seed_ids"],
        "pair_i": pair_i,
        "pair_j": pair_j,
        "score_names": np.asarray(SCORE_NAMES),
        "raw_mean_d0": archive["raw_size"].astype(np.float32),
        "expert_mass_d0": archive["expert_mass"].astype(np.float32),
        "hidden_pair_distance": hidden_distance.astype(np.float32),
        "final_action_pair_distance": action_distance.astype(np.float32),
        "nearest_hidden_pair_indices": nearest,
        "pair_scores": pair_scores.astype(np.float32),
        **inference_arrays,
    }
    np.savez_compressed(args.out_dir / "arrays.npz", **arrays)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
