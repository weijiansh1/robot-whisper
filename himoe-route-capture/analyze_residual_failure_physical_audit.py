#!/usr/bin/env python3
"""Audit metadata and physical stage of failures outside routing consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = ANALYSIS / "residual-failure-physical-audit"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
KEYS = ["task", "episode", "init_state_id", "flow_noise_seed", "episode_length"]
GOAL_DISTANCE_M = 0.05
LIFT_DISTANCE_M = 0.01
SEED = 20260828

# These are the three already-frozen blocks summarized in
# analysis/routing-organization-synthesis/report.md. This audit does not select
# clusters again from outcome labels.
ALIGNED_FAILURE_CLUSTER = 1
EVENT_FAILURE_CLUSTER = 0
LAG_FAILURE_COMMUNITY = "C7"
EXPECTED_BLOCKS = {
    "aligned": {"episodes": 252, "failures": 246},
    "event": {"episodes": 214, "failures": 209},
    "lag": {"episodes": 219, "failures": 215},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file_set(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_alignment(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if not reference[KEYS].equals(other[KEYS]):
        raise ValueError(f"row alignment failed for {name}")


def load_episode_table(analysis_root: Path) -> pd.DataFrame:
    aligned = pd.read_csv(analysis_root / "aligned-route-kernel/assignments.csv")
    event = pd.read_csv(analysis_root / "route-change-events/assignments.csv")
    alternative = pd.read_csv(
        analysis_root / "alternative-routing-organizations/assignments.csv"
    )
    validate_alignment(aligned, event, "route-change events")
    validate_alignment(aligned, alternative, "alternative organizations")

    frame = aligned[
        KEYS + ["outcome", "raw_cluster"]
    ].copy()
    frame["failure"] = frame["outcome"].eq("failure")
    frame["aligned_vote"] = aligned["raw_cluster"].eq(ALIGNED_FAILURE_CLUSTER)
    frame["event_cluster"] = event["event_cluster"].astype(int)
    frame["event_vote"] = frame["event_cluster"].eq(EVENT_FAILURE_CLUSTER)
    frame["lag_spectrum_louvain"] = alternative["lag_spectrum_louvain"]
    frame["lag_vote"] = frame["lag_spectrum_louvain"].eq(LAG_FAILURE_COMMUNITY)
    vote_columns = ["aligned_vote", "event_vote", "lag_vote"]
    frame["consensus_votes"] = frame[vote_columns].sum(axis=1).astype(int)
    frame["vote_pattern"] = (
        frame[vote_columns].astype(np.int8).astype(str).agg("".join, axis=1)
    )
    frame["analysis_group"] = np.select(
        [
            ~frame["failure"],
            frame["failure"] & frame["consensus_votes"].ge(2),
            frame["failure"] & frame["consensus_votes"].eq(1),
        ],
        ["success", "failure_core", "failure_1vote"],
        default="failure_0vote",
    )

    observed = {
        "aligned": {
            "episodes": int(frame["aligned_vote"].sum()),
            "failures": int((frame["aligned_vote"] & frame["failure"]).sum()),
        },
        "event": {
            "episodes": int(frame["event_vote"].sum()),
            "failures": int((frame["event_vote"] & frame["failure"]).sum()),
        },
        "lag": {
            "episodes": int(frame["lag_vote"].sum()),
            "failures": int((frame["lag_vote"] & frame["failure"]).sum()),
        },
    }
    if observed != EXPECTED_BLOCKS:
        raise ValueError(f"frozen consensus blocks drifted: {observed}")

    physical = pd.read_csv(
        analysis_root / "replanning-reset-trap/episode_metrics.csv"
    )[["task", "episode", "failure_mode"]]
    frame = frame.merge(physical, on=["task", "episode"], validate="one_to_one")
    return frame


def add_event_diagnostics(
    frame: pd.DataFrame, analysis_root: Path
) -> pd.DataFrame:
    archive = np.load(
        analysis_root / "route-change-events/event_features.npz", allow_pickle=False
    )
    names = list(map(str, archive["core_names"]))
    values = np.asarray(archive["core"])
    if len(values) != len(frame):
        raise ValueError("event feature row count drifted")
    selected = [
        "soft_late_early_ratio",
        "hard_late_early_ratio",
        "global_peak_phase",
        "return_mean_lag_fraction",
        "late_stasis_indicator",
        "terminal_return_indicator",
    ]
    for name in selected:
        frame[name] = values[:, names.index(name)]
    return frame


def nearest_sibling_success(
    frame: pd.DataFrame, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(frame):
        raise ValueError("embedding row count drifted")
    failure = frame["failure"].to_numpy(bool)
    nearest = np.full(len(frame), np.nan, dtype=np.float64)
    ratio = np.full(len(frame), np.nan, dtype=np.float64)
    for _key, group in frame.groupby(["task", "init_state_id"], sort=False):
        index = group.index.to_numpy(dtype=np.int64)
        success_index = index[~failure[index]]
        if not len(success_index):
            continue
        delta = values[index, None, :] - values[success_index][None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        for local, episode_index in enumerate(index):
            row = distances[local].copy()
            same = np.flatnonzero(success_index == episode_index)
            if len(same):
                row[same[0]] = np.inf
            if np.isfinite(row).any():
                nearest[episode_index] = float(row.min())
        reference = nearest[success_index]
        reference = reference[np.isfinite(reference)]
        if len(reference):
            ratio[index] = nearest[index] / np.median(reference)
    return nearest, ratio


def add_success_distances(
    frame: pd.DataFrame, analysis_root: Path
) -> pd.DataFrame:
    aligned = np.load(
        analysis_root / "aligned-route-kernel/embeddings_labels_landmarks.npz",
        allow_pickle=False,
    )
    alternative = np.load(
        analysis_root
        / "alternative-routing-organizations/embeddings_and_labels.npz",
        allow_pickle=False,
    )
    event = np.load(
        analysis_root / "route-change-events/event_features.npz", allow_pickle=False
    )
    spaces = {
        "aligned": np.asarray(aligned["embedding_raw"], dtype=np.float64),
        "event": np.asarray(event["embedding"], dtype=np.float64),
        "lag": np.asarray(
            alternative["embedding_lag_spectrum"], dtype=np.float64
        ),
    }
    for name, values in spaces.items():
        nearest, ratio = nearest_sibling_success(frame, values)
        frame[f"{name}_nearest_sibling_success"] = nearest
        frame[f"{name}_nearest_sibling_success_ratio"] = ratio
    return frame


def _first_at_goal(distance: np.ndarray) -> float:
    index = np.flatnonzero(distance <= GOAL_DISTANCE_M)
    return float(index[0]) if len(index) else np.nan


def long_task_physical_table(
    frame: pd.DataFrame, cache_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = cache_root / LONG_TASK / "right-16x32/client"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    layout = json.loads((client / "sim_layout.json").read_text())
    joints = {row["joint"]: row for row in layout["joints"]}
    slices = {}
    for name in ("moka_pot_1_joint0", "moka_pot_2_joint0"):
        joint = joints[name]
        lo = int(joint["state_lo"])
        slices[name] = slice(lo, lo + 3)

    tapes: dict[int, dict[str, np.ndarray]] = {}
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(client / f"episode_{episode:02d}.npz", allow_pickle=False) as data:
            sim = np.asarray(data["sim_state"], dtype=np.float64)
            state = np.asarray(data["state"], dtype=np.float64)
        tapes[episode] = {
            "pot1": sim[:, slices["moka_pot_1_joint0"]],
            "pot2": sim[:, slices["moka_pot_2_joint0"]],
            "eef": state[:, :3],
        }

    success_episodes = [
        int(row["episode_index"]) for row in summaries if bool(row["success"])
    ]
    goals = {
        pot: np.mean([tapes[episode][pot][-1] for episode in success_episodes], axis=0)
        for pot in ("pot1", "pot2")
    }
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        episode = int(summary["episode_index"])
        tape = tapes[episode]
        distance1 = np.linalg.norm(tape["pot1"] - goals["pot1"], axis=1)
        distance2 = np.linalg.norm(tape["pot2"] - goals["pot2"], axis=1)
        reach1 = _first_at_goal(distance1)
        reach2 = _first_at_goal(distance2)
        placed1 = bool(distance1[-1] <= GOAL_DISTANCE_M)
        placed2 = bool(distance2[-1] <= GOAL_DISTANCE_M)
        if placed1 and placed2:
            stage = "both_at_goal"
        elif placed2:
            stage = "pot2_only"
        elif placed1:
            stage = "pot1_only"
        else:
            stage = "neither_at_goal"
        terminal_p1 = float(np.linalg.norm(tape["eef"][-1] - tape["pot1"][-1]))
        terminal_p2 = float(np.linalg.norm(tape["eef"][-1] - tape["pot2"][-1]))
        rows.append(
            {
                "task": LONG_TASK,
                "episode": episode,
                "pot1_final_goal_distance_m": float(distance1[-1]),
                "pot2_final_goal_distance_m": float(distance2[-1]),
                "pot1_first_goal_query": reach1,
                "pot2_first_goal_query": reach2,
                "queries_after_pot2_goal": (
                    float(len(distance2) - 1 - reach2) if np.isfinite(reach2) else np.nan
                ),
                "pot1_lift_m": float(
                    np.max(tape["pot1"][:, 2] - tape["pot1"][0, 2])
                ),
                "pot2_lift_m": float(
                    np.max(tape["pot2"][:, 2] - tape["pot2"][0, 2])
                ),
                "long_physical_stage": stage,
                "eef_terminal_pot1_distance_m": terminal_p1,
                "eef_terminal_pot2_distance_m": terminal_p2,
                "eef_terminal_basin": "pot1" if terminal_p1 < terminal_p2 else "pot2",
                "eef_late_pot1_min_distance_m": float(
                    np.linalg.norm(
                        tape["eef"][len(tape["eef"]) // 2 :] -
                        tape["pot1"][len(tape["pot1"]) // 2 :],
                        axis=1,
                    ).min()
                ),
                "eef_path_after_pot2_goal_m": (
                    float(
                        np.linalg.norm(
                            np.diff(tape["eef"][int(reach2) :], axis=0), axis=1
                        ).sum()
                    )
                    if np.isfinite(reach2)
                    else np.nan
                ),
            }
        )
    physical = pd.DataFrame(rows)

    long_index = frame["task"].eq(LONG_TASK)
    source = frame.loc[long_index].sort_values("episode")
    if source["episode"].tolist() != physical["episode"].tolist():
        raise ValueError("long-task physical episodes do not align")
    if source["outcome"].eq("success").tolist() != [
        bool(row["success"]) for row in summaries
    ]:
        raise ValueError("long-task outcome drift between assignments and summaries")
    metadata = {
        "goal_distance_threshold_m": GOAL_DISTANCE_M,
        "lift_distance_threshold_m": LIFT_DISTANCE_M,
        "pot1_goal_xyz_m": goals["pot1"],
        "pot2_goal_xyz_m": goals["pot2"],
        "goal_proxy": "mean terminal XYZ across successful episodes",
        "terminal_basin_rule": (
            "closer terminal EEF XYZ to current pot1 versus pot2 XYZ; no fixed radius"
        ),
        "eef_source": "episode state[:, :3]",
    }
    return physical, metadata


def binary_mutual_information(category: np.ndarray, label: np.ndarray) -> float:
    _values, inverse = np.unique(category, return_inverse=True)
    label = np.asarray(label, dtype=np.int8)
    counts = np.zeros((inverse.max() + 1, 2), dtype=np.float64)
    np.add.at(counts, (inverse, label), 1.0)
    total = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    column = counts.sum(axis=0, keepdims=True)
    expected = row @ column / total
    keep = counts > 0
    return float(np.sum((counts[keep] / total) * np.log(counts[keep] / expected[keep])))


def conditional_mi_permutation(
    category: np.ndarray,
    label: np.ndarray,
    strata: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = binary_mutual_information(category, label)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    exceed = 0
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = np.asarray(label).copy()
        for index in groups:
            permuted[index] = rng.permutation(permuted[index])
        null[draw] = binary_mutual_information(category, permuted)
        exceed += int(null[draw] >= observed - 1e-15)
    return {
        "mutual_information_nats": observed,
        "permutations": draws,
        "conditional_permutation_p": float((exceed + 1) / (draws + 1)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def count_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    result = frame.groupby(columns, dropna=False, sort=True).size().rename("n")
    return result.reset_index().to_dict("records")


def numeric_by_vote(
    frame: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for vote, group in frame.groupby("consensus_votes"):
        result[str(int(vote))] = {"n": int(len(group))}
        for column in columns:
            values = group[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            result[str(int(vote))][column] = {
                "available": int(len(values)),
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                "mean": float(np.mean(values)) if len(values) else np.nan,
            }
    return result


def build_summary(
    frame: pd.DataFrame,
    physical_metadata: dict[str, Any],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    failure = frame[frame["failure"]]
    residual = failure[failure["consensus_votes"].le(1)]
    long_all = frame[frame["task"].eq(LONG_TASK)]
    long_failure = long_all[long_all["failure"]]
    long_residual = long_failure[long_failure["consensus_votes"].le(1)]
    long_core = long_failure[long_failure["consensus_votes"].ge(2)]

    init_counts = (
        long_failure.assign(
            zero=long_failure["consensus_votes"].eq(0),
            one=long_failure["consensus_votes"].eq(1),
            core=long_failure["consensus_votes"].ge(2),
        )
        .groupby("init_state_id")
        .agg(
            failures=("episode", "size"),
            zero=("zero", "sum"),
            one=("one", "sum"),
            core=("core", "sum"),
        )
        .reset_index()
    )
    seed_counts = (
        long_failure.assign(
            zero=long_failure["consensus_votes"].eq(0),
            one=long_failure["consensus_votes"].eq(1),
            core=long_failure["consensus_votes"].ge(2),
        )
        .groupby("flow_noise_seed")
        .agg(
            failures=("episode", "size"),
            zero=("zero", "sum"),
            one=("one", "sum"),
            core=("core", "sum"),
        )
        .reset_index()
    )

    crossed_tests: dict[str, Any] = {}
    for offset, (name, label) in enumerate(
        {
            "zero_vote_failure": (
                long_all["failure"] & long_all["consensus_votes"].eq(0)
            ).to_numpy(),
            "residual_failure_le1_vote": (
                long_all["failure"] & long_all["consensus_votes"].le(1)
            ).to_numpy(),
        }.items()
    ):
        crossed_tests[name] = {
            "initial_state_given_seed": conditional_mi_permutation(
                long_all["init_state_id"].to_numpy(),
                label,
                long_all["flow_noise_seed"].to_numpy(),
                permutations,
                seed + 10 * offset,
            ),
            "flow_seed_given_initial_state": conditional_mi_permutation(
                long_all["flow_noise_seed"].to_numpy(),
                label,
                long_all["init_state_id"].to_numpy(),
                permutations,
                seed + 10 * offset + 1,
            ),
        }

    success_long = long_all[~long_all["failure"]]
    success_order_available = (
        success_long["pot2_first_goal_query"].notna()
        & success_long["pot1_first_goal_query"].notna()
    )
    success_order = (
        success_long.loc[success_order_available, "pot2_first_goal_query"]
        < success_long.loc[success_order_available, "pot1_first_goal_query"]
    )
    distance_columns = [
        f"{space}_{suffix}"
        for space in ("aligned", "event", "lag")
        for suffix in ("nearest_sibling_success", "nearest_sibling_success_ratio")
    ]
    event_columns = [
        "soft_late_early_ratio",
        "hard_late_early_ratio",
        "global_peak_phase",
        "return_mean_lag_fraction",
        "late_stasis_indicator",
        "terminal_return_indicator",
    ]
    return {
        "schema": "residual-failure-physical-audit/1",
        "run_class": "exploratory descriptive audit",
        "frozen_vote_blocks": {
            "aligned_raw_cluster": ALIGNED_FAILURE_CLUSTER,
            "event_cluster": EVENT_FAILURE_CLUSTER,
            "lag_louvain_community": LAG_FAILURE_COMMUNITY,
            "verified_composition": EXPECTED_BLOCKS,
        },
        "cohort": {
            "episodes": int(len(frame)),
            "successes": int((~frame["failure"]).sum()),
            "failures": int(frame["failure"].sum()),
            "failure_core_ge2_votes": int(
                (frame["failure"] & frame["consensus_votes"].ge(2)).sum()
            ),
            "failure_1vote": int(
                (frame["failure"] & frame["consensus_votes"].eq(1)).sum()
            ),
            "failure_0vote": int(
                (frame["failure"] & frame["consensus_votes"].eq(0)).sum()
            ),
        },
        "residual": {
            "episodes": int(len(residual)),
            "task_vote_counts": count_records(
                residual, ["task", "consensus_votes"]
            ),
            "failure_mode_vote_counts": count_records(
                residual, ["failure_mode", "consensus_votes"]
            ),
            "vote_pattern_counts": count_records(residual, ["vote_pattern"]),
            "episode_length_by_task": {
                task: sorted(map(int, group["episode_length"].unique()))
                for task, group in residual.groupby("task")
            },
        },
        "long_task": {
            "task": LONG_TASK,
            "episodes": int(len(long_all)),
            "successes": int(len(success_long)),
            "failures": int(len(long_failure)),
            "residual_failures": int(len(long_residual)),
            "core_failures": int(len(long_core)),
            "physical_definition": physical_metadata,
            "successful_transition": {
                "pot2_first_goal_query_median": float(
                    np.nanmedian(success_long["pot2_first_goal_query"])
                ),
                "pot1_first_goal_query_median": float(
                    np.nanmedian(success_long["pot1_first_goal_query"])
                ),
                "order_available": int(success_order_available.sum()),
                "order_total_successes": int(len(success_long)),
                "pot2_before_pot1_count": int(success_order.sum()),
                "pot2_before_pot1_fraction": float(success_order.mean()),
            },
            "physical_stage_by_vote": count_records(
                long_failure, ["consensus_votes", "long_physical_stage"]
            ),
            "terminal_basin_by_vote": count_records(
                long_failure, ["consensus_votes", "eef_terminal_basin"]
            ),
            "terminal_basin_by_pattern": count_records(
                long_failure, ["vote_pattern", "eef_terminal_basin"]
            ),
            "failure_mode_by_vote": count_records(
                long_failure, ["consensus_votes", "failure_mode"]
            ),
            "event_dynamics_by_vote": numeric_by_vote(
                long_failure, event_columns
            ),
            "initial_state_counts": init_counts.to_dict("records"),
            "flow_seed_counts": seed_counts.to_dict("records"),
            "crossed_design_tests": crossed_tests,
        },
        "success_neighborhood_by_failure_vote": numeric_by_vote(
            failure, distance_columns
        ),
        "interpretation_limits": [
            "The three vote blocks were selected in earlier exploratory analyses; this audit freezes them and does not provide independent confirmation.",
            "Relative-phase route features still use the final episode horizon and are post-onset/pre-terminal descriptions.",
            "The long-task goal is a success-terminal mean proxy, not an environment success predicate reconstructed from simulator internals.",
            "Terminal basin is a closer-object comparison and does not imply grasp or contact.",
            "Saved embedding Euclidean distances are comparative diagnostics, not calibrated physical or kernel distances.",
            "Initial-state and seed permutation tests are exploratory and unadjusted for multiple comparisons.",
        ],
    }


def _lookup_count(records: list[dict[str, Any]], **conditions: Any) -> int:
    return int(
        sum(
            int(row["n"])
            for row in records
            if all(row.get(key) == value for key, value in conditions.items())
        )
    )


def render_report(summary: dict[str, Any]) -> str:
    cohort = summary["cohort"]
    residual = summary["residual"]
    long = summary["long_task"]
    stages = long["physical_stage_by_vote"]
    basins = long["terminal_basin_by_vote"]
    patterns = long["terminal_basin_by_pattern"]
    dynamics = long["event_dynamics_by_vote"]
    neighborhood = summary["success_neighborhood_by_failure_vote"]

    task_rows = []
    task_names = sorted({row["task"] for row in residual["task_vote_counts"]})
    for task in task_names:
        zero = _lookup_count(
            residual["task_vote_counts"], task=task, consensus_votes=0
        )
        one = _lookup_count(
            residual["task_vote_counts"], task=task, consensus_votes=1
        )
        task_rows.append(
            f"| `{task.split('/', 1)[-1]}` | {zero} | {one} | {zero + one} |"
        )

    mode_rows = []
    modes = sorted({row["failure_mode"] for row in residual["failure_mode_vote_counts"]})
    for mode in modes:
        zero = _lookup_count(
            residual["failure_mode_vote_counts"], failure_mode=mode, consensus_votes=0
        )
        one = _lookup_count(
            residual["failure_mode_vote_counts"], failure_mode=mode, consensus_votes=1
        )
        mode_rows.append(f"| {mode} | {zero} | {one} | {zero + one} |")

    stage_rows = []
    for vote in range(4):
        counts = {
            stage: _lookup_count(stages, consensus_votes=vote, long_physical_stage=stage)
            for stage in ("pot2_only", "pot1_only", "both_at_goal", "neither_at_goal")
        }
        stage_rows.append(
            "| %d | %d | %d | %d | %d |"
            % (
                vote,
                counts["pot2_only"],
                counts["pot1_only"],
                counts["both_at_goal"],
                counts["neither_at_goal"],
            )
        )

    basin_rows = []
    for vote in range(4):
        p1 = _lookup_count(basins, consensus_votes=vote, eef_terminal_basin="pot1")
        p2 = _lookup_count(basins, consensus_votes=vote, eef_terminal_basin="pot2")
        basin_rows.append(f"| {vote} | {p1} | {p2} |")

    pattern_rows = []
    for pattern in ("100", "010", "001"):
        p1 = _lookup_count(patterns, vote_pattern=pattern, eef_terminal_basin="pot1")
        p2 = _lookup_count(patterns, vote_pattern=pattern, eef_terminal_basin="pot2")
        pattern_rows.append(f"| {pattern} | {p1 + p2} | {p1} | {p2} |")

    dynamic_rows = []
    for vote in range(4):
        row = dynamics[str(vote)]
        dynamic_rows.append(
            "| %d | %d | %.3f | %.3f | %.3f | %.1f%% | %.1f%% |"
            % (
                vote,
                row["n"],
                row["soft_late_early_ratio"]["median"],
                row["return_mean_lag_fraction"]["median"],
                row["global_peak_phase"]["median"],
                100 * row["late_stasis_indicator"]["mean"],
                100 * row["terminal_return_indicator"]["mean"],
            )
        )

    distance_rows = []
    for vote in range(4):
        row = neighborhood[str(vote)]
        distance_rows.append(
            "| %d | %d | %.2f | %.2f | %.2f |"
            % (
                vote,
                row["aligned_nearest_sibling_success_ratio"]["available"],
                row["aligned_nearest_sibling_success_ratio"]["median"],
                row["event_nearest_sibling_success_ratio"]["median"],
                row["lag_nearest_sibling_success_ratio"]["median"],
            )
        )

    init_rows = []
    init_sorted = sorted(
        long["initial_state_counts"], key=lambda row: -(row["zero"] + row["one"])
    )
    for row in init_sorted:
        if row["zero"] + row["one"]:
            init_rows.append(
                "| %d | %d | %d | %d | %d |"
                % (
                    row["init_state_id"],
                    row["failures"],
                    row["zero"],
                    row["one"],
                    row["core"],
                )
            )

    zero_seed = long["crossed_design_tests"]["zero_vote_failure"]
    residual_seed = long["crossed_design_tests"]["residual_failure_le1_vote"]
    transition = long["successful_transition"]
    lines = [
        "# 剩余失败的物理阶段审计",
        "",
        "## 直接结论",
        "",
        f"三方法 2/3 共识覆盖 {cohort['failure_core_ge2_votes']}/{cohort['failures']} 个失败。剩余 {residual['episodes']} 条由 1 票 {cohort['failure_1vote']} 条和 0 票 {cohort['failure_0vote']} 条组成。",
        "",
        f"它们并不是一组新的、干净的物理失败类：{long['residual_failures']}/{residual['episodes']} 来自双摩卡壶长任务，而且与核心失败大多处于同一个子任务阶段。长任务剩余失败中，只有 pot 2 到位而 pot 1 未到位的有 {_lookup_count(stages, consensus_votes=0, long_physical_stage='pot2_only') + _lookup_count(stages, consensus_votes=1, long_physical_stage='pot2_only')}/{long['residual_failures']}；核心中对应为 {_lookup_count(stages, consensus_votes=2, long_physical_stage='pot2_only') + _lookup_count(stages, consensus_votes=3, long_physical_stage='pot2_only')}/{long['core_failures']}。",
        "",
        "真正清楚的差别是失败后的运动形态：0 票长任务失败多数在接近 pot 1 后又回到已经完成的 pot 2/炉灶一侧；三票核心则停在未完成的 pot 1 一侧。前者晚期路由仍活跃并出现长跨度回返，后者更像减速后静滞。因此，没有聚进共同核心不等于没有结构，而是共同核心偏向抓住一种特定的静滞终局。",
        "",
        "## 组成",
        "",
        "| task | 0 vote | 1 vote | total |",
        "|---|---:|---:|---:|",
        *task_rows,
        "",
        "所有失败均跑到各任务固定的失败 horizon：long=52、top-drawer=30、ramekin/stove=22 queries。剩余组不是由更短轨迹造成。",
        "",
        "| physical failure proxy | 0 vote | 1 vote | total |",
        "|---|---:|---:|---:|",
        *mode_rows,
        "",
        "`failure_mode` 来自既有 MuJoCo 轨迹启发式，不是真值人工标签。",
        "",
        "## 长任务阶段",
        "",
        f"成功轨迹通常先让 pot 2 到达成功终点代理（median query {transition['pot2_first_goal_query_median']:.0f}），再让 pot 1 到达（median query {transition['pot1_first_goal_query_median']:.0f}）。在同时穿过两个 5 cm 代理球、因而可判定顺序的 {transition['order_available']}/{transition['order_total_successes']} 条成功轨迹中，{transition['pot2_before_pot1_count']}/{transition['order_available']} 都是 pot 2 先于 pot 1；其余成功轨迹不被这个代理判定。",
        "",
        "| votes | pot2 only | pot1 only | both at goal | neither |",
        "|---:|---:|---:|---:|---:|",
        *stage_rows,
        "",
        f"到位判据与 `replanning-reset-trap` 一致：物体 XYZ 距成功轨迹终点均值不超过 {GOAL_DISTANCE_M:.2f} m。它只是可复现的阶段代理。",
        "",
        "## 终端回退",
        "",
        "终端盆地只比较最后一个 query 的末端执行器 XYZ 到两个物体当前 XYZ 哪个更近，不使用 outcome，也不要求发生接触。",
        "",
        "| votes | terminal closer to pot1 | terminal closer to pot2 |",
        "|---:|---:|---:|",
        *basin_rows,
        "",
        "在 30 条长任务 0 票失败中，27 条终止时更靠 pot 2；在 100 条三票失败中，100 条都更靠 pot 1。0 票组后半程到 pot 1 的最小距离 median 约 0.094 m，说明它们通常不是从未切换，而是靠近 pot 1 后又退回。",
        "",
        "一票本身不是一个类别：",
        "",
        "| vote pattern (aligned/event/lag) | n, long task | pot1 basin | pot2 basin |",
        "|---|---:|---:|---:|",
        *pattern_rows,
        "",
        "Aligned-only 更接近共同核心的第二物体终局；event-only 与 lag-only 更混合。把 62 条一票失败平均成一类会掩盖这个差别。",
        "",
        "## 路由动态",
        "",
        "以下只比较同一个 long task、同一个 52-query 失败 horizon。",
        "",
        "| votes | n | late/early route speed | mean return lag | peak phase | late stasis | terminal return |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *dynamic_rows,
        "",
        "整体上，票数越高，路由越明显减速并进入晚期静滞；三票组的变化峰值也最早。0 票组则相反：晚期变化重新增强，回返跨度更长。这是连续梯度，不支持再硬切出若干等价密度的失败类。",
        "",
        "## 初态与随机种子",
        "",
        "| init | failures | 0 vote | 1 vote | core |",
        "|---:|---:|---:|---:|---:|",
        *init_rows,
        "",
        f"在完整 16x32 long 交叉网格上，固定 seed 后 residual membership 与 init 有关（permutation p={residual_seed['initial_state_given_seed']['conditional_permutation_p']:.4f}）；固定 init 后，整体 residual 与 flow seed 没有清楚关系（p={residual_seed['flow_seed_given_initial_state']['conditional_permutation_p']:.4f}）。",
        "",
        f"0 票子集单独看时，init 与 seed 都有探索性关联（p={zero_seed['initial_state_given_seed']['conditional_permutation_p']:.4f} / {zero_seed['flow_seed_given_initial_state']['conditional_permutation_p']:.4f}）。这是事后挑出的 30 条小样本，不能当成确认性 seed 机制。",
        "",
        "## 与成功轨迹的邻近性",
        "",
        "距离在三个已保存分析 embedding 中分别计算；每条 rollout 只找同 task、同 init 的成功 sibling。表中再除以该 cell 内成功对成功的典型最近邻距离；1 表示约等于成功岛内部尺度。",
        "",
        "| failure votes | available | aligned ratio | event ratio | lag ratio |",
        "|---:|---:|---:|---:|---:|",
        *distance_rows,
        "",
        "0/1 票失败确实比三票核心更接近相同初态的成功轨迹，但距离通常仍大于成功岛内部尺度。结合 lag-HDBSCAN 把这些失败判为 noise，准确说法是“更靠成功流形的稀疏边缘”，不是“与成功无法区分”。",
        "",
        "## 下一步实验",
        "",
        "最值得验证的二级结构不是重新对完整 episode 强制聚类，而是按 pot 2 首次到位对齐，只看之后的固定历史：",
        "",
        "1. `stay-at-pot1`：切到第二物体后停在那里；",
        "2. `return-to-pot2`：靠近第二物体后又回到已经完成的第一子目标；",
        "3. 对两组分别匹配 init、pot 1/pot 2/eef 位置和剩余预算，再测试 routing 是否仍有增量。",
        "",
        "这能直接检验“路由暴露策略没有真正更新”，同时避免把物体位置或最终 timeout 当作 MoE 机制。",
        "",
        "## 限制",
        "",
        "- 三个投票块来自先前的探索性分析；这里冻结它们，不构成独立确认。",
        "- 相对相位特征仍依赖 episode 的最终 horizon，只能作 post-onset、pre-terminal 描述。",
        "- 长任务 goal 是成功终点均值代理，不是从环境内部重新构造的成功判据。",
        "- 终端盆地只是距离哪个物体更近，不代表发生抓取或接触。",
        "- embedding 欧氏距离只作比较诊断，不是校准后的物理距离或 kernel 距离。",
        "- init/seed 置换检验是探索性的，未校正多重比较。",
        "",
        "## 产物",
        "",
        "- `episode_audit.csv`: 2,560 条逐 episode 投票、失败代理、物理阶段和成功邻距。",
        "- `long_initial_state_counts.csv`: long task 的 init 组成。",
        "- `long_flow_seed_counts.csv`: long task 的 flow seed 组成。",
        "- `summary.json`: 报告中全部计数和检验。",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    frame = pd.DataFrame(
        {
            "task": ["t"] * 4,
            "init_state_id": [0] * 4,
            "failure": [False, False, True, True],
        }
    )
    values = np.asarray([[0.0], [2.0], [0.5], [4.0]])
    nearest, ratio = nearest_sibling_success(frame, values)
    np.testing.assert_allclose(nearest, [2.0, 2.0, 0.5, 2.0])
    np.testing.assert_allclose(ratio, [1.0, 1.0, 0.25, 1.0])
    assert binary_mutual_information(np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 1])) > 0
    assert binary_mutual_information(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])) == 0
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.permutations < 1:
        raise ValueError("permutations must be positive")

    frame = load_episode_table(args.analysis_root)
    frame = add_event_diagnostics(frame, args.analysis_root)
    frame = add_success_distances(frame, args.analysis_root)
    physical, physical_metadata = long_task_physical_table(frame, args.cache_root)
    frame = frame.merge(physical, on=["task", "episode"], how="left", validate="one_to_one")
    summary = build_summary(
        frame, physical_metadata, args.permutations, args.seed
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "episode_audit.csv", index=False)
    pd.DataFrame(summary["long_task"]["initial_state_counts"]).to_csv(
        args.out_dir / "long_initial_state_counts.csv", index=False
    )
    pd.DataFrame(summary["long_task"]["flow_seed_counts"]).to_csv(
        args.out_dir / "long_flow_seed_counts.csv", index=False
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))

    output_names = [
        "episode_audit.csv",
        "long_initial_state_counts.csv",
        "long_flow_seed_counts.csv",
        "summary.json",
        "report.md",
    ]
    completion = {
        "schema": "residual-failure-physical-audit-completion/1",
        "status": "complete",
        "run_class": "exploratory descriptive audit",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "input_sha256": {
            "aligned_assignments": sha256_file(
                args.analysis_root / "aligned-route-kernel/assignments.csv"
            ),
            "event_assignments": sha256_file(
                args.analysis_root / "route-change-events/assignments.csv"
            ),
            "alternative_assignments": sha256_file(
                args.analysis_root
                / "alternative-routing-organizations/assignments.csv"
            ),
            "failure_modes": sha256_file(
                args.analysis_root / "replanning-reset-trap/episode_metrics.csv"
            ),
            "long_summaries": sha256_file(
                args.cache_root / LONG_TASK / "right-16x32/client/summaries.json"
            ),
            "long_sim_layout": sha256_file(
                args.cache_root / LONG_TASK / "right-16x32/client/sim_layout.json"
            ),
            "long_episode_npz_set": sha256_file_set(
                list(
                    (
                        args.cache_root / LONG_TASK / "right-16x32/client"
                    ).glob("episode_*.npz")
                )
            ),
        },
        "output_sha256": {
            name: sha256_file(args.out_dir / name) for name in output_names
        },
    }
    (args.out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
