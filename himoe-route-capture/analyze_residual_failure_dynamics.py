#!/usr/bin/env python3
"""Describe routing dynamics outside the cross-method failure consensus core."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_replanning_reset_trap import GOAL_M, LIFT_M, identify_targets
from analyze_route_change_events import extract_event_features


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
EVENT_DIR = ANALYSIS / "route-change-events"
ALIGNED_DIR = ANALYSIS / "aligned-route-kernel"
ALTERNATIVE_DIR = ANALYSIS / "alternative-routing-organizations"
OUT_DIR = ANALYSIS / "residual-failure-dynamics"
FEATURE_CACHE = ANALYSIS / "all-outcome-routing-clusters/feature_cache"
FAILURE_MODE_CSV = ANALYSIS / "replanning-reset-trap/episode_metrics.csv"
LONG_CLIENT = (
    HERE.parent
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
    / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32/client"
)

KEY_COLUMNS = [
    "task",
    "episode",
    "init_state_id",
    "flow_noise_seed",
    "episode_length",
]
GROUP_ORDER = ["success", "failure_core", "failure_1vote", "failure_0vote"]
GROUP_LABELS = {
    "success": "success",
    "failure_core": "failure core (2-3 votes)",
    "failure_1vote": "failure, 1 vote",
    "failure_0vote": "failure, 0 votes",
}
COLORS = {
    "success": "#6B7280",
    "failure_core": "#C43C39",
    "failure_1vote": "#D08B17",
    "failure_0vote": "#188977",
}


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def validate_alignment(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    if not left[KEY_COLUMNS].equals(right[KEY_COLUMNS]):
        raise ValueError(f"row alignment failed for {name}")


def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    event = pd.read_csv(EVENT_DIR / "assignments.csv")
    aligned = pd.read_csv(ALIGNED_DIR / "assignments.csv")
    alternative = pd.read_csv(ALTERNATIVE_DIR / "assignments.csv")
    validate_alignment(event, aligned, "aligned kernel")
    validate_alignment(event, alternative, "alternative organizations")

    arrays_file = np.load(EVENT_DIR / "event_features.npz")
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    if len(event) != len(arrays["core"]):
        raise ValueError("event feature row count does not match assignments")

    event["aligned_vote"] = aligned["raw_cluster"].eq(1)
    event["event_vote"] = event["event_cluster"].eq(0)
    event["lag_vote"] = alternative["lag_spectrum_louvain"].eq("C7")
    event["consensus_votes"] = event[
        ["aligned_vote", "event_vote", "lag_vote"]
    ].sum(axis=1)
    event["vote_pattern"] = (
        event[["aligned_vote", "event_vote", "lag_vote"]]
        .astype(np.int8)
        .astype(str)
        .agg("".join, axis=1)
    )
    event["analysis_group"] = np.select(
        [
            ~event["failure"],
            event["failure"] & event["consensus_votes"].ge(2),
            event["failure"] & event["consensus_votes"].eq(1),
        ],
        ["success", "failure_core", "failure_1vote"],
        default="failure_0vote",
    )

    feature_frame = pd.DataFrame(arrays["core"], columns=arrays["core_names"])
    for column in feature_frame:
        event[column] = feature_frame[column].to_numpy()

    soft_speed = arrays["soft_speed"].astype(np.float64)
    speed_scale = np.maximum(np.median(soft_speed, axis=1, keepdims=True), 1e-8)
    arrays["soft_speed_normalized"] = soft_speed / speed_scale
    event["early_low_speed_fraction"] = arrays["low_speed_event"][:, 1:4].mean(axis=1)
    event["late_low_speed_fraction_3"] = arrays["low_speed_event"][:, -3:].mean(axis=1)
    event["terminal_return_rate"] = arrays["return_event"][:, -1].astype(np.float64)

    template_distance = np.full(len(event), np.nan, dtype=np.float64)
    for _, indices in event.groupby("task", sort=False).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        success_index = index[~event.loc[index, "failure"].to_numpy()]
        if not len(success_index):
            continue
        template = np.median(arrays["soft_speed_normalized"][success_index], axis=0)
        template_distance[index] = np.sqrt(
            np.mean(
                (arrays["soft_speed_normalized"][index] - template[None, :]) ** 2,
                axis=1,
            )
        )
    event["task_success_speed_distance"] = template_distance
    return event, arrays


def load_truncate90_events() -> dict[str, np.ndarray]:
    manifest = json.loads((FEATURE_CACHE / "manifest.json").read_text())
    item = manifest["arrays"]["feature_truncate90_recurrence"]
    recurrence = np.load(FEATURE_CACHE / item["file"], mmap_mode="r")
    if list(recurrence.shape) != item["shape"] or str(recurrence.dtype) != item["dtype"]:
        raise ValueError("truncate90 recurrence metadata mismatch")
    core, names, absolute, absolute_names, tapes = extract_event_features(recurrence)
    result = {
        "core": core,
        "core_names": np.asarray(names),
        "absolute_scale": absolute,
        "absolute_scale_names": np.asarray(absolute_names),
        **tapes,
    }
    speed = result["soft_speed"].astype(np.float64)
    result["soft_speed_normalized"] = speed / np.maximum(
        np.median(speed, axis=1, keepdims=True), 1e-8
    )
    return result


def event_feature_frame(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(arrays["core"], columns=arrays["core_names"])


def exact_vote_axis(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    long_failure_mask: np.ndarray,
    phase_window: tuple[float, float],
) -> dict[str, object]:
    features = event_feature_frame(arrays)
    votes = frame.loc[long_failure_mask, "consensus_votes"].to_numpy()
    selected_features = features.loc[long_failure_mask]
    speed = arrays["soft_speed_normalized"][long_failure_mask]
    metrics = [
        "soft_late_early_ratio",
        "global_peak_phase",
        "late_stasis_indicator",
        "soft_speed_cv",
        "return_count_fraction",
        "layer_peak_within1_fraction",
        "change_max_contrast",
    ]
    result: dict[str, object] = {"phase_window": phase_window, "by_vote": {}}
    for vote in range(4):
        mask = votes == vote
        means = selected_features.loc[mask, metrics].mean().to_dict()
        means["global_peak_episode_phase"] = phase_window[0] + (
            phase_window[1] - phase_window[0]
        ) * means["global_peak_phase"]
        result["by_vote"][str(vote)] = {
            "n": int(mask.sum()),
            "mean": means,
            "median": selected_features.loc[mask, metrics].median().to_dict(),
            "q25": selected_features.loc[mask, metrics].quantile(0.25).to_dict(),
            "q75": selected_features.loc[mask, metrics].quantile(0.75).to_dict(),
            "speed_mean": speed[mask].mean(axis=0),
        }
    trends = {}
    for metric in metrics:
        statistic = spearmanr(votes, selected_features[metric]).statistic
        trends[metric] = float(statistic)
    result["spearman_vote_trends"] = trends
    return result


def nonzero_sign_flip_indices(values: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(values, dtype=np.float64))
    nonzero = np.flatnonzero(signs != 0)
    if len(nonzero) < 2:
        return np.empty(0, dtype=np.int64)
    return nonzero[1:][signs[nonzero[1:]] != signs[nonzero[:-1]]]


PHYSICAL_COLUMNS = [
    "eef_path_m",
    "eef_path_late_half_m",
    "gripper_sign_flips",
    "gripper_sign_flips_late_half",
    "targets_lifted",
    "targets_placed",
    "targets_reached",
    "lift_onsets",
]


def attach_long_task_physics(
    frame: pd.DataFrame, long_failure_mask: np.ndarray
) -> tuple[pd.DataFrame, list[str]]:
    summaries = sorted(
        json.loads((LONG_CLIENT / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    layout = json.loads((LONG_CLIENT / "sim_layout.json").read_text())
    sims = []
    for episode in range(len(summaries)):
        with np.load(LONG_CLIENT / f"episode_{episode:02d}.npz", allow_pickle=False) as data:
            sims.append(np.asarray(data["sim_state"], dtype=np.float32))
    targets = identify_targets(summaries, layout, sims)
    target_names = [str(target["name"]) for target in targets]
    if target_names != ["moka_pot_1_joint0", "moka_pot_2_joint0"]:
        raise ValueError(f"unexpected long-task targets: {target_names}")

    modes = pd.read_csv(FAILURE_MODE_CSV)
    modes = modes[modes["task"].str.contains("moka_pots", regex=False)]
    if modes.duplicated(["task", "episode"]).any():
        raise ValueError("duplicate long-task failure-mode rows")
    mode_by_episode = modes.set_index("episode")["failure_mode"].to_dict()

    for column in PHYSICAL_COLUMNS:
        frame[column] = np.nan
    frame["failure_mode_proxy"] = "not_applicable"
    for index in np.flatnonzero(long_failure_mask):
        episode = int(frame.at[index, "episode"])
        with np.load(LONG_CLIENT / f"episode_{episode:02d}.npz", allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float64)
            actions = np.asarray(data["actions"], dtype=np.float64)
            sim = np.asarray(data["sim_state"], dtype=np.float64)
        if state.shape != (52, 8) or actions.shape != (52, 10, 7):
            raise ValueError(f"unexpected long failure tape shape for episode {episode}")

        eef_steps = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1)
        gripper_command = actions[:, :, 6].mean(axis=1)
        flip_indices = nonzero_sign_flip_indices(gripper_command)
        lifted = placed = reached = lift_onsets = 0
        for target in targets:
            position = sim[:, int(target["lo"]) : int(target["hi"])]
            is_lifted = position[:, 2] - position[0, 2] > LIFT_M
            lifted += int(np.any(is_lifted))
            lift_onsets += int(np.sum(is_lifted & ~np.r_[False, is_lifted[:-1]]))
            distance = np.linalg.norm(position - np.asarray(target["goal"]), axis=1)
            placed += int(distance[-1] <= GOAL_M)
            reached += int(np.min(distance) <= GOAL_M)

        values = {
            "eef_path_m": float(eef_steps.sum()),
            "eef_path_late_half_m": float(eef_steps[len(eef_steps) // 2 :].sum()),
            "gripper_sign_flips": float(len(flip_indices)),
            "gripper_sign_flips_late_half": float(
                np.sum(flip_indices >= len(gripper_command) // 2)
            ),
            "targets_lifted": float(lifted),
            "targets_placed": float(placed),
            "targets_reached": float(reached),
            "lift_onsets": float(lift_onsets),
        }
        for column, value in values.items():
            frame.at[index, column] = value
        frame.at[index, "failure_mode_proxy"] = str(mode_by_episode[episode])
    return frame, target_names


def bootstrap_matched_initial_state_delta(
    frame: pd.DataFrame,
    left: str,
    right: str,
    columns: list[str],
    repeats: int = 5_000,
) -> dict[str, object]:
    contrasts = []
    states = []
    for initial_state, subset in frame.groupby("init_state_id"):
        left_rows = subset[subset["physical_group"].eq(left)]
        right_rows = subset[subset["physical_group"].eq(right)]
        if left_rows.empty or right_rows.empty:
            continue
        contrasts.append(
            left_rows[columns].mean().to_numpy() - right_rows[columns].mean().to_numpy()
        )
        states.append(int(initial_state))
    values = np.asarray(contrasts, dtype=np.float64)
    rng = np.random.default_rng(8181)
    bootstrap = np.empty((repeats, len(columns)), dtype=np.float64)
    for repeat in range(repeats):
        selected = rng.integers(0, len(values), size=len(values))
        bootstrap[repeat] = values[selected].mean(axis=0)
    return {
        "left_minus_right": f"{left} - {right}",
        "matched_initial_states": states,
        "n_initial_states": len(states),
        "mean": dict(zip(columns, values.mean(axis=0))),
        "ci95_low": dict(zip(columns, np.quantile(bootstrap, 0.025, axis=0))),
        "ci95_high": dict(zip(columns, np.quantile(bootstrap, 0.975, axis=0))),
    }


def physical_validation(
    frame: pd.DataFrame, long_failure_mask: np.ndarray, target_names: list[str]
) -> dict[str, object]:
    failure = frame.loc[long_failure_mask].copy()
    failure["physical_group"] = np.where(
        failure["consensus_votes"].lt(2), "tail_0_1", "core_2_3"
    )
    by_vote = {}
    for vote, subset in failure.groupby("consensus_votes"):
        by_vote[str(int(vote))] = {
            "n": len(subset),
            "failure_mode_counts": subset["failure_mode_proxy"].value_counts().to_dict(),
            "mean": subset[PHYSICAL_COLUMNS].mean().to_dict(),
            "median": subset[PHYSICAL_COLUMNS].median().to_dict(),
        }

    partial = failure[failure["failure_mode_proxy"].eq("partial")].copy()
    partial_by_vote = {}
    for vote, subset in partial.groupby("consensus_votes"):
        partial_by_vote[str(int(vote))] = {
            "n": len(subset),
            "mean": subset[PHYSICAL_COLUMNS].mean().to_dict(),
            "median": subset[PHYSICAL_COLUMNS].median().to_dict(),
        }
    partial_group = {
        group: {
            "n": len(subset),
            "mean": subset[PHYSICAL_COLUMNS].mean().to_dict(),
            "median": subset[PHYSICAL_COLUMNS].median().to_dict(),
        }
        for group, subset in partial.groupby("physical_group")
    }
    trend_columns = [
        "eef_path_m",
        "eef_path_late_half_m",
        "gripper_sign_flips",
        "gripper_sign_flips_late_half",
    ]
    trends = {
        column: float(spearmanr(partial["consensus_votes"], partial[column]).statistic)
        for column in trend_columns
    }
    matched = bootstrap_matched_initial_state_delta(
        partial, "tail_0_1", "core_2_3", trend_columns
    )
    return {
        "source": str(LONG_CLIENT),
        "grouping_features_used": False,
        "target_names": target_names,
        "all_failure_by_vote": by_vote,
        "failure_mode_table": pd.crosstab(
            failure["consensus_votes"], failure["failure_mode_proxy"]
        ).to_dict(),
        "partial_definition": "one target placed at terminal; the other incomplete",
        "partial_n": len(partial),
        "partial_by_vote": partial_by_vote,
        "partial_tail_vs_core": partial_group,
        "partial_spearman_vote_trends": trends,
        "partial_matched_initial_state_delta": matched,
    }


def group_counts(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in GROUP_ORDER:
        subset = frame[frame["analysis_group"].eq(group)]
        result[group] = {
            "n": len(subset),
            "tasks": subset["task"].value_counts().sort_index().to_dict(),
            "episode_length_mean": subset["episode_length"].mean(),
            "episode_length_median": subset["episode_length"].median(),
        }
    return result


def bootstrap_curve_by_initial_state(
    values: np.ndarray,
    initial_state: np.ndarray,
    rng: np.random.Generator,
    repeats: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(initial_state)
    samples = np.empty((repeats, values.shape[1]), dtype=np.float64)
    blocks = {state: values[initial_state == state] for state in unique}
    for repeat in range(repeats):
        selected = rng.choice(unique, size=len(unique), replace=True)
        samples[repeat] = np.concatenate([blocks[state] for state in selected]).mean(axis=0)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def curve_profiles(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(20260828)
    result: dict[str, object] = {}
    tapes = {
        "speed": arrays["soft_speed_normalized"],
        "return": arrays["return_event"].astype(np.float64),
        "joint_return": arrays["joint_return_event"].astype(np.float64),
        "low_speed": arrays["low_speed_event"].astype(np.float64),
    }
    for group in GROUP_ORDER:
        selected = mask & frame["analysis_group"].eq(group).to_numpy()
        group_result: dict[str, object] = {"n": int(selected.sum())}
        for name, values in tapes.items():
            selected_values = values[selected]
            initial_state = frame.loc[selected, "init_state_id"].to_numpy()
            low, high = bootstrap_curve_by_initial_state(
                selected_values, initial_state, rng
            )
            group_result[name] = {
                "mean": selected_values.mean(axis=0),
                "ci95_low": low,
                "ci95_high": high,
            }
        result[group] = group_result
    return result


SCALAR_COLUMNS = [
    "soft_late_early_ratio",
    "return_count_fraction",
    "joint_return_count_fraction",
    "terminal_return_rate",
    "early_low_speed_fraction",
    "late_low_speed_fraction_3",
    "layer_peak_within1_fraction",
    "layer_speed_curve_correlation",
    "change_max_contrast",
    "task_success_speed_distance",
]


def scalar_profiles(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in GROUP_ORDER:
        subset = frame.loc[mask & frame["analysis_group"].eq(group), SCALAR_COLUMNS]
        result[group] = {
            "n": len(subset),
            "mean": subset.mean().to_dict(),
            "median": subset.median().to_dict(),
        }
    return result


MATCHED_COLUMNS = [
    "soft_late_early_ratio",
    "return_count_fraction",
    "terminal_return_rate",
    "early_low_speed_fraction",
    "late_low_speed_fraction_3",
    "layer_peak_within1_fraction",
    "change_max_contrast",
]


def matched_initial_state_contrasts(
    frame: pd.DataFrame,
    long_mask: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(314159)
    result: dict[str, object] = {}
    long_frame = frame.loc[long_mask]
    for group in GROUP_ORDER[1:]:
        rows: list[np.ndarray] = []
        states: list[int] = []
        for initial_state, subset in long_frame.groupby("init_state_id"):
            group_rows = subset[subset["analysis_group"].eq(group)]
            success_rows = subset[subset["analysis_group"].eq("success")]
            if group_rows.empty or success_rows.empty:
                continue
            rows.append(
                group_rows[MATCHED_COLUMNS].mean().to_numpy()
                - success_rows[MATCHED_COLUMNS].mean().to_numpy()
            )
            states.append(int(initial_state))
        contrasts = np.asarray(rows, dtype=np.float64)
        bootstrap = np.empty((5_000, len(MATCHED_COLUMNS)), dtype=np.float64)
        for repeat in range(len(bootstrap)):
            selected = rng.integers(0, len(contrasts), size=len(contrasts))
            bootstrap[repeat] = contrasts[selected].mean(axis=0)
        result[group] = {
            "matched_initial_states": states,
            "n_initial_states": len(states),
            "mean_delta_from_success": dict(zip(MATCHED_COLUMNS, contrasts.mean(axis=0))),
            "ci95_low": dict(
                zip(MATCHED_COLUMNS, np.quantile(bootstrap, 0.025, axis=0))
            ),
            "ci95_high": dict(
                zip(MATCHED_COLUMNS, np.quantile(bootstrap, 0.975, axis=0))
            ),
        }
    return result


def pattern_profiles(frame: pd.DataFrame, long_mask: np.ndarray) -> dict[str, object]:
    failure = frame.loc[long_mask & frame["failure"]]
    columns = [
        "soft_late_early_ratio",
        "return_count_fraction",
        "terminal_return_rate",
        "early_low_speed_fraction",
        "late_low_speed_fraction_3",
        "layer_peak_within1_fraction",
        "change_max_contrast",
    ]
    result: dict[str, object] = {}
    for pattern, subset in failure.groupby("vote_pattern"):
        result[pattern] = {
            "n": len(subset),
            "event_cluster_counts": subset["event_cluster"].value_counts().to_dict(),
            "mean": subset[columns].mean().to_dict(),
        }
    return result


def zero_vote_event_profiles(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    long_mask: np.ndarray,
) -> dict[str, object]:
    selected = long_mask & frame["failure"].to_numpy() & frame["consensus_votes"].eq(0).to_numpy()
    result: dict[str, object] = {}
    for cluster in sorted(frame.loc[selected, "event_cluster"].unique()):
        mask = selected & frame["event_cluster"].eq(cluster).to_numpy()
        result[f"event_C{int(cluster)}"] = {
            "n": int(mask.sum()),
            "speed_mean": arrays["soft_speed_normalized"][mask].mean(axis=0),
            "return_probability": arrays["return_event"][mask].mean(axis=0),
            "low_speed_probability": arrays["low_speed_event"][mask].mean(axis=0),
        }
    return result


def make_plot(
    curves: dict[str, object],
    scalars: dict[str, object],
    output: Path,
) -> None:
    speed_phase = np.linspace(0.0, 1.0, 9)
    return_phase = np.linspace(2 / 9, 1.0, 8)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    for group in GROUP_ORDER:
        profile = curves[group]
        color = COLORS[group]
        label = GROUP_LABELS[group]
        speed = profile["speed"]
        axes[0, 0].plot(speed_phase, speed["mean"], color=color, lw=2.2, label=label)
        axes[0, 0].fill_between(
            speed_phase,
            speed["ci95_low"],
            speed["ci95_high"],
            color=color,
            alpha=0.13,
        )
        axes[0, 1].plot(
            return_phase,
            profile["return"]["mean"],
            color=color,
            lw=2.2,
            label=label,
        )
        axes[1, 0].plot(
            speed_phase,
            profile["low_speed"]["mean"],
            color=color,
            lw=2.2,
            label=label,
        )

    axes[0, 0].set_title("Route speed shape (median-normalized)")
    axes[0, 0].set_ylabel("adjacent-phase route distance")
    axes[0, 1].set_title("Nonlocal return probability")
    axes[0, 1].set_ylabel("fraction of episodes")
    axes[1, 0].set_title("Low-speed probability")
    axes[1, 0].set_ylabel("fraction of episodes")
    for axis in axes.flat[:3]:
        axis.set_xlabel("normalized rollout phase")
        axis.grid(alpha=0.2)
        axis.set_xlim(0.0, 1.0)

    bar_columns = [
        "soft_late_early_ratio",
        "return_count_fraction",
        "terminal_return_rate",
        "layer_peak_within1_fraction",
        "change_max_contrast",
    ]
    bar_labels = ["late/early\nspeed", "return\nfrequency", "terminal\nreturn", "layer\npeak sync", "change\ncontrast"]
    x = np.arange(len(bar_columns), dtype=np.float64)
    width = 0.19
    for offset, group in enumerate(GROUP_ORDER):
        values = [scalars[group]["mean"][column] for column in bar_columns]
        axes[1, 1].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            color=COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[1, 1].set_title("Long-task scalar dynamics")
    axes[1, 1].set_xticks(x, bar_labels)
    axes[1, 1].grid(axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Routing dynamics in the long moka-pot task", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_vote_axis_plot(
    full_axis: dict[str, object],
    truncate_axis: dict[str, object],
    physical: dict[str, object],
    output: Path,
) -> None:
    votes = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    windows = [("full", full_axis, "#C43C39"), ("truncate90", truncate_axis, "#188977")]
    metrics = [
        ("soft_late_early_ratio", "Late / early route speed"),
        ("global_peak_episode_phase", "Episode phase of route-speed peak"),
        ("late_stasis_indicator", "Late-stasis fraction"),
    ]
    for axis, (metric, title) in zip(axes.flat[:3], metrics):
        for label, result, color in windows:
            values = [result["by_vote"][str(vote)]["mean"][metric] for vote in votes]
            axis.plot(votes, values, marker="o", lw=2.3, color=color, label=label)
        axis.set_title(title)
        axis.set_xlabel("number of consensus votes")
        axis.set_xticks(votes)
        axis.grid(alpha=0.2)

    partial = physical["partial_by_vote"]
    late_path = [partial[str(vote)]["mean"]["eef_path_late_half_m"] for vote in votes]
    late_flips = [
        partial[str(vote)]["mean"]["gripper_sign_flips_late_half"] for vote in votes
    ]
    physical_axis = axes[1, 1]
    flip_axis = physical_axis.twinx()
    path_line = physical_axis.plot(
        votes, late_path, marker="o", lw=2.3, color="#2E6E9E", label="EEF path"
    )
    flip_line = flip_axis.plot(
        votes,
        late_flips,
        marker="s",
        lw=2.3,
        color="#D08B17",
        label="gripper flips",
    )
    physical_axis.set_title("Late-half physical activity (partial failures)")
    physical_axis.set_xlabel("number of consensus votes")
    physical_axis.set_ylabel("EEF path (m)", color="#2E6E9E")
    flip_axis.set_ylabel("gripper sign flips", color="#D08B17")
    physical_axis.set_xticks(votes)
    physical_axis.grid(alpha=0.2)
    physical_axis.legend(path_line + flip_line, ["EEF path", "gripper flips"], frameon=False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Long-task failures form an activity-to-stasis axis", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def report_text(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    long_counts = summary["long_task_counts"]
    scalar = summary["long_task_scalar_profiles"]
    matched = summary["long_task_matched_initial_state_contrasts"]
    patterns = summary["long_task_vote_pattern_profiles"]
    zero_profiles = summary["long_task_zero_vote_event_profiles"]
    full_axis = summary["long_task_exact_vote_axis"]["full"]
    truncate_axis = summary["long_task_exact_vote_axis"]["truncate90"]
    physical = summary["long_task_physical_validation"]
    residual_total = counts["failure_1vote"]["n"] + counts["failure_0vote"]["n"]
    residual_long = long_counts["failure_1vote"]["n"] + long_counts["failure_0vote"]["n"]

    lines = [
        "# Residual failure routing dynamics",
        "",
        "## Direct answer",
        "",
        f"The consensus core contains {counts['failure_core']['n']} of 307 failures. "
        f"The residual contains {residual_total}: {counts['failure_1vote']['n']} with "
        f"one method vote and {counts['failure_0vote']['n']} with zero votes. "
        f"{residual_long}/{residual_total} residual failures come from the long moka-pot task.",
        "",
        "Within that task, the residual failures are not routing-normal successes:",
        "",
        "- The strongest organization is a continuous activity-to-stasis axis, not another set of discrete failure classes.",
        "- At 0 votes routing remains active or restarts late; at 3 votes its largest change happens early and it then slows strongly. Votes 1 and 2 mostly lie between them.",
        "- Independent robot-state tapes agree: among failures at the same partial-completion stage, low-vote rollouts move the EEF farther and reverse the gripper more often in the second half.",
        "",
        "The remaining failures are therefore best read as active retrying or differently timed variants along the same broad noncompletion process. Small subgroups remain candidates, not validated new failure types.",
        "",
        "## Composition",
        "",
        "| group | all n | long moka-pot n | mean queries |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        lines.append(
            f"| {GROUP_LABELS[group]} | {counts[group]['n']} | "
            f"{long_counts[group]['n']} | {fmt(counts[group]['episode_length_mean'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Long-task dynamics",
            "",
            "All phases are relative phases. Speed is divided by each rollout's median speed; no absolute control step is used.",
            "",
            "| group | n | late/early speed | return frequency | terminal return | early-mid low speed | late low speed | layer peak sync | change contrast | distance to success speed template |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUP_ORDER:
        mean = scalar[group]["mean"]
        lines.append(
            f"| {GROUP_LABELS[group]} | {scalar[group]['n']} | "
            f"{fmt(mean['soft_late_early_ratio'])} | "
            f"{fmt(mean['return_count_fraction'])} | "
            f"{fmt(mean['terminal_return_rate'])} | "
            f"{fmt(mean['early_low_speed_fraction'])} | "
            f"{fmt(mean['late_low_speed_fraction_3'])} | "
            f"{fmt(mean['layer_peak_within1_fraction'])} | "
            f"{fmt(mean['change_max_contrast'])} | "
            f"{fmt(mean['task_success_speed_distance'])} |"
        )

    lines.extend(
        [
            "",
            "The speed-template distance gives a useful gradient: success is compact, the consensus core is far away, and the one-/zero-vote failures are intermediate. They are therefore weaker or differently timed deviations, not simply missed copies of the core.",
            "",
            "## Initial-state-matched check",
            "",
            "For every initial state containing both successes and the target failure group, the table subtracts the success mean first, then averages states equally.",
            "",
            "| failure group | matched states | delta late/early speed | delta return frequency | delta terminal return | delta early-mid low speed | delta late low speed | delta layer peak sync | delta change contrast |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUP_ORDER[1:]:
        row = matched[group]
        mean = row["mean_delta_from_success"]
        lines.append(
            f"| {GROUP_LABELS[group]} | {row['n_initial_states']} | "
            f"{fmt(mean['soft_late_early_ratio'])} | "
            f"{fmt(mean['return_count_fraction'])} | "
            f"{fmt(mean['terminal_return_rate'])} | "
            f"{fmt(mean['early_low_speed_fraction'])} | "
            f"{fmt(mean['late_low_speed_fraction_3'])} | "
            f"{fmt(mean['layer_peak_within1_fraction'])} | "
            f"{fmt(mean['change_max_contrast'])} |"
        )

    lines.extend(
        [
            "",
            "The zero-vote terminal-return excess is especially large after this control. The pause/restart observation is therefore not only due to difficult initial states being overrepresented.",
            "",
            "## Exact-vote axis at identical length",
            "",
            "All 216 failed moka-pot rollouts contain exactly 52 queries. The same full-window vote membership is frozen below; `truncate90` recomputes route dynamics on phase 0.50-0.90 and excludes phase 0.90-1.00.",
            "",
            "| votes | n | full late/early | trunc late/early | full peak phase | trunc peak phase | full late stasis | trunc late stasis |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for vote in range(4):
        full = full_axis["by_vote"][str(vote)]
        trunc = truncate_axis["by_vote"][str(vote)]
        lines.append(
            f"| {vote} | {full['n']} | "
            f"{fmt(full['mean']['soft_late_early_ratio'])} | "
            f"{fmt(trunc['mean']['soft_late_early_ratio'])} | "
            f"{fmt(full['mean']['global_peak_episode_phase'])} | "
            f"{fmt(trunc['mean']['global_peak_episode_phase'])} | "
            f"{fmt(full['mean']['late_stasis_indicator'])} | "
            f"{fmt(trunc['mean']['late_stasis_indicator'])} |"
        )
    lines.extend(
        [
            "",
            "| ordinal trend | full Spearman rho | truncate90 Spearman rho |",
            "|---|---:|---:|",
            f"| late/early speed | {fmt(full_axis['spearman_vote_trends']['soft_late_early_ratio'])} | {fmt(truncate_axis['spearman_vote_trends']['soft_late_early_ratio'])} |",
            f"| route-speed peak phase | {fmt(full_axis['spearman_vote_trends']['global_peak_phase'])} | {fmt(truncate_axis['spearman_vote_trends']['global_peak_phase'])} |",
            f"| late stasis | {fmt(full_axis['spearman_vote_trends']['late_stasis_indicator'])} | {fmt(truncate_axis['spearman_vote_trends']['late_stasis_indicator'])} |",
            "",
            "The ordering is strong but not perfectly stepwise in every full-window group mean. After terminal truncation, late/early speed and peak timing become strictly ordered across 0/1/2/3 votes. This is more consistent with one severity/timing axis than four discrete mechanisms.",
            "",
            "## Independent physical check",
            "",
            "The physical measurements below are read from client `state`, `actions`, and `sim_state`; none participates in the routing vote or grouping. EEF path uses query-boundary positions. A gripper flip is a sign change in the mean command of consecutive action chunks.",
            "",
            "The terminal proxy labels 198/216 failures as `partial`: exactly one of the two moka pots is placed and the other is incomplete. This supplies a same-task, same-length, same-coarse-stage comparison.",
            "",
            "| votes | partial n | EEF path total (m) | EEF path late half (m) | gripper flips | late-half flips | targets lifted | targets placed |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for vote in range(4):
        row = physical["partial_by_vote"][str(vote)]
        mean = row["mean"]
        lines.append(
            f"| {vote} | {row['n']} | {fmt(mean['eef_path_m'])} | "
            f"{fmt(mean['eef_path_late_half_m'])} | "
            f"{fmt(mean['gripper_sign_flips'])} | "
            f"{fmt(mean['gripper_sign_flips_late_half'])} | "
            f"{fmt(mean['targets_lifted'])} | {fmt(mean['targets_placed'])} |"
        )
    tail = physical["partial_tail_vs_core"]["tail_0_1"]
    core = physical["partial_tail_vs_core"]["core_2_3"]
    physical_match = physical["partial_matched_initial_state_delta"]
    lines.extend(
        [
            "",
            f"Pooling 0/1 votes, the late-half EEF path is {fmt(tail['mean']['eef_path_late_half_m'])} m and late gripper flips are {fmt(tail['mean']['gripper_sign_flips_late_half'])}; for 2/3 votes they are {fmt(core['mean']['eef_path_late_half_m'])} m and {fmt(core['mean']['gripper_sign_flips_late_half'])}.",
            "",
            f"After subtracting 2/3-vote rollouts within each of {physical_match['n_initial_states']} shared initial states, the 0/1-vote excess remains {fmt(physical_match['mean']['eef_path_late_half_m'])} m of late EEF travel and {fmt(physical_match['mean']['gripper_sign_flips_late_half'])} late gripper flips.",
            "",
            "This supports the plain-language interpretation: the residual tail is often still moving and reopening/reclosing, while the high-consensus core has largely stopped changing physically. It does not show that every retry is useful or that routing causes the retry.",
            "",
            "## Why one vote is not one class",
            "",
            "Vote bit order is aligned-kernel / event / lag-spectrum.",
            "",
            "| pattern | interpretation | n | late/early speed | return frequency | terminal return | early-mid low speed | layer sync | change contrast |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    descriptions = {
        "100": "aligned only; mostly flat-speed recurrence",
        "010": "event only; terminal slowdown",
        "001": "lag only; middle pause then reactivation",
    }
    for pattern in ["100", "010", "001"]:
        row = patterns[pattern]
        mean = row["mean"]
        lines.append(
            f"| {pattern} | {descriptions[pattern]} | {row['n']} | "
            f"{fmt(mean['soft_late_early_ratio'])} | "
            f"{fmt(mean['return_count_fraction'])} | "
            f"{fmt(mean['terminal_return_rate'])} | "
            f"{fmt(mean['early_low_speed_fraction'])} | "
            f"{fmt(mean['layer_peak_within1_fraction'])} | "
            f"{fmt(mean['change_max_contrast'])} |"
        )

    lines.extend(
        [
            "",
            "The 59 long-task one-vote failures split into 42/9/8 across these three signatures. Averaging them into one curve hides that structure.",
            "",
            "## Zero-vote candidates",
            "",
            "The 30 long-task zero-vote failures are already split by the independent event partition into "
            + ", ".join(
                f"{name.replace('event_', '')} n={row['n']}" for name, row in zero_profiles.items()
            )
            + ".",
            "",
            "- Event C2 is the clearest pause/restart candidate: the normalized speed trough is in the early-middle phase, followed by late reactivation.",
            "- Event C3 has repeated nonlocal returns, especially near the end, without a sustained terminal slowdown.",
            "- Event C1 is flatter and has no single dominant route event.",
            "",
            "These are hypotheses for targeted replay or physical-event labeling. With n=15/9/6 they should not yet be presented as stable failure types.",
            "",
            "## Limits",
            "",
            "- Consensus membership and these summaries come from the same routing data, so the analysis is descriptive rather than an independent validation.",
            "- Relative phase removes rollout-length scale but still uses the final episode horizon; it does not establish prediction before the first physical error.",
            "- The residual is dominated by one task. Cross-task claims for its candidate subtypes are not supported.",
            "- A nonlocal route return means an older routing state is closer than the immediately previous state. It does not by itself prove that the robot repeated the same physical action.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`: counts, curves, scalar profiles, matched-state contrasts, and subgroup profiles.",
            "- `assignments.csv`: episode-level consensus membership, selected dynamics, and independent physical measurements for long-task failures.",
            "- `long_task_dynamics.png`: speed, return, low-speed, and scalar comparisons.",
            "- `vote_axis_and_physics.png`: full/truncated routing trends and partial-failure physical activity.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, arrays = load_data()
    long_mask = frame["task"].str.contains("moka_pots", regex=False).to_numpy()
    long_failure_mask = long_mask & frame["failure"].to_numpy()
    truncate90_arrays = load_truncate90_events()
    frame, target_names = attach_long_task_physics(frame, long_failure_mask)

    counts = group_counts(frame)
    long_counts = group_counts(frame.loc[long_mask])
    curves = curve_profiles(frame, arrays, long_mask)
    scalars = scalar_profiles(frame, long_mask)
    matched = matched_initial_state_contrasts(frame, long_mask)
    patterns = pattern_profiles(frame, long_mask)
    zero_profiles = zero_vote_event_profiles(frame, arrays, long_mask)
    full_axis = exact_vote_axis(frame, arrays, long_failure_mask, (0.5, 1.0))
    truncate_axis = exact_vote_axis(
        frame, truncate90_arrays, long_failure_mask, (0.5, 0.9)
    )
    physical = physical_validation(frame, long_failure_mask, target_names)

    assert int(frame["failure"].sum()) == 307
    assert int(long_failure_mask.sum()) == 216
    assert frame.loc[long_failure_mask, "episode_length"].eq(52).all()
    assert (
        frame.loc[long_failure_mask, "consensus_votes"].value_counts().sort_index().to_dict()
        == {0: 30, 1: 59, 2: 27, 3: 100}
    )
    assert counts["failure_core"]["n"] == 213
    assert counts["failure_1vote"]["n"] == 62
    assert counts["failure_0vote"]["n"] == 32
    assert long_counts["failure_1vote"]["n"] == 59
    assert long_counts["failure_0vote"]["n"] == 30
    assert physical["partial_n"] == 198
    assert {
        int(vote): row["n"] for vote, row in physical["partial_by_vote"].items()
    } == {0: 27, 1: 51, 2: 20, 3: 100}
    partial_mask = long_failure_mask & frame["failure_mode_proxy"].eq("partial").to_numpy()
    assert frame.loc[partial_mask, "targets_placed"].eq(1).all()
    assert set(frame.loc[long_failure_mask, "failure_mode_proxy"]) == {
        "misplaced",
        "partial",
        "reached_then_lost",
    }
    assert np.isfinite(frame.loc[long_failure_mask, PHYSICAL_COLUMNS].to_numpy()).all()

    summary = {
        "schema": "himoe.residual_failure_dynamics.v2",
        "consensus_definition": {
            "aligned_vote": "aligned-route-kernel raw_cluster == 1",
            "event_vote": "route-change-events event_cluster == 0",
            "lag_vote": "alternative-routing lag_spectrum_louvain == C7",
            "failure_core": "failure with at least two votes",
        },
        "phase_definition": {
            "speed": np.linspace(0.0, 1.0, 9),
            "return": np.linspace(2 / 9, 1.0, 8),
            "normalization": "each speed curve divided by its episode median",
        },
        "counts": counts,
        "long_task_counts": long_counts,
        "long_task_curve_profiles": curves,
        "long_task_scalar_profiles": scalars,
        "long_task_matched_initial_state_contrasts": matched,
        "long_task_exact_vote_axis": {
            "full": full_axis,
            "truncate90": truncate_axis,
            "membership_note": "full-window votes are frozen for both feature windows",
        },
        "long_task_physical_validation": physical,
        "long_task_vote_pattern_profiles": patterns,
        "long_task_zero_vote_event_profiles": zero_profiles,
    }

    export_columns = KEY_COLUMNS + [
        "failure",
        "analysis_group",
        "aligned_vote",
        "event_vote",
        "lag_vote",
        "consensus_votes",
        "vote_pattern",
        "event_cluster",
        "failure_mode_proxy",
    ] + SCALAR_COLUMNS + PHYSICAL_COLUMNS
    frame[export_columns].to_csv(OUT_DIR / "assignments.csv", index=False)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_plot(curves, scalars, OUT_DIR / "long_task_dynamics.png")
    make_vote_axis_plot(
        full_axis, truncate_axis, physical, OUT_DIR / "vote_axis_and_physics.png"
    )
    (OUT_DIR / "report.md").write_text(report_text(plain(summary)), encoding="utf-8")


if __name__ == "__main__":
    main()
