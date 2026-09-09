#!/usr/bin/env python3
"""Audit failures outside the three-view MoE routing consensus core.

The analysis deliberately restricts subtype discovery to failures from one
task and one episode horizon.  This removes the two strongest nuisance axes
before asking whether the residual failures contain reproducible route types.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from analyze_failure_routing_clusters import (
    PhaseConfig,
    build_representations,
    query_tapes,
)
from analyze_route_change_events import extract_event_features


HERE = pathlib.Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
OUT_DIR = ANALYSIS / "residual-failure-subtypes"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LONG_RUN = (
    HERE.parent
    / "VLA_MUI_HUB/cache/HiMoE-VLA"
    / LONG_TASK
    / "right-16x32/client"
)
SEED = 20260829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--subsamples", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def robust_pca(values: np.ndarray, seed: int) -> np.ndarray:
    values = RobustScaler(quantile_range=(25, 75)).fit_transform(
        np.asarray(values, dtype=np.float64)
    )
    values = values[:, values.std(axis=0) > 1e-10]
    if not values.shape[1]:
        raise ValueError("all feature dimensions are constant")
    maximum = min(20, len(values) - 1, values.shape[1])
    model = PCA(n_components=maximum, random_state=seed)
    embedded = model.fit_transform(values)
    components = int(
        max(
            2,
            min(
                maximum,
                np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.90)
                + 1,
            ),
        )
    )
    return embedded[:, :components]


def density_labels(values: np.ndarray, seed: int) -> np.ndarray:
    embedded = robust_pca(values, seed)
    return HDBSCAN(
        min_cluster_size=8,
        min_samples=4,
        cluster_selection_method="eom",
        copy=True,
    ).fit_predict(embedded)


def label_summary(labels: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {"noise" if value < 0 else f"C{value}": int(count) for value, count in zip(unique, counts)}
    clusters = int(np.sum(unique >= 0))
    return {
        "clusters": clusters,
        "coverage": float(np.mean(labels >= 0)),
        "sizes": sizes,
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.sum(left | right)
    return float(np.sum(left & right) / union) if union else 1.0


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_csv(ANALYSIS / "aligned-route-kernel/assignments.csv")
    events = pd.read_csv(ANALYSIS / "route-change-events/assignments.csv")
    alternative = pd.read_csv(
        ANALYSIS / "alternative-routing-organizations/assignments.csv"
    )
    keys = ["task", "episode"]
    if not aligned[keys].equals(events[keys]) or not aligned[keys].equals(
        alternative[keys]
    ):
        raise ValueError("source assignment row order drifted")
    return aligned, events, alternative


def consensus_votes(
    aligned: pd.DataFrame, events: pd.DataFrame, alternative: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    flags = np.column_stack(
        [
            aligned["raw_cluster"].to_numpy() == 1,
            events["event_cluster"].to_numpy() == 0,
            alternative["lag_spectrum_louvain"].to_numpy() == "C7",
        ]
    )
    return flags, flags.sum(axis=1)


def load_views() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    aligned = np.load(
        ANALYSIS / "aligned-route-kernel/embeddings_labels_landmarks.npz",
        allow_pickle=False,
    )
    aligned_truncated = np.load(
        ANALYSIS
        / "aligned-route-kernel-truncate90/embeddings_labels_landmarks.npz",
        allow_pickle=False,
    )
    event = np.load(
        ANALYSIS / "route-change-events/event_features.npz", allow_pickle=False
    )
    alternative = np.load(
        ANALYSIS
        / "alternative-routing-organizations/representation_features.npz",
        allow_pickle=False,
    )
    grammar = np.load(
        ANALYSIS / "routing-state-grammar/arrays.npz", allow_pickle=False
    )
    views = {
        "aligned_raw": aligned["landmark_distance_raw"],
        "aligned_task_init": aligned["landmark_distance_task_init_residual"],
        "event": event["core"],
        "peer_rank": alternative["peer_rank"],
        "lag_spectrum": alternative["lag_spectrum"],
        "path_signature": alternative["path_signature"],
        "route_topology": alternative["route_topology"],
        "layer_wave": alternative["layer_wave"],
        "state_grammar": grammar["grammar_features"],
    }
    truncated = {
        "aligned_raw": aligned_truncated["landmark_distance_raw"],
        "aligned_task_residual": aligned_truncated[
            "landmark_distance_task_residual"
        ],
    }
    return views, truncated


def load_truncated_event_features() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = ANALYSIS / "all-outcome-routing-clusters/feature_cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    item = manifest["arrays"]["feature_truncate90_recurrence"]
    recurrence = np.load(cache / item["file"], allow_pickle=False)
    core, _names, _scale, _scale_names, tapes = extract_event_features(recurrence)
    return core, tapes


def physical_metrics(episodes: np.ndarray) -> pd.DataFrame:
    layout = json.loads((LONG_RUN / "sim_layout.json").read_text())
    joints = {item["joint"]: item for item in layout["joints"]}
    pot1 = joints["moka_pot_1_joint0"]
    pot2 = joints["moka_pot_2_joint0"]
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        with np.load(
            LONG_RUN / f"episode_{int(episode):02d}.npz", allow_pickle=False
        ) as archive:
            state = np.asarray(archive["state"], dtype=np.float64)
            actions = np.asarray(archive["actions"], dtype=np.float64)
            sim = np.asarray(archive["sim_state"], dtype=np.float64)
        first = sim[:, int(pot1["state_lo"]) : int(pot1["state_lo"]) + 3]
        second = sim[:, int(pot2["state_lo"]) : int(pot2["state_lo"]) + 3]
        first_final = float(np.linalg.norm(first[-1] - first[0]))
        first_maximum = float(np.linalg.norm(first - first[0], axis=1).max())
        eef_terminal = state[-1, :3]
        distance_to_pot1 = float(np.linalg.norm(eef_terminal - first[-1]))
        distance_to_pot2 = float(np.linalg.norm(eef_terminal - second[-1]))
        gripper = np.sign(actions[:, 0, -1])
        gripper = gripper[gripper != 0]
        rows.append(
            {
                "task": LONG_TASK,
                "episode": int(episode),
                "eef_query_path": float(
                    np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).sum()
                ),
                "gripper_query_flips": int(np.sum(gripper[1:] != gripper[:-1])),
                "pot1_final_displacement": first_final,
                "pot1_max_displacement": first_maximum,
                "pot2_final_displacement": float(
                    np.linalg.norm(second[-1] - second[0])
                ),
                "eef_terminal_distance_to_pot1": distance_to_pot1,
                "eef_terminal_distance_to_pot2": distance_to_pot2,
                "eef_terminal_side": (
                    "pot1_unfinished_side"
                    if distance_to_pot1 < distance_to_pot2
                    else "pot2_stove_side"
                ),
                "physical_stage": (
                    "pot1_not_attempted"
                    if first_maximum < 0.05
                    else "pot1_returned"
                    if first_final < 0.05
                    else "pot1_moved"
                ),
            }
        )
    return pd.DataFrame(rows)


def post_pot2_representations(
    episodes: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    summaries = pd.DataFrame(json.loads((LONG_RUN / "summaries.json").read_text()))
    lengths = summaries["inference_calls"].to_numpy(dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    layout = json.loads((LONG_RUN / "sim_layout.json").read_text())
    joints = {item["joint"]: item for item in layout["joints"]}
    pot2_lo = int(joints["moka_pot_2_joint0"]["state_lo"])

    successful_final = []
    for episode in np.flatnonzero(summaries["success"].to_numpy(dtype=bool)):
        with np.load(
            LONG_RUN / f"episode_{int(episode):02d}.npz", allow_pickle=False
        ) as archive:
            successful_final.append(archive["sim_state"][-1, pot2_lo : pot2_lo + 3])
    pot2_goal = np.median(np.stack(successful_final), axis=0)

    configurations = (
        PhaseConfig("full", 0.0, 1.0, 10),
        PhaseConfig("truncate90", 0.0, 0.9, 10),
        PhaseConfig("prefix60", 0.0, 0.6, 10),
    )
    names = ("geometry", "change", "recurrence")
    features: dict[str, dict[str, list[np.ndarray]]] = {
        config.name: {name: [] for name in names} for config in configurations
    }
    starts = []
    route = zarr.open_group(str(LONG_RUN.parent / "server/routes.zarr"), mode="r")
    for episode in episodes:
        episode = int(episode)
        with np.load(
            LONG_RUN / f"episode_{episode:02d}.npz", allow_pickle=False
        ) as archive:
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
        distance = np.linalg.norm(sim[:, pot2_lo : pot2_lo + 3] - pot2_goal, axis=1)
        hits = np.flatnonzero(distance <= 0.05)
        if not len(hits):
            raise ValueError(f"pot2 placement milestone missing in episode {episode}")
        post_start = int(hits[0])
        starts.append(post_start)
        first = int(offsets[episode] + post_start)
        stop = int(offsets[episode] + lengths[episode])
        probabilities = np.asarray(route["hb_router_probs"][first:stop], np.float32)
        expert_ids = np.asarray(route["hb_expert_ids"][first:stop], np.uint8)
        tapes = query_tapes(probabilities, expert_ids)
        for config in configurations:
            values, _diagnostics = build_representations(tapes, config)
            for name in names:
                features[config.name][name].append(values[name])
    stacked = {
        window: {name: np.stack(values) for name, values in views.items()}
        for window, views in features.items()
    }
    return stacked, np.asarray(starts, dtype=np.int32)


def grouped_side_separability(
    values: np.ndarray,
    side: np.ndarray,
    initial_state: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = np.zeros(len(values), dtype=np.float64)
    fold_counts = []
    for train, test in splitter.split(values, side, groups=initial_state):
        scaler = RobustScaler(quantile_range=(25, 75)).fit(values[train])
        train_values = scaler.transform(values[train])
        test_values = scaler.transform(values[test])
        keep = train_values.std(axis=0) > 1e-10
        train_values = train_values[:, keep]
        test_values = test_values[:, keep]
        maximum = min(20, len(train) - 1, train_values.shape[1])
        pca = PCA(n_components=maximum, random_state=seed).fit(train_values)
        components = int(
            max(
                2,
                min(
                    maximum,
                    np.searchsorted(
                        np.cumsum(pca.explained_variance_ratio_), 0.90
                    )
                    + 1,
                ),
            )
        )
        train_values = pca.transform(train_values)[:, :components]
        test_values = pca.transform(test_values)[:, :components]
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000
        ).fit(train_values, side[train])
        probability[test] = model.predict_proba(test_values)[:, 1]
        fold_counts.append(np.bincount(side[test], minlength=2).astype(int).tolist())
    return {
        "grouping": "five-fold StratifiedGroupKFold by init_state_id",
        "roc_auc": float(roc_auc_score(side, probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(side, probability >= 0.5)
        ),
        "fold_class_counts": fold_counts,
    }


def permutation_nmi(
    initial_state: np.ndarray, labels: np.ndarray, draws: int, seed: int
) -> dict[str, float]:
    observed = float(normalized_mutual_info_score(initial_state, labels))
    adjusted = float(adjusted_mutual_info_score(initial_state, labels))
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [
            normalized_mutual_info_score(initial_state, rng.permutation(labels))
            for _ in range(draws)
        ]
    )
    return {
        "nmi": observed,
        "adjusted_mi": adjusted,
        "permutation_p": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def event_subsample_stability(
    values: np.ndarray, full_labels: np.ndarray, draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_size = int(np.ceil(0.8 * len(values)))
    ari: list[float] = []
    clusters: list[int] = []
    for draw in range(draws):
        index = np.sort(rng.choice(len(values), sample_size, replace=False))
        labels = density_labels(values[index], seed + draw + 1)
        ari.append(float(adjusted_rand_score(full_labels[index], labels)))
        clusters.append(len(set(labels)) - int(-1 in labels))
    return {
        "draws": draws,
        "ari_median": float(np.median(ari)),
        "ari_p10": float(np.quantile(ari, 0.10)),
        "zero_cluster_draws": int(np.sum(np.asarray(clusters) == 0)),
        "cluster_count_histogram": {
            str(value): int(np.sum(np.asarray(clusters) == value))
            for value in sorted(set(clusters))
        },
    }


def group_numeric(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "mean": float(frame[column].mean()),
        "median": float(frame[column].median()),
    }


def make_plot(gradient: list[dict[str, Any]], output: pathlib.Path) -> None:
    votes = np.asarray([row["vote"] for row in gradient])
    counts = np.asarray([row["count"] for row in gradient])
    speed = np.asarray([row["soft_late_early_ratio"]["median"] for row in gradient])
    stasis = np.asarray([row["late_stasis_rate"] for row in gradient])
    path = np.asarray([row["eef_query_path"]["median"] for row in gradient])
    flips = np.asarray([row["gripper_query_flips"]["median"] for row in gradient])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar(votes, counts, color="#607D8B")
    axes[0].set(xlabel="consensus votes", ylabel="long-task failures", xticks=votes)
    axes[1].plot(votes, speed, "o-", label="late/early route speed")
    axes[1].plot(votes, stasis, "s-", label="late stasis rate")
    axes[1].set(xlabel="consensus votes", xticks=votes)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(votes, path, "o-", label="EEF query path")
    axes[2].plot(votes, flips, "s-", label="gripper flips")
    axes[2].set(xlabel="consensus votes", xticks=votes)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aligned, events, alternative = load_tables()
    flags, votes = consensus_votes(aligned, events, alternative)
    failure = aligned["outcome"].to_numpy() == "failure"
    residual = failure & (votes < 2)
    if int(failure.sum()) != 307 or int(residual.sum()) != 94:
        raise ValueError("formal failure/consensus cohort drifted")
    long_failure = failure & (aligned["task"].to_numpy() == LONG_TASK)
    long_residual = residual & (aligned["task"].to_numpy() == LONG_TASK)
    long_index = np.flatnonzero(long_residual)
    if len(long_index) != 89:
        raise ValueError("expected 89 residual long-task failures")
    if set(aligned.loc[long_residual, "episode_length"]) != {52}:
        raise ValueError("long residual cohort no longer has a fixed horizon")

    long_episodes = aligned.loc[long_failure, "episode"].to_numpy()
    physical = physical_metrics(long_episodes)
    residual_episodes = aligned.loc[long_residual, "episode"].to_numpy()
    physical_by_episode = physical.set_index("episode")
    residual_physical = physical_by_episode.loc[residual_episodes]
    endpoint_side = (
        residual_physical["eef_terminal_side"].to_numpy()
        == "pot1_unfinished_side"
    ).astype(np.int32)

    views, truncated_aligned = load_views()
    full_labels = {
        name: density_labels(values[long_index], args.seed + offset)
        for offset, (name, values) in enumerate(views.items())
    }
    view_summary = {
        name: label_summary(labels) for name, labels in full_labels.items()
    }
    for name, labels in full_labels.items():
        view_summary[name]["endpoint_side_nmi"] = float(
            normalized_mutual_info_score(endpoint_side, labels)
        )
        view_summary[name]["endpoint_side_adjusted_mi"] = float(
            adjusted_mutual_info_score(endpoint_side, labels)
        )
        cross = pd.crosstab(labels, endpoint_side)
        view_summary[name]["cluster_by_endpoint_side"] = {
            "noise" if int(label) < 0 else f"C{int(label)}": {
                "pot2_stove_side": int(row.get(0, 0)),
                "pot1_unfinished_side": int(row.get(1, 0)),
            }
            for label, row in cross.iterrows()
        }
    viable = [name for name, item in view_summary.items() if item["clusters"] >= 2]
    pairwise = []
    for left, right in combinations(viable, 2):
        pairwise.append(
            {
                "left": left,
                "right": right,
                "ari": float(adjusted_rand_score(full_labels[left], full_labels[right])),
            }
        )

    event_archive = np.load(
        ANALYSIS / "route-change-events/event_features.npz", allow_pickle=False
    )
    truncated_event, truncated_tapes = load_truncated_event_features()
    event_full = full_labels["event"]
    event_truncated = density_labels(truncated_event[long_index], args.seed + 100)
    no_return_full = ~event_archive["return_event"][long_index].any(axis=1)
    no_return_truncated = ~truncated_tapes["return_event"][long_index].any(axis=1)
    exact_no_return_clusters = [
        int(label)
        for label in np.unique(event_full[event_full >= 0])
        if np.array_equal(event_full == label, no_return_full)
    ]
    if len(exact_no_return_clusters) != 1:
        raise ValueError("full event no-return density cluster is no longer exact")
    exact_no_return_cluster = exact_no_return_clusters[0]

    aligned_full_raw = full_labels["aligned_raw"]
    aligned_truncated_raw = density_labels(
        truncated_aligned["aligned_raw"][long_index], args.seed + 101
    )
    aligned_full_task = density_labels(
        np.load(
            ANALYSIS / "aligned-route-kernel/embeddings_labels_landmarks.npz",
            allow_pickle=False,
        )["landmark_distance_task_residual"][long_index],
        args.seed + 102,
    )
    aligned_truncated_task = density_labels(
        truncated_aligned["aligned_task_residual"][long_index], args.seed + 103
    )
    truncation = {
        "event": {
            "full": label_summary(event_full),
            "truncate90": label_summary(event_truncated),
            "ari": float(adjusted_rand_score(event_full, event_truncated)),
            "full_no_return_count": int(no_return_full.sum()),
            "truncate90_no_return_count": int(no_return_truncated.sum()),
            "no_return_intersection": int(np.sum(no_return_full & no_return_truncated)),
            "no_return_jaccard": jaccard(no_return_full, no_return_truncated),
            "exact_full_no_return_cluster": f"C{exact_no_return_cluster}",
        },
        "aligned_raw": {
            "full": label_summary(aligned_full_raw),
            "truncate90": label_summary(aligned_truncated_raw),
            "ari": float(adjusted_rand_score(aligned_full_raw, aligned_truncated_raw)),
        },
        "aligned_task_residual": {
            "full": label_summary(aligned_full_task),
            "truncate90": label_summary(aligned_truncated_task),
            "ari": float(
                adjusted_rand_score(aligned_full_task, aligned_truncated_task)
            ),
        },
    }

    residual_frame = aligned.loc[residual].copy()
    residual_frame["aligned_vote"] = flags[residual, 0]
    residual_frame["event_vote"] = flags[residual, 1]
    residual_frame["lag_vote"] = flags[residual, 2]
    residual_frame["vote_count"] = votes[residual]
    residual_frame["residual_group"] = np.where(
        residual_frame["task"] != LONG_TASK,
        "other_task_isolate",
        np.where(residual_frame["vote_count"] == 0, "long_zero_vote", "long_one_vote"),
    )
    residual_frame["event_density_cluster"] = -99
    residual_frame["aligned_density_cluster"] = -99
    residual_frame["no_return_full"] = False
    residual_frame.loc[
        residual_frame["task"] == LONG_TASK, "event_density_cluster"
    ] = event_full
    residual_frame.loc[
        residual_frame["task"] == LONG_TASK, "aligned_density_cluster"
    ] = aligned_full_raw
    residual_frame.loc[
        residual_frame["task"] == LONG_TASK, "no_return_full"
    ] = no_return_full

    long_frame = aligned.loc[long_failure].copy()
    long_frame["vote"] = votes[long_failure]
    long_frame = long_frame.merge(
        physical, on=["task", "episode"], validate="one_to_one"
    )
    event_names = list(event_archive["core_names"].astype(str))
    event_core = event_archive["core"][np.flatnonzero(long_failure)]
    for name in (
        "soft_late_early_ratio",
        "global_peak_phase",
        "late_stasis_indicator",
        "return_count_fraction",
    ):
        long_frame[name] = event_core[:, event_names.index(name)]

    gradient = []
    for vote, group in long_frame.groupby("vote", sort=True):
        gradient.append(
            {
                "vote": int(vote),
                "count": int(len(group)),
                "soft_late_early_ratio": group_numeric(
                    group, "soft_late_early_ratio"
                ),
                "global_peak_phase": group_numeric(group, "global_peak_phase"),
                "late_stasis_rate": float(group["late_stasis_indicator"].mean()),
                "return_count_fraction": group_numeric(
                    group, "return_count_fraction"
                ),
                "eef_query_path": group_numeric(group, "eef_query_path"),
                "gripper_query_flips": group_numeric(
                    group, "gripper_query_flips"
                ),
                "physical_stage_counts": {
                    str(key): int(value)
                    for key, value in group["physical_stage"].value_counts().items()
                },
                "eef_terminal_side_counts": {
                    str(key): int(value)
                    for key, value in group["eef_terminal_side"].value_counts().items()
                },
            }
        )

    residual_long_frame = long_frame[long_frame["vote"] < 2]
    initial_shadow = permutation_nmi(
        residual_long_frame["init_state_id"].to_numpy(),
        residual_long_frame["vote"].to_numpy(),
        args.permutations,
        args.seed,
    )
    initial_shadow["vote_by_initial_state"] = {
        str(initial): {str(vote): int(count) for vote, count in counts.items()}
        for initial, counts in pd.crosstab(
            residual_long_frame["init_state_id"], residual_long_frame["vote"]
        ).to_dict(orient="index").items()
    }

    physical_residual = physical[
        physical["episode"].isin(aligned.loc[long_residual, "episode"])
    ]
    residual_frame = residual_frame.merge(
        physical_residual,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    )
    other_task = residual_frame["task"] != LONG_TASK
    if residual_frame.loc[other_task, "eef_terminal_side"].notna().any():
        raise ValueError("long-task physical metadata leaked onto another task")

    post_features, post_start = post_pot2_representations(residual_episodes)
    post_labels: dict[str, dict[str, np.ndarray]] = {}
    post_summary: dict[str, Any] = {}
    post_arrays: dict[str, np.ndarray] = {
        "episode": residual_episodes.astype(np.int32),
        "post_pot2_start_query": post_start,
        "endpoint_side": endpoint_side.astype(np.int8),
    }
    residual_initial_state = aligned.loc[long_residual, "init_state_id"].to_numpy()
    for view_offset, view in enumerate(("geometry", "change", "recurrence")):
        post_labels[view] = {}
        windows = {}
        for window_offset, window in enumerate(("full", "truncate90", "prefix60")):
            values = post_features[window][view]
            labels = density_labels(
                values, args.seed + 200 + 10 * view_offset + window_offset
            )
            post_labels[view][window] = labels
            item = label_summary(labels)
            item["endpoint_side_nmi"] = float(
                normalized_mutual_info_score(endpoint_side, labels)
            )
            item["endpoint_side_adjusted_mi"] = float(
                adjusted_mutual_info_score(endpoint_side, labels)
            )
            item["endpoint_side_separability"] = grouped_side_separability(
                values,
                endpoint_side,
                residual_initial_state,
                args.seed + 300,
            )
            windows[window] = item
            post_arrays[f"feature_{view}_{window}"] = values.astype(np.float32)
            post_arrays[f"label_{view}_{window}"] = labels.astype(np.int16)
        post_summary[view] = {
            "windows": windows,
            "full_truncate90_ari": float(
                adjusted_rand_score(
                    post_labels[view]["full"], post_labels[view]["truncate90"]
                )
            ),
            "full_prefix60_ari": float(
                adjusted_rand_score(
                    post_labels[view]["full"], post_labels[view]["prefix60"]
                )
            ),
        }
    post_summary["milestone_query"] = {
        "minimum": int(post_start.min()),
        "median": float(np.median(post_start)),
        "maximum": int(post_start.max()),
        "definition": "first query with pot2 within 0.05 m of median successful terminal pot2 position",
        "relative_windows": {
            "full": "0--100% of the milestone-to-terminal interval",
            "truncate90": "0--90% of the milestone-to-terminal interval",
            "prefix60": "0--60% of the milestone-to-terminal interval",
        },
        "online_caveat": "window lengths use the eventual terminal query and are not fixed-horizon online prefixes",
    }
    np.savez_compressed(args.out_dir / "post_pot2_features.npz", **post_arrays)
    residual_frame.to_csv(args.out_dir / "assignments.csv", index=False)

    pairwise_values = [item["ari"] for item in pairwise]
    independent_pairwise_values = [
        item["ari"]
        for item in pairwise
        if {item["left"], item["right"]}
        != {"aligned_raw", "aligned_task_init"}
    ]
    summary = {
        "cohort": {
            "all_episodes": int(len(aligned)),
            "all_failures": int(failure.sum()),
            "consensus_core_failures": int(np.sum(failure & (votes >= 2))),
            "residual_failures": int(residual.sum()),
            "residual_by_task": {
                str(key): int(value)
                for key, value in residual_frame["task"].value_counts().items()
            },
            "long_residual": int(long_residual.sum()),
            "long_residual_horizon": 52,
            "long_residual_vote_counts": {
                str(key): int(value)
                for key, value in residual_long_frame["vote"].value_counts().sort_index().items()
            },
        },
        "failure_only_density_clustering": {
            "views": view_summary,
            "pairwise_ari": pairwise,
            "pairwise_ari_median": float(np.median(pairwise_values)),
            "pairwise_ari_maximum": float(np.max(pairwise_values)),
            "pairwise_ari_maximum_excluding_aligned_variants": float(
                np.max(independent_pairwise_values)
            ),
            "event_subsample_stability": event_subsample_stability(
                views["event"][long_index],
                event_full,
                args.subsamples,
                args.seed + 1000,
            ),
        },
        "terminal_truncation": truncation,
        "endpoint_state_shadow": {
            "side_definition": "terminal EEF is assigned to the nearer final pot1 or pot2 position",
            "pot1_unfinished_side_count": int(endpoint_side.sum()),
            "pot2_stove_side_count": int(len(endpoint_side) - endpoint_side.sum()),
            "residual_vote_side_nmi": float(
                normalized_mutual_info_score(
                    residual_long_frame["vote"],
                    (
                        residual_long_frame["eef_terminal_side"]
                        == "pot1_unfinished_side"
                    ).astype(np.int32),
                )
            ),
            "vote_by_side_all_long_failures": {
                str(vote): {str(side): int(count) for side, count in counts.items()}
                for vote, counts in pd.crosstab(
                    long_frame["vote"], long_frame["eef_terminal_side"]
                ).to_dict(orient="index").items()
            },
        },
        "post_pot2_relative_phase": post_summary,
        "initial_state_shadow": initial_shadow,
        "long_failure_vote_gradient": gradient,
        "interpretation": {
            "stable_discrete_subtypes_found": False,
            "primary_structure": "continuous noncompletion/activity axis plus physical-progress variation",
            "event_no_return_subtype_rejected": "full/truncate90 membership is not stable",
            "other_task_residuals": "five isolated episodes are insufficient for clustering",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    make_plot(gradient, args.out_dir / "residual_gradient.png")

    view_lines = []
    for name, item in view_summary.items():
        view_lines.append(
            f"| {name} | {item['clusters']} | {item['coverage']:.3f} | "
            f"{item['endpoint_side_adjusted_mi']:.3f} | "
            f"{json.dumps(item['sizes'], ensure_ascii=False)} |"
        )
    gradient_lines = []
    for row in gradient:
        gradient_lines.append(
            f"| {row['vote']} | {row['count']} | "
            f"{row['soft_late_early_ratio']['median']:.3f} | "
            f"{row['global_peak_phase']['median']:.3f} | "
            f"{row['late_stasis_rate']:.3f} | "
            f"{row['eef_query_path']['median']:.3f} | "
            f"{row['gripper_query_flips']['median']:.1f} |"
        )
    post_lines = []
    for view in ("geometry", "change", "recurrence"):
        item = post_summary[view]
        full = item["windows"]["full"]
        truncated = item["windows"]["truncate90"]
        prefix = item["windows"]["prefix60"]
        post_lines.append(
            f"| {view} | {full['clusters']} | {truncated['clusters']} | "
            f"{prefix['clusters']} | {item['full_truncate90_ari']:.3f} | "
            f"{item['full_prefix60_ari']:.3f} | "
            f"{full['endpoint_side_separability']['roc_auc']:.3f} | "
            f"{truncated['endpoint_side_separability']['roc_auc']:.3f} | "
            f"{prefix['endpoint_side_separability']['roc_auc']:.3f} |"
        )
    report = f"""# 共识核心之外失败的再分析

## 直接结论

没有证据把剩余 94 条失败再切成若干稳定的 MoE 路由类别。

- `89/94` 来自同一个 long moka-pot 任务，而且全部正好跑满 `52` 次 query。
- 这 89 条中，`59` 条已被三种核心方法中的一种选中，更像共同核心的边界；只有 `30` 条是三种方法均未选中的远端样本。
- 另外 5 条分散在三个任务，单个任务只有 `1/1/3` 条，不能做可靠聚类。
- 在同一任务、同一长度的 89 条上重做九种表示的 HDBSCAN，三种表示全为 noise，其余表示给出的分组彼此不一致；可行视图两两 ARI 中位数为 `{summary['failure_only_density_clustering']['pairwise_ari_median']:.3f}`。除去同源的两个 aligned 变体，最高也只有 `{summary['failure_only_density_clustering']['pairwise_ari_maximum_excluding_aligned_variants']:.3f}`。
- 删除末尾 10% 后，event 分组 ARI 为 `{truncation['event']['ari']:.3f}`，aligned-raw 分组 ARI 为 `{truncation['aligned_raw']['ari']:.3f}`，task-residual 分组 ARI 为 `{truncation['aligned_task_residual']['ari']:.3f}`。它们都没有复现。
- 最强的 aligned 分组主要恢复终端机械臂停在 pot-1 侧还是 pot-2/炉灶侧，而不是新的失败原因。

因此，剩余样本更适合描述成一条连续的“仍在活动/逐渐停滞”轴，加上不同物理进度，而不是新的离散失败类型。

## 剩余 94 条是什么

| 范围 | 数量 | 解释 |
|---|---:|---|
| long task，1 票 | 59 | 靠近共同失败核心，但没有达到 2-of-3 共识 |
| long task，0 票 | 30 | 路由仍较活跃、离共同停滞核心更远 |
| 其他任务 | 5 | 三个任务中的孤立尾部，样本不足 |

这里的 `0/1 票` 不是新标签，只表示 rollout 被 aligned-kernel、change-event、lag-graph 三种方法中的几种选中。

## 连续轴

把同一个 long task 的全部 216 条失败按 0--3 票排列，会看到渐变而不是清楚断层：

| 票数 | n | late/early route speed 中位数 | route peak 中位相位 | late stasis 比例 | EEF query path 中位数 | gripper query flips 中位数 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(gradient_lines)}

0 票端的路由晚期仍然较快，变化峰值更晚，几乎不进入 late stasis；3 票端的路由较早减速并长期平台化。1 票样本多数位于两端之间，但不同指标并非严格单调，这正是“连续轴 + 物理阶段差异”而非单一离散类别的表现。

物理代理只用于解释，不参与路由聚类。`pot1_max_displacement < 0.05` 时记作尚未明显尝试第二阶段要处理的 pot-1。0 票中为 `27/30`，1 票中为 `44/59`；另有 `3/30` 和 `14/59` 已明显移动 pot-1。说明一部分 1 票轨迹只是走到了不同任务阶段，不能据此命名新的路由失败类。

终端 EEF 位置提供了更直接的解释。按终端 EEF 更靠近 pot-1 还是已放到炉灶侧的 pot-2 划分：

| 票数 | pot-2/炉灶侧 | 未完成 pot-1 侧 |
|---:|---:|---:|
| 0 | 27 | 3 |
| 1 | 12 | 47 |
| 2 | 1 | 26 |
| 3 | 0 | 100 |

所以 0 票并不是一种新“错误机制”：它主要表示 episode 终止时机械臂又回到了已完成的 pot-2/炉灶一侧；共同核心则几乎全停在未完成的 pot-1 一侧。对剩余 89 条，0/1 票与这个终端侧别的 NMI 为 `{summary['endpoint_state_shadow']['residual_vote_side_nmi']:.3f}`。

## 聚类核验

所有聚类都只在相同 task、相同 52-query horizon 的 89 条失败内拟合：

| 路由表示 | 密度簇数 | 被分配比例 | 与终端侧别 adjusted MI | 大小（noise 单列） |
|---|---:|---:|---:|---|
{chr(10).join(view_lines)}

aligned-raw 的终端侧别 adjusted MI 为 `{view_summary['aligned_raw']['endpoint_side_adjusted_mi']:.3f}`。其两个主簇分别为 `35 pot-2 + 1 pot-1` 和 `0 pot-2 + 35 pot-1`；aligned task-init residual 仍有 `{view_summary['aligned_task_init']['endpoint_side_adjusted_mi']:.3f}`。这说明即使在 failure-only 内部，最清楚的“分类”也是状态/任务阶段阴影。

唯一看上去很干净的是 event 表示中的 12 条 `{truncation['event']['exact_full_no_return_cluster']}`：代码断言它与 full window 内“没有超过阈值的 nonlocal routing return”逐条完全相同。但删除末尾 10% 后，无回返样本变成 22 条，原 12 条只有 4 条仍保留，Jaccard 仅 `{truncation['event']['no_return_jaccard']:.3f}`。所以这个小簇是窗口与阈值敏感的，不保留为亚型。

event 小簇在 80% 子采样下的 ARI median/P10 为 `{summary['failure_only_density_clustering']['event_subsample_stability']['ari_median']:.3f}/{summary['failure_only_density_clustering']['event_subsample_stability']['ari_p10']:.3f}`；它在 full window 内看似稳定，但终端截尾失败，不能过关。

## 从 pot-2 完成后重新对齐

为了避免整条 episode 的前半段主导结果，我把每条轨迹中 pot-2 第一次进入成功终点 5 cm 内的 query 当作新零点，之后只使用相对于这个事件的 0--1 阶段，不把绝对 query 当分析尺度。`prefix60` 指“该里程碑到最终终止区间”的前 60%，区间长度仍使用了未来终点，所以它不是严格在线、固定 query 数的预测窗口。然后分别分析 full、删末 10%、只取前 60% 路由：

| 表示 | full 簇数 | truncate90 簇数 | prefix60 簇数 | full/tr90 ARI | full/prefix60 ARI | 终端侧别 AUC full | AUC tr90 | AUC prefix60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(post_lines)}

这里的 AUC 是按 init-state 分组的五折诊断分类，终端侧别只用于事后解释，没有参与 HDBSCAN。结果说明两件事：

1. post-pot2 路由确实很容易读出轨迹最后会停在哪一侧；full/truncate90 的 AUC 很高。
2. 但无监督簇仍不稳定，full/truncate90 的 ARI 最高只有 recurrence 的 `{post_summary['recurrence']['full_truncate90_ari']:.3f}`；只保留 post-pot2 前 60% 后，与 full 的 ARI 进一步降到 `0.16--0.25`，且大量样本被判为 noise。

因此这更像路由随闭环物理状态逐渐分叉，而不是 pot-2 完成后立刻进入两个固定的内部失败盆地。prefix60 的 AUC 高于 0.5 只能作为后续固定时点预测实验的线索，不能在这次 post-hoc、按最终长度归一化的分析里当成因果前兆。

## 初态阴影

任务和长度在 89 条内已经固定，但初始场景仍有影响。0/1 票与 init-state 的 NMI 为 `{initial_shadow['nmi']:.3f}`，置换 `p={initial_shadow['permutation_p']:.4f}`。例如 init 39 的 long-task 失败中有 `10/28` 是 0 票，而 init 49 是 `0/32`。因此即使强行切开，部分结构也会是在恢复初始物体布局，而不是失败机制。

`peer_rank` 已对 task x init-state 的 32 个 sibling 做了相对化；它在这 89 条上全部被 HDBSCAN 判为 noise。这是没有稳定、去初态离散亚型的直接证据。

## 最终组织

目前最忠实的组织是：

```text
共同 noncompletion 核心（>=2 票）
    -> 1 票边界带（59 条 long residual）
    -> 0 票活跃尾部（30 条 long residual）
    -> 5 条跨任务孤立点
```

箭头表示连续远近，不表示四种失败原因。要把尾部进一步命名成“滑落、错放、反复重抓”等物理亚型，需要按物理事件对齐或人工/视觉标注；仅靠当前整条 rollout 的 MoE 路由不能可靠完成。
"""
    (args.out_dir / "report.md").write_text(report)


def self_test() -> None:
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(-2, 0.1, (12, 5)), rng.normal(2, 0.1, (12, 5))]
    labels = density_labels(values, 0)
    assert len(set(labels) - {-1}) == 2
    assert jaccard(np.asarray([True, False]), np.asarray([True, True])) == 0.5


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
