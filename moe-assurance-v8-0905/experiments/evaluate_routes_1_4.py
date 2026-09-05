#!/usr/bin/env python3
"""Evaluate commitment, coherence, graph, and healthy-manifold routes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "assurance"))

from evaluation import episode_rows, load_task, matched_onset_auc, profile_inventory  # noqa: E402
from profile_schema import empirical_percentile  # noqa: E402


OUTPUT = BUNDLE / "results/routes_1_4"
MANIFOLD_FEATURES = (
    "late_flow_volatility",
    "route_acceleration",
    "graph_commitment_mean",
    "graph_token_consensus_mean",
    "graph_effective_rank_mean",
    "graph_token_expert_mi_mean",
    "graph_occupancy_concentration_mean",
    "graph_support_union_fraction_mean",
    "state_action_gap",
    "long_lag_return_advantage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--healthy-rows-per-episode", type=int, default=8)
    return parser.parse_args()


def matrix(data, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([data.profile.feature(name) for name in names]).astype(np.float64)


def subsample(rows: np.ndarray, limit: int) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    positions = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return rows[positions]


def reference_values(tasks, feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    reference = {}
    for name in feature_names:
        values = [
            data.profile.feature(name)
            for data in tasks
            if data.corpus == "main16x32"
        ]
        reference[name] = np.concatenate(values).astype(np.float64)
    return reference


def rank(reference: dict[str, np.ndarray], name: str, values: np.ndarray) -> np.ndarray:
    return empirical_percentile(reference[name], values)


def fit_healthy_manifold(tasks, rows_per_episode: int):
    samples = []
    episodes = 0
    for data in tasks:
        if data.corpus != "main16x32":
            continue
        events = data.event_by_episode
        values = matrix(data, MANIFOLD_FEATURES)
        for episode, event in events.items():
            is_healthy = (
                int(event["success"]) == 1
                and int(event["loop_onset_q"]) < 0
                and int(event["static_onset_q"]) < 0
            )
            if not is_healthy:
                continue
            rows = subsample(episode_rows(data.profile, episode), rows_per_episode)
            samples.append(values[rows])
            episodes += 1
    raw = np.concatenate(samples, axis=0)
    medians = np.nanmedian(raw, axis=0)
    raw_filled = np.where(np.isfinite(raw), raw, medians)
    ecdfs = [np.sort(raw_filled[:, column]) for column in range(raw.shape[1])]
    gaussian = np.column_stack(
        [
            norm.ppf(empirical_percentile(ecdfs[column], raw_filled[:, column]))
            for column in range(raw.shape[1])
        ]
    )
    estimator = LedoitWolf(store_precision=True, assume_centered=False).fit(gaussian)
    return {
        "medians": medians,
        "ecdfs": ecdfs,
        "location": estimator.location_,
        "precision": estimator.precision_,
        "shrinkage": float(estimator.shrinkage_),
        "rows": len(raw),
        "episodes": episodes,
    }


def healthy_energy(values: np.ndarray, model) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, model["medians"])
    gaussian = np.column_stack(
        [
            norm.ppf(empirical_percentile(model["ecdfs"][column], filled[:, column]))
            for column in range(values.shape[1])
        ]
    )
    centered = gaussian - model["location"]
    return np.einsum("ni,ij,nj->n", centered, model["precision"], centered)


def make_scores(data, reference, manifold):
    feature = data.profile.feature
    commitment = feature("graph_commitment_mean")
    entropy = feature("graph_entropy_normalized_mean")
    margin = feature("graph_top12_margin_mean")
    top4 = feature("graph_top4_mass_mean")
    volatility = rank(reference, "late_flow_volatility", feature("late_flow_volatility"))
    acceleration = rank(reference, "route_acceleration", feature("route_acceleration"))
    dispersion = rank(reference, "graph_token_dispersion_mean", feature("graph_token_dispersion_mean"))
    mutual_information = rank(
        reference, "graph_token_expert_mi_mean", feature("graph_token_expert_mi_mean")
    )
    commitment_rank = rank(reference, "graph_commitment_mean", commitment)
    entropy_rank = rank(reference, "graph_entropy_normalized_mean", entropy)
    consensus_rank = rank(
        reference, "graph_token_consensus_mean", feature("graph_token_consensus_mean")
    )
    rank_low = 1.0 - rank(
        reference, "graph_effective_rank_mean", feature("graph_effective_rank_mean")
    )
    occupancy_high = rank(
        reference,
        "graph_occupancy_concentration_mean",
        feature("graph_occupancy_concentration_mean"),
    )
    support_low = 1.0 - rank(
        reference,
        "graph_support_union_fraction_mean",
        feature("graph_support_union_fraction_mean"),
    )
    lag1_low = 1.0 - rank(reference, "lag1_distance", feature("lag1_distance"))
    return {
        "r1_commitment": commitment,
        "r1_entropy": entropy,
        "r1_margin": margin,
        "r1_top4_mass": top4,
        "r2_late_volatility": volatility,
        "r2_route_acceleration": acceleration,
        "r2_flow_instability": 0.5 * (volatility + acceleration),
        "r3_token_consensus": feature("graph_token_consensus_mean"),
        "r3_effective_rank": feature("graph_effective_rank_mean"),
        "r3_token_expert_mi": feature("graph_token_expert_mi_mean"),
        "r3_occupancy_concentration": feature("graph_occupancy_concentration_mean"),
        "r3_support_union_fraction": feature("graph_support_union_fraction_mean"),
        "r3_loop_graph_evidence": np.mean(
            np.column_stack((commitment_rank, dispersion, mutual_information)), axis=1
        ),
        "r3_static_authority_loss": np.mean(
            np.column_stack(
                (entropy_rank, consensus_rank, rank_low, occupancy_high, support_low)
            ),
            axis=1,
        ),
        "r3_static_lockin": lag1_low,
        "r4_healthy_energy": healthy_energy(matrix(data, MANIFOLD_FEATURES), manifold),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(item) for item in profile_inventory()]
    reference_names = (
        "late_flow_volatility",
        "route_acceleration",
        "graph_commitment_mean",
        "graph_entropy_normalized_mean",
        "graph_token_dispersion_mean",
        "graph_token_consensus_mean",
        "graph_effective_rank_mean",
        "graph_token_expert_mi_mean",
        "graph_occupancy_concentration_mean",
        "graph_support_union_fraction_mean",
        "lag1_distance",
    )
    reference = reference_values(tasks, reference_names)
    manifold = fit_healthy_manifold(tasks, args.healthy_rows_per_episode)

    task_scores = {}
    score_names = None
    score_root = args.output / "scores"
    for data in tasks:
        scores = make_scores(data, reference, manifold)
        if score_names is None:
            score_names = tuple(scores)
        elif tuple(scores) != score_names:
            raise RuntimeError("score schema drift")
        key = (data.corpus, data.suite, data.task)
        task_scores[key] = scores
        destination = score_root / data.corpus / data.suite / f"{data.task}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            episode_id=data.profile.episode_id,
            query=data.profile.query,
            score_names=np.asarray(score_names),
            scores=np.column_stack([scores[name] for name in score_names]).astype(np.float32),
        )

    rows = []
    for corpus in ("main16x32", "grid50x8"):
        subset = [data for data in tasks if data.corpus == corpus]
        for event_type in ("loop", "static"):
            for lead in (-2, 0):
                for score_name in score_names:
                    selected = {
                        key: values[score_name]
                        for key, values in task_scores.items()
                        if key[0] == corpus
                    }
                    result = matched_onset_auc(subset, selected, event_type, lead)
                    rows.append(
                        {
                            "corpus": corpus,
                            "event": event_type,
                            "lead": lead,
                            "score": score_name,
                            **result,
                        }
                    )
    csv_path = args.output / "matched_onset_auc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model_path = args.output / "healthy_manifold_model.npz"
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(MANIFOLD_FEATURES),
        medians=manifold["medians"],
        location=manifold["location"],
        precision=manifold["precision"],
        **{f"ecdf_{index}": values for index, values in enumerate(manifold["ecdfs"])},
    )
    summary = {
        "schema": "himoe.assurance.routes_1_4.v1",
        "runtime_task_metadata": False,
        "reference_corpus": "main16x32",
        "percentile_reference_uses_outcomes": False,
        "healthy_manifold_uses_outcome_labels_for_reference_selection": True,
        "healthy_manifold_selection": "successful episodes with neither loop nor static",
        "healthy_reference_episodes": manifold["episodes"],
        "healthy_reference_rows": manifold["rows"],
        "healthy_rows_per_episode": args.healthy_rows_per_episode,
        "healthy_manifold_features": list(MANIFOLD_FEATURES),
        "ledoit_wolf_shrinkage": manifold["shrinkage"],
        "score_names": list(score_names),
        "evaluation": "same task, scene, and query; event episode versus no-event controls",
        "rows": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    headline = [
        row
        for row in rows
        if row["lead"] == -2
        and row["score"]
        in {"r1_commitment", "r2_flow_instability", "r3_loop_graph_evidence", "r3_static_authority_loss", "r3_static_lockin", "r4_healthy_energy"}
    ]
    print(json.dumps({"manifold": {k: summary[k] for k in ("healthy_reference_episodes", "healthy_reference_rows", "ledoit_wolf_shrinkage")}, "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
