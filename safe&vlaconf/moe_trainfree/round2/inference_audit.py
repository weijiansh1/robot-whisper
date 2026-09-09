"""Supplemental null audit, preserving the shared noise-seed dependence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

from paired_core import SEED, digest, write_json

HERE = Path(__file__).resolve().parent


def shared_seed_null(frame, scores, draws=5000):
    seeds = np.sort(frame.flow_noise_seed.unique())
    rng = np.random.default_rng(SEED)
    permutations = np.argsort(rng.random((draws, len(seeds))), axis=1)
    null = np.full((draws, scores.shape[1]), 0.5)
    observed = np.full(scores.shape[1], 0.5)
    tasks = sorted(frame.task.unique())
    mixed_groups = 0
    for task in tasks:
        groups = [np.asarray(rows) for _, rows in frame[frame.task == task].groupby("init_state_id").groups.items()
                  if frame.loc[rows, "failure"].nunique() == 2]
        mixed_groups += len(groups)
        if not groups:
            raise ValueError("task has no mixed-outcome initial states")
        for rows in groups:
            rows = rows[np.argsort(frame.loc[rows, "flow_noise_seed"].to_numpy())]
            if not np.array_equal(frame.loc[rows, "flow_noise_seed"], seeds):
                raise ValueError("shared-seed permutation requires a complete seed grid")
            truth = frame.loc[rows, "failure"].to_numpy(float)
            p, n = truth.sum(), len(truth) - truth.sum()
            rank = rankdata(scores[rows], axis=0, method="average")
            weight = (rank - (len(rows) + 1) / 2) / (p * n * len(groups) * len(tasks))
            observed += truth @ weight
            # The same seed permutation is applied to all tasks and initial states.
            null += truth[permutations] @ weight
    return observed, null, mixed_groups


def independent_random_controls(frame, draws=1000):
    episodes, row_index = np.unique(frame.episode, return_inverse=True)
    values = np.random.default_rng(SEED + 12345).random((len(episodes), draws))[row_index]
    result = np.full(draws, 0.5)
    tasks = sorted(frame.task.unique())
    for task in tasks:
        groups = [np.asarray(rows) for _, rows in frame[frame.task == task].groupby("init_state_id").groups.items()
                  if frame.loc[rows, "failure"].nunique() == 2]
        for rows in groups:
            y = frame.loc[rows, "failure"].to_numpy(float)
            rank = rankdata(values[rows], axis=0)
            result += y @ (rank - (len(rows) + 1) / 2) / (
                y.sum() * (len(rows) - y.sum()) * len(groups) * len(tasks))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round2")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    ranking = pd.read_csv(output / "ranking_metrics.csv")
    tables = []
    group_counts, random_checks = {}, {}
    for dataset in ("q0", "q34"):
        path = output / f"{dataset}_init_state_id.npz"
        if digest(path) != manifest["artifacts"][path.name]:
            raise ValueError("changed sealed predictions")
        frame = pd.read_csv(output / f"{dataset}_label_alignment.csv")
        with np.load(path, allow_pickle=False) as archive:
            scores, methods = archive["scores"], archive["method_names"].astype(str)
        observed, null, mixed = shared_seed_null(frame, scores)
        group_counts[dataset] = mixed
        random_auc = independent_random_controls(frame)
        original_random = float(observed[list(methods).index("random")])
        random_checks[dataset] = {
            "draws": len(random_auc), "original_random_auc": original_random,
            "independent_draw_median": float(np.median(random_auc)),
            "independent_draw_95_range": np.quantile(random_auc, [0.025, 0.975]),
            "independent_draw_min_max": [float(random_auc.min()), float(random_auc.max())],
            "draws_at_least_original": int((random_auc >= original_random).sum()),
        }
        expected = ranking[(ranking.dataset == dataset) & (ranking.setting == "init_state_id") &
                           (ranking.scope == "all")].set_index("method").loc[methods, "within_init_macro_auc"]
        if not np.allclose(observed, expected):
            raise ValueError("rank-sum and evaluation AUC disagree")
        sd = null.std(axis=0)
        supported = sd > 1e-12
        z = np.divide(observed - 0.5, sd, out=np.zeros_like(sd), where=supported)
        z_null = np.divide(null - 0.5, sd, out=np.zeros_like(null), where=supported)
        maximum = z_null.max(axis=1)
        low, high = np.quantile(null, [0.025, 0.975], axis=0)
        for m, name in enumerate(methods):
            tables.append({"dataset": dataset, "method": name, "observed_auc": observed[m],
                           "null_lo": low[m], "null_hi": high[m], "mixed_init_groups": mixed,
                           "permutation_p_one_sided": (1 + np.count_nonzero(null[:, m] >= observed[m] - 1e-12)) / (len(null) + 1),
                           "family_maxT_p_one_sided": (1 + np.count_nonzero(maximum >= z[m] - 1e-12)) / (len(null) + 1)})
        np.savez_compressed(output / f"{dataset}_shared_seed_null.npz", null_auc=null,
                            method_names=methods, observed_auc=observed)
    pd.DataFrame(tables).to_csv(output / "shared_seed_null_metrics.csv", index=False)

    frame = pd.read_csv(output / "pilot_label_alignment.csv")
    with np.load(output / "pilot_predictions.npz", allow_pickle=False) as archive:
        scores, names = archive["scores"], archive["method_names"].astype(str)
    intervals = []
    for lead in (-4, -2):
        subset = frame[(frame.lead == lead) & frame.qualified]
        differences = []
        for _, rows in subset.groupby("pair_id").groups.items():
            y = frame.loc[rows, "failure"].to_numpy(bool)
            delta = scores[np.asarray(rows)[y][0]] - scores[np.asarray(rows)[~y][0]]
            differences.append(delta)
        delta = np.stack(differences)
        for m, name in enumerate(names):
            wins, ties, total = int((delta[:, m] > 0).sum()), int((delta[:, m] == 0).sum()), len(delta)
            if ties:
                low = high = np.nan
            else:
                low = 0.0 if wins == 0 else beta.ppf(0.025, wins, total - wins + 1)
                high = 1.0 if wins == total else beta.ppf(0.975, wins + 1, total - wins)
            intervals.append({"lead": lead, "method": name, "wins": wins, "ties": ties, "pairs": total,
                              "binomial_exact_lo": low, "binomial_exact_hi": high})
    pd.DataFrame(intervals).to_csv(output / "pilot_pair_count_intervals.csv", index=False)
    write_json(output / "inference_audit.json", {
        "added_after_initial_round2_metrics": True, "predictions_changed": False,
        "null_draws": 5000, "permutation_unit": "shared global noise-seed permutation across all task/init groups",
        "mixed_groups": group_counts,
        "random_control_repetition": random_checks,
        "scope": "conditional diagnostic on historically explored data, not prospective confirmation",
        "pilot_interval_caveat": "Exact binomial intervals assume exchangeable independent sampled pairs; this selected pilot is not population-representative.",
        "source_sha256": digest(Path(__file__)),
    })
    print(pd.DataFrame(tables).groupby("dataset").agg(
        candidates=("method", "count"), maxT_below_005=("family_maxT_p_one_sided", lambda x: int((x < 0.05).sum()))
    ).to_string())


if __name__ == "__main__":
    main()
