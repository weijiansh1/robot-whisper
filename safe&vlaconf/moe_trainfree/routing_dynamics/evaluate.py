"""Fixed retrospective, same-query checks of the new routing feature representation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import zarr

from encoder import (EncoderConfig, FEATURE_NAMES, LAYER_FEATURE_NAMES, LAYER_NAMES,
                     ReadoutEncoder, RoutingDynamicsEncoder, encode_readouts)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from run_analysis import BASE, ROOT, bootstrap_indices, digest, write_json


S05 = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
FIELDS = ("decoupling", "path_decoupling", "acceleration_log_ratio", "mobility_log_ratio",
          "token_diversity_contraction", "layer_agreement", "recent_decoupling",
          "persistent_decoupling", "decoupling_occupancy", "routing_recovery")


def pair_auc(labels, values):
    positive, negative = values[labels], np.sort(values[~labels])
    if not len(positive) or not len(negative):
        return np.nan
    ranks = np.searchsorted(negative, positive, side="left") + np.searchsorted(negative, positive, side="right")
    return float(ranks.sum() / (2 * len(positive) * len(negative)))


def interval(values, suites):
    values = np.asarray(values, float)
    if not len(values):
        return dict(estimate=np.nan, lo=np.nan, hi=np.nan, tasks=0)
    draws = bootstrap_indices(np.arange(len(values)), suites)
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(estimate=float(values.mean()), lo=float(low), hi=float(high), tasks=len(values))


def verify_encodings(frame, readouts, valid, features, layers, manifest):
    readout_checks, raw_checks, maximum_error = 0, 0, 0.
    for i in manifest["representative_replay_rows"]:
        row = frame.iloc[i]
        live = ReadoutEncoder()
        for q in range(row.length):
            result = live.update(readouts[i, q])
            np.testing.assert_allclose(result.values, features[i, q], atol=1e-6, rtol=1e-6, equal_nan=True)
            np.testing.assert_allclose(result.per_layer, layers[i, q], atol=1e-6, rtol=1e-6, equal_nan=True)
            readout_checks += 1
        if row.episode not in (77, 96, 174, 190):
            continue
        run = ROOT / "VLA_MUI_HUB" / row.source
        summaries = sorted(json.loads((run / "client/summaries.json").read_text()), key=lambda x: x["episode_index"])
        start = sum(s["inference_calls"] for s in summaries[:row.episode])
        raw = zarr.open_group(str(run / "server/routes.zarr"), mode="r")["hb_router_probs"][start:start + row.length]
        live = RoutingDynamicsEncoder()
        for q, query in enumerate(raw):
            result = live.update(query)
            expected = features[i, q]
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(np.isfinite(result.values), finite)
            # Historical caches use alternate finite-precision Hellinger/entropy reductions.
            if finite.any():
                error = float(np.max(np.abs(result.values[finite] - expected[finite])))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(result.values[finite], expected[finite], atol=2e-3, rtol=2e-3)
            raw_checks += 1
    return dict(stream_readout_checks=readout_checks, raw_query_checks=raw_checks,
                maximum_raw_cache_encoding_error=maximum_error)


def strata(frame, features, old_scores):
    records, paired_states = [], []
    indices = [FEATURE_NAMES.index(name) for name in FIELDS]
    for (cohort, task), part in frame.groupby(["cohort", "task"], sort=True):
        rows = part.index.to_numpy()
        labels = part.failure.to_numpy(bool)
        states = [np.flatnonzero(part.init_state_id.eq(state).to_numpy()) for state in part.init_state_id.unique()]
        for q in range(7, features.shape[1]):
            for name, column in zip(FIELDS, indices):
                values = features[rows, q, column].astype(float)
                available = np.isfinite(values)
                if not (labels & available).any() or not (~labels & available).any():
                    continue
                score = pair_auc(labels[available], values[available])
                reference_auc = np.nan
                if cohort == "B":
                    reference = old_scores[rows, q]
                    assert np.isfinite(reference[available]).all()
                    reference_auc = pair_auc(labels[available], reference[available])
                records.append(dict(cohort=cohort, suite=part.suite.iloc[0], task=task, query=q, feature=name,
                                    auc=score, v82_auc=reference_auc, delta=score - reference_auc,
                                    failures=int((labels & available).sum()), successes=int((~labels & available).sum()),
                                    nonzero_failure=int((labels & available & (values > 0)).sum()),
                                    nonzero_success=int((~labels & available & (values > 0)).sum())))
                if name != "decoupling":
                    continue
                within, references, failed_n, success_n = [], [], 0, 0
                for state_rows in states:
                    state_rows = state_rows[available[state_rows]]
                    y = labels[state_rows]
                    if not y.any() or y.all():
                        continue
                    within.append(pair_auc(y, values[state_rows]))
                    if cohort == "B":
                        references.append(pair_auc(y, reference[state_rows]))
                    failed_n += int(y.sum())
                    success_n += int((~y).sum())
                if within:
                    old = float(np.mean(references)) if references else np.nan
                    paired_states.append(dict(cohort=cohort, suite=part.suite.iloc[0], task=task, query=q,
                                              feature=name, auc=float(np.mean(within)), v82_auc=old,
                                              delta=float(np.mean(within)) - old, states=len(within),
                                              failures=failed_n, successes=success_n))
        print(f"CHECKED {cohort} {task}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(paired_states)


def summarize(table, level):
    tasks, summaries = [], []
    for window, selected in (("q7_13", table.loc[table["query"].between(7, 13)]), ("all_comparable_queries", table)):
        grouped = selected.groupby(["cohort", "suite", "task", "feature"], as_index=False).agg(
            auc=("auc", "mean"), v82_auc=("v82_auc", "mean"), delta=("delta", "mean"),
            queries=("query", "nunique"), first_query=("query", "min"), last_query=("query", "max"),
            failure_observations=("failures", "sum"), success_observations=("successes", "sum"))
        grouped["window"], grouped["level"] = window, level
        tasks.append(grouped)
        for (cohort, feature), part in grouped.groupby(["cohort", "feature"]):
            scopes = [("all", part), ("excluding_S05", part.loc[part.task.ne(S05)])] + list(part.groupby("suite"))
            for scope, current in scopes:
                for metric in ("auc", "v82_auc", "delta"):
                    available = current.loc[current[metric].notna()].sort_values(["suite", "task"])
                    if len(available):
                        summaries.append(dict(level=level, cohort=cohort, window=window, feature=feature,
                                              scope=scope, metric=metric, **interval(available[metric], available.suite)))
    return pd.concat(tasks, ignore_index=True), pd.DataFrame(summaries)


def plots(output, frame, features, task_table):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    selected = task_table.loc[task_table.cohort.eq("B") & task_table.feature.eq("decoupling")
                              & task_table.window.eq("q7_13")].sort_values(["suite", "task"])
    names, counts = [], {}
    for row in selected.itertuples():
        short = {"libero_goal": "G", "libero_long": "L", "libero_object": "O", "libero_spatial": "S"}[row.suite]
        counts[short] = counts.get(short, 0) + 1
        names.append(f"{short}{counts[short]:02d}")
    pd.DataFrame(dict(plot_id=names, task=selected.task.to_list())).to_csv(output / "plot_task_map.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(np.arange(len(selected)), selected.auc, "o", color="#167e76", label="Decoupling feature")
    ax.plot(np.arange(len(selected)), selected.v82_auc, "x", color="#9c5976", label="Existing v8.2 score, same observations")
    ax.axhline(.5, color="#777777", linestyle=":")
    ax.set_xticks(np.arange(len(names)), names, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Within-task/query AUROC, averaged over q7..q13")
    ax.set_title("B: retrospective cross-task check; no feature fitting or threshold selection")
    ax.legend(loc="best")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cross_task_check.{extension}", dpi=170, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    examples = [(77, "Failure, detected", "#b94d48"), (96, "Failure, missed", "#a87425"),
                (174, "Success, alarm", "#167e76"), (190, "Success, no alarm", "#715887")]
    for ax, name in zip(axes.flat, ("mobility_log_ratio", "acceleration_log_ratio", "decoupling", "routing_recovery")):
        for episode, label, color in examples:
            row = frame.loc[frame.cohort.eq("B") & frame.task.eq(S05) & frame.episode.eq(episode)].iloc[0]
            q = np.arange(row.length)
            ax.plot(q, features[row.name, :row.length, FEATURE_NAMES.index(name)], color=color, label=f"{episode}: {label}")
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Observed query")
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("S05 examples: a routing state feature is not an irreversible-failure label")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"s05_encoded_states.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "routing_dynamics_20260908")
    args = parser.parse_args()
    output = args.input.resolve()
    if (output / "features.npz").exists():
        raise FileExistsError("analysis outputs already exist")
    manifest = json.loads((output / "input_verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    contract = json.loads((output / "execution_contract.json").read_text())
    assert contract["encoder_sha256"] == digest(HERE / "encoder.py")
    assert contract["protocol_sha256"] == digest(HERE / "PROTOCOL_ZH.md")
    write_json(output / "analysis_contract.json", dict(
        fields=FIELDS, primary="decoupling", raw_value_orientation=True,
        comparisons="identical task/query and observed episodes for each new feature and existing v8.2",
        config=asdict(EncoderConfig()), source_sha256=digest(__file__), no_new_detector=True))
    frame = pd.read_csv(output / "index.csv")
    frame["cohort"] = frame.run_id.map({"right-50x8-20260903": "A", "right-50x8b-20260903": "B"})
    with np.load(output / "readouts.npz", allow_pickle=False) as z:
        readouts, valid = z["values"], z["valid"]
    features = np.full((*valid.shape, len(FEATURE_NAMES)), np.nan, np.float32)
    layers = np.full((*valid.shape, 8, 3), np.nan, np.float32)
    for start in range(0, len(frame), 1000):
        values, layer = encode_readouts(readouts[start:start + 1000], valid[start:start + 1000])
        features[start:start + 1000], layers[start:start + 1000] = values, layer
    checks = verify_encodings(frame, readouts, valid, features, layers, manifest)
    np.savez_compressed(output / "features.npz", features=features, per_layer=layers, valid=valid,
                        global_rows=frame.global_row.to_numpy(), names=np.asarray(FEATURE_NAMES),
                        layer_names=np.asarray(LAYER_NAMES), layer_feature_names=np.asarray(LAYER_FEATURE_NAMES))
    del layers, readouts
    prediction_path = BASE / "v82_validation_20260908/crossfit_predictions.npz"
    prior = json.loads((BASE / "v82_validation_20260908/analysis_verification.json").read_text())
    assert digest(prediction_path) == prior["artifacts"]["crossfit_predictions.npz"]
    old_scores = np.full(valid.shape, np.nan)
    with np.load(prediction_path, allow_pickle=False) as z:
        position = list(z["methods"]).index("v82")
        old_scores[z["global_rows"]] = z["scores"][position]
    observations, same_initial = strata(frame, features, old_scores)
    observations.to_csv(output / "query_comparisons.csv", index=False)
    same_initial.to_csv(output / "same_initial_comparisons.csv", index=False)
    task_table, summary = summarize(observations, "same_task_query")
    state_tasks, state_summary = summarize(same_initial, "same_task_init_query")
    pd.concat((task_table, state_tasks), ignore_index=True).to_csv(output / "task_summary.csv", index=False)
    pd.concat((summary, state_summary), ignore_index=True).to_csv(output / "summary.csv", index=False)
    checked_auc = 0
    for row in observations.loc[observations.feature.eq("decoupling")].itertuples():
        selected = frame.loc[frame.task.eq(row.task) & frame.cohort.eq(row.cohort)]
        x = features[selected.index, row.query, FEATURE_NAMES.index("decoupling")].astype(float)
        y, present = selected.failure.to_numpy(bool), np.isfinite(x)
        expected = mannwhitneyu(x[y & present], x[~y & present]).statistic / ((y & present).sum() * (~y & present).sum())
        np.testing.assert_allclose(row.auc, expected, atol=1e-12, rtol=0)
        checked_auc += 1
    s05 = frame.loc[frame.cohort.eq("B") & frame.task.eq(S05)]
    diagnostic = []
    for i, row in s05.iterrows():
        for q in np.flatnonzero(valid[i]):
            diagnostic.append(dict(episode=row.episode, failure=row.failure, query=q,
                                   **dict(zip(FEATURE_NAMES, features[i, q].tolist()))))
    pd.DataFrame(diagnostic).to_csv(output / "s05_features.csv", index=False)
    plots(output, frame, features, task_table)
    write_json(output / "verification.json", dict(
        all_checks_passed=True, episodes=len(frame), queries=int(valid.sum()), features=len(FEATURE_NAMES),
        **checks, independent_auc_checks=checked_auc, encoder_sha256=digest(HERE / "encoder.py"),
        analysis_sha256=digest(__file__), reference_predictions_sha256=digest(prediction_path),
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(summary.loc[summary.feature.eq("decoupling") & summary.scope.isin(["all", "excluding_S05"])].to_string(index=False))
    print("ENCODING AND CHECKS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
