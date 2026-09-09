"""Audit frozen alarms on the shared external cohort without refitting scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "v8_alarm_positions"
HISTORICAL = {
    "v8": ("v8_full_corpus", "v8"),
    "v81": ("v81", "v8.1"),
    "v82": ("v82", "v8.2"),
    "v83": ("v83", "v8.3"),
}
GEOMETRIC = {
    "knn20": "knn20_a05_first_alarm_query",
    "c4_a05": "c4_assigned_radius_a05_first_alarm_query",
    "c4_a02": "c4_assigned_radius_a02_first_alarm_query",
}
METHODS = tuple(HISTORICAL) + tuple(GEOMETRIC)
PLOTTED = ("v82", "knn20", "c4_a05", "c4_a02")
LABELS = dict(v82="v8.2", knn20="kNN-20 (5%)", c4_a05="C4 (5%)", c4_a02="C4 (2%)")
COLORS = dict(v82="#168579", knn20="#b94f3c", c4_a05="#376da5", c4_a02="#737373")
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
POSITION_BINS = ("0-25%", "25-50%", "50-75%", "75-100%", "no_alarm")
BIN_COLORS = ("#63bfa3", "#77a9d2", "#e5bc65", "#c8716a", "#dedede")
CHUNK = 10


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantiles(values, prefix):
    values = np.asarray(values)
    q = np.quantile(values, [.25, .5, .75]) if len(values) else np.full(3, np.nan)
    return {f"{prefix}_{name}": float(value) for name, value in zip(("p25", "median", "p75"), q)}


def load_positions():
    inputs = {}

    def checked(path):
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return path

    source = checked(RESULTS / "round11_kmeans/trajectory_results.csv")
    columns = ["global_row", "source", "run_id", "task", "suite", "episode", "init_state_id", "noise_seed",
               "length", "actual_action_steps", "failure", "primary_failure_reason", "drop_goal_release",
               *GEOMETRIC.values()]
    master = pd.read_csv(source, usecols=columns).set_index(["run_id", "task", "episode"], drop=False)
    assert master.index.is_unique
    meta_path = checked(ROOT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz")
    with np.load(meta_path, allow_pickle=False) as meta:
        index = pd.MultiIndex.from_arrays([
            np.repeat(str(meta["run_id"]), len(meta["episode"])),
            meta["task_names"][meta["task_index"]], meta["episode"],
        ])
        frame = master.loc[index].reset_index(drop=True)
        for old, new in (("length", "length"), ("init_state_id", "init_state_id"), ("flow_noise_seed", "noise_seed")):
            np.testing.assert_array_equal(meta[old], frame[new])
    label_path = checked(ROOT / "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv")
    labels = pd.read_csv(label_path).set_index(["task", "episode"])
    np.testing.assert_array_equal(labels.loc[index.droplevel(0), "original_failure"], frame.failure)
    assert len(frame) == frame.global_row.nunique() == 15600 and frame.failure.sum() == 564
    np.testing.assert_array_equal(frame.length, (frame.actual_action_steps + CHUNK - 1) // CHUNK)

    audit = []
    for method, (archive, version) in HISTORICAL.items():
        path = checked(ROOT / f"moe-v8-0906/results/{archive}_alarms.npz")
        with np.load(path, allow_pickle=False) as data:
            original = data[f"external_8b|{version}"].astype(int)
        assert original.shape == (len(frame),)
        valid = (original >= 0) & (original < frame.length.to_numpy())
        frame[f"{method}_original_q"] = original
        frame[f"{method}_q"] = np.where(valid, original, -1)
        audit.append(dict(method=method, removed_outside_execution=int(((original >= 0) & ~valid).sum())))
    for method, column in GEOMETRIC.items():
        frame[f"{method}_q"] = frame[column].to_numpy(int)

    successes = frame.loc[~frame.failure]
    references = successes.groupby("task").actual_action_steps.agg(
        success_count="size", success_end_min="min", success_end_median="median", success_end_max="max")
    references["success_end_p95"] = successes.groupby("task").actual_action_steps.quantile(.95)
    for column in references.columns:
        frame[column] = frame.task.map(references[column])
    assert frame.success_count.notna().all()
    for method in METHODS:
        q = frame[f"{method}_q"].to_numpy()
        fired = q >= 0
        assert ((q == -1) | ((q >= 0) & (q < frame.length))).all()
        actions = CHUNK * q
        frame[f"{method}_actions"] = np.where(fired, actions, np.nan)
        frame[f"{method}_position"] = np.where(fired, actions / frame.actual_action_steps, np.nan)
        frame[f"{method}_remaining_actions"] = np.where(fired, frame.actual_action_steps - actions, np.nan)
        frame[f"{method}_event_delay_q"] = np.where(fired, q - frame.drop_goal_release, np.nan)
        frame[f"{method}_after_last_success"] = fired & (actions >= frame.success_end_max)
    frame = frame.drop(columns=list(GEOMETRIC.values()))
    return frame, references, audit, inputs


def summarize(frame):
    summaries, bins, queries, paired = [], [], [], []
    scopes = [("all", frame)] + [(suite, frame.loc[frame.suite.eq(suite)]) for suite in SUITES]
    for suite, scoped in scopes:
        for method in METHODS:
            for failure in (True, False):
                part = scoped.loc[scoped.failure.eq(failure)]
                q = part[f"{method}_q"].to_numpy()
                fired = q >= 0
                hit = part.loc[fired]
                position = hit[f"{method}_position"].to_numpy()
                event = part.drop_goal_release.to_numpy()
                eligible = part.failure.to_numpy() & np.isfinite(event)
                delay = q[eligible & fired] - event[eligible & fired]
                meta = dict(suite=suite, method=method, failure=failure, episodes=len(part))
                summary = dict(meta, alarms=int(fired.sum()), no_alarm=int((~fired).sum()),
                    **quantiles(q[fired], "query"), **quantiles(position, "position"),
                    **quantiles(hit[f"{method}_remaining_actions"], "remaining_actions"),
                    after_last_success=int(hit[f"{method}_after_last_success"].sum()),
                    after_median_success=int(hit[f"{method}_actions"].ge(hit.success_end_median).sum()),
                    event_episodes=int(eligible.sum()), event_before=int((eligible & fired & (q < event)).sum()),
                    event_same=int((eligible & fired & (q == event)).sum()),
                    event_after=int((eligible & fired & (q > event)).sum()),
                    event_missed=int((eligible & ~fired).sum()), **quantiles(delay, "event_delay_q"))
                summaries.append(summary)
                counts = [int(((position >= i / 4) & (position < (i + 1) / 4)).sum()) for i in range(4)]
                counts.append(int((~fired).sum()))
                assert sum(counts) == len(part)
                for label, count in zip(POSITION_BINS, counts):
                    bins.append(dict(meta, bin=label, count=count, fraction=count / len(part)))
                for query in range(int(scoped.length.max())):
                    running = part.length.to_numpy() > query
                    at_risk = running & ((q < 0) | (q >= query))
                    cumulative = int((fired & (q <= query)).sum())
                    queries.append(dict(meta, query=query, executed_actions=CHUNK * query,
                        new_alarms=int((q == query).sum()), cumulative_alarms=cumulative,
                        cumulative_fraction=cumulative / len(part), running=int(running.sum()),
                        running_not_previously_alarm=int(at_risk.sum())))
            for failure in (True, False):
                part = scoped.loc[scoped.failure.eq(failure)]
                baseline, other = part.v82_q.to_numpy(), part[f"{method}_q"].to_numpy()
                both = (baseline >= 0) & (other >= 0)
                delta = other[both] - baseline[both]
                paired.append(dict(suite=suite, method=method, baseline="v82", failure=failure,
                    episodes=len(part), both=int(both.sum()), only_v82=int(((baseline >= 0) & (other < 0)).sum()),
                    only_other=int(((baseline < 0) & (other >= 0)).sum()),
                    neither=int(((baseline < 0) & (other < 0)).sum()),
                    other_earlier=int((delta < 0).sum()), same=int((delta == 0).sum()),
                    other_later=int((delta > 0).sum()), **quantiles(delta, "other_minus_v82_q")))
    return {"summary": pd.DataFrame(summaries), "position_bins": pd.DataFrame(bins),
            "query_distribution": pd.DataFrame(queries), "paired_with_v82": pd.DataFrame(paired)}


def figures(output, tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    bins = tables["position_bins"]
    for ax, suite in zip(axes.flat, SUITES):
        scoped = bins.loc[bins.suite.eq(suite) & bins.failure]
        left = np.zeros(len(PLOTTED))
        for label, color in zip(POSITION_BINS, BIN_COLORS):
            part = scoped.loc[scoped.bin.eq(label)].set_index("method").loc[list(PLOTTED)]
            width = part.fraction.to_numpy() * 100
            ax.barh(np.arange(len(PLOTTED)), width, left=left, color=color, label=label, height=.6)
            for i, (w, count) in enumerate(zip(width, part["count"])):
                if w >= 7:
                    ax.text(left[i] + w / 2, i, str(count), ha="center", va="center", fontsize=9)
            left += width
        assert np.allclose(left, 100)
        n = int(scoped.episodes.iloc[0])
        ax.set(title=f"{suite.removeprefix('libero_').title()} | {n} failed trajectories",
               yticks=np.arange(len(PLOTTED)), yticklabels=[LABELS[m] for m in PLOTTED],
               xlabel="Fraction of all failed trajectories", xlim=(0, 100))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter())
    fig.suptitle("First alarm within actual trajectory duration | shared external cohort\n"
                 "Positions use executed actions / final actions; unalarmed failures remain in the denominator")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=5,
               title="Position at first alarm (retrospective); numbers inside bars are trajectory counts", frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"failure_positions.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    data = tables["query_distribution"]
    for col, suite in enumerate(SUITES):
        for row, failure in enumerate((True, False)):
            ax = axes[row, col]
            scoped = data.loc[data.suite.eq(suite) & data.failure.eq(failure)]
            for method in PLOTTED:
                part = scoped.loc[scoped.method.eq(method)]
                ax.step(part.executed_actions, part.cumulative_fraction * 100, where="post",
                        label=LABELS[method], color=COLORS[method], linestyle="--" if method == "c4_a02" else "-")
            count = int(scoped.episodes.iloc[0])
            ax.set(title=f"{suite.removeprefix('libero_').title()} | n={count}",
                   xlabel="Executed actions", ylabel="Failure recall (%)" if failure else "Success false alarms (%)",
                   xlim=(0, int(scoped.executed_actions.max())))
            ax.set_ylim(0, 102 if failure else max(1, 105 * scoped.cumulative_fraction.max()))
            ax.grid(alpha=.2)
    fig.suptitle("Cumulative first alarms | all failures / successes retained\n"
                 "Each suite has its own action horizon; lower panels use different vertical scales")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"timing_curves.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    frame, references, audit, inputs = load_positions()
    tables = summarize(frame)
    summary = tables["summary"].loc[lambda x: x.suite.eq("all")].set_index(["method", "failure"])
    for method, (tp, fp) in {"v8": (460, 90), "v81": (495, 105), "v82": (475, 98), "v83": (478, 95),
                             "knn20": (499, 1403), "c4_a05": (485, 632), "c4_a02": (369, 223)}.items():
        assert int(summary.loc[(method, True), "alarms"]) == tp
        assert int(summary.loc[(method, False), "alarms"]) == fp
    for row in tables["paired_with_v82"].itertuples():
        assert row.both + row.only_v82 + row.only_other + row.neither == row.episodes
        assert row.other_earlier + row.same + row.other_later == row.both
    for row in tables["summary"].itertuples():
        assert row.event_before + row.event_same + row.event_after + row.event_missed == row.event_episodes
        assert row.alarms + row.no_alarm == row.episodes
    output.mkdir(parents=True)
    frame.to_csv(output / "episode_positions.csv", index=False)
    references.to_csv(output / "task_success_endpoints.csv")
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    figures(output, tables)
    manifest = dict(passed=True, episodes=len(frame), failures=int(frame.failure.sum()),
        methods=METHODS, original_alarms_audit=audit, inputs=inputs, source_sha256=digest(Path(__file__)),
        alarm_validity="0 <= query < actual query count", calibration="unchanged frozen profiles",
        temporal_position="10 * query / actual_action_steps; descriptive, not an online input",
        success_endpoint_reference="same-task successful B trajectories; retrospective diagnosis only",
        quantile_population="alarmed trajectories only; no-alarm counts also reported",
        artifacts={path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.loc[(slice(None), True), ["episodes", "alarms", "query_median", "position_median",
        "remaining_actions_median", "after_last_success", "event_delay_q_median"]].to_string())
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
