"""Locate routing-based first alarms within each task's configured action budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_analysis import BASE, RUN_B, digest, write_json


CAPS = dict(libero_goal=300, libero_long=520, libero_object=280, libero_spatial=220)
METHODS = ("v7", "v8", "v82")
SOURCE_NAMES = ("v7", "v8_fixed", "v82")
COLORS = ("#777777", "#376da5", "#14836b")
LABELS = ("v7", "v8", "v8.2")
CUTOFFS = (25, 50, 60, 70, 75, 80, 90, 100)


def describe(q, caps):
    fired = q >= 0
    pct = 1000 * q[fired] / caps[fired]
    quartiles = np.quantile(pct, [.25, .5, .75]) if len(pct) else [np.nan] * 3
    return dict(episodes=len(q), alarms=int(fired.sum()), no_alarm=int((~fired).sum()),
                alarm_rate=float(fired.mean()) if len(q) else np.nan,
                p25_pct=quartiles[0], median_pct=quartiles[1], p75_pct=quartiles[2],
                median_query=float(np.median(q[fired])) if fired.any() else np.nan,
                at_or_after_75pct=int((fired & (1000 * q >= 75 * caps)).sum()))


def task_label(task, suite):
    label = task.split("/", 1)[1]
    label = re.sub(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", label)
    if suite == "libero_object":
        label = label.removeprefix("pick_up_the_").removesuffix("_and_place_it_in_the_basket") + "_to_basket"
    elif suite == "libero_spatial":
        label = "bowl_" + label.removeprefix("pick_up_the_black_bowl_").removesuffix("_and_place_it_on_the_plate")
    return textwrap.shorten(label.replace("_", " "), width=62, placeholder="...")


def render_task_table(output, task_map, summary):
    records = summary.loc[summary.family.eq("frozen") & summary.level.eq("task")]
    lines = ["# First Alarm Positions by Task", "",
             "Position = executed actions / configured task budget. All values describe the original frozen routing guards.",
             "Cells show the median position and detected failures / all failures. IQR is the middle 50% of v8.2 failure alarms.",
             "v8.2 FP cells show median position and false alarms / all successful episodes. Missing medians mean no alarm.", ""]
    for suite, group in task_map.groupby("suite", sort=True):
        lines.extend(["## " + suite, "", "| ID | Task | v7 | v8 | v8.2 | v8.2 IQR | v8.2 FP |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for item in group.itertuples():
            part = records.loc[records.scope.eq(item.task)]
            cells = []
            for method in METHODS:
                row = part.loc[part.method.eq(method) & part.outcome.eq("failure")].iloc[0]
                median = f"{row.median_pct:.1f}%" if row.alarms else "NA"
                cells.append(f"{median} ({row.alarms}/{row.episodes})")
            row = part.loc[part.method.eq("v82") & part.outcome.eq("failure")].iloc[0]
            iqr = f"{row.p25_pct:.1f}-{row.p75_pct:.1f}%" if row.alarms else "NA"
            fp = part.loc[part.method.eq("v82") & part.outcome.eq("success")].iloc[0]
            fp_median = f"{fp.median_pct:.1f}%" if fp.alarms else "NA"
            lines.append("| " + " | ".join([item.task_id, item.label, *cells, iqr,
                                              f"{fp_median} ({fp.alarms}/{fp.episodes})"]) + " |")
        lines.append("")
    lines.extend(["Full task identifiers: [task_map.csv](task_map.csv).", "",
                  "![Failure alarm medians and interquartile ranges](task_failure_positions.png)", ""])
    (output / "TASKS.md").write_text("\n".join(lines))


def plot_tasks(output, task_map, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    selected = summary.loc[summary.family.eq("frozen") & summary.level.eq("task") & summary.outcome.eq("failure")]
    fig, axes = plt.subplots(2, 2, figsize=(21, 14))
    for ax, (suite, current) in zip(axes.flat, task_map.groupby("suite", sort=True)):
        current = current.reset_index(drop=True)
        y = np.arange(len(current))
        for method, color, label, offset in zip(METHODS, COLORS, LABELS, (-.21, 0, .21)):
            part = selected.loc[selected.method.eq(method)].set_index("scope").loc[current.task]
            good = part.alarms.to_numpy() > 0
            med = part.median_pct.to_numpy()[good]
            lo, hi = part.p25_pct.to_numpy()[good], part.p75_pct.to_numpy()[good]
            ax.errorbar(med, y[good] + offset, xerr=np.stack((med - lo, hi - med)),
                        fmt="o", markersize=4, capsize=2, elinewidth=1.1, color=color, label=label)
        v82 = selected.loc[selected.method.eq("v82")].set_index("scope").loc[current.task]
        for i, row in enumerate(v82.itertuples()):
            ax.text(1.02, i, f"{row.alarms}/{row.episodes}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=9)
        ax.text(1.02, 1.03, "v8.2\nTP / failures", transform=ax.transAxes, ha="left", fontsize=9)
        ax.set_yticks(y, [row.task_id + "  " + textwrap.fill(row.label, 35) for row in current.itertuples()], fontsize=9)
        ax.set_ylim(len(y) - .5, -.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(axis="x", color="#dddddd", linewidth=.7)
        ax.set_title(suite.removeprefix("libero_").capitalize() + f" ({CAPS[suite]} action budget)", pad=18)
        ax.set_xlabel("Task budget used at first alarm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .967))
    fig.suptitle("Frozen routing guards: first failure alarm positions by task", y=.994, fontsize=15)
    fig.text(.5, .014, "Dots: median. Lines: middle 50% of alarms. Blank: no detected failure. Counts retain all failures.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(.005, .04, .965, .93), h_pad=3, w_pad=5)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("task_failure_positions." + suffix), dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "v82_validation_20260908")
    parser.add_argument("--output", type=Path, default=BASE / "v82_alarm_budget_positions_20260908")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_contract.json", dict(
        task_action_budgets=CAPS, methods=METHODS, primary_family="frozen",
        supplementary_family="standardized_task_init_1pct", cutoffs_pct=CUTOFFS,
        location_definition="100 * 10 * first_query / configured_task_action_budget",
        duration_used_as_detection_score=False, task_budget_used_only_as_reporting_coordinate=True,
        thresholds_refit=False, medians_condition_on_alarm=True, missing_alarms_retained_in_counts=True,
        source_sha256=digest(__file__),
    ))
    old = json.loads((source / "analysis_verification.json").read_text())
    names = ("index.csv", "frozen_first_alarms.csv", "crossfit_predictions.npz", "alarm_metrics.csv", "cumulative_curves.csv")
    hashes = {name: digest(source / name) for name in names}
    for name, value in hashes.items():
        assert value == old["artifacts"][name], name
    frame = pd.read_csv(source / "index.csv")
    b = frame.loc[frame.run_id.eq(RUN_B)].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564 and not b.global_row.duplicated().any()
    frozen = pd.read_csv(source / "frozen_first_alarms.csv").set_index("global_row").loc[b.global_row]
    for field in ("source", "episode", "length", "failure"):
        np.testing.assert_array_equal(frozen[field], b[field])
    first = {"frozen": np.stack([frozen[name + "_frozen"].to_numpy(int) for name in METHODS])}
    with np.load(source / "crossfit_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["global_rows"], b.global_row)
        kind = saved["kinds"].tolist().index("task_init")
        alpha = np.flatnonzero(np.isclose(saved["alphas"], .01)).item()
        method_ids = [saved["methods"].tolist().index(name) for name in SOURCE_NAMES]
        first["standardized_task_init_1pct"] = saved["first"][kind, alpha, method_ids].astype(np.int64)
    task_map = b[["task", "suite"]].drop_duplicates().sort_values(["suite", "task"]).reset_index(drop=True)
    task_map["task_id"] = [suite.removeprefix("libero_")[0].upper() + f"{i + 1:02d}"
                           for suite, group in task_map.groupby("suite", sort=True) for i in range(len(group))]
    task_map["label"] = [task_label(row.task, row.suite) for row in task_map.itertuples()]
    task_map["budget_actions"] = task_map.suite.map(CAPS)
    task_map.to_csv(output / "task_map.csv", index=False)
    b["task_id"] = b.task.map(task_map.set_index("task").task_id)
    b["budget_actions"] = b.suite.map(CAPS)
    exported = b[["global_row", "source", "episode", "task", "task_id", "suite", "failure", "length", "budget_actions"]].copy()
    summary, bins, checkpoints, paired = [], [], [], []
    scopes = [("all", "all", b)] + [("suite", suite, part) for suite, part in b.groupby("suite", sort=True)]
    scopes += [("task", task, part) for task, part in b.groupby("task", sort=True)]
    for family, alarms in first.items():
        assert np.all((alarms == -1) | ((alarms >= 0) & (alarms < b.length.to_numpy()[None])))
        assert np.all(10 * alarms < b.budget_actions.to_numpy()[None])
        for mi, method in enumerate(METHODS):
            exported[family + "_" + method + "_query"] = alarms[mi]
            exported[family + "_" + method + "_pct"] = np.where(alarms[mi] >= 0, 1000 * alarms[mi] / b.budget_actions, np.nan)
        for level, scope, part in scopes:
            for outcome, failure in (("failure", True), ("success", False)):
                current = part.loc[part.failure.eq(failure)]
                ix, caps = current.index.to_numpy(), current.budget_actions.to_numpy()
                for mi, method in enumerate(METHODS):
                    q = alarms[mi, ix]
                    fired = q >= 0
                    meta = dict(family=family, level=level, scope=scope, outcome=outcome, method=method)
                    summary.append(dict(meta, **describe(q, caps)))
                    counts, edges = np.histogram(1000 * q[fired] / caps[fired], bins=np.arange(0, 101, 10))
                    assert counts.sum() + (~fired).sum() == len(current)
                    for lo, hi, count in zip(edges[:-1], edges[1:], counts):
                        bins.append(dict(meta, bin=f"{lo:02d}-{hi:03d}%", count=int(count), episodes=len(current)))
                    bins.append(dict(meta, bin="no_alarm", count=int((~fired).sum()), episodes=len(current)))
                    for cutoff in CUTOFFS:
                        checkpoints.append(dict(meta, cutoff_pct=cutoff, episodes=len(current),
                                                alarms=int((fired & (1000 * q <= cutoff * caps)).sum())))
                for mi, method in enumerate(METHODS[:2]):
                    baseline, candidate = alarms[mi, ix], alarms[2, ix]
                    both = (baseline >= 0) & (candidate >= 0)
                    delta = candidate[both] - baseline[both]
                    paired.append(dict(family=family, level=level, scope=scope, outcome=outcome,
                                       baseline=method, both=int(both.sum()), v82_earlier=int((delta < 0).sum()),
                                       same=int((delta == 0).sum()), v82_later=int((delta > 0).sum()),
                                       only_v82=int(((candidate >= 0) & (baseline < 0)).sum()),
                                       only_baseline=int(((baseline >= 0) & (candidate < 0)).sum())))
    tables = dict(summary=pd.DataFrame(summary), position_bins=pd.DataFrame(bins),
                  checkpoints=pd.DataFrame(checkpoints), paired_with_v82=pd.DataFrame(paired), episode_positions=exported)
    anchors = pd.read_csv(source / "alarm_metrics.csv")
    for family, calibration, alpha in (("frozen", "original", None), ("standardized_task_init_1pct", "task_init", .01)):
        for mi, method in enumerate(METHODS):
            name = method + "_frozen" if family == "frozen" else SOURCE_NAMES[mi]
            expected = anchors.loc[anchors.method.eq(name) & anchors.scope.eq("all") & anchors.calibration.eq(calibration)]
            if alpha is not None:
                expected = expected.loc[np.isclose(expected.alpha, alpha)]
            assert len(expected) == 1
            for outcome, column in (("failure", "tp"), ("success", "fp")):
                actual = tables["summary"].query("family == @family and level == 'all' and method == @method and outcome == @outcome")
                assert int(actual.alarms.iloc[0]) == int(expected[column].iloc[0])
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    render_task_table(output, task_map, tables["summary"])
    plot_tasks(output, task_map, tables["summary"])
    write_json(output / "verification.json", dict(
        all_checks_passed=True, test_episodes=len(b), tasks=len(task_map),
        first_alarms_checked=sum(a.size for a in first.values()), alarm_count_anchors_checked=12,
        duration_used_as_detection_score=False, input_sha256=hashes, source_sha256=digest(__file__),
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    ))
    print(tables["summary"].query("family == 'frozen' and level != 'task'").to_string(index=False))
    print(tables["checkpoints"].query("family == 'frozen' and level == 'all' and method == 'v82'").to_string(index=False))
    print(tables["paired_with_v82"].query("family == 'frozen' and level == 'all' and outcome == 'failure'").to_string(index=False))


if __name__ == "__main__":
    main()
