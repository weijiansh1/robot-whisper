"""Explain frozen routing alarms for the spatial bowl-next-to-plate task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitor import ROOT, first_alarm, first_and, intrinsic_score_arrays, persistent, union
from run_analysis import BASE, RUN_B, digest, historical_series, write_json


TASK = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
HEADS = ("freeze", "turbulence", "frontback", "curvature")


def longest_run(values, threshold, eligible):
    longest = current = 0
    for value, active in zip(values, eligible):
        current = current + 1 if active and np.isfinite(value) and value > threshold else 0
        longest = max(longest, current)
    return longest


def plot_example(output, episodes, streams, extra, profile):
    item = episodes.loc[episodes.episode.eq(77)].iloc[0]
    i, n = int(item.name), int(item.length)
    q = np.arange(7, n)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    settings = (("acceleration", "Route-flow acceleration", ("acceleration_raw", "acceleration", "acceleration_persistent")),
                ("periodicity", "Recurrence-loss score", ("periodicity", "periodicity_persistent")),
                ("freeze", "Relative routing freeze", ("freeze_raw", "freeze")))
    labels = dict(acceleration_raw="Raw", acceleration="3-query mean", acceleration_persistent="8-query confirmation",
                  periodicity="6-query mean", periodicity_persistent="4-query confirmation",
                  freeze_raw="Raw", freeze="6-query mean")
    colors = ("#9b9b9b", "#376da5", "#14836b")
    for ax, (head, title, keys) in zip(axes.flat, settings):
        for key, color in zip(keys, colors[-len(keys):]):
            ax.plot(q, streams[key][i, 7:n], marker="o", markersize=3, color=color, label=labels[key])
        ax.axhline(profile[head + "_threshold"], color="#b44f43", linestyle="--", label="Threshold")
        ax.set_title(title)
    axes[0, 0].axvspan(13, 20, color="#dcebe5", alpha=.5)
    axes[0, 0].text(13.2, .27, "8 consecutive smoothed scores\nq13 through q20", fontsize=9)
    for j, name in enumerate(("frontback", "curvature")):
        threshold = 0.0 if j == 0 else .3890000581741333
        margin = persistent(extra[:, :, j] + .0015 * np.arange(extra.shape[1])[None], 2) - threshold
        axes[1, 1].plot(q, margin[i, 7:n], marker="o", markersize=3,
                        label=name, color=("#376da5", "#b48228")[j])
    axes[1, 1].axhline(0, color="#b44f43", linestyle="--", label="Threshold margin = 0")
    axes[1, 1].set_title("v8.2 new heads: confirmed threshold margins")
    for ax in axes.flat:
        ax.axvline(item.release_query, color="#333333", linestyle=":", label="Physical release")
        ax.axvline(item.v82_first, color="#855391", linestyle="-.", label="First alarm")
        ax.set_xlim(7, 21)
        ax.set_xticks(np.arange(7, 22, 2))
        ax.set_xlabel("Policy query q (10 actions per query)")
        ax.set_ylabel("Score")
        ax.grid(alpha=.17)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("S05, B episode 77: release q9, persistent acceleration confirmed q20", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("episode_77_diagnosis." + extension), dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_s05_diagnosis_20260908")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = BASE / "v82_validation_20260908"
    write_json(output / "execution_contract.json", dict(
        task=TASK, cohort=RUN_B, primary="frozen v8.2 branch and event timing decomposition",
        diagnostic="acceleration confirmations 8 to 1, other heads and all thresholds fixed",
        diagnostic_scope="S05 and all B, retrospective explanation only; no replacement selected",
        raw_head_first_crossing_requires_observed_baseline=True,
        trajectory_duration_used_as_score=False, new_rollouts=False, source_sha256=digest(__file__),
    ))
    prior = json.loads((source / "analysis_verification.json").read_text())
    inputs = {}
    for name in ("index.csv", "frozen_first_alarms.csv", "head_attribution.csv"):
        inputs[str(source / name)] = digest(source / name)
        assert inputs[str(source / name)] == prior["artifacts"][name]
    frame = pd.read_csv(source / "index.csv")
    b = frame.loc[frame.run_id.eq(RUN_B)].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564
    rows = b.global_row.to_numpy(int)
    cached = BASE / "round3_safe/v7/v7_inputs.npz"
    inputs[str(cached)] = digest(cached)
    old_inputs = json.loads((source / "input_verification.json").read_text())["inputs"]
    assert inputs[str(cached)] == old_inputs[str(cached.relative_to(ROOT))]
    with np.load(cached, allow_pickle=False) as z:
        cache = {key: z[key][rows] for key in ("mobility", "acceleration", "periodicity", "valid")}
    profile_path = ROOT / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
    inputs[str(profile_path)] = digest(profile_path)
    with np.load(profile_path, allow_pickle=False) as z:
        profile = {key: float(z[key]) for key in ("freeze_threshold", "acceleration_threshold", "periodicity_threshold", "periodicity_scale")}
    all_streams = intrinsic_score_arrays(cache["mobility"], cache["acceleration"], cache["periodicity"], profile["periodicity_scale"])
    frozen = pd.read_csv(source / "frozen_first_alarms.csv").set_index("global_row").loc[rows].reset_index(drop=True)
    eligible_acc = cache["valid"] & (np.arange(52)[None] >= 7)
    freeze = first_alarm(all_streams["freeze"], cache["valid"], profile["freeze_threshold"])
    acc = first_alarm(all_streams["acceleration_persistent"], eligible_acc, profile["acceleration_threshold"])
    period = first_alarm(all_streams["periodicity_persistent"], cache["valid"], profile["periodicity_threshold"])
    turbulence = first_and(acc, period)
    np.testing.assert_array_equal(union(freeze, turbulence), frozen.v7_frozen)
    acc_one = first_alarm(all_streams["acceleration"], eligible_acc, profile["acceleration_threshold"])
    short_turbulence = first_and(acc_one, period)
    assert np.all((turbulence < 0) | ((short_turbulence >= 0) & (short_turbulence <= turbulence)))
    short_guard = union(frozen.v82_frozen.to_numpy(), short_turbulence)
    task_positions = np.flatnonzero(b.task.eq(TASK).to_numpy())
    task = b.iloc[task_positions].reset_index(drop=True)
    assert len(task) == 400 and task.failure.sum() == 24
    streams = {key: values[task_positions] for key, values in all_streams.items()}
    valid = cache["valid"][task_positions]
    raw_path = BASE / "round7_temporal_fusion/v8_inputs.npz"
    inputs[str(raw_path)] = digest(raw_path)
    assert inputs[str(raw_path)] == old_inputs[str(raw_path.relative_to(ROOT))]
    with np.load(raw_path, allow_pickle=False) as z:
        raw = z["raw"][task.global_row.to_numpy(int)]
    extra = historical_series(raw, valid)
    extra[:, :6] = np.nan
    new_heads = [first_alarm(persistent(extra[:, :, i] + .0015 * np.arange(52)[None], 2), valid, threshold, True)
                 for i, threshold in enumerate((0.0, .3890000581741333))]
    task_first = union(freeze[task_positions], turbulence[task_positions], *new_heads)
    np.testing.assert_array_equal(task_first, frozen.v82_frozen.to_numpy()[task_positions])
    episodes = task[["global_row", "source", "episode", "init_state_id", "noise_seed", "failure", "length", "primary_failure_reason"]].copy()
    episodes["v82_first"] = task_first
    episodes["diagnostic_no_acc_confirmation_first"] = short_guard[task_positions]
    components = [freeze[task_positions], turbulence[task_positions], *new_heads]
    episodes["cause"] = ["+".join(name for name, values in zip(HEADS, components) if first >= 0 and values[i] == first)
                         for i, first in enumerate(task_first)]
    expected_causes = pd.read_csv(source / "head_attribution.csv")
    expected_causes = expected_causes.loc[expected_causes.task.eq(TASK) & expected_causes.version.eq("frozen")].set_index("global_row")
    np.testing.assert_array_equal(episodes.loc[task_first >= 0, "cause"], expected_causes.loc[episodes.loc[task_first >= 0, "global_row"], "cause"])
    for head, start, count in (("freeze", 5, 1), ("acceleration", 7, 8), ("periodicity", 6, 4)):
        eligible = valid & (np.arange(52)[None] >= start)
        for suffix, key in (("raw", head + "_raw"), ("smooth", head),
                            ("confirmed", head if count == 1 else head + "_persistent")):
            episodes[head + "_" + suffix + "_first"] = first_alarm(streams[key], eligible, profile[head + "_threshold"])
        episodes[head + "_longest_smooth_run"] = [longest_run(values, profile[head + "_threshold"], mask)
                                                   for values, mask in zip(streams[head], eligible)]
    for i, head in enumerate(("frontback", "curvature")):
        threshold = 0 if i == 0 else .3890000581741333
        margin = persistent(extra[:, :, i] + .0015 * np.arange(52)[None], 2) - threshold
        episodes[head + "_first"] = new_heads[i]
        episodes[head + "_max_margin"] = np.nanmax(np.where(valid, margin, np.nan), axis=1)
    physics = BASE / "v82_physical_controls_20260908"
    manifest = json.loads((physics / "verification.json").read_text())
    physical_run = next(row for row in manifest["runs"] if row["source"] == task.source.iloc[0])
    physical_path = physics / "runs" / physical_run["output"]
    inputs[str(physical_path)] = digest(physical_path)
    assert inputs[str(physical_path)] == physical_run["output_sha256"]
    physical_rows = []
    with np.load(physical_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["global_rows"], task.global_row)
        np.testing.assert_array_equal(z["lengths"], task.length)
        assert z["subjects"].tolist() == ["akita_black_bowl_1"]
        offset = 0
        for row, n in zip(task.itertuples(), z["lengths"]):
            grasp = z["grasps"][offset:offset + n, 0]
            goal = z["goals"][offset:offset + n, 0]
            releases = np.flatnonzero(grasp[:-1] & ~grasp[1:] & ~goal[1:]) + 1
            first = int(releases[0]) if len(releases) else -1
            physical_rows.append(dict(release_query=first, release_count=len(releases),
                                      observed_goal_true=bool(goal.any())))
            offset += int(n)
    episodes = pd.concat((episodes, pd.DataFrame(physical_rows)), axis=1)
    events = pd.read_csv(physics / "events.csv")
    event_first = events.loc[events.task.eq(TASK) & events.kind.eq("release_outside_goal")].groupby("global_row")["query"].min()
    np.testing.assert_array_equal(episodes.release_query, episodes.global_row.map(event_first).fillna(-1).astype(int))
    episodes["release_to_alarm_queries"] = np.where((episodes.release_query >= 0) & (episodes.v82_first >= 0),
                                                      episodes.v82_first - episodes.release_query, np.nan)
    episodes.to_csv(output / "episodes.csv", index=False)
    details = []
    baseline_ready = dict(freeze=5, acceleration=7, periodicity=6)
    for i, row in episodes.iterrows():
        for q in range(row.length):
            details.append(dict(global_row=row.global_row, episode=row.episode, failure=row.failure, query=q,
                                **{key: float(values[i, q]) if q >= baseline_ready[key.split("_", 1)[0]] else np.nan
                                   for key, values in streams.items()},
                                frontback_smooth=float(extra[i, q, 0]), curvature_smooth=float(extra[i, q, 1])))
    pd.DataFrame(details).to_csv(output / "score_streams.csv", index=False)
    # Verify prefix availability independently of the full cached baseline calculation.
    prefix_checks = 0
    for n in range(8, 23):
        prefix = intrinsic_score_arrays(cache["mobility"][task_positions, :n], cache["acceleration"][task_positions, :n],
                                        cache["periodicity"][task_positions, :n], profile["periodicity_scale"])
        active = valid[:, n - 1]
        for key in streams:
            np.testing.assert_allclose(prefix[key][active, -1], streams[key][active, n - 1], atol=1e-7, rtol=0, equal_nan=True)
            prefix_checks += int(active.sum())
    diagnostic = []
    for scope, mask in (("S05", b.task.eq(TASK).to_numpy()), ("all_B", np.ones(len(b), bool)),
                        ("other_B_tasks", b.task.ne(TASK).to_numpy())):
        failure = b.failure.to_numpy(bool)[mask]
        for name, first in (("frozen_v82", frozen.v82_frozen.to_numpy()[mask]),
                            ("diagnostic_no_acc_confirmation", short_guard[mask])):
            fired = first >= 0
            diagnostic.append(dict(scope=scope, method=name, failures=int(failure.sum()), successes=int((~failure).sum()),
                                   tp=int((fired & failure).sum()), fp=int((fired & ~failure).sum()),
                                   median_failure_query=float(np.median(first[fired & failure]))))
    pd.DataFrame(diagnostic).to_csv(output / "fixed_threshold_diagnostic.csv", index=False)
    write_json(output / "profile.json", profile)
    plot_example(output, episodes, streams, extra, profile)
    write_json(output / "verification.json", dict(all_checks_passed=True, v7_first_alarms_replayed=16000,
               v82_task_alarms_replayed=400, task_physical_release_records_checked=400,
               prefix_score_checks=prefix_checks, input_sha256=inputs, source_sha256=digest(__file__),
               artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(pd.DataFrame(diagnostic).to_string(index=False))
    print(episodes.loc[episodes.failure, ["episode", "v82_first", "cause", "release_query", "release_to_alarm_queries",
          "acceleration_smooth_first", "acceleration_confirmed_first", "acceleration_longest_smooth_run",
          "periodicity_confirmed_first", "periodicity_longest_smooth_run"]].to_string(index=False))


if __name__ == "__main__":
    main()
