"""Compare physical events and successful controls without selecting on alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import ALPHAS, HERE, METHODS, ROOT
from run_analysis import RUN_B, SEED, archive, bootstrap_indices, digest, write_json

KINDS = ("release_outside_goal", "release_height_loss", "goal_regression")
OFFSETS = tuple(range(-6, 11))


def verify_physics(frame, physics):
    episodes = pd.read_csv(physics / "episodes.csv")
    b = frame.loc[frame.run_id.eq(RUN_B)]
    assert len(episodes) == len(b) == 16000 and not episodes.global_row.duplicated().any()
    episodes = episodes.set_index("global_row").loc[b.global_row]
    for field in ("length", "failure", "init_state_id", "episode", "source", "action_steps"):
        np.testing.assert_array_equal(episodes[field].to_numpy(), b[field].to_numpy())
    events = pd.read_csv(physics / "events.csv")
    assert events.global_row.isin(b.global_row).all()
    assert ((events["query"] >= 0) & (events["query"] < events.length)).all()
    first_release = events.loc[events.kind.isin(["release_in_goal", "release_outside_goal"])].groupby(
        ["source", "episode", "subject"])["query"].min().to_dict()
    lookup = frame.set_index(["source", "episode"])
    anchors = []
    with (ROOT / "VLA_MUI_HUB/physical-failure-labels/results/failures.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["run_id"] != RUN_B:
                continue
            source, ep = record["source"]["run"], int(record["episode_index"])
            row = lookup.loc[(source, ep)]
            assert row.failure and row.length == record["inference_calls"]
            for subject, values in record["goal_subject_physics"].items():
                old = values["first_release_snapshot"]
                old = -1 if old is None else int(old)
                new = int(first_release.get((source, ep, subject), -1))
                anchors.append(dict(global_row=int(row.global_row), subject=subject, old_release=old, new_release=new))
                if old != new:
                    raise AssertionError("physical release mismatch: %s ep%d %s %d != %d" % (source, ep, subject, old, new))
    return episodes, events, pd.DataFrame(anchors)


def timing_table(frame, events, alarm_methods, output):
    primary = events.loc[events.kind.isin(KINDS)].sort_values(["query", "subject"]).drop_duplicates(["global_row", "kind"])
    primary = primary.sort_values(["global_row", "kind"])
    primary.to_csv(output / "physical_primary_events.csv", index=False)
    rows, details = [], []
    for (kind, failure), part in primary.groupby(["kind", "failure"], sort=True):
        scopes = [("all", part)] + list(part.groupby("suite", sort=True))
        for scope, current in scopes:
            ix = current.global_row.to_numpy(int)
            event_q = current["query"].to_numpy(int)
            length = current.length.to_numpy(int)
            for name, first in alarm_methods.items():
                alarm = first[ix]
                fired = (alarm >= 0) & (alarm < length)
                delay = alarm - event_q
                before, same, after = fired & (delay < 0), fired & (delay == 0), fired & (delay > 0)
                row = dict(kind=kind, failure=bool(failure), scope=scope, method=name, episodes=len(current),
                           before=int(before.sum()), same=int(same.sum()), after=int(after.sum()), missed=int((~fired).sum()),
                           median_delay_q=float(np.median(delay[fired])) if fired.any() else np.nan,
                           median_post_event_delay_q=float(np.median(delay[after])) if after.any() else np.nan)
                assert row["before"] + row["same"] + row["after"] + row["missed"] == row["episodes"]
                for lead in (2, 4):
                    row["lead_ge_%dq" % lead] = int((fired & (delay <= -lead)).sum())
                for lag in (2, 4):
                    row["detected_by_plus_%dq" % lag] = int((fired & (delay <= lag)).sum())
                rows.append(row)
                if scope == "all":
                    details.extend(dict(kind=kind, failure=bool(failure), method=name, global_row=int(i),
                                        event_q=int(e), first=int(a), delay_q=int(d) if f else np.nan)
                                   for i, e, a, d, f in zip(ix, event_q, alarm, delay, fired))
    pd.DataFrame(rows).to_csv(output / "event_timing.csv", index=False)
    pd.DataFrame(details).to_csv(output / "event_alarm_details.csv", index=False)
    return primary


def event_curves(primary, raw_scores, names, output):
    rows = []
    for (kind, failure), part in primary.groupby(["kind", "failure"], sort=True):
        for scope, current in [("all", part)] + list(part.groupby("suite", sort=True)):
            for coverage in ("available", "complete_minus3_plus4"):
                selected = current
                if coverage == "complete_minus3_plus4":
                    selected = current.loc[(current["query"] >= 9) & (current.length > current["query"] + 4)]
                ix = selected.global_row.to_numpy(int)
                q0 = selected["query"].to_numpy(int)
                for delta in OFFSETS:
                    q = q0 + delta
                    observed = (q >= 6) & (q < selected.length.to_numpy(int))
                    for hi, head in enumerate(names):
                        values = raw_scores[hi, ix[observed], q[observed]]
                        values = values[np.isfinite(values)]
                        summary = np.quantile(values, [.25, .5, .75]) if len(values) else [np.nan] * 3
                        rows.append(dict(kind=kind, failure=bool(failure), scope=scope, coverage=coverage,
                                         head=head, offset=delta, total_events=len(selected), observed=len(values),
                                         p25=summary[0], median=summary[1], p75=summary[2]))
    pd.DataFrame(rows).to_csv(output / "event_aligned_curves.csv", index=False)


def matched_controls(frame, primary, score_bank, output):
    rng = np.random.default_rng(SEED)
    b = frame.loc[frame.run_id.eq(RUN_B)]
    pairs = []
    for row in primary.loc[primary.failure].sort_values(["kind", "task", "global_row"]).itertuples():
        q = int(getattr(row, "query"))
        candidates = b.loc[(~b.failure) & b.task.eq(row.task) & (b.length > q)]
        same = candidates.loc[candidates.init_state_id.eq(row.init_state_id)]
        available = same if len(same) else candidates
        control = int(rng.choice(available.global_row)) if len(available) else -1
        pairs.append(dict(kind=row.kind, case_row=int(row.global_row), control_row=control, event_q=q,
                          task=row.task, suite=row.suite, matching="same_init" if len(same) else
                          ("same_task" if len(candidates) else "no_active_success")))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(output / "event_control_pairs.csv", index=False)
    rows = []
    for (kind, scope), selected in [( (kind, "all"), part) for kind, part in pairs.groupby("kind", sort=True)] + [
            ((kind, suite), part) for (kind, suite), part in pairs.groupby(["kind", "suite"], sort=True)]:
        current = selected.loc[selected.control_row >= 0]
        for delta in (-2, 0, 2, 4):
            cases = current.case_row.to_numpy(int)
            controls = current.control_row.to_numpy(int)
            q = current.event_q.to_numpy(int) + delta
            available = (q >= 6) & (q < frame.iloc[cases].length.to_numpy()) & (q < frame.iloc[controls].length.to_numpy())
            for name, score in score_bank.items():
                case_values = score[cases[available], q[available]]
                control_values = score[controls[available], q[available]]
                finite = np.isfinite(case_values) & np.isfinite(control_values)
                difference = case_values[finite] - control_values[finite]
                weights = (difference > 0).astype(float) + .5 * (difference == 0)
                group = current.loc[available].iloc[np.flatnonzero(finite)]
                unit = pd.DataFrame(dict(task=group.task.to_numpy(), suite=group.suite.to_numpy(), win=weights))
                task = unit.groupby(["suite", "task"], as_index=False).win.mean()
                lo = hi = estimate = np.nan
                if len(task):
                    draws = bootstrap_indices(task.task.to_numpy(), task.suite.to_numpy())
                    lo, hi = np.quantile(task.win.to_numpy()[draws].mean(axis=1), [.025, .975])
                    estimate = float(task.win.mean())
                rows.append(dict(kind=kind, scope=scope, offset=delta, method=name,
                                 failure_events=len(selected), matched_at_event=len(current),
                                 observed_pairs=len(weights), unique_controls=int(group.control_row.nunique()),
                                 tasks=len(task), case_higher_task_mean=estimate, lo=lo, hi=hi,
                                 median_difference=float(np.median(difference)) if len(difference) else np.nan))
    pd.DataFrame(rows).to_csv(output / "event_matched_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=HERE.parent / "results/v82_validation_20260908")
    parser.add_argument("--physics", type=Path, default=HERE.parent / "results/v82_physical_controls_20260908")
    args = parser.parse_args()
    output = args.analysis
    frame = pd.read_csv(output / "index.csv")
    episodes, events, anchors = verify_physics(frame, args.physics)
    anchors.to_csv(output / "physical_label_anchors.csv", index=False)
    frozen = pd.read_csv(output / "frozen_first_alarms.csv")
    alarm_methods = {name: frozen[name].to_numpy(int) for name in ("v7_frozen", "v82_frozen", "v82_padding_corrected")}
    predictions = archive(output / "crossfit_predictions.npz")
    rows = predictions["global_rows"]
    for name in ("v7", "v82", "eef_motion_low", "clock"):
        first = np.full(len(frame), -1, np.int16)
        first[rows] = predictions["first"][1, ALPHAS.index(.01), METHODS.index(name)]
        alarm_methods[name + "_budget01"] = first
    primary = timing_table(frame, events, alarm_methods, output)
    raw = archive(output / "frozen_scores.npz")
    names = raw["names"].astype(str).tolist()
    event_curves(primary, raw["scores"], names, output)
    scores = {name: raw["scores"][i] for i, name in enumerate(names)}
    for name in ("v7", "v82", "eef_motion_low"):
        values = np.full((len(frame), 52), np.nan, np.float32)
        values[rows] = predictions["scores"][METHODS.index(name)]
        scores[name] = values
    matched_controls(frame, primary, scores, output)
    counts = primary.groupby(["kind", "failure"]).size().rename("episodes").reset_index()
    counts.to_csv(output / "physical_event_counts.csv", index=False)
    goal_early = episodes.loc[episodes.first_full_goal_q >= 0].copy()
    goal_early.to_csv(output / "restored_goal_satisfied_successes.csv")
    write_json(output / "event_verification.json", dict(physics_verification_sha256=digest(args.physics / "verification.json"),
                events_sha256=digest(args.physics / "events.csv"), original_label_anchors=len(anchors),
                original_release_labels_agree=True, physical_episodes=len(episodes), physical_events=len(events),
                primary_event_counts=counts.to_dict("records"),
                successes_with_full_goal_at_restored_checkpoint=len(goal_early),
                new_rollouts=False, model_training=False, release_is_not_an_irreversible_failure_label=True,
                sources={p.name: digest(p) for p in (Path(__file__), HERE / "physical_controls.py")}))
    print("EVENTS complete: %d episodes, %d raw events, %d legacy anchors" %
          (len(episodes), len(events), len(anchors)), flush=True)


if __name__ == "__main__":
    main()
