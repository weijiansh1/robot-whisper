"""Audit sealed predictions; no fitting or changes to the round-one methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import (CHECKPOINTS, PRIMARY, SEED, binary_metrics, digest,
                  intrinsic_score_arrays, prefix_aggregate, score_at, write_json)
from evaluate import PHYSICAL, hierarchical_weights, ranking_metrics
from intrinsic_guard_monitor import GlobalIntrinsicProfile
from unlabeled_budget_calibration import alarms_from_scores

HERE = Path(__file__).resolve().parent
DROP_REASONS = ("object_released_or_dropped_before_goal", "object_released_outside_goal")


def subject_release(record: dict) -> int | None:
    """Use only subjects of goals whose annotated failure is release-related."""
    predicates = {p["id"]: p["expression"] for p in record["goal_predicates"]}
    onsets = []
    for goal in record["goal_failure_labels"]:
        if goal["reason"] not in DROP_REASONS:
            continue
        subject = predicates[goal["goal_id"]][1]
        onset = record["goal_subject_physics"].get(subject, {}).get("first_release_snapshot")
        if onset is not None:
            onsets.append(int(onset))
    return min(onsets) if onsets else None


def physical_audit(frame: pd.DataFrame):
    lookup = {(r.run_id, r.task, int(r.episode)): int(r.row) for r in frame.itertuples()}
    seen, events = set(), []
    with (PHYSICAL / "failures.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["run_id"], record["suite"] + "/" + record["task_name"],
                   record["episode_index"])
            if key not in lookup:
                continue
            row = lookup[key]
            if row in seen or not frame.iloc[row].failure:
                raise ValueError("duplicate or non-failure physical record")
            seen.add(row)
            physics = record["physics_validation"]
            length = int(frame.iloc[row].length)
            if record["inference_calls"] != length or physics["snapshots_restored"] != length:
                raise ValueError("physical snapshot/query alignment mismatch")
            if physics["status"] != "passed" or physics["snapshot_stride_actions"] != 10:
                raise ValueError("unexpected physical sampling protocol")
            if record["primary_failure_reason"] != frame.iloc[row].primary_failure_reason:
                raise ValueError("physical reason mismatch")
            if record["primary_failure_reason"] in DROP_REASONS:
                onset = subject_release(record)
                if onset is not None:
                    if not 0 <= onset < length:
                        raise ValueError("subject release outside recorded queries")
                    events.append({"row": row, "onset": onset,
                                   "reason": record["primary_failure_reason"]})
    if len(seen) != int(frame.failure.sum()):
        raise ValueError("physical label coverage mismatch")
    return pd.DataFrame(events), len(seen)


def causal_replay(output: Path, first: np.ndarray) -> dict:
    with np.load(output / "test_inputs.npz", allow_pickle=False) as archive:
        rows = np.sort(np.random.default_rng(SEED).choice(len(first), 64, replace=False))
        values = {k: archive[k][rows] for k in ("mobility", "acceleration", "periodicity", "valid")}
    profile = GlobalIntrinsicProfile(**json.loads(
        (output / "cohort_transfer_profiles.json").read_text())[0]["v7"]["0.03"]["profile"])
    for stop in range(1, values["valid"].shape[1] + 1):
        scores = intrinsic_score_arrays(values["mobility"][:, :stop],
                                        values["acceleration"][:, :stop],
                                        values["periodicity"][:, :stop],
                                        profile.periodicity_scale)
        replay = alarms_from_scores(scores, values["valid"][:, :stop], profile)["guard"]
        expected = np.where((first[rows] >= 0) & (first[rows] < stop), first[rows], -1)
        if not np.array_equal(replay, expected):
            raise ValueError(f"non-causal guard prediction at prefix {stop}")
    return {"episodes": len(rows), "prefixes_per_episode": stop, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results/round1")
    args = parser.parse_args()
    output = args.input.resolve()
    summary = json.loads((output / "evaluation_summary.json").read_text())
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in summary["result_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"changed evaluation: {name}")
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"changed prediction artifact: {name}")
    frame = pd.read_csv(output / "label_alignment.csv")
    y, length = frame.failure.to_numpy(bool), frame.length.to_numpy(int)
    tasks = frame.task.to_numpy(str)
    groups = [np.flatnonzero(tasks == t) for t in sorted(frame.task.unique())]
    scopes = {"all": np.ones(len(frame), bool),
              "standard_3_suites": frame.suite.to_numpy() != "libero_long",
              "long": frame.suite.to_numpy() == "libero_long"}
    alarms = np.load(output / "first_alarms.npz", allow_pickle=False)
    names, budgets = alarms["method_names"].astype(str).tolist(), alarms["budgets"]
    primary_i, budget_i = names.index(PRIMARY), int(np.flatnonzero(np.isclose(budgets, 0.03))[0])
    events, matched = physical_audit(frame)
    events.to_csv(output / "subject_release_events.csv", index=False)
    event_rows, prefix_rows, scope_rows, references = [], [], [], []
    for setting_info in manifest["settings"]:
        setting = setting_info["setting"]
        for b, budget in enumerate(budgets):
            for m, method in enumerate(names):
                first = alarms[setting][b, m]
                available = first != -2
                for point in (*CHECKPOINTS, None):
                    fired = (first >= 0) & (True if point is None else first <= point)
                    retained = available & ~fired
                    prefix_rows.append({
                        "setting": setting, "method": method, "budget": budget,
                        "checkpoint": "full_rollout" if point is None else f"q{point}",
                        "available_episodes": int(available.sum()), "retained": int(retained.sum()),
                        "coverage": float(retained.sum() / max(available.sum(), 1)),
                        "retained_failures": int((retained & y).sum()),
                        "selective_risk": float((retained & y).sum() / max(retained.sum(), 1)),
                        "detected_failures": int((available & fired & y).sum()),
                        "false_alarms": int((available & fired & ~y).sum()),
                    })
                if np.isclose(budget, 0.03):
                    for scope, rows in scopes.items():
                        take = rows & available
                        scope_rows.append({"setting": setting, "method": method, "scope": scope,
                                           **binary_metrics(first[take], y[take], length[take])})
                    if setting in ("cohort_transfer", "task_heldout"):
                        rows, onset = events.row.to_numpy(int), events.onset.to_numpy(int)
                        fired = first[rows] >= 0
                        delta = first[rows] - onset
                        event_rows.append({
                            "setting": setting, "method": method, "events": len(rows),
                            "detected": int(fired.sum()),
                            "strictly_before": int((fired & (delta < 0)).sum()),
                            "at_or_before": int((fired & (delta <= 0)).sum()),
                            "before2": int((fired & (delta <= -2)).sum()),
                            "before4": int((fired & (delta <= -4)).sum()),
                            "median_delay_detected": float(np.median(delta[fired])) if fired.any() else np.nan,
                        })
        for profile in json.loads((output / f"{setting}_profiles.json").read_text()):
            for key, info in profile["thresholds"].items():
                method, budget = key.rsplit("|", 1)
                references.append({"setting": setting, "fold": profile["fold"],
                                   "method": method, "budget": float(budget),
                                   "episodes": info["reference_episodes"],
                                   "alarms": info["reference_alarms"]})
            for budget, info in profile["v7"].items():
                if "audit" in info:
                    references.append({"setting": setting, "fold": profile["fold"],
                                       "method": PRIMARY, "budget": float(budget),
                                       "episodes": info["audit"]["reference_episodes"],
                                       "alarms": info["audit"]["union_reference_alarms"]})
    reference_table = pd.DataFrame(references)
    if (reference_table.alarms > np.floor(reference_table.budget * reference_table.episodes)).any():
        raise ValueError("a reference budget was exceeded")
    reference_table["actual_rate"] = reference_table.alarms / reference_table.episodes
    reference_table.to_csv(output / "reference_budget_audit.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "subject_release_metrics.csv", index=False)
    pd.DataFrame(prefix_rows).to_csv(output / "selective_risk.csv", index=False)
    pd.DataFrame(scope_rows).to_csv(output / "scope_alarm_metrics.csv", index=False)

    with np.load(output / "cohort_transfer_scores.npz", allow_pickle=False) as archive:
        scores, score_names = archive["scores"], archive["method_names"].astype(str)
    minimum = np.empty(len(frame), int)
    for rows in groups:
        minimum[rows] = length[rows].min() - 1
    rankings = []
    for name, score in zip(score_names, scores):
        points = [(f"q{q}", np.full(len(frame), q), length > q, score) for q in CHECKPOINTS]
        points.append(("retrospective_half", (length - 1) // 2, scopes["all"], score))
        if not name.endswith("__max"):
            maximum = prefix_aggregate(score, "max")
            points.extend([("safe_task_min_prefixmax", minimum, scopes["all"], maximum),
                           ("full_rollout_prefixmax", length - 1, scopes["all"], maximum)])
        for point, positions, eligible, values in points:
            for scope, take in scopes.items():
                if scope == "all":
                    continue
                result = ranking_metrics(y, score_at(values, positions), tasks,
                                         eligible & take, groups)
                result["overall_coverage"] = result["scored_episodes"] / int(take.sum())
                result["failure_coverage"] = result["scored_failures"] / int((take & y).sum())
                rankings.append({"setting": "cohort_transfer", "method": name,
                                 "checkpoint": point, "scope": scope, **result})
    pd.DataFrame(rankings).to_csv(output / "scope_ranking_metrics.csv", index=False)

    group, weights = hierarchical_weights(frame, 1000)
    differences = []
    for setting in ("cohort_transfer", "state_heldout", "task_heldout", "suite_heldout"):
        a = alarms[setting][budget_i, primary_i]
        for baseline in ("clock", "freeze__current"):
            b = alarms[setting][budget_i, names.index(baseline)]
            for metric, take, hit_a, hit_b in (
                ("recall", y, a >= 0, b >= 0),
                ("fpr", ~y, a >= 0, b >= 0),
                ("recall_lead4", y, (a >= 0) & (length - 1 - a >= 4),
                 (b >= 0) & (length - 1 - b >= 4)),
            ):
                numerator = np.bincount(group, weights=(hit_a.astype(int) - hit_b) * take,
                                        minlength=weights.shape[1])
                denominator = np.bincount(group, weights=take, minlength=weights.shape[1])
                boot = (weights @ numerator) / (weights @ denominator)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                differences.append({"setting": setting, "baseline": baseline, "metric": metric,
                                    "primary_minus_baseline": numerator.sum() / denominator.sum(),
                                    "lo": lo, "hi": hi})
    pd.DataFrame(differences).to_csv(output / "paired_comparisons.csv", index=False)
    replay = causal_replay(output, alarms["cohort_transfer"][budget_i, primary_i])
    write_json(output / "diagnostic_summary.json", {
        "diagnostics_added_after_initial_results": True,
        "scores_thresholds_or_methods_changed": False,
        "physical_failures_length_and_stride_verified": matched,
        "subject_matched_release_episodes": len(events),
        "release_caveat": "First observed grasp loss of a release-related failed goal subject; not proven irreversible failure onset.",
        "selective_risk_caveat": "Passive outcome risk of unflagged rollouts; no recovery or assisted success is simulated.",
        "reference_budget_checks": len(references), "causal_guard_replay": replay,
        "physical_jsonl_sha256": digest(PHYSICAL / "failures.jsonl"),
        "source_sha256": digest(Path(__file__)),
        "original_evaluation_sha256": digest(output / "evaluation_summary.json"),
    })
    print(json.dumps({"physical_records_verified": matched, "subject_events": len(events),
                      "budget_checks": len(references), "causal_replay": replay}))


if __name__ == "__main__":
    main()
