"""Post-analysis comparison that preserves v8.2 alarms and limits extra false alarms."""

import csv
import datetime
import json
from pathlib import Path

import numpy as np

from develop_v82_alarm import (METHODS, OUTPUT, SOURCE, PRIMARY_QUERY, candidate_profiles,
    frozen_sources, paired_changes, pilot_check, save_csv, save_json, select_profile,
    sha256_file, summarize, verify_exported_monitor)


def choose_with_reference_budget(first, labels, train, profiles):
    reference_fp = int(((first[0, train] >= 0) & ~labels[train]).sum())
    eligible = []
    for i, profile in enumerate(profiles):
        if profile.curvature_confirm != 2:
            continue
        calls = first[i, train]
        fp = int(((calls >= 0) & ~labels[train]).sum())
        if fp <= reference_fp:
            tp_mid = int(((calls >= 0) & (calls <= PRIMARY_QUERY) & labels[train]).sum())
            tp_full = int(((calls >= 0) & labels[train]).sum())
            ratio = 1. if profile.weak_freeze_ratio is None else profile.weak_freeze_ratio
            eligible.append(((-tp_mid, -tp_full, -ratio, profile.name), i))
    return min(eligible)[1]


def main():
    root = OUTPUT
    protocol_path = root / "coverage-selection-protocol.json"
    if protocol_path.exists():
        raise ValueError("Coverage comparison already exists; preserve its results")
    hashes = {str(path): sha256_file(path) for path in (
        Path(__file__), METHODS / "v82_alarm_candidate.py", root / "protocol.json", root / "results.json",
        root / "all-candidate-episodes.csv", SOURCE / "summary.json")}
    save_json(protocol_path, dict(
        schema="local.v82.coverage_selection.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_hashes=hashes,
        status="Post-analysis exploratory follow-up; previous candidate comparison remains unchanged",
        reason="Longer curvature confirmation removed one true alarm and delayed alarms on separate pilot traces",
        changes="Restrict to the already evaluated candidates that keep original curvature confirmation at two",
        family="Original v8.2 OR four consecutive weak-freeze scores; ratios off / 0.8 / 0.7",
        selection_rule=["Training false alarms no greater than original v8.2 on the same training tasks",
            "Most failure detections by q36", "Most full-episode failure detections", "Higher weak-freeze ratio"],
        primary_query=PRIMARY_QUERY, held_out_unit="Whole base task, including both benchmarks and all initial states",
        design_has_seen_all_outcomes=True, independent_blind_test=False,
        policy_training=False, alarm_only=True))
    summary = json.loads((SOURCE / "summary.json").read_text())
    episodes, profiles = summary["episodes"], candidate_profiles()
    episode_index = {e["name"]: i for i, e in enumerate(episodes)}
    profile_index = {p.name: i for i, p in enumerate(profiles)}
    first = np.full((len(profiles), len(episodes)), -2, int)
    with (root / "all-candidate-episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = profile_index[row["profile"]], episode_index[row["episode"]]
            if first[key] != -2:
                raise ValueError("Duplicate candidate/episode prediction")
            first[key] = int(row["first_query"])
    if np.any(first < -1):
        raise ValueError("Missing candidate prediction")
    labels = np.asarray([not e["source_result"]["success"] for e in episodes], bool)
    tasks = np.asarray([e["base_task_id"] for e in episodes], int)
    original_hashes = frozen_sources()
    if original_hashes != summary["source_sha256"]:
        raise ValueError("Frozen reference changed")
    for j, profile in enumerate(profiles):
        if profile.curvature_confirm == 2:
            originally_alarmed = first[0] >= 0
            if np.any((first[j, originally_alarmed] < 0) |
                      (first[j, originally_alarmed] > first[0, originally_alarmed])):
                raise ValueError("A coverage candidate loses or delays an original alarm")
    oof = np.full(len(episodes), -2, int)
    selected_names, folds = {}, []
    for task in sorted(set(tasks.tolist())):
        train, test = np.flatnonzero(tasks != task), np.flatnonzero(tasks == task)
        chosen = choose_with_reference_budget(first, labels, train, profiles)
        poisoned_first, poisoned_labels = first.copy(), labels.copy()
        poisoned_first[:, test], poisoned_labels[test] = 0, ~poisoned_labels[test]
        if choose_with_reference_budget(poisoned_first, poisoned_labels, train, profiles) != chosen:
            raise ValueError("Held-out values affect selection")
        oof[test] = first[chosen, test]
        selected_names[task] = profiles[chosen].name
        folds.append(dict(held_out_task=task, selected_profile=profiles[chosen].name,
            train_episodes=len(train), test_episodes=[episodes[i]["name"] for i in test]))
    if np.any(oof < -1):
        raise ValueError("Missing held-out prediction")
    chosen = choose_with_reference_budget(first, labels, np.arange(len(episodes)), profiles)
    profile_path = root / "coverage-profile.json"
    save_json(profile_path, dict(schema="local.v82.alarm_candidate_profile.v1", name=profiles[chosen].name,
        parameters=profiles[chosen].to_dict(), status="Experimental alarm-only coverage candidate",
        selection_scope="Post-analysis reference-FP-budget comparison; all-cohort refit",
        independent_blind_test=False, checkpoint_sha256=episodes[0]["policy_identity"]["checkpoint_sha256"],
        frozen_source_sha256=original_hashes, input_shape=[8, 10, 11, 32], reset_between_episodes=True))
    online = verify_exported_monitor(SOURCE, episodes, first[chosen], profile_path)
    pilots = pilot_check(profile_path, episodes)
    rows = []
    for i, episode in enumerate(episodes):
        rows.append(dict(episode=episode["name"], benchmark=episode["benchmark"],
            base_task_id=episode["base_task_id"], failed=bool(labels[i]),
            original_first_action_step=int(first[0, i] * 10) if first[0, i] >= 0 else None,
            held_out_profile=selected_names[episode["base_task_id"]],
            held_out_first_action_step=int(oof[i] * 10) if oof[i] >= 0 else None))
    save_csv(root / "coverage-episode-comparison.csv", rows)
    result = dict(complete=True, independent_blind_test=False,
        study_status="Post-analysis exploratory comparison on previously inspected episodes",
        selected_profile=profiles[chosen].name, selected_parameters=profiles[chosen].to_dict(),
        original=summarize(first[0], labels, episodes), task_held_out=summarize(oof, labels, episodes),
        all_cohort_fitted=summarize(first[chosen], labels, episodes),
        task_held_out_paired_changes=paired_changes(first[0], oof, labels),
        fold_selection_counts={p.name: sum(name == p.name for name in selected_names.values()) for p in profiles},
        folds=folds, pilot_checks=pilots,
        verification=dict(all_original_alarm_times_preserved=True, held_out_selection_poison_checks_passed=True,
            exported_monitor=online, frozen_original_unchanged=frozen_sources() == original_hashes))
    if any(sha256_file(Path(path)) != expected for path, expected in hashes.items()):
        raise ValueError("A source artifact changed during the follow-up")
    save_json(root / "coverage-results.json", result)
    lines = ["# Alarm coverage follow-up", "",
        "This post-analysis follow-up preserves the first comparison and changes the selection objective explicitly. "
        "The first comparison prioritized the fewest false alarms, selecting four-query curvature confirmation. "
        "That suppressed one false alarm in-sample but also lost one true alarm and delayed the pilot alarms. "
        "This follow-up keeps the original two-query curvature rule and compares only the already evaluated "
        "weak-freeze additions, allowing no more training false alarms than the original detector.", "",
        "The candidate is exactly original v8.2 OR persistent weak freeze. It cannot delay or remove an original "
        "alarm. A new weak-freeze alarm requires four consecutive scores at or above 80% of the original "
        "freeze threshold (approximately 0.45925). Original feature calculations and moving thresholds are retained.", "",
        "Each fold holds out a complete base task, including both benchmarks and all initial states. "
        "The family and selection objective were informed by this inspected cohort; these results are "
        "development evidence, not independent confirmation.", "",
        "| Deadline | Original TP / 51 | Task-held-out candidate TP / 51 |",
        "|---|---:|---:|"]
    for window in ("q15", "q18", "q26", "q30", "q36", "q40", "full_episode"):
        old, new = result["original"][window]["all"], result["task_held_out"][window]["all"]
        name = "Full observed episode" if window == "full_episode" else str(int(window[1:]) * 10) + " actions"
        lines.append(f"| {name} | {old['tp']} | {new['tp']} |")
    old, new = result["original"]["full_episode"]["all"], result["task_held_out"]["full_episode"]["all"]
    lines += [f"| Any false alarm / 9 successes | {old['fp']} | {new['fp']} |", "",
        "Selected profile: `" + profiles[chosen].name + "`. Fold selections:", "", "```json",
        json.dumps({k: v for k, v in result["fold_selection_counts"].items() if v}, indent=2), "```", "",
        "## Separate pilot checks", "", "| Pilot | Outcome | Original action | Candidate action |",
        "|---|---|---:|---:|"]
    for p in pilots:
        old = str(p["reference_first_query"] * 10) if p["reference_first_query"] >= 0 else "none"
        new = str(p["candidate_first_query"] * 10) if p["candidate_first_query"] >= 0 else "none"
        lines.append(f"| {p['name']} | {'success' if p['success'] else 'failure'} | {old} | {new} |")
    lines += ["", "These traces were not used for parameter choice, but were already collected and inspected. "
        "They do not establish benchmark-wide accuracy.", "",
        "## Online use", "", "```python", "import sys",
        "sys.path.insert(0, '/data/coding/v8-methods')", "from v82_alarm_candidate import V82AlarmCandidate", "",
        "monitor = V82AlarmCandidate.from_profile(" + repr(str(profile_path)) + ")",
        "status = monitor.update(hb_router_probs)", "alarm = status['alarm']",
        "# Call monitor.reset() at the start of each new episode.", "```", "",
        "All 2,891 complete routing queries reproduce through this exported monitor; fresh truncated-prefix "
        "results also match exactly. Existing original alarm times are preserved. No model training, "
        "simulator actions, recovery controller, or policy-service replacement is involved.", "",
        "Successes stop at their actual endpoints, without padding. The mid-episode metrics are therefore "
        "operating-budget metrics, not equal-length discrimination tests. Only nine successes are available, "
        "all from Plus; one false alarm remains. The remaining failure cases and independent new successes "
        "are the next development and validation targets.", ""]
    (root / "COVERAGE_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(event="coverage_followup_complete", profile=profiles[chosen].name,
        primary=result["task_held_out"]["q36"]["all"], full=result["task_held_out"]["full_episode"]["all"],
        folds=result["fold_selection_counts"])), flush=True)


if __name__ == "__main__":
    main()
