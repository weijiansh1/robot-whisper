"""Evaluate small, alarm-only v8.2 changes with task-held-out parameter selection."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_v82_sample import frozen_sources, sha256_file
from validate_v82_mechanism import confusion, save_csv, save_json

METHODS = Path("/data/coding/v8-methods")
sys.path.insert(0, str(METHODS))
from v82_alarm_candidate import AlarmProfile, SustainedAlarmState, V82AlarmCandidate, candidate_profiles
from v82_closed_loop import V82Monitor
from v8_closed_loop import SIGNALS, score_status

SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
PILOT = Path("/data/coding/v8-signal-test-CTUaAJ")
OUTPUT = Path("/data/libero-runtime/samples/v82-alarm-development-20260915")
PRIMARY_QUERY = 36
WINDOWS = (15, 18, 26, 30, 36, 40, -1)


def source_files():
    return {str(p): sha256_file(p) for p in (
        Path(__file__), METHODS / "v82_alarm_candidate.py", METHODS / "test_v82_alarm_candidate.py")}


def plan(source, output):
    summary = json.loads((source / "summary.json").read_text())
    if not summary["complete"] or len(summary["episodes"]) != 60:
        raise ValueError("Expected the original complete cohort")
    if frozen_sources() != summary["source_sha256"]:
        raise ValueError("Original frozen detector changed")
    protocol = dict(schema="local.v82.alarm_development.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=str(source),
        study_status="Exploratory development; method design used previously inspected cohort diagnostics",
        independent_blind_test=False, policy_training=False, policy_intervention=False,
        scope="Alarm only; unchanged original routing features and v8.2 threshold schedule",
        hypotheses=[
            "Persistent subthreshold freeze evidence may cover missed or late alarms",
            "Longer curvature confirmation may suppress transient false alarms",
        ],
        profiles=[dict(name=p.name, parameters=p.to_dict()) for p in candidate_profiles()],
        primary_query=PRIMARY_QUERY, primary_action_steps=PRIMARY_QUERY * 10,
        secondary_queries=[q for q in WINDOWS if q != PRIMARY_QUERY],
        positive_metric="Failure episodes alarmed by the stated action budget",
        negative_metric="Any alarm before the successful episode actually ends; never clear a counted false alarm",
        selection_order=["Fewest successful-episode false alarms", "Most failures detected by q36",
            "Most failures detected over the complete observed episode", "Fewest changes from frozen v8.2",
            "Higher weak-freeze ratio", "Shorter curvature confirmation", "Stable profile name"],
        validation="Leave one base task out; both benchmarks and all initial states of that task stay together",
        tasks=sorted({e["base_task_id"] for e in summary["episodes"]}),
        no_query_level_split=True, no_padding_after_termination=True,
        runtime_inputs="Current complete routing tensor only; no task, outcome, horizon or wall-clock fields",
        caveats=[
            "Nine successes only, all from Plus; Pro false-alarm rate is unidentifiable",
            "The entire cohort was inspected before candidate design, so task-held-out selection is not a blind test",
            "Mid-episode metrics allow naturally terminated successes and therefore unequal observed lengths",
            "A remaining step budget does not establish physical recoverability or failure-onset timing",
            "Candidate thresholds are developed on these samples; the underlying policy is not fine-tuned",
        ],
        source_sha256={name: sha256_file(source / name) for name in ("summary.json", "signals.csv", "plan.json")},
        detector_sha256=frozen_sources(), development_code_sha256=source_files(),
        pilot_check=dict(path=str(PILOT), summary_sha256=sha256_file(PILOT / "summary.json"),
            used_for_selection=False, status="Previously collected pilot; separate traces, not a new blind cohort"))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", protocol)
    print(json.dumps(dict(event="development_protocol_saved", output=str(output))), flush=True)


def first_confirmed(hits, confirmation=1):
    hits = np.asarray(hits, bool)
    if len(hits) < confirmation:
        return -1
    counts = np.convolve(hits.astype(int), np.ones(confirmation, int), mode="valid")
    indices = np.flatnonzero(counts == confirmation)
    return int(indices[0] + confirmation - 1) if len(indices) else -1


def vectorized_first(history, profile, threshold):
    q = np.arange(len(history))
    freeze = np.asarray([s["freeze_score"] for s in history])
    curvature = np.asarray([s["v8_scores"][1] for s in history])
    curvature_threshold = np.asarray([s["v82_thresholds"][1] for s in history])
    starts = [first_confirmed([s["freeze_alarm"] for s in history]),
              first_confirmed([s["turbulence_alarm"] for s in history]),
              first_confirmed([s["v82_head_first"][0] >= 0 for s in history]),
              first_confirmed((q >= 6) & np.isfinite(curvature) & (curvature >= curvature_threshold),
                              profile.curvature_confirm)]
    if profile.weak_freeze_ratio is not None:
        starts.append(first_confirmed((q >= 6) & np.isfinite(freeze) &
            (freeze >= profile.weak_freeze_ratio * threshold), profile.weak_freeze_confirm))
    return min((s for s in starts if s >= 0), default=-1)


def replay(source, output, episodes, profiles):
    first = np.full((len(profiles), len(episodes)), -1, int)
    histories, query_records = {}, []
    for i, episode in enumerate(episodes):
        name = episode["name"]
        archive = source / name / "full-hb-routes.npz"
        if sha256_file(archive) != episode["integrity"]["full_hb_sha256"]:
            raise ValueError("Route archive changed: " + name)
        with np.load(archive, allow_pickle=False) as data:
            probabilities = data["hb_router_probs"]
        with (source / name / "signals.csv").open(newline="") as stream:
            original = list(csv.DictReader(stream))
        if probabilities.shape != (episode["queries"], 8, 10, 11, 32) or len(original) != len(probabilities):
            raise ValueError("Incomplete source routing history")
        frozen, history = V82Monitor(), []
        threshold = frozen.config["v7"]["freeze_threshold"]
        states = [SustainedAlarmState(p, threshold) for p in profiles]
        for q, probability in enumerate(probabilities):
            status = frozen.update(probability)
            expected_scores = [float(original[q][s + "_score"]) if original[q][s + "_score"] else np.nan
                               for s in SIGNALS]
            np.testing.assert_allclose(score_status(status), expected_scores, rtol=0, atol=0, equal_nan=True)
            if bool(status["v82_alarm"]) != (original[q]["v82_alarm"] == "True"):
                raise ValueError("Frozen v8.2 source alarm mismatch")
            if int(original[q]["query"]) != q or int(original[q]["action_step_before"]) != q * 10:
                raise ValueError("Unexpected query/action alignment")
            for j, state in enumerate(states):
                alarm = state.update(status)
                if j == 0 and alarm["alarm"] != bool(status["v82_alarm"]):
                    raise ValueError("Reference candidate differs from original v8.2")
                query_records.append(dict(episode=name, profile=profiles[j].name, query=q,
                    alarm=alarm["alarm"], first_alarm_query=alarm["first_alarm_query"],
                    state=alarm["state"], weak_freeze_streak=alarm["evidence"]["weak_freeze_streak"],
                    curvature_streak=alarm["evidence"]["curvature_streak"]))
            history.append(status)
        for j, state in enumerate(states):
            first[j, i] = state.first_alarm_query
            if first[j, i] != vectorized_first(history, profiles[j], threshold):
                raise ValueError("Online state and independent window calculation disagree")
        if first[0, i] != episode["first_alarm_query_zero_based"]["v82"]:
            raise ValueError("Original first alarm does not reproduce")
        histories[name] = history
        if (i + 1) % 10 == 0:
            print(json.dumps(dict(event="candidates_replayed", episodes=i + 1)), flush=True)
    save_csv(output / "candidate-query-states.csv", query_records)
    return first, histories


def alarm_by(first, query):
    first = np.asarray(first)
    return (first >= 0) & ((first <= query) if query >= 0 else True)


def select_profile(first, labels, indices, profiles):
    local_labels = labels[indices]
    keys = []
    for i, p in enumerate(profiles):
        calls = first[i, indices]
        fp = int((alarm_by(calls, -1) & ~local_labels).sum())
        tp_mid = int((alarm_by(calls, PRIMARY_QUERY) & local_labels).sum())
        tp_full = int((alarm_by(calls, -1) & local_labels).sum())
        changes = int(p.weak_freeze_ratio is not None) + int(p.curvature_confirm != 2)
        weakening = 0. if p.weak_freeze_ratio is None else 1. - p.weak_freeze_ratio
        keys.append((fp, -tp_mid, -tp_full, changes, weakening, p.curvature_confirm, p.name))
    return min(range(len(profiles)), key=lambda i: keys[i])


def summarize(first, labels, episodes):
    results = {}
    for q in WINDOWS:
        predicted = alarm_by(first, q)
        window = "full_episode" if q < 0 else "q%d" % q
        results[window] = {}
        for benchmark in ("all", "plus", "pro"):
            subset = np.asarray([benchmark == "all" or e["benchmark"] == benchmark for e in episodes])
            results[window][benchmark] = confusion(labels[subset], predicted[subset])
    detected = labels & (first >= 0)
    results["timing_on_detected_failures"] = dict(
        count=int(detected.sum()), median_action_step=float(np.median(first[detected]) * 10) if detected.any() else None,
        median_steps_left_to_520=float(np.median(520 - first[detected] * 10)) if detected.any() else None)
    return results


def paired_changes(reference, candidate, labels):
    results = {}
    for q in WINDOWS:
        old, new = alarm_by(reference, q), alarm_by(candidate, q)
        window = "full_episode" if q < 0 else "q%d" % q
        results[window] = dict(new_failure_detections=int((labels & ~old & new).sum()),
            lost_failure_detections=int((labels & old & ~new).sum()),
            new_false_alarms=int((~labels & ~old & new).sum()),
            removed_false_alarms=int((~labels & old & ~new).sum()))
    return results


def verify_exported_monitor(source, episodes, first, profile_path):
    queries = 0
    for i, episode in enumerate(episodes):
        with np.load(source / episode["name"] / "full-hb-routes.npz", allow_pickle=False) as archive:
            probabilities = archive["hb_router_probs"]
        monitor = V82AlarmCandidate.from_profile(profile_path)
        results = [monitor.update(p) for p in probabilities]
        for q, status in enumerate(results):
            if status["alarm"] != bool(first[i] >= 0 and q >= first[i]):
                raise ValueError("Exported monitor differs from development replay")
        for q in sorted({15, min(PRIMARY_QUERY, len(probabilities) - 1)}):
            prefix_monitor = V82AlarmCandidate.from_profile(profile_path)
            for probability in probabilities[:q + 1]:
                truncated = prefix_monitor.update(probability)
            if truncated != results[q]:
                raise ValueError("Truncated-prefix monitor differs from full replay")
        queries += len(results)
    return dict(episodes=len(episodes), queries=queries, online_export_exact=True, truncated_prefixes_exact=True)


def pilot_check(profile_path, main_episodes):
    pilot = json.loads((PILOT / "summary.json").read_text())
    main_traces = {e["source_trace_sha256"] for e in main_episodes}
    results = []
    for episode in pilot["episodes"]:
        if episode["source_trace_sha256"] in main_traces:
            raise ValueError("Pilot and development traces overlap")
        archive = PILOT / episode["name"] / "full-hb-routes.npz"
        if sha256_file(archive) != episode["integrity"]["full_hb_sha256"]:
            raise ValueError("Pilot archive identity changed")
        monitor = V82AlarmCandidate.from_profile(profile_path)
        with np.load(archive, allow_pickle=False) as data:
            for probability in data["hb_router_probs"]:
                status = monitor.update(probability)
        results.append(dict(name=episode["name"], success=episode["source_result"]["success"],
            reference_first_query=episode["first_alarm_query_zero_based"]["v82"],
            candidate_first_query=status["first_alarm_query"], candidate_alarm=status["alarm"],
            source_trace_sha256=episode["source_trace_sha256"]))
    return results


def run(source, output):
    protocol = json.loads((output / "protocol.json").read_text())
    if source_files() != protocol["development_code_sha256"] or frozen_sources() != protocol["detector_sha256"]:
        raise ValueError("Analysis or frozen source changed after protocol creation")
    for name, expected in protocol["source_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError("Source evaluation changed")
    if sha256_file(PILOT / "summary.json") != protocol["pilot_check"]["summary_sha256"]:
        raise ValueError("Pilot summary changed")
    summary = json.loads((source / "summary.json").read_text())
    episodes, profiles = summary["episodes"], candidate_profiles()
    labels = np.asarray([not e["source_result"]["success"] for e in episodes], bool)
    tasks = np.asarray([e["base_task_id"] for e in episodes], int)
    first, _ = replay(source, output, episodes, profiles)
    oof_first, assignments = np.full(len(episodes), -1, int), np.full(len(episodes), -1, int)
    folds = []
    for task in sorted(set(tasks.tolist())):
        train, test = np.flatnonzero(tasks != task), np.flatnonzero(tasks == task)
        selected = select_profile(first, labels, train, profiles)
        altered_labels = labels.copy()
        altered_labels[test] = ~altered_labels[test]
        altered_first = first.copy()
        altered_first[:, test] = 0
        if select_profile(altered_first, altered_labels, train, profiles) != selected:
            raise ValueError("Held-out values influence parameter selection")
        oof_first[test], assignments[test] = first[selected, test], selected
        folds.append(dict(held_out_task=task, train_episodes=len(train), test_episodes=len(test),
            selected_profile=profiles[selected].name, parameters=profiles[selected].to_dict(),
            test_episode_names=[episodes[i]["name"] for i in test],
            test_metrics=summarize(first[selected, test], labels[test], [episodes[i] for i in test]),
            reference_test_metrics=summarize(first[0, test], labels[test], [episodes[i] for i in test]),
            held_out_values_cannot_change_selection=True))
    if np.any(assignments < 0):
        raise ValueError("An episode lacks a held-out prediction")
    selected = select_profile(first, labels, np.arange(len(episodes)), profiles)
    selected_profile = profiles[selected]
    profile_path = output / "candidate-profile.json"
    save_json(profile_path, dict(schema="local.v82.alarm_candidate_profile.v1",
        name=selected_profile.name, parameters=selected_profile.to_dict(),
        status="Experimental alarm-only profile; not automatically activated in the policy service",
        selection_scope="Refit on all 60 development episodes after task-held-out evaluation",
        independent_blind_test=False, checkpoint_sha256=episodes[0]["policy_identity"]["checkpoint_sha256"],
        frozen_source_sha256=protocol["detector_sha256"], development_source_sha256=source_files(),
        input_shape=[8, 10, 11, 32], reset_between_episodes=True))
    reference = summarize(first[0], labels, episodes)
    oof = summarize(oof_first, labels, episodes)
    fitted = summarize(first[selected], labels, episodes)
    ref_primary, oof_primary = reference["q36"]["all"], oof["q36"]["all"]
    improves = bool(oof_primary["tp"] > ref_primary["tp"] and
        oof["full_episode"]["all"]["fp"] <= reference["full_episode"]["all"]["fp"] and
        oof["full_episode"]["all"]["tp"] >= reference["full_episode"]["all"]["tp"])
    episode_rows, grid_rows = [], []
    for i, episode in enumerate(episodes):
        episode_rows.append(dict(episode=episode["name"], benchmark=episode["benchmark"],
            base_task_id=episode["base_task_id"], failed=bool(labels[i]), queries=episode["queries"],
            original_first_action_step=int(first[0, i] * 10) if first[0, i] >= 0 else None,
            held_out_profile=profiles[assignments[i]].name,
            held_out_first_action_step=int(oof_first[i] * 10) if oof_first[i] >= 0 else None,
            fitted_first_action_step=int(first[selected, i] * 10) if first[selected, i] >= 0 else None))
        for j, profile in enumerate(profiles):
            grid_rows.append(dict(episode=episode["name"], profile=profile.name,
                benchmark=episode["benchmark"], base_task_id=episode["base_task_id"], failed=bool(labels[i]),
                first_query=int(first[j, i]), first_action_step=int(first[j, i] * 10) if first[j, i] >= 0 else None))
    save_csv(output / "episode-comparison.csv", episode_rows)
    save_csv(output / "all-candidate-episodes.csv", grid_rows)
    grid = {p.name: summarize(first[i], labels, episodes) for i, p in enumerate(profiles)}
    save_json(output / "folds.json", folds)
    save_json(output / "candidate-grid.json", grid)
    print(json.dumps(dict(event="task_held_out_complete", selected=selected_profile.name,
        original_tp_by_360=ref_primary["tp"], held_out_tp_by_360=oof_primary["tp"],
        held_out_full_fp=oof["full_episode"]["all"]["fp"])), flush=True)
    online_check = verify_exported_monitor(source, episodes, first[selected], profile_path)
    pilot = pilot_check(profile_path, episodes)
    if frozen_sources() != protocol["detector_sha256"] or source_files() != protocol["development_code_sha256"]:
        raise ValueError("Source code changed during evaluation")
    result = dict(complete=True, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        study_status=protocol["study_status"], independent_blind_test=False,
        source=str(source), development_episodes=len(episodes), failures=int(labels.sum()), successes=int((~labels).sum()),
        selected_profile=selected_profile.name, selected_parameters=selected_profile.to_dict(),
        original=reference, task_held_out=oof, all_cohort_fitted=fitted,
        task_held_out_paired_changes=paired_changes(first[0], oof_first, labels),
        fitted_paired_changes=paired_changes(first[0], first[selected], labels),
        task_held_out_improvement_gate_passed=improves,
        interpretation="Promising development candidate" if improves else "No task-held-out improvement established",
        fold_selection_counts={p.name: sum(f["selected_profile"] == p.name for f in folds) for p in profiles},
        pilot_checks=pilot,
        verification=dict(original_v82_exact=True, all_nine_candidates_match_independent_window_calculation=True,
            task_disjoint_folds=True, held_out_selection_poison_checks_passed=True,
            frozen_source_unchanged=True, online_profile=online_check),
        limitations=protocol["caveats"])
    save_json(output / "results.json", result)
    report(output, result)
    print(json.dumps(dict(event="alarm_development_complete", output=str(output),
        selected_profile=selected_profile.name, original=reference["full_episode"]["all"],
        held_out=oof["full_episode"]["all"], fitted=fitted["full_episode"]["all"])), flush=True)


def report(output, result):
    lines = ["# Local v8.2 alarm development", "", result["interpretation"] + ".", "",
        "This is alarm-only development on 60 previously inspected episodes (51 failures, nine successes). "
        "The policy is not trained or changed; no recovery actions or new simulator rollouts are executed. "
        "Original v8.2 remains unchanged. Candidate design used existing outcome diagnostics, so even "
        "the task-held-out results are exploratory, not a new blind test.", "",
        "## Change", "",
        "The original strong freeze, turbulence, and inversion branches are preserved. A new optional "
        "weak-freeze branch requires four consecutive frozen freeze scores above a lower relative threshold. "
        "Curvature can require two, four, or six consecutive crossings of its original v8.2 moving threshold. "
        "An alarm remains latched until episode reset; clearing evidence never removes a counted false alarm.", "",
        "Nine configurations were specified before evaluating their results: weak-freeze ratio off / 0.8 / 0.7, "
        "crossed with curvature confirmation 2 / 4 / 6. No new input features, task-specific parameters, "
        "horizon inputs, or wall-clock inputs are added.", "",
        "Selected development profile: `" + result["selected_profile"] + "`.", "",
        "```json", json.dumps(result["selected_parameters"], indent=2), "```", "",
        "## Task-held-out evaluation", "",
        "For each of ten folds, both benchmarks and all initial states of one base task are held out together. "
        "Parameter choice uses only the other nine tasks: minimize full-episode false alarms, then maximize "
        "failure detections by 360 actions, then full-episode detections, with conservative fixed tie breaks. "
        "The held-out column combines those ten independently selected fold profiles. The fitted column "
        "uses one profile chosen on all 60 episodes and is an in-sample development result.", "",
        "| Observation deadline | Original v8.2 TP / 51 | Task-held-out TP / 51 | All-cohort fitted TP / 51 |",
        "|---|---:|---:|---:|"]
    for q in WINDOWS:
        key = "full_episode" if q < 0 else "q%d" % q
        label = "Full observed episode" if q < 0 else "%d actions" % (q * 10)
        counts = [str(result[k][key]["all"]["tp"]) for k in ("original", "task_held_out", "all_cohort_fitted")]
        lines.append("| " + label + " | " + " | ".join(counts) + " |")
    fps = [str(result[k]["full_episode"]["all"]["fp"]) for k in ("original", "task_held_out", "all_cohort_fitted")]
    lines += ["| Full-episode false alarms / 9 | " + " | ".join(fps) + " |", "",
        "The primary deadline of 360 actions leaves 160 of the 520 action slots. This is observation-budget "
        "headroom, not demonstrated recoverability or a labeled failure onset. Mid-episode comparisons "
        "respect natural termination; no successes are padded with artificial future routes. Consequently "
        "they are not equal-length-prefix discrimination tests after the shortest success has ended.", "",
        "## Paired changes", "",
        "Task-held-out changes at the primary deadline:", "", "```json",
        json.dumps(result["task_held_out_paired_changes"]["q36"], indent=2), "```", "",
        "Task-held-out changes over the complete observed episode:", "", "```json",
        json.dumps(result["task_held_out_paired_changes"]["full_episode"], indent=2), "```", "",
        "## Verification", "",
        "All original v8.2 scores and alarms reproduce exactly from the hashed complete routing archives. "
        "All nine state-machine candidates agree with an independent sliding-window first-alarm calculation. "
        "Every episode appears in exactly one task-held-out fold. Deliberately altering held-out outcomes "
        "and candidate predictions cannot change that fold's selected parameters.", "",
        "The exported online monitor is replayed over all 2,891 real queries and checked against development "
        "predictions. Fresh truncated-prefix monitors give identical outputs. Seven focused state tests cover "
        "persistence, interrupted evidence, warmup, latching, reset, retained original branches and invalid inputs.", "",
        "Three earlier pilot traces are disjoint from the development cohort and were not used for selection. "
        "They are a small additional check, not a new blind benchmark:", "",
        "| Pilot | Outcome | Original first alarm action | Candidate first alarm action |",
        "|---|---|---:|---:|"]
    for p in result["pilot_checks"]:
        old = str(p["reference_first_query"] * 10) if p["reference_first_query"] >= 0 else "none"
        new = str(p["candidate_first_query"] * 10) if p["candidate_first_query"] >= 0 else "none"
        lines.append(f"| {p['name']} | {'success' if p['success'] else 'failure'} | {old} | {new} |")
    lines += ["", "## Use", "",
        "The candidate is available for shadow alarming; the existing policy service is not changed.", "",
        "```python", "import sys", "sys.path.insert(0, '/data/coding/v8-methods')",
        "from v82_alarm_candidate import V82AlarmCandidate", "",
        "monitor = V82AlarmCandidate.from_profile(" + repr(str(output / "candidate-profile.json")) + ")",
        "status = monitor.update(hb_router_probs)  # Current [8, 10, 11, 32] tensor",
        "alarm = status['alarm']", "reasons = status['first_alarm_reasons']",
        "# New episode:", "monitor.reset()", "```", "",
        "WARMUP is insufficient history, WATCH is unconfirmed evidence, and ALARM is latched. These states "
        "are not calibrated probabilities. The exported profile is exploratory and is not automatically "
        "activated in any inference or control service.", "",
        "## Limits", ""]
    lines.extend("- " + caveat for caveat in result["limitations"])
    lines += ["- Fewer false alarms in nine successes does not establish a low population false-alarm rate.",
              "- Independent new trajectories are needed before claiming generalization or replacing the reference.",
              "", "Detailed counts, confidence intervals, selected folds, per-episode differences, all candidate "
              "states, profile parameters and code/data hashes are saved alongside this report.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    (plan if args.command == "plan" else run)(args.source, args.output)
