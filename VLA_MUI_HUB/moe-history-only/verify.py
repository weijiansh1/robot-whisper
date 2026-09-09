#!/usr/bin/env python3
"""Raw-stream parity, coverage, clustered uncertainty, and event specificity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from analyze import HUB, RESULTS, auc, digest_file, episode_metrics, load_features, source_inventory, write_json
from monitor import HistoryOnlyMonitor, PRIMARY, first_alarm, fixed_rules, score_stream


def raw_replay(frame: pd.DataFrame, audits: list[dict]) -> dict:
    rng = np.random.default_rng(20260906)
    sampled = []
    max_error, steps_checked = 0., 0
    for audit in audits:
        group = frame[frame.source_run == audit["source_run"]]
        # Every source run contributes one label-blind sampled episode/subtask.
        local = int(rng.integers(len(group)))
        row = group.iloc[local]
        offset = int(group.length.iloc[:local].sum())
        length = int(row.length)
        store = zarr.open_group(str(HUB / row.source_run / "server/routes.zarr"), mode="r")
        raw = np.asarray(store["hb_router_probs"][offset:offset+length])
        with np.load(RESULTS / "features" / audit["cache_file"], allow_pickle=False) as archive:
            block = archive["features"][local:local+1, :length]
        for rule in [r for r in fixed_rules() if r.name in (PRIMARY.name, "freeze_back_half_k4")]:
            monitor = HistoryOnlyMonitor(rule)
            actual = np.asarray([monitor.update(query)["score"] for query in raw])
            expected = score_stream(block, rule)[0]
            np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=2e-6, atol=2e-7)
            a = first_alarm(actual[None], rule.threshold)[0]
            b = first_alarm(expected[None], rule.threshold)[0]
            assert a == b == int(row[rule.name])
            finite = np.isfinite(actual) & np.isfinite(expected)
            if finite.any():
                max_error = max(max_error, float(np.abs(actual[finite]-expected[finite]).max()))
        sampled.append(dict(source_run=row.source_run, episode=int(row.episode), row=int(row.row)))
        steps_checked += length
    return dict(source_runs=len(audits), episodes_checked=len(sampled), queries_checked=steps_checked,
                rules_checked=[PRIMARY.name, "freeze_back_half_k4"],
                max_float32_cache_score_error=max_error, identical_first_alarms=True, samples=sampled)


def cluster_intervals(frame: pd.DataFrame, rule_name: str) -> list[dict]:
    primary = frame[rule_name].to_numpy(int)
    labels = ~frame.success.to_numpy(bool)
    hit = primary >= 0
    data = frame.copy()
    data["tp"] = (hit & labels).astype(int)
    data["fp"] = (hit & ~labels).astype(int)
    data["nf"] = labels.astype(int)
    data["ns"] = (~labels).astype(int)
    data["early_tp"] = (hit & labels & ((primary+1)/frame.length.to_numpy() <= 0.65)).astype(int)
    data["half_tp"] = (hit & labels & ((primary+1)/frame.length.to_numpy() <= 0.5)).astype(int)
    rows = []
    rng = np.random.default_rng(20260906)
    for label, columns in (("task", ["suite", "task"]), ("task_init", ["suite", "task", "init_state"]),
                           ("run_seed", ["run_id", "seed"])):
        counts = data.groupby(columns)[["tp", "fp", "nf", "ns", "early_tp", "half_tp"]].sum().to_numpy(float)
        weights = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=2000)
        draws = weights @ counts
        for metric, numerator, denominator in (("recall", draws[:,0], draws[:,2]), ("fpr", draws[:,1], draws[:,3]),
                                             ("precision", draws[:,0], draws[:,0]+draws[:,1]),
                                             ("recall_by_065", draws[:,4], draws[:,2]),
                                             ("recall_by_half", draws[:,5], draws[:,2])):
            values = numerator[denominator>0] / denominator[denominator>0]
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append(dict(rule=rule_name, cluster_axis=label, clusters=len(counts), bootstrap_repeats=2000,
                             metric=metric, lower=float(lower), upper=float(upper)))
    return rows


def landmark_uncertainty(frame: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    """Confidence intervals over task/init strata; no threshold selection."""
    rng = np.random.default_rng(20260906)
    out = []
    for (run_id, suite), group in frame.groupby(["run_id", "suite"]):
        for q in (7, 9, 11, 13, 15, 19, 25, 31):
            group_q = group[(group.length > q) & np.isfinite(scores[group.row, q])]
            pairs = []
            for _, sub in group_q.groupby(["task", "init_state"]):
                value, weight = auc(~sub.success.to_numpy(bool), -scores[sub.row, q])
                if weight:
                    pairs.append((value*weight, weight))
            if not pairs:
                continue
            pairs = np.asarray(pairs)
            value = float(pairs[:,0].sum()/pairs[:,1].sum())
            weights = rng.multinomial(len(pairs), np.full(len(pairs), 1/len(pairs)), size=2000)
            draws = weights @ pairs
            lower, upper = np.quantile(draws[:,0]/draws[:,1], [0.025,0.975])
            out.append(dict(run_id=run_id, suite=suite, query=q, matched_auc=value,
                            ci_low=float(lower), ci_high=float(upper), mixed_init_strata=len(pairs),
                            pairs=int(pairs[:,1].sum()), active=len(group_q),
                            active_tasks=int(group_q.task.nunique())))
    return out


def alarm_hazard_control(frame: pd.DataFrame, rule_name: str) -> list[dict]:
    """Compare first alarms with still-running, not-yet-alarmed matched peers.

    The null fixes each (run,task,query)'s number of new alerts and randomizes
    their recipients among eligible episodes. Conditional counts are used only
    to audit specificity; they never enter the monitor.
    """
    out = []
    for (run_id, suite), suite_group in frame.groupby(["run_id", "suite"]):
        for stratification in ("task", "task_init"):
            observed, expected, total, variance = 0, 0., 0, 0.
            keys = ["task"] + (["init_state"] if stratification == "task_init" else [])
            for _, task in suite_group.groupby(keys):
                first = task[rule_name].to_numpy(int)
                labels = ~task.success.to_numpy(bool)
                length = task.length.to_numpy(int)
                for q in np.unique(first[first >= 0]):
                    eligible = (length > q) & ((first < 0) | (first >= q))
                    new = first == q
                    observed += int((new & labels).sum())
                    n, k, p = int(eligible.sum()), int(new.sum()), float(labels[eligible].mean())
                    expected += k*p
                    variance += k*p*(1-p)*(n-k)/(n-1) if n > 1 else 0
                    total += k
            out.append(dict(rule=rule_name, run_id=run_id, suite=suite, stratification=stratification,
                            observed_failure_alarms=observed, matched_expected_failure_alarms=expected,
                            conditional_variance_diagnostic=variance, new_alarms=total,
                            observed_precision=observed/total if total else None,
                            matched_survival_precision=expected/total if total else None,
                            observed_over_expected=observed/expected if expected else None))
    return out


def physical_event_lead(frame: pd.DataFrame, audited_rules: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict]]:
    """Timestamped event proxies are evaluation targets, never detector inputs."""
    lookup = frame.set_index(["source_run", "episode"])
    rows = []
    path = HUB / "physical-failure-labels/results/failures.jsonl"
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["source"]["run"], int(record["episode_index"]))
            if key not in lookup.index:
                continue
            episode = lookup.loc[key]
            reason = record["primary_failure_reason"]
            related = {g["goal_id"] for g in record["goal_failure_labels"] if g["reason"] == reason}
            events = []
            for goal in record["goal_predicates"]:
                if goal["id"] not in related:
                    continue
                if reason == "goal_predicate_regressed":
                    q = goal["first_regression_snapshot"]
                    if q is not None:
                        events.append((int(q), "first_goal_regression"))
                elif reason in ("object_released_or_dropped_before_goal", "object_released_outside_goal"):
                    subject = goal["expression"][1]
                    q = record["goal_subject_physics"].get(subject, {}).get("first_release_snapshot")
                    if q is not None:
                        events.append((int(q), "first_observed_release_after_grasp"))
            event_q, kind = min(events) if events else (-1, "no_timestamped_event_proxy")
            for rule in audited_rules:
                first = int(episode[rule])
                rows.append(dict(row=int(episode.row), source_run=key[0], episode=key[1], suite=episode.suite,
                                 rule=rule, reason=reason, first_alarm=first, observed_event_query=event_q,
                                 event_kind=kind, before_event=(first >= 0 and event_q >= 0 and first < event_q),
                                 event_minus_alarm=(event_q-first) if min(event_q,first) >= 0 else np.nan))
    detail = pd.DataFrame(rows)
    result = []
    for (rule, kind), group in detail.groupby(["rule", "event_kind"]):
        with_alarm = group.first_alarm >= 0
        result.append(dict(rule=rule, event_kind=kind, episodes=len(group), alarmed=int(with_alarm.sum()),
                           strictly_before_event=int(group.before_event.sum()),
                           same_query_as_event=int((with_alarm & (group.first_alarm == group.observed_event_query)).sum()),
                           median_event_minus_alarm=float(group.event_minus_alarm.median()) if group.event_minus_alarm.notna().any() else None))
    return detail, result


def main() -> None:
    frame = pd.read_csv(RESULTS / "episode_alarms.csv")
    source, _ = source_inventory()
    for column in ("source_run", "episode", "success", "length"):
        assert np.array_equal(frame[column], source[column]), column
    audits = json.loads((RESULTS / "extraction_audit.json").read_text())
    natural = frame[frame.regime == "libero_natural"]
    assert len(natural) == 34560 and int((~natural.success).sum()) == 1403
    failures = pd.read_csv(RESULTS / "failure_keypoints.csv")
    assert len(failures) == 1403 and (~failures.success).all()
    print("checking raw online monitor against each source run", flush=True)
    parity = raw_replay(frame, audits)
    write_json(RESULTS / "raw_replay_verification.json", parity)
    print("checking source coverage and uncertainty", flush=True)
    audited_rules = (PRIMARY.name, "freeze_back_half", "freeze_all_half", "freeze_back_half_k4")
    pd.DataFrame([row for name in audited_rules for row in cluster_intervals(natural, name)]).to_csv(RESULTS / "cluster_intervals.csv", index=False)
    with np.load(RESULTS / "primary_scores.npz", allow_pickle=False) as archive:
        scores = archive["scores"]
        assert np.array_equal(archive["row"], frame.row)
    pd.DataFrame(landmark_uncertainty(natural, scores)).to_csv(RESULTS / "landmark_intervals.csv", index=False)
    pd.DataFrame([row for name in audited_rules for row in alarm_hazard_control(natural, name)]).to_csv(RESULTS / "matched_alarm_specificity.csv", index=False)
    detail, events = physical_event_lead(natural, audited_rules)
    detail.to_csv(RESULTS / "physical_event_lead.csv", index=False)
    write_json(RESULTS / "physical_event_summary.json", events)
    # A clock is constant in each matched comparison and must have AUC exactly 1/2.
    assert auc(np.array([True, False, True, False]), np.ones(4))[0] == 0.5
    suite_rows = []
    for name in audited_rules:
        for suite, block in natural.groupby("suite"):
            suite_rows.append(dict(rule=name, suite=suite, **episode_metrics(block, block[name].to_numpy())))
    pd.DataFrame(suite_rows).to_csv(RESULTS / "suite_metrics.csv", index=False)
    delayed, deadlines, keys = [], [], []
    for name in audited_rules:
        first = natural[name].to_numpy(int)
        length = natural.length.to_numpy(int)
        lagged = np.where((first >= 0) & (first + 1 < length), first+1, -1)
        delayed.append(dict(rule=name, **episode_metrics(natural, lagged)))
        for q in range(int(length.max())):
            at_q = np.where((first >= 0) & (first <= q), first, -1)
            deadlines.append(dict(rule=name, query=q, **episode_metrics(natural, at_q)))
        for _, row in natural[~natural.success].iterrows():
            q = int(row[name])
            keys.append(dict(rule=name, row=int(row.row), source_run=row.source_run, episode=int(row.episode),
                             suite=row.suite, task=row.task, init_state=int(row.init_state), seed=int(row.seed),
                             first_alarm_query=q, first_alarm_inference_number=q+1 if q>=0 else -1,
                             actions_executed_before_alarm=q*int(row.replan_steps) if q>=0 else -1,
                             length=int(row.length), prefix_only_previous_query_alarm=q+1 if 0<=q+1<int(row.length) and q>=0 else -1,
                             detected_by_half=bool(q>=0 and (q+1)/int(row.length)<=0.5),
                             physical_reason=row.primary_failure_reason))
    pd.DataFrame(delayed).to_csv(RESULTS / "previous_query_only_metrics.csv", index=False)
    pd.DataFrame(deadlines).to_csv(RESULTS / "rule_deadlines.csv", index=False)
    pd.DataFrame(keys).to_csv(RESULTS / "keypoint_index.csv", index=False)
    all_rows = []
    for regime, block in frame.groupby("regime"):
        all_rows.append(dict(regime=regime, **episode_metrics(block, block[PRIMARY.name].to_numpy())))
    write_json(RESULTS / "verification.json", dict(raw_parity=parity["identical_first_alarms"],
               source_coverage_verified=True, all_1403_natural_failures_listed=True,
               clock_matched_auc=0.5, raw_episodes_checked=parity["episodes_checked"],
               regime_metrics=all_rows))
    paths = list(Path(__file__).resolve().parent.glob("*.py")) + list(Path(__file__).resolve().parent.glob("*.md"))
    paths += [p for p in RESULTS.rglob("*") if p.is_file() and p.name != "manifest.json"]
    write_json(RESULTS / "manifest.json", {str(p.relative_to(RESULTS.parent)): digest_file(p) for p in sorted(paths)})
    print(json.dumps(dict(coverage=True, raw_replay=parity["episodes_checked"], matched_clock_auc=0.5)), flush=True)


if __name__ == "__main__":
    main()
