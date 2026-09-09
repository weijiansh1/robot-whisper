"""Evaluate sealed MoE predictions using SAFE/VLAConf-style endpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import (ROOT, SEED, CHECKPOINTS, PRIMARY, binary_metrics, digest,
                  prefix_aggregate, score_at, write_json)

HERE = Path(__file__).resolve().parent
PHYSICAL = ROOT / "VLA_MUI_HUB/physical-failure-labels/results"


def verify_seal(output: Path) -> dict:
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"changed sealed prediction: {name}")
    for name, expected in manifest["sources"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"changed sealed source: {name}")
    return manifest


def load_index(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        frame = pd.DataFrame({k: data[k] for k in
                              ("task", "episode", "init_state_id", "flow_noise_seed", "length", "run_id")})
        valid = data["valid"]
    frame["suite"] = frame.task.str.split("/", n=1).str[0]
    frame["row"] = np.arange(len(frame))
    return frame, valid


def join_labels(index: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(PHYSICAL / "episodes.csv")
    labels["task"] = labels.suite + "/" + labels.task_name
    labels = labels.rename(columns={"episode_index": "episode"})
    keys = ["run_id", "task", "episode"]
    needed = pd.MultiIndex.from_frame(index[keys])
    labels = labels.loc[pd.MultiIndex.from_frame(labels[keys]).isin(needed)]
    if labels.duplicated(keys).any():
        raise ValueError("ambiguous physical label key")
    joined = index.merge(labels[[*keys, "recorded_success", "primary_failure_reason",
                                 "init_state_id", "flow_noise_seed"]], on=keys,
                         how="left", validate="one_to_one", suffixes=("", "_label")).sort_values("row")
    if joined.recorded_success.isna().any():
        missing = joined.loc[joined.recorded_success.isna(), keys].head().to_dict("records")
        raise ValueError(f"unmatched outcome labels: {missing}")
    for field in ("init_state_id", "flow_noise_seed"):
        if not np.array_equal(joined[field], joined[field + "_label"]):
            raise ValueError(f"outcome metadata mismatch: {field}")
    if not pd.api.types.is_bool_dtype(joined.recorded_success):
        raise ValueError("unexpected success encoding")
    joined["failure"] = ~joined.recorded_success
    return joined.reset_index(drop=True)


def ranking_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Tie-aware ROC-AUC and average precision, sharing one stable sort."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return np.nan, np.nan
    order = np.argsort(-score, kind="stable")
    s, truth = score[order], y[order]
    end = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(truth)[end]
    fp = end + 1 - tp
    positive = np.diff(np.r_[0, tp])
    negative = np.diff(np.r_[0, fp])
    auc = np.sum(positive * (n - fp + 0.5 * negative)) / (p * n)
    ap = np.sum(positive * tp / (end + 1)) / p
    return float(auc), float(ap)


def ranking_metrics(y, score, tasks, eligible, groups) -> dict:
    finite = eligible & np.isfinite(score)
    auc, ap = ranking_pair(y[finite], score[finite])
    macro = []
    for rows in groups:
        take = rows[finite[rows]]
        if len(take):
            a, b = ranking_pair(y[take], score[take])
            if np.isfinite(a):
                macro.append((a, b))
    average = np.mean(macro, axis=0) if macro else (np.nan, np.nan)
    return {"eligible_episodes": int(eligible.sum()), "scored_episodes": int(finite.sum()),
            "overall_coverage": float(finite.mean()),
            "eligible_score_coverage": float(finite.sum() / max(eligible.sum(), 1)),
            "scored_failures": int((finite & y).sum()),
            "scored_successes": int((finite & ~y).sum()),
            "failure_coverage": float((finite & y).sum() / max(y.sum(), 1)),
            "auc": auc, "ap": ap, "task_macro_auc": float(average[0]),
            "task_macro_ap": float(average[1]), "macro_tasks": len(macro)}


def hierarchical_weights(frame: pd.DataFrame, draws: int):
    key = frame.task + ":" + frame.init_state_id.astype(str)
    group, unique = pd.factorize(key, sort=True)
    task_groups = [np.unique(group[frame.task.to_numpy() == task]) for task in sorted(frame.task.unique())]
    rng = np.random.default_rng(SEED)
    weights = np.zeros((draws, len(unique)), dtype=np.float64)
    for b in range(draws):
        selected = rng.integers(len(task_groups), size=len(task_groups))
        for task_i, count in enumerate(np.bincount(selected, minlength=len(task_groups))):
            if count:
                ids = task_groups[task_i]
                sample = rng.choice(ids, size=len(ids) * int(count), replace=True)
                weights[b] += np.bincount(sample, minlength=len(unique))
    return group, weights


def rate_intervals(first, y, length, group, weights):
    fired = first >= 0
    columns = [fired & y, fired & ~y, y, ~y,
               fired & y & ((length - 1 - first) >= 4)]
    counts = np.stack([np.bincount(group, weights=c, minlength=weights.shape[1]) for c in columns], axis=1)
    sampled = weights @ counts
    result = {}
    for name, a, b in (("recall", 0, 2), ("fpr", 1, 3), ("recall_lead4", 4, 2)):
        ratios = np.divide(sampled[:, a], sampled[:, b],
                           out=np.full(len(sampled), np.nan), where=sampled[:, b] > 0)
        result[name + "_lo"], result[name + "_hi"] = np.nanquantile(ratios, [0.025, 0.975])
    return result


def release_events(frame: pd.DataFrame):
    key_to_row = {(r.run_id, r.task, int(r.episode)): int(r.row) for r in frame.itertuples()}
    events = []
    matched_failures = 0
    with (PHYSICAL / "failures.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["run_id"], record["suite"] + "/" + record["task_name"], record["episode_index"])
            if key not in key_to_row:
                continue
            row = key_to_row[key]
            if not bool(frame.iloc[row].failure):
                raise ValueError("failure event matched a successful rollout")
            matched_failures += 1
            objects = record["goal_subject_physics"]
            if not isinstance(objects, dict):
                raise ValueError("unexpected object physics schema")
            release = [v["first_release_snapshot"] for v in objects.values()
                       if v.get("first_release_snapshot") is not None]
            if release:
                onset = int(min(release))
                if not 0 <= onset < int(frame.iloc[row].length):
                    raise ValueError("physical onset outside observed queries")
                events.append({"row": row, "onset": onset, "reason": record["primary_failure_reason"]})
    if matched_failures != int(frame.failure.sum()):
        raise ValueError("incomplete physical failure coverage")
    return pd.DataFrame(events, columns=["row", "onset", "reason"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results/round1")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    output = args.input.resolve()
    started = time.perf_counter()
    print("Verifying prediction and source hashes before opening outcomes", flush=True)
    manifest = verify_seal(output)
    index, valid = load_index(output / "test_index.npz")
    frame = join_labels(index)
    reference_index, _ = load_index(output / "reference_index.npz")
    reference_frame = join_labels(reference_index)
    frame.to_csv(output / "label_alignment.csv", index=False)
    write_json(output / "label_audit.json", {
        "test_episodes": len(frame), "test_failures": int(frame.failure.sum()),
        "test_tasks": frame.task.nunique(), "reference_episodes": len(reference_frame),
        "reference_failures": int(reference_frame.failure.sum()),
        "outcomes_opened_after_seal": True,
        "outcome_definition": "original recorded success, without timeout extension",
        "labels_sha256": digest(PHYSICAL / "episodes.csv")})
    print(f"Labels aligned: test={len(frame)}, failures={frame.failure.sum()}, "
          f"reference={len(reference_frame)}", flush=True)
    y = frame.failure.to_numpy(bool)
    length = frame.length.to_numpy(int)
    tasks = frame.task.to_numpy(str)
    groups = [np.flatnonzero(tasks == task) for task in sorted(frame.task.unique())]
    minimum = np.empty(len(frame), dtype=int)
    for rows in groups:
        minimum[rows] = length[rows].min() - 1
    group, weights = hierarchical_weights(frame, args.bootstrap)
    events = release_events(frame)
    events.to_csv(output / "release_event_index.csv", index=False)
    alarm_file = np.load(output / "first_alarms.npz", allow_pickle=False)
    alarm_names = alarm_file["method_names"].astype(str).tolist()
    budgets = alarm_file["budgets"]
    ranking, alarms, suites, modes, event_rows = [], [], [], [], []
    for setting_info in manifest["settings"]:
        setting = setting_info["setting"]
        print(f"Evaluating {setting}", flush=True)
        with np.load(output / f"{setting}_scores.npz", allow_pickle=False) as archive:
            score_bank = archive["scores"]
            score_names = archive["method_names"].astype(str).tolist()
        for method_i, name in enumerate(score_names):
            score = score_bank[method_i]
            checkpoints = [(f"q{q}", np.full(len(frame), q), length > q) for q in CHECKPOINTS]
            checkpoints.append(("retrospective_half", (length - 1) // 2, np.ones(len(frame), dtype=bool)))
            for point, positions, eligible in checkpoints:
                values = score_at(score, positions)
                ranking.append({"setting": setting, "method": name, "checkpoint": point,
                                **ranking_metrics(y, values, tasks, eligible, groups)})
            if not name.endswith("__max"):
                maximum = prefix_aggregate(score, "max")
                for point, positions in (("safe_task_min_prefixmax", minimum),
                                         ("full_rollout_prefixmax", length - 1)):
                    ranking.append({"setting": setting, "method": name, "checkpoint": point,
                                    **ranking_metrics(y, score_at(maximum, positions), tasks,
                                                      np.ones(len(frame), dtype=bool), groups)})
        all_first = alarm_file[setting]
        for budget_i, budget in enumerate(budgets):
            for method_i, name in enumerate(alarm_names):
                first = all_first[budget_i, method_i]
                available = first != -2
                if np.any((first >= 0) & (first >= length)):
                    raise ValueError("alarm after termination")
                row = {"setting": setting, "method": name, "budget": budget,
                       "available_episodes": int(available.sum()),
                       **binary_metrics(first[available], y[available], length[available])}
                if setting in ("cohort_transfer", "state_heldout", "task_heldout", "suite_heldout") and np.isclose(budget, 0.03) and available.all():
                    row.update(rate_intervals(first, y, length, group, weights))
                alarms.append(row)
                if np.isclose(budget, 0.03):
                    for suite in sorted(frame.suite.unique()):
                        take = (frame.suite.to_numpy() == suite) & available
                        suites.append({"setting": setting, "method": name, "suite": suite,
                                       **binary_metrics(first[take], y[take], length[take])})
                    if setting in ("cohort_transfer", "task_heldout"):
                        for reason in sorted(frame.loc[frame.failure, "primary_failure_reason"].unique()):
                            take = frame.primary_failure_reason.to_numpy() == reason
                            take &= available
                            modes.append({"setting": setting, "method": name, "reason": reason,
                                          **binary_metrics(first[take], y[take], length[take])})
                        for scope in ("all_observed_releases_in_failures", "drop_related_primary_reason"):
                            event_subset = events
                            if scope == "drop_related_primary_reason":
                                event_subset = events[events.reason.isin(("object_released_or_dropped_before_goal",
                                                                         "object_released_outside_goal"))]
                            rows = event_subset.row.to_numpy(int)
                            onset = event_subset.onset.to_numpy(int)
                            detected = first[rows] >= 0
                            delta = first[rows] - onset
                            event = {"setting": setting, "method": name, "scope": scope,
                                     "events": len(rows), "detected": int(detected.sum()),
                                     "median_delay_detected": float(np.median(delta[detected])) if detected.any() else np.nan,
                                     "within2_fraction_all_events": float((detected & (np.abs(delta) <= 2)).mean()) if len(rows) else np.nan}
                            for lead in (0, 2, 4):
                                event[f"recall_before{lead}"] = float((detected & (delta <= -lead)).mean()) if len(rows) else np.nan
                            event_rows.append(event)
        del score_bank
    tables = {"ranking_metrics.csv": ranking, "alarm_metrics.csv": alarms,
              "suite_metrics.csv": suites, "failure_modes.csv": modes,
              "physical_timing.csv": event_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False)
    alarm_frame = pd.DataFrame(alarms)
    primary = alarm_frame[(alarm_frame.method == PRIMARY) & np.isclose(alarm_frame.budget, 0.03)]
    print(primary[["setting", "tp", "fp", "recall", "fpr", "recall_lead4", "t_det"]].to_string(index=False), flush=True)
    write_json(output / "evaluation_summary.json", {
        "schema": manifest["schema"], "primary": primary.to_dict("records"),
        "test_episodes": len(frame), "test_failures": int(y.sum()), "test_tasks": len(groups),
        "physical_releases_in_failures": len(events), "bootstrap_draws": args.bootstrap,
        "bootstrap_unit": "hierarchical task then task/init; all noise branches retained",
        "lead_definition": "last observed query index (length-1) minus alarm query",
        "t_det_definition": "alarm query / max(length-1,1), with misses assigned 1",
        "probability_calibration_run": False,
        "not_run": ["calibrated success probability", "controlled LIBERO-Pro/Plus shifts",
                    "real robot", "prospective expert assistance", "SAFE final-layer baseline"],
        "evaluation_seconds": time.perf_counter() - started,
        "evaluation_source_sha256": digest(Path(__file__)),
        "prediction_manifest_sha256": digest(output / "sealed_manifest.json"),
        "result_hashes": {name: digest(output / name) for name in tables}})


if __name__ == "__main__":
    main()
