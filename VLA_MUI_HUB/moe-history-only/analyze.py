#!/usr/bin/env python3
"""Rebuild MoE-only features from source stores and audit fixed causal rules."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import zarr

from monitor import FEATURES, PRIMARY, fixed_rules, first_alarm, root_action_routes, features_from_roots, score_stream

HERE = Path(__file__).resolve().parent
HUB = HERE.parent
RESULTS = HERE / "results"
HORIZONS = (5, 7, 9, 11, 13, 15, 19, 25, 31)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_inventory() -> tuple[pd.DataFrame, list[dict]]:
    rows, jobs = [], []
    for path in sorted(HUB.glob("cache*/HiMoE-VLA/libero_*/*/*/client/summaries.json")):
        run = path.parents[1]
        source = str(run.relative_to(HUB))
        records = sorted(json.loads(path.read_text()), key=lambda x: x["episode_index"])
        run_id, task, suite = run.name, run.parent.name, run.parent.parent.name
        if run_id.startswith("right-"):
            regime = "libero_natural"
        elif run_id == "pin-base":
            regime = "duplicate_pin_base"
        else:
            regime = "pin_intervention" if run_id in ("pin-on", "pin-off") else "pin_smoke"
        lengths, episode_ids = [], []
        for record in records:
            length = int(record["inference_calls"])
            if length < 1:
                raise ValueError(f"empty episode: {source}")
            episode = int(record["episode_index"])
            lengths.append(length)
            episode_ids.append(episode)
            rows.append(dict(source_run=source, run_id=run_id, suite=suite, task=task,
                             episode=episode, sequence=-1, subtask=-1,
                             init_state=int(record["init_state_id"]), seed=int(record["flow_noise_seed"]),
                             success=bool(record["success"]), length=length, regime=regime,
                             replan_steps=10))
        jobs.append(dict(source_run=source, lengths=lengths, episode_ids=episode_ids,
                         reconstructed=False, summary_sha256=digest_file(path)))
    for path in sorted(HUB.glob("cache*/HiMoE-VLA/calvin_d/task_D_D/*/client/sequences.jsonl")):
        run = path.parents[1]
        source = str(run.relative_to(HUB))
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        config = json.loads((path.parent / "run-config.json").read_text())
        replan = int(config["replan_steps"])
        lengths = []
        for record in records:
            current = [math.ceil(s["environment_steps"] / replan) for s in record["subtasks"]]
            if sum(current) != int(record["inference_calls"]):
                raise ValueError(f"cannot reconstruct CALVIN subtask boundaries: {source}")
            for subtask, length in zip(record["subtasks"], current):
                episode = len(lengths)
                lengths.append(length)
                rows.append(dict(source_run=source, run_id=run.name, suite="calvin_d", task=subtask["task"],
                                 episode=episode, sequence=int(record["sequence_index"]), subtask=int(subtask["subtask_index"]),
                                 init_state=int(record["sequence_index"]), seed=-1,
                                 success=bool(subtask["success"]), length=length, regime="calvin_partial",
                                 replan_steps=replan))
        jobs.append(dict(source_run=source, lengths=lengths, episode_ids=list(range(len(lengths))),
                         reconstructed=True, summary_sha256=digest_file(path)))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["source_run", "episode"]).any():
        raise ValueError("duplicate episode key")
    frame.insert(0, "row", np.arange(len(frame)))
    return frame, jobs


def extract_run(job: dict) -> dict:
    source = job["source_run"]
    destination = RESULTS / "features" / (hashlib.sha256(source.encode()).hexdigest()[:20] + ".npz")
    input_id = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            if str(archive["input_id"]) != input_id:
                raise ValueError(f"changed input for existing cache: {source}")
            audit = json.loads(str(archive["audit"]))
        return audit
    group = zarr.open_group(str(HUB / source / "server/routes.zarr"), mode="r")
    ids = np.asarray(group["episode_id"][:])
    steps = np.asarray(group["control_step"][:])
    lengths = np.asarray(job["lengths"])
    if len(ids) != sum(lengths) or len(steps) != len(ids) or not np.all(np.diff(steps) == 1):
        raise ValueError(f"row count / time order mismatch: {source}")
    if not job["reconstructed"]:
        if not np.array_equal(ids, np.repeat(job["episode_ids"], lengths)):
            raise ValueError(f"server episode order mismatch: {source}")
    elif not np.all(ids == 0):
        raise ValueError(f"unexpected CALVIN episode_id schema: {source}")
    raw = np.asarray(group["hb_router_probs"][:, :, 9, 1:, :])
    if raw.shape != (sum(lengths), 8, 10, 32):
        raise ValueError(f"router dimensions mismatch: {source}")
    checksum = hashlib.sha256(raw.tobytes()).hexdigest()
    output = np.full((len(lengths), max(lengths), 8, len(FEATURES)), np.nan, dtype=np.float32)
    cursor = 0
    for i, length in enumerate(lengths):
        root = root_action_routes(raw[cursor:cursor + length])
        output[i, :length] = features_from_roots(root)
        cursor += length
    audit = dict(source_run=source, episodes=len(lengths), queries=int(sum(lengths)),
                 server_time_order_verified=True, reconstructed_subtask_boundaries=job["reconstructed"],
                 final_route_sha256=checksum, summary_sha256=job["summary_sha256"], cache_file=destination.name)
    np.savez_compressed(destination, input_id=np.asarray(input_id), features=output,
                        episode_ids=np.asarray(job["episode_ids"]), lengths=lengths,
                        audit=np.asarray(json.dumps(audit)))
    return audit


def load_features(frame: pd.DataFrame, audits: list[dict]) -> np.ndarray:
    features = np.full((len(frame), int(frame.length.max()), 8, len(FEATURES)), np.nan, dtype=np.float32)
    for audit in audits:
        take = np.flatnonzero(frame.source_run.to_numpy() == audit["source_run"])
        with np.load(RESULTS / "features" / audit["cache_file"], allow_pickle=False) as archive:
            if not np.array_equal(archive["episode_ids"], frame.iloc[take].episode):
                raise ValueError("cache episode order changed")
            if not np.array_equal(archive["lengths"], frame.iloc[take].length):
                raise ValueError("cache lengths changed")
            block = archive["features"]
            features[take, :block.shape[1]] = block
    return features


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def median(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    z = 1.95996398454
    center = (k / n + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return max(0., float(center-half)), min(1., float(center+half))


def episode_metrics(frame: pd.DataFrame, first: np.ndarray) -> dict:
    fail = ~frame.success.to_numpy(bool)
    length = frame.length.to_numpy(int)
    hit = (first >= 0) & (first < length)
    tp, fp = int((hit & fail).sum()), int((hit & ~fail).sum())
    nf, ns = int(fail.sum()), int((~fail).sum())
    phase = (first + 1) / length
    lead = length - 1 - first
    return dict(episodes=len(frame), failures=nf, successes=ns, tp=tp, fp=fp,
                recall=rate(tp, nf), fpr=rate(fp, ns), precision=rate(tp, tp+fp),
                recall_ci_low=wilson(tp, nf)[0], recall_ci_high=wilson(tp, nf)[1],
                fpr_ci_low=wilson(fp, ns)[0], fpr_ci_high=wilson(fp, ns)[1],
                first_query_median=median(first[hit & fail]),
                phase_median=median(phase[hit & fail]), lead_median=median(lead[hit & fail]),
                recall_by_half=rate(int((hit & fail & (phase <= 0.5)).sum()), nf),
                recall_by_065=rate(int((hit & fail & (phase <= 0.65)).sum()), nf),
                recall_lead3=rate(int((hit & fail & (lead >= 3)).sum()), nf))


def auc(labels: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    pos = np.asarray(labels, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan"), 0
    ranks = rankdata(values)
    return float((ranks[pos].sum() - n1*(n1+1)/2)/(n1*n0)), n1*n0


def landmark_metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float, name: str) -> list[dict]:
    output = []
    for (regime, run_id, suite), group in frame.groupby(["regime", "run_id", "suite"], sort=True):
        for q in HORIZONS:
            active = group[(group.length > q) & np.isfinite(scores[group.row, q])]
            if active.empty:
                continue
            labels = ~active.success.to_numpy(bool)
            values = -scores[active.row, q]
            hit = values >= -threshold
            pool_auc, _ = auc(labels, values)
            for stratify in ("task", "task_init"):
                strata = ["source_run", "task"] + (["init_state"] if stratify == "task_init" else [])
                weighted, pairs, count, positive, equal = 0., 0, 0, 0, []
                expected_tp, alerts_supported, observed_tp = 0., 0, 0
                for _, sub in active.groupby(strata, sort=False):
                    sublabels = ~sub.success.to_numpy(bool)
                    subvalues = -scores[sub.row, q]
                    value, weight = auc(sublabels, subvalues)
                    alerts = subvalues >= -threshold
                    expected_tp += float(alerts.sum()) * float(sublabels.mean())
                    observed_tp += int((alerts & sublabels).sum())
                    alerts_supported += int(alerts.sum())
                    if weight:
                        weighted += value*weight
                        pairs += weight
                        count += 1
                        positive += int(value > 0.5)
                        equal.append(value)
                output.append(dict(rule=name, regime=regime, run_id=run_id, suite=suite, query=q,
                                   stratification=stratify, active=len(active), failures=int(labels.sum()),
                                   pooled_auc=pool_auc, matched_auc=weighted/pairs if pairs else np.nan,
                                   equal_stratum_auc=np.mean(equal) if equal else np.nan,
                                   pairs=pairs, comparable_strata=count, positive_strata=positive,
                                   alerts=int(hit.sum()), tp=int((hit & labels).sum()),
                                   fpr_active=float(hit[~labels].mean()) if (~labels).any() else np.nan,
                                   matched_expected_tp=expected_tp,
                                   matched_lift=observed_tp/expected_tp if expected_tp else np.nan))
    return output


def clock_controls(frame: pd.DataFrame, first: np.ndarray) -> list[dict]:
    rows = []
    for (run_id, suite), group in frame[frame.regime == "libero_natural"].groupby(["run_id", "suite"]):
        labels = ~group.success.to_numpy(bool)
        length = group.length.to_numpy(int)
        alarms = first[group.row]
        hit = alarms >= 0
        fp = int((hit & ~labels).sum())
        candidates = []
        for q in range(int(length.max())):
            fired = length > q
            clock_fp = int((fired & ~labels).sum())
            if clock_fp <= fp:
                candidates.append((int((fired & labels).sum()), q, clock_fp))
        best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0] if candidates else (0, -1, 0)
        rows.append(dict(run_id=run_id, suite=suite, moe_tp=int((hit & labels).sum()), moe_fp=fp,
                         moe_first_median=median(alarms[hit & labels]),
                         retrospective_clock_tp=best[0], retrospective_clock_q=best[1],
                         retrospective_clock_fp=best[2],
                         moe_before_clock=int((hit & labels & (alarms < best[1])).sum()) if best[1] >= 0 else 0))
    return rows


def attach_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(HUB / "physical-failure-labels/results/episodes.csv")
    labels = labels.rename(columns={"episode_index": "episode"})
    merged = frame.merge(labels[["source_run", "episode", "primary_failure_reason", "recorded_success"]],
                         how="left", on=["source_run", "episode"], validate="one_to_one", sort=False)
    known = merged.regime != "calvin_partial"
    if merged.loc[known, "recorded_success"].isna().any():
        raise ValueError("missing physical-label coverage")
    if not np.array_equal(merged.loc[known, "success"], merged.loc[known, "recorded_success"].astype(bool)):
        raise ValueError("source success labels do not match physical-label table")
    return merged.drop(columns="recorded_success")


def run_analysis(frame: pd.DataFrame, features: np.ndarray) -> dict:
    frame = attach_reasons(frame)
    table, groups, reasons, landmarks, rules = [], [], [], [], fixed_rules()
    alarms = frame.copy()
    primary_scores = None
    for rule in rules:
        scores = score_stream(features, rule)
        first = first_alarm(scores, rule.threshold)
        if np.any((first >= 0) & (first >= frame.length.to_numpy())):
            raise AssertionError("alarm in padded suffix")
        alarms[rule.name] = first
        for (regime, run_id), block in frame.groupby(["regime", "run_id"]):
            table.append(dict(rule=rule.name, regime=regime, run_id=run_id,
                              **episode_metrics(block, first[block.row])))
        natural = frame[frame.regime == "libero_natural"]
        table.append(dict(rule=rule.name, regime="libero_natural", run_id="all",
                          **episode_metrics(natural, first[natural.row])))
        if rule.name == PRIMARY.name:
            primary_scores = scores
            for (run_id, suite, task), block in natural.groupby(["run_id", "suite", "task"]):
                groups.append(dict(run_id=run_id, suite=suite, task=task, **episode_metrics(block, first[block.row])))
            for reason, block in natural[~natural.success].groupby("primary_failure_reason"):
                reasons.append(dict(reason=reason, **episode_metrics(block, first[block.row])))
        if rule.name in (PRIMARY.name, "freeze_back_half", "freeze_back_eighth", "recurrence_back_025", "collapse_back_half", "freeze_back_half_k4"):
            landmarks.extend(landmark_metrics(frame, scores, rule.threshold, rule.name))
        print(f"scored {rule.name}", flush=True)
    assert primary_scores is not None
    first = alarms[PRIMARY.name].to_numpy(int)
    lagged = np.where((first >= 0) & (first + 1 < frame.length.to_numpy()), first + 1, -1)
    alarms["primary_previous_query_only"] = lagged
    alarms["primary_observation_onset_q"] = np.where(first >= 0, first - PRIMARY.width - PRIMARY.confirmations + 2, -1)
    alarms["primary_actions_executed_before_alarm"] = np.where(first >= 0, first * frame.replan_steps.to_numpy(), -1)
    alarms.to_csv(RESULTS / "episode_alarms.csv", index=False)
    alarms[(alarms.regime == "libero_natural") & ~alarms.success].to_csv(RESULTS / "failure_keypoints.csv", index=False)
    pd.DataFrame(table).to_csv(RESULTS / "operating_points.csv", index=False)
    pd.DataFrame(groups).to_csv(RESULTS / "task_metrics.csv", index=False)
    pd.DataFrame(reasons).to_csv(RESULTS / "failure_reason_metrics.csv", index=False)
    pd.DataFrame(landmarks).to_csv(RESULTS / "fixed_query_metrics.csv", index=False)
    pd.DataFrame(clock_controls(frame, first)).to_csv(RESULTS / "clock_controls.csv", index=False)
    np.savez_compressed(RESULTS / "primary_scores.npz", scores=primary_scores.astype(np.float32), row=frame.row.to_numpy())
    natural = frame.regime.to_numpy() == "libero_natural"
    deadlines = []
    for q in range(features.shape[1]):
        by_q = np.where((first >= 0) & (first <= q), first, -1)
        deadlines.append(dict(query=q, **episode_metrics(frame[natural], by_q[natural])))
    pd.DataFrame(deadlines).to_csv(RESULTS / "early_deadlines.csv", index=False)
    summary = dict(primary_rule=asdict(PRIMARY), rules=[asdict(r) for r in rules],
                   no_training=True, detector_inputs=["current_and_past_hb_router_probs"],
                   primary=episode_metrics(frame[natural], first[natural]),
                   previous_query_only=episode_metrics(frame[natural], lagged[natural]),
                   coverage=frame.groupby(["regime", "run_id"]).agg(episodes=("row", "size"),
                       failures=("success", lambda x: int((~x).sum())), queries=("length", "sum")).reset_index().to_dict("records"),
                   note="Retrospective, fixed-rule evaluation. Wilson intervals ignore correlated seeds/init states.")
    write_json(RESULTS / "summary.json", summary)
    make_figure(alarms, primary_scores, first, pd.DataFrame(table), pd.DataFrame(deadlines))
    return summary


def make_figure(frame, scores, first, table, deadlines) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    chosen = table[(table.regime == "libero_natural") & (table.run_id == "all")]
    for signal, color in (("freeze", "#126b73"), ("recurrence", "#ac3653"), ("collapse", "#8063a0"), ("joint", "#50733c")):
        block = chosen[chosen.rule.str.startswith(signal)]
        axes[0, 0].scatter(block.fpr, block.recall, label=signal, color=color)
    primary = chosen[chosen.rule == PRIMARY.name].iloc[0]
    axes[0, 0].scatter([primary.fpr], [primary.recall], marker="*", s=170, color="black", label="fixed primary")
    axes[0, 0].set(xlabel="Successful episodes alarmed", ylabel="Failed episodes detected", title="Fixed rules; no parameter fitting")
    axes[0, 0].legend(fontsize=8)
    natural = frame.regime.to_numpy() == "libero_natural"
    half_first = first_alarm(scores, 0.5)
    half_deadlines = pd.DataFrame([
        episode_metrics(frame[natural], np.where((half_first[natural] >= 0) & (half_first[natural] <= q), half_first[natural], -1))
        for q in deadlines["query"]
    ])
    axes[0, 1].plot(deadlines["query"], half_deadlines.recall, label="Half threshold: recall", color="#126b73")
    axes[0, 1].plot(deadlines["query"], half_deadlines.fpr, label="Half threshold: false alarms", color="#ac3653")
    confirmed = frame["freeze_back_half_k4"].to_numpy()
    confirmed_recall = [episode_metrics(frame[natural], np.where((confirmed[natural] >= 0) & (confirmed[natural] <= q), confirmed[natural], -1))["recall"] for q in deadlines["query"]]
    axes[0, 1].plot(deadlines["query"], confirmed_recall, label="Half threshold, K=4: recall", color="#8063a0")
    axes[0, 1].plot(deadlines["query"], deadlines.recall, label="Quarter threshold: recall", color="#50733c", linestyle="--")
    axes[0, 1].set(xlabel="Decision query (zero based)", ylabel="Cumulative fraction", title="Timing across fixed thresholds")
    axes[0, 1].legend(fontsize=8)
    hit = (half_first >= 0) & natural
    candidates = [np.flatnonzero(hit & ~frame.success.to_numpy()), np.flatnonzero(hit & frame.success.to_numpy())]
    for ax, group, title in zip(axes[1], candidates, ("Failure", "False alarm on a success")):
        if not len(group):
            ax.set_title(title + ": none")
            continue
        ordered = group[np.argsort(half_first[group], kind="stable")]
        row = ordered[len(ordered)//2]
        n = int(frame.iloc[row].length)
        ax.plot(np.arange(n), scores[row, :n], color="#126b73")
        ax.axhline(0.5, color="#ac3653", linestyle="--")
        ax.axvline(half_first[row], color="black", linestyle=":")
        ax.set(xlabel="Query", ylabel="Persistent mobility ratio", title=f"{title}: row {row}, alarm q{half_first[row]}", ylim=(0, 1.5))
    fig.savefig(RESULTS / "overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        frame = pd.read_csv(RESULTS / "episode_alarms.csv")
        with np.load(RESULTS / "primary_scores.npz", allow_pickle=False) as archive:
            scores = archive["scores"]
        make_figure(frame, scores, frame[PRIMARY.name].to_numpy(),
                    pd.read_csv(RESULTS / "operating_points.csv"), pd.read_csv(RESULTS / "early_deadlines.csv"))
        return
    (RESULTS / "features").mkdir(parents=True, exist_ok=True)
    frame, jobs = source_inventory()
    frame.to_csv(RESULTS / "episode_index.csv", index=False)
    print(f"inventory: {len(frame)} episodes/subtasks, {len(jobs)} source runs", flush=True)
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_run, job): job for job in jobs}
        for future in as_completed(futures):
            audit = future.result()
            audits.append(audit)
            print(f"features {len(audits)}/{len(jobs)}: {audit['source_run']}", flush=True)
    audits.sort(key=lambda x: x["source_run"])
    write_json(RESULTS / "extraction_audit.json", audits)
    if args.extract_only:
        return
    features = load_features(frame, audits)
    summary = run_analysis(frame, features)
    print(json.dumps(summary["primary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
