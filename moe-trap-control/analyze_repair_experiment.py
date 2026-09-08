#!/usr/bin/env python3
"""Endpoints, stratification and internal readouts for the late-alarm repair experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from repair_control import ARMS, PROTOCOL, WINDOW_STEPS

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
STALL_WINDOW, STALL_DIAMETER_M = 6, 0.005


def arc_metrics(probs):
    """State-token sharpness and action-token arc geometry from one [8,10,11,32] fp16 route tensor."""
    p = np.asarray(probs, np.float64)[:, -1]                      # final denoise step: [8, 11, 32]
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    r = np.sqrt(p)
    out = {}
    for li, layer in enumerate(HB_LAYERS):
        state = p[li, 0]
        out["state_top1_L%d" % layer] = float(state.max())
        out["state_entropy_L%d" % layer] = float(-(state * np.log(np.maximum(state, 1e-12))).sum())
        g = r[li] @ r[li].T                                       # 11 x 11 Bhattacharyya kernel
        k = g[1:, 1:] - np.outer(g[0, 1:], g[0, 1:])              # Schur complement of the state token
        d = np.sqrt(np.maximum(np.diag(k)[:, None] + np.diag(k)[None, :] - 2 * k, 0.0))
        iu = np.triu_indices(10, 1)
        lag = np.abs(iu[0] - iu[1]).astype(np.float64)
        dist = d[iu]
        if dist.std() > 0:
            out["arc_bandedness_L%d" % layer] = float(np.corrcoef(np.argsort(np.argsort(dist)), np.argsort(np.argsort(lag)))[0, 1])
        else:
            out["arc_bandedness_L%d" % layer] = float("nan")
        centred = k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()
        out["arc_size_L%d" % layer] = float(np.trace(centred) / 10.0)
    out["state_top1_front"] = float(np.mean([out["state_top1_L%d" % l] for l in HB_LAYERS[:4]]))
    out["arc_bandedness_back"] = float(np.nanmean([out["arc_bandedness_L%d" % l] for l in HB_LAYERS[4:]]))
    out["arc_size_back"] = float(np.mean([out["arc_size_L%d" % l] for l in HB_LAYERS[4:]]))
    return out


def stalled_after(positions):
    """First query index at which the trailing 6-query end-effector window has diameter <= 5 mm."""
    for end in range(STALL_WINDOW, len(positions) + 1):
        window = np.asarray(positions[end - STALL_WINDOW:end], np.float64)
        if np.linalg.norm(window[:, None] - window[None, :], axis=-1).max() <= STALL_DIAMETER_M:
            return end - 1
    return None


def branch_rows(run_dir):
    rows = []
    for task_dir in sorted((run_dir / "tasks").glob("*/branches")):
        result = json.loads((task_dir / "result.json").read_text())
        if result["status"] != "completed":
            continue
        main_id, failed = result["main_id"], bool(result["failed"])
        replay = json.loads((task_dir.parent / "replay/result.json").read_text())
        physics_by_event = {e["event_id"]: json.loads((task_dir.parent / "replay/events" / e["event_id"] / "physics.json").read_text())
                            for e in replay["events"]}
        parent_rows = None
        for branch in result["branches"]:
            if branch["status"] != "completed":
                continue
            directory = task_dir / "branches" / branch["event_id"] / ("repeat%d" % branch["replicate"]) / branch["arm"]
            physics = physics_by_event[branch["event_id"]]
            row = dict(main_id=main_id, failed=failed, base_task=result["variant"]["task_name"].split("_view_")[0],
                benchmark=result["variant"]["benchmark"], category=result["variant"]["category"],
                event_id=branch["event_id"], timing=branch["timing"], start_query=branch["start_query"],
                physical_class=branch["physical_class"], target_name=branch["target_name"],
                eef_to_target_m=physics.get("eef_to_target_m"), aperture_at_fork=physics.get("aperture"),
                arm=branch["arm"], replicate=branch["replicate"],
                success_window=bool(branch["success"]), success_original=bool(branch["success_within_original"]),
                success_step=branch.get("success_step"), original_remaining=branch["original_remaining_steps"],
                repair_kind=branch.get("repair_kind"), repair_reason=branch.get("repair_reason"),
                repair_steps=branch.get("repair_steps", 0), repair_chunks=branch.get("repair_chunks", 0),
                retract_reason=branch.get("retract_reason"), lifted=branch.get("lifted"),
                handback_step=branch.get("handback_step"), queries=branch["queries"],
                model_queries=branch["deployment_model_queries"])
            suffix = directory / "suffix"
            positions, first_probs, knn_at_8, freeze_at_1 = [], None, None, None
            if (suffix / "manifest.json").exists():
                for index, rec in enumerate(records(suffix)):
                    positions.append(np.asarray(rec["proprio"][:3], np.float64))
                    if index == 0:
                        first_probs = np.asarray(rec[PROBS_KEY])
                        freeze_at_1 = float(rec["alarm_scores"][0])
                    if index == 7:
                        knn_at_8 = float(rec["knn_score"])
            row.update(knn_after_8=knn_at_8, freeze_after_1=freeze_at_1)
            stall = stalled_after(positions) if positions else None
            row["restalled_query"] = stall
            row["restalled"] = bool(stall is not None and not row["success_window"])
            if first_probs is not None:
                row.update({"post_" + k: v for k, v in arc_metrics(first_probs).items()
                            if k in ("state_top1_front", "arc_bandedness_back", "arc_size_back")})
            if parent_rows is None:
                parent_rows = {int(r["query"]): r for r in records(Path(result["parent_directory"]) / "main")}
            pre = parent_rows.get(branch["start_query"] - 1)
            if pre is not None:
                row.update({"pre_" + k: v for k, v in arc_metrics(np.asarray(pre[PROBS_KEY])).items()
                            if k in ("state_top1_front", "arc_bandedness_back", "arc_size_back")})
            rows.append(row)
    return rows


def counts(rows, key):
    table = defaultdict(lambda: dict(n=0, window=0, original=0, restalled=0, repair_reached=0))
    for row in rows:
        cell = table[key(row)]
        cell["n"] += 1
        cell["window"] += row["success_window"]
        cell["original"] += row["success_original"]
        cell["restalled"] += row["restalled"]
        cell["repair_reached"] += row.get("retract_reason") == "reached"
    return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): v for k, v in table.items()}


def paired_vs_baseline(rows):
    baseline = {(r["main_id"], r["event_id"], r["replicate"]): r for r in rows if r["arm"] == "new_noise"}
    out = defaultdict(lambda: dict(pairs=0, wins=0, losses=0))
    for r in rows:
        if r["arm"] == "new_noise":
            continue
        base = baseline.get((r["main_id"], r["event_id"], r["replicate"]))
        if base is None:
            continue
        cell = out[(r["timing"], r["arm"])]
        cell["pairs"] += 1
        cell["wins"] += int(r["success_window"] and not base["success_window"])
        cell["losses"] += int(base["success_window"] and not r["success_window"])
    return {"|".join(k): v for k, v in out.items()}


def cluster_bootstrap(rows, arm, timing, baseline="new_noise", n_boot=2000, seed=20260908):
    """Paired difference in window success (arm minus baseline) among failed parents, resampled by main_id."""
    rng = np.random.default_rng(seed)
    per_main = defaultdict(list)
    base = {(r["main_id"], r["event_id"], r["replicate"]): r["success_window"] for r in rows if r["arm"] == baseline}
    for r in rows:
        if r["arm"] != arm or r["timing"] != timing or not r["failed"]:
            continue
        b = base.get((r["main_id"], r["event_id"], r["replicate"]))
        if b is not None:
            per_main[r["main_id"]].append(int(r["success_window"]) - int(b))
    mains = list(per_main)
    if not mains:
        return None
    means = np.array([np.mean(per_main[m]) for m in mains])
    point = float(means.mean())
    draws = [means[rng.integers(0, len(mains), len(mains))].mean() for _ in range(n_boot)]
    return dict(mains=len(mains), point=point, ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def run(args):
    rows = branch_rows(args.run)
    failed = [r for r in rows if r["failed"]]
    success = [r for r in rows if not r["failed"]]
    summary = dict(protocol=PROTOCOL, run=str(args.run), branches=len(rows),
        parents=len({r["main_id"] for r in rows}), failed_parents=len({r["main_id"] for r in failed}),
        success_parents=len({r["main_id"] for r in success}),
        by_timing_arm=counts(failed, lambda r: (r["timing"], r["arm"])),
        by_timing_class_arm=counts(failed, lambda r: (r["timing"], r["physical_class"], r["arm"])),
        by_class=counts(failed, lambda r: (r["physical_class"],)),
        harm_by_timing_arm=counts(success, lambda r: (r["timing"], r["arm"])),
        paired_vs_new_noise=paired_vs_baseline(failed),
        bootstrap={"%s|%s" % (t, a): cluster_bootstrap(failed, a, t) for t in ("early", "mid", "late") for a in ARMS if a != "new_noise"},
        readout_correlation={})
    for name in ("post_state_top1_front", "post_arc_bandedness_back", "post_arc_size_back", "knn_after_8", "freeze_after_1"):
        values = [(r[name], r["success_window"]) for r in failed if r.get(name) is not None and np.isfinite(r[name])]
        if len(values) >= 8 and len({v[1] for v in values}) == 2:
            x = np.array([v[0] for v in values]); y = np.array([v[1] for v in values], bool)
            pos, neg = x[y], x[~y]
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
            summary["readout_correlation"][name] = dict(n=len(values), positives=int(y.sum()), auc_success_high=auc)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "summary.json", summary)
    fields = sorted({k for r in rows for k in r})
    with (args.out / "branches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    print(json.dumps(dict(branches=len(rows), failed_parents=summary["failed_parents"],
        by_timing_arm=summary["by_timing_arm"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
