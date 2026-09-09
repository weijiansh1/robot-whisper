#!/usr/bin/env python3
"""What happens to internal signals after a kNN-20 alarm: successes that survive the alarm versus failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from adaptive_control import RouteRisk
from collection_storage import atomic_json

SCALARS = ["knn", "freeze", "acceleration", "periodicity", "v8_inversion", "v8_curvature", "state_top1_front",
           "state_top1_back", "state_entropy_front", "arc_bandedness_back", "arc_size_back", "state_drift_from_alarm",
           "action_drift_from_alarm", "state_recurrence_lag3", "action_recurrence_lag3"]


def base_task(name):
    return name.split("_view_")[0].split("_light_")[0].split("_table_")[0].split("_add_")[0].split("_level")[0].split("_initstate")[0]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def scalars(f, t, alarm):
    """Scalar features at query index t for one episode (alarm = first-alarm query)."""
    sp, ap = f["state_probs"], f["action_probs"]
    hell = lambda a, b: float(np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2.0) / np.sqrt(a.shape[0]))
    out = dict(knn=f["knn"][t], freeze=f["v7"][t, 0], acceleration=f["v7"][t, 1], periodicity=f["v7"][t, 2],
               v8_inversion=f["v8"][t, 0], v8_curvature=f["v8"][t, 1], state_top1_front=f["state_top1"][t, :4].mean(),
               state_top1_back=f["state_top1"][t, 4:].mean(), state_entropy_front=f["state_entropy"][t, :4].mean(),
               arc_bandedness_back=f["arc"][t, 1], arc_size_back=f["arc"][t, 2],
               state_drift_from_alarm=hell(sp[t], sp[alarm]), action_drift_from_alarm=hell(ap[t], ap[alarm]),
               state_recurrence_lag3=hell(sp[t], sp[t - 3]) if t >= 3 else np.nan,
               action_recurrence_lag3=hell(ap[t], ap[t - 3]) if t >= 3 else np.nan)
    return {k: float(v) for k, v in out.items()}


def logistic_cv(X, y, groups, l2=1.0, iters=500, lr=0.1):
    """Leave-one-group-out logistic regression in numpy; returns out-of-fold probabilities."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    oof = np.full(len(y), np.nan)
    for g in sorted(set(groups)):
        train = np.array([gg != g for gg in groups]); test = ~train
        w, b = np.zeros(Z.shape[1]), 0.0
        Zt, yt = Z[train], y[train]
        pw = 0.5 / max(yt.mean(), 1e-6); nw = 0.5 / max(1 - yt.mean(), 1e-6)
        sw = np.where(yt == 1, pw, nw)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Zt @ w + b)))
            grad_w = (Zt * ((p - yt) * sw)[:, None]).mean(0) + l2 * w / len(yt)
            grad_b = ((p - yt) * sw).mean()
            w -= lr * grad_w; b -= lr * grad_b
        oof[test] = 1 / (1 + np.exp(-(Z[test] @ w + b)))
    return oof


def run(args):
    meta = json.loads((args.features / "meta.json").read_text())
    risk = RouteRisk()
    threshold = risk.threshold
    episodes = {}
    for main_id, m in meta.items():
        if m["analysis_role"] != "perturbation" or m["knn20"] < 0:
            continue
        with np.load(args.features / ("%s.npz" % main_id)) as z:
            episodes[main_id] = {k: z[k] for k in z.files}
    succ = [m for m in episodes if not meta[m]["failure"]]
    fail = [m for m in episodes if meta[m]["failure"]]
    summary = dict(alarmed_successes=len(succ), alarmed_failures=len(fail), threshold=float(threshold))
    print("alarmed successes", len(succ), "alarmed failures", len(fail))
    # 1) aligned medians and per-offset AUC
    offsets = list(range(-4, 25))
    table = {}
    for name in SCALARS:
        row = {}
        for dt in offsets:
            vals = {"success": [], "failure": []}
            for group, ids in (("success", succ), ("failure", fail)):
                for m in ids:
                    f, a = episodes[m], meta[m]["knn20"]
                    t = a + dt
                    if 0 <= t < len(f["knn"]):
                        vals[group].append(scalars(f, t, a)[name])
            row[dt] = dict(success_median=float(np.nanmedian(vals["success"])) if vals["success"] else None,
                           failure_median=float(np.nanmedian(vals["failure"])) if vals["failure"] else None,
                           auc_failure_high=auc(vals["failure"], vals["success"]), n_success=len(vals["success"]), n_failure=len(vals["failure"]))
        table[name] = row
    summary["aligned"] = table
    print("\n== AUC (failure high) by offset from the first kNN alarm; n_success at +1/+5/+10:",
          table["knn"][1]["n_success"], table["knn"][5]["n_success"], table["knn"][10]["n_success"])
    print("%-26s" % "feature" + "".join("%7s" % ("t%+d" % d) for d in (0, 1, 2, 3, 5, 8, 12, 16, 20)))
    for name in SCALARS:
        print("%-26s" % name + "".join("%7.2f" % table[name][d]["auc_failure_high"] for d in (0, 1, 2, 3, 5, 8, 12, 16, 20)))
    print("\n== medians (success | failure) at t+1, t+5, t+10")
    for name in SCALARS:
        print("%-26s" % name + "".join("  %6.3f|%6.3f" % (table[name][d]["success_median"] or np.nan, table[name][d]["failure_median"] or np.nan) for d in (1, 5, 10)))
    # 2) alarm clearing: first offset after alarm with knn below threshold, and fraction still above at +5/+10
    clear = {}
    for group, ids in (("success", succ), ("failure", fail)):
        times, above5, above10, n5, n10 = [], 0, 0, 0, 0
        for m in ids:
            f, a = episodes[m], meta[m]["knn20"]
            after = f["knn"][a + 1:]
            below = np.where(after < threshold)[0]
            times.append(int(below[0]) + 1 if len(below) else None)
            if a + 5 < len(f["knn"]):
                n5 += 1; above5 += f["knn"][a + 5] >= threshold
            if a + 10 < len(f["knn"]):
                n10 += 1; above10 += f["knn"][a + 10] >= threshold
        cleared = [t for t in times if t is not None]
        clear[group] = dict(n=len(ids), cleared=len(cleared), median_clear_queries=float(np.median(cleared)) if cleared else None,
                            still_above_at_5="%d/%d" % (above5, n5), still_above_at_10="%d/%d" % (above10, n10))
    summary["clearing"] = clear
    print("\n== alarm clearing:", json.dumps(clear))
    # 3) predictor: features from t+1..t+K -> success, leave-one-base-task-out
    for K in (1, 3, 5):
        X, y, groups, ids = [], [], [], []
        for m in succ + fail:
            f, a = episodes[m], meta[m]["knn20"]
            if a + K >= len(f["knn"]):
                continue
            feats = []
            for dt in range(1, K + 1):
                s = scalars(f, a + dt, a)
                feats += [s[k] for k in SCALARS]
            if not np.isfinite(feats).all():
                continue
            X.append(feats); y.append(0 if meta[m]["failure"] else 1); groups.append(base_task(meta[m]["task_name"])); ids.append(m)
        oof = logistic_cv(X, y, groups)
        y = np.asarray(y)
        a = auc(oof[y == 1], oof[y == 0])
        # operating point: spare successes: threshold at which 80% of successes are above
        thr = np.percentile(oof[y == 1], 20)
        spared = float((oof[y == 1] >= thr).mean()); kept = float((oof[y == 0] < thr).mean())
        summary["predictor_K%d" % K] = dict(n=len(y), successes=int(y.sum()), auc_success_high=a, spare80_success_fraction=spared,
                                             failures_still_flagged=kept, base_tasks=len(set(groups)))
        print("predictor K=%d: n=%d (successes %d) leave-task-out AUC(success high)=%.3f; at 80%% successes spared, failures still flagged %.1f%%" % (K, len(y), y.sum(), a, 100 * kept))
    # 4) contrastive routing direction after the alarm: success minus failure mean centred-log routing at t+1..t+5
    def centred_log(p):
        v = np.log(np.maximum(p, 1e-12)); return v - v.mean(-1, keepdims=True)
    acc = {"success": [], "failure": []}
    for group, ids in (("success", succ), ("failure", fail)):
        for m in ids:
            f, a = episodes[m], meta[m]["knn20"]
            for dt in range(1, 6):
                if a + dt < len(f["knn"]):
                    acc[group].append(np.stack([centred_log(f["state_probs"][a + dt]), centred_log(f["action_probs"][a + dt])]))
    S, F = np.asarray(acc["success"]), np.asarray(acc["failure"])
    diff = S.mean(0) - F.mean(0)                    # [2 (state, action), 8 layers, 32 experts]
    pooled = np.sqrt(0.5 * (S.var(0) + F.var(0))) + 1e-9
    effect = diff / pooled
    summary["contrast"] = dict(state_rms_by_layer=np.sqrt((effect[0] ** 2).mean(-1)).round(3).tolist(),
                               action_rms_by_layer=np.sqrt((effect[1] ** 2).mean(-1)).round(3).tolist(),
                               max_abs_effect=float(np.abs(effect).max()))
    np.savez_compressed(args.out / "contrast_direction.npz", diff=diff, effect=effect, success_mean=S.mean(0), failure_mean=F.mean(0))
    print("\n== contrastive routing (success - failure, t+1..t+5), standardised effect RMS per layer")
    print("   state token :", summary["contrast"]["state_rms_by_layer"])
    print("   action tokens:", summary["contrast"]["action_rms_by_layer"], " max |effect|", round(summary["contrast"]["max_abs_effect"], 3))
    atomic_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("design/post_alarm_routing_20260909"))
    parser.add_argument("--out", type=Path, default=Path("design/post_alarm_routing_analysis_20260909"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args)
