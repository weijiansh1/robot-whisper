#!/usr/bin/env python3
"""Q4 properly: single heads and multi-head bundles on the same false-alarm axis.

The pre-registered comparison matches development false alarms, but development
false-alarm counts do not transfer to external (a bundle at 16 development FP
lands at 40 external FP). So two comparisons are reported:

  confirmatory   matched on DEVELOPMENT false alarms (pre-registered; in
                 multihead.json). Selection never sees external.
  descriptive    the full external TP-vs-FP frontier, produced by sweeping a
                 development-side knob (the pre-declared quantile grid, or the
                 greedy budget) and reporting where each point lands on
                 external. This is a sweep report, not a selection.

Every point on both frontiers is a causal, train-free, threshold-on-a-routing-
statistic rule. No physical label is used anywhere in this file.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
import evaluate_layerwise_alarm_development as dev

OUT = C.BUNDLE / "results"
MAX_HEADS = 4
BUDGET_GRID = tuple(range(4, 205, 4))


def build_sweeps() -> dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    config = pd.read_csv(C.EXTERNAL_DETECTORS)
    config = config[config["feasible"].fillna(False).astype(bool)]
    graphs = {n: C.load_graph(n)
              for n in ("development_main", "development_extra", "external_8b")}
    mob = {n: C.load_npz(C.MOBILITY_PATHS[n]) for n in graphs}
    dev_valid = graphs["development_main"]["valid"].astype(bool)
    ext_valid = graphs["external_8b"]["valid"].astype(bool)
    ext_task = C.task_of(graphs["external_8b"])
    ref_task = {n: C.task_of(graphs[n]) for n in ("development_main", "development_extra")}
    task_index = graphs["development_main"]["task_index"].astype(int)
    init_state = graphs["development_main"]["init_state_id"].astype(int)

    sweeps: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for quantity, block in config.groupby("quantity"):
        caches = {n: C.quantity_cache(graphs[n], mob[n], quantity) for n in graphs}
        reprs = {n: {k: v for k, (v, _) in dev.representations(caches[n]).items()}
                 for n in caches}
        for _, row in block.iterrows():
            representation, direction = str(row["representation"]), str(row["direction"])
            mode = str(row["mode"])
            key = f"{quantity}|{mode}"
            dev_or = C.oriented(reprs["development_main"][representation], direction)
            dev_p = dev.persistent_score(dev_or, C.CONFIRMATIONS)
            ext_p = dev.persistent_score(
                C.oriented(reprs["external_8b"][representation], direction), C.CONFIRMATIONS)
            pooled = np.concatenate(
                [dev.row_max(C.oriented(reprs[n][representation], direction))
                 for n in ("development_main", "development_extra")])
            crossfit = (dev.crossfit_thresholds(dev.row_max(dev_or), task_index, init_state)
                        if mode == "per_task" else None)
            per_task_lines = {}
            if mode == "per_task":
                for task in np.unique(ext_task):
                    peaks = [dev.row_max(C.oriented(
                        reprs[n][representation][ref_task[n] == task], direction))
                        for n in ("development_main", "development_extra")
                        if (ref_task[n] == task).any()]
                    per_task_lines[task] = np.concatenate(peaks)
            table = {}
            for position, q in enumerate(dev.QUANTILES):
                if mode == "global":
                    line = dev.quantile_higher(pooled, q)
                    if not np.isfinite(line):
                        continue
                    d = dev.first_query(np.isfinite(dev_p) & (dev_p > line) & dev_valid)
                    e = dev.first_query(np.isfinite(ext_p) & (ext_p > line) & ext_valid)
                else:
                    d = dev.first_query(np.isfinite(dev_p)
                                        & (dev_p > crossfit[:, position][:, None]) & dev_valid)
                    e = np.full(len(ext_p), -1, dtype=np.int16)
                    for task, peaks in per_task_lines.items():
                        take = np.flatnonzero(ext_task == task)
                        line = dev.quantile_higher(peaks, q)
                        e[take] = dev.first_query(np.isfinite(ext_p[take])
                                                  & (ext_p[take] > line) & ext_valid[take])
                table[float(q)] = (d.astype(np.int16), e.astype(np.int16))
            sweeps[key] = table
        print(f"  swept {quantity}", flush=True)
    return sweeps


def or_alarm(arrays):
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        fired = np.asarray(a) >= 0
        out[fired] = np.minimum(out[fired], np.asarray(a)[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


def pareto(points: pd.DataFrame) -> pd.DataFrame:
    """Keep points not dominated on (external fp low, external tp high)."""
    frame = points.sort_values(["ext_fp", "ext_tp"], ascending=[True, False])
    best, keep = -1, []
    for _, r in frame.iterrows():
        if r["ext_tp"] > best:
            keep.append(r)
            best = r["ext_tp"]
    return pd.DataFrame(keep)


def main() -> None:
    sweeps = build_sweeps()
    dev_frame = C.cohort_index("development_main")
    ext_frame = C.cohort_index("external_8b")
    dev_risk = dev_frame.risk.to_numpy(bool); ext_risk = ext_frame.risk.to_numpy(bool)
    dev_suite = dev_frame.suite.to_numpy(str); ext_suite = ext_frame.suite.to_numpy(str)
    dev_priors = C.survival_prior(dev_suite, dev_frame.length.to_numpy(int), dev_risk)
    ext_priors = C.survival_prior(ext_suite, ext_frame.length.to_numpy(int), ext_risk)
    v7 = C.load_npz(C.V7_ALARMS)

    rows = []

    def add(kind, name, quantile, d, e):
        d, e = np.asarray(d, int), np.asarray(e, int)
        ds = C.score_candidate(d, dev_risk, C.prior_of(d, dev_suite, dev_priors))
        es = C.score_candidate(e, ext_risk, C.prior_of(e, ext_suite, ext_priors))
        rows.append({"kind": kind, "name": name, "quantile": quantile,
                     "dev_tp": ds["tp"], "dev_fp": ds["fp"],
                     "ext_tp": es["tp"], "ext_fp": es["fp"],
                     "ext_precision": es["precision"], "ext_recall": es["risk_recall"],
                     "ext_lift": es["lift"], "ext_early_tp": es["low_prior_tp"],
                     "ext_early_fp": es["low_prior_fp"]})

    for key, table in sweeps.items():
        for q, (d, e) in sorted(table.items()):
            add("single", key, q, d, e)
    for head in ("freeze", "acceleration", "periodicity", "turbulence", "guard"):
        add("single", f"v7_{head}|task_agnostic", np.nan,
            v7[f"main_{head}"], v7[f"external_{head}"])

    # greedy OR bundles, selected on development, swept over the FP budget
    families = {
        "global": [k for k in sweeps if k.endswith("|global")],
        "per_task": [k for k in sweeps if k.endswith("|per_task")],
    }
    frozen_q = {}
    config = pd.read_csv(C.EXTERNAL_DETECTORS)
    config = config[config["feasible"].fillna(False).astype(bool)]
    for _, r in config.iterrows():
        frozen_q[f"{r['quantity']}|{r['mode']}"] = float(r["quantile"])

    for family, members in families.items():
        pool_dev = {k: sweeps[k][frozen_q[k]][0] for k in members}
        pool_ext = {k: sweeps[k][frozen_q[k]][1] for k in members}
        for budget in BUDGET_GRID:
            chosen, best_tp = [], 0
            while len(chosen) < MAX_HEADS:
                pick, gain = None, 0
                for k in members:
                    if k in chosen:
                        continue
                    merged = or_alarm([pool_dev[c] for c in chosen] + [pool_dev[k]])
                    fired = merged >= 0
                    tp, fp = int((fired & dev_risk).sum()), int((fired & ~dev_risk).sum())
                    if fp <= budget and tp - best_tp > gain:
                        pick, gain = k, tp - best_tp
                if pick is None:
                    break
                chosen.append(pick)
                best_tp = int(((or_alarm([pool_dev[c] for c in chosen]) >= 0) & dev_risk).sum())
            if chosen:
                add(f"bundle_{family}", "+".join(chosen), float(budget),
                    or_alarm([pool_dev[c] for c in chosen]),
                    or_alarm([pool_ext[c] for c in chosen]))

    # v7 guard OR global heads, swept over the budget
    members = [k for k in sweeps if k.endswith("|global")]
    for budget in BUDGET_GRID:
        chosen, best_tp = ["v7_guard"], 0
        pool_dev = {k: sweeps[k][frozen_q[k]][0] for k in members}
        pool_ext = {k: sweeps[k][frozen_q[k]][1] for k in members}
        pool_dev["v7_guard"] = v7["main_guard"]; pool_ext["v7_guard"] = v7["external_guard"]
        best_tp = int(((np.asarray(v7["main_guard"]) >= 0) & dev_risk).sum())
        if int(((np.asarray(v7["main_guard"]) >= 0) & ~dev_risk).sum()) > budget:
            continue
        while len(chosen) < MAX_HEADS:
            pick, gain = None, 0
            for k in members:
                if k in chosen:
                    continue
                merged = or_alarm([pool_dev[c] for c in chosen] + [pool_dev[k]])
                fired = merged >= 0
                tp, fp = int((fired & dev_risk).sum()), int((fired & ~dev_risk).sum())
                if fp <= budget and tp - best_tp > gain:
                    pick, gain = k, tp - best_tp
            if pick is None:
                break
            chosen.append(pick)
            best_tp = int(((or_alarm([pool_dev[c] for c in chosen]) >= 0) & dev_risk).sum())
        add("bundle_v7_plus_global", "+".join(chosen), float(budget),
            or_alarm([pool_dev[c] for c in chosen]), or_alarm([pool_ext[c] for c in chosen]))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "frontier_points.csv", index=False)
    np.savez_compressed(OUT / "sweep_index.npz",
                        keys=np.asarray(sorted(sweeps)),
                        quantiles=np.asarray(sorted(next(iter(sweeps.values())))))

    summary = {"schema": "himoe.failure_modes_0906.frontier.v1", "matched_external_fp": {}}
    single = frame[frame.kind == "single"]
    for family, kinds in (("global", ["bundle_global"]),
                          ("per_task", ["bundle_per_task"]),
                          ("task_agnostic", ["bundle_v7_plus_global"])):
        if family == "global":
            sing = single[single.name.str.endswith("|global")]
        elif family == "per_task":
            sing = single[single.name.str.endswith("|per_task")]
        else:
            sing = single[single.name.str.endswith("|global")
                          | single.name.str.endswith("|task_agnostic")]
        bund = frame[frame.kind.isin(kinds)]
        block = {}
        for cap in (17, 20, 40, 57, 80, 93, 100, 120, 155):
            s = sing[sing.ext_fp <= cap]
            b = bund[bund.ext_fp <= cap]
            block[cap] = {
                "best_single_ext_tp": int(s.ext_tp.max()) if len(s) else None,
                "best_single": s.loc[s.ext_tp.idxmax(), "name"] if len(s) else None,
                "best_single_ext_fp": int(s.loc[s.ext_tp.idxmax(), "ext_fp"]) if len(s) else None,
                "best_bundle_ext_tp": int(b.ext_tp.max()) if len(b) else None,
                "best_bundle": b.loc[b.ext_tp.idxmax(), "name"] if len(b) else None,
                "best_bundle_ext_fp": int(b.loc[b.ext_tp.idxmax(), "ext_fp"]) if len(b) else None,
            }
            block[cap]["bundle_minus_single_tp"] = (
                block[cap]["best_bundle_ext_tp"] - block[cap]["best_single_ext_tp"]
                if block[cap]["best_bundle_ext_tp"] is not None
                and block[cap]["best_single_ext_tp"] is not None else None)
        summary["matched_external_fp"][family] = block
    (OUT / "frontier.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary["matched_external_fp"], indent=2, default=str))


if __name__ == "__main__":
    main()
