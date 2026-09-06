#!/usr/bin/env python3
"""Q3 + Q4: build a multi-head detector, name its heads from physical modes, and
test it against a single head at matched false alarms.

Selection is on development outcomes only (risk / not risk). Physical labels are
used only afterwards, to name heads and to describe what the bundle catches; they
never touch an alarm. External is opened once, at the end.

Greedy OR, exactly as pre-registered:
    start empty; repeatedly add the head with the largest development TP gain
    that keeps development FP within budget; stop at 4 heads or no gain.

Families are never mixed, because only `global` proves the signal is MoE-side:
    global               12 pooled-quantile detectors, no task identity
    per_task             12 same-task-quantile detectors
    task_agnostic        the 12 global detectors + the 5 v7 heads
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
import evaluate_layerwise_alarm_development as dev

OUT = C.BUNDLE / "results"
MAX_HEADS = 4

DEV_NONRISK, EXT_NONRISK = 14313, 15036
BUDGETS = {
    "global_single_17fp": ("global", 17),
    "global_combination_13fp": ("global", 13),
    "per_task_single_93fp": ("per_task", 93),
    "per_task_combination_57fp": ("per_task", 57),
    "task_agnostic_v7guard_80fp": ("task_agnostic", 80),
}


def or_alarm(arrays: list[np.ndarray]) -> np.ndarray:
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        fired = a >= 0
        out[fired] = np.minimum(out[fired], a[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


def family_members(keys: list[str], family: str) -> list[str]:
    if family == "global":
        return [k for k in keys if k.endswith("|global")]
    if family == "per_task":
        return [k for k in keys if k.endswith("|per_task")]
    return [k for k in keys if k.endswith("|global") or k.endswith("|task_agnostic")]


def greedy(cands: dict[str, np.ndarray], risk: np.ndarray, budget: int) -> list[str]:
    chosen: list[str] = []
    best_tp = -1
    while len(chosen) < MAX_HEADS:
        best_key, best_gain, best_fp = None, 0, None
        for key, arr in cands.items():
            if key in chosen:
                continue
            merged = or_alarm([cands[c] for c in chosen] + [arr]) if chosen else arr
            fired = merged >= 0
            tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
            if fp > budget:
                continue
            if tp > best_tp and tp - max(best_tp, 0) > best_gain:
                best_key, best_gain, best_fp = key, tp - max(best_tp, 0), fp
        if best_key is None:
            break
        chosen.append(best_key)
        merged = or_alarm([cands[c] for c in chosen])
        best_tp = int(((merged >= 0) & risk).sum())
        print(f"    + {best_key}: dev tp={best_tp} fp={best_fp}", flush=True)
    return chosen


def evaluate(first, risk, suite, priors):
    return C.score_candidate(np.asarray(first, int), risk,
                             C.prior_of(np.asarray(first, int), suite, priors))


def quantile_sweep(quantity: str, representation: str, direction: str, mode: str):
    """Rebuild one detector at every pre-declared quantile (train-free grid)."""
    graphs = {n: C.load_graph(n)
              for n in ("development_main", "development_extra", "external_8b")}
    mob = {n: C.load_npz(C.MOBILITY_PATHS[n]) for n in graphs}
    caches = {n: C.quantity_cache(graphs[n], mob[n], quantity) for n in graphs}
    reprs = {n: {k: v for k, (v, _) in dev.representations(caches[n]).items()}
             for n in caches}
    dev_or = C.oriented(reprs["development_main"][representation], direction)
    dev_p = dev.persistent_score(dev_or, C.CONFIRMATIONS)
    ext_p = dev.persistent_score(
        C.oriented(reprs["external_8b"][representation], direction), C.CONFIRMATIONS)
    dev_valid = graphs["development_main"]["valid"].astype(bool)
    ext_valid = graphs["external_8b"]["valid"].astype(bool)
    pooled = np.concatenate([dev.row_max(C.oriented(reprs[n][representation], direction))
                             for n in ("development_main", "development_extra")])
    ext_task = C.task_of(graphs["external_8b"])
    ref_task = {n: C.task_of(graphs[n]) for n in ("development_main", "development_extra")}
    crossfit = dev.crossfit_thresholds(dev.row_max(dev_or),
                                       graphs["development_main"]["task_index"].astype(int),
                                       graphs["development_main"]["init_state_id"].astype(int))
    out = {}
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
            for task in np.unique(ext_task):
                take = np.flatnonzero(ext_task == task)
                peaks = [dev.row_max(C.oriented(reprs[n][representation][ref_task[n] == task],
                                                direction))
                         for n in ("development_main", "development_extra")
                         if (ref_task[n] == task).any()]
                line = dev.quantile_higher(np.concatenate(peaks), q)
                e[take] = dev.first_query(np.isfinite(ext_p[take]) & (ext_p[take] > line)
                                          & ext_valid[take])
        out[q] = (d.astype(np.int16), e.astype(np.int16))
    return out


def name_heads(heads: list[str], recall: pd.DataFrame) -> dict:
    """Pre-registered naming rule. Evidence is the physical label, never the head."""
    dev_t = recall[(recall.cohort == "development_main") & (recall.population == "all_risks")]
    ext_t = recall[(recall.cohort == "external_8b") & (recall.population == "all_risks")]
    names = {}
    for head in heads:
        d = dev_t[dev_t["detector"] == head].replace([np.inf, -np.inf], np.nan)
        e = ext_t[ext_t["detector"] == head].replace([np.inf, -np.inf], np.nan)
        d = d[d["n_mode"] >= 30]
        if d.empty or d.sel_suite.isna().all():
            names[head] = {"name": "unselective", "reason": "no usable cell"}
            continue
        top = d.loc[d.sel_suite.idxmax()]
        row = e[e["mode"] == top["mode"]]
        strict = bool(top.sel_suite > 0 and top.p_suite < 0.05 and top.p_task < 0.05)
        relaxed = bool(top.sel_suite > 0 and top.p_suite < 0.05)
        confirmed = bool(
            not row.empty
            and float(row.sel_suite.iloc[0]) > 0
            and top["mode"] == e.loc[e[e["n_mode"] >= 30].sel_suite.idxmax(), "mode"]
        ) if not e[e["n_mode"] >= 30].empty else False
        names[head] = {
            "name": C.MODE_SHORT.get(top["mode"], top["mode"]) if strict else "unselective",
            "relaxed_name": C.MODE_SHORT.get(top["mode"], top["mode"]) if relaxed else "unselective",
            "development_argmax_mode": top["mode"],
            "development_sel_suite": float(top.sel_suite),
            "development_p_suite": float(top.p_suite),
            "development_p_task": float(top.p_task),
            "external_sel_suite": float(row.sel_suite.iloc[0]) if not row.empty else None,
            "external_p_suite": float(row.p_suite.iloc[0]) if not row.empty else None,
            "external_confirms_argmax": confirmed,
            "passes_strict_rule": strict,
        }
    return names


def main() -> None:
    recall = pd.read_csv(OUT / "mode_recall.csv")
    dev_alarms = C.load_npz(OUT / "first_alarms_development.npz"); dev_alarms.pop("schema")
    ext_alarms = C.load_npz(OUT / "first_alarms_external.npz"); ext_alarms.pop("schema")

    dev_frame = C.cohort_index("development_main")
    ext_frame = C.cohort_index("external_8b")
    dev_risk = dev_frame.risk.to_numpy(bool); ext_risk = ext_frame.risk.to_numpy(bool)
    dev_suite = dev_frame.suite.to_numpy(str); ext_suite = ext_frame.suite.to_numpy(str)
    dev_priors = C.survival_prior(dev_suite, dev_frame.length.to_numpy(int), dev_risk)
    ext_priors = C.survival_prior(ext_suite, ext_frame.length.to_numpy(int), ext_risk)
    ext_mode = ext_frame.primary_failure_reason.fillna("").to_numpy(str)

    keys = sorted(dev_alarms)
    results = {}
    per_mode_rows = []

    for label, (family, ext_fp) in BUDGETS.items():
        dev_budget = int(round(ext_fp * DEV_NONRISK / EXT_NONRISK))
        members = family_members(keys, family)
        cands = {k: np.asarray(dev_alarms[k], np.int16) for k in members}
        print(f"  {label}: family={family} dev FP budget={dev_budget} "
              f"({len(members)} candidate heads)", flush=True)
        chosen = greedy(cands, dev_risk, dev_budget)
        if not chosen:
            results[label] = {"heads": [], "note": "no head fits the budget"}
            continue
        dev_first = or_alarm([dev_alarms[k] for k in chosen])
        ext_first = or_alarm([ext_alarms[k] for k in chosen])
        results[label] = {
            "family": family,
            "external_fp_reference": ext_fp,
            "development_fp_budget": dev_budget,
            "heads": chosen,
            "development": evaluate(dev_first, dev_risk, dev_suite, dev_priors),
            "external": evaluate(ext_first, ext_risk, ext_suite, ext_priors),
            "head_names": name_heads(chosen, recall),
            "per_head_external": {
                k: evaluate(ext_alarms[k], ext_risk, ext_suite, ext_priors) for k in chosen
            },
        }
        for m in pd.Series(ext_mode[ext_risk]).value_counts().index:
            take = (ext_mode == m) & ext_risk
            row = {"bundle": label, "mode": m, "mode_short": C.MODE_SHORT.get(m, m),
                   "n_mode": int(take.sum()),
                   "bundle_caught": int(((ext_first >= 0) & take).sum())}
            for k in chosen:
                row[f"head::{k}"] = int(((np.asarray(ext_alarms[k]) >= 0) & take).sum())
            per_mode_rows.append(row)
        np.savez_compressed(OUT / f"bundle_alarms_{label}.npz",
                            development=dev_first, external=ext_first,
                            heads=np.asarray(chosen))

    # --- Q4: matched-false-alarm comparison against the best single head -----
    singles = {
        "global_single_17fp": ("mobility", "L12", "low", "global", "mobility|global"),
        "global_combination_13fp": ("mobility", "L12", "low", "global", "mobility|global"),
        "per_task_single_93fp": ("expert_load_effective_rank", "L3", "low", "per_task",
                                 "expert_load_effective_rank|per_task"),
        "per_task_combination_57fp": ("expert_load_effective_rank", "L3", "low", "per_task",
                                      "expert_load_effective_rank|per_task"),
        "task_agnostic_v7guard_80fp": (None, None, None, None, "v7_guard|task_agnostic"),
    }
    matched = {}
    for label, (quantity, representation, direction, mode, frozen_key) in singles.items():
        if label not in results or not results[label].get("heads"):
            continue
        bundle_dev_fp = int(results[label]["development"]["fp"])
        bundle = results[label]["external"]
        entry = {"frozen_single_head": frozen_key,
                 "frozen_single_head_external": evaluate(
                     ext_alarms[frozen_key], ext_risk, ext_suite, ext_priors),
                 "bundle_external": bundle,
                 "bundle_development_fp": bundle_dev_fp}
        if quantity is not None:
            sweep = quantile_sweep(quantity, representation, direction, mode)
            best = None
            for q, (d, e) in sorted(sweep.items()):
                fired = d >= 0
                fp = int((fired & ~dev_risk).sum()); tp = int((fired & dev_risk).sum())
                if fp <= bundle_dev_fp and (best is None or tp > best[1]):
                    best = (q, tp, fp, e)
            if best is not None:
                q, tp, fp, e = best
                entry["matched_single_head"] = {
                    "quantile": q, "development_tp": tp, "development_fp": fp,
                    "external": evaluate(e, ext_risk, ext_suite, ext_priors),
                    "external_per_mode": {
                        C.MODE_SHORT.get(m, m): int(((e >= 0) & (ext_mode == m) & ext_risk).sum())
                        for m in pd.Series(ext_mode[ext_risk]).value_counts().index},
                }
        matched[label] = entry

    pd.DataFrame(per_mode_rows).to_csv(OUT / "bundle_per_mode_external.csv", index=False)
    (OUT / "multihead.json").write_text(
        json.dumps({"schema": "himoe.failure_modes_0906.multihead.v1",
                    "max_heads": MAX_HEADS,
                    "selection": "greedy OR, development outcomes only",
                    "bundles": results,
                    "matched_false_alarm_comparison": matched},
                   indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for label, block in results.items():
        if block.get("heads"):
            e = block["external"]
            print(f"{label}: heads={block['heads']} -> external tp={e['tp']} fp={e['fp']} "
                  f"prec={e['precision']:.3f} recall={e['risk_recall']:.3f} "
                  f"lift={e['lift']:.3f} earlyTP={e['low_prior_tp']} earlyFP={e['low_prior_fp']}")


if __name__ == "__main__":
    main()
