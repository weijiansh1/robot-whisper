#!/usr/bin/env python3
"""Select on development, score external once, and test the mode x arm interaction.

Protocol, fixed before any external number was read:

  * an operating point is (channel, layer, direction, statistic, threshold);
  * a *selection* maximises true positives of one target group at lead >= 4
    subject to a false-alarm budget, on development episodes only;
  * the selection is then applied to external unchanged, two ways - at
    development's numeric threshold, and at the same grid index re-derived from
    external's own unlabelled routing distribution;
  * leave-one-task-out repeats the selection with one task removed from the
    development side and scores only that task on the external side, so a LOTO
    number and the pooled number it is compared with are the same counts summed
    over different task sets;
  * every reported operating point carries the true positives the cap-free
    fixed-chunk baseline ("still running at chunk q0") would get at the same
    false-alarm count, per mode.  That is the floor, not the standard of proof.

Five channel pools are reported, all fixed in advance; see `arms.pool_masks`.
The two that matter for reading the result are `headline` (routing channels
that move inside an episode) and `null_elapsed` (the chunk counter itself, on
which a persistence statistic *is* the fixed-chunk baseline).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import arms as A
import modes_common as M
import capfree_protocol as CF

BUDGETS = (19, 38, 57, 76)
HEADLINE_BUDGET = 38
LEAD_INDEX = {b: i for i, b in enumerate(M.LEADS)}
BOOT_DRAWS = 4000
EXT_TABLES = ("frozenvalue", "ratematched")
PRIMARY_EXT = "ratematched"
# The protocol baseline sweep, and an extended diagnostic sweep.  The protocol
# stops at 39; a persistence statistic can alarm at chunk 48, and comparing it
# to a baseline that is not allowed to wait that long would flatter it.
Q0_PROTOCOL = tuple(CF.Q0_GRID)
Q0_EXTENDED = tuple(range(2, 52))


def load_table(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    ops = pd.DataFrame(z["ops"])
    for col in ("quantity", "layer", "stat", "family", "arm"):
        ops[col] = ops[col].astype(str)
    for col in ("threshold", "median_chunk", "within_episode_constant"):
        ops[col] = ops[col].astype(float)
    ops["sign"] = ops["sign"].astype(int)
    ops["grid_k"] = ops["grid_k"].astype(int)
    ops["is_control"] = ops["quantity"].str.startswith("ctrl_")
    return {"ops": ops, "counts": z["counts"],
            "tasks": np.asarray(z["tasks"], dtype=object).astype(str),
            "n_risk": int(z["n_risk"]), "n_episode": int(z["n_episode"])}


_FLAT: dict[int, np.ndarray] = {}


def _flat(counts: np.ndarray, lead: int) -> np.ndarray:
    """(n_ops * 4 groups, n_task) float32 view, cached, for fast task subsetting."""
    key = (id(counts), lead)
    if key not in _FLAT:
        li = LEAD_INDEX[lead]
        blk = np.ascontiguousarray(
            counts[..., li].transpose(0, 2, 1).reshape(-1, counts.shape[1]),
            dtype=np.float32)
        _FLAT[key] = blk
    return _FLAT[key]


def totals(counts: np.ndarray, task_take: np.ndarray, lead: int) -> dict:
    """Alarm counts summed over a subset of tasks, for every operating point."""
    blk = _flat(counts, lead)
    sub = (blk @ task_take.astype(np.float32)).reshape(-1, A.N_GROUP)
    sub = np.rint(sub).astype(np.int64)
    return {"fp": sub[:, A.GROUP_NONRISK],
            "tp_drop": sub[:, A.GROUP_DROP],
            "tp_grasp": sub[:, A.GROUP_GRASP],
            "tp_other": sub[:, A.GROUP_OTHER],
            "tp_all": sub[:, 1:].sum(axis=1)}


def pick(table: dict, rows: np.ndarray, task_take: np.ndarray, target: str,
         budget: int, lead: int = M.HEADLINE_LEAD) -> int | None:
    """Operating point maximising the target at lead >= `lead`, at most `budget` FP.

    Ties break on fewer false alarms, then earlier median alarm chunk, then
    table order - deterministic and fixed in advance.
    """
    tot = totals(table["counts"], task_take, lead)
    key = "tp_all" if target == "all" else f"tp_{target}"
    ok = rows & (tot["fp"] <= budget)
    if not ok.any():
        return None
    best = np.where(ok, tot[key], -1).max()
    cand = np.flatnonzero(ok & (tot[key] == best))
    fp = tot["fp"][cand]
    cand = cand[fp == fp.min()]
    med = table["ops"]["median_chunk"].to_numpy()[cand]
    return int(cand[np.argsort(np.where(np.isfinite(med), med, 1e9),
                               kind="stable")[0]])


# --------------------------------------------------------------------------- #
# cap-free fixed-chunk baseline, per mode


def baseline_frame(frame: dict, q0_grid) -> pd.DataFrame:
    risk, length, mode = frame["risk"], frame["length"], frame["mode"]
    drop = risk & (mode == M.MODE_DROP)
    grasp = risk & (mode == M.MODE_GRASP)
    rows = []
    for q0 in q0_grid:
        first = np.where(length > q0, q0, -1)
        fired = first >= 0
        lead = np.where(fired, length - first, -1)
        row = {"q0": q0}
        for b in M.LEADS:
            timely = fired & (lead >= b)
            row[f"fp_lead{b}"] = int((timely & ~risk).sum())
            row[f"tp_all_lead{b}"] = int((timely & risk).sum())
            row[f"tp_drop_lead{b}"] = int((timely & drop).sum())
            row[f"tp_grasp_lead{b}"] = int((timely & grasp).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_at(base: pd.DataFrame, fp_budget: float, key: str,
                lead: int = M.HEADLINE_LEAD) -> int:
    take = base[f"fp_lead{lead}"] <= fp_budget
    return int(base.loc[take, f"{key}_lead{lead}"].max()) if take.any() else 0


# --------------------------------------------------------------------------- #
# alarm vectors, needed only for the union arms


def alarm_vector(frame: dict, calib: dict, row: pd.Series) -> np.ndarray:
    x = M.channel(frame, row["quantity"], row["layer"])
    c = calib[(row["quantity"], row["layer"])]
    u, ok = A.standardised(x, frame["valid"], c["ref"], int(row["sign"]))
    stat = A.statistics(u, ok, c["gates"][int(row["sign"])])[row["stat"]]
    return M.first_alarm(A._episode_running_max(stat), float(row["threshold"]))


def union(*vectors: np.ndarray) -> np.ndarray:
    """Earliest alarm of any member; -1 only when no member fires."""
    stacked = np.stack(vectors)
    masked = np.where(stacked >= 0, stacked, np.iinfo(np.int64).max)
    out = masked.min(axis=0)
    return np.where(out == np.iinfo(np.int64).max, -1, out)


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=M.CACHE)
    p.add_argument("--out", type=Path, default=M.RESULTS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(M.SEED)

    frames = {c: M.load(c) for c in ("development_main", "external_8b")}
    dev = load_table(args.cache / "development_main_ops.npz")
    ext = {v: load_table(args.cache / f"external_8b_ops_{v}.npz") for v in EXT_TABLES}
    for v in EXT_TABLES:
        for col in ("quantity", "layer", "sign", "stat", "grid_k"):
            assert (dev["ops"][col].to_numpy() == ext[v]["ops"][col].to_numpy()).all(), col
    assert np.allclose(dev["ops"]["threshold"], ext["frozenvalue"]["ops"]["threshold"])

    n_mode = {c: {"drop": int((f["risk"] & (f["mode"] == M.MODE_DROP)).sum()),
                  "grasp": int((f["risk"] & (f["mode"] == M.MODE_GRASP)).sum()),
                  "all": int(f["risk"].sum()),
                  "other": int((f["risk"] & ~np.isin(f["mode"], M.MODES)).sum()),
                  "nonrisk": int((~f["risk"]).sum())}
              for c, f in frames.items()}
    M.write_json(args.out / "mode_sizes.json", n_mode)

    base = {c: {"protocol": baseline_frame(f, Q0_PROTOCOL),
                "extended": baseline_frame(f, Q0_EXTENDED)}
            for c, f in frames.items()}
    for c in frames:
        M.write_csv(args.out / f"baseline_per_mode_{c}.csv",
                    base[c]["extended"].assign(
                        in_protocol_sweep=lambda d: d.q0.isin(Q0_PROTOCOL)))

    # Why the protocol sweep stops at 39, stated as a computed identity rather
    # than an assertion: for q0 >= cap - 4, "still running at q0" with lead >= 4
    # requires length >= cap, which *is* the risk label, so the rule scores zero
    # false alarms by definition and is not a detector at all.
    leak = []
    for c, f in frames.items():
        caps = {s: int(f["length"][f["suite"] == s].max()) for s in set(f["suite"])}
        for s, cap in sorted(caps.items()):
            q0 = cap - M.HEADLINE_LEAD
            here = f["suite"] == s
            timely = here & (f["length"] > q0) & ((f["length"] - q0) >= M.HEADLINE_LEAD)
            leak.append({"cohort": c, "suite": s, "cap": cap, "q0_leak": q0,
                         "tp_in_suite": int((timely & f["risk"]).sum()),
                         "fp_in_suite": int((timely & ~f["risk"]).sum()),
                         "n_risk_in_suite": int((here & f["risk"]).sum()),
                         "identical_to_risk_label_in_suite":
                             bool(np.array_equal(timely[here], f["risk"][here]))})
    M.write_csv(args.out / "baseline_leak_diagnostic.csv", pd.DataFrame(leak))

    ops = dev["ops"]
    arm_of = ops["arm"].to_numpy()
    pools = A.pool_masks(ops)
    rowsets = {(pool, arm): mask & (arm_of == arm)
               for pool, mask in pools.items() for arm in A.ARMS}
    for pool, mask in pools.items():
        rowsets[(pool, "any")] = mask.copy()
    M.write_json(args.out / "pool_sizes.json",
                 {f"{p}|{a}": int(m.sum()) for (p, a), m in rowsets.items()})

    n_dev_task, n_ext_task = len(dev["tasks"]), len(ext[PRIMARY_EXT]["tasks"])
    all_dev = np.ones(n_dev_task, bool)
    all_ext = np.ones(n_ext_task, bool)

    def describe(op: int, budget: int, pool: str, arm: str, target: str) -> dict:
        r = ops.iloc[op]
        out = {"budget_dev_fp_lead4": budget, "pool": pool, "arm": arm,
               "selected_for": target, "op": op,
               "channel": f"{r['quantity']}|{r['layer']}", "sign": int(r["sign"]),
               "stat": r["stat"], "grid_k": int(r["grid_k"]),
               "within_episode_constant": float(r["within_episode_constant"]),
               "dev_threshold": float(r["threshold"]),
               "dev_median_alarm_chunk": float(r["median_chunk"])}
        for tag, table, mask, cohort in (
                ("dev", dev, all_dev, "development_main"),
                ("extF", ext["frozenvalue"], all_ext, "external_8b"),
                ("extR", ext["ratematched"], all_ext, "external_8b")):
            for lead in M.LEADS:
                t = totals(table["counts"], mask, lead)
                out[f"{tag}_fp_lead{lead}"] = int(t["fp"][op])
                out[f"{tag}_tp_all_lead{lead}"] = int(t["tp_all"][op])
                out[f"{tag}_tp_drop_lead{lead}"] = int(t["tp_drop"][op])
                out[f"{tag}_tp_grasp_lead{lead}"] = int(t["tp_grasp"][op])
            out[f"{tag}_median_alarm_chunk"] = float(
                table["ops"]["median_chunk"].iloc[op])
            fp4 = out[f"{tag}_fp_lead4"]
            # The protocol sweep stops at q0 = 39 for a reason: at q0 >= cap - 4
            # the rule "still running at q0, with lead >= 4" is *identical* to
            # the risk label (length == cap), so it scores 0 false alarms by
            # definition.  Only the protocol sweep is used for excess; the
            # extended sweep is carried as a leak diagnostic.
            for sweep, short in (("protocol", "pro"), ("extended", "leak")):
                b = base[cohort][sweep]
                for key in ("tp_all", "tp_drop", "tp_grasp"):
                    got = baseline_at(b, fp4, key)
                    out[f"{tag}_base{short}_{key}"] = got
                    if sweep == "protocol":
                        out[f"{tag}_excess_{key}"] = out[f"{tag}_{key}_lead4"] - got
        return out

    # ---------------- selections ------------------------------------------ #
    rows = []
    for budget in BUDGETS:
        for (pool, arm), mask in rowsets.items():
            for target in ("all", "drop", "grasp"):
                op = pick(dev, mask, all_dev, target, budget)
                if op is not None:
                    rows.append(describe(op, budget, pool, arm, target))
    picks = pd.DataFrame(rows)
    M.write_csv(args.out / "selected_operating_points.csv", picks)

    # ---------------- the 2 x 3 table and the interaction ------------------ #
    head = picks[(picks.budget_dev_fp_lead4 == HEADLINE_BUDGET)
                 & (picks.pool == "headline") & (picks.arm.isin(A.ARMS))
                 & (picks.selected_for.isin(("drop", "grasp")))]
    op_list = head["op"].tolist()
    keys = [f"{r.arm}|{r.selected_for}" for r in head.itertuples()]
    col = {k: i for i, k in enumerate(keys)}
    li = LEAD_INDEX[M.HEADLINE_LEAD]
    sub = ext[PRIMARY_EXT]["counts"][np.asarray(op_list)][:, :, :, li]
    draw = rng.integers(0, n_ext_task, size=(BOOT_DRAWS, n_ext_task))
    boot = {name: sub[:, :, gi][:, draw].sum(axis=2).T
            for gi, name in ((A.GROUP_NONRISK, "fp"), (A.GROUP_DROP, "tp_drop"),
                             (A.GROUP_GRASP, "tp_grasp"))}

    cells, tab = {}, []
    for mode in ("drop", "grasp"):
        for arm in A.ARMS:
            k = f"{arm}|{mode}"
            if k not in col:
                continue
            r = head[(head.arm == arm) & (head.selected_for == mode)].iloc[0]
            d = boot[f"tp_{mode}"][:, col[k]]
            cells[k] = {"point": int(r[f"extR_tp_{mode}_lead4"]),
                        "ci_lo": float(np.quantile(d, 0.025)),
                        "ci_hi": float(np.quantile(d, 0.975)),
                        "n_mode": n_mode["external_8b"][mode]}
            tab.append({
                "mode": mode, "arm": arm, "channel": r["channel"],
                "sign": r["sign"], "stat": r["stat"],
                "dev_tp": int(r[f"dev_tp_{mode}_lead4"]),
                "dev_n": n_mode["development_main"][mode],
                "dev_fp": int(r["dev_fp_lead4"]),
                "ext_tp": int(r[f"extR_tp_{mode}_lead4"]),
                "ext_n": n_mode["external_8b"][mode],
                "ext_fp": int(r["extR_fp_lead4"]),
                "ext_ci_lo": cells[k]["ci_lo"], "ext_ci_hi": cells[k]["ci_hi"],
                "ext_tp_frozen_value": int(r[f"extF_tp_{mode}_lead4"]),
                "ext_fp_frozen_value": int(r["extF_fp_lead4"]),
                "ext_baseline_tp": int(r[f"extR_basepro_tp_{mode}"]),
                "ext_excess_over_baseline": int(r[f"extR_excess_tp_{mode}"]),
                "ext_median_alarm_chunk": float(r["extR_median_alarm_chunk"]),
            })
    M.write_csv(args.out / "mode_arm_table.csv", pd.DataFrame(tab))

    contrasts = {}
    for arm in ("change_point", "persistence"):
        if f"{arm}|drop" not in col or f"{arm}|grasp" not in col:
            continue
        d_drop = (boot["tp_drop"][:, col[f"{arm}|drop"]]
                  - boot["tp_drop"][:, col["threshold|drop"]])
        d_grasp = (boot["tp_grasp"][:, col[f"{arm}|grasp"]]
                   - boot["tp_grasp"][:, col["threshold|grasp"]])
        did = d_drop - d_grasp
        pt_drop = cells[f"{arm}|drop"]["point"] - cells["threshold|drop"]["point"]
        pt_grasp = cells[f"{arm}|grasp"]["point"] - cells["threshold|grasp"]["point"]
        contrasts[arm] = {
            "gain_on_drop": {"point": pt_drop, "n_mode": n_mode["external_8b"]["drop"],
                             "ci_lo": float(np.quantile(d_drop, 0.025)),
                             "ci_hi": float(np.quantile(d_drop, 0.975))},
            "gain_on_grasp": {"point": pt_grasp, "n_mode": n_mode["external_8b"]["grasp"],
                              "ci_lo": float(np.quantile(d_grasp, 0.025)),
                              "ci_hi": float(np.quantile(d_grasp, 0.975))},
            "difference_in_differences": {
                "point": pt_drop - pt_grasp,
                "ci_lo": float(np.quantile(did, 0.025)),
                "ci_hi": float(np.quantile(did, 0.975)),
                "p_two_sided_sign": float(2 * min((did <= 0).mean(),
                                                  (did >= 0).mean()))},
        }
    M.write_json(args.out / "interaction.json",
                 {"cells": cells, "contrasts": contrasts,
                  "budget_dev_fp_lead4": HEADLINE_BUDGET,
                  "external_variant": PRIMARY_EXT,
                  "bootstrap_draws": BOOT_DRAWS,
                  "bootstrap_unit": f"task (external_8b, {n_ext_task} tasks)"})

    # ---------------- leave one task out ----------------------------------- #
    f_dev = frames["development_main"]
    _, dev_code = A.task_codes(f_dev)
    nonrisk_dev = np.bincount(dev_code[~f_dev["risk"]], minlength=n_dev_task)
    dev_tasks, ext_tasks = list(dev["tasks"]), list(ext[PRIMARY_EXT]["tasks"])

    loto_rows, loto_sum = [], []
    for budget in BUDGETS:
        for (pool, arm), mask in rowsets.items():
            if pool == "null_epconst":
                continue
            for target in ("all", "drop", "grasp"):
                acc = {"tp_drop": 0, "tp_grasp": 0, "tp_all": 0, "fp": 0}
                folds = 0
                for ti, name in enumerate(ext_tasks):
                    keep = np.ones(n_dev_task, bool)
                    if name in dev_tasks:
                        keep[dev_tasks.index(name)] = False
                    scale = nonrisk_dev[keep].sum() / nonrisk_dev.sum()
                    op = pick(dev, mask, keep, target, int(round(budget * scale)))
                    if op is None:
                        continue
                    take = np.zeros(n_ext_task, bool)
                    take[ti] = True
                    t = totals(ext[PRIMARY_EXT]["counts"], take, M.HEADLINE_LEAD)
                    r = ops.iloc[op]
                    fold = {"budget_dev_fp_lead4": budget, "pool": pool,
                            "arm": arm, "selected_for": target,
                            "held_out_task": name,
                            "in_development": name in dev_tasks,
                            "channel": f"{r['quantity']}|{r['layer']}",
                            "sign": int(r["sign"]), "stat": r["stat"],
                            "tp_drop": int(t["tp_drop"][op]),
                            "tp_grasp": int(t["tp_grasp"][op]),
                            "tp_all": int(t["tp_all"][op]),
                            "fp": int(t["fp"][op])}
                    loto_rows.append(fold)
                    for k in acc:
                        acc[k] += fold[k]
                    folds += 1
                loto_sum.append({"budget_dev_fp_lead4": budget, "pool": pool,
                                 "arm": arm, "selected_for": target,
                                 "n_folds": folds, **acc,
                                 "n_drop": n_mode["external_8b"]["drop"],
                                 "n_grasp": n_mode["external_8b"]["grasp"],
                                 "n_risk": n_mode["external_8b"]["all"]})
    M.write_csv(args.out / "loto_detail.csv", pd.DataFrame(loto_rows))
    M.write_csv(args.out / "loto_summary.csv", pd.DataFrame(loto_sum))

    # ---------------- per-task breakdown of the headline cells ------------- #
    f_ext = frames["external_8b"]
    _, ext_code = A.task_codes(f_ext)
    per_task = []
    for _, r in head.iterrows():
        op = int(r["op"])
        for ti, name in enumerate(ext_tasks):
            take = np.zeros(n_ext_task, bool)
            take[ti] = True
            t = totals(ext[PRIMARY_EXT]["counts"], take, M.HEADLINE_LEAD)
            n_d = int((f_ext["risk"] & (f_ext["mode"] == M.MODE_DROP)
                       & (ext_code == ti)).sum())
            n_g = int((f_ext["risk"] & (f_ext["mode"] == M.MODE_GRASP)
                       & (ext_code == ti)).sum())
            if n_d + n_g + int(t["fp"][op]) == 0:
                continue
            per_task.append({"arm": r["arm"], "selected_for": r["selected_for"],
                             "task": name, "n_drop": n_d, "n_grasp": n_g,
                             "tp_drop": int(t["tp_drop"][op]),
                             "tp_grasp": int(t["tp_grasp"][op]),
                             "fp": int(t["fp"][op])})
    M.write_csv(args.out / "per_task_headline.csv", pd.DataFrame(per_task))

    print("\n== 2x3 mode x arm (external_8b, rate-matched, lead>=4, dev FP<=%d) =="
          % HEADLINE_BUDGET)
    print(pd.DataFrame(tab).to_string(index=False))
    print("\n== interaction (difference in differences, task-clustered bootstrap) ==")
    print(json.dumps(contrasts, indent=2))
    print("\n== LOTO, headline pool, budget %d ==" % HEADLINE_BUDGET)
    ls = pd.DataFrame(loto_sum)
    print(ls[(ls.budget_dev_fp_lead4 == HEADLINE_BUDGET)
             & (ls.pool == "headline")].to_string(index=False))


if __name__ == "__main__":
    main()
