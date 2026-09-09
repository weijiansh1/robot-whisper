"""Step 11: can routing see a DROPPED OBJECT early, or not?

goal's 250 risks are 194 `object_released_or_dropped_before_goal`; object's 81
are 36.  These are the 331 "reachable but missed" cases.  The mode is one where
routing is expected to look normal -- the object leaves the gripper, but the
plan the model is executing does not change -- so the honest question is
whether routing can see it at all, early, and if not, to say so cleanly.

Three tests, all on development for the estimator and repeated on the sealed
cohorts for the detector:

  1. detection by failure mode, at lead >= 4 / 12 / 16, for v8.3 and for the
     two frozen arms -- does the gain land on drop-type failures?
  2. within-task fixed-chunk AUC, survivors only, computed SEPARATELY for
     drop-type and non-drop-type risks against the same successes.  If routing
     carried the drop, drop-type risks would separate at least as well as the
     rest; if the drop is invisible, they separate worse.
  3. the same at the chunk where a lead >= 16 alarm would have to land.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
from step3_ceiling import task_auc, tp_at_fp
from step4_ceiling_raw import build_bank
from step9_freeze_and_seal import apply_arms, fit_on_development

RUN_ID = {"development_main": "right-50x8-20260903",
          "external_8b": "right-50x8b-20260903",
          "legacy_main16x32": "right-16x32"}
DROP = "object_released_or_dropped_before_goal"
DEADLINE = {"libero_goal": 14, "libero_long": 36,
            "libero_object": 12, "libero_spatial": 6}
SERIES = ("set_inflow_all", "set_outflow_all", "set_jacc_adj_all",
          "state_mob_all", "flow_path_back", "mob_back_s9", "tok_disp_fbr")


def attach_modes(data):
    lab = pd.read_csv(C.ROOT / "VLA_MUI_HUB/physical-failure-labels/results"
                      "/episodes.csv")
    lab = lab[["suite", "task_name", "run_id", "episode_index",
               "primary_failure_reason"]]
    for cohort, d in data.items():
        key = pd.DataFrame({"suite": d["suite"], "task_name": d["task"],
                            "run_id": RUN_ID[cohort],
                            "episode_index": np.arange(len(d["risk"]))})
        # episode_index in the cache is the per-task episode id
        key["episode_index"] = _episode_ids(cohort, d)
        merged = key.merge(lab, on=["suite", "task_name", "run_id",
                                    "episode_index"], how="left")
        d["mode"] = merged.primary_failure_reason.fillna("(none)").to_numpy()
        matched = int((d["mode"][d["risk"]] != "(none)").sum())
        print("  %-18s risks %4d, labelled %4d (%.0f%%)"
              % (cohort, int(d["risk"].sum()), matched,
                 100 * matched / max(int(d["risk"].sum()), 1)))
    return data


def _episode_ids(cohort, d):
    from step9_freeze_and_seal import ARM_A  # noqa: F401  (path setup)
    V4 = C.ROOT / "moe-v4-0904/results/layerwise_mobility"
    path = {"development_main": V4 / "main_reference.npz",
            "external_8b": V4 / "external_8b.npz",
            "legacy_main16x32": C.ROOT / "moe-v4-0904/results/cache16x32_v4"
                                         "/layerwise_mobility.npz"}[cohort]
    with np.load(path, allow_pickle=False) as a:
        return a["episode"].astype(int)


def main() -> None:
    data = C.load_all()
    print("=== joining physical failure labels ===")
    data = attach_modes(data)
    series = {c: build_bank(c, d) for c, d in data.items()}
    fitted = fit_on_development(series["development_main"])
    arms = {}
    for c, d in data.items():
        a, b = apply_arms(series[c], fitted)
        arms.setdefault("v8.3", {})[c] = d["v83"]
        arms.setdefault("v8.3+A+B", {})[c] = C.union(d["v83"], a, b)

    # ---- 1. detection by failure mode -----------------------------------
    print("\n=== detection by physical failure mode (whole corpus) ===")
    modes = {}
    for c, d in data.items():
        for m in np.unique(d["mode"][d["risk"]]):
            modes[m] = modes.get(m, 0)
    rows = []
    for mode in sorted(modes):
        for lead in (4, 12, 16):
            n = tp83 = tpnew = 0
            for c, d in data.items():
                sel = d["risk"] & (d["mode"] == mode)
                n += int(sel.sum())
                for arm, acc in (("v8.3", 0), ("v8.3+A+B", 1)):
                    f = arms[arm][c]
                    t = int((sel & (f >= 0) & ((d["length"] - f) >= lead)).sum())
                    if acc == 0:
                        tp83 += t
                    else:
                        tpnew += t
            rows.append({"mode": mode, "lead": lead, "n": n,
                         "v83": tp83, "new": tpnew, "gain": tpnew - tp83})
    modetab = pd.DataFrame(rows)
    modetab.to_csv(C.RESULTS / "failure_mode_detection.csv", index=False)
    print("%-46s %5s | %-16s | %-16s | %-16s"
          % ("mode", "n", "lead>=4", "lead>=12", "lead>=16"))
    for mode in sorted(modes, key=lambda m: -int(
            modetab[(modetab["mode"] == m) & (modetab.lead == 4)].n.iloc[0])):
        cells = []
        for lead in (4, 12, 16):
            r = modetab[(modetab["mode"] == mode) & (modetab.lead == lead)].iloc[0]
            cells.append("%4d->%4d (%+d)" % (r.v83, r.new, r.gain))
        n = int(modetab[(modetab["mode"] == mode) & (modetab.lead == 4)].n.iloc[0])
        print("%-46s %5d | %-16s | %-16s | %-16s"
              % (mode[:46], n, cells[0], cells[1], cells[2]))

    # ---- 2/3. is the drop visible in routing at all? --------------------
    print("\n=== fixed-chunk separability: drop-type risks vs the rest ===")
    print("  within task, survivors only, same successes as the negative"
          " class; development + external pooled for n")
    out = []
    for suite in ("libero_goal", "libero_object"):
        for chunk in (8, DEADLINE[suite], 18):
            for name in SERIES:
                vals, risk_d, risk_o, task = [], [], [], []
                for c, d in data.items():
                    m = (d["suite"] == suite) & (d["length"] > chunk)
                    if not m.any():
                        continue
                    vals.append(series[c][name][m, chunk])
                    isdrop = d["risk"][m] & (d["mode"][m] == DROP)
                    isoth = d["risk"][m] & (d["mode"][m] != DROP)
                    risk_d.append(isdrop)
                    risk_o.append(isoth)
                    task.append(np.char.add(c + "|", d["task"][m].astype(str)))
                if not vals:
                    continue
                v = np.concatenate(vals)
                rd, ro = np.concatenate(risk_d), np.concatenate(risk_o)
                tk = np.concatenate(task)
                safe = ~(rd | ro)
                a_d = task_auc(v[rd | safe], rd[rd | safe], tk[rd | safe])
                a_o = task_auc(v[ro | safe], ro[ro | safe], tk[ro | safe])
                out.append({"suite": suite, "chunk": chunk, "series": name,
                            "n_drop": int(rd.sum()), "n_other": int(ro.sum()),
                            "n_safe": int(safe.sum()),
                            "auc_drop": a_d, "auc_other": a_o,
                            "gap": (abs(a_o - 0.5) - abs(a_d - 0.5))})
    sep = pd.DataFrame(out)
    sep.to_csv(C.RESULTS / "drop_separability.csv", index=False)
    for suite in ("libero_goal", "libero_object"):
        print("\n  --- %s ---" % suite)
        sub = sep[sep.suite == suite]
        print("  %-4s %-20s %7s %7s | %8s %8s %8s"
              % ("q", "series", "n_drop", "n_other", "AUC drop", "AUC oth",
                 "|oth|-|drop|"))
        for _, r in sub.iterrows():
            print("  q%-3d %-20s %7d %7d | %8.3f %8.3f %+8.3f"
                  % (r.chunk, r.series, r.n_drop, r.n_other, r.auc_drop,
                     r.auc_other, r.gap))
        print("  mean |AUC-0.5|: drop %.3f, other %.3f  over %d cells"
              % ((sub.auc_drop - 0.5).abs().mean(),
                 (sub.auc_other - 0.5).abs().mean(), len(sub)))

    print("\n=== summary of the drop question ===")
    d4 = modetab[(modetab["mode"] == DROP) & (modetab.lead == 4)].iloc[0]
    d12 = modetab[(modetab["mode"] == DROP) & (modetab.lead == 12)].iloc[0]
    d16 = modetab[(modetab["mode"] == DROP) & (modetab.lead == 16)].iloc[0]
    print("  %s: %d risks corpus-wide" % (DROP, d4.n))
    print("    lead>=4  %d -> %d (%+d)   lead>=12 %d -> %d (%+d)"
          "   lead>=16 %d -> %d (%+d)"
          % (d4.v83, d4.new, d4.gain, d12.v83, d12.new, d12.gain,
             d16.v83, d16.new, d16.gain))
    pooled = sep[sep.suite.isin(("libero_goal", "libero_object"))]
    print("  fixed-chunk separability, pooled over %d cells: drop-type risks"
          " reach mean |AUC-0.5| = %.3f, all other risks %.3f"
          % (len(pooled), (pooled.auc_drop - 0.5).abs().mean(),
             (pooled.auc_other - 0.5).abs().mean()))

    (C.RESULTS / "failure_modes.json").write_text(json.dumps({
        "modes": modetab.to_dict("records"),
        "separability_mean": {
            "drop": float((pooled.auc_drop - 0.5).abs().mean()),
            "other": float((pooled.auc_other - 0.5).abs().mean()),
            "n_cells": int(len(pooled))},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
