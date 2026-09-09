"""Step 0: reproduce every published anchor before doing anything new.

Anchors (frozen v8.3, cap-free, lead in absolute chunks):
    lead >= 4   development 344/487 47FP | external 398/564 70FP
                legacy 225/307 17FP      | total 967/1358 134FP
    profile     L0 1167/209  L4 967/134  L8 711/104
                L12 617/66   L16 533/30  L20 352/14
    v7 profile  L0 1078/172  L4 870/111  L8 638/82
                L12 564/60   L16 513/29  L20 341/14
Also verifies the structural reachability table:
    goal q14 29/250, object q12 0/81, spatial q6 0/315, long q36 499/712.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def main() -> None:
    data = C.load_all()

    v83 = {c: d["v83"] for c, d in data.items()}
    v7 = {c: d["v7"] for c, d in data.items()}

    # 1. published lead>=4 per cohort
    print("=== anchor A: v8.3 @ lead>=4 per cohort ===")
    ok = True
    for c, (tp_e, n_e, fp_e) in C.ANCHOR_LEAD4.items():
        s = C.score(v83[c], data[c]["risk"], data[c]["length"], 4)
        n = int(data[c]["risk"].sum())
        good = (s["tp"], n, s["fp"]) == (tp_e, n_e, fp_e)
        ok &= good
        print("  %-18s %4d/%-4d %3dFP   expected %4d/%-4d %3dFP  %s"
              % (c, s["tp"], n, s["fp"], tp_e, n_e, fp_e,
                 "OK" if good else "MISMATCH"))

    # 2. published lead profile
    print("\n=== anchor B: v8.3 lead profile (whole corpus) ===")
    prof83 = C.profile(v83, data)
    prof7 = C.profile(v7, data)
    for lead in C.LEADS:
        exp = C.ANCHOR_PROFILE[lead]
        good = tuple(prof83[lead]) == exp
        ok &= good
        print("  lead>=%-2d  v7 %4d/%-4d   v8.3 %4d/%-4d  expected %4d/%-4d  %s"
              % (lead, prof7[lead][0], prof7[lead][1], prof83[lead][0],
                 prof83[lead][1], exp[0], exp[1], "OK" if good else "MISMATCH"))

    # 3. v8.3 rebuilt from flow_speed must equal the frozen vectors
    print("\n=== anchor C: v8.3 rebuilt from flow_speed ===")
    rebuilt = C.rebuild_v83(data)
    for c in C.COHORTS:
        same = int((rebuilt[c] == v83[c]).sum())
        print("  %-18s identical %5d/%-5d" % (c, same, len(v83[c])))
        ok &= same == len(v83[c])

    # 4. structural reachability: risks run to the cap, so lead >= L needs an
    #    alarm at chunk <= cap - L.  Confirm both the caps and the misses.
    print("\n=== anchor D: reachability at lead>=16 ===")
    rows = []
    for suite in C.SUITES:
        cap = C.CAP[suite]
        lens, hit16, n = [], 0, 0
        for c, d in data.items():
            m = (d["suite"] == suite) & d["risk"]
            if not m.any():
                continue
            lens.append(d["length"][m])
            n += int(m.sum())
            f = v83[c]
            hit16 += int((m & (f >= 0) & ((d["length"] - f) >= 16)).sum())
        lens = np.concatenate(lens)
        rows.append({"suite": suite, "cap": cap, "n_risk": n,
                     "len_min": int(lens.min()), "len_max": int(lens.max()),
                     "len_at_cap": int((lens == cap).sum()),
                     "deadline_q": cap - 16, "v83_lead16": hit16})
        print("  %-16s cap %2d  risks %4d  lengths %d..%d (%d at cap)"
              "  deadline q%-3d  v8.3 %3d/%-4d"
              % (suite, cap, n, lens.min(), lens.max(),
                 (lens == cap).sum(), cap - 16, hit16, n))
    reach = pd.DataFrame(rows)
    reach.to_csv(C.RESULTS / "anchor_reachability.csv", index=False)

    # 5. per-task concentration warning
    pt = C.per_task(v83, data, 4)
    print("\n=== anchor E: risk concentration (top 6 tasks) ===")
    for _, r in pt.head(6).iterrows():
        print("  %-64s %4d risks  v8.3 %4d TP" % (r.task[:64], r.n_risk, r.tp))

    (C.RESULTS / "anchor.json").write_text(json.dumps({
        "all_anchors_reproduced": bool(ok),
        "v83_profile": {str(k): list(v) for k, v in prof83.items()},
        "v7_profile": {str(k): list(v) for k, v in prof7.items()},
        "per_cohort_lead4": {c: C.score(v83[c], data[c]["risk"],
                                        data[c]["length"], 4)
                             for c in C.COHORTS},
    }, indent=2))
    print("\nALL ANCHORS REPRODUCED:", ok)
    assert ok, "anchors did not reproduce -- stop"


if __name__ == "__main__":
    main()
