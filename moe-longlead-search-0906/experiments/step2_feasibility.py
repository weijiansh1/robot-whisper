"""Step 2: what does the arithmetic allow in the early window?

At chunk q an online detector only sees episodes still running (length > q);
alarms after an episode ends score lead < 0 and are discarded, so every
detector is already survivor-conditioned for free.  That makes the *survivor
composition* the hard ceiling on early precision:

    alarming on every survivor at chunk q  ->  TP = risks alive, FP = successes
                                               alive

This script prints the survivor table per suite and, for the deadlines that
matter (goal q14, object q12, spatial q6, long q36), the false alarms a
perfect-recall early head would have to pay.  It also reports the pooled
false-alarm budget: at lead >= 4 the frozen v8.3 spends 134 FP, so a new head
that adds more than a few dozen is not affordable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

DEADLINE = {"libero_goal": 14, "libero_long": 36,
            "libero_object": 12, "libero_spatial": 6}


def main() -> None:
    data = C.load_all(with_speed=False)

    rows = []
    for suite in C.SUITES:
        cap = C.CAP[suite]
        for q in range(0, cap + 1):
            ra = sa = 0
            for c, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                alive = m & (d["length"] > q)
                ra += int((alive & d["risk"]).sum())
                sa += int((alive & ~d["risk"]).sum())
            rows.append({"suite": suite, "chunk": q, "risk_alive": ra,
                         "safe_alive": sa,
                         "prior": ra / max(ra + sa, 1)})
    surv = pd.DataFrame(rows)
    surv.to_csv(C.RESULTS / "survivor_table.csv", index=False)

    print("=== survivor composition at the lead>=16 deadline ===")
    print("%-16s %8s %10s %10s %10s" % ("suite", "deadline", "risks alive",
                                        "safes alive", "prior"))
    for suite in C.SUITES:
        q = DEADLINE[suite]
        r = surv[(surv.suite == suite) & (surv.chunk == q)].iloc[0]
        print("%-16s %8d %10d %10d %9.3f"
              % (suite, q, r.risk_alive, r.safe_alive, r.prior))

    print("\n=== survivor prior by chunk (goal / object / spatial) ===")
    print("%-6s %s" % ("chunk", "  ".join("%-22s" % s for s in
                                          ("libero_goal", "libero_object",
                                           "libero_spatial"))))
    for q in range(4, 23, 2):
        cells = []
        for suite in ("libero_goal", "libero_object", "libero_spatial"):
            sub = surv[(surv.suite == suite) & (surv.chunk == q)]
            if sub.empty:
                cells.append("%-22s" % "-")
                continue
            r = sub.iloc[0]
            cells.append("%-22s" % ("%d risk / %d safe %.3f"
                                    % (r.risk_alive, r.safe_alive, r.prior)))
        print("q%-5d %s" % (q, "  ".join(cells)))

    # ---- what a perfect-recall early head would cost ---------------------
    print("\n=== cost of a PERFECT-RECALL alarm at the deadline ===")
    print("  (alarm every survivor at chunk = deadline; this is the ceiling on"
          " recall and the floor on how precise a head must be)")
    for suite in C.SUITES:
        q = DEADLINE[suite]
        r = surv[(surv.suite == suite) & (surv.chunk == q)].iloc[0]
        print("  %-16s +%4d TP  +%5d FP   -> precision %.4f"
              % (suite, r.risk_alive, r.safe_alive, r.prior))

    # ---- affordable precision -------------------------------------------
    print("\n=== affordable operating points, given v8.3 spends 134 FP @L4 ===")
    for suite, target in (("libero_goal", 213), ("libero_object", 81)):
        q = DEADLINE[suite]
        r = surv[(surv.suite == suite) & (surv.chunk == q)].iloc[0]
        for budget in (20, 40, 80):
            need = target / (target + budget)
            enrich = need / r.prior
            print("  %-14s recover %3d late risks for +%2d FP -> precision"
                  " %.3f, i.e. %.1fx the survivor prior %.4f"
                  % (suite, target, budget, need, enrich, r.prior))

    # ---- the same for a partial recovery ---------------------------------
    print("\n=== partial recovery: enrichment needed for +N TP at +20 FP ===")
    for suite in ("libero_goal", "libero_object"):
        q = DEADLINE[suite]
        r = surv[(surv.suite == suite) & (surv.chunk == q)].iloc[0]
        line = []
        for n_tp in (20, 50, 100, 200):
            prec = n_tp / (n_tp + 20)
            line.append("+%3d TP: %.1fx" % (n_tp, prec / r.prior))
        print("  %-16s (prior %.4f)  %s" % (suite, r.prior, "  ".join(line)))

    (C.RESULTS / "feasibility.json").write_text(json.dumps({
        "deadline": DEADLINE,
        "survivor_at_deadline": {
            s: surv[(surv.suite == s) & (surv.chunk == DEADLINE[s])]
            .iloc[0][["risk_alive", "safe_alive", "prior"]].to_dict()
            for s in C.SUITES},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
