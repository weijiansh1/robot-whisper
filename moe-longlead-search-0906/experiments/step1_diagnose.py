"""Step 1: where does the lead>=16 shortfall actually come from?

Two mutually exclusive ways a risk can fail the lead>=16 bar:
    (a) NEVER DETECTED  -- v8.3 never fires (first < 0), so a new head must
        supply a genuinely new detection, paying full false-alarm price;
    (b) DETECTED LATE   -- v8.3 fires but after the deadline cap-16.  Moving
        that alarm earlier converts a lead-4 TP into a lead-16 TP.  Under the
        union rule, an early head firing on an episode v8.3 *already* flags
        costs ZERO extra false alarms.

If (b) dominates, the search should be "same episodes, earlier" rather than
"more episodes"; the false-alarm budget for that is only the successes v8.3
does not already flag.  This script measures the split, per suite and per
task, and also reports the earliest chunk each frozen arm can physically fire.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def main() -> None:
    data = C.load_all(with_speed=False)
    v83 = {c: d["v83"] for c, d in data.items()}
    v7 = {c: d["v7"] for c, d in data.items()}

    print("=== earliest chunk each frozen arm ever fires ===")
    for arm, alarms in (("v7", v7), ("v8.3", v83)):
        allf = np.concatenate([alarms[c] for c in C.COHORTS])
        fired = allf[allf >= 0]
        print("  %-5s min q%-3d  p01 q%-3d  median q%-3d  fires on %d/%d"
              % (arm, fired.min(), np.percentile(fired, 1), np.median(fired),
                 len(fired), len(allf)))

    # ---- decomposition of the lead>=16 shortfall -------------------------
    rows = []
    for suite in C.SUITES:
        cap = C.CAP[suite]
        for lead in (4, 12, 16, 20):
            deadline = cap - lead
            n = never = late = intime = 0
            fp_ep = 0            # successes v8.3 flags at all (free budget)
            n_safe = 0
            late_first = []
            for c, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                f = v83[c]
                risk = m & d["risk"]
                n += int(risk.sum())
                never += int((risk & (f < 0)).sum())
                good = risk & (f >= 0) & (f <= deadline)
                intime += int(good.sum())
                bad = risk & (f >= 0) & (f > deadline)
                late += int(bad.sum())
                late_first.append(f[bad])
                safe = m & ~d["risk"]
                n_safe += int(safe.sum())
                fp_ep += int((safe & (f >= 0)).sum())
            lf = np.concatenate(late_first) if late_first else np.array([])
            rows.append({"suite": suite, "cap": cap, "lead": lead,
                         "deadline_q": deadline, "n_risk": n,
                         "in_time": intime, "late": late, "never": never,
                         "late_median_q": float(np.median(lf)) if len(lf) else np.nan,
                         "n_safe": n_safe, "safe_already_flagged": fp_ep})
    split = pd.DataFrame(rows)
    split.to_csv(C.RESULTS / "shortfall_decomposition.csv", index=False)

    print("\n=== v8.3 lead>=16 shortfall: late vs never ===")
    print("%-16s %4s %8s | %6s %6s %6s | %s"
          % ("suite", "cap", "deadline", "in", "late", "never", "late median q"))
    for _, r in split[split.lead == 16].iterrows():
        print("%-16s %4d %8d | %6d %6d %6d | %.1f"
              % (r.suite, r.cap, r.deadline_q, r.in_time, r.late, r.never,
                 r.late_median_q))
    tot = split[split.lead == 16]
    print("%-16s %4s %8s | %6d %6d %6d"
          % ("TOTAL", "", "", tot.in_time.sum(), tot.late.sum(),
             tot.never.sum()))

    print("\n=== same at lead>=12 ===")
    for _, r in split[split.lead == 12].iterrows():
        print("%-16s deadline q%-3d | in %4d late %4d never %4d | late med q%.1f"
              % (r.suite, r.deadline_q, r.in_time, r.late, r.never,
                 r.late_median_q))

    # ---- the free false-alarm budget -------------------------------------
    print("\n=== union arithmetic: free budget for an early head ===")
    n_safe_tot = sum(int((~d["risk"]).sum()) for d in data.values())
    flagged_safe = sum(int(((~d["risk"]) & (v83[c] >= 0)).sum())
                       for c, d in data.items())
    print("  successes total                     %6d" % n_safe_tot)
    print("  successes v8.3 already flags        %6d  (these are FREE: an early"
          " head firing on them adds no new FP)" % flagged_safe)
    print("  successes v8.3 never flags          %6d  (every early alarm here"
          " is a NEW false alarm)" % (n_safe_tot - flagged_safe))

    # ---- alarm-chunk histogram for goal / object risks -------------------
    print("\n=== when does v8.3 fire on goal / object risks? ===")
    for suite in ("libero_goal", "libero_object"):
        fs = []
        for c, d in data.items():
            m = (d["suite"] == suite) & d["risk"]
            fs.append(v83[c][m])
        fs = np.concatenate(fs)
        fired = fs[fs >= 0]
        print("  %-16s risks %4d  fired %4d  q: min %2d p10 %2d p25 %2d "
              "median %2d p75 %2d" % (suite, len(fs), len(fired), fired.min(),
                                      np.percentile(fired, 10),
                                      np.percentile(fired, 25),
                                      np.median(fired),
                                      np.percentile(fired, 75)))
        hist = np.bincount(fired, minlength=C.CAP[suite] + 1)
        print("       chunk hist:", " ".join("q%d:%d" % (i, v)
                                             for i, v in enumerate(hist) if v))

    # ---- per-task late/never for goal and object -------------------------
    print("\n=== goal / object per task at lead>=16 (deadline q14 / q12) ===")
    recs = {}
    for c, d in data.items():
        keys = np.array([f"{s}/{t}" for s, t in zip(d["suite"], d["task"])])
        f = v83[c]
        for k in np.unique(keys):
            if not (k.startswith("libero_goal") or k.startswith("libero_object")):
                continue
            suite = k.split("/")[0]
            dl = C.CAP[suite] - 16
            m = (keys == k) & d["risk"]
            if not m.any():
                continue
            r = recs.setdefault(k, {"task": k, "n": 0, "in": 0, "late": 0,
                                    "never": 0})
            r["n"] += int(m.sum())
            r["in"] += int((m & (f >= 0) & (f <= dl)).sum())
            r["late"] += int((m & (f >= 0) & (f > dl)).sum())
            r["never"] += int((m & (f < 0)).sum())
    tasks = pd.DataFrame(sorted(recs.values(), key=lambda r: -r["n"]))
    tasks.to_csv(C.RESULTS / "goal_object_per_task.csv", index=False)
    for _, r in tasks.iterrows():
        print("  %-58s n %4d  in %3d late %3d never %3d"
              % (r.task[-58:], r.n, r["in"], r.late, r.never))

    (C.RESULTS / "shortfall.json").write_text(json.dumps({
        "safe_total": n_safe_tot, "safe_already_flagged": flagged_safe,
        "lead16": {r.suite: {"in": int(r.in_time), "late": int(r.late),
                             "never": int(r.never)}
                   for _, r in split[split.lead == 16].iterrows()},
        "lead12": {r.suite: {"in": int(r.in_time), "late": int(r.late),
                             "never": int(r.never)}
                   for _, r in split[split.lead == 12].iterrows()},
    }, indent=2))


if __name__ == "__main__":
    main()
