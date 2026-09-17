"""Alarm statistics of the online recurrence-rate detector on the 610 new episodes (shadow run in the native arm)."""
import json, glob, collections
from pathlib import Path
import numpy as np

OUT = Path("/home/swj/data/libero-runtime/simulations/control-scale")
jobs = {j["tag"]: j for j in json.loads((OUT / "jobs.json").read_text())}
eps = []
for tag, j in jobs.items():
    f = OUT / "native" / tag / "summary.json"
    if not f.exists(): continue
    s = json.loads(f.read_text()); c = np.load(OUT / "native" / tag / "control.npz")
    if s.get("status") != "completed": continue
    rr = c["per_query"][:, 2]
    eps.append(dict(tag=tag, bench=j["bench"], task=j["task_id"], category=j["category"] or "-", success=bool(s["success"]), steps=int(s["action_steps"]), rr=rr))


def first_alarm(rr, theta, k):
    run = 0
    for q in range(13, len(rr)):
        if not np.isfinite(rr[q]): continue
        run = run + 1 if rr[q] >= theta else 0
        if run >= k: return q
    return None


def stats(sub, theta=0.4, k=2):
    fa = [first_alarm(e["rr"], theta, k) for e in sub]
    fails = [(e, a) for e, a in zip(sub, fa) if not e["success"]]; succs = [(e, a) for e, a in zip(sub, fa) if e["success"]]
    tp = sum(1 for _, a in fails if a is not None); fp = sum(1 for _, a in succs if a is not None)
    lead = [520 - 10 * a for _, a in fails if a is not None]
    fq = [a for _, a in fails if a is not None]
    return dict(n_fail=len(fails), n_succ=len(succs), recall=tp / len(fails) if fails else float("nan"), fa_rate=fp / len(succs) if succs else float("nan"),
                precision=tp / (tp + fp) if tp + fp else float("nan"), tp=tp, fp=fp,
                first_q_median=float(np.median(fq)) if fq else float("nan"), first_q_iqr=[float(np.percentile(fq, 25)), float(np.percentile(fq, 75))] if fq else None,
                budget_used_median=float(np.median(fq)) * 10 / 520 if fq else float("nan"), lead_steps_median=float(np.median(lead)) if lead else float("nan"),
                fp_first_q_median=float(np.median([a for _, a in succs if a is not None])) if fp else float("nan"),
                fp_steps_to_success_median=float(np.median([e["steps"] - 10 * a for e, a in succs if a is not None])) if fp else float("nan"))


res = {}
print("%-10s %-22s n(fail/succ)  recall   FA rate  precision  first alarm q (IQR)   budget used  lead steps | FP alarm q, steps to success" % ("bench", "category"))
for bench in ("libero10", "pro-swap", "plus"):
    sub = [e for e in eps if e["bench"] == bench]
    st = stats(sub); res[bench] = st
    print("%-10s %-22s %3d/%3d      %5.1f%%  %5.1f%%   %5.1f%%    q%.0f (%s)   %4.0f%%      %4.0f   | q%.0f, %.0f" % (bench, "all", st["n_fail"], st["n_succ"], 100 * st["recall"], 100 * st["fa_rate"], 100 * st["precision"],
          st["first_q_median"], "-".join("%.0f" % v for v in st["first_q_iqr"]) if st["first_q_iqr"] else "-", 100 * st["budget_used_median"], st["lead_steps_median"], st["fp_first_q_median"], st["fp_steps_to_success_median"]))
    if bench == "plus":
        res["plus_by_category"] = {}
        for cat in sorted({e["category"] for e in sub}):
            s2 = stats([e for e in sub if e["category"] == cat]); res["plus_by_category"][cat] = s2
            print("%-10s %-22s %3d/%3d      %5.1f%%  %5.1f%%   %5.1f%%    q%.0f" % ("", cat, s2["n_fail"], s2["n_succ"], 100 * s2["recall"], 100 * s2["fa_rate"], 100 * s2["precision"], s2["first_q_median"]))
allst = stats(eps); res["all"] = allst
print("%-10s %-22s %3d/%3d      %5.1f%%  %5.1f%%   %5.1f%%    q%.0f" % ("all", "", allst["n_fail"], allst["n_succ"], 100 * allst["recall"], 100 * allst["fa_rate"], 100 * allst["precision"], allst["first_q_median"]))
print("\nper base task (libero10 + plus pooled): recall / FA")
res["by_task"] = {}
for t in range(10):
    s2 = stats([e for e in eps if e["task"] == t and e["bench"] != "pro-swap"]); res["by_task"][t] = s2
    print("  task%02d  fail %2d succ %2d  recall %5.1f%%  FA %5.1f%%" % (t, s2["n_fail"], s2["n_succ"], 100 * s2["recall"], 100 * s2["fa_rate"]))
print("\nthreshold sweep on the new episodes (libero10 + plus; pro-swap has no successes):")
res["sweep"] = []
nonpro = [e for e in eps if e["bench"] != "pro-swap"]
for theta in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    for k in (2, 3):
        s2 = stats(nonpro, theta, k); res["sweep"].append(dict(theta=theta, k=k, **s2))
        print("  theta %.1f k=%d: recall %5.1f%%  FA %5.1f%%  precision %5.1f%%  first alarm q%.0f  lead %.0f steps" % (theta, k, 100 * s2["recall"], 100 * s2["fa_rate"], 100 * s2["precision"], s2["first_q_median"], s2["lead_steps_median"]))
json.dump(res, open(OUT / "alarm-stats.json", "w"), indent=2, default=float)
