"""Evaluate the stored OOF predictions on strata where tier 1 is structurally blind.

The corpus has a near-fixed scene layout (pot 1 at (-0.05, 0.25) +- 0.013 m, pot 2 at
(0.04, 0.05) +- 0.015 m in every held-out init state, 0.17-0.25 m apart), so eef
position alone identifies which pot the arm is working on.  The only way to remove that
channel is to restrict evaluation to chunks where the arm is far from BOTH pots, so its
own coordinates cannot say which object is in play.  Models are unchanged; this is a
slice of the same out-of-fold predictions.
"""
import csv
import os
import numpy as np

CACHE = "/tmp/moe_future_cache"
ROOT = "/home/jovyan/work/himoe-vla"
BLOCKS = ("clock_causal", "clock_rich", "route", "t1", "t1+route", "t2")
BINARY = {"undo", "undo_nc", "obj_nc", "obj_ncfar", "obj_gt_arm",
          "which_moves", "which_moves_late"}
HOR = (1, 2, 4, 8)


def auc(y, p):
    o = np.argsort(p, kind="mergesort")
    ys = y[o].astype(np.float64)
    Wp = ys.sum()
    Wn = len(ys) - Wp
    if Wp <= 0 or Wn <= 0:
        return np.nan
    wn = 1.0 - ys
    return float((ys * (np.cumsum(wn) - wn)).sum() / (Wp * Wn))


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return np.nan if ss <= 0 else 1.0 - ((y - p) ** 2).sum() / ss


def main(corpus):
    d = np.load(os.path.join(CACHE, f"corpus{corpus}.npz"))
    z = np.load(os.path.join(CACHE, f"oof_{corpus}_main.npz"))
    group, branch = d["group"], d["branch"]
    T2 = d["T2"]
    # tier-2 raw geometry columns: t2 = [t1(251) | g2_now | lag | d1 | d2 | ...]
    # recompute the distances directly instead of trusting offsets
    sim_off = 251
    p1 = T2[:, sim_off + 47: sim_off + 50]
    p2 = T2[:, sim_off + 50: sim_off + 53]
    eef = d["T1"][:, 0:3]
    dmin = np.minimum(np.linalg.norm(eef - p1, axis=1), np.linalg.norm(eef - p2, axis=1))
    aperture = d["T1"][:, 8]
    rows = []
    strata = {
        "arm_clear_0.12": dmin > 0.12,
        "arm_clear_0.12_open": (dmin > 0.12) & (aperture > 0.04),
        "arm_clear_0.20": dmin > 0.20,
    }
    for t in ("which_moves", "which_moves_late", "obj_nc", "obj_gt_arm",
              "obj_disp", "obj_disp_p1", "obj_disp_p2", "eef_disp", "grip_flips"):
        for mi, m in enumerate(HOR):
            key = f"{t}|{m}"
            if f"{key}|__y" not in z.files:
                continue
            Y = d[f"Y_{t}"][:, mi]
            idx = np.concatenate([np.flatnonzero(group == g)[
                np.isfinite(Y[np.flatnonzero(group == g)])] for g in np.unique(group)])
            y = z[f"{key}|__y"]
            assert len(idx) == len(y) and np.allclose(branch[idx], z[f"{key}|__b"])
            g = group[idx]
            fn = auc if t in BINARY else r2
            mn = "auc" if t in BINARY else "r2"
            for sname, smask in strata.items():
                s = smask[idx]
                if s.sum() < 200:
                    continue
                if t in BINARY and len(np.unique(y[s])) < 2:
                    continue
                for b in BLOCKS:
                    k = f"{key}|{b}"
                    if k not in z.files:
                        continue
                    p = z[k]
                    v = []
                    for gg in np.unique(g):
                        sel = s & (g == gg)
                        if sel.sum() < 50:
                            continue
                        if t in BINARY and len(np.unique(y[sel])) < 2:
                            continue
                        r = fn(y[sel], p[sel])
                        if np.isfinite(r):
                            v.append(r)
                    if v:
                        rows.append((corpus, t, m, b, sname, mn,
                                     float(np.mean(v)), int(s.sum())))
                if t in BINARY:
                    rows.append((corpus, t, m, "__prevalence", sname, "rate",
                                 float(y[s].mean()), int(s.sum())))
    return rows


if __name__ == "__main__":
    out = []
    for c in ("A", "B"):
        out += main(c)
    for path in ("/tmp/moe_future.csv",
                 os.path.join(ROOT, "analysis_future/moe_future.csv")):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(out)
    print(f"appended {len(out)} stratified rows")
