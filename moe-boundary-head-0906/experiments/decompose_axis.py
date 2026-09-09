"""Is the chunk boundary actually a new axis?

The boundary is one leg of a triangle in sqrt-probability space, per chunk q:

    A = p[q-1, layer, step 9]      the end of the previous pass
    B = p[q,   layer, step 9]      the end of this pass
    C = p[q,   layer, step 0]      the start of this pass

    seam  = || A - C || / sqrt(2)   <- the "boundary reshuffle"
    step9 = || A - B || / sqrt(2)   <- adjacent-query mobility, i.e. v7's freeze
    chord = || C - B || / sqrt(2)   <- within-query denoising displacement,
                                       i.e. the axis v8's flow/curvature heads read

If `seam` is a function of `step9` and `chord` then it is not a new axis: it is
a re-mixture of the two axes v7 and v8 already have.  This script measures how
much of it is left over, separately for the state token and the action tokens,
and it is the thing that decides the question before any detector is built.

The state token is checked first, because the whole "the reshuffle energy is
mostly in the state token, 3.9x the action token" claim rests on it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import BASE_CELLS, COHORTS, OUT, load_all

CHUNKS = (5, 7, 9, 12, 16, 20)


def r2(y, x):
    """R^2 of an OLS fit of y on [1, x...]; x is a list of regressors."""
    design = np.column_stack([np.ones(len(y))] + list(x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss if ss else np.nan


def main() -> None:
    data = load_all()
    report = {"cohorts": {}}

    print("===== 1. state token 在去噪轴上根本不动 =====")
    print("chord = ||p[q,step0] - p[q,step9]||/sqrt2，即一次前向内部的位移\n")
    print("%-18s %-14s %10s %10s %10s %10s"
          % ("cohort", "cell", "seam均值", "step9均值", "chord均值", "chord/seam"))
    state_rows = []
    for cohort in COHORTS:
        d = data[cohort]
        for cell in BASE_CELLS:
            s, w, c = (float(np.nanmean(d[cell])),
                       float(np.nanmean(d[f"wc_{cell}"])),
                       float(np.nanmean(d[f"chord_{cell}"])))
            print("%-18s %-14s %10.5f %10.5f %10.5f %9.4f"
                  % (cohort, cell, s, w, c, c / s))
            state_rows.append({"cohort": cohort, "cell": cell, "seam": s,
                               "step9": w, "chord": c, "chord_over_seam": c / s})
    pd.DataFrame(state_rows).to_csv(OUT / "triangle_magnitudes.csv", index=False)

    print("\n===== 2. seam 与 step9 的相关（同一条序列？）=====")
    print("%-18s %-14s %10s %10s %10s"
          % ("cohort", "cell", "Pearson r", "Spearman", "逐元素相同"))
    same_rows = []
    for cohort in COHORTS:
        d = data[cohort]
        for cell in BASE_CELLS:
            a, b = d[cell], d[f"wc_{cell}"]
            m = np.isfinite(a) & np.isfinite(b)
            pear = float(np.corrcoef(a[m], b[m])[0, 1])
            spear = float(stats.spearmanr(a[m], b[m]).statistic)
            ident = bool(np.array_equal(a[m], b[m]))
            print("%-18s %-14s %10.6f %10.6f %10s"
                  % (cohort, cell, pear, spear, ident))
            same_rows.append({"cohort": cohort, "cell": cell, "pearson": pear,
                              "spearman": spear, "bit_identical": ident})
    pd.DataFrame(same_rows).to_csv(OUT / "seam_vs_step9.csv", index=False)

    print("\n===== 3. seam 能被 {step9, chord} 解释多少（OLS R^2）=====")
    print("R^2=1 表示 seam 不是新轴，只是 v7 freeze 轴与 v8 flow 轴的重新组合")
    print("%-18s %-14s %10s %10s %10s"
          % ("cohort", "cell", "R2~step9", "R2~chord", "R2~两者"))
    dec_rows = []
    for cohort in COHORTS:
        d = data[cohort]
        for cell in BASE_CELLS:
            a, w, c = d[cell], d[f"wc_{cell}"], d[f"chord_{cell}"]
            m = np.isfinite(a) & np.isfinite(w) & np.isfinite(c)
            y, xw, xc = a[m], w[m], c[m]
            one = r2(y, [xw])
            two = r2(y, [xc])
            both = r2(y, [xw, xc, xw * xc, xw ** 2, xc ** 2])
            print("%-18s %-14s %10.4f %10.4f %10.4f"
                  % (cohort, cell, one, two, both))
            dec_rows.append({"cohort": cohort, "cell": cell, "r2_step9": one,
                             "r2_chord": two, "r2_both_quadratic": both})
    pd.DataFrame(dec_rows).to_csv(OUT / "seam_decomposition.csv", index=False)

    print("\n===== 4. 残差里还剩什么：seam 对 {step9, chord} 的残差的 AUC =====")
    print("如果 seam 有独立于两条已有轴的判别力，残差应当仍然显著")
    from reproduce_exploratory import auc_at_chunk

    rng = np.random.default_rng(20260906)
    res_rows = []
    for cohort in COHORTS:
        d = data[cohort]
        for cell in ("front_action", "back_action"):
            resid = np.full(d[cell].shape, np.nan)
            a, w, c = d[cell], d[f"wc_{cell}"], d[f"chord_{cell}"]
            m = np.isfinite(a) & np.isfinite(w) & np.isfinite(c)
            design = np.column_stack([np.ones(m.sum()), w[m], c[m],
                                      w[m] * c[m], w[m] ** 2, c[m] ** 2])
            beta, *_ = np.linalg.lstsq(design, a[m], rcond=None)
            resid[m] = a[m] - design @ beta
            probe = dict(d)
            probe[f"resid_{cell}"] = resid
            for q in CHUNKS:
                got = auc_at_chunk(probe, f"resid_{cell}", q, rng)
                raw = auc_at_chunk(d, cell, q, rng)
                if got is None or raw is None:
                    continue
                res_rows.append({"cohort": cohort, "cell": cell, "chunk": q,
                                 "auc_raw": raw["auc"], "sig_raw": raw["sig"],
                                 "auc_resid": got["auc"], "lo": got["lo"],
                                 "hi": got["hi"], "sig_resid": got["sig"]})
    res = pd.DataFrame(res_rows)
    res.to_csv(OUT / "seam_residual_auc.csv", index=False)
    for cohort in COHORTS:
        print("\n-- %s --" % cohort)
        print("%-14s %s" % ("cell", "  ".join("q%-18d" % q for q in CHUNKS)))
        for cell in ("front_action", "back_action"):
            cells = []
            for q in CHUNKS:
                r = res[(res.cohort == cohort) & (res.cell == cell)
                        & (res.chunk == q)]
                if not len(r):
                    cells.append("%-20s" % "-")
                    continue
                r = r.iloc[0]
                cells.append("%.3f%s->%.3f[%.2f,%.2f]%s"
                             % (r.auc_raw, "*" if r.sig_raw else " ",
                                r.auc_resid, r.lo, r.hi,
                                "*" if r.sig_resid else " "))
            print("%-14s %s" % (cell, "  ".join(cells)))

    report["state_moves_on_denoising_axis"] = {
        c: {cell: float(np.nanmean(data[c][f"chord_{cell}"]))
            for cell in BASE_CELLS} for c in COHORTS}
    report["seam_vs_step9_pearson"] = {
        c: {cell: float(np.corrcoef(
            data[c][cell][np.isfinite(data[c][cell])],
            data[c][f"wc_{cell}"][np.isfinite(data[c][cell])])[0, 1])
            for cell in BASE_CELLS} for c in COHORTS}
    report["decomposition"] = pd.DataFrame(dec_rows).to_dict("records")
    report["residual_auc"] = res.to_dict("records")
    report["n_residual_significant"] = int(res.sig_resid.sum())
    report["n_residual_cells"] = int(len(res))
    (OUT / "decompose_axis.json").write_text(json.dumps(report, indent=2,
                                                        default=float))

    print("\n结论：残差 AUC 在 %d/%d 个格子上显著（原始 seam 是 %d/%d）"
          % (int(res.sig_resid.sum()), len(res), int(res.sig_raw.sum()), len(res)))


if __name__ == "__main__":
    main()
