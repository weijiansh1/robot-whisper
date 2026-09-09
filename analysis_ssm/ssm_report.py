"""Final tables: headline comparison, rank sensitivity, real-vs-shuffle per cell."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L

MAINR, MAIND = 32, 8
PRIM = ["innov_w1", "innov_w3", "innov_w8", "innov_w12",
        "nis_w1", "nis_w3", "nis_w8", "nis_w12", "zfnorm_w8", "dzf_w8", "pcaz_w8"]


def main():
    d = pd.read_csv(f"{L.OUT}/moe_ssm.csv")
    p = d.pivot_table(index=["corpus", "variant", "r", "d", "t", "stat", "metric"],
                      columns="name", values="value", aggfunc="first").reset_index()
    p.to_csv(f"{L.OUT}/moe_ssm_pivot.csv", index=False)
    p.to_csv("/tmp/moe_ssm_pivot.csv", index=False)

    L.note("\n## 3. Innovation detection vs the scalar baseline "
           f"(pre-declared main latent config r={MAINR}, d={MAIND})")
    L.note("`auc_succ` = P(score | success > score | failure); `det` = max(auc, 1-auc) "
           "(the sign is chosen with the label, as in the previous rounds).\n")
    L.note("| corpus | t | baseline det | stat | auc_succ | raw det | p_perm raw | "
           "resid det | p_perm resid | rho vs baseline |")
    L.note("|---|---|---|---|---|---|---|---|---|---|")
    for corpus in ("A", "B"):
        for t in (20, 25, 30):
            q = p[(p.corpus == corpus) & (p.variant == "real") & (p.t == t)]
            bd = q[(q.stat == "base_w8") & (q.metric == "raw")].det.iloc[0]
            bp = q[(q.stat == "base_w8") & (q.metric == "raw")].p_perm.iloc[0]
            L.note(f"| {corpus} | {t} | {bd:.3f} (p={bp:.3f}) | *baseline* | | | | | | |")
            for st in ("innov_w8", "nis_w8", "nis_w12", "zfnorm_w8"):
                rw = q[(q.stat == st) & (q.metric == "raw") & (q.r == MAINR) & (q.d == MAIND)]
                rs = q[(q.stat == st) & (q.metric == "resid") & (q.r == MAINR) & (q.d == MAIND)]
                if not len(rw):
                    continue
                L.note(f"| {corpus} | {t} | {bd:.3f} | {st} | {rw.auc_succ.iloc[0]:.3f} | "
                       f"{rw.det.iloc[0]:.3f} | {rw.p_perm.iloc[0]:.3f} | "
                       f"{rs.det.iloc[0]:.3f} | {rs.p_perm.iloc[0]:.3f} | "
                       f"{rw.rho_base.iloc[0]:+.3f} |")

    L.note("\n## 5. Sensitivity to latent rank (raw det AUC, real data)")
    for corpus in ("A", "B"):
        for t in (25, 30):
            q = p[(p.corpus == corpus) & (p.variant == "real") & (p.t == t) &
                  (p.metric == "raw") & (p.stat.isin(["innov_w8", "nis_w8", "nis_w12"]))]
            tab = q.pivot_table(index=["r", "d"], columns="stat", values="det").round(3)
            bd = p[(p.corpus == corpus) & (p.variant == "real") & (p.t == t) &
                   (p.stat == "base_w8") & (p.metric == "raw")].det.iloc[0]
            L.note(f"\n**{corpus}, t={t}** (baseline det {bd:.3f})\n")
            L.note("```\n" + tab.to_string() + "\n```")

    L.note("\n## Sentinel 1 detail: same statistic, real vs order-shuffled")
    L.note("| corpus | t | family max REAL | family max SHUFFLE | cells where real>shuffle | "
           "median(real - shuffle) |")
    L.note("|---|---|---|---|---|---|")
    for corpus in ("A", "B"):
        for t in (20, 25, 30):
            base_q = (p.corpus == corpus) & (p.t == t) & (p.metric == "raw") & \
                     (~p.stat.isin(["base_w8", "base_cum"]))
            r_ = p[base_q & (p.variant == "real")].set_index(["r", "d", "stat"]).det
            s_ = p[base_q & (p.variant == "shuffle")].set_index(["r", "d", "stat"]).det
            j = pd.concat([r_.rename("real"), s_.rename("shuf")], axis=1).dropna()
            L.note(f"| {corpus} | {t} | {j['real'].max():.3f} | {j['shuf'].max():.3f} | "
                   f"{(j['real'] > j['shuf']).mean():.2f} ({len(j)} cells) | "
                   f"{(j['real'] - j['shuf']).median():+.3f} |")
            L.emit(corpus=corpus, t=t, stat="shuffle_sentinel", metric="family",
                   real_max=j["real"].max(), shuf_max=j["shuf"].max(),
                   frac_real_gt=float((j["real"] > j["shuf"]).mean()),
                   med_diff=float((j["real"] - j["shuf"]).median()), ncells=len(j))
    L.flush("moe_ssm_probe.csv")


if __name__ == "__main__":
    main()
