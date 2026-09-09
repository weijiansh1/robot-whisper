"""Assemble the deliverable: matched-false-alarm comparison, class breakouts,
alarm-phase distributions, ARL, and the honest cross-corpus config check."""
import sys

import numpy as np
import pandas as pd

import driver
import evalm
import lib
import seq

TARGET = {"A": 0.137, "B": 0.051}
KEY = ["method", "W", "rule", "dens", "whiten", "tstart"]


def interp_at(sub, target, cols):
    """Interpolate metrics along the achieved-FA axis at the target FA."""
    s = sub.sort_values("fa")
    fa = s["fa"].values
    out = {}
    for c in cols:
        v = s[c].values.astype(float)
        m = np.isfinite(v)
        out[c] = float(np.interp(target, fa[m], v[m])) if m.sum() >= 2 else np.nan
    lo = s[s.fa <= target]
    hi = s[s.fa >= target]
    out["fa_lo"] = float(lo["fa"].max()) if len(lo) else np.nan
    out["fa_hi"] = float(hi["fa"].min()) if len(hi) else np.nan
    return out


def load_sweeps():
    a = pd.read_csv("sweep_full.csv")
    b = pd.read_csv("sweep_tvd.csv")
    a["tstart"] = 1
    df = pd.concat([a, b], ignore_index=True)
    df["W"] = df["W"].astype(int)
    df["tstart"] = df["tstart"].astype(int)
    for c in ("dens", "whiten"):
        df[c] = df[c].fillna("-")
    return df


def main():
    df = load_sweeps()
    cols = ["det", "phase_p50", "phase_p25", "phase_p75", "t_p50", "arl0", "arl1",
            "arl1_mean_alarm_t", "t_dip", "frac_early", "delay_p50"]
    # ---------- matched-FA table for every config, both corpora ----------
    recs = []
    for (corpus, *k), sub in df[df["pop"] == "all_fail"].groupby(["corpus"] + KEY):
        r = interp_at(sub, TARGET[corpus], cols)
        recs.append(dict(zip(["corpus"] + KEY, [corpus] + list(k))),)
        recs[-1].update(r)
        recs[-1]["peak_auc"] = float(sub["peak_auc"].iloc[0])
    M = pd.DataFrame(recs)
    M.to_csv("matched_fa_all_configs.csv", index=False)

    seqM = M[M.method == "sequential"]
    pa = seqM[seqM.corpus == "A"].set_index(KEY)
    pb = seqM[seqM.corpus == "B"].set_index(KEY)
    common = pa.index.intersection(pb.index)
    rho_det = pd.Series(pa.loc[common, "det"].values).corr(
        pd.Series(pb.loc[common, "det"].values), method="spearman")
    rho_ph = pd.Series(pa.loc[common, "phase_p50"].values).corr(
        pd.Series(pb.loc[common, "phase_p50"].values), method="spearman")
    best_a = pa.loc[common, "det"].idxmax()
    best_b = pb.loc[common, "det"].idxmax()

    lines = ["", "## 2. Matched-false-alarm comparison", "",
             f"Config grid: {len(common)} sequential configurations shared by both corpora.",
             f"Spearman rank correlation of detection-at-matched-FA between corpus A "
             f"and corpus B across configs: **{rho_det:.3f}**; of median alarm phase: "
             f"**{rho_ph:.3f}**.",
             f"Best config on A: {dict(zip(KEY, best_a))} -> A det "
             f"{pa.loc[best_a, 'det']:.3f}, but on B det {pb.loc[best_a, 'det']:.3f}.",
             f"Best config on B: {dict(zip(KEY, best_b))} -> B det "
             f"{pb.loc[best_b, 'det']:.3f}, but on A det {pa.loc[best_b, 'det']:.3f}.",
             ""]
    print("\n".join(lines))

    # ---------- headline operating points ----------
    heads = {
        "seq_cusum_primary": dict(method="sequential", W=1, rule="cusum", dens="tvd",
                                  whiten="ar1", tstart=1),
        "seq_sprt_primary": dict(method="sequential", W=1, rule="sprt", dens="tvd",
                                 whiten="ar1", tstart=1),
        "seq_cusum_W8": dict(method="sequential", W=8, rule="cusum", dens="tvd",
                             whiten="ar1", tstart=1),
        "seq_xcorpus": None,      # filled per corpus below
        "fixed_W8K3": dict(method="fixed", W=8, rule="K3", dens="-", whiten="-", tstart=1),
        "fixed_W12K3": dict(method="fixed", W=12, rule="K3", dens="-", whiten="-", tstart=1),
    }
    xsel = {"A": dict(zip(KEY, best_b)), "B": dict(zip(KEY, best_a))}

    tbl = ["| corpus | detector | FA | det | phase p25 | **phase p50** | phase p75 | "
           "t p50 | ARL0 | ARL1 | modes | dip |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_out = []
    for corpus in "AB":
        for name, spec in heads.items():
            sp = xsel[corpus] if spec is None else spec
            m = (df.corpus == corpus) & (df["pop"] == "all_fail")
            for k, v in sp.items():
                m &= df[k] == v
            sub = df[m]
            if len(sub) == 0:
                continue
            r = interp_at(sub, TARGET[corpus], cols)
            tbl.append(f"| {corpus} | {name} | {TARGET[corpus]:.3f} | {r['det']:.3f} | "
                       f"{r['phase_p25']:.3f} | **{r['phase_p50']:.3f}** | "
                       f"{r['phase_p75']:.3f} | {r['t_p50']:.1f} | {r['arl0']:.1f} | "
                       f"{r['arl1']:.1f} | {r['t_dip']:.2f} | {r['frac_early']:.2f} |")
            rows_out.append(dict(section="headline", corpus=corpus, detector=name,
                                 **{k: str(v) for k, v in sp.items()}, fa_target=TARGET[corpus],
                                 **r))
    lines += tbl
    print("\n".join(tbl))
    lib.append_rows(rows_out)
    lib.append_md("\n".join(lines))
    pd.DataFrame(rows_out).to_csv("headline.csv", index=False)
    M.to_csv("matched_fa_all_configs.csv", index=False)
    with open("xsel.txt", "w") as f:
        f.write(repr({"A": xsel["A"], "B": xsel["B"], "rho_det": rho_det,
                      "rho_phase": rho_ph}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
