"""Turn analysis_future/moe_future.csv into the deliverable tables."""
import sys
import numpy as np
import pandas as pd

CSV = "/home/jovyan/work/himoe-vla/analysis_future/moe_future.csv"
VISION = ["obj_disp_p1", "obj_disp_p2", "which_moves", "which_moves_late",
          "obj_nc", "obj_ncfar", "obj_free_disp", "obj_gt_arm", "undo_nc"]
CORE = ["eef_disp", "eef_path", "obj_disp", "dgoal", "undo", "grip_flips"]
BINARY = {"undo", "undo_nc", "obj_nc", "obj_ncfar", "obj_gt_arm",
          "which_moves", "which_moves_late"}


def load():
    df = pd.read_csv(CSV)
    return df.drop_duplicates(
        subset=["corpus", "target", "horizon", "feature_block", "t_bin", "metric"],
        keep="last")


def wide(df, corpus, metric_rows=("r2", "auc"), tbin="all", shuf=False):
    d = df[(df.corpus == corpus) & (df.t_bin == tbin) & df.metric.isin(metric_rows)]
    d = d[d.feature_block.str.startswith("SHUF_") == shuf]
    if shuf:
        d = d.assign(feature_block=d.feature_block.str.replace("SHUF_", "", regex=False))
    return d.pivot_table(index=["target", "horizon"], columns="feature_block",
                         values="value")


def fmt(v, w=6):
    return " " * w if pd.isna(v) else f"{v:+.3f}"


def main():
    df = load()
    out = []
    P = out.append
    for corpus in sorted(df.corpus.unique()):
        sub = df[df.corpus == corpus]
        P(f"\n{'='*100}\nCORPUS {corpus}\n{'='*100}")

        # ---------------- sentinels
        P("\n## Sentinel 1 - shuffle (chunk order permuted within branch)")
        sh = wide(df, corpus, shuf=True)
        tr = wide(df, corpus, shuf=False)
        if len(sh):
            cols = [c for c in ("route", "t1", "t2") if c in sh.columns]
            P(f"{'target':18s} {'m':>2s} " + "".join(
                f"{c+'(true)':>14s}{c+'(shuf)':>14s}" for c in cols))
            for (t, m), r in sh.iterrows():
                if (t, m) not in tr.index:
                    continue
                P(f"{t:18s} {m:2d} " + "".join(
                    f"{fmt(tr.loc[(t, m), c]):>14s}{fmt(r[c]):>14s}" for c in cols))

        P("\n## Sentinel 2 - clock only (t, T, t/T; clock_rich = smooth basis, "
          "clock_causal = t only)")
        w = wide(df, corpus)
        cols = [c for c in ("clock_causal", "clock", "clock_rich", "route", "t1")
                if c in w.columns]
        P(f"{'target':18s} {'m':>2s} " + "".join(f"{c:>15s}" for c in cols))
        for (t, m), r in w.iterrows():
            P(f"{t:18s} {m:2d} " + "".join(f"{fmt(r[c]):>15s}" for c in cols))

        # ---------------- main table
        P("\n## Main table  (within-group OOF; R2 for continuous, AUC for binary)")
        cols = [c for c in ("route", "t1", "t1+route", "t2", "t2+route") if c in w.columns]
        P(f"{'target':18s} {'m':>2s} " + "".join(f"{c:>13s}" for c in cols)
          + f"{'inc/t1':>12s}{'CI':>22s}{'n':>8s}")
        inc = sub[sub.feature_block == "inc_route_over_t1"].pivot_table(
            index=["target", "horizon"], columns="metric", values="value")
        nn = sub[(sub.feature_block == "t1") & (sub.metric.isin(("r2", "auc")))
                 & (sub.t_bin == "all")].set_index(["target", "horizon"])["n"]
        for t in CORE + VISION:
            for m in (1, 2, 4, 8):
                if (t, m) not in w.index:
                    continue
                r = w.loc[(t, m)]
                line = f"{t:18s} {m:2d} " + "".join(f"{fmt(r[c]):>13s}" for c in cols)
                if (t, m) in inc.index:
                    i = inc.loc[(t, m)]
                    line += f"{i.get('delta', np.nan):+12.4f}" + \
                        f"   [{i.get('delta_lo95', np.nan):+.4f},{i.get('delta_hi95', np.nan):+.4f}]"
                line += f"{int(nn.get((t, m), 0)):8d}"
                P(line)

        # ---------------- increments
        P("\n## Routing increment over each tier (grouped/branch cluster bootstrap 95% CI)")
        for base in ("t1", "t2"):
            b = sub[sub.feature_block == f"inc_route_over_{base}"].pivot_table(
                index=["target", "horizon"], columns="metric", values="value")
            if not len(b):
                continue
            P(f"\n  --- over {base} ---")
            for (t, m), r in b.iterrows():
                sig = "  *" if (r.get("delta_lo95", -1) > 0 or r.get("delta_hi95", 1) < 0) else ""
                P(f"  {t:18s} m={m:2d}  delta={r.get('delta', np.nan):+.4f}"
                  f"  [{r.get('delta_lo95', np.nan):+.4f}, {r.get('delta_hi95', np.nan):+.4f}]{sig}")

        # ---------------- time bins
        for tb in ("q0-11", "q12-23", "q24+"):
            wb = wide(df, corpus, tbin=tb)
            if not len(wb):
                continue
            P(f"\n## Time bin {tb}")
            cols = [c for c in ("route", "t1", "t1+route") if c in wb.columns]
            P(f"{'target':18s} {'m':>2s} " + "".join(f"{c:>13s}" for c in cols))
            for (t, m), r in wb.iterrows():
                if m != 4:
                    continue
                P(f"{t:18s} {m:2d} " + "".join(f"{fmt(r[c]):>13s}" for c in cols))

    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    main()
