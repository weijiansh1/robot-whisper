"""Write the grid to /tmp/moe_sweep_stat.csv in the coordinator's schema (idempotent append)."""
import re, os, sys
import numpy as np, pandas as pd
OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
CSV = "/tmp/moe_sweep_stat.csv"
COLS = ["corpus", "statistic", "window", "t", "auc", "n_pos", "n_neg",
        "slice", "family", "n_pairs", "n_alive", "n_scored", "censored", "auc_failure"]


def group_counts(tag):
    meta = pd.read_csv(f"{OUT}/{tag}_meta.csv")
    out = {}
    for t in [15, 20, 25, 30, 35]:
        al = meta[meta["T"] > t]
        npos = nneg = 0
        for g, d in al.groupby("group"):
            p, n = int(d.success.sum()), int((~d.success.astype(bool)).sum())
            if p and n:                      # groups with 0 pairs contribute nothing
                npos += p; nneg += n
        out[t] = (npos, nneg)
    return out


def win_of(stat):
    m = re.search(r"_W(\d+)$", stat)
    if m:
        return int(m.group(1))
    m = re.search(r"_a([\d.]+)$", stat)
    if m:
        return f"ewm_a{m.group(1)}"
    m = re.search(r"lag(\d)$", stat)
    if m:
        return f"lag{m.group(1)}"
    return "none" if stat.startswith("I:") else "full"


def fam_of(stat):
    if stat.startswith("I:"):
        return "instantaneous"
    if stat.startswith("R:"):
        return "reference_other_branches"
    return "historical_self"


if __name__ == "__main__":
    g = pd.read_csv(f"{OUT}/grid_raw.csv")
    gc = {tag: group_counts(tag) for tag in ["A", "B"]}
    g["n_pos"] = [gc[c][t][0] for c, t in zip(g.corpus, g.t)]
    g["n_neg"] = [gc[c][t][1] for c, t in zip(g.corpus, g.t)]
    g["statistic"] = g["stat"]
    g["window"] = g["stat"].map(win_of)
    g["family"] = g["stat"].map(fam_of)
    g["auc"] = g["auc_success"]                       # P(score higher | success)
    g["auc_failure"] = 1.0 - g["auc_success"]         # P(score higher | failure)
    g["n_pairs"] = g["npairs"]
    out = g[COLS].sort_values(["corpus", "slice", "family", "statistic", "t"])
    hdr = not os.path.exists(CSV)
    out.to_csv(CSV, index=False, mode="w")            # full rewrite: canonical snapshot
    print("wrote", CSV, out.shape)
    print(out.head(3).to_string(index=False))
