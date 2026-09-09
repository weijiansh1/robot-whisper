import sys, time, json
sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_stat_axis")
import numpy as np, pandas as pd
from stats_lib import (build_dense, consecutive_series, instantaneous_bank,
                       historical_bank, reference_bank, within_group_auc)

OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
TS = [15, 20, 25, 30, 35]


def load(tag, slice_name):
    X = np.load(f"{OUT}/{tag}_{slice_name}.npy")
    meta = pd.read_csv(f"{OUT}/{tag}_meta.csv")
    eid = np.load(f"{OUT}/{tag}_episode_id.npy")
    if X.ndim == 3:                      # front slice: (N, layers, 32) -> add token axis
        X = X[:, :, None, :]
    n_layers, n_tokens = X.shape[1], X.shape[2]
    P, valid = build_dense(X, meta, eid, contiguous=(tag == "B"))
    return P, valid, meta, n_layers, n_tokens


def score_bank(P, valid, meta, n_layers, n_tokens, t, dser):
    d_hel, d_js, d_l1 = dser
    alive = valid[:, t]
    B = P.shape[0]
    bank = {}
    inst = instantaneous_bank(P[alive, t], n_tokens, n_layers)
    for k, v in inst.items():
        a = np.full(B, np.nan, np.float32); a[alive] = v; bank[k] = a
    bank.update(historical_bank(P, valid, d_hel, d_js, d_l1, t))
    bank.update(reference_bank(P, valid, d_hel, t,
                               meta["group"].values, meta["success"].values.astype(int)))
    return bank


def run(tag, slice_name):
    P, valid, meta, nl, nt = load(tag, slice_name)
    y = meta["success"].values.astype(bool); g = meta["group"].values
    t0 = time.time()
    dser = (consecutive_series(P, valid, "hellinger"),
            consecutive_series(P, valid, "js"),
            consecutive_series(P, valid, "l1"))
    rows = []; banks = {}
    for t in TS:
        if not valid[:, t].any():
            continue
        bank = score_bank(P, valid, meta, nl, nt, t, dser)
        banks[t] = bank
        nalive = int(valid[:, t].sum())
        for k, v in bank.items():
            auc, npairs = within_group_auc(v, y, g)
            rows.append(dict(corpus=tag, slice=slice_name, stat=k, t=t,
                             auc_success=auc, npairs=npairs, n_alive=nalive,
                             n_scored=int(np.isfinite(v).sum()),
                             censored=bool(nalive < len(meta))))
    df = pd.DataFrame(rows)
    print(f"{tag}/{slice_name} grid done {time.time()-t0:.1f}s  {len(df)} cells", flush=True)
    return df, banks, (P, valid, meta, dser, nl, nt)


if __name__ == "__main__":
    allr = []
    ctx = {}
    for tag in ["A", "B"]:
        for sl in ["main", "front"]:
            df, banks, c = run(tag, sl)
            allr.append(df); ctx[(tag, sl)] = (banks, c)
    res = pd.concat(allr, ignore_index=True)
    res.to_csv(f"{OUT}/grid_raw.csv", index=False)
    print(res.groupby(["corpus", "slice"]).size())
    # sanity: baseline
    b = res[(res.stat == "H:hel_prev_mean_W8") & (res.slice == "main")]
    print(b[["corpus", "t", "auc_success", "npairs", "n_alive"]].to_string(index=False))
