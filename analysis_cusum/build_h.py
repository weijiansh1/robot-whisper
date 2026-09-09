"""Per-branch full pairwise Hellinger matrix H[b] (T,T), cell-mean (V2 placement).

H[b][i, j] = mean over the 40 (layer, token) cells of sqrt(1 - BC(P_i, P_j)).
The real per-step score is H[b][t-1, t]; the shuffle sentinel reads the same
matrix along a permuted chunk order, so the shuffled run uses exactly the same
distance definition and the same numbers, only reordered.
"""
import numpy as np
import pandas as pd

DATA = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
OUT = "/home/jovyan/work/himoe-vla/analysis_cusum"
EPS = 1e-12


def build(tag):
    X = np.load(f"{DATA}/{tag}_main.npy", mmap_mode="r")
    eid = np.load(f"{DATA}/{tag}_episode_id.npy")
    meta = pd.read_csv(f"{DATA}/{tag}_meta.csv")
    srt = np.argsort(eid, kind="stable")
    e = eid[srt]
    uq = np.unique(e)
    st = np.searchsorted(e, uq, "left")
    en = np.append(st[1:], len(e))
    order = {int(u): np.sort(srt[s:t]) for u, s, t in zip(uq, st, en)}
    Tmax = int(meta["T"].max())
    H = np.full((len(meta), Tmax, Tmax), np.nan, np.float32)
    for i, (ep, T) in enumerate(zip(meta["episode_id"].values, meta["T"].values)):
        rows = order[int(ep)]
        assert len(rows) == T
        P = np.asarray(X[rows], np.float32).reshape(T, 40, 32)
        P = P / np.maximum(P.sum(-1, keepdims=True), EPS)
        R = np.sqrt(P)
        bc = np.einsum("ice,jce->ijc", R, R, optimize=True)
        H[i, :T, :T] = np.sqrt(np.clip(1.0 - bc, 0.0, None)).mean(-1)
    np.save(f"{OUT}/H_{tag}.npy", H)
    print(tag, H.shape, "done", flush=True)


if __name__ == "__main__":
    for t in "AB":
        build(t)
