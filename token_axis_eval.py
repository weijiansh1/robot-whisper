"""Within-group AUC evaluation of the token-axis sweep + permutation null."""
import json
import numpy as np
import pandas as pd

ROOT = "/home/jovyan/work/himoe-vla"
A_LAB = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/analysis/candidate_physical_labels.csv"
B_DIR = f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
WIN = 8
TS = [15, 20, 25, 30, 35]


def load(tag, d):
    z = np.load(f"{ROOT}/.tokaxis_{tag}_d{d}.npz")
    eids = z["__eids__"]
    lens = z["__lens__"]
    off = np.concatenate([[0], np.cumsum(lens)])
    feats = [k for k in z.files if not k.startswith("__")]
    return eids, lens, off, {k: z[k] for k in feats}, feats


def window_score(series, off, i, t, win=WIN):
    """mean of series over steps [t-win+1 .. t] for branch i (nan-safe)."""
    lo, hi = off[i] + t - win + 1, off[i] + t + 1
    v = series[lo:hi]
    v = v[~np.isnan(v)]
    return v.mean() if len(v) else np.nan


def group_auc(scores, y, groups):
    """AUC = P(score_fail > score_succ) within group, pooled by pair count.
    y: 1 = success, 0 = failure."""
    num = 0.0
    den = 0.0
    per = {}
    for g in np.unique(groups):
        m = groups == g
        sf = scores[m][y[m] == 0]
        ss = scores[m][y[m] == 1]
        sf = sf[~np.isnan(sf)]
        ss = ss[~np.isnan(ss)]
        if len(sf) == 0 or len(ss) == 0:
            continue
        # rank-based AUC
        allv = np.concatenate([sf, ss])
        r = pd.Series(allv).rank().values
        a = (r[:len(sf)].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sf) * len(ss))
        n = len(sf) * len(ss)
        num += a * n
        den += n
        per[int(g)] = (a, n)
    return (num / den if den else np.nan), den, per


def build_matrix(tag, d, t):
    """Return (X [n_branch x n_feat], featnames, alive mask)."""
    eids, lens, off, F, feats = load(tag, d)
    feats = sorted(feats)
    alive = lens > t
    X = np.full((len(eids), len(feats)), np.nan, np.float32)
    for j, f in enumerate(feats):
        s = F[f]
        for i in np.where(alive)[0]:
            X[i, j] = window_score(s, off, i, t)
    return eids, X, feats, alive


def meta(tag):
    if tag == "A":
        lab = pd.read_csv(A_LAB)
        return (lab.set_index("episode_id")["success"].astype(int).to_dict(),
                lab.set_index("episode_id")["worker"].to_dict())
    summ = json.load(open(f"{B_DIR}/client/summaries.json"))
    return ({i: int(bool(s["success"])) for i, s in enumerate(summ)},
            {i: s["init_state_id"] for i, s in enumerate(summ)})


def run(tag, d):
    ymap, gmap = meta(tag)
    res = {}
    for t in TS:
        eids, X, feats, alive = build_matrix(tag, d, t)
        idx = np.where(alive)[0]
        y = np.array([ymap[int(e)] for e in eids])[idx]
        g = np.array([gmap[int(e)] for e in eids])[idx]
        Xa = X[idx]
        row = {}
        for j, f in enumerate(feats):
            a, n, _ = group_auc(Xa[:, j], y, g)
            row[f] = a
        res[t] = dict(row, __n__=len(idx), __npairs__=n,
                      __nsucc__=int(y.sum()), __nfail__=int((1 - y).sum()))
    return res, feats


if __name__ == "__main__":
    for tag in ("A", "B"):
        r, feats = run(tag, 9)
        print(f"=== {tag} d9 baseline act_all_1_10|hell ===")
        for t in TS:
            print(f"  t={t:2d} n={r[t]['__n__']:3d} pairs={r[t]['__npairs__']:5.0f} "
                  f"AUC={r[t]['act_all_1_10|hell']:.3f}")
