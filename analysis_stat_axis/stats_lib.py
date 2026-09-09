"""Statistic bank + within-group AUC evaluation for MoE routing failure detection."""
import numpy as np, pandas as pd

LOG32 = np.log(32.0)
EPS = 1e-12


# ---------------------------------------------------------------- data assembly
def build_dense(X, meta, eid, n_group_col="group", contiguous=False):
    """Return P (B, Tmax, C, 32) float32, valid (B,Tmax) bool, meta rows aligned.

    X: (N, ..., 32) row-major slice array. meta: DataFrame with episode_id, T.
    Rows for an episode are ordered by ascending global row index (== control_step order).
    """
    C = int(np.prod(X.shape[1:-1]))
    Xf = X.reshape(X.shape[0], C, 32)
    B = len(meta)
    Tmax = int(meta["T"].max())
    P = np.zeros((B, Tmax, C, 32), np.float32)
    valid = np.zeros((B, Tmax), bool)
    order = {}
    if not contiguous:
        srt = np.argsort(eid, kind="stable")
        e_srt = eid[srt]
        bounds = np.searchsorted(e_srt, np.unique(e_srt), side="left")
        uq = np.unique(e_srt)
        ends = np.append(bounds[1:], len(e_srt))
        for u, s, e in zip(uq, bounds, ends):
            order[int(u)] = np.sort(srt[s:e])       # ascending global row idx = temporal order
    for i, (epid, T) in enumerate(zip(meta["episode_id"].values, meta["T"].values)):
        if contiguous:
            r0 = int(meta["row0"].values[i]); rows = np.arange(r0, r0 + T)
        else:
            rows = order[int(epid)]
        assert len(rows) == T, (epid, len(rows), T)
        P[i, :T] = Xf[rows]
        valid[i, :T] = True
    return P, valid


# ---------------------------------------------------------------- primitives
def _sorted_desc(P):
    return np.sort(P, axis=-1)[..., ::-1]


def hellinger(P, Q):
    """Per-cell Hellinger; P,Q broadcastable (..., C, 32) -> (..., C)."""
    bc = np.sqrt(np.clip(P, 0, None) * np.clip(Q, 0, None)).sum(-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, None))


def js_div(P, Q):
    M = 0.5 * (P + Q)
    def kl(a, b):
        m = a > EPS
        return np.where(m, a * np.log(np.where(m, a, 1.0) / np.clip(b, EPS, None)), 0.0).sum(-1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def l1_dist(P, Q):
    return np.abs(P - Q).sum(-1)


def entropy(P):
    m = P > EPS
    return -(np.where(m, P * np.log(np.where(m, P, 1.0)), 0.0)).sum(-1)


def gini(P):
    s = np.sort(P, axis=-1)          # ascending
    n = P.shape[-1]
    idx = np.arange(1, n + 1, dtype=np.float32)
    return (2.0 * (idx * s).sum(-1)) / n - (n + 1.0) / n


# ---------------------------------------------------------------- instantaneous bank
def instantaneous_bank(Pt, n_tokens, n_layers):
    """Pt: (B, C, 32) probs at one query index. Returns dict name -> (B,) score.
    Cell ordering is (layer, token) C-order so reshape (n_layers, n_tokens)."""
    B, C, _ = Pt.shape
    ent = entropy(Pt)                                  # (B,C)
    srt = _sorted_desc(Pt)
    top1 = srt[..., 0]
    top4 = srt[..., :4].sum(-1)
    gap45 = srt[..., 3] - srt[..., 4]
    coll = (Pt ** 2).sum(-1)
    pr = 1.0 / np.clip(coll, EPS, None)
    bc_u = Pt.clip(0).__pow__(0.5).sum(-1) / np.sqrt(32.0)   # Bhattacharyya coeff to uniform
    bhat = -np.log(np.clip(bc_u, EPS, None))
    hel_u = np.sqrt(np.clip(1 - bc_u, 0, None))
    kl_u_p = (-LOG32 - np.log(np.clip(Pt, EPS, None)).mean(-1))  # KL(u||p)
    g = gini(Pt)

    out = {}
    out["I:ent_norm"] = (ent / LOG32).mean(1)
    out["I:top1"] = top1.mean(1)
    out["I:top4mass"] = top4.mean(1)
    out["I:gap45"] = gap45.mean(1)
    out["I:kl_p_unif"] = (LOG32 - ent).mean(1)
    out["I:kl_unif_p"] = kl_u_p.mean(1)
    out["I:bhatt_unif"] = bhat.mean(1)
    out["I:hell_unif"] = hel_u.mean(1)
    out["I:gini"] = g.mean(1)
    out["I:partratio"] = (pr / 32.0).mean(1)

    # dispersion across the n_tokens tokens (std over tokens, mean over layers)
    if n_tokens > 1:
        sh = (B, n_layers, n_tokens)
        out["I:ent_tokstd"] = (ent.reshape(sh)).std(2).mean(1)
        out["I:top1_tokstd"] = (top1.reshape(sh)).std(2).mean(1)
        out["I:pr_tokstd"] = (pr.reshape(sh) / 32.0).std(2).mean(1)
        # routing heterogeneity across tokens: Hellinger of each token to the layer-token-mean
        Pl = Pt.reshape(B, n_layers, n_tokens, 32)
        pbar = Pl.mean(2, keepdims=True); pbar = pbar / pbar.sum(-1, keepdims=True)
        out["I:tok_heterog"] = hellinger(Pl, pbar).mean((1, 2))
    if n_layers > 1:
        sh2 = (B, n_layers, n_tokens)
        out["I:ent_laystd"] = (ent.reshape(sh2)).std(1).mean(1)
        Pl2 = Pt.reshape(B, n_layers, n_tokens, 32)
        lbar = Pl2.mean(1, keepdims=True); lbar = lbar / lbar.sum(-1, keepdims=True)
        out["I:lay_heterog"] = hellinger(Pl2, lbar).mean((1, 2))
    return out


# ---------------------------------------------------------------- historical bank
def consecutive_series(P, valid, metric="hellinger"):
    """d[b,t] = mean-over-cells distance between step t and t-1 (t>=1); NaN where unavailable."""
    B, T, C, E = P.shape
    f = {"hellinger": hellinger, "js": js_div, "l1": l1_dist}[metric]
    d = np.full((B, T), np.nan, np.float32)
    for t in range(1, T):
        ok = valid[:, t] & valid[:, t - 1]
        if ok.any():
            d[ok, t] = f(P[ok, t], P[ok, t - 1]).mean(-1)
    return d


def historical_bank(P, valid, d_hel, d_js, d_l1, t, windows=(3, 5, 8, 12, 16)):
    """Return dict name -> (B,) score at query index t (NaN where not computable)."""
    B = P.shape[0]
    alive = valid[:, t]
    out = {}

    def emit(name, arr):
        v = np.full(B, np.nan, np.float32)
        v[alive] = arr
        out[name] = v

    Pt = P[alive, t]                                     # (b, C, 32)
    for W in windows:
        if t - W < 0:
            continue
        seg = d_hel[alive, t - W + 1:t + 1]               # W consecutive distances
        emit(f"H:hel_prev_mean_W{W}", seg.mean(1))
        emit(f"H:hel_prev_std_W{W}", seg.std(1))
        emit(f"H:hel_prev_max_W{W}", seg.max(1))
        # distance to own window mean (window of W states ending at t, excluding t)
        Wm = P[alive, t - W:t].mean(1)
        Wm = Wm / Wm.sum(-1, keepdims=True)
        emit(f"H:hel_to_winmean_W{W}", hellinger(Pt, Wm).mean(-1))
    # metric controls at W=8
    emit("H:js_prev_mean_W8", d_js[alive, t - 7:t + 1].mean(1))
    emit("H:l1_prev_mean_W8", d_l1[alive, t - 7:t + 1].mean(1))

    # drift from origin / path statistics
    P0 = P[alive, 0]
    hf = hellinger(Pt, P0).mean(-1)
    emit("H:hel_to_first", hf)
    path = d_hel[alive, 1:t + 1].sum(1)
    emit("H:cum_speed", path / t)
    emit("H:straightness", hf / np.clip(path, EPS, None))
    # exponentially weighted change rate
    for a in (0.15, 0.3, 0.5):
        ser = d_hel[alive, 1:t + 1]                       # (b, t)
        L = ser.shape[1]
        w = a * (1 - a) ** np.arange(L - 1, -1, -1, dtype=np.float32)
        w = w / w.sum()
        emit(f"H:ewm_rate_a{a}", (ser * w).sum(1))
    # autocorrelation of the change-rate series
    ser = d_hel[alive, 1:t + 1]
    ser = ser - ser.mean(1, keepdims=True)
    den = (ser ** 2).sum(1)
    for lag in (1, 2, 3):
        num = (ser[:, lag:] * ser[:, :-lag]).sum(1)
        emit(f"H:acf_lag{lag}", num / np.clip(den, EPS, None))
    return out


def reference_bank(P, valid, d_hel, t, groups, labels, n_folds=5, seed=0,
                   windows=(3, 5, 8, 12)):
    """Distances to reference distributions built from OTHER branches.

    Supervised references (success/failure class means at the same query index) are
    cross-fitted 5-fold over branches WITHIN each group, stratified by label, so the
    scored branch never contributes to its own reference. The cohort references use
    plain leave-one-out and consume NO labels.
    """
    B = P.shape[0]
    alive = valid[:, t]
    keys = ["R:hel_to_cohort_LOO", "R:hel_to_succ_ref", "R:hel_to_fail_ref"]
    keys += [f"R:hel_to_succ_ref_W{W}" for W in windows]
    keys += [f"R:hel_to_cohort_LOO_W{W}" for W in windows]
    keys += [f"R:hel_to_fail_ref_W{W}" for W in windows]
    out = {k: np.full(B, np.nan, np.float32) for k in keys}
    rng = np.random.default_rng(seed)
    for g in np.unique(groups):
        idx = np.where((groups == g) & alive)[0]
        if len(idx) < 3:
            continue
        n = len(idx)
        yg = labels[idx]
        fold = np.empty(n, int)
        for cls in (0, 1):
            w = np.where(yg == cls)[0]
            perm = rng.permutation(len(w))
            fold[w[perm]] = np.arange(len(w)) % n_folds

        def do(Pg, suf):
            # label-free LOO cohort mean
            tot = Pg.sum(0)
            loo = (tot[None] - Pg) / (n - 1)
            loo = loo / loo.sum(-1, keepdims=True)
            out["R:hel_to_cohort_LOO" + suf][idx] = hellinger(Pg, loo).mean(-1)
            # supervised class references, cross-fitted
            for cls, base in ((1, "R:hel_to_succ_ref"), (0, "R:hel_to_fail_ref")):
                for f in range(n_folds):
                    tr = np.where((fold != f) & (yg == cls))[0]
                    te = np.where(fold == f)[0]
                    if len(tr) == 0 or len(te) == 0:
                        continue
                    ref = Pg[tr].mean(0); ref = ref / ref.sum(-1, keepdims=True)
                    out[base + suf][idx[te]] = hellinger(Pg[te], ref[None]).mean(-1)

        do(P[idx, t], "")
        for W in windows:
            if t - W + 1 < 0:
                continue
            Pw = P[idx, t - W + 1:t + 1].mean(1)
            Pw = Pw / Pw.sum(-1, keepdims=True)
            do(Pw, f"_W{W}")
    for W in ("",) + tuple(f"_W{w}" for w in windows):
        out["R:succ_minus_fail_ref" + W] = out["R:hel_to_succ_ref" + W] - out["R:hel_to_fail_ref" + W]
    return out


# ---------------------------------------------------------------- AUC
def _auc_pairs(scores, y):
    """Mann-Whitney AUC P(score_pos > score_neg) with ties=0.5; returns (auc, npairs)."""
    ok = np.isfinite(scores)
    s, yy = scores[ok], y[ok]
    npos, nneg = int(yy.sum()), int((~yy).sum())
    if npos == 0 or nneg == 0:
        return np.nan, 0
    r = pd.Series(s).rank(method="average").values
    auc = (r[yy].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return auc, npos * nneg


def within_group_auc(scores, y, groups):
    """Pool per-group AUC by pair count. y: True=success. Returns AUC of P(score|success > score|failure)."""
    num = 0.0; den = 0
    for g in np.unique(groups):
        m = groups == g
        a, p = _auc_pairs(scores[m], y[m])
        if p > 0 and np.isfinite(a):
            num += a * p; den += p
    return (num / den, den) if den else (np.nan, 0)


# ---------------------------------------------------------------- residualisation
def _avg_rank(x):
    """Average ranks (ties -> mean rank), numpy-only, ~10x faster than pandas here."""
    n = len(x)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    r = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        r[o[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return r


def rank_residualise(scores, base, groups):
    """Within each group: rank-transform `scores` and `base`, OLS-regress rank(scores)
    on rank(base), return the residual. Answers 'what does this statistic carry that the
    baseline does not already have'. Label-free. NaN in either -> NaN residual."""
    out = np.full(len(scores), np.nan, np.float64)
    for g in np.unique(groups):
        m = np.where((groups == g) & np.isfinite(scores) & np.isfinite(base))[0]
        if len(m) < 3:
            continue
        rs = _avg_rank(scores[m]); rb = _avg_rank(base[m])
        rs -= rs.mean(); rb -= rb.mean()
        den = (rb * rb).sum()
        beta = (rs * rb).sum() / den if den > 1e-12 else 0.0
        out[m] = rs - beta * rb
    return out
