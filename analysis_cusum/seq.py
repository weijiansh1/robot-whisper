"""Sequential (SPRT / CUSUM) detector core.

Everything here is strictly causal: at index t the statistic uses only
score increments at indices <= t.  All model parameters (time profile,
AR coefficient, class-conditional densities, alarm bound) are fitted
leave-one-group-out; a branch never contributes to the model that scores it.

The decision rule is ABSOLUTE: the bound is a single number applied to every
branch, and the time profile / densities are fixed functions learnt on the
training folds.  No test-group statistics are ever used.
"""
import numpy as np

TMAX = 35          # detection window is t in [ALO, TMAX)
ALO = 8            # earliest allowed alarm index
LLR_CLIP = 6.0     # per-step log-likelihood-ratio clip (both directions)
SD_FLOOR = 1e-9


# ------------------------------------------------------------------ smoothing
def smooth(S, W):
    """Trailing mean of width W over the per-step score.  W == 1 -> raw."""
    if W == 1:
        return S.copy()
    B, T = S.shape
    M = np.full_like(S, np.nan)
    C = np.nan_to_num(S, nan=0.0)
    ok = np.isfinite(S).astype(np.float64)
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(C, 1)], 1)
    co = np.concatenate([np.zeros((B, 1)), np.cumsum(ok, 1)], 1)
    for t in range(W - 1, T):
        n = co[:, t + 1] - co[:, t + 1 - W]
        s = cs[:, t + 1] - cs[:, t + 1 - W]
        f = n == W
        M[f, t] = s[f] / W
    return M


# ------------------------------------------------------------------ densities
def _gauss_ll(x, mu, sd):
    return -0.5 * np.log(2 * np.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2


class GaussDens:
    """Location-scale Gaussian per class (captures shift AND scale)."""

    def __init__(self, x, y):
        self.p = {}
        for c, m in (("f", y), ("s", ~y)):
            v = x[m]
            self.p[c] = (v.mean(), max(v.std(), 1e-6))

    def llr(self, x):
        mf, sf = self.p["f"]
        ms, ss = self.p["s"]
        return _gauss_ll(x, mf, sf) - _gauss_ll(x, ms, ss)


class KdeDens:
    """Gaussian KDE per class on a shared grid, linear interpolation."""

    def __init__(self, x, y, ngrid=257, pad=1.5):
        lo, hi = np.percentile(x, [0.2, 99.8])
        rng = hi - lo
        self.g = np.linspace(lo - pad * rng, hi + pad * rng, ngrid)
        self.ll = {}
        for c, m in (("f", y), ("s", ~y)):
            v = x[m]
            n = len(v)
            bw = 1.06 * max(v.std(), 1e-6) * n ** (-0.2)
            d = np.exp(-0.5 * ((self.g[:, None] - v[None, :]) / bw) ** 2).sum(1)
            d = d / (n * bw * np.sqrt(2 * np.pi))
            # floor at a small fraction of the peak so tails stay finite
            d = np.maximum(d, 1e-4 * d.max())
            self.ll[c] = np.log(d)

    def llr(self, x):
        a = np.interp(x, self.g, self.ll["f"], left=self.ll["f"][0], right=self.ll["f"][-1])
        b = np.interp(x, self.g, self.ll["s"], left=self.ll["s"][0], right=self.ll["s"][-1])
        return a - b


DENS = {"gauss": GaussDens, "kde": KdeDens}


def _ksmooth(v, ok, bw):
    """Gaussian kernel smooth of a per-t curve over the valid entries."""
    t = np.arange(len(v), dtype=float)
    K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / bw) ** 2) * ok[None, :]
    num = K @ np.nan_to_num(v * ok)
    den = K.sum(1)
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


class TvdDens:
    """Time-VARYING location-scale Gaussian per class.

    p_t(z | class) with mu and sd estimated per index t on the training folds
    and kernel-smoothed across t.  This is the properly specified LLR: at
    indices where the two classes coincide the per-step evidence is ~0
    automatically, instead of being forced through one time-homogeneous
    density.  t is observable at run time so this stays strictly causal.
    """

    def __init__(self, x, y, tt, T, bw=3.0, sd_floor=0.15):
        self.mu = {}
        self.sd = {}
        for c, m in (("f", y), ("s", ~y)):
            mu = np.full(T, np.nan)
            sd = np.full(T, np.nan)
            ok = np.zeros(T)
            for t in range(T):
                v = x[m & (tt == t)]
                if len(v) >= 8:
                    mu[t], sd[t], ok[t] = v.mean(), v.std(), 1.0
            gm, gs = x[m].mean(), max(x[m].std(), 1e-6)
            if ok.sum() == 0:
                mu[:], sd[:], ok[:] = gm, gs, 1.0
            self.mu[c] = np.nan_to_num(_ksmooth(mu, ok, bw), nan=gm)
            s = np.nan_to_num(_ksmooth(sd, ok, bw), nan=gs)
            self.sd[c] = np.maximum(s, sd_floor * gs)

    def llr(self, x, tt):
        a = _gauss_ll(x, self.mu["f"][tt], self.sd["f"][tt])
        b = _gauss_ll(x, self.mu["s"][tt], self.sd["s"][tt])
        return a - b


# ------------------------------------------------------------------ pipeline
def build_llr(S, meta, cfg):
    """Return llr (B, Tmax) with NaN where undefined, fitted leave-one-group-out.

    cfg keys: W (int), tstd (bool), whiten ('none'|'ar1'), dens ('gauss'|'kde').
    """
    W = cfg["W"]
    M = smooth(S, W)
    B, T = M.shape
    y = ~meta["success"].values.astype(bool)          # True == eventual failure
    grp = meta["group"].values
    Tb = meta["T"].values
    t0 = max(W, 1)

    # mask of usable (b, t): score defined and t < branch length and t < TMAX
    use = np.isfinite(M)
    for b in range(B):
        use[b, Tb[b]:] = False
    use[:, :max(t0, cfg.get("tstart", 1))] = False
    use[:, TMAX:] = False
    t0 = max(t0, cfg.get("tstart", 1))

    llr = np.full((B, T), np.nan)
    aux = {}
    for g in np.unique(grp):
        tr = grp != g
        te = grp == g
        # ---- 1. time profile (train folds, both classes pooled) -> absolute
        Z = np.full((B, T), np.nan)
        if cfg["tstd"]:
            mu = np.full(T, np.nan)
            sd = np.full(T, np.nan)
            for t in range(t0, TMAX):
                v = M[tr & use[:, t], t]
                if len(v) >= 10:
                    mu[t], sd[t] = v.mean(), max(v.std(), SD_FLOOR)
            # fill any gap with the pooled train value
            vp = M[np.ix_(tr, np.arange(t0, TMAX))][use[np.ix_(tr, np.arange(t0, TMAX))]]
            mup, sdp = vp.mean(), max(vp.std(), SD_FLOOR)
            bad = ~np.isfinite(mu)
            mu[bad], sd[bad] = mup, sdp
            for t in range(t0, TMAX):
                m = use[:, t]
                Z[m, t] = (M[m, t] - mu[t]) / sd[t]
            aux["mu"], aux["sd"] = mu, sd
        else:
            vp = M[np.ix_(tr, np.arange(t0, TMAX))][use[np.ix_(tr, np.arange(t0, TMAX))]]
            mup, sdp = vp.mean(), max(vp.std(), SD_FLOOR)
            Z[use] = (M[use] - mup) / sdp

        # ---- 2. whitening (AR(1) fitted on TRAIN SUCCESS branches only)
        if cfg["whiten"] == "ar1":
            num = den = 0.0
            for b in np.where(tr & ~y)[0]:
                z = Z[b, t0:min(TMAX, Tb[b])]
                z = z[np.isfinite(z)]
                if len(z) > 3:
                    num += (z[:-1] * z[1:]).sum()
                    den += (z[:-1] ** 2).sum()
            phi = float(np.clip(num / max(den, 1e-9), -0.95, 0.95))
            aux["phi"] = phi
            E = np.full((B, T), np.nan)
            sc = np.sqrt(max(1 - phi * phi, 1e-6))
            for b in range(B):
                idx = np.where(use[b])[0]
                if len(idx) == 0:
                    continue
                z = Z[b, idx]
                e = np.empty_like(z)
                e[0] = z[0]
                e[1:] = (z[1:] - phi * z[:-1]) / sc
                E[b, idx] = e
        else:
            aux["phi"] = 0.0
            E = Z

        # ---- 3. class-conditional densities on train folds
        trm = np.zeros((B, T), bool)
        trm[tr] = use[tr]
        x = E[trm]
        yy = np.repeat(y[:, None], T, 1)[trm]
        tem = np.zeros((B, T), bool)
        tem[te] = use[te]
        tt_tr = np.repeat(np.arange(T)[None, :], B, 0)[trm]
        tt_te = np.repeat(np.arange(T)[None, :], B, 0)[tem]

        # ---- 4. per-step LLR for the held-out group
        if cfg["dens"] == "tvd":
            d = TvdDens(x, yy, tt_tr, T, bw=cfg.get("tvd_bw", 3.0))
            v = d.llr(E[tem], tt_te)
        else:
            d = DENS[cfg["dens"]](x, yy)
            v = d.llr(E[tem])
        llr[tem] = np.clip(v, -LLR_CLIP, LLR_CLIP)
    return llr, use, aux


def accumulate(llr, use, rule):
    """Lambda (B, T).  rule 'sprt' = plain cumulative sum, 'cusum' = reset at 0."""
    B, T = llr.shape
    L = np.full((B, T), np.nan)
    for b in range(B):
        idx = np.where(use[b])[0]
        acc = 0.0
        for t in idx:
            acc = acc + llr[b, t]
            if rule == "cusum":
                acc = max(0.0, acc)
            L[b, t] = acc
    return L


def alarm_times(L, Tb, h, lo=ALO, hi=TMAX):
    """First index t in [lo, hi) with L[b, t] >= h.  -1 if none."""
    B = L.shape[0]
    out = np.full(B, -1)
    for b in range(B):
        top = min(hi, Tb[b])
        seg = L[b, lo:top]
        ok = np.isfinite(seg) & (seg >= h)
        w = np.flatnonzero(ok)
        if len(w):
            out[b] = lo + w[0]
    return out


def running_peak(L, Tb, sel=None, lo=ALO, hi=TMAX):
    """max over the detection window of the causal statistic, per branch."""
    B = L.shape[0]
    idx = np.arange(B) if sel is None else np.where(sel)[0]
    peak = np.full(len(idx), -np.inf)
    for j, b in enumerate(idx):
        seg = L[b, lo:min(hi, Tb[b])]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            peak[j] = seg.max()
    return peak


def calibrate_bound(L, Tb, sel, alpha, lo=ALO, hi=TMAX):
    """Smallest bound h whose alarm rate among branches in `sel` is <= alpha.
    `sel` should be the TRAIN-fold success branches."""
    peak = running_peak(L, Tb, sel, lo, hi)
    if len(peak) == 0:
        return np.inf
    peak = np.sort(peak)[::-1]
    k = int(np.floor(alpha * len(peak)))
    if k >= len(peak):
        return -np.inf
    return float(np.nextafter(peak[k], np.inf))


def logo_alarms(L, meta, alpha, lo=ALO, hi=TMAX):
    """Leave-one-group-out bound calibration -> alarm index per branch.

    Returns (alarm, bounds dict).  For each held-out group the bound is chosen
    on the OTHER groups' success branches to hit nominal FA = alpha, then
    applied unchanged to the held-out branches (absolute rule, no group
    centring).
    """
    grp = meta["group"].values
    su = meta["success"].values.astype(bool)
    Tb = meta["T"].values
    alarm = np.full(len(meta), -1)
    bounds = {}
    for g in np.unique(grp):
        tr = (grp != g) & su
        h = calibrate_bound(L, Tb, tr, alpha, lo, hi)
        bounds[int(g)] = h
        a = alarm_times(L, Tb, h, lo, hi)
        te = grp == g
        alarm[te] = a[te]
    return alarm, bounds


# --------------------------------------------------------- fixed-threshold rule
def fixed_statistic(S, W=8, K=3):
    """The K-consecutive-below rule written as a crossing rule.

    An alarm at index t requires the W-window mean to be < thr at each of
    t-K+1 .. t, i.e. max(M[t-K+1..t]) < thr.  So with L = -max(M[t-K+1..t])
    the rule is exactly "L >= -thr", the same shape as the sequential rule,
    which lets both be calibrated by the identical bound search.
    """
    M = smooth(S, W)
    B, T = M.shape
    R = np.full((B, T), np.nan)
    for t in range(K - 1, T):
        blk = M[:, t - K + 1:t + 1]
        ok = np.isfinite(blk).all(1)
        R[ok, t] = blk[ok].max(1)
    return -R


def fixed_alarms(S, Tb, thr, K=3, W=8, lo=ALO, hi=TMAX):
    return alarm_times(fixed_statistic(S, W, K), Tb, -thr, lo, hi)
