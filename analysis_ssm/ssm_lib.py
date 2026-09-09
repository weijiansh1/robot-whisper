"""Linear-Gaussian state-space (Kalman) model of HiMoE-VLA MoE routing.

Emission  y_k = clr(R_k)        centred log-ratio of the 32-simplex at each (layer, token) site
Latent    z_{k+1} = F z_k + B u_k + b + w_k ,   u_k = flatten(executed 10x7 action chunk)
Observed  x_k = W z_k + v_k     x = whitened PCA scores of y (basis fit on TRAIN folds only)

Quantity of interest: the innovation e_k = x_k - W z_{k|k-1} and its Mahalanobis form (NIS).

Nothing here touches labels: PCA, EM and the filter are all label-free. Labels enter only
in the AUC evaluation. Model parameters are fit leave-one-group-out; a branch never
contributes to the model that scores it. The filter is strictly forward (causal): the score
at query t depends only on chunks 0..t of that branch.
"""
import os

import numpy as np
import pandas as pd
import torch

ROOT = "/home/jovyan/work/himoe-vla"
CACHE = f"{ROOT}/analysis_ssm/cache"
OUT = f"{ROOT}/analysis_ssm"
TMP = "/tmp"
EPS = 1e-12
SEED = 20260829

DEEP = [4, 5, 6, 7]          # HB layers 12,13,14,15 along the stored 8-layer axis
ACT_TOK = list(range(1, 11))  # action tokens 1-10


# ------------------------------------------------------------------ data
def load(tag, sites="all88"):
    """Dense per-branch routing probs, actions, validity mask, meta.

    P     (Bn, Tmax, S, 32) float32 routing probabilities
    A     (Bn, Tmax, 10, 7) float32 action chunks
    valid (Bn, Tmax) bool
    """
    full = np.load(f"{CACHE}/{tag}_full.npy", mmap_mode="r")   # (N, 8, 11, 32)
    rowidx = np.load(f"{CACHE}/{tag}_rowidx.npy")
    act = np.load(f"{CACHE}/{tag}_act.npy")
    meta = pd.read_csv(f"{CACHE}/{tag}_meta.csv")
    meta["success"] = meta["success"].astype(bool)
    Bn, Tmax = rowidx.shape
    valid = rowidx >= 0
    if sites == "all88":
        lay, tok = list(range(8)), list(range(11))
    elif sites == "back40":
        lay, tok = DEEP, ACT_TOK
    else:
        raise ValueError(sites)
    S = len(lay) * len(tok)
    P = np.zeros((Bn, Tmax, S, 32), np.float32)
    for b in range(Bn):
        r = rowidx[b][valid[b]]
        blk = np.asarray(full[r])                      # (T, 8, 11, 32)
        P[b, :len(r)] = blk[:, lay][:, :, tok].reshape(len(r), S, 32)
    return P, act, valid, meta


def clr(P, valid):
    """Centred log-ratio per (layer, token) site; flattened. (Bn,T,S,32)->(Bn,T,S*32) f32."""
    Bn, T, S, E = P.shape
    L = np.log(np.clip(P, 1e-7, None), dtype=np.float32)
    L -= L.mean(-1, keepdims=True)
    Y = L.reshape(Bn, T, S * E)
    Y[~valid] = 0.0
    return Y


def hellinger_prev(P, valid):
    """Protocol baseline series: Hellinger(chunk t, chunk t-1) per site, mean over sites."""
    Bn, T, S, E = P.shape
    d = np.full((Bn, T), np.nan, np.float32)
    for t in range(1, T):
        ok = valid[:, t]
        if ok.any():
            bc = np.sqrt(P[ok, t] * P[ok, t - 1]).sum(-1)
            d[ok, t] = np.sqrt(np.clip(1.0 - bc, 0.0, None)).mean(-1)
    return d


def shuffle_within_branch(P, A, valid, seed=SEED):
    """Acausal control: permute the chunk order inside each branch (routing+action together)."""
    rng = np.random.default_rng(seed)
    P2, A2 = P.copy(), A.copy()
    for b in range(P.shape[0]):
        T = int(valid[b].sum())
        pi = rng.permutation(T)
        P2[b, :T] = P[b, pi]
        A2[b, :T] = A[b, pi]
    return P2, A2


# ------------------------------------------------------------------ PCA (label-free, train-only)
def fit_pca(Xtr, r, dev="cpu"):
    """Exact PCA by eigendecomposition of the train covariance. Returns mu, V (D,r), sd (r,).

    Gram on CPU (fast BLAS), eigendecomposition on `dev` (GPU is ~4x faster at D=2816).
    """
    X = torch.as_tensor(Xtr)
    mu = X.mean(0)
    X = X - mu
    C = (X.T @ X).double() / (X.shape[0] - 1)
    try:
        w, V = torch.linalg.eigh(C.to(dev))
        w, V = w.cpu(), V.cpu()
    except Exception:
        w, V = torch.linalg.eigh(C)
    V = V[:, -r:].flip(1).float()                      # descending eigenvalue order
    lam = w[-r:].flip(0).clamp_min(1e-10).float()
    return mu.numpy(), V.numpy(), lam.sqrt().numpy()


def apply_pca(Y, mu, V, sd):
    """(Bn,T,D) -> whitened scores (Bn,T,r) float64 numpy."""
    Bn, T, D = Y.shape
    Yt = torch.as_tensor(Y.reshape(-1, D))
    Z = (Yt - torch.as_tensor(mu)) @ torch.as_tensor(V)
    Z = Z / torch.as_tensor(sd)
    return Z.reshape(Bn, T, -1).double().numpy()


# ------------------------------------------------------------------ LGSSM: filter / smoother / EM
class Params:
    __slots__ = ("F", "Bm", "b", "Q", "W", "R", "m0", "P0")

    def __init__(self, F, Bm, b, Q, W, R, m0, P0):
        self.F, self.Bm, self.b, self.Q = F, Bm, b, Q
        self.W, self.R, self.m0, self.P0 = W, R, m0, P0


def init_params(r, d, m, dev):
    W = torch.zeros(r, d, dtype=torch.float64, device=dev)
    W[:d, :d] = torch.eye(d, dtype=torch.float64, device=dev)
    return Params(F=0.9 * torch.eye(d, dtype=torch.float64, device=dev),
                  Bm=torch.zeros(d, m, dtype=torch.float64, device=dev),
                  b=torch.zeros(d, dtype=torch.float64, device=dev),
                  Q=0.1 * torch.eye(d, dtype=torch.float64, device=dev),
                  W=W,
                  R=torch.eye(r, dtype=torch.float64, device=dev),
                  m0=torch.zeros(d, dtype=torch.float64, device=dev),
                  P0=torch.eye(d, dtype=torch.float64, device=dev))


def cov_seq(p, T, jitter=1e-9):
    """Kalman covariance recursion. Data-independent, hence SHARED across branches.

    Every branch is valid from chunk 0 up to its own T_b, so for any chunk k the
    predicted covariance of every still-alive branch is identical; branches that have
    ended are masked out of everything downstream, so one recursion suffices.
    """
    d, r = p.F.shape[0], p.W.shape[0]
    dev = p.F.device
    Ir = torch.eye(r, dtype=torch.float64, device=dev)
    Id = torch.eye(d, dtype=torch.float64, device=dev)
    Pp = torch.zeros(T, d, d, dtype=torch.float64, device=dev)
    Pf = torch.zeros(T, d, d, dtype=torch.float64, device=dev)
    Kg = torch.zeros(T, d, r, dtype=torch.float64, device=dev)
    Lc = torch.zeros(T, r, r, dtype=torch.float64, device=dev)
    ld = torch.zeros(T, dtype=torch.float64, device=dev)
    P = p.P0.clone()
    Wt = p.W.T
    for k in range(T):
        S = p.W @ P @ Wt + p.R
        S = 0.5 * (S + S.T) + jitter * Ir
        Lk = torch.linalg.cholesky(S)
        Lc[k] = Lk
        ld[k] = 2.0 * torch.log(torch.diagonal(Lk)).sum()
        Pp[k] = P
        K = torch.cholesky_solve((P @ Wt).T, Lk).T                 # (d,r)
        Kg[k] = K
        Pk = P - K @ (p.W @ P)
        Pf[k] = 0.5 * (Pk + Pk.T)
        P = p.F @ Pf[k] @ p.F.T + p.Q
        P = 0.5 * (P + P.T) + jitter * Id
    return dict(Pp=Pp, Pf=Pf, Kg=Kg, Lc=Lc, logdet=ld)


def kalman_filter(X, U, M, p, want_seq=True, cov=None):
    """Batched forward filter (means batched, covariance shared).

    X (n,T,r), U (n,T,m), M (n,T) bool. Returns causal per-step quantities:
    innov (n,T,r), nis (n,T), nll (n,T), zf/zp (n,T,d) and the shared covariance seq.
    """
    n, T, r = X.shape
    d = p.F.shape[0]
    dev = X.device
    if cov is None:
        cov = cov_seq(p, T)
    innov = torch.zeros(n, T, r, dtype=torch.float64, device=dev)
    nis = torch.zeros(n, T, dtype=torch.float64, device=dev)
    zf = torch.zeros(n, T, d, dtype=torch.float64, device=dev)
    zp = torch.zeros(n, T, d, dtype=torch.float64, device=dev)
    z = p.m0.expand(n, d).clone()
    Wt = p.W.T
    for k in range(T):
        mk = M[:, k].unsqueeze(-1)
        e = X[:, k] - z @ Wt                                       # (n,r)
        alpha = torch.cholesky_solve(e.T, cov["Lc"][k]).T          # (n,r) = S^{-1} e
        nis[:, k] = (e * alpha).sum(-1)
        zp[:, k] = z
        zk = z + e @ cov["Kg"][k].T
        zk = torch.where(mk, zk, z)
        zf[:, k] = zk
        innov[:, k] = torch.where(mk, e, torch.zeros_like(e))
        z = zk @ p.F.T + U[:, k] @ p.Bm.T + p.b
    Mf = M.double()
    nis = nis * Mf
    nll = 0.5 * (nis + (cov["logdet"] + r * np.log(2 * np.pi)).unsqueeze(0)) * Mf
    return dict(innov=innov, nis=nis, nll=nll, zf=zf, zp=zp, cov=cov)


def rts_smooth(zf, zp, p, cov, jitter=1e-9):
    """RTS smoother. Means batched; smoothed covariances are shared across branches."""
    n, T, d = zf.shape
    dev = zf.device
    Id = torch.eye(d, dtype=torch.float64, device=dev)
    zs = torch.zeros_like(zf)
    Ps = torch.zeros(T, d, d, dtype=torch.float64, device=dev)
    Pc = torch.zeros(T, d, d, dtype=torch.float64, device=dev)
    J = torch.zeros(T, d, d, dtype=torch.float64, device=dev)
    zs[:, T - 1] = zf[:, T - 1]
    Ps[T - 1] = cov["Pf"][T - 1]
    for k in range(T - 2, -1, -1):
        Pp1 = cov["Pp"][k + 1] + jitter * Id
        Jk = torch.linalg.solve(Pp1, p.F @ cov["Pf"][k]).T          # (d,d)
        J[k] = Jk
        zs[:, k] = zf[:, k] + (zs[:, k + 1] - zp[:, k + 1]) @ Jk.T
        Ps[k] = cov["Pf"][k] + Jk @ (Ps[k + 1] - Pp1) @ Jk.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
        Pc[k + 1] = Ps[k + 1] @ Jk.T
    return zs, Ps, Pc


def em_fit(X, U, M, r, d, n_iter=25, dev="cpu", seed=SEED, verbose=False, batch=None,
           tol=1e-6):
    """EM for the LGSSM with control input. X (n,T,r), U (n,T,m), M (n,T) bool."""
    n, T, _ = X.shape
    m = U.shape[2]
    p = init_params(r, d, m, dev)
    Id = torch.eye(d, dtype=torch.float64, device=dev)
    Ir = torch.eye(r, dtype=torch.float64, device=dev)
    G = d + m + 1
    Mf = M.double()
    Mp = (M[:, :-1] & M[:, 1:]).double()                            # (n,T-1)
    Nk = Mf.sum(0)                                                  # (T,)
    Npk = Mp.sum(0)                                                 # (T-1,)
    Ntot, Np = float(Mf.sum()), float(Mp.sum())
    Xf = X.reshape(-1, r)
    Xw = (X * Mf.unsqueeze(-1)).reshape(-1, r)
    Sxx = Xw.T @ Xf
    Uc = U[:, :-1]
    lls = []
    for it in range(n_iter):
        f = kalman_filter(X, U, M, p, want_seq=True)
        cov = f["cov"]
        lls.append(float(-f["nll"].sum()))
        zs, Ps, Pc = rts_smooth(f["zf"], f["zp"], p, cov)
        zsw = zs * Mf.unsqueeze(-1)
        Szz = (Nk.view(T, 1, 1) * Ps).sum(0) + zsw.reshape(-1, d).T @ zs.reshape(-1, d)
        Sxz = Xw.T @ zs.reshape(-1, d)
        p.W = torch.linalg.solve(Szz + 1e-8 * Id, Sxz.T).T
        Rn = (Sxx - p.W @ Sxz.T - Sxz @ p.W.T + p.W @ Szz @ p.W.T) / Ntot
        p.R = 0.5 * (Rn + Rn.T) + 1e-6 * Ir
        g = torch.cat([zs[:, :-1], Uc,
                       torch.ones(n, T - 1, 1, dtype=torch.float64, device=dev)], -1)
        gf = g.reshape(-1, G)
        gw = (g * Mp.unsqueeze(-1)).reshape(-1, G)
        Egg = gw.T @ gf
        Egg[:d, :d] += (Npk.view(T - 1, 1, 1) * Ps[:-1]).sum(0)
        z1 = zs[:, 1:].reshape(-1, d)
        Ezg = ((zs[:, 1:] * Mp.unsqueeze(-1)).reshape(-1, d)).T @ gf
        Ezg[:, :d] += (Npk.view(T - 1, 1, 1) * Pc[1:]).sum(0)
        A = torch.linalg.solve(Egg + 1e-8 * torch.eye(G, dtype=torch.float64, device=dev),
                               Ezg.T).T
        p.F = A[:, :d].contiguous()
        p.Bm = A[:, d:d + m].contiguous()
        p.b = A[:, d + m].contiguous()
        Ez1z1 = (Npk.view(T - 1, 1, 1) * Ps[1:]).sum(0) + \
            ((zs[:, 1:] * Mp.unsqueeze(-1)).reshape(-1, d)).T @ z1
        Qn = (Ez1z1 - A @ Ezg.T - Ezg @ A.T + A @ Egg @ A.T) / Np
        p.Q = 0.5 * (Qn + Qn.T) + 1e-6 * Id
        p.m0 = zs[:, 0].mean(0)
        dz = zs[:, 0] - p.m0
        p.P0 = Ps[0] + (dz.T @ dz) / n + 1e-6 * Id
        if verbose:
            print(f"  EM {it:2d} ll={lls[-1]:.1f}", flush=True)
        if it > 4 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-2]):
            break
    return p, lls


def filter_batched(X, U, M, p, batch=None):
    return kalman_filter(X, U, M, p, want_seq=False)


# ------------------------------------------------------------------ evaluation
def _avg_rank(x):
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


def within_group_auc(scores, y, groups):
    num, den = 0.0, 0
    for g in np.unique(groups):
        msk = (groups == g) & np.isfinite(scores)
        s, yy = scores[msk], y[msk]
        npos, nneg = int(yy.sum()), int((~yy).sum())
        if npos == 0 or nneg == 0:
            continue
        r = _avg_rank(s)
        auc = (r[yy].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
        num += auc * npos * nneg
        den += npos * nneg
    return (num / den, den) if den else (np.nan, 0)


def rank_residualise(scores, base, groups):
    out = np.full(len(scores), np.nan, np.float64)
    for g in np.unique(groups):
        msk = np.where((groups == g) & np.isfinite(scores) & np.isfinite(base))[0]
        if len(msk) < 3:
            continue
        rs = _avg_rank(scores[msk]); rb = _avg_rank(base[msk])
        rs -= rs.mean(); rb -= rb.mean()
        den = (rb * rb).sum()
        beta = (rs * rb).sum() / den if den > 1e-12 else 0.0
        res = rs - beta * rb
        res[np.abs(res) < 1e-8 * len(msk)] = 0.0
        out[msk] = res
    return out


def within_group_spearman(a, b, groups):
    """Equal-weighted mean and per-group values of Spearman rho within group."""
    vals = []
    for g in np.unique(groups):
        msk = (groups == g) & np.isfinite(a) & np.isfinite(b)
        if msk.sum() < 5:
            continue
        ra, rb = _avg_rank(a[msk]), _avg_rank(b[msk])
        ra = ra - ra.mean(); rb = rb - rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        vals.append(float((ra * rb).sum() / den) if den > 0 else np.nan)
    vals = np.array(vals, float)
    return float(np.nanmean(vals)), vals


def perm_p(scores, y, groups, obs_det, ndraw=200, seed=SEED):
    """Within-group branch-level label permutation; p on the det AUC."""
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(ndraw):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            yp[idx] = y[idx][rng.permutation(len(idx))]
        a, _ = within_group_auc(scores, yp, groups)
        if np.isfinite(a) and max(a, 1 - a) >= obs_det - 1e-12:
            ge += 1
    return (ge + 1) / (ndraw + 1)


# ------------------------------------------------------------------ folds
def logo_folds(meta):
    """Leave-one-group-out: fold id == group."""
    return meta["group"].values.copy()


# ------------------------------------------------------------------ persistence
_ROWS = []
KEYCOLS = ("corpus", "sites", "r", "d", "t", "variant", "stat", "metric")


def emit(**row):
    _ROWS.append(row)


def flush(name="moe_ssm.csv"):
    if not _ROWS:
        return
    recs = []
    for r in _ROWS:
        keys = {k: r.get(k, "") for k in KEYCOLS}
        for k, v in r.items():
            if k in KEYCOLS:
                continue
            recs.append({**keys, "name": k, "value": v})
    df = pd.DataFrame(recs)
    for dd in (TMP, OUT):
        p = os.path.join(dd, name)
        hdr = not os.path.exists(p)
        df.to_csv(p, mode="a", header=hdr, index=False)
    _ROWS.clear()


def note(text, name="moe_ssm.md"):
    for dd in (TMP, OUT):
        with open(os.path.join(dd, name), "a") as fh:
            fh.write(text.rstrip() + "\n")
    print(text, flush=True)
