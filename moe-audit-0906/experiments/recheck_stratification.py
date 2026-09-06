"""Audit re-check: survival-conditioned AUC under pooled / within-suite / within-task
stratification, recomputed from the cached conditional kernels of
moe-token-geometry-0906 (read-only) with labels from moe-failure-modes-0906.

Nothing outside moe-audit-0906/ is written."""
import numpy as np, csv, json, os, sys

ROOT = '/home/jovyan/work/himoe-vla'
OUT  = os.path.join(ROOT, 'moe-audit-0906', 'results')
os.makedirs(OUT, exist_ok=True)
CAPS = {'libero_goal': 30, 'libero_long': 52, 'libero_object': 28, 'libero_spatial': 22}
CHUNKS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
WIDTH = 4          # trailing-mean width, frozen protocol
MIN_STRATUM = 30   # matches moe-token-geometry-0906/results/effect/effect.json

def load(cohort, joined):
    z = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/{cohort}_index.npz',
                allow_pickle=True)
    rows = list(csv.DictReader(open(joined)))
    L  = np.array([int(r['length']) for r in rows])
    R  = np.array([r['risk'] == 'True' for r in rows])
    SU = np.array([r['suite'] for r in rows])
    TK = np.array([r['task_key'] for r in rows])
    assert np.array_equal(L, z['length'].astype(int)), 'length misalignment'
    assert np.array_equal(TK, z['task_names'][z['task_index'].astype(int)]), 'task misalignment'
    K = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/{cohort}_conditional55.npy',
                mmap_mode='r')
    return z, K, L, R, SU, TK

def quantities(z, K):
    """conditional_energy = mean(diag C);  size2 = tr(J C J) (centred configuration size^2)."""
    ur, uc = z['upper_row'].astype(int), z['upper_col'].astype(int)
    dg, off = ur == uc, ur != uc
    Kd = np.asarray(K)                              # (nq, 8, 55)
    diag_sum = Kd[:, :, dg].sum(axis=2)             # tr(C)
    off_sum  = Kd[:, :, off].sum(axis=2)
    total    = diag_sum + 2.0 * off_sum             # 1' C 1
    ce   = diag_sum / 10.0                          # conditional_energy
    size2 = diag_sum - total / 10.0                 # tr(J C J)
    return {'conditional_energy': ce,
            'size': np.sqrt(np.maximum(size2, 0.0)),
            'common_component': total / 100.0}      # mean of all entries of C

def to_series(vals, valid, n_ep, n_chunk):
    """scatter (nq, 8) -> (n_ep, n_chunk, 8) with NaN padding"""
    out = np.full((n_ep, n_chunk, vals.shape[1]), np.nan, np.float64)
    out[valid] = vals
    return out

def trailing_mean(x, w):
    """causal trailing mean of width w over axis 1; NaN until w observations exist"""
    n_ep, n_chunk, n_l = x.shape
    out = np.full_like(x, np.nan)
    cs = np.nancumsum(np.nan_to_num(x), axis=1)
    ct = np.cumsum(~np.isnan(x), axis=1)
    for q in range(n_chunk):
        lo = q - w
        s = cs[:, q] - (cs[:, lo] if lo >= 0 else 0)
        c = ct[:, q] - (ct[:, lo] if lo >= 0 else 0)
        ok = c == w
        out[:, q][ok] = (s / np.maximum(c, 1))[ok]
    return out

def auc(pos, neg):
    """Mann-Whitney AUC with tie correction"""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return np.nan, 0
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0), n1 * n0

def stratified_auc(vals, risk, strata, alive):
    """weighted pooled AUC over strata; strata=None -> single pooled stratum"""
    num = den = 0.0
    keys = [None] if strata is None else np.unique(strata[alive])
    for k in keys:
        m = alive & (np.ones_like(alive) if k is None else (strata == k))
        m = m & np.isfinite(vals)
        p, n = vals[m & risk], vals[m & ~risk]
        if len(p) < 1 or len(n) < 1: continue
        if strata is not None and (len(p) + len(n)) < MIN_STRATUM: continue
        a, w = auc(p, n)
        if np.isfinite(a): num += a * w; den += w
    return (num / den if den else np.nan), den

def run(cohort, joined):
    z, K, L, R, SU, TK = load(cohort, joined)
    valid = z['valid']
    n_ep, n_chunk = valid.shape
    qs = quantities(z, K)
    layers = list(z['layer_names'])
    atcap = np.array([L[i] == CAPS[SU[i]] for i in range(n_ep)])
    rows = []
    for qname, v in qs.items():
        ser = trailing_mean(to_series(v, valid, n_ep, n_chunk), WIDTH)
        for li, lay in enumerate(layers):
            for c in CHUNKS:
                if c >= n_chunk: continue
                alive = L > c
                if alive.sum() < 50: continue
                x = ser[:, c, li]
                a_p, _ = stratified_auc(x, R, None, alive)
                a_s, _ = stratified_auc(x, R, SU, alive)
                a_t, _ = stratified_auc(x, R, TK, alive)
                rows.append(dict(cohort=cohort, quantity=qname, layer=lay, chunk=c,
                                 alive=int(alive.sum()), alive_risk=int((alive & R).sum()),
                                 prior=float((alive & R).sum() / alive.sum()),
                                 auc_pooled=a_p, auc_within_suite=a_s, auc_within_task=a_t))
    return rows, dict(n_episodes=n_ep, n_risk=int(R.sum()), n_atcap=int(atcap.sum()),
                      risk_subset_of_atcap=bool((R & ~atcap).sum() == 0),
                      atcap_not_risk=int((atcap & ~R).sum()))

if __name__ == '__main__':
    all_rows, meta = [], {}
    for cohort, joined in [('external_8b', f'{ROOT}/moe-failure-modes-0906/results/joined_external_8b.csv'),
                           ('development_main', f'{ROOT}/moe-failure-modes-0906/results/joined_development_main.csv')]:
        r, m = run(cohort, joined)
        all_rows += r; meta[cohort] = m
        print(cohort, m, file=sys.stderr)
    with open(f'{OUT}/recheck_survival_auc.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    json.dump(meta, open(f'{OUT}/recheck_meta.json', 'w'), indent=1)
    print('wrote', len(all_rows), 'rows', file=sys.stderr)
