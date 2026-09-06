"""Replicate moe-circuit-analogy-0906's survival-conditioned AUC metric exactly
(strata = (group, chunk); require n+>=20 and n->=20; Mann-Whitney n+*n- weights),
then swap group = suite -> group = task and report the difference."""
import numpy as np, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recheck_stratification import CAPS, WIDTH, to_series, trailing_mean, auc, ROOT, OUT

MIN_POS = MIN_NEG = 20

def series_for(cohort, joined):
    rows_j = list(csv.DictReader(open(joined)))
    L  = np.array([int(r['length']) for r in rows_j])
    R  = np.array([r['risk'] == 'True' for r in rows_j])
    SU = np.array([r['suite'] for r in rows_j]); TK = np.array([r['task_key'] for r in rows_j])
    zc = np.load(f'{ROOT}/moe-circuit-analogy-0906/results/circuit_features/{cohort}.npz', allow_pickle=True)
    zt = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/{cohort}_index.npz', allow_pickle=True)
    assert np.array_equal(L, zc['length'].astype(int))
    valid = zc['valid']; n_ep, n_chunk = valid.shape
    out = {}
    feats = list(zc['feature_names']); F = zc['features']
    for name in ['norm_fiedler','gap_ratio','spectral_erank','kirchhoff_efficiency','res_cv','vol','fiedler','ground_leak_mean']:
        out[name] = trailing_mean(to_series(np.asarray(F[:, :, feats.index(name)]), valid, n_ep, n_chunk), WIDTH)
    # conditional_energy + centred configuration size from the token-geometry kernels
    Kk = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/{cohort}_conditional55.npy', mmap_mode='r')
    ur, uc = zt['upper_row'].astype(int), zt['upper_col'].astype(int)
    Kd = np.asarray(Kk); ds = Kd[:, :, ur == uc].sum(2); os_ = Kd[:, :, ur != uc].sum(2)
    out['conditional_energy'] = trailing_mean(to_series(ds / 10.0, valid, n_ep, n_chunk), WIDTH)
    out['size'] = trailing_mean(to_series(np.sqrt(np.maximum(ds - (ds + 2*os_)/10.0, 0)), valid, n_ep, n_chunk), WIDTH)
    return out, L, R, SU, TK, list(zc['layer_names']), n_chunk

def metric(x_all, L, R, groups, n_chunk, li):
    num = den = 0.0; nstrata = 0
    for g in np.unique(groups):
        gm = groups == g
        for q in range(n_chunk):
            m = gm & (L > q)
            if m.sum() == 0: continue
            v = x_all[:, q, li]
            pos = v[m & R]; neg = v[m & ~R]
            pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
            if len(pos) < MIN_POS or len(neg) < MIN_NEG: continue
            a, w = auc(pos, neg)
            if np.isfinite(a): num += a*w; den += w; nstrata += 1
    return (num/den if den else np.nan), nstrata, den

out_rows = []
for cohort, joined in [('external_8b', f'{ROOT}/moe-failure-modes-0906/results/joined_external_8b.csv'),
                       ('development_main', f'{ROOT}/moe-failure-modes-0906/results/joined_development_main.csv')]:
    S, L, R, SU, TK, layers, n_chunk = series_for(cohort, joined)
    for name, x in S.items():
        for li, lay in enumerate(layers):
            a_s, ns, ws = metric(x, L, R, SU, n_chunk, li)
            a_t, nt, wt = metric(x, L, R, TK, n_chunk, li)
            out_rows.append(dict(cohort=cohort, quantity=name, layer=lay,
                                 auc_suite_strat=a_s, n_suite_strata=ns, pairs_suite=ws,
                                 auc_task_strat=a_t, n_task_strata=nt, pairs_task=wt,
                                 dev_suite=abs(a_s-.5) if np.isfinite(a_s) else np.nan,
                                 dev_task=abs(a_t-.5) if np.isfinite(a_t) else np.nan))
with open(f'{OUT}/recheck_circuit_metric_suite_vs_task.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
print('wrote', len(out_rows), file=sys.stderr)
