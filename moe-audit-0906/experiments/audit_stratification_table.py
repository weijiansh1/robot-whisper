"""Canonical stratification audit table.
Reproduces moe-circuit-analogy-0906's survival-conditioned AUC exactly (raw values,
strata=(group,chunk), n+>=20 & n->=20, Mann-Whitney n+*n- weights), then re-runs the
identical metric with group=task instead of group=suite."""
import numpy as np, csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recheck_stratification import to_series, auc, ROOT, OUT
MIN = 20

rows_j = list(csv.DictReader(open(f'{ROOT}/moe-failure-modes-0906/results/joined_external_8b.csv')))
L  = np.array([int(r['length']) for r in rows_j]); R = np.array([r['risk'] == 'True' for r in rows_j])
SU = np.array([r['suite'] for r in rows_j]);       TK = np.array([r['task_key'] for r in rows_j])
zc = np.load(f'{ROOT}/moe-circuit-analogy-0906/results/circuit_features/external_8b.npz', allow_pickle=True)
zt = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/external_8b_index.npz', allow_pickle=True)
feats = list(zc['feature_names']); F = zc['features']; valid = zc['valid']; layers = list(zc['layer_names'])
n_ep, n_chunk = valid.shape

series = {n: to_series(np.asarray(F[:, :, feats.index(n)]), valid, n_ep, n_chunk) for n in feats}
Kk = np.load(f'{ROOT}/moe-token-geometry-0906/cache/kernels/external_8b_conditional55.npy', mmap_mode='r')
ur, uc = zt['upper_row'].astype(int), zt['upper_col'].astype(int)
Kd = np.asarray(Kk); ds = Kd[:, :, ur == uc].sum(2); os_ = Kd[:, :, ur != uc].sum(2)
series['conditional_energy'] = to_series(ds / 10.0, valid, n_ep, n_chunk)
series['centred_config_size'] = to_series(np.sqrt(np.maximum(ds - (ds + 2*os_)/10.0, 0)), valid, n_ep, n_chunk)

def metric(x, groups, li):
    num = den = 0.0; k = 0
    for g in np.unique(groups):
        gm = groups == g
        for q in range(n_chunk):
            m = gm & (L > q)
            if m.sum() == 0: continue
            v = x[:, q, li]; p = v[m & R]; n = v[m & ~R]
            p = p[np.isfinite(p)]; n = n[np.isfinite(n)]
            if len(p) < MIN or len(n) < MIN: continue
            a, w = auc(p, n)
            if np.isfinite(a): num += a*w; den += w; k += 1
    return (num/den if den else np.nan), k

out = []
for name, x in series.items():
    for li, lay in enumerate(layers):
        s, ks = metric(x, SU, li); t, kt = metric(x, TK, li)
        if not (np.isfinite(s) and np.isfinite(t)): continue
        out.append(dict(quantity=name, layer=lay,
                        auc_suite_stratified=round(s, 4), dev_suite=round(abs(s-.5), 4), n_suite_strata=ks,
                        auc_task_stratified=round(t, 4), dev_task=round(abs(t-.5), 4), n_task_strata=kt,
                        shrinkage=round(1 - abs(t-.5)/abs(s-.5), 3) if abs(s-.5) > 1e-9 else None,
                        sign_flip=bool((s-.5)*(t-.5) < 0)))
out.sort(key=lambda r: -r['dev_suite'])
with open(f'{OUT}/audit_stratification_survival_auc.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('wrote', len(out), 'rows', file=sys.stderr)
top = [r for r in out if r['dev_suite'] >= 0.04]
print(f"cells with suite-stratified |AUC-0.5| >= 0.04 : {len(top)}", file=sys.stderr)
print(f"  of those, task-stratified |AUC-0.5| >= 0.04 : {sum(1 for r in top if r['dev_task']>=0.04)}", file=sys.stderr)
print(f"  median shrinkage : {np.median([r['shrinkage'] for r in top]):.1%}", file=sys.stderr)
print(f"  sign flips       : {sum(1 for r in top if r['sign_flip'])}", file=sys.stderr)
