"""Re-check the circuit bundle's suite-stratified survival AUC under task stratification.
Reads moe-circuit-analogy-0906 cached features read-only."""
import numpy as np, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recheck_stratification import (CAPS, CHUNKS, WIDTH, to_series, trailing_mean,
                                    stratified_auc, ROOT, OUT)

z = np.load(f'{ROOT}/moe-circuit-analogy-0906/results/circuit_features/external_8b.npz',
            allow_pickle=True)
rows_j = list(csv.DictReader(open(f'{ROOT}/moe-failure-modes-0906/results/joined_external_8b.csv')))
L  = np.array([int(r['length']) for r in rows_j])
R  = np.array([r['risk'] == 'True' for r in rows_j])
SU = np.array([r['suite'] for r in rows_j])
TK = np.array([r['task_key'] for r in rows_j])
assert np.array_equal(L, z['length'].astype(int))
assert np.array_equal(TK, z['task_names'][z['task_index'].astype(int)])

feats  = z['feature_names']; layers = list(z['layer_names']); valid = z['valid']
F = z['features']            # (248255, 8, 31)
n_ep, n_chunk = valid.shape
WANT = ['norm_fiedler', 'gap_ratio', 'spectral_erank', 'kirchhoff_efficiency',
        'res_cv', 'vol', 'fiedler', 'log_kirchhoff', 'log_spanning_trees',
        'ground_leak_mean', 'th_mean', 'res_mean', 'dd_deficit', 'th_cv', 'lam_max']
out = []
for name in WANT:
    if name not in list(feats):
        print('  (absent)', name, file=sys.stderr); continue
    fi = list(feats).index(name)
    ser = trailing_mean(to_series(np.asarray(F[:, :, fi]), valid, n_ep, n_chunk), WIDTH)
    for li, lay in enumerate(layers):
        for c in CHUNKS:
            if c >= n_chunk: continue
            alive = L > c
            if alive.sum() < 50: continue
            x = ser[:, c, li]
            ap, _ = stratified_auc(x, R, None, alive)
            asu, _ = stratified_auc(x, R, SU, alive)
            at, _ = stratified_auc(x, R, TK, alive)
            out.append(dict(cohort='external_8b', quantity=name, layer=lay, chunk=c,
                            alive=int(alive.sum()), alive_risk=int((alive & R).sum()),
                            auc_pooled=ap, auc_within_suite=asu, auc_within_task=at))
with open(f'{OUT}/recheck_circuit_auc.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('wrote', len(out), 'rows', file=sys.stderr)
