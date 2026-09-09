"""Compute the three-method consensus-core statistics from three assignment CSVs."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

ak_p, ce_p, al_p = sys.argv[1], sys.argv[2], sys.argv[3]
ak = pd.read_csv(ak_p); ce = pd.read_csv(ce_p); al = pd.read_csv(al_p)
key = ["task", "episode"]
ak["fail"] = ak.outcome != "success"
m = (ak[key + ["fail", "raw_cluster"]]
     .merge(ce[key + ["event_cluster"]], on=key)
     .merge(al[key + ["lag_spectrum_louvain"]], on=key))
assert len(m) == 2560, len(m)
f = m.fail.values
F = f.sum()

# Blocks are identified by outcome-blind *content*, not by label index:
# pick the block of each partition with the highest failure count (the "core" block).
def pick_core(series):
    counts = {}
    for v in series.unique():
        sel = (series == v).values
        counts[v] = (sel & f).sum()
    best = max(counts, key=counts.get)
    return best, (series == best).values

ak_lab, A = pick_core(m.raw_cluster)
ce_lab, B = pick_core(m.event_cluster)
al_lab, C = pick_core(m.lag_spectrum_louvain)

def stat(sel):
    n = int(sel.sum()); nf = int((sel & f).sum())
    return dict(n=n, success=n - nf, failure=nf,
                precision=round(nf / max(n, 1), 4),
                failure_recall=round(nf / F, 4))

votes = A.astype(int) + B.astype(int) + C.astype(int)
def jac(x, y): return round(float((x & y).sum() / (x | y).sum()), 4)

miss = f & (votes == 0)
out = dict(
    core_labels=dict(raw=str(ak_lab), event=str(ce_lab), lag=str(al_lab)),
    raw_core=stat(A), event_core=stat(B), lag_core=stat(C),
    vote_ge2=stat(votes >= 2), intersection=stat(votes == 3), union=stat(votes >= 1),
    jaccard=dict(raw_event=jac(A, B), raw_lag=jac(A, C), event_lag=jac(B, C)),
    union_missed_failures=int(miss.sum()),
    union_missed_by_task={k: int(v) for k, v in m.loc[miss, "task"].value_counts().items()},
    partition_sizes=dict(
        raw={str(k): int(v) for k, v in m.raw_cluster.value_counts().sort_index().items()},
        event={str(k): int(v) for k, v in m.event_cluster.value_counts().sort_index().items()},
        lag_n_communities=int(m.lag_spectrum_louvain.nunique()),
    ),
)
# member id sets for cross-seed Jaccard
ids = (m.task + "|" + m.episode.astype(str)).values
out["_members"] = dict(raw=sorted(ids[A].tolist()), event=sorted(ids[B].tolist()),
                       lag=sorted(ids[C].tolist()), ge2=sorted(ids[votes >= 2].tolist()))
out["_labels"] = dict(raw=m.raw_cluster.astype(str).tolist(),
                      event=m.event_cluster.astype(str).tolist(),
                      lag=m.lag_spectrum_louvain.astype(str).tolist(),
                      ids=ids.tolist())
print(json.dumps(out))
