"""Wire-equivalent route recurrence: top-4 expert ids + weights (what the recorder server returns on the wire),
computed offline from the topology captures, to calibrate an online trigger without new GPU servers."""
import json, glob, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

CAP = Path("/home/swj/data/moe-capture/topo-20260916")
GRID = CAP / "_grid/episodes"
OUT = Path("/home/swj/data/libero-runtime/samples/moe-recurrence-20260916")
W, WT, ALPHA = 14, 3, 0.10


def sparse_points(ids, probs):
    """ids/probs (8,10,11,4) -> back_path scope (layers 4..7, all 10 flows, action tokens 1..10) as sqrt-prob vectors."""
    ids = ids[4:, :, 1:, :]; probs = probs[4:, :, 1:, :].astype(np.float64)
    probs = probs / np.maximum(probs.sum(-1, keepdims=True), 1e-12)   # renormalise the top-4 mass (wire has no full distribution)
    dense = np.zeros(ids.shape[:-1] + (32,))
    np.put_along_axis(dense, ids.astype(np.int64), probs, axis=-1)
    cells = int(np.prod(dense.shape[:-1]))
    return np.sqrt(dense).reshape(-1) / np.sqrt(2 * cells)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3: return np.nan
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def rr_window(D, q, eps):
    S = D[q - W + 1:q + 1, q - W + 1:q + 1]; J = np.arange(W)
    m = (J[None, :] - J[:, None]) >= WT
    return float((S[m] <= eps).mean()), float(S.max())


eps_meta = json.loads((GRID.parent / "episodes.json").read_text())
episodes = []
for m in eps_meta:
    tag, ds = m["tag"], m["dataset"]
    server_dirs = sorted(CAP.glob("server-p95*/" + tag))
    files = sorted(glob.glob(str(server_dirs[0] / "q*.npz"))) if server_dirs else []
    if len(files) != m["n_queries"]:
        print("skip", tag, len(files), m["n_queries"]); continue
    pts = []
    for f in files:
        z = np.load(f)
        pts.append(sparse_points(z["hb_expert_ids"], z["hb_selected_prob"]))
    P = np.stack(pts)
    D = np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1), 0.0))
    episodes.append(dict(tag=tag, ds=ds, success=bool(m["success"]), Q=len(files), D=D, base=int(tag[4:6])))
print("episodes", len(episodes))
vals = []
for e in episodes:
    S = e["D"][:W, :W]; J = np.arange(len(S)); vals.append(S[(J[None, :] - J[:, None]) >= WT])
eps = float(np.quantile(np.concatenate(vals), ALPHA))
print("global eps (wire route, back_path, alpha %.2f) = %.6f" % (ALPHA, eps))
succ = np.array([e["success"] for e in episodes]); tasks = np.array([e["base"] for e in episodes])
RR = np.full((len(episodes), 52), np.nan); DM = np.full((len(episodes), 52), np.nan)
for i, e in enumerate(episodes):
    for q in range(W - 1, e["Q"]):
        RR[i, q], DM[i, q] = rr_window(e["D"], q, eps)
for a in (16, 20, 24):
    ok = np.array([e["Q"] > a for e in episodes])
    ta = [auroc(RR[ok & ~succ & (tasks == t), a], RR[ok & succ & (tasks == t), a]) for t in range(10)]
    print("q%d: RR AUROC failure>success pooled %.3f task-mean %.3f (n %d/%d); diameter (oriented) %.3f" % (
        a, auroc(RR[ok & ~succ, a], RR[ok & succ, a]), np.nanmean(ta), (ok & ~succ).sum(), (ok & succ).sum(), 1 - auroc(DM[ok & ~succ, a], DM[ok & succ, a])))
print("trigger calibration (RR >= theta for k consecutive queries, q >= 13):")
for theta in (0.3, 0.4, 0.5, 0.6, 0.7):
    for k in (2, 3):
        first = np.full(len(episodes), -1)
        for i, e in enumerate(episodes):
            run = 0
            for q in range(13, e["Q"]):
                run = run + 1 if RR[i, q] >= theta else 0
                if run >= k: first[i] = q; break
        tp = ((first >= 0) & ~succ).sum(); fp = ((first >= 0) & succ).sum()
        print("  theta %.1f k=%d: failures %d/84 (median first q %s), successes %d/86" % (theta, k, tp, np.median(first[(first >= 0) & ~succ]) if tp else -1, fp))
np.savez_compressed(OUT / "wire-route-rr.npz", RR=RR, DM=DM, success=succ, tasks=tasks, eps=eps, tags=np.array([e["tag"] for e in episodes]), ds=np.array([e["ds"] for e in episodes]))
json.dump(dict(eps=eps, alpha=ALPHA, window=W, theiler=WT, scope="route back_path from wire top-4"), open(OUT / "wire-route-calibration.json", "w"), indent=2)
