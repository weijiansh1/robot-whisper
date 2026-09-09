"""Phase 1 (LABEL-BLIND): exhaustive NONPARAMETRIC pair scan, y ~ f(x).

The linear scan can only see straight relations.  This scan bins x into 256
equal-count bins and uses E[y | bin] as f, which upper-bounds what any function
of x alone can explain.  All 384 x 383 ordered pairs are scanned, so the search
is exhaustive rather than cherry-picked, and the identical scan is run on two
surrogates so the reported floor is matched to the same search.

  rel_resid_nl(y|x) = sqrt(1 - eta^2(y | bins of x))

No outcome label anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_definitional import classify, family  # noqa: E402

BUNDLE = HERE.parent
BANK = BUNDLE / "results" / "bank"
OUT = BUNDLE / "results"
NBINS = 256
RNG = np.random.default_rng(20260906)


def scan(Z: np.ndarray, nbins: int = NBINS) -> np.ndarray:
    """eta2[j, i] = variance of standardised column i explained by bins of j.

    The bin sums are formed as a one-hot GEMM rather than a gather-reduceat;
    that is 14x faster here and agrees to float32 rounding."""
    n, V = Z.shape
    edges = np.linspace(0, n, nbins + 1).astype(np.int64)
    cnt = np.diff(edges).astype(np.float64)
    binid = np.repeat(np.arange(nbins), np.diff(edges))
    Y = np.ascontiguousarray(Z, dtype=np.float32)
    Y -= Y.mean(axis=0)
    sd = Y.std(axis=0)
    Y /= np.where(sd > 0, sd, 1.0)
    onehot = np.zeros((n, nbins), dtype=np.float32)
    eta2 = np.empty((V, V), dtype=np.float64)
    for j in range(V):
        order = np.argsort(Y[:, j], kind="stable")
        onehot[:] = 0.0
        onehot[order, binid] = 1.0
        sums = (onehot.T @ Y).astype(np.float64)
        means = sums / cnt[:, None]
        eta2[j] = (cnt[:, None] * means ** 2).sum(axis=0) / n
    return np.clip(eta2, 0.0, 1.0)


def permute_within(Z: np.ndarray, g: np.ndarray, rng) -> np.ndarray:
    """Independent permutation of every column inside every group of `g`."""
    n, V = Z.shape
    base = np.argsort(g, kind="stable")
    gf = g.astype(np.float64)
    out = np.empty_like(Z)
    for v in range(V):
        ordv = np.argsort(gf + rng.random(n), kind="stable")
        out[base, v] = Z[ordv, v]
    return out


def main() -> None:
    f = np.load(BANK / "development_main_bank.npz", allow_pickle=True)
    names = [str(s) for s in f["var_names"]]
    V = len(names)
    X = f["X"].astype(np.float64)
    Z = (X - X.mean(0)) / X.std(0)
    task = f["row_task"].astype(np.int64)
    chunk = f["row_chunk"].astype(np.int64)
    epi = f["row_query"].astype(np.int64)
    cell = task * (int(chunk.max()) + 1) + chunk
    _, cell = np.unique(cell, return_inverse=True)

    fams = [family(n) for n in names]
    fid = np.array([{k: i for i, k in enumerate(sorted(set(fams)))}[x] for x in fams])
    same_fam = fid[:, None] == fid[None, :]

    print("observed scan...", file=sys.stderr)
    eta = scan(Z)
    print("null_cell scan...", file=sys.stderr)
    eta_nc = scan(permute_within(Z, cell, RNG))
    print("null_epi scan...", file=sys.stderr)
    eta_ne = scan(permute_within(Z, epi, RNG))

    rr = np.sqrt(np.clip(1 - eta, 0, 1))        # rr[j, i]: predict i from j
    rr_nc = np.sqrt(np.clip(1 - eta_nc, 0, 1))
    rr_ne = np.sqrt(np.clip(1 - eta_ne, 0, 1))
    off = ~np.eye(V, dtype=bool)
    nondef = off & ~same_fam

    order = np.argsort(np.where(nondef, rr, np.inf), axis=None)
    rows = []
    for o in order[:6000]:
        j, i = np.unravel_index(o, rr.shape)
        if not nondef[j, i]:
            continue
        rows.append({
            "target": names[i], "predictor": names[j],
            "rel_resid_nl": float(rr[j, i]),
            "null_cell": float(rr_nc[j, i]), "null_epi": float(rr_ne[j, i]),
            "category": classify([names[i], names[j]]),
        })
    print("external replication...", file=sys.stderr)
    g = np.load(BANK / "external_8b_bank.npz", allow_pickle=True)
    assert [str(s) for s in g["var_names"]] == names
    Xe = g["X"].astype(np.float64)
    eta_ext = scan((Xe - Xe.mean(0)) / Xe.std(0))
    rr_ext = np.sqrt(np.clip(1 - eta_ext, 0, 1))
    for r in rows:
        i, j = names.index(r["target"]), names.index(r["predictor"])
        r["rel_resid_nl_external_8b_refit"] = float(rr_ext[j, i])

    res = {
        "n_bins": NBINS, "n_vars": V, "n_rows": int(Z.shape[0]),
        "floor_all_pairs": {"N_cell": float(rr_nc[off].min()),
                            "N_epi": float(rr_ne[off].min())},
        "floor_nondefinitional_pairs": {"N_cell": float(rr_nc[nondef].min()),
                                        "N_epi": float(rr_ne[nondef].min())},
        "observed_percentiles_nondefinitional":
            {f"p{q}": float(np.percentile(rr[nondef], q))
             for q in (0.01, 0.1, 1, 5, 25, 50)},
        "best_definitional_pair": best_of(rr, off & same_fam, names),
        "resolution_floor": {
            "note": "flow_path = 9 * mean flow_speed is EXACT in the extractor "
                    "(phase1_controls C3, rel_resid 2.4e-4).  Whatever this "
                    "scan reports for that pair is the binning resolution "
                    "floor: nothing below it is resolvable.",
            "pair": ["sp.flow_speed@L2[mean]", "lg.flow_path@L2[agg]"],
            "rel_resid_nl": float(rr[names.index("lg.flow_path@L2[agg]"),
                                     names.index("sp.flow_speed@L2[mean]")]),
        },
        "top_nondefinitional": [r for r in rows
                                if r["category"].startswith("empirical")][:400],
    }
    (OUT / "phase1_nonlinear.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("floor_all_pairs", "floor_nondefinitional_pairs",
                       "observed_percentiles_nondefinitional")}, indent=1))
    for r in res["top_nondefinitional"][:10]:
        print("  %.4f (null %.3f/%.3f)  %s ~ f(%s)  [%s]"
              % (r["rel_resid_nl"], r["null_cell"], r["null_epi"],
                 r["target"], r["predictor"], r["category"]))


def best_of(rr, mask, names):
    m = np.where(mask, rr, np.inf)
    o = int(np.argmin(m))
    j, i = np.unravel_index(o, rr.shape)
    return {"target": names[i], "predictor": names[j],
            "rel_resid_nl": float(rr[j, i])}


if __name__ == "__main__":
    main()
