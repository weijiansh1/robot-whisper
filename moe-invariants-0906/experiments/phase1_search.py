"""Phase 1 (LABEL-BLIND): systematic pair and triple search for near-identities.

NO outcome label is loaded anywhere in this file or in anything it imports.
Ranking is by tightness only.  The output is frozen and hashed before phase 2.

Two passes:
  free   - unrestricted; used to check that the known identity tops the ranking
  nondef - canonical duplicates of the target or of each other are banned, so
           the ranking is over relations the definitions do not already force
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_definitional import canonical, classify, family  # noqa: E402
from phase1_lib import (  # noqa: E402
    combo_std, cov_decomposition, fit_std_coefs, null_corr, oos_rel_resid,
    to_corr, triple_best,
)

BUNDLE = HERE.parent
BANK = BUNDLE / "results" / "bank"
OUT = BUNDLE / "results"
COHORTS = ("development_main", "development_extra", "external_8b")
DUP_CUT = 0.9999
COLLINEAR_CUT = 0.999
TOP_PER_TARGET = 12
RNG = np.random.default_rng(20260906)


def load_bank(cohort: str) -> dict:
    f = np.load(BANK / f"{cohort}_bank.npz", allow_pickle=True)
    return {k: f[k] for k in f.files}


def main() -> None:
    dev = load_bank("development_main")
    names = [str(s) for s in dev["var_names"]]
    V = len(names)
    canon = [canonical(n) for n in names]
    canon_id = {c: i for i, c in enumerate(sorted(set(canon)))}
    cid = np.array([canon_id[c] for c in canon])
    same_canon = cid[:, None] == cid[None, :]
    fams = [family(n) for n in names]
    fam_id = {f: i for i, f in enumerate(sorted(set(fams)))}
    fid = np.array([fam_id[f] for f in fams])
    same_fam = fid[:, None] == fid[None, :]

    X = dev["X"].astype(np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    assert (sd > 0).all()
    Z = (X - mu) / sd

    task = dev["row_task"].astype(np.int64)
    chunk = dev["row_chunk"].astype(np.int64)
    epi = dev["row_query"].astype(np.int64)
    cell = task * (int(chunk.max()) + 1) + chunk
    _, cell = np.unique(cell, return_inverse=True)

    dec_cell = cov_decomposition(Z, cell)
    dec_epi = cov_decomposition(Z, epi)
    R_obs = to_corr(dec_cell["tot"])
    R_Ncell = null_corr(dec_cell["tot"], dec_cell["bet"])
    R_Nepi = null_corr(dec_epi["tot"], dec_epi["bet"])
    R_wcell = to_corr(dec_cell["wit"])
    R_wepi = to_corr(dec_epi["wit"])
    perm_check = shuffle_check(Z, cell, R_Ncell)

    dup = np.abs(R_obs) > DUP_CUT
    np.fill_diagonal(dup, False)

    # ---------------- pairs -------------------------------------------------
    rel_obs = np.sqrt(np.clip(1 - R_obs ** 2, 0, 1))
    rel_nc = np.sqrt(np.clip(1 - R_Ncell ** 2, 0, 1))
    rel_ne = np.sqrt(np.clip(1 - R_Nepi ** 2, 0, 1))
    iu = np.triu_indices(V, 1)
    order = np.argsort(rel_obs[iu])
    all_pairs = []
    for o in order[:20000]:
        i, j = int(iu[0][o]), int(iu[1][o])
        cat = classify([names[i], names[j]])
        all_pairs.append({
            "kind": "pair", "target": names[i], "preds": [names[j]],
            "rel_resid": float(rel_obs[i, j]),
            "null_cell": float(rel_nc[i, j]), "null_epi": float(rel_ne[i, j]),
            "rel_resid_within_cell": float(np.sqrt(max(0, 1 - R_wcell[i, j] ** 2))),
            "rel_resid_within_epi": float(np.sqrt(max(0, 1 - R_wepi[i, j] ** 2))),
            "r": float(R_obs[i, j]), "category": cat,
        })
    pair_floor = {"N_cell": float(rel_nc[iu].min()),
                  "N_epi": float(rel_ne[iu].min())}
    # the same floor computed only over pairs the definitions do not force
    ndu = (~same_fam)[iu]
    pair_floor_nondef = {"N_cell": float(rel_nc[iu][ndu].min()),
                         "N_epi": float(rel_ne[iu][ndu].min())}

    # ---------------- triples ----------------------------------------------
    triples_free, triples_nd = triple_pass(
        R_obs, R_Ncell, R_Nepi, names, same_fam)

    # ---------------- enrichment -------------------------------------------
    idx_of = {n: i for i, n in enumerate(names)}
    Zs = {}
    for c in COHORTS:
        b = load_bank(c)
        assert [str(s) for s in b["var_names"]] == names
        Zs[c] = (b["X"].astype(np.float64) - mu) / sd

    def enrich(rec):
        t = idx_of[rec["target"]]
        pr = [idx_of[p] for p in rec["preds"]]
        coefs = fit_std_coefs(R_obs, t, pr)
        rec["coefs_std_dev_main"] = [float(c) for c in coefs]
        rec["coefs_raw_dev_main"] = [float(coefs[k] * sd[t] / sd[p])
                                     for k, p in enumerate(pr)]
        rec["combo_std"] = combo_std(R_obs, [t] + pr)[0]
        rec["combo_std_null_cell"] = combo_std(R_Ncell, [t] + pr)[0]
        rec["combo_std_null_epi"] = combo_std(R_Nepi, [t] + pr)[0]
        for c in COHORTS:
            rec[f"oos_{c}"] = oos_rel_resid(Zs[c], t, pr, coefs)
        return rec

    def top_of(rows, cat_prefix, n):
        sel = [r for r in rows if r["category"].startswith(cat_prefix)]
        return sel[:n]

    headline = (top_of(all_pairs, "empirical:cross_metric", 150)
                + top_of(all_pairs, "empirical:same_metric", 150)
                + top_of(triples_nd, "empirical:cross_metric", 150)
                + top_of(triples_nd, "empirical:same_metric", 150)
                + all_pairs[:60] + triples_free[:60])
    seen = set()
    for rec in headline:
        key = (rec["target"], tuple(rec["preds"]))
        if key in seen:
            continue
        seen.add(key)
        enrich(rec)

    pos = positive_control(R_obs, R_Ncell, R_Nepi, idx_of, Zs, sd, triples_free)
    neg = negative_control(R_obs, V, same_fam)

    out = {
        "n_vars": V,
        "n_rows_dev_main": int(Z.shape[0]),
        "n_canonical_vars": len(canon_id),
        "n_family_vars": len(fam_id),
        "pair_null_floor": pair_floor,
        "pair_null_floor_nondefinitional": pair_floor_nondef,
        "shuffle_check": perm_check,
        "positive_control": pos,
        "negative_control": neg,
        "pairs_free": all_pairs[:800],
        "pairs_empirical_cross_metric": top_of(all_pairs, "empirical:cross_metric", 800),
        "pairs_empirical_same_metric": top_of(all_pairs, "empirical:same_metric", 800),
        "triples_free": triples_free[:400],
        "triples_empirical_cross_metric": top_of(triples_nd, "empirical:cross_metric", 800),
        "triples_empirical_same_metric": top_of(triples_nd, "empirical:same_metric", 800),
        "category_counts_pairs": count(all_pairs),
        "category_counts_triples_nondef": count(triples_nd),
    }
    (OUT / "phase1_relations.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "phase1_corr.npz", R_obs=R_obs.astype(np.float32),
                        R_Ncell=R_Ncell.astype(np.float32),
                        R_Nepi=R_Nepi.astype(np.float32),
                        R_wcell=R_wcell.astype(np.float32),
                        R_wepi=R_wepi.astype(np.float32),
                        var_names=np.array(names), mu=mu, sd=sd)
    print("pair floor", pair_floor, file=sys.stderr)
    for k in ("pairs_empirical_cross_metric", "triples_empirical_cross_metric"):
        r = out[k][0]
        print(k, "%.4f" % r["rel_resid"], r["target"], r["preds"], file=sys.stderr)


def count(rows):
    c = {}
    for r in rows:
        c[r["category"]] = c.get(r["category"], 0) + 1
    return c


def triple_pass(R_obs, R_Ncell, R_Nepi, names, same_fam):
    V = R_obs.shape[0]
    free, nd = {}, {}
    for t in range(V):
        rr, _ = triple_best(R_obs, t, collinear_cut=COLLINEAR_CUT)
        rnc, _ = triple_best(R_Ncell, t, collinear_cut=COLLINEAR_CUT)
        rne, _ = triple_best(R_Nepi, t, collinear_cut=COLLINEAR_CUT)
        base = np.where(np.isfinite(rr), rr, np.inf)
        base = np.triu(base, 1) + np.tril(np.full_like(base, np.inf))
        for store, mat in (("free", base), ("nd", None)):
            if mat is None:
                banned = same_fam.copy()
                banned |= same_fam[t][:, None] | same_fam[t][None, :]
                mat = np.where(banned, np.inf, base)
            tgt = free if store == "free" else nd
            k = TOP_PER_TARGET
            flat = mat.ravel()
            idx = np.argpartition(flat, k)[:k] if flat.size > k else np.arange(flat.size)
            for o in idx[np.argsort(flat[idx])]:
                j, kk = np.unravel_index(o, mat.shape)
                if not np.isfinite(mat[j, kk]):
                    continue
                key = (t, *sorted((int(j), int(kk))))
                if key in tgt:
                    continue
                tgt[key] = {
                    "kind": "triple", "target": names[t],
                    "preds": [names[j], names[kk]],
                    "rel_resid": float(rr[j, kk]),
                    "null_cell": float(rnc[j, kk]), "null_epi": float(rne[j, kk]),
                    "r_pred_pred": float(R_obs[j, kk]),
                    "category": classify([names[t], names[j], names[kk]]),
                }
    return (sorted(free.values(), key=lambda d: d["rel_resid"]),
            sorted(nd.values(), key=lambda d: d["rel_resid"]))


def shuffle_check(Z, cell, R_null, n_rep: int = 2) -> dict:
    order = np.argsort(cell, kind="stable")
    bounds = np.searchsorted(cell[order], np.arange(int(cell.max()) + 2))
    probe = RNG.choice(Z.shape[1], size=24, replace=False)
    devs = []
    for _ in range(n_rep):
        Zs = Z[:, probe][order].copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            for v in range(len(probe)):
                Zs[a:b, v] = Zs[a + RNG.permutation(b - a), v]
        Zr = np.empty_like(Zs)
        Zr[order] = Zs
        Rr = to_corr(np.cov(Zr, rowvar=False, bias=True))
        devs.append(float(np.abs(Rr - R_null[np.ix_(probe, probe)]).max()))
    return {"probe_vars": int(len(probe)),
            "max_abs_dev_closed_form_vs_permutation": devs}


def positive_control(R_obs, R_Ncell, R_Nepi, idx_of, Zs, sd, triples_free):
    recs = []
    for ly in ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15"):
        for red in ("mean", "s0", "s9"):
            keys = [f"sp.{m}@{ly}[{red}]" for m in
                    ("load_entropy", "token_entropy", "token_differentiation")]
            if any(k not in idx_of for k in keys):
                continue
            t, p1, p2 = (idx_of[k] for k in keys)
            rr, _ = triple_best(R_obs, t, collinear_cut=1.1)
            rnc, _ = triple_best(R_Ncell, t, collinear_cut=1.1)
            rne, _ = triple_best(R_Nepi, t, collinear_cut=1.1)
            coefs = fit_std_coefs(R_obs, t, [p1, p2])
            recs.append({
                "layer": ly, "reduction": red,
                "rel_resid": float(rr[p1, p2]),
                "null_cell": float(rnc[p1, p2]), "null_epi": float(rne[p1, p2]),
                "raw_coefs": [float(coefs[0] * sd[t] / sd[p1]),
                              float(coefs[1] * sd[t] / sd[p2])],
                "oos_external_8b": oos_rel_resid(Zs["external_8b"], t, [p1, p2], coefs),
                "oos_development_extra": oos_rel_resid(Zs["development_extra"], t,
                                                       [p1, p2], coefs),
            })
    ranks = [k + 1 for k, r in enumerate(triples_free)
             if r["category"] == "definitional:entropy_identity"]
    return {
        "per_layer_reduction": recs,
        "worst_rel_resid": max(r["rel_resid"] for r in recs),
        "worst_oos_external": max(r["oos_external_8b"] for r in recs),
        "raw_coef_min": min(min(r["raw_coefs"]) for r in recs),
        "raw_coef_max": max(max(r["raw_coefs"]) for r in recs),
        "best_rank_in_free_triple_search": min(ranks) if ranks else None,
        "n_in_top100_of_free_triple_search": sum(1 for r in ranks if r <= 100),
    }


def negative_control(R_obs, V, same_fam, n: int = 40000):
    i = RNG.integers(0, V, n)
    j = RNG.integers(0, V, n)
    ok = (i != j) & ~same_fam[i, j]
    i, j = i[ok], j[ok]
    rp = np.sqrt(np.clip(1 - R_obs[i, j] ** 2, 0, 1))
    tri = []
    for a, b, c in zip(i[:6000], j[:6000], RNG.integers(0, V, 6000)):
        if len({a, b, c}) < 3 or same_fam[a, c] or same_fam[b, c]:
            continue
        S = R_obs[np.ix_([a, b, c], [a, b, c])]
        r2 = S[0, 1:] @ np.linalg.solve(S[1:, 1:] + 1e-9 * np.eye(2), S[0, 1:])
        tri.append(np.sqrt(max(0.0, 1 - min(r2, 1.0))))
    tri = np.array(tri)
    q = [1, 5, 25, 50]
    return {
        "note": "random pairs/triples of functionals in DISTINCT definitional families",
        "pair": {"n": int(len(rp)), "min": float(rp.min()),
                 "median": float(np.median(rp)),
                 **{f"p{k}": float(np.percentile(rp, k)) for k in q}},
        "triple": {"n": int(len(tri)), "min": float(tri.min()),
                   "median": float(np.median(tri)),
                   **{f"p{k}": float(np.percentile(tri, k)) for k in q}},
    }


if __name__ == "__main__":
    main()
