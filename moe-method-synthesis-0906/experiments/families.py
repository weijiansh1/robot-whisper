"""Cluster the saved detectors by *which episodes they fire on*, not by name.

The prior belief under test is that the detector names overstate diversity.
The test is purely structural: two detectors belong to the same family if their
in-window alarm sets agree beyond what their firing rates alone would produce.

Three similarities are reported, because they answer different questions and a
single number would hide the answer:

  jaccard          |A n B| / |A u B|.  Identity of *operating points*.  A
                   detector at quantile 0.5 and the same detector at 0.99 have
                   low Jaccard purely because the sets differ in size, so this
                   measures threshold granularity, not method diversity.
  excess_jaccard   jaccard minus its analytic rate-matched expectation.
  excess_overlap   |A n B| / min(|A|,|B|) minus its rate-matched expectation
                   max(|A|,|B|)/n.  Nested sets score 1, so a threshold ladder
                   over one score collapses to one family; a detector that
                   fires on almost everything gets no credit for containing the
                   others.  This is the *method*-level similarity and the one
                   the headline family count uses.

Every arm carries a null: rate-matched random alarm vectors with the per-suite
firing rate of each detector preserved.  A family a random vector of the same
rate would also join is not a family.

Families are derived on *development* for the 419 detectors present in both
cohorts (held-out capable) and, separately and in-sample, on external for all
622, because four bundles published external vectors only and can never be
selected honestly.  Everything is also reported per suite.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

import synth_core as sc

CUTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
HEADLINE_CUT = 0.5
HEADLINE_SIM = "excess_overlap"
SEED = 20260906


def quantity_of(detector: str) -> str:
    """The named quantity a detector claims to measure, scope/threshold stripped."""
    bundle, name = detector.split("::", 1)
    parts = name.split("|")
    if bundle == "moe-v7-0905":
        return f"v7_{name}"
    if bundle == "moe-circuit-analogy-0906" and parts[0] in (
        "circuit", "published", "conjunction"
    ):
        return parts[1] if parts[0] == "published" else f"{parts[0]}_{parts[1]}"
    return parts[0]


def root_of(quantity: str) -> str:
    """Coarser root: denoise-step suffixes collapsed."""
    return re.sub(r"_s\d+(_and_s\d+|_or_s\d+)?$", "", quantity)


def _counts(fire: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = fire.astype(np.float32)
    return f @ f.T, f.sum(1)


def similarity(fire: np.ndarray, kind: str) -> np.ndarray:
    n = fire.shape[1]
    inter, a = _counts(fire)
    union = a[:, None] + a[None, :] - inter
    mn = np.minimum(a[:, None], a[None, :])
    mx = np.maximum(a[:, None], a[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "jaccard":
            s = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        elif kind == "excess_jaccard":
            e_int = a[:, None] * a[None, :] / n
            e_un = a[:, None] + a[None, :] - e_int
            s = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0) - np.where(
                e_un > 0, e_int / np.maximum(e_un, 1e-9), 0.0
            )
        elif kind == "overlap":
            s = np.where(mn > 0, inter / np.maximum(mn, 1e-9), 0.0)
        elif kind == "excess_overlap":
            s = np.where(mn > 0, inter / np.maximum(mn, 1e-9), 0.0) - mx / n
        else:
            raise ValueError(kind)
    s = np.clip(s, 0.0, 1.0)
    np.fill_diagonal(s, 1.0)
    return s


def cluster(sim: np.ndarray, cut: float) -> np.ndarray:
    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=1.0 - cut, criterion="distance")


def within_between(sim: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    iu = np.triu_indices_from(sim, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    w = float(sim[iu][same].mean()) if same.any() else float("nan")
    b = float(sim[iu][~same].mean()) if (~same).any() else float("nan")
    return w, b


def effective_rank(fire: np.ndarray) -> dict:
    """How many independent signals are in the alarm matrix at all.

    Singular values of the column-centred binary in-window fire matrix.  The
    participation ratio and the 90%-variance component count say how many
    directions the whole detector bank actually spans; a bank of genuinely
    independent detectors would need as many components as detectors.
    """
    x = fire.astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    var = sv**2
    if var.sum() <= 0:
        return {"participation_ratio": float("nan"), "n_pc_90": float("nan")}
    p = var / var.sum()
    return {
        "participation_ratio": float(1.0 / np.sum(p**2)),
        "n_pc_90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "n_pc_99": int(np.searchsorted(np.cumsum(p), 0.99) + 1),
        "top_pc_share": float(p[0]),
    }


def describe(labels: np.ndarray, names: list[str], sim: np.ndarray,
             tp: np.ndarray, fp: np.ndarray, rate: np.ndarray) -> pd.DataFrame:
    rows = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        sub = sim[np.ix_(idx, idx)]
        medoid = idx[int(np.argmax(sub.mean(1)))]
        qs = [quantity_of(names[i]) for i in idx]
        roots = [root_of(q) for q in qs]
        rows.append({
            "family": int(lab),
            "size": len(idx),
            "medoid": names[medoid],
            "medoid_quantity": quantity_of(names[medoid]),
            "n_quantities": len(set(qs)),
            "n_roots": len(set(roots)),
            "modal_root": pd.Series(roots).mode().iloc[0],
            "modal_root_share": float(pd.Series(roots).value_counts(normalize=True).iloc[0]),
            "n_bundles": len({names[i].split("::")[0] for i in idx}),
            "mean_within_sim": float(sub[np.triu_indices(len(idx), 1)].mean())
            if len(idx) > 1 else float("nan"),
            "min_fire_rate": float(rate[idx].min()),
            "max_fire_rate": float(rate[idx].max()),
            "best_tp": int(tp[idx].max()),
            "best_tp_detector": names[idx[int(np.argmax(tp[idx]))]],
            "fp_at_best_tp": int(fp[idx[int(np.argmax(tp[idx]))]]),
            "roots": "; ".join(sorted(set(roots))),
            "members": "; ".join(sorted(names[i] for i in idx)),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    ext = sc.load_cohort("external_8b")
    dev = sc.load_cohort("development_main")
    shared = sc.shared_detectors(ext, dev)

    out_fam, out_prof = [], []
    meta = {"schema": "himoe.method_synthesis.families.v2",
            "headline_similarity": HEADLINE_SIM, "headline_cut": HEADLINE_CUT,
            "similarity_note": (
                "excess_overlap = |AnB|/min(|A|,|B|) - max(|A|,|B|)/n; it is "
                "invariant to threshold level (nested sets score 1) and gives a "
                "detector that fires on nearly everything no credit."
            )}
    store: dict[str, dict] = {}

    arms = [
        ("development_shared419", dev, shared),
        ("external_shared419", ext, shared),
        ("external_all622", ext, ext.detectors),
    ]
    for tag, coh, keep in arms:
        idx = [coh.detectors.index(k) for k in keep]
        alarms = coh.alarms[idx]
        iw = (alarms >= 0) & (alarms < coh.deadline[None, :])
        live = np.flatnonzero(iw.any(1))
        dead = [keep[i] for i in range(len(keep)) if i not in set(live.tolist())]
        iw = iw[live]
        names = [keep[i] for i in live]
        tp = (iw & coh.risk).sum(1)
        fp = (iw & ~coh.risk).sum(1)
        rate = iw.sum(1) / coh.n
        null_fire = sc.rate_matched_null(iw, rng, strata=coh.suite)

        sims = {k: similarity(iw, k) for k in
                ("jaccard", "excess_jaccard", "overlap", "excess_overlap")}
        null_sims = {k: similarity(null_fire, k) for k in sims}

        for kind in sims:
            for cut in CUTS:
                lab = cluster(sims[kind], cut)
                lab_null = cluster(null_sims[kind], cut)
                w, b = within_between(sims[kind], lab)
                wn, bn = within_between(null_sims[kind], lab_null)
                out_prof.append({
                    "arm": tag, "suite": "pooled", "similarity": kind, "cut": cut,
                    "n_detectors": len(names),
                    "n_families_observed": int(len(np.unique(lab))),
                    "n_families_null": int(len(np.unique(lab_null))),
                    "largest_family_observed": int(np.bincount(lab).max()),
                    "largest_family_null": int(np.bincount(lab_null).max()),
                    "within_sim_observed": w, "between_sim_observed": b,
                    "within_sim_null": wn, "between_sim_null": bn,
                })

        labels = cluster(sims[HEADLINE_SIM], HEADLINE_CUT)
        fam = describe(labels, names, sims[HEADLINE_SIM], tp, fp, rate)
        fam.insert(0, "arm", tag)
        out_fam.append(fam)
        store[tag] = {"labels": labels, "names": names, "sims": sims}

        er = effective_rank(iw)
        er_null = effective_rank(null_fire)
        meta[tag] = {
            "n_detectors": len(keep),
            "n_never_fire_in_window": len(dead),
            "never_fire_in_window": dead,
            "n_families_headline": int(len(np.unique(labels))),
            "n_singleton_families": int((fam["size"] == 1).sum()),
            "n_families_null_headline": int(
                len(np.unique(cluster(null_sims[HEADLINE_SIM], HEADLINE_CUT)))
            ),
            "n_named_quantities": len({quantity_of(k) for k in names}),
            "n_named_roots": len({root_of(quantity_of(k)) for k in names}),
            "families_spanning_multiple_named_roots": int((fam["n_roots"] > 1).sum()),
            "detectors_in_multi_root_families": int(fam[fam.n_roots > 1]["size"].sum()),
            "effective_rank_observed": er,
            "effective_rank_null": er_null,
        }

        for s in sorted(set(coh.suite)):
            m = coh.suite == s
            iws = iw[:, m]
            alive = np.flatnonzero(iws.any(1))
            if len(alive) < 3:
                continue
            sub = iws[alive]
            sim_s = similarity(sub, HEADLINE_SIM)
            null_s = similarity(sc.rate_matched_null(sub, rng), HEADLINE_SIM)
            lab_s = cluster(sim_s, HEADLINE_CUT)
            lab_ns = cluster(null_s, HEADLINE_CUT)
            w, b = within_between(sim_s, lab_s)
            wn, bn = within_between(null_s, lab_ns)
            out_prof.append({
                "arm": tag, "suite": s, "similarity": HEADLINE_SIM, "cut": HEADLINE_CUT,
                "n_detectors": int(len(alive)),
                "n_families_observed": int(len(np.unique(lab_s))),
                "n_families_null": int(len(np.unique(lab_ns))),
                "largest_family_observed": int(np.bincount(lab_s).max()),
                "largest_family_null": int(np.bincount(lab_ns).max()),
                "within_sim_observed": w, "between_sim_observed": b,
                "within_sim_null": wn, "between_sim_null": bn,
            })

    # Do the families derived on development reproduce on external?
    common = [n for n in store["development_shared419"]["names"]
              if n in set(store["external_shared419"]["names"])]
    a = np.asarray([store["development_shared419"]["labels"][
        store["development_shared419"]["names"].index(n)] for n in common])
    b = np.asarray([store["external_shared419"]["labels"][
        store["external_shared419"]["names"].index(n)] for n in common])
    rng2 = np.random.default_rng(SEED + 1)
    meta["family_replication_dev_to_ext"] = {
        "n_detectors_compared": len(common),
        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
        "ari_null_permuted": float(
            np.mean([adjusted_rand_score(a, rng2.permutation(b)) for _ in range(200)])
        ),
        "n_families_dev": int(len(np.unique(a))),
        "n_families_ext": int(len(np.unique(b))),
    }

    pd.concat(out_fam, ignore_index=True).to_csv(sc.RESULTS / "families.csv", index=False)
    pd.DataFrame(out_prof).to_csv(sc.RESULTS / "family_cut_profile.csv", index=False)
    np.savez_compressed(
        sc.RESULTS / "family_labels.npz",
        names_dev=np.array(store["development_shared419"]["names"]),
        labels_dev=np.asarray(store["development_shared419"]["labels"]),
        names_ext_shared=np.array(store["external_shared419"]["names"]),
        labels_ext_shared=np.asarray(store["external_shared419"]["labels"]),
        names_ext_all=np.array(store["external_all622"]["names"]),
        labels_ext_all=np.asarray(store["external_all622"]["labels"]),
    )
    (sc.RESULTS / "families.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
