"""判出率 (discrimination = max(AUC, 1-AUC)) of action tokens vs the whole MoE tensor.

Two targets:
  T1  final BDDL success/failure, risk set = branches still alive at query t
  T2  future trap onset within lead L, risk set = alive AND not yet trapped   (corpus A only)

Protocol is inherited verbatim from the 2026-08-29 organization sweep:
  - deep block HB 12-15, denoise 9, 8-step trailing window   (the frozen baseline slice)
  - AUC paired WITHIN group only (A: worker, B: init_state_id), pooled by pair count
  - families priced by within-group label permutation + maxT
  - soft probabilities only; hard top-4 ids are never touched
"""
import json
import sys

import numpy as np
import pandas as pd

ROOT = "/home/jovyan/work/himoe-vla"
RS = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
A_LAB = f"{RS}/analysis/candidate_physical_labels.csv"
A_ONSET = f"{RS}/analysis_trap_event_moe_20260829/event_onsets.csv"
B_DIR = (f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
         "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")

WIN = 8
DENOISE = 9
NPERM = 2000
RNG = np.random.default_rng(20260901)

TOK_ORGS = ([f"tok{k:02d}" for k in range(11)]
            + ["act1_3", "act4_7", "act8_10", "act_all_1_10", "all11"])
KINDS = ["hell", "ent"]
FAMILY = [f"{o}|{k}" for o in TOK_ORGS for k in KINDS]
BASELINE = "act_all_1_10|hell"

PRETTY = {"tok00": "tok00 (state)", "act_all_1_10": "action_all (1-10)",
          "all11": "all_tokens (0-10)"}


# --------------------------------------------------------------- data loading
def load_feats(tag, d=DENOISE):
    z = np.load(f"{ROOT}/.tokaxis_{tag}_d{d}.npz")
    eids = z["__eids__"]
    lens = z["__lens__"]
    off = np.concatenate([[0], np.cumsum(lens)])
    return eids, lens, off, {f: z[f] for f in FAMILY}


def meta(tag):
    """Return (y, g) dicts keyed by episode id. y: 1 = success."""
    if tag == "A":
        lab = pd.read_csv(A_LAB)
        return (lab.set_index("episode_id")["success"].astype(int).to_dict(),
                lab.set_index("episode_id")["worker"].to_dict())
    summ = json.load(open(f"{B_DIR}/client/summaries.json"))
    return ({i: int(bool(s["success"])) for i, s in enumerate(summ)},
            {i: s["init_state_id"] for i, s in enumerate(summ)})


def window_matrix(F, off, lens, t):
    """(n_branch, n_feat) trailing-window mean ending at step t; nan if not alive."""
    n = len(lens)
    X = np.full((n, len(FAMILY)), np.nan, np.float64)
    alive = lens > t
    lo_rel, hi_rel = t - WIN + 1, t + 1
    for j, f in enumerate(FAMILY):
        s = F[f]
        for i in np.where(alive)[0]:
            v = s[off[i] + lo_rel: off[i] + hi_rel]
            v = v[~np.isnan(v)]
            if len(v):
                X[i, j] = v.mean()
    return X, alive


# ------------------------------------------------------------------- AUC core
def _group_rank(X, groups):
    """Within-group ranks (average ties). Returns R same shape as X."""
    R = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        R[m] = pd.DataFrame(X[m]).rank().to_numpy()
    return R


def grouped_auc_from_ranks(R, fail, groups):
    """Pooled within-group AUC = P(score_fail > score_succ), weighted by pair count.

    Only groups holding both classes contribute. Returns (auc[n_feat], W, C).
    """
    num = np.zeros(R.shape[1])
    W = 0.0
    C = 0.0
    for g in np.unique(groups):
        m = groups == g
        nf = int(fail[m].sum())
        ns = int(m.sum() - nf)
        if nf == 0 or ns == 0:
            continue
        num += R[m][fail[m]].sum(0)
        W += nf * ns
        C += nf * (nf + 1) / 2.0
    if W == 0:
        return np.full(R.shape[1], np.nan), 0.0, 0.0
    return (num - C) / W, W, C


def perm_null(R, fail, groups, nperm=NPERM):
    """maxT null for discrimination under within-group label permutation."""
    idx_by_g = [np.where(groups == g)[0] for g in np.unique(groups)]
    valid = []
    W = 0.0
    C = 0.0
    for ix in idx_by_g:
        nf = int(fail[ix].sum())
        ns = len(ix) - nf
        if nf and ns:
            valid.append((ix, nf))
            W += nf * ns
            C += nf * (nf + 1) / 2.0
    maxT = np.empty(nperm)
    per_feat = np.empty((nperm, R.shape[1]))
    for p in range(nperm):
        tot = np.zeros(R.shape[1])
        for ix, nf in valid:
            pick = RNG.choice(len(ix), nf, replace=False)
            tot += R[ix[pick]].sum(0)
        auc = (tot - C) / W
        disc = np.maximum(auc, 1.0 - auc)
        per_feat[p] = disc
        maxT[p] = disc.max()
    return maxT, per_feat


def rank_residualise(X, groups, base_col):
    """Within-group rank residual of every feature on the baseline feature."""
    R = _group_rank(X, groups)
    out = np.array(R, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        b = R[m][:, base_col]
        b = b - b.mean()
        denom = (b * b).sum()
        if denom <= 0:
            continue
        blk = R[m] - R[m].mean(0)
        beta = (blk * b[:, None]).sum(0) / denom
        out[m] = blk - beta[None, :] * b[:, None]
    return out


# ---------------------------------------------------------------- T1: outcome
def target1(tag, ts):
    eids, lens, off, F = load_feats(tag)
    ymap, gmap = meta(tag)
    rows = []
    for t in ts:
        X, alive = window_matrix(F, off, lens, t)
        idx = np.where(alive)[0]
        Xa = X[idx]
        if np.isnan(Xa).any():
            bad = np.isnan(Xa).sum()
            print(f"  [warn] {tag} t={t}: {bad} nan cells", file=sys.stderr)
        y = np.array([ymap[int(e)] for e in eids])[idx]
        g = np.array([gmap[int(e)] for e in eids])[idx]
        fail = y == 0

        R = _group_rank(Xa, g)
        auc, W, _ = grouped_auc_from_ranks(R, fail, g)
        disc = np.maximum(auc, 1.0 - auc)
        maxT, _ = perm_null(R, fail, g)
        p_family = np.array([(maxT >= d).mean() for d in disc])

        Rres = rank_residualise(Xa, g, FAMILY.index(BASELINE))
        Rres = _group_rank(Rres, g)
        auc_r, _, _ = grouped_auc_from_ranks(Rres, fail, g)
        disc_r = np.maximum(auc_r, 1.0 - auc_r)
        maxT_r, _ = perm_null(Rres, fail, g)
        p_family_r = np.array([(maxT_r >= d).mean() for d in disc_r])

        for j, f in enumerate(FAMILY):
            org, kind = f.split("|")
            rows.append(dict(corpus=tag, t=t, org=org, kind=kind,
                             n=len(idx), n_fail=int(fail.sum()),
                             n_succ=int((~fail).sum()), pairs=W,
                             auc=auc[j], disc=disc[j], p_family=p_family[j],
                             auc_resid=auc_r[j], disc_resid=disc_r[j],
                             p_family_resid=p_family_r[j],
                             null_med=np.median(maxT)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- T2: trap onset
def target2(leads, qs):
    """Corpus A only: predict first trap onset in (q, q+lead], strata = (worker, q)."""
    eids, lens, off, F = load_feats("A")
    on = pd.read_csv(A_ONSET).set_index("episode_id")
    onset = on["trap_onset"].to_dict()
    worker = on["worker"].to_dict()
    pos = {int(e): i for i, e in enumerate(eids)}

    rows = []
    for lead_lo, lead_hi in leads:
        Xs, fails, strata = [], [], []
        for q in qs:
            X, alive = window_matrix(F, off, lens, q)
            for i in np.where(alive)[0]:
                e = int(eids[i])
                o = onset.get(e, -1)
                if o != -1 and o <= q:          # already trapped -> out of risk set
                    continue
                hit = (o != -1) and (q + lead_lo <= o <= q + lead_hi)
                Xs.append(X[i])
                fails.append(hit)
                strata.append(f"w{worker[e]}_q{q}")
        Xa = np.array(Xs)
        fail = np.array(fails)
        g = np.array(strata)
        keep = ~np.isnan(Xa).any(1)
        Xa, fail, g = Xa[keep], fail[keep], g[keep]

        R = _group_rank(Xa, g)
        auc, W, _ = grouped_auc_from_ranks(R, fail, g)
        disc = np.maximum(auc, 1.0 - auc)
        maxT, _ = perm_null(R, fail, g)
        p_family = np.array([(maxT >= d).mean() for d in disc])

        Rres = _group_rank(rank_residualise(Xa, g, FAMILY.index(BASELINE)), g)
        auc_r, _, _ = grouped_auc_from_ranks(Rres, fail, g)
        disc_r = np.maximum(auc_r, 1.0 - auc_r)
        maxT_r, _ = perm_null(Rres, fail, g)
        p_family_r = np.array([(maxT_r >= d).mean() for d in disc_r])

        for j, f in enumerate(FAMILY):
            org, kind = f.split("|")
            rows.append(dict(corpus="A", lead=f"{lead_lo}-{lead_hi}", org=org, kind=kind,
                             n_rows=len(fail), n_event=int(fail.sum()),
                             strata=len(np.unique(g)), pairs=W,
                             auc=auc[j], disc=disc[j], p_family=p_family[j],
                             auc_resid=auc_r[j], disc_resid=disc_r[j],
                             p_family_resid=p_family_r[j],
                             null_med=np.median(maxT)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    TS = [10, 15, 20, 25, 30]
    t1 = pd.concat([target1("A", TS), target1("B", TS)], ignore_index=True)
    t1.to_csv(f"{ROOT}/tokdisc_target1.csv", index=False)
    print("T1 baseline reproduction check (act_all_1_10|hell):")
    chk = t1[(t1.org == "act_all_1_10") & (t1.kind == "hell") & (t1.t == 30)]
    for _, r in chk.iterrows():
        print(f"  corpus {r.corpus} t=30 AUC={r.auc:.3f}  (sweep doc records A 0.795 / B 0.759)")

    t2 = target2([(1, 4), (5, 8), (9, 12)], list(range(10, 31)))
    t2.to_csv(f"{ROOT}/tokdisc_target2.csv", index=False)
    print("\nwrote tokdisc_target1.csv / tokdisc_target2.csv")


# ------------------------------------------- T2b: single-q (no repeated measures)
def target2_singleq(leads, qs):
    """Same as target2 but ONE query position per reported number, strata = worker.

    Each branch contributes exactly one row, so the within-worker label
    permutation is exactly valid (target2 pools 21 q's per branch and its
    permutation null is therefore anti-conservative).
    """
    eids, lens, off, F = load_feats("A")
    on = pd.read_csv(A_ONSET).set_index("episode_id")
    onset = on["trap_onset"].to_dict()
    worker = on["worker"].to_dict()
    rows = []
    for q in qs:
        X, alive = window_matrix(F, off, lens, q)
        for lead_lo, lead_hi in leads:
            keep, fails, g = [], [], []
            for i in np.where(alive)[0]:
                e = int(eids[i])
                o = onset.get(e, -1)
                if o != -1 and o <= q:
                    continue
                keep.append(i)
                fails.append((o != -1) and (q + lead_lo <= o <= q + lead_hi))
                g.append(worker[e])
            Xa, fail, g = X[keep], np.array(fails), np.array(g)
            ok = ~np.isnan(Xa).any(1)
            Xa, fail, g = Xa[ok], fail[ok], g[ok]
            if fail.sum() < 3 or (~fail).sum() < 3:
                continue
            R = _group_rank(Xa, g)
            auc, W, _ = grouped_auc_from_ranks(R, fail, g)
            disc = np.maximum(auc, 1.0 - auc)
            maxT, _ = perm_null(R, fail, g)
            p = np.array([(maxT >= d).mean() for d in disc])
            Rres = _group_rank(rank_residualise(Xa, g, FAMILY.index(BASELINE)), g)
            auc_r, _, _ = grouped_auc_from_ranks(Rres, fail, g)
            disc_r = np.maximum(auc_r, 1.0 - auc_r)
            maxT_r, _ = perm_null(Rres, fail, g)
            p_r = np.array([(maxT_r >= d).mean() for d in disc_r])
            for j, f in enumerate(FAMILY):
                org, kind = f.split("|")
                rows.append(dict(q=q, lead=f"{lead_lo}-{lead_hi}", org=org, kind=kind,
                                 n=len(fail), n_event=int(fail.sum()), pairs=W,
                                 auc=auc[j], disc=disc[j], p_family=p[j],
                                 disc_resid=disc_r[j], p_family_resid=p_r[j],
                                 null_med=np.median(maxT)))
    return pd.DataFrame(rows)
