"""Build per-chunk feature/target caches for corpus A (rolling-star branches)
and corpus B (KITCHEN_SCENE8 replication)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_events")
from common import (RUN, HUB, OUT, G, G_CENT, G_SLOT, DENOISE, TABLE_Z, TH,
                    Q_LO, Q_HI, Q_RANGE, S_ARM, S_FING, S_P1, S_Q1, S_P2, S_Q2,
                    S_BTN, S_ARMV, dgoal, tilt_deg)

HOR = [1, 2, 4]
SUF = os.environ.get("SUF", "")
DISP_TH = 0.010          # m, per-object displacement threshold
TILT_TH = 10.0           # deg
JLIM_TH = 0.10           # fraction of joint range from a limit
TWIST_DQ = 0.15          # rad
TWIST_DE = 0.010         # m
GRIP_TH = 0.030          # finger-width binarisation (half-width per finger ~0.04 open)
FAR_M = 0.15             # end-effector considered far from an object beyond this


def chunk_summary(a):
    """a (10,7) action chunk -> compact summary."""
    tr = a[:, :3]
    ro = a[:, 3:6]
    gp = a[:, 6]
    cs = tr.cumsum(0)
    sg = np.sign(gp)
    nflip = int((np.diff(sg) != 0).sum())
    return np.concatenate([
        cs[-1], [np.linalg.norm(cs[-1])],
        [np.abs(tr).sum(), np.linalg.norm(tr, axis=1).mean(), np.linalg.norm(tr, axis=1).max()],
        ro.sum(0), [np.abs(ro).sum()],
        [gp.mean(), gp.min(), gp.max(), gp[0], gp[-1], nflip],
    ])


def build_branch(ss, ps, ac, extra_state=None):
    """ss (K,47) per-query sim states, ps (K,8) proprio, ac (K,10,7) action chunks.
    extra_state: optional (47,) terminal sim state appended as index K."""
    K = ss.shape[0]
    SS = ss.astype(np.float64)
    if extra_state is not None:
        SS = np.vstack([SS, extra_state.astype(np.float64)[None]])
    n = SS.shape[0]                       # n = K or K+1 state samples
    eef = ps[:, :3].astype(np.float64)
    aa = ps[:, 3:6].astype(np.float64)
    fing = ps[:, 6:8].astype(np.float64)
    width = np.abs(fing[:, 0] - fing[:, 1])

    q = SS[:, S_ARM]
    p1 = SS[:, S_P1]
    p2 = SS[:, S_P2]
    qu1 = SS[:, S_Q1]
    qu2 = SS[:, S_Q2]
    btn = SS[:, S_BTN]
    qv = SS[:, S_ARMV]
    fingv = SS[:, 32:34]
    p1v = SS[:, 34:40]
    p2v = SS[:, 40:46]
    fingq = SS[:, S_FING]
    widthq = np.abs(fingq[:, 0] - fingq[:, 1])
    # end-effector position for the state array (needs n rows); for the appended
    # terminal state we approximate with the last query eef (only used for twist
    # target at the final index, guarded below).
    eef_n = np.vstack([eef, eef[-1:]]) if n > K else eef

    t1 = tilt_deg(qu1)
    t2 = tilt_deg(qu2)
    dg1 = dgoal(p1)
    dg2 = dgoal(p2)
    jn = np.minimum(q - Q_LO, Q_HI - q) / Q_RANGE
    jmin = jn.min(1)
    gopen = (widthq > GRIP_TH).astype(np.int8)

    # ---------------- targets ----------------
    tg = {}
    for m in HOR:
        valid = np.zeros(K, bool)
        valid[: max(0, n - m)] = True          # k+m must exist
        idx = np.arange(K)
        j = np.clip(idx + m, 0, n - 1)

        d1 = np.linalg.norm(p1[j] - p1[idx], axis=1)
        d2 = np.linalg.norm(p2[j] - p2[idx], axis=1)
        tg[f"disp1_m{m}"] = (d1 > DISP_TH, valid)
        tg[f"disp2_m{m}"] = (d2 > DISP_TH, valid)
        tg[f"dispmag1_m{m}"] = (d1, valid)
        tg[f"dispmag2_m{m}"] = (d2, valid)
        tg[f"tilt1_m{m}"] = (np.abs(t1[j] - t1[idx]) > TILT_TH, valid)
        tg[f"tilt2_m{m}"] = (np.abs(t2[j] - t2[idx]) > TILT_TH, valid)

        # windowed extremes over (k, k+m]
        def wmin(x):
            out = np.full(K, np.nan)
            for k in range(K):
                hi = min(k + m, n - 1)
                if hi > k:
                    out[k] = x[k + 1: hi + 1].min()
            return out

        def wmax(x):
            out = np.full(K, np.nan)
            for k in range(K):
                hi = min(k + m, n - 1)
                if hi > k:
                    out[k] = x[k + 1: hi + 1].max()
            return out

        jm_w = wmin(jmin)
        tg[f"jlim_m{m}"] = (jm_w < JLIM_TH, valid & ~np.isnan(jm_w))
        tg[f"AUXjminw_m{m}"] = (jm_w, valid & ~np.isnan(jm_w))

        dq = np.linalg.norm(q[j] - q[idx], axis=1)
        de = np.linalg.norm(eef_n[j] - eef_n[idx], axis=1)
        tg[f"twist_m{m}"] = ((dq > TWIST_DQ) & (de < TWIST_DE), valid)
        tg[f"dqmag_m{m}"] = (dq, valid)
        tg[f"AUXde_m{m}"] = (de, valid)

        gflip = np.zeros(K, bool)
        for k in range(K):
            hi = min(k + m, n - 1)
            if hi > k:
                gflip[k] = (gopen[k + 1: hi + 1] != gopen[k]).any()
        tg[f"grip_m{m}"] = (gflip, valid)

        # ---- proprio-blind family: motion with the gripper open and far away ----
        gopen_k = gopen[:K].astype(bool)
        de1k = np.linalg.norm(eef - p1[:K], axis=1)
        de2k = np.linalg.norm(eef - p2[:K], axis=1)
        # min eef-object distance over the whole window [k, k+m]
        def wmin_pair(po):
            dd = np.linalg.norm(eef_n[: n] - po[: n], axis=1)
            out = np.full(K, np.inf)
            for k in range(K):
                hi = min(k + m, n - 1)
                out[k] = dd[k: hi + 1].min()
            return out
        nc1 = wmin_pair(p1)
        nc2 = wmin_pair(p2)
        mv1 = d1 > DISP_TH
        mv2 = d2 > DISP_TH
        for o, (dek, ncw, mvo) in enumerate([(de1k, nc1, mv1), (de2k, nc2, mv2)], start=1):
            pop_nc = valid & gopen_k & (dek > FAR_M)
            tg[f"dispnc{o}_m{m}"] = (mvo, pop_nc)
            tg[f"dispfar{o}_m{m}"] = (mvo, valid & gopen_k & (ncw > FAR_M))
        # which pot moves (exactly one moves): identity is the vision-dependent bit
        one = valid & (mv1 ^ mv2)
        tg[f"whichpot_m{m}"] = (mv2.astype(np.float32), one)
        tg[f"whichpotnc_m{m}"] = (mv2.astype(np.float32),
                                  one & gopen_k & (de1k > FAR_M) & (de2k > FAR_M))

        # conditional populations, per object
        for o, (dgo, po) in enumerate([(dg1, p1), (dg2, p2)], start=1):
            dek = de1k if o == 1 else de2k
            ncw = nc1 if o == 1 else nc2
            popg = valid & (dgo[:K] < 0.09)
            reg = wmax(dgo) > (dgo[:K] + 0.02)
            tg[f"goalreg{o}_m{m}"] = (reg, popg)
            tg[f"goalregnc{o}_m{m}"] = (reg, popg & gopen_k & (ncw > FAR_M))
            pop = valid & (dgo[:K] < TH["goal_enter_m"])
            lv = wmax(dgo) > TH["goal_exit_m"]
            tg[f"leavegoal{o}_m{m}"] = (lv, pop)
            liftv = po[:K, 2] - TABLE_Z
            popl = valid & (liftv > TH["lift_enter_m"])
            zw = wmin(po[:, 2])
            dr = zw < (po[:K, 2] - TH["drop_m"])
            tg[f"drop{o}_m{m}"] = (dr, popl)
            # height loss that is NOT a controlled placement onto a goal slot
            not_placed = dgo[j] > TH["goal_enter_m"]
            tg[f"dropfail{o}_m{m}"] = (dr & not_placed, popl)

    # ---------------- features ----------------
    def lag(x, L):
        y = np.empty_like(x)
        y[:L] = x[:1] if x.ndim == 1 else x[:1]
        y[L:] = x[:-L]
        return y

    eefq = eef
    ph = np.stack([np.arange(K), np.full(K, K), np.arange(K) / max(K - 1, 1),
                   K - np.arange(K), np.log1p(np.arange(K))], 1)

    acs = np.stack([chunk_summary(ac[k]) for k in range(K)])
    acs_p = lag(acs, 1)
    predend = eefq + acs[:, :3]

    def dslot(p):
        return np.stack([np.linalg.norm(p - G_SLOT[0], axis=1),
                         np.linalg.norm(p - G_SLOT[1], axis=1)], 1)

    # --- strictly causal proprio history: makes the deployable control honest ---
    gop = gopen[:K].astype(np.float64)
    flip = np.r_[0.0, (np.diff(gop) != 0).astype(np.float64)]
    since = np.zeros(K)
    c = 999.0
    nfl = np.zeros(K)
    for k in range(K):
        c = 0.0 if flip[k] else c + 1.0
        since[k] = min(c, 50.0)
        nfl[k] = flip[: k + 1].sum()
    step = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1)]
    cpath = np.cumsum(step)
    path4 = cpath - np.r_[np.zeros(min(4, K)), cpath[:-4]] if K > 4 else cpath
    hist = np.stack([gop, flip, since, nfl, step, cpath, path4], 1)
    lagf = np.concatenate([eefq - lag(eefq, 3), eefq - lag(eefq, 4),
                           lag(gop, 1)[:, None], lag(gop, 2)[:, None],
                           lag(gop, 3)[:, None], lag(gop, 4)[:, None],
                           lag(width, 3)[:, None], lag(width, 4)[:, None],
                           lag(acs[:, 15], 1)[:, None], lag(acs[:, 16], 1)[:, None]], 1)

    t1_feats = [
        eefq, aa, fing, width[:, None],
        eefq - lag(eefq, 1), eefq - lag(eefq, 2),
        (width - lag(width, 1))[:, None], (width - lag(width, 2))[:, None],
        dgoal(eefq)[:, None], dslot(eefq), eefq - G_CENT,
        np.linalg.norm(eefq - G_CENT, axis=1)[:, None],
        ac.reshape(K, -1), acs, acs_p,
        predend, dgoal(predend)[:, None], dslot(predend),
        hist, lagf,
        ph,
    ]
    T1 = np.concatenate([np.atleast_2d(x.T).T if x.ndim == 1 else x for x in t1_feats], 1)

    P1 = p1[:K]; P2 = p2[:K]; QU1 = qu1[:K]; QU2 = qu2[:K]
    t2_extra = [
        SS[:K],                                            # full 47-dim sim_state
        P1, P2, QU1, QU2, t1[:K, None], t2[:K, None], btn[:K, None],
        q[:K], qv[:K], fingv[:K], p1v[:K], p2v[:K], jn[:K], jmin[:K, None],
        dg1[:K, None], dg2[:K, None], dslot(P1), dslot(P2),
        np.linalg.norm(eefq - P1, axis=1)[:, None],
        np.linalg.norm(eefq - P2, axis=1)[:, None],
        np.linalg.norm(P1 - P2, axis=1)[:, None],
        (P1[:, 2] - TABLE_Z)[:, None], (P2[:, 2] - TABLE_Z)[:, None],
        eefq - P1, eefq - P2, P1 - P2,
        P1 - lag(P1, 1), P2 - lag(P2, 1), P1 - lag(P1, 2), P2 - lag(P2, 2),
        (dg1[:K] - lag(dg1[:K], 1))[:, None], (dg2[:K] - lag(dg2[:K], 1))[:, None],
        (dg1[:K] - lag(dg1[:K], 2))[:, None], (dg2[:K] - lag(dg2[:K], 2))[:, None],
        np.linalg.norm(predend - P1, axis=1)[:, None],
        np.linalg.norm(predend - P2, axis=1)[:, None],
        (t1[:K] - lag(t1[:K], 1))[:, None], (t2[:K] - lag(t2[:K], 1))[:, None],
    ]
    T2 = np.concatenate([T1] + t2_extra, 1)
    return T1.astype(np.float32), T2.astype(np.float32), ph.astype(np.float32), tg


def route_feats(zg, rows):
    """rows: array of zarr row indices in query order -> (len,2816) float32."""
    out = np.empty((len(rows), 8 * 11 * 32), np.float32)
    arr = zg["hb_router_probs"]
    B = 256
    for i in range(0, len(rows), B):
        r = rows[i: i + B]
        blk = arr[r[0]: r[-1] + 1, :, DENOISE, :, :]
        sel = r - r[0]
        out[i: i + B] = blk[sel].reshape(len(r), -1).astype(np.float32)
    return out


# ------------------------------------------------------------------ corpus A
def prep_A():
    zg = zarr.open_group(f"{RUN}/formal/server/routes.zarr", mode="r")
    ep = zg["episode_id"][:]
    cs = zg["control_step"][:]
    lab = pd.read_csv(f"{RUN}/analysis/candidate_physical_labels.csv")
    order = np.argsort(ep, kind="stable")
    ep_s = ep[order]
    bounds = np.searchsorted(ep_s, np.unique(ep_s), side="left")
    rows_by_ep = {}
    uq = np.unique(ep_s)
    ends = np.r_[bounds[1:], len(ep_s)]
    for e, a, b in zip(uq, bounds, ends):
        r = order[a:b]
        rows_by_ep[int(e)] = r[np.argsort(cs[r])]

    T1L, T2L, PHL, RL, META = [], [], [], [], []
    TG = {}
    for _, row in lab.iterrows():
        f = f"{RUN}/formal/worker{int(row.worker)}/snapshot_{int(row.snapshot):03d}/candidate_{int(row.candidate):02d}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ss, ps, ac = z["sim_state"], z["policy_state"], z["action_chunks"]
        css = z["control_sim_state"]
        K = ss.shape[0]
        eid = int(row.episode_id)
        r = rows_by_ep.get(eid)
        if r is None or len(r) != K:
            continue
        t1, t2, ph, tg = build_branch(ss, ps, ac, extra_state=css[-1])
        rf = route_feats(zg, r)
        T1L.append(t1); T2L.append(t2); PHL.append(ph); RL.append(rf)
        META.append(pd.DataFrame(dict(worker=int(row.worker), snapshot=int(row.snapshot),
                                      candidate=int(row.candidate), episode=eid,
                                      k=np.arange(K), K=K, success=bool(row.success))))
        for name, (v, m) in tg.items():
            TG.setdefault(name, [[], []])
            TG[name][0].append(np.asarray(v, np.float32))
            TG[name][1].append(np.asarray(m, bool))
    return _pack(T1L, T2L, PHL, RL, META, TG, "A")


# ------------------------------------------------------------------ corpus B
def prep_B():
    zg = zarr.open_group(f"{HUB}/server/routes.zarr", mode="r")
    summ = json.load(open(f"{HUB}/client/summaries.json"))
    off = np.r_[0, np.cumsum([s["inference_calls"] for s in summ])]
    ep = zg["episode_id"][:]
    T1L, T2L, PHL, RL, META = [], [], [], [], []
    TG = {}
    for i, s in enumerate(summ):
        f = f"{HUB}/client/episode_{i:02d}.npz" if i < 100 else f"{HUB}/client/episode_{i}.npz"
        if not os.path.exists(f):
            f = f"{HUB}/client/episode_{i:02d}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ss, ps, ac = z["sim_state"], z["state"], z["actions"]
        K = ss.shape[0]
        a, b = int(off[i]), int(off[i + 1])
        if b - a != K:
            continue
        assert (ep[a:b] == ep[a]).all(), (i, "episode block mismatch")
        r = np.arange(a, b)
        t1, t2, ph, tg = build_branch(ss, ps, ac, extra_state=None)
        rf = route_feats(zg, r)
        T1L.append(t1); T2L.append(t2); PHL.append(ph); RL.append(rf)
        META.append(pd.DataFrame(dict(worker=int(s["init_state_id"]), snapshot=0,
                                      candidate=int(s["repeat"]), episode=i,
                                      k=np.arange(K), K=K, success=bool(s["success"]))))
        for name, (v, m) in tg.items():
            TG.setdefault(name, [[], []])
            TG[name][0].append(np.asarray(v, np.float32))
            TG[name][1].append(np.asarray(m, bool))
    return _pack(T1L, T2L, PHL, RL, META, TG, "B")


def _pack(T1L, T2L, PHL, RL, META, TG, tag):
    tag = tag + SUF
    T1 = np.concatenate(T1L); T2 = np.concatenate(T2L)
    PH = np.concatenate(PHL); R = np.concatenate(RL)
    M = pd.concat(META, ignore_index=True)
    d = dict(T1=T1, T2=T2, PH=PH, R=R.astype(np.float16))
    Y, MK = {}, {}
    for name, (vs, ms) in TG.items():
        Y[name] = np.concatenate(vs)
        MK[name] = np.concatenate(ms)
    # corpus-relative kinematic-extreme definitions (marginal thresholds only)
    for m in [1, 2, 4]:
        jw, jm = Y[f"AUXjminw_m{m}"], MK[f"AUXjminw_m{m}"]
        thr = np.percentile(jw[jm], 2.0)
        Y[f"jext_m{m}"] = (jw < thr).astype(np.float32); MK[f"jext_m{m}"] = jm
        dq, de, vm = Y[f"dqmag_m{m}"], Y[f"AUXde_m{m}"], MK[f"dqmag_m{m}"]
        tq = np.percentile(dq[vm], 90.0)
        te = np.percentile(de[vm], 25.0)
        Y[f"twistq_m{m}"] = ((dq > tq) & (de < te)).astype(np.float32)
        MK[f"twistq_m{m}"] = vm
        print(f"  {tag} m={m} jext thr={thr:.4f} twistq dq>{tq:.3f} de<{te:.4f}")
    for name in list(Y):
        d["y_" + name] = Y[name]
        d["m_" + name] = MK[name]
    np.savez(f"{OUT}/cache_{tag}.npz", **d)
    M.to_parquet(f"{OUT}/meta_{tag}.parquet")
    print(tag, "rows", len(M), "branches", M.episode.nunique(),
          "T1", T1.shape[1], "T2", T2.shape[1], "R", R.shape[1])
    return M, TG


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which == "A":
        prep_A()
    else:
        prep_B()
