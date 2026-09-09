"""Extract per-chunk features and physical-future targets for corpora A and B.

Unit of analysis = (branch, query index k).  Every feature uses only chunks <= k.
Outputs one .npz per corpus into CACHE with:
    P        (N, 2816) float32  routing probs at denoise 9, flattened (8 layer x 11 token x 32 expert)
    T1       (N, d1)   float32  deployable control features
    T2       (N, d2)   float32  privileged control features
    CLOCK    (N, 5)    float32  t, T, t/T, T-t, global_t   (acausal: uses T)
    CLOCKC   (N, 2)    float32  t, global_t                (causal)
    Y_*      (N, 4)    float32  targets, one column per horizon m in (1,2,4,8); nan = undefined
    branch   (N,)      int32    branch index
    group    (N,)      int32    grouping variable (worker / init_state_id)
    kidx     (N,)      int32    query index within branch
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
CACHE = "/tmp/moe_future_cache"
MIRROR = os.path.join(ROOT, "analysis_future/cache")
RUN_A = os.path.join(ROOT, "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828")
RUN_B = os.path.join(
    ROOT,
    "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32",
)
HORIZONS = (1, 2, 4, 8)
DENOISE = 9
GOAL_ENTER = 0.055
GOAL_EXIT = 0.100
GRIP_OPEN = 0.040          # aperture threshold; distribution is bimodal at 0.001 / 0.080
POT = {"p1": slice(10, 13), "p2": slice(17, 20)}


# ---------------------------------------------------------------- routing io

def read_routes(zpath, block=1024):
    """Return (probs float32 (N,2816), episode_id, control_step)."""
    g = zarr.open_group(zpath, mode="r")
    ep = np.asarray(g["episode_id"][:])
    cs = np.asarray(g["control_step"][:])
    arr = g["hb_router_probs"]
    n = arr.shape[0]
    out = np.empty((n, arr.shape[1] * arr.shape[3] * arr.shape[4]), dtype=np.float32)
    for i in range(0, n, block):
        j = min(i + block, n)
        chunk = np.asarray(arr[i:j, :, DENOISE, :, :], dtype=np.float32)
        out[i:j] = chunk.reshape(j - i, -1)
    return out, ep, cs


# ------------------------------------------------------------ goal geometry

def dist_to_refs(pos, refs):
    """pos (T,3), refs (R,3) -> (T,) min distance."""
    return np.linalg.norm(pos[:, None, :] - refs[None, :, :], axis=-1).min(axis=1)


def nearest_ref_delta(pos, refs):
    d = np.linalg.norm(pos[:, None, :] - refs[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    return pos - refs[idx]


def goal_exit_events(dist_series):
    """Hysteresis state machine.  Returns bool array 'exited at this query'."""
    n = len(dist_series)
    ev = np.zeros(n, dtype=bool)
    inside = False
    for i in range(n):
        d = dist_series[i]
        if not inside and d < GOAL_ENTER:
            inside = True
        elif inside and d > GOAL_EXIT:
            inside = False
            ev[i] = True
    return ev


# ------------------------------------------------------------------ targets

TARGET_NAMES = (
    "eef_disp", "eef_path", "obj_disp", "dgoal", "undo", "grip_flips",
    # ---- vision-dependent set: proprioception is blind to these by construction
    "obj_disp_p1", "obj_disp_p2",      # per pot, never pooled
    "which_moves",                      # which pot moves more (nan when neither moves)
    "which_moves_late",                 # same, but only after the scene prior is broken
    "obj_nc", "obj_ncfar",              # pot moves with gripper open and arm >8cm / >12cm away
    "obj_free_disp",                    # metres of such no-contact pot motion
    "obj_gt_arm",                       # pot moves further than the arm did
    "undo_nc",                          # goal exit while the arm is not in contact
)
FAR = 0.08          # eef-pot distance; median distance while a pot is being moved is 0.064 m
FAR2 = 0.12
MOVE = 0.005        # per-query pot displacement that counts as "moving"
PRIOR_BROKEN = 0.05  # pot displacement from its own initial position


def make_targets(eef, pots, goal_agg, aperture):
    """All per-(k,m) targets.  eef (T,3); pots (T,2,3); goal_agg (T,); aperture (T,)."""
    T = len(eef)
    eef_step = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1)]
    eef_cum = np.cumsum(eef_step)
    grip_open = (aperture > GRIP_OPEN).astype(np.int8)
    grip_flip = np.r_[0, (np.diff(grip_open) != 0).astype(np.int8)]
    grip_flip_cum = np.cumsum(grip_flip)
    # ---- contact geometry, per pot, per query transition j-1 -> j
    d_ep = np.linalg.norm(eef[:, None, :] - pots, axis=2)          # (T,2)
    op = aperture > GRIP_OPEN
    pot_step = np.linalg.norm(np.diff(pots, axis=0), axis=2)        # (T-1,2)
    moving = pot_step > MOVE
    no_contact = (d_ep[1:] > FAR) & (d_ep[:-1] > FAR) & op[1:, None] & op[:-1, None]
    clear = (d_ep[1:].min(1) > FAR2) & (d_ep[:-1].min(1) > FAR2) & op[1:] & op[:-1]
    nc_move = moving & no_contact                                  # (T-1,2)
    ncfar_move = moving & clear[:, None]

    exits2 = np.stack([goal_exit_events(dist_to_refs(pots[:, o, :], REFS))
                       for o in range(2)], axis=1)                  # (T,2)
    exits = exits2.any(axis=1)
    exit_nc = (exits2[1:] & no_contact).any(axis=1)
    exit_cum = np.cumsum(exits.astype(np.int32))
    exit_nc_cum = np.r_[0, np.cumsum(exit_nc.astype(np.int32))]
    nc_cum = np.r_[0, np.cumsum(nc_move.any(1).astype(np.int32))]
    ncfar_cum = np.r_[0, np.cumsum(ncfar_move.any(1).astype(np.int32))]
    free_cum = np.vstack([np.zeros((1, 2), np.float32),
                          np.cumsum(pot_step * nc_move, axis=0)])   # (T,2)
    prior_broken = np.linalg.norm(pots - pots[:1], axis=2).max(1) > PRIOR_BROKEN

    out = {k: np.full((T, len(HORIZONS)), np.nan, np.float32) for k in TARGET_NAMES}
    for mi, m in enumerate(HORIZONS):
        k = np.arange(T)
        kk = k[k + m <= T - 1]
        j = kk + m
        dp = np.linalg.norm(pots[j] - pots[kk], axis=2)              # (n,2)
        de = np.linalg.norm(eef[j] - eef[kk], axis=1)
        out["eef_disp"][kk, mi] = de
        out["eef_path"][kk, mi] = eef_cum[j] - eef_cum[kk]
        out["obj_disp"][kk, mi] = dp.max(axis=1)
        out["obj_disp_p1"][kk, mi] = dp[:, 0]
        out["obj_disp_p2"][kk, mi] = dp[:, 1]
        out["dgoal"][kk, mi] = goal_agg[j] - goal_agg[kk]
        out["undo"][kk, mi] = (exit_cum[j] - exit_cum[kk] > 0).astype(np.float32)
        out["undo_nc"][kk, mi] = (exit_nc_cum[j] - exit_nc_cum[kk] > 0).astype(np.float32)
        out["grip_flips"][kk, mi] = grip_flip_cum[j] - grip_flip_cum[kk]
        out["obj_nc"][kk, mi] = (nc_cum[j] - nc_cum[kk] > 0).astype(np.float32)
        out["obj_ncfar"][kk, mi] = (ncfar_cum[j] - ncfar_cum[kk] > 0).astype(np.float32)
        out["obj_free_disp"][kk, mi] = (free_cum[j] - free_cum[kk]).max(axis=1)
        out["obj_gt_arm"][kk, mi] = (dp.max(axis=1) > de).astype(np.float32)
        valid = dp.max(axis=1) > MOVE
        wm = np.where(valid, (dp[:, 1] > dp[:, 0]).astype(np.float32), np.nan)
        out["which_moves"][kk, mi] = wm
        out["which_moves_late"][kk, mi] = np.where(prior_broken[kk], wm, np.nan)
    return out


# ----------------------------------------------------------------- features

def lag(x, n=1):
    """Causal lag: row k gets row k-n (row 0 repeats itself)."""
    y = np.empty_like(x)
    y[n:] = x[:-n]
    y[:n] = x[:n][:1] if x.ndim == 1 else x[:1]
    return y


def chunk_summary(A):
    """A (T,10,7) planned/executed action chunk -> derived summary."""
    T = A.shape[0]
    pos = A[:, :, :3]
    rot = A[:, :, 3:6]
    grip = A[:, :, 6]
    cum = pos.cumsum(axis=1)
    feats = [
        A.reshape(T, -1),                                    # 70 raw
        cum[:, -1, :],                                       # 3  net commanded translation
        np.linalg.norm(pos, axis=2),                         # 10 per-step magnitude
        np.linalg.norm(pos, axis=2).sum(axis=1, keepdims=True),   # 1 path
        np.linalg.norm(cum[:, -1, :], axis=1, keepdims=True),     # 1 net norm
        rot.sum(axis=1),                                     # 3 net rotation
        np.linalg.norm(rot, axis=2).sum(axis=1, keepdims=True),   # 1
        grip.mean(axis=1, keepdims=True),                    # 1
        grip[:, :1], grip[:, -1:],                           # 2
        (np.diff(np.sign(grip), axis=1) != 0).sum(axis=1, keepdims=True).astype(np.float32),
    ]
    return np.concatenate(feats, axis=1).astype(np.float32), cum[:, -1, :]


def build_branch(state, sim, A):
    """state (T,8) [eef3, aa3, g1, g2]; sim (T,47); A (T,10,7)."""
    T = len(state)
    eef = state[:, :3]
    aa = state[:, 3:6]
    aperture = state[:, 6] - state[:, 7]
    pots = np.stack([sim[:, POT["p1"]], sim[:, POT["p2"]]], axis=1)      # (T,2,3)

    d_ref_pot = np.stack([dist_to_refs(pots[:, o, :], REFS) for o in range(2)], axis=1)
    goal_agg = d_ref_pot.max(axis=1)                       # aggregate = max over pots
    goal_mean = d_ref_pot.mean(axis=1)
    d_ref_eef = dist_to_refs(eef, REFS)

    chunk_f, net_cmd = chunk_summary(A)
    eef_end = eef + net_cmd                                 # planned chunk endpoint (control-side)

    # ---- tier 1 : proprio + own action chunk + eef/goal-landmark geometry
    t1_now = np.concatenate([
        state,                                                  # 8 eef pos/axis-angle/fingers
        aperture[:, None],
        d_ref_eef[:, None],                                     # eef -> nearest goal landmark
        nearest_ref_delta(eef, REFS),                           # 3 signed
        np.linalg.norm(aa, axis=1, keepdims=True),
        (d_ref_eef * aperture)[:, None],
    ], axis=1).astype(np.float32)
    t1 = np.concatenate([
        t1_now, lag(t1_now), t1_now - lag(t1_now), t1_now - lag(t1_now, 2),
        chunk_f, lag(chunk_f),
        eef_end, dist_to_refs(eef_end, REFS)[:, None],
        (dist_to_refs(eef_end, REFS) - d_ref_eef)[:, None],
    ], axis=1).astype(np.float32)

    # ---- tier 2 : full sim_state + full relative geometry (+ everything in tier 1)
    p1, p2 = pots[:, 0, :], pots[:, 1, :]
    g2_now = np.concatenate([
        sim,                                                       # 47
        p1, p2,
        eef - p1, eef - p2, p1 - p2,
        np.linalg.norm(eef - p1, axis=1, keepdims=True),
        np.linalg.norm(eef - p2, axis=1, keepdims=True),
        np.linalg.norm(p1 - p2, axis=1, keepdims=True),
        d_ref_pot,                                                 # 2
        goal_agg[:, None], goal_mean[:, None], d_ref_eef[:, None],
        nearest_ref_delta(p1, REFS), nearest_ref_delta(p2, REFS),
        np.minimum(np.linalg.norm(eef - p1, axis=1),
                   np.linalg.norm(eef - p2, axis=1))[:, None],
        (np.linalg.norm(eef - p1, axis=1) <
         np.linalg.norm(eef - p2, axis=1)).astype(np.float32)[:, None],
        (aperture * np.linalg.norm(eef - p1, axis=1))[:, None],
        (aperture * np.linalg.norm(eef - p2, axis=1))[:, None],
        (eef[:, 2] - p1[:, 2])[:, None], (eef[:, 2] - p2[:, 2])[:, None],
    ], axis=1).astype(np.float32)
    t2 = np.concatenate([
        t1,
        g2_now, lag(g2_now), g2_now - lag(g2_now), g2_now - lag(g2_now, 2),
        np.linalg.norm(eef_end - p1, axis=1, keepdims=True),
        np.linalg.norm(eef_end - p2, axis=1, keepdims=True),
        dist_to_refs(eef_end, REFS)[:, None],
        (np.linalg.norm(eef_end - p1, axis=1)
         - np.linalg.norm(eef - p1, axis=1))[:, None],
        (np.linalg.norm(eef_end - p2, axis=1)
         - np.linalg.norm(eef - p2, axis=1))[:, None],
    ], axis=1).astype(np.float32)

    targets = make_targets(eef, pots, goal_agg, aperture)
    return t1, t2, targets, T


def eef_ref_delta(eef):
    return nearest_ref_delta(eef, REFS)



# ------------------------------------------------------------------ corpus A

def load_A():
    global REFS
    lab = pd.read_csv(os.path.join(RUN_A, "analysis/candidate_physical_labels.csv"))
    defs = json.load(open(os.path.join(
        RUN_A, "analysis_dryrun_v10/physical_label_definitions.json")))
    REFS = np.asarray(defs["success_terminal_references"]["moka_pot_1_joint0"], np.float32)
    print(f"[A] {len(lab)} branches, {len(REFS)} goal references")

    probs, ep, cs = read_routes(os.path.join(RUN_A, "formal/server/routes.zarr"))
    order = np.argsort(cs, kind="stable")
    ep, probs = ep[order], probs[order]
    row_of = {}
    for e in np.unique(ep):
        row_of[int(e)] = np.flatnonzero(ep == e)

    P, T1, T2, CL, CLC, BR, GR, KI = [], [], [], [], [], [], [], []
    TG = {k: [] for k in TARGET_NAMES}
    for bi, r in lab.reset_index(drop=True).iterrows():
        f = os.path.join(RUN_A, f"formal/worker{r.worker}",
                         f"snapshot_{int(r.snapshot):03d}",
                         f"candidate_{int(r.candidate):02d}.npz")
        z = np.load(f)
        state = z["policy_state"].astype(np.float32)
        sim = z["sim_state"].astype(np.float32)
        A = z["action_chunks"].astype(np.float32)
        rows = row_of[int(r.episode_id)]
        T = min(len(state), len(rows))
        state, sim, A, rows = state[:T], sim[:T], A[:T], rows[:T]
        t1, t2, tg, T = build_branch(state, sim, A)
        P.append(probs[rows])
        T1.append(t1); T2.append(t2)
        t = np.arange(T, dtype=np.float32)
        gt = t + int(r.snapshot)          # snapshot offset = global phase
        CL.append(np.stack([t, np.full(T, T, np.float32), t / T,
                            T - t, gt], axis=1))
        CLC.append(np.stack([t, gt], axis=1))
        BR.append(np.full(T, bi, np.int32))
        GR.append(np.full(T, int(r.worker), np.int32))
        KI.append(t.astype(np.int32))
        for k in TG:
            TG[k].append(tg[k])
        del z
    return dict(P=np.concatenate(P), T1=np.concatenate(T1), T2=np.concatenate(T2),
                CLOCK=np.concatenate(CL).astype(np.float32),
                CLOCKC=np.concatenate(CLC).astype(np.float32),
                branch=np.concatenate(BR), group=np.concatenate(GR),
                kidx=np.concatenate(KI),
                success=lab.success.values.astype(np.int8),
                **{f"Y_{k}": np.concatenate(v) for k, v in TG.items()})


# ------------------------------------------------------------------ corpus B

def load_B():
    global REFS
    summ = json.load(open(os.path.join(RUN_B, "client/summaries.json")))
    ic = np.array([e["inference_calls"] for e in summ])
    off = np.r_[0, np.cumsum(ic)]
    # goal references: pooled terminal pot positions of successful episodes
    refs = []
    for i, e in enumerate(summ):
        if not e["success"]:
            continue
        z = np.load(os.path.join(RUN_B, f"client/episode_{i:02d}.npz"), allow_pickle=True)
        s = z["sim_state"]
        refs.append(s[-1, POT["p1"]]); refs.append(s[-1, POT["p2"]])
    REFS = np.asarray(refs, np.float32)
    print(f"[B] {len(summ)} episodes, {len(REFS)} goal references")

    probs, ep, cs = read_routes(os.path.join(RUN_B, "server/routes.zarr"))

    P, T1, T2, CL, CLC, BR, GR, KI = [], [], [], [], [], [], [], []
    TG = {k: [] for k in TARGET_NAMES}
    for i, e in enumerate(summ):
        z = np.load(os.path.join(RUN_B, f"client/episode_{i:02d}.npz"), allow_pickle=True)
        state = z["state"].astype(np.float32)
        sim = z["sim_state"].astype(np.float32)
        A = z["actions"].astype(np.float32)
        rows = np.arange(off[i], off[i + 1])
        T = min(len(state), len(rows))
        state, sim, A, rows = state[:T], sim[:T], A[:T], rows[:T]
        t1, t2, tg, T = build_branch(state, sim, A)
        P.append(probs[rows]); T1.append(t1); T2.append(t2)
        t = np.arange(T, dtype=np.float32)
        CL.append(np.stack([t, np.full(T, T, np.float32), t / T, T - t, t], axis=1))
        CLC.append(np.stack([t, t], axis=1))
        BR.append(np.full(T, i, np.int32))
        GR.append(np.full(T, int(e["init_state_id"]), np.int32))
        KI.append(t.astype(np.int32))
        for k in TG:
            TG[k].append(tg[k])
    return dict(P=np.concatenate(P), T1=np.concatenate(T1), T2=np.concatenate(T2),
                CLOCK=np.concatenate(CL).astype(np.float32),
                CLOCKC=np.concatenate(CLC).astype(np.float32),
                branch=np.concatenate(BR), group=np.concatenate(GR),
                kidx=np.concatenate(KI),
                success=np.array([e["success"] for e in summ], np.int8),
                **{f"Y_{k}": np.concatenate(v) for k, v in TG.items()})


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(MIRROR, exist_ok=True)
    which = sys.argv[1]
    d = load_A() if which == "A" else load_B()
    for k, v in d.items():
        print(f"  {k:14s} {v.shape} {v.dtype}")
    out = os.path.join(CACHE, f"corpus{which}.npz")
    np.savez(out, **d)
    print("wrote", out, os.path.getsize(out) / 1e6, "MB")
