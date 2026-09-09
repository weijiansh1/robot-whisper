"""Aggregation-method brute force for the HiMoE-VLA MoE routing tensor.

Axis under study: HOW the four axes of X[t] in R^(8 layer x 10 denoise x 11 token x 32 expert)
are reduced to one scalar per chunk pair.  Held fixed (owned by a sibling agent):
divergence = Hellinger, temporal aggregation = mean over an 8-step window.

Three choices are crossed:
  1. PLACEMENT   -- each of layer/denoise/token is reduced BEFORE the divergence
                    (aggregate the probability vectors, then one divergence) or AFTER
                    (one divergence per element, then aggregate the scalars).  2^3 = 8.
  2. OPERATOR    -- mean / max / min / median / std / q75 at each axis.
  3. AXIS SCOPE  -- layer: all 8 / back 12-15 / front 2-5
                    denoise: all 10 / d9 only
                    token: all 11 / action 1-10 / state only

Canonical reduction order inside the "before" group and inside the "after" group is
layer -> denoise -> token (the order among before-ops is itself non-commutative; a
separate order-sensitivity probe lives in agg_order_probe.py).

Only rows 12..30 of each branch are needed (windows are mean over [t-7..t] for
t in {20,25,30}; row 12 is the predecessor of row 13).  This keeps the working
tensor under the 8 GB cgroup cap.

Never touches hard top-4 expert ids.  Probabilities only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
A_ZARR = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr"
A_LAB = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/analysis/candidate_physical_labels.csv"
B_DIR = f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"

TS = [20, 25, 30]
WIN = 8
T0 = min(TS) - WIN            # 12: first row kept (predecessor only)
SLAB = max(TS) - T0 + 1       # 19 rows per branch
EPS = 1e-12

OPS = ["mean", "max", "min", "median", "std", "q75"]
LAYER_SCOPES = {"all": slice(0, 8), "back": slice(4, 8), "front": slice(0, 4)}
DEN_SCOPES = {"all": slice(0, 10), "d9": slice(9, 10)}
TOK_SCOPES = {"all11": slice(0, 11), "act": slice(1, 11), "state": slice(0, 1)}

LAYER_OPTS = [(s, p, o) for s in LAYER_SCOPES for p in ("B", "A") for o in OPS]
DEN_OPTS = [("all", p, o) for p in ("B", "A") for o in OPS] + [("d9", "-", "id")]
TOK_OPTS = ([(s, p, o) for s in ("all11", "act") for p in ("B", "A") for o in OPS]
            + [("state", "-", "id")])

CFGS = [(l, d, t) for l in LAYER_OPTS for d in DEN_OPTS for t in TOK_OPTS]
CFG_IDX = {c: i for i, c in enumerate(CFGS)}
NCFG = len(CFGS)

BASELINE = (("back", "A", "mean"), ("d9", "-", "id"), ("act", "A", "mean"))


def cfg_name(c):
    (ls, lp, lo), (ds, dp, do), (ts, tp, to) = c
    return f"L[{ls}/{lp}/{lo}]_D[{ds}/{dp}/{do}]_T[{ts}/{tp}/{to}]"


def placement_str(c):
    return "".join(x[1] for x in c)


# ------------------------------------------------------------------ reductions
def _red(X, ax, op):
    if op == "mean":
        return X.mean(ax)
    if op == "max":
        return X.max(ax)
    if op == "min":
        return X.min(ax)
    if op == "median":
        return np.median(X, ax)
    if op == "std":
        return X.std(ax)
    if op == "q75":
        return np.quantile(X, 0.75, axis=ax)
    raise KeyError(op)


def red_prob(X, ax, op):
    """Reduce a probability tensor along `ax`, then re-normalise over the expert axis."""
    Y = np.ascontiguousarray(_red(X, ax, op), dtype=np.float32)
    np.clip(Y, EPS, None, out=Y)
    Y /= Y.sum(-1, keepdims=True)
    return Y


def red_scalar(X, ax, op):
    return np.ascontiguousarray(_red(X, ax, op), dtype=np.float32)


def hell_consec(Q, first_row):
    """Element-wise Hellinger between consecutive rows; nan on branch-starting rows."""
    S = np.sqrt(Q)
    bc = (S[1:] * S[:-1]).sum(-1)
    del S
    H = np.empty(Q.shape[:-1], np.float32)
    H[0] = np.nan
    np.sqrt(np.clip(1.0 - bc, 0.0, None), out=H[1:])
    del bc
    H[first_row] = np.nan
    return H


# ------------------------------------------------------------------ data
def corpus_meta(tag):
    if tag == "A":
        z = zarr.open(store=A_ZARR, mode="r")
        lab = pd.read_csv(A_LAB)
        ep = z["episode_id"][:]
        cs = z["control_step"][:]
        eids = sorted(int(e) for e in lab.episode_id)
        ymap = dict(zip(lab.episode_id.astype(int), lab.success.astype(int)))
        gmap = dict(zip(lab.episode_id.astype(int), lab.worker.astype(int)))
        rows = []
        for e in eids:
            r = np.where(ep == e)[0]
            r = r[np.argsort(cs[r])]
            assert len(r) > max(TS), (e, len(r))
            rows.append(r[T0:T0 + SLAB])
        arr = z["hb_router_probs"]
    else:
        summ = json.load(open(f"{B_DIR}/client/summaries.json"))
        z = zarr.open(store=f"{B_DIR}/server/routes.zarr", mode="r")
        ep = z["episode_id"][:]
        cs = z["control_step"][:]
        eids = list(range(len(summ)))
        ymap = {i: int(bool(s["success"])) for i, s in enumerate(summ)}
        gmap = {i: int(s["init_state_id"]) for i, s in enumerate(summ)}
        rows = []
        for i, s in enumerate(summ):
            r = np.where(ep == i)[0]
            assert len(r) == s["inference_calls"], (i, len(r), s["inference_calls"])
            r = r[np.argsort(cs[r])]
            assert len(r) > max(TS), (i, len(r))
            rows.append(r[T0:T0 + SLAB])
        arr = z["hb_router_probs"]
    return dict(arr=arr, rows=np.array(rows), eids=np.array(eids),
                y=np.array([ymap[e] for e in eids]),
                g=np.array([gmap[e] for e in eids]))


def read_block(arr, rowmat):
    """rowmat: (nb, SLAB) source row ids -> float32 (nb*SLAB, 8,10,11,32) renormalised."""
    flat = rowmat.reshape(-1)
    order = np.argsort(flat, kind="stable")
    srt = flat[order]
    raw = arr.oindex[srt, :, :, :, :]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    P = raw[inv].astype(np.float32)
    del raw
    np.clip(P, EPS, None, out=P)
    P /= P.sum(-1, keepdims=True)
    return P


# ------------------------------------------------------------------ config tree
def block_configs(P, nb, out):
    """Fill out[(nb, NCFG, len(TS))] for one block of branches."""
    first_row = np.zeros(nb * SLAB, bool)
    first_row[::SLAB] = True
    # window row indices inside the slab
    W = np.empty((nb, len(TS), WIN), np.int64)
    for i in range(nb):
        for j, t in enumerate(TS):
            W[i, j] = i * SLAB + np.arange(t - T0 - WIN + 1, t - T0 + 1)

    def emit(cfg, s):
        out[:, CFG_IDX[cfg], :] = s[W].mean(-1)

    layer_nodes = [("A", None)] + [("B", (s, o)) for s in LAYER_SCOPES for o in OPS]
    den_nodes = [("A", None)] + [("B", ("all", o)) for o in OPS]
    tok_nodes = [("A", None)] + [("B", (s, o)) for s in ("all11", "act") for o in OPS]

    done = 0
    for lkind, lb in layer_nodes:
        if lkind == "B":
            ls, lo = lb
            Q1 = red_prob(P[:, LAYER_SCOPES[ls]], 1, lo)
            l_cfgs = [(ls, "B", lo)]
            ax_d, ax_t = 1, 2
        else:
            Q1 = P
            l_cfgs = [(s, "A", o) for s in LAYER_SCOPES for o in OPS]
            ax_d, ax_t = 2, 3

        for dkind, db in den_nodes:
            if dkind == "B":
                ds, do = db
                Q2 = red_prob(Q1[(slice(None),) * ax_d + (DEN_SCOPES[ds],)], ax_d, do)
                d_cfgs = [(ds, "B", do)]
                ax_t2 = ax_t - 1
            else:
                Q2 = Q1
                d_cfgs = [("all", "A", o) for o in OPS] + [("d9", "-", "id")]
                ax_t2 = ax_t

            for tkind, tb in tok_nodes:
                if tkind == "B":
                    ts_, to = tb
                    Q3 = red_prob(Q2[(slice(None),) * ax_t2 + (TOK_SCOPES[ts_],)], ax_t2, to)
                    t_cfgs = [(ts_, "B", to)]
                else:
                    Q3 = Q2
                    t_cfgs = ([(s, "A", o) for s in ("all11", "act") for o in OPS]
                              + [("state", "-", "id")])

                H = hell_consec(Q3, first_row)
                if Q3 is not Q2:
                    del Q3
                pos = 1
                pl = pos if lkind == "A" else None
                pos += (lkind == "A")
                pdn = pos if dkind == "A" else None
                pos += (dkind == "A")
                pt = pos if tkind == "A" else None

                for lc in l_cfgs:
                    HL = H if pl is None else red_scalar(H[:, LAYER_SCOPES[lc[0]]], 1, lc[2])
                    shift_l = 1 if pl is not None else 0
                    for dc in d_cfgs:
                        if pdn is None:
                            HD = HL
                        else:
                            a = pdn - shift_l
                            sl = (slice(None),) * a + (DEN_SCOPES[dc[0]],)
                            HD = red_scalar(HL[sl], a, "mean" if dc[2] == "id" else dc[2])
                        shift_d = shift_l + (1 if pdn is not None else 0)
                        for tc in t_cfgs:
                            if pt is None:
                                s = HD
                            else:
                                a = pt - shift_d
                                sl = (slice(None),) * a + (TOK_SCOPES[tc[0]],)
                                s = red_scalar(HD[sl], a, "mean" if tc[2] == "id" else tc[2])
                            emit((lc, dc, tc), s)
                            done += 1
                del H
                if dkind == "A" and lkind == "A" and tkind == "A":
                    pass
            if Q2 is not Q1:
                del Q2
        if Q1 is not P:
            del Q1
    assert done == NCFG, (done, NCFG)


def sweep(tag, blk=96, verbose=True):
    t0 = time.time()
    M = corpus_meta(tag)
    nbr = len(M["eids"])
    cube = np.full((nbr, NCFG, len(TS)), np.nan, np.float32)
    for i0 in range(0, nbr, blk):
        i1 = min(i0 + blk, nbr)
        P = read_block(M["arr"], M["rows"][i0:i1])
        block_configs(P, i1 - i0, cube[i0:i1])
        del P
        if verbose:
            print(f"[{tag}] branches {i0}-{i1}/{nbr}  {time.time()-t0:.0f}s", flush=True)
    assert not np.isnan(cube).any(), "nan in cube"
    if verbose:
        print(f"[{tag}] done {NCFG} cfgs x {nbr} branches in {time.time()-t0:.0f}s", flush=True)
    return dict(cube=cube, y=M["y"], g=M["g"], eids=M["eids"])


def save(tag, out):
    np.savez(f"/tmp/agg_cube_{tag}.npz", **out)
    print(f"wrote /tmp/agg_cube_{tag}.npz", flush=True)


if __name__ == "__main__":
    import sys
    for tag in (sys.argv[1:] or ["A", "B"]):
        save(tag, sweep(tag))
