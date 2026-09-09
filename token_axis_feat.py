"""Token-axis sweep for HiMoE-VLA MoE routing failure detection.

Extracts per-control-step chunk-level scalars for many *token organizations*
(the axis under study), holding layers = deep block 12-15 (indices 4:8) and
denoise iteration fixed.

Never touches hard top-4 expert ids -- probability vectors only.
"""
import json
import os
import numpy as np
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
A_ZARR = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr"
A_LAB = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/analysis/candidate_physical_labels.csv"
B_DIR = f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"

DEEP = slice(4, 8)          # HB layers 12,13,14,15
NTOK = 11
NEXP = 32
LOG_NEXP = np.log(NEXP)

# ---------------------------------------------------------------- token orgs
TOKEN_SUBSETS = {
    **{f"tok{k:02d}": [k] for k in range(NTOK)},
    "act1_3": [1, 2, 3],
    "act4_7": [4, 5, 6, 7],
    "act8_10": [8, 9, 10],
    "act_all_1_10": list(range(1, 11)),   # <- baseline organization
    "all11": list(range(0, 11)),
}
CROSS_ORGS = ["disp11_hell", "disp11_bhat", "dispAct_hell", "dispAct_bhat",
              "state_vs_actmean_hell"]


def _rows_probs(z, rows, denoise):
    """Return float32 (n, 4, 11, 32) renormalised prob tensor for given rows."""
    a = z["hb_router_probs"]
    out = np.empty((len(rows), 4, NTOK, NEXP), dtype=np.float32)
    B = 2048
    for i in range(0, len(rows), B):
        sl = rows[i:i + B]
        blk = a.oindex[sl, DEEP, denoise, :, :] if False else a[sl.min():sl.max() + 1, DEEP, denoise, :, :]
        out[i:i + len(sl)] = blk[sl - sl.min()].astype(np.float32)
    out = np.clip(out, 1e-12, None)
    out /= out.sum(-1, keepdims=True)
    return out


def per_step_scalars(P):
    """P: (T,4,11,32) float32, time-ordered for ONE branch.

    Returns dict name -> (T,) array.  Entries whose value at step 0 is
    undefined (temporal differences) are set to nan at t=0.
    """
    T = P.shape[0]
    S = np.sqrt(P)                                     # (T,4,11,32)
    out = {}

    # --- normalised entropy per (layer,token) -----------------------------
    ent = -(P * np.log(P)).sum(-1) / LOG_NEXP          # (T,4,11)

    # --- Hellinger to previous control step per (layer,token) -------------
    bc = np.einsum("tlke,tlke->tlk", S[1:], S[:-1])    # (T-1,4,11)
    hell = np.sqrt(np.clip(1.0 - bc, 0.0, None))
    hell = np.concatenate([np.full((1, 4, NTOK), np.nan, np.float32), hell], 0)

    for name, toks in TOKEN_SUBSETS.items():
        out[f"{name}|hell"] = hell[:, :, toks].mean((1, 2))
        out[f"{name}|ent"] = ent[:, :, toks].mean((1, 2))

    # --- cross-token dispersion (per chunk, per layer) --------------------
    # pairwise Bhattacharyya coefficient among tokens: (T,4,11,11)
    G = np.einsum("tlke,tlme->tlkm", S, S)
    G = np.clip(G, 1e-12, 1.0)
    iu = np.triu_indices(NTOK, 1)
    ia = np.triu_indices(10, 1)
    Gall = G[:, :, iu[0], iu[1]]                        # (T,4,55)
    Gact = G[:, :, 1:, 1:][:, :, ia[0], ia[1]]          # (T,4,45)
    lvl = {
        "disp11_hell": np.sqrt(np.clip(1 - Gall, 0, None)).mean((1, 2)),
        "disp11_bhat": (-np.log(Gall)).mean((1, 2)),
        "dispAct_hell": np.sqrt(np.clip(1 - Gact, 0, None)).mean((1, 2)),
        "dispAct_bhat": (-np.log(Gact)).mean((1, 2)),
    }
    # --- state token vs mean action token ---------------------------------
    pact = P[:, :, 1:, :].mean(2)                       # (T,4,32)
    pact = pact / pact.sum(-1, keepdims=True)
    bc0 = (np.sqrt(P[:, :, 0, :]) * np.sqrt(pact)).sum(-1)   # (T,4)
    lvl["state_vs_actmean_hell"] = np.sqrt(np.clip(1 - bc0, 0, None)).mean(1)

    for k, v in lvl.items():
        out[f"{k}|ent"] = v.astype(np.float32)           # "level" slot
        d = np.abs(np.diff(v))
        out[f"{k}|hell"] = np.concatenate(
            [np.array([np.nan], np.float32), d]).astype(np.float32)
    return out


FEATNAMES = ([f"{n}|{q}" for n in TOKEN_SUBSETS for q in ("hell", "ent")] +
             [f"{n}|{q}" for n in CROSS_ORGS for q in ("hell", "ent")])


def build_corpus(zpath, order_rows_per_ep, denoise_list, tag):
    z = zarr.open(store=zpath, mode="r")
    a = z["hb_router_probs"]
    for d in denoise_list:
        cache = f"{ROOT}/.tokaxis_{tag}_d{d}.npz"
        if os.path.exists(cache):
            continue
        # read all needed rows once (float16 -> compact)
        n = a.shape[0]
        buf = np.empty((n, 4, NTOK, NEXP), np.float16)
        B = 4096
        for i in range(0, n, B):
            buf[i:i + B] = a[i:i + B, DEEP, d, :, :]
        feats = {}
        for eid, rows in order_rows_per_ep.items():
            P = buf[rows].astype(np.float32)
            P = np.clip(P, 1e-12, None)
            P /= P.sum(-1, keepdims=True)
            feats[eid] = per_step_scalars(P)
        del buf
        # pack: dict feat -> object array of per-episode series
        eids = sorted(order_rows_per_ep)
        lens = np.array([len(order_rows_per_ep[e]) for e in eids])
        packed = {"__eids__": np.array(eids), "__lens__": lens}
        for f in FEATNAMES:
            packed[f] = np.concatenate([feats[e][f] for e in eids]).astype(np.float32)
        np.savez_compressed(cache, **packed)
        print(f"  wrote {cache}")


def main():
    import pandas as pd
    # ---------------- corpus A ----------------
    za = zarr.open(store=A_ZARR, mode="r")
    ep = za["episode_id"][:]
    cs = za["control_step"][:]
    lab = pd.read_csv(A_LAB)
    keep = set(lab.episode_id.tolist())
    orderA = {}
    for e in np.unique(ep):
        if int(e) not in keep:
            continue
        r = np.where(ep == e)[0]
        orderA[int(e)] = r[np.argsort(cs[r])]
    print("corpus A episodes", len(orderA))
    build_corpus(A_ZARR, orderA, [0, 5, 9], "A")

    # ---------------- corpus B ----------------
    summ = json.load(open(f"{B_DIR}/client/summaries.json"))
    zb = zarr.open(store=f"{B_DIR}/server/routes.zarr", mode="r")
    epb = zb["episode_id"][:]
    csb = zb["control_step"][:]
    orderB = {}
    for i, s in enumerate(summ):
        r = np.where(epb == i)[0]
        assert len(r) == s["inference_calls"], (i, len(r), s["inference_calls"])
        orderB[i] = r[np.argsort(csb[r])]
    print("corpus B episodes", len(orderB))
    build_corpus(f"{B_DIR}/server/routes.zarr", orderB, [0, 5, 9], "B")


if __name__ == "__main__":
    main()
