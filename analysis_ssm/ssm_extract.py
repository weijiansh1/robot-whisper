"""Extract the FULL denoise-9 routing tensor (8 layers x 11 tokens x 32 experts)
and the executed action chunks, for both corpora, into dense per-branch arrays.

Outputs (in analysis_ssm/cache):
  {tag}_full.npy    (N, 8, 11, 32) float32   denoise step 9, renormalised
  {tag}_act.npy     (B, Tmax, 10, 7) float32 action chunks, 0 where invalid
  {tag}_rowidx.npy  (B, Tmax) int32          row into _full for each (branch, k), -1 invalid
  {tag}_meta.csv    episode_id, group, success, T
"""
import json
import os
import time

import numpy as np
import pandas as pd
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
OUT = f"{ROOT}/analysis_ssm/cache"
A_RUN = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
A_ZARR = f"{A_RUN}/formal/server/routes.zarr"
A_LAB = f"{A_RUN}/analysis/candidate_physical_labels.csv"
B_DIR = (f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
         f"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
DEN = 9


def extract_routes(zpath, tag):
    z = zarr.open_group(zpath, mode="r")
    P = z["hb_router_probs"]
    N = P.shape[0]
    full = np.empty((N, 8, 11, 32), np.float32)
    bs = 512
    t0 = time.time()
    for s in range(0, N, bs):
        e = min(s + bs, N)
        full[s:e] = P[s:e, :, DEN, :, :].astype(np.float32)
    full /= full.sum(-1, keepdims=True)
    np.save(f"{OUT}/{tag}_full.npy", full)
    eid = np.asarray(z["episode_id"][:])
    np.save(f"{OUT}/{tag}_episode_id.npy", eid)
    print(tag, "routes", full.shape, f"{time.time()-t0:.1f}s", flush=True)
    return eid, N


def row_order(eid):
    """episode_id -> ascending global row indices (control_step is a global arange)."""
    srt = np.argsort(eid, kind="stable")
    e_srt = eid[srt]
    uq = np.unique(e_srt)
    st = np.searchsorted(e_srt, uq, side="left")
    en = np.append(st[1:], len(e_srt))
    return {int(u): np.sort(srt[s:e]) for u, s, e in zip(uq, st, en)}


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- corpus A ----------------
    eidA, NA = extract_routes(A_ZARR, "A")
    lab = pd.read_csv(A_LAB)
    metaA = lab[["episode_id", "worker", "success", "inference_calls",
                 "snapshot", "candidate"]].copy()
    metaA = metaA.rename(columns={"worker": "group", "inference_calls": "T"})
    metaA["success"] = metaA["success"].astype(bool)
    ordA = row_order(eidA)
    Tmax = int(metaA["T"].max())
    actA = np.zeros((len(metaA), Tmax, 10, 7), np.float32)
    rowA = np.full((len(metaA), Tmax), -1, np.int32)
    for i, r in metaA.iterrows():
        T = int(r["T"])
        rows = ordA[int(r["episode_id"])]
        assert len(rows) == T, ("A", r["episode_id"], len(rows), T)
        rowA[i, :T] = rows
        p = (f"{A_RUN}/formal/worker{int(r['group'])}/snapshot_{int(r['snapshot']):03d}"
             f"/candidate_{int(r['candidate']):02d}.npz")
        ac = np.load(p)["action_chunks"]
        assert ac.shape == (T, 10, 7), ("A act", p, ac.shape, T)
        actA[i, :T] = ac
    np.save(f"{OUT}/A_act.npy", actA)
    np.save(f"{OUT}/A_rowidx.npy", rowA)
    metaA[["episode_id", "group", "success", "T"]].to_csv(f"{OUT}/A_meta.csv", index=False)
    print("A done", actA.shape, rowA.shape, flush=True)

    # ---------------- corpus B ----------------
    eidB, NB = extract_routes(B_DIR + "/server/routes.zarr", "B")
    summ = pd.DataFrame(json.load(open(B_DIR + "/client/summaries.json")))
    summ = summ.sort_values("episode_index").reset_index(drop=True)
    off = np.concatenate([[0], np.cumsum(summ["inference_calls"].values)])
    assert off[-1] == NB, ("B rows", off[-1], NB)
    metaB = pd.DataFrame({"episode_id": summ["episode_index"].values,
                          "group": summ["init_state_id"].values,
                          "success": summ["success"].values.astype(bool),
                          "T": summ["inference_calls"].values})
    Tmax = int(metaB["T"].max())
    actB = np.zeros((len(metaB), Tmax, 10, 7), np.float32)
    rowB = np.full((len(metaB), Tmax), -1, np.int32)
    for i in range(len(metaB)):
        T = int(metaB["T"].iloc[i])
        rowB[i, :T] = np.arange(off[i], off[i + 1])
        ep = int(metaB["episode_id"].iloc[i])
        z = np.load(f"{B_DIR}/client/episode_{ep:02d}.npz", allow_pickle=True)
        ac = z["actions"]
        assert ac.shape == (T, 10, 7), ("B act", ep, ac.shape, T)
        actB[i, :T] = ac
    np.save(f"{OUT}/B_act.npy", actB)
    np.save(f"{OUT}/B_rowidx.npy", rowB)
    metaB.to_csv(f"{OUT}/B_meta.csv", index=False)
    print("B done", actB.shape, rowB.shape, flush=True)


if __name__ == "__main__":
    main()
