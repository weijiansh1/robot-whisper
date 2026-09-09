"""Extract fixed slices of hb_router_probs for both corpora and cache as .npy."""
import json, os, sys, time
import numpy as np, pandas as pd, zarr

OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
A_ZARR = "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr"
A_LAB  = "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/analysis/candidate_physical_labels.csv"
B_DIR  = "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"

DEEP = slice(4, 8)     # HB layers 12,13,14,15
FRONT = slice(0, 4)    # HB layers 2,3,4,5
DEN = 9                # denoise step 9 (last)


def extract(zpath, tag):
    z = zarr.open_group(zpath, mode="r")
    P = z["hb_router_probs"]
    N = P.shape[0]
    main = np.empty((N, 4, 10, 32), np.float32)   # deep layers, action tokens 1-10, denoise 9
    front = np.empty((N, 4, 32), np.float32)      # front layers, state token 0, denoise 9
    bs = 1024
    t0 = time.time()
    for s in range(0, N, bs):
        e = min(s + bs, N)
        blk = P[s:e]                              # (b,8,10,11,32) float16
        main[s:e] = blk[:, DEEP, DEN, 1:11, :].astype(np.float32)
        front[s:e] = blk[:, FRONT, DEN, 0, :].astype(np.float32)
    # renormalise (fp16 rounding leaves sums off by ~2e-4)
    main /= main.sum(-1, keepdims=True)
    front /= front.sum(-1, keepdims=True)
    np.save(f"{OUT}/{tag}_main.npy", main)
    np.save(f"{OUT}/{tag}_front.npy", front)
    np.save(f"{OUT}/{tag}_episode_id.npy", z["episode_id"][:])
    np.save(f"{OUT}/{tag}_control_step.npy", z["control_step"][:])
    print(tag, "extracted", main.shape, front.shape, f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    extract(A_ZARR, "A")
    extract(B_DIR + "/server/routes.zarr", "B")

    # --- metadata A ---
    lab = pd.read_csv(A_LAB)
    eidA = np.load(f"{OUT}/A_episode_id.npy")
    csA = np.load(f"{OUT}/A_control_step.npy")
    assert np.array_equal(csA, np.arange(len(csA))), "control_step is not arange"
    metaA = lab[["episode_id", "worker", "success", "inference_calls"]].copy()
    metaA.columns = ["episode_id", "group", "success", "T"]
    metaA["success"] = metaA["success"].astype(bool)
    metaA.to_csv(f"{OUT}/A_meta.csv", index=False)

    # --- metadata B: offsets = cumsum of inference_calls ---
    summ = json.load(open(B_DIR + "/client/summaries.json"))
    df = pd.DataFrame(summ)
    df = df.sort_values("episode_index").reset_index(drop=True)
    off = np.concatenate([[0], np.cumsum(df["inference_calls"].values)])
    Nb = np.load(f"{OUT}/B_main.npy", mmap_mode="r").shape[0]
    print("B rows", Nb, "cumsum", off[-1], "match", off[-1] == Nb)
    eidB = np.load(f"{OUT}/B_episode_id.npy")
    # cross-check against stored episode_id
    derived = np.zeros(Nb, np.int64)
    for i in range(len(df)):
        derived[off[i]:off[i + 1]] = df["episode_index"].values[i]
    print("B derived-vs-stored episode_id agreement:", float((derived == eidB).mean()),
          "n uniq stored", len(np.unique(eidB)))
    metaB = pd.DataFrame({"episode_id": df["episode_index"].values,
                          "group": df["init_state_id"].values,
                          "success": df["success"].values.astype(bool),
                          "T": df["inference_calls"].values,
                          "row0": off[:-1]})
    metaB.to_csv(f"{OUT}/B_meta.csv", index=False)
    print(metaB.groupby("group")["success"].agg(["size", "sum"]))
