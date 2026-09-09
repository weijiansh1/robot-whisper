"""One pass over routes.zarr for fourteen quantities nobody has computed.

Existing work mines mobility, flow speed / path, entropies, Schur-complement
geometry, top-k churn, recurrence and spectra -- all of them *amplitude*
statistics of a single summarised routing vector.  This extracts four families
that are structurally different, all read at the final denoising step s=9 (v7's
own reference step) unless noted:

SET STRUCTURE (from `hb_expert_ids`, the actual top-4 selection).  Expert index
is permutation-symmetric, so only set operations are meaningful.  Crucially the
gain and the loss are *not* the same number once they are weighted by mass:
    set_jacc_adj    |S(q) & S(q-1)| / |S(q) | S(q-1)|
    set_inflow      mass at q sitting on experts that were NOT selected at q-1
    set_outflow     mass at q-1 sitting on experts NOT selected at q
    set_asym        inflow - outflow   (signed: expanding vs retreating)
    set_novel_open  |S(q) \\ union of S over the opening chunks 1..3| / 4

RANK STRUCTURE (from `hb_router_probs`, magnitudes discarded).  A permutation
of the probability values that preserves their order leaves these unchanged, so
they cannot be a repackaging of any amplitude head:
    rank_footrule_adj   normalised Spearman footrule between q and q-1
    rank_footrule_open  the same against the episode's own opening rank profile

TOKEN-JOINT STRUCTURE (the ten action tokens as a whole, not summarised):
    tok_disp        mean pairwise Hellinger among the ten action tokens
    tok_near_far    Hellinger between the near (t1..t3) and far (t8..t10) ends
                    of the action chunk -- does routing disagree about the far
                    future more than the near future?
    tok_effrank     participation ratio of the singular values of the centred
                    10 x 32 token-expert matrix: the effective number of
                    distinct routing modes the ten tokens occupy

STATE / LOAD:
    state_action    Hellinger(state token 0, mean action token)
    load_top4mass   selected mass (sum of `hb_selected_prob`)
    load_pr         1 / sum p^2 of the mean action distribution
    step_endpoint   Hellinger(s=0, s=9) -- net denoising displacement, the
                    chord that flow_speed's path length integrates over

Everything is per (episode, chunk, layer) and written as [n, 52, 8] float32,
aligned row-for-row with the published mobility caches.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V4 = ROOT / "moe-v4-0904/results/layerwise_mobility"

COHORTS = {
    "development_main": {"cache": V4 / "main_reference.npz",
                         "root": ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
                         "run_id": None},
    "external_8b": {"cache": V4 / "external_8b.npz",
                    "root": ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
                    "run_id": None},
    "legacy_main16x32": {
        "cache": ROOT / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
        "root": ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA", "run_id": "right-16x32"},
}

STEP = 9          # final denoising step, v7's reference
OPEN_LO, OPEN_HI = 1, 4    # the episode's own opening regime, chunks 1..3
NAMES = ("set_jacc_adj", "set_inflow", "set_outflow", "set_asym",
         "set_novel_open", "rank_footrule_adj", "rank_footrule_open",
         "tok_disp", "tok_near_far", "tok_effrank", "state_action",
         "load_top4mass", "load_pr", "step_endpoint",
         "state_mob", "state_top4mass", "state_set_jacc")


def _norm(p):
    p = np.clip(p.astype(np.float32), 0.0, None)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def _hell(sa, sb):
    """Hellinger from sqrt-probabilities."""
    return np.sqrt(np.clip(1.0 - (sa * sb).sum(-1), 0.0, 1.0))


def episode_features(probs, sel, ids, probs_s0):
    """probs [T,8,11,32] at s=9, sel [T,8,11,4], ids [T,8,11,4],
    probs_s0 [T,8,11,32] at s=0.  Returns {name: [T,8] float32}."""
    T = probs.shape[0]
    p = _norm(probs)                       # [T,8,11,32]
    pa = p[:, :, 1:, :]                    # action tokens [T,8,10,32]
    sq = np.sqrt(pa)
    out = {}

    # --- token-joint -----------------------------------------------------
    gram = np.einsum("tlie,tlje->tlij", sq, sq)            # Bhattacharyya
    hell = np.sqrt(np.clip(1.0 - gram, 0.0, 1.0))
    iu = np.triu_indices(10, k=1)
    out["tok_disp"] = hell[:, :, iu[0], iu[1]].mean(-1)

    near = _norm(pa[:, :, 0:3, :].sum(2))
    far = _norm(pa[:, :, 7:10, :].sum(2))
    out["tok_near_far"] = _hell(np.sqrt(near), np.sqrt(far))

    centred = pa - pa.mean(2, keepdims=True)
    g = np.einsum("tlie,tlje->tlij", centred, centred)     # [T,8,10,10]
    ev = np.linalg.eigvalsh(g.astype(np.float64))
    ev = np.clip(ev, 0.0, None)
    s1 = np.sqrt(ev).sum(-1)
    s2 = ev.sum(-1)
    out["tok_effrank"] = np.where(s2 > 1e-18, s1 ** 2 / np.maximum(s2, 1e-18),
                                  np.nan).astype(np.float32)

    # --- state / load ----------------------------------------------------
    abar = _norm(pa.sum(2))
    out["state_action"] = _hell(np.sqrt(_norm(p[:, :, 0, :])), np.sqrt(abar))
    out["load_top4mass"] = sel[:, :, 1:, :].astype(np.float32).sum(-1).mean(-1)
    out["load_pr"] = 1.0 / np.maximum((abar ** 2).sum(-1), 1e-12)

    p0 = _norm(probs_s0)[:, :, 1:, :]
    out["step_endpoint"] = _hell(np.sqrt(p0), sq).mean(-1)

    # --- support sets ----------------------------------------------------
    mask = np.zeros((T, 8, 10, 32), dtype=bool)
    idx = ids[:, :, 1:, :].astype(np.int64)
    np.put_along_axis(mask, idx, True, axis=-1)

    jacc = np.full((T, 8), np.nan, np.float32)
    inflow = np.full((T, 8), np.nan, np.float32)
    outflow = np.full((T, 8), np.nan, np.float32)
    if T > 1:
        inter = (mask[1:] & mask[:-1]).sum(-1).astype(np.float32)
        uni = (mask[1:] | mask[:-1]).sum(-1).astype(np.float32)
        jacc[1:] = (inter / np.maximum(uni, 1e-9)).mean(-1)
        inflow[1:] = (pa[1:] * ~mask[:-1]).sum(-1).mean(-1)
        outflow[1:] = (pa[:-1] * ~mask[1:]).sum(-1).mean(-1)
    out["set_jacc_adj"] = jacc
    out["set_inflow"] = inflow
    out["set_outflow"] = outflow
    out["set_asym"] = inflow - outflow

    hi = min(OPEN_HI, T)
    novel = np.full((T, 8), np.nan, np.float32)
    if hi > OPEN_LO:
        seen = mask[OPEN_LO:hi].any(0)                     # [8,10,32]
        novel[:] = (mask & ~seen[None]).sum(-1).mean(-1) / 4.0
        novel[:hi] = np.nan
    out["set_novel_open"] = novel

    # --- ranks (magnitudes discarded) ------------------------------------
    order = np.argsort(pa, axis=-1)
    rank = np.empty_like(order, dtype=np.int16)
    np.put_along_axis(rank, order,
                      np.broadcast_to(np.arange(32, dtype=np.int16),
                                      order.shape).copy(), axis=-1)
    rf_adj = np.full((T, 8), np.nan, np.float32)
    if T > 1:
        rf_adj[1:] = (np.abs(rank[1:].astype(np.float32)
                             - rank[:-1]).sum(-1).mean(-1) / 512.0)
    out["rank_footrule_adj"] = rf_adj

    rf_open = np.full((T, 8), np.nan, np.float32)
    if hi > OPEN_LO:
        ref = rank[OPEN_LO:hi].astype(np.float32).mean(0)   # [8,10,32]
        rf_open[:] = (np.abs(rank.astype(np.float32) - ref[None])
                      .sum(-1).mean(-1) / 512.0)
        rf_open[:hi] = np.nan
    out["rank_footrule_open"] = rf_open

    # --- state token (token 0) in its own right --------------------------
    # v7's mobility primitive slices `[..., 1:, :]`, so the state token has
    # never entered any mobility-family head -- yet it is the only token that
    # routes sharply (entropy ~1.7-3.0 against ~3.45 for action tokens).
    ps = _norm(p[:, :, 0, :])
    sqs = np.sqrt(ps)
    smob = np.full((T, 8), np.nan, np.float32)
    sjac = np.full((T, 8), np.nan, np.float32)
    if T > 1:
        smob[1:] = _hell(sqs[1:], sqs[:-1])
        m0 = np.zeros((T, 8, 32), dtype=bool)
        np.put_along_axis(m0, ids[:, :, 0, :].astype(np.int64), True, axis=-1)
        inter0 = (m0[1:] & m0[:-1]).sum(-1).astype(np.float32)
        uni0 = (m0[1:] | m0[:-1]).sum(-1).astype(np.float32)
        sjac[1:] = inter0 / np.maximum(uni0, 1e-9)
    out["state_mob"] = smob
    out["state_set_jacc"] = sjac
    out["state_top4mass"] = sel[:, :, 0, :].astype(np.float32).sum(-1)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def run_task(job):
    cohort, path, rows, episodes, max_query = job
    res = {n: {} for n in NAMES}
    group = zarr.open_group(path, mode="r")
    # read once per run: zarr chunks span the whole (layer, step, token,
    # expert) block, so per-episode fancy slicing would re-decompress it.
    _p = group["hb_router_probs"]
    probs = np.stack([np.asarray(_p[:, :, 0]), np.asarray(_p[:, :, STEP])],
                     axis=2)
    sel = np.asarray(group["hb_selected_prob"][:, :, STEP])
    ids = np.asarray(group["hb_expert_ids"][:, :, STEP])
    raw_episode = np.asarray(group["episode_id"][:], dtype=int)
    starts = np.flatnonzero(np.r_[True, np.diff(raw_episode) != 0])
    ends = np.r_[starts[1:], len(raw_episode)]
    block = {int(raw_episode[s]): (int(s), int(e))
             for s, e in zip(starts, ends)}
    for row, ep in zip(rows, episodes):
        if ep not in block:
            continue
        lo, hi = block[ep]
        if hi - lo < 2:
            continue
        feats = episode_features(probs[lo:hi, :, 1], sel[lo:hi], ids[lo:hi],
                                 probs[lo:hi, :, 0])
        q = min(hi - lo, max_query)
        for n in NAMES:
            res[n][int(row)] = feats[n][:q]
    return cohort, res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    jobs, meta = [], {}
    for cohort, cfg in COHORTS.items():
        if only and cohort != only:
            continue
        with np.load(cfg["cache"], allow_pickle=False) as archive:
            cache = {k: np.asarray(archive[k]) for k in archive.files}
        task_names = cache["task_names"].astype(str)
        task_index = cache["task_index"].astype(int)
        episodes = cache["episode"].astype(int)
        max_query = cache["valid"].shape[1]
        run_id = cfg["run_id"] or str(cache["run_id"])
        meta[cohort] = (len(episodes), max_query)
        for position, task in enumerate(task_names):
            rows = np.flatnonzero(task_index == position)
            path = cfg["root"] / task / run_id / "server/routes.zarr"
            if not path.is_dir():
                raise FileNotFoundError(path)
            jobs.append((cohort, str(path), rows.tolist(),
                         [int(episodes[r]) for r in rows], max_query))
    print("jobs: %d" % len(jobs), flush=True)

    store = {c: {n: np.full((meta[c][0], meta[c][1], 8), np.nan, np.float32)
                 for n in NAMES} for c in meta}
    t0 = time.time()
    with Pool(processes=min(24, len(jobs))) as pool:
        for k, (cohort, res) in enumerate(
                pool.imap_unordered(run_task, jobs), 1):
            for n, per_row in res.items():
                arr = store[cohort][n]
                for row, vals in per_row.items():
                    arr[row, :len(vals)] = vals
            print("  %3d/%d  %s  %.0fs" % (k, len(jobs), cohort,
                                           time.time() - t0), flush=True)

    for cohort, d in store.items():
        np.savez_compressed(OUT / f"{cohort}_rawq.npz", **d)
        finite = {n: float(np.isfinite(v).mean()) for n, v in d.items()}
        print("wrote %s %s finite=%s" % (cohort, d[NAMES[0]].shape,
                                         {k: round(v, 3)
                                          for k, v in finite.items()}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
