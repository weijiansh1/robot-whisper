#!/usr/bin/env python3
"""Do the routing states of successful and failed rollouts differ?  libero_10/t08.

The question is asked of one task, 512 episodes, 16 initial states x 32 flow-noise
draws, so the two obvious traps are both live:

  scene difficulty   base rates run from 0/32 (s49) to 32/32 (s33, s36).  Pooling
                     across scenes measures which scene an episode is in, not
                     what its routing did.  Every test here is stratified: the
                     comparison happens inside a scene and is then pooled, and
                     the null is a within-scene label shuffle.

  episode length     every one of the 216 failures ran the full 52 inference
                     calls; the shortest success took 35.  So "how long did it
                     run" separates the classes perfectly and is not a finding.
                     All 512 episodes are still running at control step 34, so
                     steps 0..34 are compared at a fixed index with no
                     survivorship and no episode having yet ended.

Three scenes are unusable for a paired test and are dropped by construction, not
by choice: s33 and s36 (32/32) and s49 (0/32) contribute no discordant pair.
That leaves 13 scenes, 296 successes and 216 failures.

Reported statistic is the stratified AUC (Mantel-Haenszel pooled Mann-Whitney):
0.5 is no signal, and its null is obtained by shuffling the outcome labels within
each scene.  Because the feature x step grid is scanned, the null is taken as the
max |AUC-0.5| over the whole grid per shuffle, which controls the family-wise
error rather than reporting the best cell of a few hundred.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
T_MAX = 35          # steps 0..34, the window where all 512 episodes are alive
N_PERM = 2000
RNG = np.random.default_rng(20260817)


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- scipy is not installed in this env."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average over tie groups
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def strat_auc(x: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> float:
    """Pooled within-scene AUC of x for outcome y.  x may be [N] or [N, F]."""
    x = np.atleast_2d(x.T).T
    num = np.zeros(x.shape[1])
    den = 0.0
    for m in groups:
        yy = y[m]
        n1 = int(yy.sum())
        n0 = int((~yy).sum())
        for f in range(x.shape[1]):
            r = rankdata(x[m, f])
            num[f] += r[yy].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    out = num / den
    return out if out.size > 1 else float(out[0])


def main() -> int:
    S = json.loads((RUN / "client/summaries.json").read_text())
    S.sort(key=lambda s: s["episode_index"])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    n_rows = np.array([s["inference_calls"] for s in S])
    offset = np.concatenate([[0], np.cumsum(n_rows)[:-1]])

    mixed = [s for s in np.unique(scene) if 0 < y[scene == s].sum() < (scene == s).sum()]
    groups = [scene == s for s in mixed]
    print("%d episodes, %d succ / %d fail; %d/%d scenes are mixed (%s dropped)"
          % (len(S), y.sum(), (~y).sum(), len(mixed), len(np.unique(scene)),
             ", ".join("s%d" % s for s in np.unique(scene) if s not in mixed)))
    print("discordant pairs: %d\n" % sum(int(y[m].sum()) * int((~y[m]).sum()) for m in groups))

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    # ---- pull the routing of every episode at steps 0..T_MAX-1 -------------
    # rows are episode-major and each episode is contiguous, so one slice per
    # episode reads whole chunks instead of scattering.
    ent = np.empty((len(S), T_MAX, 8), np.float32)      # entropy, mean over D,S
    mass = np.empty((len(S), T_MAX, 8), np.float32)     # top-4 probability mass
    load = np.empty((len(S), T_MAX, 8, 32), np.float32)  # expert load profile
    asp = np.empty((len(S), T_MAX, 4, 3), np.float32)
    asid = np.empty((len(S), T_MAX, 4), np.uint8)
    for i in range(len(S)):
        a, b = offset[i], offset[i] + T_MAX
        ent[i] = np.asarray(z["hb_entropy"][a:b], np.float32).mean((2, 3))
        mass[i] = np.asarray(z["hb_selected_prob"][a:b], np.float32).sum(-1).mean((2, 3))
        load[i] = np.asarray(z["hb_router_probs"][a:b], np.float32).mean((2, 3))
        asp[i] = np.asarray(z["as_probs"][a:b], np.float32)
        asid[i] = np.asarray(z["as_expert_ids"][a:b])
        if i % 128 == 0:
            print("  read %d/%d episodes" % (i, len(S)), flush=True)

    np.savez_compressed("/tmp/t08_features.npz", ent=ent, mass=mass, load=load,
                        asp=asp, asid=asid, y=y, scene=scene, n_rows=n_rows)

    # ---- descriptive ------------------------------------------------------
    print("\n=== what the routing looks like (steps 0..%d, all episodes) ===" % (T_MAX - 1))
    print("layer  entropy mean+-sd    top4 mass    max expert load  n experts >1%%")
    for L in range(8):
        pr = load[:, :, L].reshape(-1, 32)
        print("  %d    %.3f +- %.3f     %.4f       %.3f            %.1f"
              % (L, ent[:, :, L].mean(), ent[:, :, L].std(), mass[:, :, L].mean(),
                 pr.mean(0).max(), (pr > 0.01).sum(1).mean()))
    print("AS: distinct expert tuples across all episodes/steps: %d  (probs min %.3f)"
          % (len(np.unique(asid.reshape(-1, 4), axis=0)), asp.max(-1).min()))

    # ---- the test ---------------------------------------------------------
    feats = {"entropy(all layers)": ent.mean(-1),
             "top4 mass(all layers)": mass.mean(-1)}
    for L in range(8):
        feats["entropy L%d" % L] = ent[:, :, L]
        feats["top4 mass L%d" % L] = mass[:, :, L]
    names = list(feats)
    X = np.stack([feats[k] for k in names], -1)          # [N, T, F]

    print("\n=== stratified AUC per control step (0.5 = nothing) ===")
    grid = np.empty((T_MAX, len(names)))
    for t in range(T_MAX):
        grid[t] = strat_auc(X[:, t, :], y, groups)

    print("step  " + "  ".join("%-6s" % n.replace("entropy", "H").replace("top4 mass", "M")
                               .replace("(all layers)", "all").replace(" ", "")
                               for n in names))
    for t in range(T_MAX):
        print("%4d  " % t + "  ".join("%6.3f" % v for v in grid[t]))

    # ---- max-statistic permutation null -----------------------------------
    print("\nrunning %d within-scene label shuffles..." % N_PERM, flush=True)
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        yp = y.copy()
        for m in groups:
            idx = np.flatnonzero(m)
            yp[idx] = y[RNG.permutation(idx)]
        g = np.empty((T_MAX, len(names)))
        for t in range(T_MAX):
            g[t] = strat_auc(X[:, t, :], yp, groups)
        null[p] = np.abs(g - 0.5).max()
        if (p + 1) % 250 == 0:
            print("  %d/%d" % (p + 1, N_PERM), flush=True)

    obs = np.abs(grid - 0.5)
    thr = np.quantile(null, 0.95)
    pval = (1 + (null >= obs.max()).sum()) / (N_PERM + 1)
    print("\nobserved max |AUC-0.5| = %.4f at step %d, feature %r"
          % (obs.max(), *np.unravel_index(obs.argmax(), obs.shape)[:1],
             names[int(np.unravel_index(obs.argmax(), obs.shape)[1])]))
    print("null 95th pct = %.4f   family-wise p = %.4f" % (thr, pval))
    sig = np.argwhere(obs > thr)
    print("cells above the family-wise threshold: %d/%d" % (len(sig), obs.size))
    for t, f in sig[np.argsort(-obs[tuple(sig.T)])][:15]:
        print("   step %2d  %-22s AUC %.3f" % (t, names[f], grid[t, f]))

    np.savez("/tmp/t08_auc.npz", grid=grid, null=null, names=np.array(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
