#!/usr/bin/env python3
"""How much input-dependence is there in HB routing to remove?

Everything measured so far says the HB choice barely matters *per site*: the
experts are near-orthogonal but the selected 4 are no closer to anything than a
random 4, and a random substitution costs 4% of the action.  The open question is
functional and end-to-end -- does HB routing change whether the episode succeeds?
That needs a rollout intervention, which costs GPU hours, so this pass first
measures the ceiling on any such intervention from data already on disk.

If the top-4 at a site is already almost fully determined by *where* the site is
(which layer, which denoise iteration, which suffix token) then freezing the
routing to that positional mode is nearly a no-op, and a null result from the
rollout arm would say nothing.  The size of the gap between "actual" and
"positional mode" is exactly the budget an intervention has to work with.

Three things, all from routes.zarr:

  1. the combine weights.  HB is top-4 with norm_topk_prob, so the four selected
     raw probabilities are renormalised to sum to 1.  If the raw scores are
     near-uniform the weights collapse to 0.25 each and the routed branch is an
     unweighted mean of four experts -- which would make *which* four the only
     lever the router has.
  2. positional predictability: how much of the actual top-4 is recovered by the
     modal set at (layer, denoise, suffix), and at (layer, suffix) alone.
  3. where the residual input-dependence lives -- is deviation from the mode
     related to the scene, the control step, or the outcome?
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN_ID = "right-16x32"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]


def load(run: pathlib.Path):
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    ids = np.asarray(z["hb_expert_ids"][:])            # [N,8,10,11,4] uint8
    sel = np.asarray(z["hb_selected_prob"][:], np.float32)
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    y = np.array([s["success"] for s in S], bool)[ep]
    scene = np.array([s["init_state_id"] for s in S])[ep]
    return ids, sel, ep, step, y, scene


def hot(ids: np.ndarray) -> np.ndarray:
    """[..., 4] expert ids -> [..., 32] membership."""
    out = np.zeros(ids.shape[:-1] + (32,), bool)
    np.put_along_axis(out, ids.astype(np.int64), True, axis=-1)
    return out


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl

    runs = []
    for suite in ("libero_goal", "libero_spatial", "libero_10"):
        base = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        for task in sorted(base.iterdir()):
            if (task / RUN_ID / "server/routes.zarr").exists():
                runs.append((suite, task.name, task / RUN_ID))

    for suite, name, run in runs:
        ids, sel, ep, step, y, scene = load(run)
        n = ids.shape[0]
        print("\n=== %s / %s  (%d control steps) ===" % (suite, name[:44], n))

        # ---- 1. combine weights ------------------------------------------
        w = sel / sel.sum(-1, keepdims=True)           # norm_topk_prob
        ws = np.sort(w, -1)[..., ::-1]
        print("  combine weights after norm_topk_prob (uniform = 0.2500):")
        print("     w1 %.4f  w2 %.4f  w3 %.4f  w4 %.4f   (means over %d sites)"
              % (*ws.mean((0, 1, 2, 3)), ws[..., 0].size))
        print("     w1 - w4 = %.4f   p99 of (w1-w4) = %.4f   max %.4f"
              % ((ws[..., 0] - ws[..., 3]).mean(),
                 np.percentile(ws[..., 0] - ws[..., 3], 99),
                 (ws[..., 0] - ws[..., 3]).max()))

        # ---- 2. positional predictability ---------------------------------
        H = hot(ids)                                   # [N,8,10,11,32]
        # modal top-4 per (layer, denoise, suffix): the 4 most-used experts
        use_lds = H.sum(0)                             # [8,10,11,32]
        mode_lds = hot(np.argsort(-use_lds, -1)[..., :4])
        use_ls = H.sum((0, 2))                         # [8,11,32]
        mode_ls = hot(np.argsort(-use_ls, -1)[..., :4])

        ov_lds = (H & mode_lds[None]).sum(-1)          # [N,8,10,11]
        ov_ls = (H & mode_ls[None, :, None]).sum(-1)
        print("  how much of the actual top-4 the positional mode recovers:")
        print("     mode(layer,denoise,suffix): %.3f / 4   exact-set match %.1f%%"
              % (ov_lds.mean(), 100 * (ov_lds == 4).mean()))
        print("     mode(layer,suffix):         %.3f / 4   exact-set match %.1f%%"
              % (ov_ls.mean(), 100 * (ov_ls == 4).mean()))
        print("     chance baseline for a fixed 4 of 32: 0.500 / 4")

        # how many distinct top-4 sets does a site ever take?
        key = (np.sort(ids.astype(np.int64), -1)
               * np.array([1, 32, 1024, 32768])).sum(-1)   # [N,8,10,11]
        k = key.reshape(n, -1)
        nuniq = np.array([len(np.unique(k[:, j])) for j in range(0, k.shape[1], 7)])
        print("     distinct top-4 sets per site over the whole task: "
              "median %d  p90 %d  max %d  (of %d steps)"
              % (np.median(nuniq), np.percentile(nuniq, 90), nuniq.max(), n))

        # ---- 3. where does the deviation live? ----------------------------
        # Averaging over an episode's own control steps would compare a 52-step
        # failure against a 39-step success on a quantity that drifts with the
        # step index, which is the episode-length leak, not routing.  Compare at
        # a fixed step instead.
        dev = 4 - ov_lds                               # experts off the mode
        print("  deviation from the positional mode:")
        print("     mean %.3f of 4 experts   by block: front %.3f  back %.3f"
              % (dev.mean(), dev[:, :4].mean(), dev[:, 4:].mean()))
        from scipy.stats import rankdata
        eps = np.unique(ep)
        yy = np.array([y[ep == e][0] for e in eps])
        sc = np.array([scene[ep == e][0] for e in eps])
        row = []
        for t in (0, 4, 8):
            have = np.array([((ep == e) & (step == t)).any() for e in eps])
            x = np.full(len(eps), np.nan)
            idx = np.flatnonzero((ep >= 0) & (step == t))
            x[ep[idx]] = dev[idx].mean((1, 2, 3))
            num = den = 0.0
            for s in np.unique(sc):
                m = (sc == s) & have & np.isfinite(x)
                if not (0 < yy[m].sum() < m.sum()):
                    continue
                r = rankdata(x[m])
                n1, n0 = int(yy[m].sum()), int((~yy[m]).sum())
                num += r[yy[m]].sum() - n1 * (n1 + 1) / 2.0
                den += n1 * n0
            row.append(num / den if den else float("nan"))
        print("     stratified AUC vs outcome at a FIXED control step: "
              "t=0 %.3f  t=4 %.3f  t=8 %.3f" % tuple(row))

        # ---- 4. is the churn driven by the observation or by the noise? ----
        # At control step 0 the 32 draws of a scene share a bit-identical
        # observation and differ only in flow_noise_seed.  So the pairwise
        # overlap there is what a *pure noise* change does to the top-4.
        s0 = step == 0
        ov_pairs, ov_mode0 = [], []
        for s in np.unique(sc):
            rows = np.flatnonzero(s0 & (scene == s))
            if len(rows) < 2:
                continue
            A = H[rows]                                # [k,8,10,11,32]
            for i in range(0, len(rows) - 1, 2):
                ov_pairs.append((A[i] & A[i + 1]).sum(-1).mean())
            ov_mode0.append(ov_lds[rows].mean())
        print("  same scene, same control step 0, only the noise seed differs:")
        print("     top-4 shared between two draws: %.3f / 4   "
              "(vs positional mode %.3f, chance 0.500)"
              % (np.mean(ov_pairs), np.mean(ov_mode0)))
        del ids, sel, H
    return 0


if __name__ == "__main__":
    sys.exit(main())
