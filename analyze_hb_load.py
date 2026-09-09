#!/usr/bin/env python3
"""Is HB expert *activation* uniform at inference?  Three different questions.

The combine weights are near-uniform (w1..w4 ~ 0.25 each), but that is about how
the four chosen experts are mixed, not about which four get chosen.  Selection is
a threshold on a near-uniform distribution, and a threshold amplifies: at layer 5
the state token puts 6.9x uniform probability on expert 12, which could mean that
expert is picked essentially every time while others are never picked.  So:

  1. GLOBAL load.  How often is each of the 32 experts in the top-4?  Uniform is
     4/32 = 12.5%.  Also the DeepSeekMoE statistic the paper's HB-Reg minimises,
     f_i = (N/KU) sum_u r_{i,u} and L_HB = sum_i f_i P_i, which is 1 at perfect
     balance -- the paper reports it "converges close to its theoretical lower
     bound", and that is checkable here rather than trusted.

  2. PER-SITE load.  HB-Reg constrains only the aggregate, so a model can be
     perfectly balanced globally while every individual site deterministically
     uses its own four.  Measured as the effective number of experts each site
     draws from, exp(H) over its selection histogram, against 32 for a site that
     spreads over everything and 4 for a site that always picks the same four.

  3. Dead experts.  Any of the 32 never selected at a layer at all.

Split by state token (suffix 0) against the ten action tokens throughout, since
that axis carries 68-94% of the router's variance.
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
N_EXPERT, TOP_K = 32, 4
CHUNK = 2048


def eff_n(counts: np.ndarray, axis: int = -1) -> np.ndarray:
    """exp(Shannon entropy) of a count histogram -- 'how many experts really'."""
    p = counts / np.maximum(counts.sum(axis, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(np.where(p > 0, p, 1))).sum(axis)
    return np.exp(h)


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
        z = zarr.open(str(run / "server/routes.zarr"), mode="r")
        n = z["hb_expert_ids"].shape[0]
        # counts[layer, suffix, expert] and the mean score P_i[layer, expert]
        cnt = np.zeros((8, 11, N_EXPERT), np.int64)
        psum = np.zeros((8, N_EXPERT), np.float64)
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            ids = np.asarray(z["hb_expert_ids"][a:b]).astype(np.int64)
            for L in range(8):
                for s in range(11):
                    cnt[L, s] += np.bincount(ids[:, L, :, s].ravel(),
                                             minlength=N_EXPERT)
            psum += np.asarray(z["hb_router_probs"][a:b],
                               np.float64).sum((0, 2, 3))
        n_tok = n * 10                                  # per (layer, suffix)
        print("\n=== %s / %s  (%d control steps) ===" % (suite, name[:44], n))

        tot = cnt.sum(1)                                # [8, 32] over all suffix
        U = n_tok * 11
        f = (N_EXPERT / (TOP_K * U)) * tot              # DeepSeek f_i, 1 = balanced
        P = psum / U                                    # mean softmax score
        L_HB = (f * P).sum(1)

        print("  layer  sel%% min/max      f_i min/max      L_HB     eff experts"
              "   dead")
        for i, L in enumerate(HB_LAYER):
            fr = tot[i] / (n_tok * 11)                  # selection frequency
            print("   %2d    %.3f / %.3f    %.3f / %.3f    %.4f    %5.1f / 32   %d"
                  % (L, fr.min(), fr.max(), f[i].min(), f[i].max(), L_HB[i],
                     eff_n(tot[i]), int((tot[i] == 0).sum())))
        print("  uniform reference: sel%% = %.3f, f_i = 1.000, L_HB = 1.0000, "
              "eff = 32.0" % (TOP_K / N_EXPERT))

        # ---- state token vs action tokens ---------------------------------
        print("  by token, selection frequency of the single most-used expert:")
        print("        " + "  ".join("L%-4d" % L for L in HB_LAYER))
        st = cnt[:, 0] / n_tok
        ac = cnt[:, 1:].sum(1) / (n_tok * 10)
        print("   state " + "  ".join("%.3f" % v for v in st.max(1)))
        print("   action" + "  ".join("%.3f" % v for v in ac.max(1)))
        print("   state eff experts " + "  ".join("%5.1f" % v for v in eff_n(cnt[:, 0])))
        print("   action eff experts" + "  ".join("%5.1f" % v
                                                  for v in eff_n(cnt[:, 1:].sum(1))))

        # ---- per-site concentration ---------------------------------------
        # a site is (layer, denoise, suffix); how many experts does it draw from?
        site = np.zeros((8, 10, 11, N_EXPERT), np.int64)
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            ids = np.asarray(z["hb_expert_ids"][a:b]).astype(np.int64)
            for L in range(8):
                for d in range(10):
                    for s in range(11):
                        site[L, d, s] += np.bincount(ids[:, L, d, s].ravel(),
                                                     minlength=N_EXPERT)
        e = eff_n(site)                                 # [8,10,11]
        print("  per-site effective number of experts (4 = always the same four,"
              " 32 = spreads):")
        print("        front L2-5 %.1f   back L12-15 %.1f   "
              "state token %.1f   action tokens %.1f"
              % (e[:4].mean(), e[4:].mean(), e[:, :, 0].mean(), e[:, :, 1:].mean()))
        # site.sum(-1) is 4 x n_steps (top-4 per step), so dividing by it gives a
        # share of *selections* against a 1/32 uniform; multiply by K to get the
        # probability the expert is in the top-4, comparable to the 12.5% above.
        top1 = TOP_K * site.max(-1) / site.sum(-1)
        print("        P(most-used expert of a site is in its top-4) = %.1f%% "
              "mean, %.1f%% at the most concentrated site  (uniform 12.5%%)"
              % (100 * top1.mean(), 100 * top1.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
