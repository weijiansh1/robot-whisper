#!/usr/bin/env python3
"""How much does the routing change from token to token inside one inference?

The earlier per-token picture was built from profiles averaged over 22883 control
steps, which shows what a token prefers on average.  It does not say how
different two tokens look at one moment, because an average can be stable while
every individual draw is noisy.

So: fix a control step, and compare within it.  Three contrasts, all at matched
resolution, so the token effect can be read against the two things it competes
with inside a single forward pass:

    token vs token     same control step, same denoise iteration
    denoise vs denoise same control step, same token, adjacent iterations
    step vs step       same token, same denoise, adjacent control steps

The first two are strictly inside one inference; the third is the nearest thing
outside it, and is the scale against which "a lot" or "a little" means anything.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
FIG = HERE / "fig"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
TOK = ["state"] + ["a%d" % k for k in range(1, 11)]
N_STEP = 300


def top4(p):
    h = np.zeros(p.shape, bool)
    np.put_along_axis(h, np.argsort(-p, -1)[..., :4], True, -1)
    return h


def main() -> int:
    FIG.mkdir(exist_ok=True)
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    rng = np.random.default_rng(1)
    # pick control steps that have a successor in the same episode, so the
    # step-to-step contrast is within an episode and not across a boundary
    ep = rng.choice(len(S), N_STEP)
    t = np.array([rng.integers(0, n_rows[e] - 1) for e in ep])
    rows = off[ep] + t

    P = np.empty((N_STEP, 2, 8, 10, 11, 32), np.float32)
    for a in range(0, N_STEP, 32):
        b = min(a + 32, N_STEP)
        P[a:b, 0] = np.asarray(z["hb_router_probs"].oindex[rows[a:b]], np.float32)
        P[a:b, 1] = np.asarray(z["hb_router_probs"].oindex[rows[a:b] + 1], np.float32)
    H = top4(P)

    def pair(A, B):
        """mean top-4 overlap and mean cosine distance over matched pairs."""
        ov = (top4(A) & top4(B)).sum(-1).mean()
        c = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
        return float(ov), float(1 - c.mean())

    print("=== the three contrasts, front block (L2-5) and back block (L12-15) ===")
    print("contrast                                    top-4 shared   1 - cosine")
    out = {}
    for tag, sl in (("front L2-5", slice(0, 4)), ("back L12-15", slice(4, 8))):
        Q = P[:, :, sl]
        # token vs token: all 55 unordered pairs, same step, same denoise
        ii, jj = np.triu_indices(11, 1)
        a = Q[:, 0][:, :, :, ii]
        b = Q[:, 0][:, :, :, jj]
        ov_t, c_t = pair(a, b)
        # denoise vs denoise: adjacent iterations, same step, same token
        ov_d, c_d = pair(Q[:, 0, :, :-1], Q[:, 0, :, 1:])
        # step vs step: same token, same denoise, t and t+1
        ov_s, c_s = pair(Q[:, 0], Q[:, 1])
        out[tag] = (ov_t, c_t, ov_d, c_d, ov_s, c_s)
        print("  %-12s token vs token (same step, same denoise)   %.2f / 4      %.5f"
              % (tag, ov_t, c_t))
        print("  %-12s denoise d vs d+1 (same step, same token)   %.2f / 4      %.5f"
              % ("", ov_d, c_d))
        print("  %-12s control step t vs t+1 (same token, same d) %.2f / 4      %.5f"
              % ("", ov_s, c_s))
    print("  chance for two unrelated top-4 sets out of 32: 0.50 / 4")

    print("\n=== within one inference: the 11 x 11 token overlap ===")
    for tag, sl in (("front L2-5", slice(0, 4)), ("back L12-15", slice(4, 8))):
        M = np.zeros((11, 11))
        Q = P[:, 0, sl]
        T = top4(Q)
        for i in range(11):
            for j in range(11):
                M[i, j] = (T[:, :, :, i] & T[:, :, :, j]).sum(-1).mean()
        out[tag + "_M"] = M
        print("  %s" % tag)
        print("        " + " ".join("%5s" % s for s in TOK))
        for i in range(11):
            print("  %-5s " % TOK[i] + " ".join("%5.2f" % v for v in M[i]))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, tag in enumerate(("front L2-5", "back L12-15")):
        a = ax[k]
        im = a.imshow(out[tag + "_M"], cmap="viridis", vmin=0.5, vmax=4)
        a.set_xticks(range(11), TOK, rotation=45)
        a.set_yticks(range(11), TOK)
        for i in range(11):
            for j in range(11):
                a.text(j, i, "%.1f" % out[tag + "_M"][i, j], ha="center",
                       va="center", fontsize=6,
                       color="w" if out[tag + "_M"][i, j] < 2.5 else "k")
        fig.colorbar(im, ax=a, label="top-4 experts shared")
        a.set_title("%s\nwithin ONE inference, token vs token" % tag, fontsize=10)

    a = ax[2]
    lbl = ["token\nvs token", "denoise\nd vs d+1", "control step\nt vs t+1"]
    x = np.arange(3)
    for k, (tag, c) in enumerate((("front L2-5", "tab:blue"),
                                  ("back L12-15", "tab:red"))):
        v = [out[tag][0], out[tag][2], out[tag][4]]
        a.bar(x + (k - .5) * .35, v, .32, color=c, label=tag)
    a.axhline(0.5, ls=":", c="k")
    a.text(2.2, .62, "chance 0.50", fontsize=8)
    a.axhline(4, ls="--", c="k", lw=.8)
    a.text(-0.4, 3.85, "identical", fontsize=8)
    a.set_xticks(x, lbl); a.set_ylabel("top-4 experts shared (of 4)")
    a.legend(fontsize=8)
    a.set_title("what changes the routing most,\nand what barely does", fontsize=10)
    fig.suptitle("MoE routing inside a single inference, libero_10/t08, "
                 "%d sampled control steps" % N_STEP, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "hb5_within.png", dpi=130)
    print("\nwrote %s" % (FIG / "hb5_within.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
