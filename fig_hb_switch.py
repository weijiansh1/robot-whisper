#!/usr/bin/env python3
"""The one input-dependent routing decision in HiMoE, and what it reads."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "fig"
RUN = (HERE / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
UNIF = 1 / 32


def main() -> int:
    FIG.mkdir(exist_ok=True)
    d = np.load("/tmp/hb_switch.npz")
    p, on, step, star = d["p"], d["on"], d["step"], d["star"]
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    prop = np.empty((p.shape[0], 8), np.float32)
    for i, s in enumerate(S):
        a, k = off[i], n_rows[i]
        prop[a:a + k] = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                                allow_pickle=True)["state"][:k]
    gap = prop[:, 6] - prop[:, 7]
    o5, o4 = on[:, 3], on[:, 2]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))

    # A: the bimodal distribution at L5 against a back-block layer
    a = ax[0]
    a.hist(p[:, 3].ravel(), bins=50, range=(0, 1), color="tab:red", alpha=.8,
           label="L5 state token (e%d)" % star[3])
    a.hist(p[:, 7].ravel(), bins=50, range=(0, 1), color="tab:blue", alpha=.6,
           label="L15 state token (e%d)" % star[7])
    a.axvline(UNIF, ls="--", c="k", lw=1)
    a.text(UNIF + .01, a.get_ylim()[1] * .6, "uniform\n1/32", fontsize=8)
    a.set_yscale("log"); a.set_xlabel("router probability of that layer's favourite expert")
    a.set_ylabel("draws (log)"); a.legend(fontsize=8)
    a.set_title("A  only the front block is bimodal\n"
                "L5 is off 61% / on 25%; L15 never leaves uniform", fontsize=10)

    # B: dose response against gripper opening
    a = ax[1]
    edges = np.array([0, .002, .006, .012, .020, .030, .040, .050, .060, .070,
                      .075, .080, .085])
    ctr, r5, r4, nn = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gap >= lo) & (gap < hi)
        if m.sum() < 100:
            continue
        ctr.append((lo + hi) / 2); r5.append(o5[m].mean()); r4.append(o4[m].mean())
        nn.append(m.sum())
    a.plot(ctr, r5, "o-", c="tab:red", label="L5 switch")
    a.plot(ctr, r4, "s-", c="tab:green", label="L4 switch")
    a.axvline(.040, ls="--", c="k", lw=1)
    a.text(.042, .6, "gap < 0.040\n94.1% accurate", fontsize=8)
    a.set_xlabel("gripper opening  (qpos$_6$ - qpos$_7$)")
    a.set_ylabel("P(switch on)"); a.legend(fontsize=8)
    a.set_title("B  the switch is a gripper gate\n"
                "L5 fires closed, L4 only when fully open", fontsize=10)
    for x, y, k in zip(ctr, r5, nn):
        a.annotate("%d" % k, (x, y), fontsize=6, xytext=(0, 6),
                   textcoords="offset points", ha="center")

    # C: on-rate along the episode
    a = ax[2]
    T = 40
    b5 = [o5[step == t].mean() for t in range(T)]
    b4 = [o4[step == t].mean() for t in range(T)]
    bg = [gap[step == t].mean() for t in range(T)]
    a.plot(range(T), b5, "-", c="tab:red", lw=2, label="L5 on-rate")
    a.plot(range(T), b4, "-", c="tab:green", lw=2, label="L4 on-rate")
    a2 = a.twinx()
    a2.plot(range(T), bg, ":", c="tab:gray", lw=2, label="gripper opening")
    a2.set_ylabel("mean gripper opening", color="tab:gray")
    a.set_xlabel("control step"); a.set_ylabel("P(switch on)")
    a.legend(fontsize=8, loc="upper left"); a2.legend(fontsize=8, loc="upper right")
    a.set_title("C  the two gates alternate along the task\n"
                "'put both moka pots on the stove' - two grasps", fontsize=10)

    fig.suptitle("HiMoE's only input-dependent routing: the state token at layers "
                 "4-5, libero_10/t08, 512 episodes", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "hb2_switch.png", dpi=130)
    print("wrote", FIG / "hb2_switch.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
