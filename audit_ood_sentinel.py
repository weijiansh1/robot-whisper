#!/usr/bin/env python3
"""Form C cheap test: can MoE internals flag a config error faster than the
surface or behavior?

Setting: right50 (correct wrist layout) is the in-distribution reference;
left50 (the config error that costs ~18pp success) is the OOD stream.  All 50
init states are shared, so step-0 proprio is an exact negative control.

Pre-declared signals (fixed before looking at numbers):
  internal  : first-inference expert-occupancy histogram, mean over denoise
              rounds and action tokens -> 8x32 = 256 dims
  surface   : first action chunk, 10x7 = 70 dims
  negcontrol: step-0 proprio (8) -- must score ~0.5 or the harness is broken
  behavior  : episode success (reference 50/50, OOD ~40/50)

Scoring: diagonal-whitened L2 distance to the reference mean (fit on 25
reference episodes).  Report single-inference AUC (25 held right vs 50 left)
and detection sample efficiency: min n with detection rate >=95% at FPR<=5%
(threshold = 95th pct of held-right n-averages, 4000 bootstrap draws).
Behavior baseline: episodes needed so that P(any failure | p=0.8) >= 0.95.
Writes audit_ood_sentinel.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "himoe-route-capture/runs"
RNG = np.random.default_rng(0)
N_BOOT = 4000


def load_arm(name):
    p = RUNS / name
    S = sorted(json.loads((p / "summaries.json").read_text()),
               key=lambda s: s.get("episode_index", s.get("init_state_id", 0)))
    feats, acts, prop, succ = [], [], [], []
    for i, ep in enumerate(sorted(p.glob("episode_*.npz"))):
        z = np.load(ep, allow_pickle=True)
        ids = z["expert_ids"][0]                    # (10 denoise, 8, 10 tok, 4)
        hot = np.zeros((8, 32), np.float32)
        for L in range(8):
            v, c = np.unique(ids[:, L], return_counts=True)
            hot[L, v] = c
        hot /= hot.sum(1, keepdims=True)
        feats.append(hot.ravel())
        acts.append(z["actions"][0].ravel())
        prop.append(z["state"][0])
    succ = np.array([s["success"] for s in S], bool)
    return np.array(feats), np.array(acts), np.array(prop), succ


def auc(neg, pos):
    x = np.concatenate([neg, pos])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return (r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sample_eff(neg_scores, pos_scores, max_n=25):
    """min n where averaging n draws detects >=95% at FPR<=5%."""
    out = {}
    for n in range(1, max_n + 1):
        nb = np.array([RNG.choice(neg_scores, n).mean() for _ in range(N_BOOT)])
        pb = np.array([RNG.choice(pos_scores, n).mean() for _ in range(N_BOOT)])
        thr = np.quantile(nb, 0.95)
        det = float((pb > thr).mean())
        out[n] = det
        if det >= 0.95:
            return n, out
    return None, out


def main() -> int:
    Fr, Ar, Pr, Sr = load_arm("right50")
    Fl, Al, Pl, Sl = load_arm("left50")
    print("right50 success %d/50, left50 %d/50" % (Sr.sum(), Sl.sum()))
    ref = np.arange(0, 50, 2)
    hold = np.arange(1, 50, 2)

    out = {"right_succ": int(Sr.sum()), "left_succ": int(Sl.sum())}
    for label, R, L in (("internal_routing_256", Fr, Fl),
                        ("surface_action_70", Ar, Al),
                        ("negcontrol_proprio_8", Pr, Pl)):
        mu = R[ref].mean(0)
        sd = R[ref].std(0) + 1e-6
        s_hold = np.linalg.norm((R[hold] - mu) / sd, axis=1)
        s_ood = np.linalg.norm((L - mu) / sd, axis=1)
        a = auc(s_hold, s_ood)
        n95, curve = sample_eff(s_hold, s_ood)
        out[label] = {"auc_single": float(a), "n_to_95pct_detect": n95,
                      "detect_curve": {str(k): round(v, 3)
                                       for k, v in curve.items() if k <= 6}}
        print("%-22s single-inference AUC %.3f   n@95%%: %s"
              % (label, a, n95))

    # behavior baseline: reference is 50/50; rule "any failure in n episodes"
    p_fail = 1 - Sl.mean()
    n_beh = int(np.ceil(np.log(0.05) / np.log(1 - p_fail)))
    out["behavior_success"] = {
        "left_fail_rate": float(p_fail),
        "episodes_to_95pct_detect": n_beh,
        "note": "rule = any failure; reference had 0 failures so FPR~0; "
                "one episode ~= 12+ inferences",
    }
    print("behavior (success-rate)  episodes@95%%: %d  (~%d+ inferences)"
          % (n_beh, n_beh * 12))

    (HERE / "audit_ood_sentinel.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_ood_sentinel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
