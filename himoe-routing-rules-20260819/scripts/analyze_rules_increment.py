#!/usr/bin/env python3
"""What is left of the rule once the *whole* physical state is accounted for?

Pose-matched twins left a real excess -- the top-20 union fails 11.9 points more
often than an arm in the same place, same scene, same phase, points [+4.5,+19.1]
-- but the match only held the robot's own 8 numbers fixed.  Inside one scene the
objects still drift apart, so "the pot is already knocked over" would produce
exactly this signature without the routing knowing anything.

So residualise instead of match.  Predict failure at each control step from the
full simulator state with a nonlinear model, leave-one-scene-out, and score the
rule on what the prediction misses:

    excess(R) = mean over firing steps of ( failed - P(fail | physical state) )

Calibration comes for free: a rule that fires at random must score 0, and that
arm is run.  Three baselines of increasing strength -- phase only, proprio 8,
full sim_state -- so the excess can be watched as the control gets harder.  If
it survives the full state, the routing carries something the simulator does not
hand to it; if it dies at proprio 8, it never did.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_rules import HB_LAYER, HUB, cooccur, lit_name, load, score  # noqa: E402

RNG = np.random.default_rng(0)


def loso_p(X, yf, ep_of_row):
    """Grouped by episode, not by scene.

    Scene identity is itself part of the physical state -- it is the initial
    condition -- so the baseline is allowed to use it; what it must never see is
    the held-out episode's own outcome, and grouping on the episode is exactly
    that.  Holding out whole scenes made the model extrapolate to unseen object
    layouts and left it miscalibrated (log-loss 1.19 bit, worse than phase
    alone), which biased every residual.
    """
    p = np.zeros(len(yf))
    for tr, te in GroupKFold(n_splits=8).split(X, yf, groups=ep_of_row):
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, random_state=0).fit(X[tr], yf[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def excess(hit, yf, p, ep_of_row):
    m = hit.ravel()
    if m.sum() < 30:
        return None
    r = (yf - p)[m]
    e = ep_of_row[m]
    eps = np.unique(e)
    bs = [np.concatenate([r[e == q] for q in RNG.choice(eps, len(eps))]).mean()
          for _ in range(1000)]
    return r.mean(), np.quantile(bs, .025), np.quantile(bs, .975), int(m.sum())


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    suite = sys.argv[2] if len(sys.argv) > 2 else "libero_long"
    run = HUB / "cache/HiMoE-VLA" / suite / task / "right-16x32"
    D = load(run)
    y, scene, M, prop, W = D["y"], D["scene"], D["M"], D["prop"], D["win"]
    fail = ~y
    n = len(y)
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.zeros((n, W, d0.shape[1]), np.float32)
    for i, s in enumerate(S):
        sim[i] = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                         allow_pickle=True)["sim_state"][:W]
    ep_of_row = np.repeat(np.arange(n), W)
    step = np.tile(np.arange(W), n).astype(np.float32)[:, None]
    yf = np.repeat(fail, W).astype(int)
    sc_row = scene[ep_of_row]
    print("%s，%d 局 x %d 步，仿真状态 %d 维，留一场景 + 非线性基线\n"
          % (task, n, W, d0.shape[1]))

    oh = np.zeros((n * W, len(np.unique(scene))), np.float32)
    oh[np.arange(n * W), np.unique(scene, return_inverse=True)[1][ep_of_row]] = 1
    base = {
        "相位+场景": np.c_[oh, step],
        "+本体感受 8 维": np.c_[oh, prop.reshape(-1, 8), step],
        "+完整仿真状态 %d 维" % d0.shape[1]: np.c_[oh, sim.reshape(-1, d0.shape[1]),
                                                   prop.reshape(-1, 8), step],
    }
    P = {}
    for k, X in base.items():
        P[k] = loso_p(X, yf, ep_of_row)
        ll = -(yf * np.log(np.clip(P[k], 1e-6, 1)) +
               (1 - yf) * np.log(np.clip(1 - P[k], 1e-6, 1))).mean() / np.log(2)
        print("  基线「%s」步级对数损失 %.3f bit" % (k, ll))

    fires = cooccur(M)
    p_fail = np.zeros(n, np.float32)
    for s in np.unique(scene):
        p_fail[scene == s] = fail[scene == s].mean()
    sup, obs, exp, z = score(fires, fail, p_fail)
    ok = (sup >= 20) & (sup <= n - 5)
    ok[np.tril_indices(256, -1)] = False
    order = np.argsort(np.where(ok, z, -np.inf).ravel())[::-1]

    rules = []
    for f in order[:3]:
        a, b = np.unravel_index(f, z.shape)
        rules.append(("%s & %s" % (lit_name(a), lit_name(b)),
                      (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)))
    pool = np.zeros(M.shape[:2], bool)
    for f in order[:20]:
        a, b = np.unravel_index(f, z.shape)
        pool |= (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
    rules.append(("前 20 条并集", pool))
    i5 = HB_LAYER.index(2) * 32 + 5
    rules.append(("单独 L2:e5", M[:, :, i5] > 0.5))
    rules.append(("对照：同频率随机", RNG.random(M.shape[:2]) < pool.mean()))

    print("\n=== 残差超额失败率（越强的基线越难剩下东西）===")
    print("  规则                     触发步数   " +
          "".join("%-22s" % k for k in base))
    for nm, hit in rules:
        row = "  %-24s %6d   " % (nm, hit.sum())
        for k in base:
            r = excess(hit, yf, P[k], ep_of_row)
            row += "%+.3f[%+.3f,%+.3f]%s " % (
                r[0], r[1], r[2], "*" if r[1] > 0 or r[2] < 0 else " ")
        print(row)
    print("\n  * = 95%% 自助区间不含 0（按局重抽，%d 局）" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
