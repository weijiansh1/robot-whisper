"""Can you read the wrist layout — or the outcome — off the routing alone?

Turns the effect into one number per question: leave-one-out cross-validated
balanced accuracy of a linear probe on per-episode expert-load features, with a
label-permutation null so chance is measured rather than assumed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from analyze_routes import HB_LAYERS, load, load_per_layer  # noqa: E402

RIGHT = "/home/jovyan/work/himoe-route-capture/runs/right50"
LEFT = "/home/jovyan/work/himoe-route-capture/runs/left50"


def features(episodes, n_win, layers=None):
    """Per-episode expert load -> [n_ep, n_layers*32]."""
    out = []
    for e in episodes:
        f = load_per_layer(e["ids"][:n_win], e["w"][:n_win])  # [8,32]
        if layers is not None:
            f = f[layers]
        out.append(f.ravel())
    return np.asarray(out)


def balanced_accuracy(y, pred):
    accs = []
    for c in np.unique(y):
        m = y == c
        accs.append((pred[m] == c).mean())
    return float(np.mean(accs))


def loo_probe(X, y, seed=0, n_perm=1000):
    """LOO-CV balanced accuracy + label-permutation null."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    pred = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    obs = balanced_accuracy(y, pred)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        p = np.empty_like(yp)
        for tr, te in LeaveOneOut().split(X):
            clf.fit(X[tr], yp[tr])
            p[te] = clf.predict(X[te])
        null.append(balanced_accuracy(yp, p))
    null = np.asarray(null)
    return obs, float(null.mean()), float((null >= obs).mean() + 1 / (n_perm + 1))


def main():
    right, left = load(RIGHT), load(LEFT)
    n_win = min(e["ids"].shape[0] for e in right + left)
    print("episodes: right %d (%d success) | left %d (%d success)"
          % (len(right), sum(e["summary"]["success"] for e in right),
             len(left), sum(e["summary"]["success"] for e in left)))
    print("matched window: first %d control steps\n" % n_win)

    results = {}

    # --- Q1: layout decodable from routing? ---
    X = np.vstack([features(right, n_win), features(left, n_win)])
    y = np.array([0] * len(right) + [1] * len(left))
    obs, null, p = loo_probe(X, y, n_perm=300)
    results["layout_all_layers"] = dict(acc=obs, null=null, p=p, n=len(y))
    print("Q1  从路由预测 wrist layout       LOO 平衡准确率 %.3f  (置换null %.3f, p=%.4f)"
          % (obs, null, p))

    # --- Q1b: per layer, to expose the depth gradient ---
    print("\n    逐层单独解码 layout:")
    per_layer = {}
    for li, layer in enumerate(HB_LAYERS):
        Xl = np.vstack([features(right, n_win, [li]), features(left, n_win, [li])])
        o, nl, pp = loo_probe(Xl, y, n_perm=200)
        per_layer[layer] = dict(acc=o, null=nl, p=pp)
        print("      layer %-3d  acc %.3f   (null %.3f, p=%.4f)" % (layer, o, nl, pp))
    results["layout_per_layer"] = per_layer

    # --- Q2: outcome decodable, within the broken layout? ---
    ys = np.array([int(e["summary"]["success"]) for e in left])
    Xs = features(left, n_win)
    print("\nQ2  released-left 组内，从路由预测成功/失败  (n=%d, 成功 %d)"
          % (len(ys), ys.sum()))
    if ys.sum() >= 3 and (1 - ys).sum() >= 3:
        obs, null, p = loo_probe(Xs, ys, n_perm=300)
        results["outcome_left"] = dict(acc=obs, null=null, p=p, n=int(len(ys)),
                                       n_success=int(ys.sum()))
        print("    LOO 平衡准确率 %.3f  (置换null %.3f, p=%.4f)" % (obs, null, p))
    else:
        print("    成功样本仍然太少，跳过")

    # --- Q3: same probe inside the healthy layout, as a control ---
    yr = np.array([int(e["summary"]["success"]) for e in right])
    if yr.sum() >= 3 and (1 - yr).sum() >= 3:
        Xr = features(right, n_win)
        obs, null, p = loo_probe(Xr, yr, n_perm=300)
        results["outcome_right"] = dict(acc=obs, null=null, p=p, n=int(len(yr)),
                                        n_success=int(yr.sum()))
        print("\nQ3  checkpoint-right 组内 成功/失败  LOO %.3f (null %.3f, p=%.4f)"
              % (obs, null, p))
    else:
        print("\nQ3  checkpoint-right 组内失败样本太少 (%d 失败)，无法做组内对比"
              % (1 - yr).sum())

    pathlib.Path("/home/jovyan/work/himoe-route-capture/decode_results.json").write_text(
        json.dumps(results, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
