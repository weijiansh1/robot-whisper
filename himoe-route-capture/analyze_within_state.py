"""Within-init-state contrast: same scene, different flow noise, different outcome.

Holding the initial state fixed removes scene difficulty entirely -- the only
thing that differs between repeats is the policy's sampled flow noise.  Any
routing difference between a success and a failure of the SAME init state
therefore cannot be "that scene was harder".

Two guards against the leaks that make this kind of analysis look better than
it is:
  * per-state centering is fitted on training folds only, never on the held-out
    episode;
  * the permutation null shuffles outcome labels WITHIN each init state, so the
    null retains all between-state structure.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from analyze_routes import HB_LAYERS, load, load_per_layer, tv  # noqa: E402

RUN = "/home/jovyan/work/himoe-route-capture/runs/within"


def balanced_accuracy(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def within_state_probe(X, y, state, seed=0, n_perm=1000):
    """LOO balanced accuracy after fold-wise per-state centering."""

    def run(labels):
        pred = np.empty_like(labels)
        for tr, te in LeaveOneOut().split(X):
            means = {}
            for s in np.unique(state):
                m = (state == s) & np.isin(np.arange(len(X)), tr)
                means[s] = X[m].mean(0) if m.any() else X[tr].mean(0)
            Xc = np.array([X[i] - means[state[i]] for i in range(len(X))])
            sc = StandardScaler().fit(Xc[tr])
            clf = LogisticRegression(max_iter=5000, C=1.0)
            # a fold whose training labels collapse to one class cannot be fit
            if len(np.unique(labels[tr])) < 2:
                pred[te] = labels[tr][0]
                continue
            clf.fit(sc.transform(Xc[tr]), labels[tr])
            pred[te] = clf.predict(sc.transform(Xc[te]))
        return balanced_accuracy(labels, pred)

    obs = run(y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(state):  # shuffle WITHIN each init state
            m = state == s
            yp[m] = rng.permutation(y[m])
        null.append(run(yp))
    null = np.asarray(null)
    return obs, float(null.mean()), float((null >= obs).mean() + 1 / (n_perm + 1))


def main():
    eps = load(RUN)
    n = min(e["ids"].shape[0] for e in eps)
    state = np.array([e["summary"]["init_state_id"] for e in eps])
    y = np.array([int(e["summary"]["success"]) for e in eps])
    X = np.array([load_per_layer(e["ids"][:n], e["w"][:n]).ravel() for e in eps])
    print("%d episodes, matched window %d control steps\n" % (len(eps), n))

    # ---- per-state outcome rates: how much is scene, how much is noise? ----
    print("=== 每个 init state 的成功率（只有 flow noise 在变）===")
    print("%-6s %-8s %-10s %s" % ("state", "成功/总", "成功率", "判定"))
    mixed = []
    for s in sorted(set(state.tolist())):
        m = state == s
        k, tot = int(y[m].sum()), int(m.sum())
        kind = "混合 ← 可用" if 0 < k < tot else ("全成功" if k == tot else "全失败")
        if 0 < k < tot:
            mixed.append(s)
        print("%-6d %d/%-6d %-10.2f %s" % (s, k, tot, k / tot, kind))
    print("\n可用于组内对比的 init state: %s" % mixed)
    if not mixed:
        print("没有任何 init state 出现混合结果 -> 结果完全由场景决定，"
              "flow noise 改变不了成败；组内对比无法进行。")
        return 0

    sel = np.isin(state, mixed)
    Xs, ys, ss = X[sel], y[sel], state[sel]
    print("组内可用样本: %d 集 (成功 %d / 失败 %d), 跨 %d 个 init state"
          % (len(ys), ys.sum(), (1 - ys).sum(), len(mixed)))

    # ---- the probe the user asked for ----
    print("\n=== 组内解码：同一 init state 内，路由能否区分成功与失败 ===")
    obs, null, p = within_state_probe(Xs, ys, ss, n_perm=500)
    print("  按 state 去均值后 LOO 平衡准确率 = %.3f  (组内置换 null %.3f, p = %.4f)"
          % (obs, null, p))

    # ---- naive version, for contrast ----
    print("\n=== 对照：不做 state 去均值（= 上一轮的做法，会混进场景难度）===")
    def naive(labels):
        pred = np.empty_like(labels)
        for tr, te in LeaveOneOut().split(Xs):
            if len(np.unique(labels[tr])) < 2:
                pred[te] = labels[tr][0]
                continue
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xs[tr]), labels[tr])
            pred[te] = clf.predict(sc.transform(Xs[te]))
        return balanced_accuracy(labels, pred)
    print("  未去均值 LOO 平衡准确率 = %.3f" % naive(ys))

    # ---- are the per-state success-minus-failure directions consistent? ----
    print("\n=== 各 init state 的「成功减失败」方向是否一致 ===")
    diffs = {}
    for s in mixed:
        m = state == s
        d = X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0)
        diffs[s] = d
    keys = sorted(diffs)
    cos = []
    print("%-8s %s" % ("", "  ".join("s%-4d" % k for k in keys)))
    for a in keys:
        row = []
        for b in keys:
            c = float(diffs[a] @ diffs[b] /
                      (np.linalg.norm(diffs[a]) * np.linalg.norm(diffs[b])))
            row.append("%5.2f" % c)
            if a < b:
                cos.append(c)
        print("s%-7d %s" % (a, "  ".join(row)))
    print("\n  非对角平均余弦 = %.3f  (0 = 各场景的差异方向互不相关)" % np.mean(cos))
    rng = np.random.default_rng(0)
    nullc = []
    for _ in range(2000):
        dd = {}
        for s in mixed:
            m = state == s
            yp = rng.permutation(y[m])
            if 0 < yp.sum() < len(yp):
                dd[s] = X[m][yp == 1].mean(0) - X[m][yp == 0].mean(0)
        ks = sorted(dd)
        cc = [float(dd[a] @ dd[b] / (np.linalg.norm(dd[a]) * np.linalg.norm(dd[b])))
              for i, a in enumerate(ks) for b in ks[i + 1:]]
        if cc:
            nullc.append(np.mean(cc))
    nullc = np.asarray(nullc)
    print("  组内置换 null = %.3f, p = %.4f"
          % (nullc.mean(), (nullc >= np.mean(cos)).mean()))

    # ---- per-layer TV, within state ----
    print("\n=== 组内负载 TV（每个 state 各自算，再平均）===")
    per_layer = np.zeros(8)
    for s in mixed:
        m = state == s
        lo = np.mean([load_per_layer(e["ids"][:n], e["w"][:n])
                      for e, keep in zip(eps, m & (y == 1)) if keep], axis=0)
        lb = np.mean([load_per_layer(e["ids"][:n], e["w"][:n])
                      for e, keep in zip(eps, m & (y == 0)) if keep], axis=0)
        per_layer += tv(lo, lb) / len(mixed)
    for li, layer in enumerate(HB_LAYERS):
        print("  layer %-3d  TV = %.4f" % (layer, per_layer[li]))
    print("  平均 = %.4f" % per_layer.mean())

    json.dump(
        {"mixed_states": [int(s) for s in mixed], "within_acc": obs,
         "within_null": null, "within_p": p, "mean_cosine": float(np.mean(cos))},
        open("/home/jovyan/work/himoe-route-capture/within_results.json", "w"),
        indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
