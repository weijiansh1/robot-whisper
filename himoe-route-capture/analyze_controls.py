"""Controls for the routing comparison: chance baselines and length confounds."""

from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from analyze_routes import HB_LAYERS, load, load_per_layer, tv  # noqa: E402

from himoe_libero_bridge.routing import sparse_routes_to_dense  # noqa: E402


def drift_first_n(ids, w, n):
    """Mean consecutive-step TV over the first n control steps only."""
    d = sparse_routes_to_dense(ids[:n], w[:n])
    d = d / d.sum(-1, keepdims=True)
    if d.shape[0] < 2:
        return float("nan")
    return float(tv(d[1:], d[:-1]).mean())


def mannwhitney_exact(a, b):
    """Exact one-sided permutation p that group a > group b."""
    pooled = list(a) + list(b)
    observed = np.mean(a) - np.mean(b)
    n = len(a)
    count = total = 0
    for combo in itertools.combinations(range(len(pooled)), n):
        rest = [i for i in range(len(pooled)) if i not in combo]
        diff = np.mean([pooled[i] for i in combo]) - np.mean([pooled[i] for i in rest])
        total += 1
        if diff >= observed - 1e-12:
            count += 1
    return count / total


def main():
    right = load("/tmp/moe-rollout/right")
    left = load("/tmp/moe-rollout/left")

    print("=" * 74)
    print("对照 A: top-4 Jaccard 的随机基线")
    rng = np.random.default_rng(0)
    js = []
    for _ in range(20000):
        a = set(rng.choice(32, 4, replace=False).tolist())
        b = set(rng.choice(32, 4, replace=False).tolist())
        js.append(len(a & b) / len(a | b))
    print("  两个随机 top-4 子集(32选4)的期望 Jaccard = %.4f" % np.mean(js))
    print("  实测 早层(2-5) 0.53-0.70 / 晚层(12-15) 0.14-0.33 —— 均显著高于随机")

    print("\n" + "=" * 74)
    print("对照 B: 等长窗口下的 drift（消除 episode 长度混淆）")
    n_min = min(e["ids"].shape[0] for e in right + left)
    print("  所有 episode 统一只取前 %d 个 control step\n" % n_min)
    print("%-18s %-5s %-8s %-9s %-9s" % ("group", "ep", "success", "drift_all", "drift_first%d" % n_min))
    rows = []
    for name, g in (("checkpoint-right", right), ("released-left", left)):
        for e in g:
            s = e["summary"]
            d_all = drift_first_n(e["ids"], e["w"], e["ids"].shape[0])
            d_win = drift_first_n(e["ids"], e["w"], n_min)
            rows.append((name, s["success"], d_all, d_win))
            print("%-18s %-5d %-8s %-9.5f %-9.5f"
                  % (name, s["init_state_id"], s["success"], d_all, d_win))

    r_win = [r[3] for r in rows if r[0] == "checkpoint-right"]
    lo_win = [r[3] for r in rows if r[0] == "released-left" and r[1]]
    lb_win = [r[3] for r in rows if r[0] == "released-left" and not r[1]]
    print("\n  等长窗口均值:  right %.5f | left成功 %.5f | left失败 %.5f"
          % (np.mean(r_win), np.mean(lo_win), np.mean(lb_win)))
    print("  right vs left(全部)      精确置换检验 p = %.4f"
          % mannwhitney_exact(r_win, lo_win + lb_win))
    print("  left成功 vs left失败      精确置换检验 p = %.4f"
          % mannwhitney_exact(lo_win, lb_win))

    print("\n" + "=" * 74)
    print("对照 C: drift 是否随 episode 内进程衰减（若是，长 episode 天然低 drift）")
    for name, g, keep in (
        ("right(成功)", right, True),
        ("left(失败)", left, False),
    ):
        sel = [e for e in g if e["summary"]["success"] == keep]
        if not sel:
            continue
        seq = []
        for e in sel:
            d = sparse_routes_to_dense(e["ids"], e["w"])
            d = d / d.sum(-1, keepdims=True)
            seq.append(tv(d[1:], d[:-1]).mean(axis=(1, 2, 3)))
        m = min(len(s) for s in seq)
        avg = np.mean([s[:m] for s in seq], axis=0)
        print("  %-12s 逐步 drift 前%d步: %s" % (name, m, np.round(avg, 3).tolist()))

    print("\n" + "=" * 74)
    print("对照 D: 层深梯度是否只是晚层路由本身更不稳定？")
    print("  用同一 layout 内不同 episode 之间的 Jaccard 作为该层的基准变异")
    for li, layer in enumerate(HB_LAYERS):
        from analyze_routes import topk_set_jaccard
        within = []
        for i, j in itertools.combinations(range(len(right)), 2):
            within.append(topk_set_jaccard(right[i]["ids"][0, :, li], right[j]["ids"][0, :, li]))
        across = []
        for ep in range(len(right)):
            across.append(topk_set_jaccard(right[ep]["ids"][0, :, li], left[ep]["ids"][0, :, li]))
        print("  layer %-3d  组内(right×right) %.3f   跨layout(配对) %.3f   差 %.3f"
              % (layer, np.mean(within), np.mean(across), np.mean(within) - np.mean(across)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
