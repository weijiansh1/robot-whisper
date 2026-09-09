#!/usr/bin/env python3
"""If failures are trapped, what is the trap -- and does the arm actually stop?

Part B found failures visiting fewer routing cells, switching less, revisiting
more and dwelling nearly twice as long (max dwell 5.31 vs 3.66 steps, scene-
adjusted z +6.2 against a permutation threshold of +-2.6, replicated at K=64 and
K=256).  A matched pose partition shows it just as strongly, so the trapping is
the arm's, not the router's -- but that makes the physical question sharper, not
less interesting.

A fixed-point attractor and a slow passage look identical in a dwell statistic
and are told apart by speed: if the arm's own velocity falls to near zero inside
the long-dwell cells, the policy has parked.  So:

    the speed test      |d proprio / d step| inside long-dwell cells against
                        everywhere else, and for failures against successes
    the trap's location decode the pose of the cells where failures dwell and
                        successes do not, and when they are occupied
    fixed point or cycle  return-time distribution -- a limit cycle returns at a
                        characteristic lag, a fixed point simply never leaves

Only t08 has a window long enough (35 steps) for any of this; on the 9-17 step
tasks max dwell is 1.0-2.0 and the statistic has no range.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_dynamics import DATA, T08, cells, read  # noqa: E402
from analyze_dynamics2 import scene_z  # noqa: E402

DIM = ["eef x", "eef y", "eef z", "轴角1", "轴角2", "轴角3", "指a", "指b"]
K = 64


def main() -> int:
    D = read(DATA / (sys.argv[1] if len(sys.argv) > 1 else T08))
    n, W, y, scene = D["n"], D["W"], D["y"], D["scene"]
    fail = ~y
    raw = D["prop"]
    P = (raw - raw.reshape(-1, 8).mean(0)) / (raw.reshape(-1, 8).std(0) + 1e-9)
    speed = np.zeros((n, W))
    speed[:, 1:] = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    print("%s，%d 局 x %d 步，成功率 %.1f%%\n" % (D["task"], n, W, 100 * y.mean()))

    print("=== 1. 速度检验：卡住的时候手臂还动吗 ===")
    v_ep = speed[:, 1:].mean(1)
    o, z = scene_z(v_ep, fail, scene)
    print("  每步位姿变化量（标准化后）  失败 %.3f   成功 %.3f   场景校正 z %+.2f （阈值 ±2.6）"
          % (v_ep[fail].mean(), v_ep[y].mean(), z))
    slow = np.array([np.min(np.convolve(speed[r, 1:], np.ones(3) / 3, "valid"))
                     for r in range(n)])
    o2, z2 = scene_z(slow, fail, scene)
    print("  一局中最慢的连续 3 步       失败 %.3f   成功 %.3f   场景校正 z %+.2f"
          % (slow[fail].mean(), slow[y].mean(), z2))

    c = cells(D["probs"].reshape(-1, 256), K, n, W)
    print("\n=== 2. 长停留格子里的速度 ===")
    run_id = np.zeros((n, W), int)
    runlen = np.zeros((n, W), int)
    for r in range(n):
        b = np.concatenate([[0], np.flatnonzero(np.diff(c[r])) + 1, [W]])
        for i in range(len(b) - 1):
            run_id[r, b[i]:b[i + 1]] = i
            runlen[r, b[i]:b[i + 1]] = b[i + 1] - b[i]
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 99)):
        m = (runlen >= lo) & (runlen <= hi)
        m[:, 0] = False
        if m.sum() < 50:
            continue
        print("  停留 %-5s 步：占 %5.1f%% 的控制步，格内速度 %.3f （全局 %.3f），"
              "其中失败局占 %.1f%%"
              % ("%d-%d" % (lo, hi) if lo != hi else str(lo), 100 * m.mean(),
                 speed[m].mean(), speed[:, 1:].mean(),
                 100 * np.repeat(fail[:, None], W, 1)[m].mean()))

    print("\n=== 3. 陷阱在哪：失败久留、成功不留的格子 ===")
    rows = []
    for k in range(K):
        mk = c == k
        if mk.sum() < 80:
            continue
        df = np.array([runlen[r][c[r] == k].max() if (c[r] == k).any() else 0
                       for r in np.flatnonzero(fail)])
        ds = np.array([runlen[r][c[r] == k].max() if (c[r] == k).any() else 0
                       for r in np.flatnonzero(y)])
        rows.append((df.mean() - ds.mean(), k, df.mean(), ds.mean(), mk,
                     speed[mk].mean()))
    rows.sort(reverse=True)
    print("  格子  失败最长停留  成功最长停留   格内速度  占用步范围   " +
          "  ".join("%-6s" % d for d in DIM[:3]) + "  夹爪开度")
    for gap, k, df, ds, mk, sp in rows[:5]:
        t = np.flatnonzero(mk.any(0))
        pose = raw[mk].mean(0)
        print("  %3d      %5.2f        %5.2f      %.3f    %2d-%2d      "
              % (k, df, ds, sp, t.min(), t.max())
              + "  ".join("%6.3f" % pose[j] for j in range(3))
              + "   %.4f" % (pose[6] - pose[7]))
    print("  全局平均速度 %.3f" % speed[:, 1:].mean())

    print("\n=== 4. 定点还是极限环：回访间隔 ===")
    for lab, sel in (("失败", fail), ("成功", y)):
        gaps = []
        for r in np.flatnonzero(sel):
            for k in np.unique(c[r]):
                t = np.flatnonzero(c[r] == k)
                b = np.flatnonzero(np.diff(t) > 1)
                gaps.extend(np.diff(t)[b])
        gaps = np.array(gaps)
        if len(gaps) < 20:
            print("  %s：离开后再回来的次数太少（%d）" % (lab, len(gaps)))
            continue
        h = np.bincount(np.clip(gaps, 0, 12), minlength=13)[2:]
        print("  %s：离开后又回到同一格 %d 次，间隔分布(2..12步) %s"
              % (lab, len(gaps), " ".join("%3d" % v for v in h)))
        print("       中位间隔 %.1f 步，其中间隔<=3 步的占 %.0f%%"
              % (np.median(gaps), 100 * (gaps <= 3).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
