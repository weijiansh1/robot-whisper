#!/usr/bin/env python3
"""Do failures park on the other tasks too, outside the comparable window?

On t08 failures slow to a near-stop -- mean step size 0.853 against 0.906 for
successes (scene-adjusted z -10.4), the slowest three consecutive steps 0.168
against 0.487, and routing cells held for 7+ steps run at a fifth of the global
speed and are 92% failures.  On the other three tasks the same test is flat, but
those windows are only 9-17 control steps because that is their shortest success,
while their failures run to the horizon.  Whatever happens late in a failure
there is simply outside the window.

No outcome comparison is available past the window -- successes have ended, so
"still running" is the label -- and this makes no attempt at one.  It only asks
what the failure trajectories themselves do: speed against progress through the
episode, for failures and (over their own shorter lives) successes, plus how much
of a failure is spent below a fixed slow threshold.
"""

from __future__ import annotations

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "himoe-routing-rules-20260819/data"


def main() -> int:
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        nr = d["n_rows"].astype(int)
        y = d["success"]
        if y.all():
            continue
        off = np.concatenate([[0], np.cumsum(nr)[:-1]])
        raw = d["proprio"]
        mu, sd = raw.mean(0), raw.std(0) + 1e-9
        print("%s  失败 %d 局（长度 %d-%d），成功 %d 局（%d-%d）"
              % (str(d["task"])[:52], (~y).sum(), nr[~y].min(), nr[~y].max(),
                 y.sum(), nr[y].min(), nr[y].max()))
        print("  局内进度        0-20%  20-40% 40-60% 60-80% 80-100%   末5步   慢步占比")
        for lab, sel in (("失败", ~y), ("成功", y)):
            prof, tail, slow = [[] for _ in range(5)], [], []
            for i in np.flatnonzero(sel):
                p = (raw[off[i]:off[i] + nr[i]] - mu) / sd
                v = np.linalg.norm(np.diff(p, axis=0), axis=-1)
                if len(v) < 6:
                    continue
                q = np.floor(np.linspace(0, 4.999, len(v))).astype(int)
                for b in range(5):
                    prof[b].append(v[q == b].mean())
                tail.append(v[-5:].mean())
                slow.append((v < 0.25).mean())
            print("  %s          %s   %.3f   %5.1f%%"
                  % (lab, "  ".join("%.3f" % np.mean(b) for b in prof),
                     np.mean(tail), 100 * np.mean(slow)))
        print()
    print("注：越过窗口之后没有成败对照可做——成功局已经结束，「还在跑」就是标签。")
    print("    这里只是描述失败轨迹自己的形状，不是一个结局检验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
