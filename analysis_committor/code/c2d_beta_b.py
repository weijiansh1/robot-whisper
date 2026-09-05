"""C2d — β_B 单独跑：q=0 的 routing 能不能识别"这个初始状态很危险"？

C2 里的 β_B 只在 main16x32 的 2–4 个 task 上算过，功效不足。这里在 grid50x8 的
**40 个 task × 50 个 scene** 上重算，独立单位 = task（每个 task 内 50 个 cell 做一次
Spearman，再对 task 做符号检验）—— 必须限定 task 内，因为 routing 100% 编码任务身份。

两个窗口：q=0（与 β_W 完全同一时刻，可直接对照）与 q∈[0,4] 均值（放宽到"开局几步"）。
"""
import os
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"


def cell_table(corpus, ykey, qlo, qhi):
    rows = []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=False)
        if ykey in ("loop", "anyloop") and not t.loop_valid:
            continue
        eps, Yd, _ = C.outcomes(t)
        _, sc, _ = C.episode_cells(t)
        y_all = Yd[ykey]
        epidx = {int(e): i for i, e in enumerate(eps)}
        m = (t.q >= qlo) & (t.q <= qhi)
        ep_m, sc_m = t.ep[m], t.scene[m]
        # 逐 episode 在窗口内取均值，再按 cell 聚合
        for s in np.unique(sc_m):
            sm = sc_m == s
            e_ids = np.unique(ep_m[sm])
            r = dict(suite=suite, task=f"{suite}/{task}", scene=int(s), K=len(e_ids),
                     chat=float(np.mean([y_all[epidx[int(e)]] for e in e_ids])))
            for f in C.FEATS:
                if f not in t.feats:
                    continue
                v = t.feats[f][m][sm]
                r[f] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            rows.append(r)
    return rows


def analyse(rows, min_cells=8):
    res = []
    tasks = sorted({r["task"] for r in rows})
    for f in C.FEATS:
        rhos, names = [], []
        for task in tasks:
            sub = [r for r in rows if r["task"] == task and np.isfinite(r.get(f, np.nan))]
            c = np.array([r["chat"] for r in sub])
            v = np.array([r[f] for r in sub])
            if len(sub) < min_cells or c.std() == 0 or v.std() == 0:
                continue
            rhos.append(stats.spearmanr(v, c).statistic)
            names.append(task)
        if len(rhos) < 5:
            continue
        rhos = np.array(rhos)
        pos = int((rhos > 0).sum())
        p = stats.binomtest(max(pos, len(rhos) - pos), len(rhos), 0.5).pvalue
        # task 级 bootstrap 的均值 CI
        rng = np.random.default_rng(20260904)
        bs = np.array([rng.choice(rhos, len(rhos), replace=True).mean() for _ in range(5000)])
        res.append(dict(feature=f, mean_rho=float(rhos.mean()), median_rho=float(np.median(rhos)),
                        n_tasks=len(rhos), n_pos=pos, p_sign=float(p),
                        ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]))
    res.sort(key=lambda r: -abs(r["mean_rho"]))
    return res


def main():
    report = {}
    for corpus in ("grid50x8", "main16x32"):
        for ykey in ("fail", "loop", "static"):
            for (lo, hi), tag in (((0, 0), "q0"), ((0, 4), "q0_4")):
                rows = cell_table(corpus, ykey, lo, hi)
                res = analyse(rows)
                if not res:
                    continue
                key = f"{corpus}:{ykey}:{tag}"
                report[key] = dict(n_cells=len(rows), n_tasks=len({r['task'] for r in rows}),
                                   results=res)
                print(f"\n=== {key}  cells={len(rows)} tasks={len({r['task'] for r in rows})}")
                for r in res[:5]:
                    star = "*" if r["p_sign"] < 0.05 else " "
                    print(f"  {star}{r['feature']:22s} rho={r['mean_rho']:+.3f} "
                          f"CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                          f"{r['n_pos']}/{r['n_tasks']} 同向 p_sign={r['p_sign']:.4f}", flush=True)
    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / "c2d_beta_b.json")


if __name__ == "__main__":
    main()
