"""C5 — 竞争风险：一条 rollout 的首次到达是 loop、static、完成，还是撞上观察边界？

框架 §2/§10 的直接落地。对每一集取**首次到达**：

  T = min(loop_onset, static_onset, 完成时刻)，类型 = 取到最小值的那一个；
  既没有任何 trap 事件、又没完成 ⇒ **右删失**（纯 timeout），不是第三类失败。

三个逐 query 的 cause-specific hazard：

  λ_k(q) = (在 q 处首次到达 k 的集数) / (在 q 处仍在风险集里的集数),  k ∈ {L,S,C}

由它给出累积发生率 F_k(q) = Σ_{j≤q} λ_k(j)·S(j−1)（竞争风险下不能用 1−KM）。

顺带读出两个数：
  · **逃逸 committor** e_k = P(最终完成 | 首次到达 k)，loop 与 static 各一个 —— 这是
    "两种 Trap 深度不同"最直接的量化；
  · **纯 timeout 占失败的比例** —— "Timeout ≠ Failure" 这句话在本语料上的规模。

只读 events/，不碰 npz，秒级。
"""
import os
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
QMAX = 60
# query 预算上限 = SUITE_MAX_STEPS / replan(10)。出处 VLA_MUI_HUB/README.md
# （spatial 220 / object 280 / goal 300 / long 520），replan=10 见各 run 的 meta.json。
SUITE_CAP = {"libero_spatial": 22, "libero_object": 28, "libero_goal": 30, "libero_long": 52}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def episode_records(corpus):
    recs = []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=False)
        eps, _, _ = C.outcomes(t)
        _, sc, _ = C.episode_cells(t)
        for i, e in enumerate(eps):
            ev = t.ev[e]
            lo = ev["loop_onset_q"] if (ev["loop_onset_q"] >= 0 and t.loop_valid) else None
            st = ev["static_onset_q"] if ev["static_onset_q"] >= 0 else None
            comp = ev["n_queries"] - 1 if ev["success"] else None
            cands = [(v, k) for v, k in ((lo, "L"), (st, "S"), (comp, "C")) if v is not None]
            if cands:
                T, kind = min(cands)
            else:
                T, kind = ev["n_queries"] - 1, "censored"
            recs.append(dict(suite=suite, task=task, scene=int(sc[i]), episode=int(e),
                             loop_valid=int(t.loop_valid), cap=SUITE_CAP[suite] - 1,
                             T=int(T), kind=kind, success=int(ev["success"]),
                             n_queries=int(ev["n_queries"]),
                             loop_onset=ev["loop_onset_q"], static_onset=ev["static_onset_q"]))
    return recs


def cause_specific(recs):
    T = np.array([r["T"] for r in recs])
    kind = np.array([r["kind"] for r in recs])
    n = len(recs)
    at_risk, lam, F = [], {k: [] for k in "LSC"}, {k: [] for k in "LSC"}
    S = 1.0
    cum = {k: 0.0 for k in "LSC"}
    for q in range(QMAX):
        risk = int(np.sum(T >= q))
        at_risk.append(risk)
        tot = 0.0
        for k in "LSC":
            d = int(np.sum((T == q) & (kind == k)))
            h = d / risk if risk else 0.0
            lam[k].append(h)
            cum[k] += S * h
            F[k].append(cum[k])
            tot += h
        S *= (1 - tot)
    return at_risk, lam, F, n


def main():
    report = {}
    for corpus in ("main16x32", "grid50x8"):
        recs = episode_records(corpus)
        at_risk, lam, F, n = cause_specific(recs)
        n_fail = sum(1 - r["success"] for r in recs)
        n_cens = sum(r["kind"] == "censored" for r in recs)
        # 逃逸 committor：首次到达 k 之后仍完成任务的比例
        esc = {}
        for k, name in (("L", "loop"), ("S", "static")):
            sub = [r for r in recs if r["kind"] == k]
            if not sub:
                continue
            s = sum(r["success"] for r in sub)
            esc[name] = dict(n=len(sub), n_escaped=s, e=s / len(sub),
                             ci=list(wilson(s, len(sub))))
        # 按首达时刻分箱的逃逸 committor
        esc_by_t = {}
        for k, name in (("L", "loop"), ("S", "static")):
            rows = []
            for a, b in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 100)):
                sub = [r for r in recs if r["kind"] == k and a <= r["T"] < b]
                if len(sub) < 5:
                    continue
                s = sum(r["success"] for r in sub)
                rows.append(dict(bin=f"[{a},{b})", n=len(sub), e=s / len(sub),
                                 ci=list(wilson(s, len(sub)))))
            esc_by_t[name] = rows
        # 等预算口径的逃逸率：首达 k 之后**固定 W 个 query** 内是否完成，且只保留
        # 剩余预算 ≥ W 的集。晚发生的 trap 剩余步数天然更少，不这么切就是在读预算。
        esc_eqb = {}
        for k, name in (("L", "loop"), ("S", "static")):
            for W in (10, 15):
                rows = []
                for a, b in ((0, 10), (10, 20), (20, 30), (30, 100)):
                    sub = [r for r in recs if r["kind"] == k and a <= r["T"] < b
                           and (r["cap"] - r["T"]) >= W]
                    if len(sub) < 5:
                        continue
                    s = sum(1 for r in sub if r["success"] and (r["n_queries"] - 1 - r["T"]) <= W)
                    rows.append(dict(bin=f"[{a},{b})", n=len(sub), e=s / len(sub),
                                     ci=list(wilson(s, len(sub)))))
                esc_eqb[f"{name}_W{W}"] = rows
        # 纯 timeout（无任何 trap 事件的失败）
        pure_to = [r for r in recs if r["kind"] == "censored" and not r["success"]]
        by_suite = {}
        for suite in sorted({r["suite"] for r in recs}):
            sr = [r for r in recs if r["suite"] == suite]
            f = [r for r in sr if not r["success"]]
            pt = [r for r in f if r["kind"] == "censored"]
            by_suite[suite] = dict(n=len(sr), n_fail=len(f), n_pure_timeout=len(pt),
                                   frac_pure_timeout=len(pt) / max(len(f), 1))
        report[corpus] = dict(
            n_episodes=n, n_fail=n_fail, n_censored=n_cens,
            first_passage_counts={k: int(sum(r["kind"] == k for r in recs))
                                  for k in ("L", "S", "C", "censored")},
            escape_committor=esc, escape_by_first_passage_time=esc_by_t,
            escape_equal_budget=esc_eqb,
            pure_timeout_n=len(pure_to),
            pure_timeout_frac_of_failures=len(pure_to) / max(n_fail, 1),
            by_suite=by_suite,
            at_risk=at_risk, hazard=lam, cuminc=F,
        )

        print(f"\n########## {corpus}   n={n}  失败={n_fail}")
        fp = report[corpus]["first_passage_counts"]
        print(f"  首次到达： loop={fp['L']}  static={fp['S']}  完成={fp['C']}  删失(纯timeout)={fp['censored']}")
        print(f"  纯 timeout 占全部失败 {report[corpus]['pure_timeout_frac_of_failures']:.1%} "
              f"({len(pure_to)}/{n_fail})")
        print("  逃逸 committor  e_k = P(最终完成 | 首达 k)：")
        for name, v in esc.items():
            print(f"    {name:7s} n={v['n']:4d}  e={v['e']:.3f}  95%CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
        for name, rows in esc_by_t.items():
            if rows:
                print(f"    {name} 按首达时刻: " +
                      "  ".join(f"{r['bin']}:{r['e']:.2f}(n={r['n']})" for r in rows))
        print("  等预算（首达后固定 W 个 query 内完成，且剩余预算 ≥W）：")
        for name, rows in esc_eqb.items():
            if rows:
                print(f"    {name:12s} " +
                      "  ".join(f"{r['bin']}:{r['e']:.2f}(n={r['n']})" for r in rows))
        # loop vs static 深度差的检验
        if "loop" in esc and "static" in esc:
            tab = [[esc["loop"]["n_escaped"], esc["loop"]["n"] - esc["loop"]["n_escaped"]],
                   [esc["static"]["n_escaped"], esc["static"]["n"] - esc["static"]["n_escaped"]]]
            p = stats.fisher_exact(tab).pvalue
            report[corpus]["loop_vs_static_escape_fisher_p"] = float(p)
            print(f"    loop vs static 逃逸率差异 Fisher p={p:.3e}")
        print("  逐 suite 纯 timeout 占失败比：")
        for s, v in by_suite.items():
            print(f"    {s:16s} 失败 {v['n_fail']:4d}  纯timeout {v['n_pure_timeout']:4d} "
                  f"({v['frac_pure_timeout']:.1%})")
        print("  cause-specific hazard（q=5..40 抽样）：")
        print("    q      " + " ".join(f"{q:6d}" for q in range(5, 45, 5)))
        for k in "LSC":
            print(f"    λ_{k}    " + " ".join(f"{lam[k][q]:6.3f}" for q in range(5, 45, 5)))
        print("    n_risk " + " ".join(f"{at_risk[q]:6d}" for q in range(5, 45, 5)))

    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / "c5_competing_risks.json")


if __name__ == "__main__":
    main()
