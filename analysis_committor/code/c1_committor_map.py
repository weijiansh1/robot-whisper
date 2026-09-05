"""C1 — 初始态 committor 分布与命运的方差分解。

问题：一条 rollout 的成败，有多少是**初始状态**决定的，有多少是**噪声**决定的？

cell = (corpus, suite, task, scene)；cell 内 K 个同胞共享初始状态。
ĉ_i = cell 内失败比例 = 初始态处的经验 committor（K=8 / 32 的分辨率）。

ICC（二元 ANOVA 估计）给出"命运由状态决定"的份额。ICC≈1 ⇒ 噪声几乎不改变结局，
候选选择在原理上就没有空间；ICC 明显小于 1 ⇒ 存在真正的 on-the-fence 状态，
候选选择的问题才是良定义的（这正是 C2 要用的人群）。

输出 out/c1_committor.json + out/c1_cells.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def icc_binary(counts_fail, counts_n):
    """二元 ICC 的 ANOVA 矩估计（Fleiss/Cuzick 口径，等价于 one-way random effects）。

    counts_fail[i], counts_n[i] = 第 i 个 cell 的失败数与同胞数。
    返回 (icc, p_bar, var_between_hat, var_within_hat)。
    """
    k = np.asarray(counts_n, dtype=float)
    x = np.asarray(counts_fail, dtype=float)
    N = k.sum()
    p = x.sum() / N
    # 组间平方和（按 Fleiss 的 unequal-size one-way ANOVA）
    k0 = (N - (k ** 2).sum() / N) / (len(k) - 1) if len(k) > 1 else np.nan
    msb = ((x - k * p) ** 2 / k).sum() / (len(k) - 1) if len(k) > 1 else np.nan
    msw = (x * (1 - x / k)).sum() / (N - len(k)) if N > len(k) else np.nan
    s2b = (msb - msw) / k0
    icc = s2b / (s2b + msw) if (s2b + msw) > 0 else 0.0
    return float(icc), float(p), float(s2b), float(msw)


def run(corpus, loop_only_valid=True):
    rows = []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task)
        eps, Y, nq = C.outcomes(t)
        _, sc, rp = C.episode_cells(t)
        for s in np.unique(sc):
            if s < 0:
                continue
            m = sc == s
            rows.append(
                dict(
                    corpus=corpus,
                    suite=suite,
                    task=task,
                    scene=int(s),
                    K=int(m.sum()),
                    n_fail=int(Y["fail"][m].sum()),
                    n_loop=int(Y["loop"][m].sum()),
                    n_static=int(Y["static"][m].sum()),
                    n_anyloop=int(Y["anyloop"][m].sum()),
                    loop_valid=int(t.loop_valid),
                    med_nq=float(np.median(nq[m])),
                )
            )
    return rows


def summarize(rows, label, key="n_fail"):
    K = np.array([r["K"] for r in rows], float)
    X = np.array([r[key] for r in rows], float)
    c = X / K
    icc, pbar, s2b, s2w = icc_binary(X, K)
    det0 = int((X == 0).sum())
    det1 = int((X == K).sum())
    sens = int(((X > 0) & (X < K)).sum())
    # 事件落在哪类 cell
    ev_in_sens = float(X[(X > 0) & (X < K)].sum() / max(X.sum(), 1))
    return dict(
        label=label,
        key=key,
        n_cells=len(rows),
        K_median=float(np.median(K)),
        base_rate=float(pbar),
        icc=icc,
        var_between=s2b,
        var_within=s2w,
        cells_all_clean=det0,
        cells_all_event=det1,
        cells_sensitive=sens,
        frac_cells_sensitive=sens / max(len(rows), 1),
        frac_events_in_sensitive_cells=ev_in_sens,
        chat_quantiles={
            q: float(np.quantile(c, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        n_cells_chat_mid=int(((c >= 0.25) & (c <= 0.75)).sum()),
    )


def main():
    all_rows = []
    summ = []
    for corpus in ("main16x32", "grid50x8"):
        rows = run(corpus)
        all_rows += rows
        for key in ("n_fail", "n_loop", "n_static", "n_anyloop"):
            sub = rows
            if key in ("n_loop", "n_anyloop"):
                sub = [r for r in rows if r["loop_valid"]]
            summ.append(summarize(sub, corpus, key))
        # 逐 suite 的失败口径
        for suite in sorted({r["suite"] for r in rows}):
            summ.append(summarize([r for r in rows if r["suite"] == suite],
                                  f"{corpus}/{suite}", "n_fail"))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "c1_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    C.jdump(summ, os.path.join(OUT, "c1_committor.json"))

    hdr = f"{'unit':28s} {'key':9s} {'cells':>6s} {'K':>3s} {'base':>6s} {'ICC':>6s} {'sens':>6s} {'%ev in sens':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for s in summ:
        print(
            f"{s['label']:28s} {s['key']:9s} {s['n_cells']:6d} {s['K_median']:3.0f} "
            f"{s['base_rate']:6.3f} {s['icc']:6.3f} {s['cells_sensitive']:6d} "
            f"{s['frac_events_in_sensitive_cells']:11.3f}"
        )


if __name__ == "__main__":
    main()
