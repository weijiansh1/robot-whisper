"""Stage 3 - fit the composite score, k, h, q_min, K-of-M, DEVELOPMENT ONLY.

Two selections are run and both are frozen before external is opened:

  A. `window`  - maximise excess TP over the cap-free baseline at lead >= 4
                 with false alarms inside the target region [80, 320].
                 This is the objective the protocol names.
  B. `v7budget`- maximise TP at lead >= 4 with false alarms at or below the
                 development-side v7_guard budget.  This is the head-to-head
                 with the 347/57 anchor and is reported separately.

External is never opened here.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402
import capfree_common as C  # noqa: E402
import detectors as D  # noqa: E402

COHORT = "development_main"
LEAD = C.HEADLINE_LEAD
# The slack `k` is swept on both signs of zero.  k < 0 gives the CUSUM a
# positive drift: with an uninformative channel (z == 0) it reduces to
# S_q = |k| q, i.e. *exactly* the cap-free fixed-chunk baseline.  So the
# negative-k arm nests the baseline, and `excess_tp` is precisely the part of
# the score that routing contributes on top of the counter.  The textbook
# k >= 0 arm is reported separately so the drift's contribution is visible.
K_GRID = (-3.0, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.35, -0.25, -0.15,
          -0.05, 0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5)
GREEDY_K = (-1.0, -0.5, -0.25, 0.0, 0.25)
QMIN_GRID = (0, 2, 4, 6)
ALARM_COUNTS = np.unique(np.round(np.geomspace(20, 6000, 60)).astype(int))
POOL_TOP = 60
CORR_MAX = 0.95
MAX_GREEDY = 10
V7_DEV_FP = 38          # v7_guard, development, lead>=4
V7_EXT_FP_SCALED = 54   # 57 external false alarms rescaled to dev negatives

_G: dict = {}


# --------------------------------------------------------------------------


def composite(Zs: np.ndarray, w: np.ndarray, valid: np.ndarray,
              mu=None, sd=None):
    """Weighted sum of sign-oriented channel z's, re-standardised per chunk."""
    s = np.tensordot(w.astype(np.float32), Zs, axes=(0, 0))
    if mu is None:
        mu, sd = C.chunk_stats(s, valid)
    z = (s - mu[None, :]) / sd[None, :]
    z = np.where(np.isfinite(z), z, 0.0)
    return np.where(valid, z, 0.0).astype(np.float32), mu, sd


def objective(rows, mode: str):
    """(value, best_row) for one of the two frozen objectives."""
    if not rows:
        return -10_000, None
    tp = np.array([r[f"tp_lead{LEAD}"] for r in rows])
    fp = np.array([r[f"fp_lead{LEAD}"] for r in rows])
    ex = np.array([r["excess_tp"] for r in rows])
    if mode == "window":
        m = (fp >= C.FP_LO) & (fp <= C.FP_HI)
        if not m.any():
            return -10_000, None
        j = np.where(m)[0][np.argmax(ex[m])]
        return int(ex[j]), rows[j]
    m = fp <= V7_EXT_FP_SCALED
    if not m.any():
        return -10_000, None
    j = np.where(m)[0][np.argmax(tp[m])]
    return int(tp[j]), rows[j]


def sweep_cusum(z, valid, risk, length, base_at, k_grid=K_GRID,
                qmin_grid=QMIN_GRID):
    rows = []
    for k in k_grid:
        S = D.cusum_path(z, valid, k)
        for qm in qmin_grid:
            cm = D.cummax_valid(S, valid, qm)
            thrs = D.alarm_count_grid(cm, ALARM_COUNTS)
            rows += D.sweep(cm, thrs, risk, length, base_at, LEAD,
                            extra={"family": "cusum", "k": k, "q_min": qm})
    return rows


def sweep_kofm(z, valid, risk, length, base_at, thr_grid, qmin_grid=QMIN_GRID):
    rows = []
    for M in (2, 3, 4, 6, 8, 10, 12):
        for thr in thr_grid:
            cnt = D.kofm_path(z, valid, thr, M).astype(np.float32)
            for K in range(1, M + 1):
                for qm in qmin_grid:
                    cm = D.cummax_valid(cnt, valid, qm)
                    first = D.first_from_cummax(cm, K)
                    if (first >= 0).sum() == 0:
                        continue
                    s = C.score(first, risk, length)
                    s.update({"family": "kofm", "K": K, "M": M,
                              "thr": float(thr), "q_min": qm,
                              "excess_tp": int(s[f"tp_lead{LEAD}"]
                                               - base_at(s[f"fp_lead{LEAD}"]))})
                    rows.append(s)
    return rows


def sweep_single(z, valid, risk, length, base_at, qmin_grid=QMIN_GRID):
    rows = []
    for qm in qmin_grid:
        cm = D.cummax_valid(z, valid, qm)
        thrs = D.alarm_count_grid(cm, ALARM_COUNTS)
        rows += D.sweep(cm, thrs, risk, length, base_at, LEAD,
                        extra={"family": "single", "q_min": qm})
    return rows


# --------------------------------------------------------------------------
# greedy forward selection


def _init(Zs, valid, risk, length):
    _G.update(Zs=Zs, valid=valid, risk=risk, length=length)
    _G["base_at"] = C.baseline_frontier(
        C.fixed_chunk_baseline(risk, length), LEAD)


def _try_add(args):
    """Each detector family gets its own forward selection, so K-of-M and the
    single-chunk floor are not handicapped by a CUSUM-chosen channel set."""
    chosen, cand, mode, family = args
    w = np.zeros(_G["Zs"].shape[0], np.float32)
    for c in chosen:
        w[c] = 1.0
    w[cand] = 1.0
    z, _, _ = composite(_G["Zs"], w, _G["valid"])
    a = (z, _G["valid"], _G["risk"], _G["length"], _G["base_at"])
    if family == "cusum":
        rows = sweep_cusum(*a, k_grid=GREEDY_K, qmin_grid=(0, 4))
    elif family == "kofm":
        rows = sweep_kofm(*a, np.quantile(z[_G["valid"]],
                                          [.5, .7, .8, .9, .95, .98, .99]),
                          qmin_grid=(0, 4))
    else:
        rows = sweep_single(*a, qmin_grid=(0, 2, 4, 6))
    val, _ = objective(rows, mode)
    return cand, val


def greedy(pool_idx, mode, pool, log, family="cusum"):
    chosen, best = [], -10_000
    for step in range(MAX_GREEDY):
        cands = [c for c in pool_idx if c not in chosen]
        if not cands:
            break
        with mp.Pool(48, initializer=_init,
                     initargs=(_G["Zs"], _G["valid"], _G["risk"],
                               _G["length"])) as p:
            out = p.map(_try_add,
                        [(tuple(chosen), c, mode, family) for c in cands])
        cand, val = max(out, key=lambda t: t[1])
        if val <= best:
            log.append(f"  [{family}] step {step}: 无改进 ({val} <= {best})，停止")
            break
        chosen.append(cand)
        best = val
        log.append(f"  [{family}] step {step}: + {pool[cand]:44s} 目标={val}")
    return chosen, best


# --------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    f = C.frame(COHORT)
    valid, risk, length = f["valid"], f["risk"], f["length"]
    base = C.fixed_chunk_baseline(risk, length)
    base_at = C.baseline_frontier(base, LEAD)

    names, X = bank.build(COHORT)
    stats = bank.fit_norm(X, valid)
    Zarm = {a: bank.apply_norm(X, valid, stats, a) for a in bank.ARMS}
    del X
    print(f"bank ready {time.time() - t0:.0f}s")

    scr = pd.read_csv(C.RESULTS / "cusum_channel_screen_development.csv")
    idx = {n: i for i, n in enumerate(names)}
    cand = []
    for col, asc in (("win_excess", False), ("v7ext_tp", False)):
        for _, r in scr.sort_values(col, ascending=asc).head(POOL_TOP).iterrows():
            sgn = int(r["win_sign"] if col == "win_excess" else r["v7ext_sign"])
            if sgn == 0:
                continue
            cand.append((r["channel"], r["arm"], sgn))
    seen, pool = set(), []
    for c in cand:
        if c[:2] in seen:
            continue
        seen.add(c[:2])
        pool.append(c)

    # sign-oriented z for every pooled candidate
    Zs = np.stack([np.float32(s) * Zarm[a][idx[n]] for n, a, s in pool])
    del Zarm
    pool_names = [f"{n}|{a}|{'+' if s > 0 else '-'}" for n, a, s in pool]
    print(f"候选池 {len(pool_names)}（去重前 {len(cand)}）")

    # correlation dedup over valid cells
    flat = Zs[:, valid]
    cc = np.corrcoef(flat)
    keep = []
    for i in range(len(pool_names)):
        if all(abs(cc[i, j]) < CORR_MAX for j in keep):
            keep.append(i)
    print(f"相关性去重 |r|<{CORR_MAX} 后保留 {len(keep)}")
    del flat

    _init(Zs, valid, risk, length)
    log = []
    frozen = {}

    # ---- logistic-regression weights on the deduped pool, for comparison ----
    from sklearn.linear_model import LogisticRegression
    Xc = Zs[keep][:, valid].T
    yc = np.repeat(risk[:, None], C.N_CHUNKS, axis=1)[valid]
    wcell = np.repeat((1.0 / length)[:, None], C.N_CHUNKS, axis=1)[valid]
    lr = LogisticRegression(C=0.05, max_iter=400, solver="lbfgs")
    lr.fit(Xc, yc, sample_weight=wcell)
    w_lr = np.zeros(len(pool_names), np.float32)
    w_lr[keep] = lr.coef_[0].astype(np.float32)
    log.append("[lr] 逐 cell 逻辑回归（按 1/length 加权），非零权重 %d"
               % int((np.abs(w_lr) > 1e-3).sum()))

    def plain(r):
        return None if r is None else {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in r.items()}

    for mode in ("window", "v7budget"):
        frozen[mode] = {"families": {}}
        all_rows = []
        for family in ("cusum", "kofm", "single"):
            log.append(f"[{mode}/{family}] 贪心前向选择")
            chosen, _ = greedy(keep, mode, pool_names, log, family)
            w_eq = np.zeros(len(pool_names), np.float32)
            w_eq[chosen] = 1.0
            variants = {"greedy_equal": w_eq}
            if family == "cusum":
                variants["lr_all"] = w_lr
            best_variant, best_val, best_pack = None, -10_000, None
            for vname, w in variants.items():
                z, mu, sd = composite(Zs, w, valid)
                a = (z, valid, risk, length, base_at)
                if family == "cusum":
                    rows = sweep_cusum(*a)
                elif family == "kofm":
                    rows = sweep_kofm(*a, np.quantile(
                        z[valid], [.5, .6, .7, .8, .9, .95, .98, .99]))
                else:
                    rows = sweep_single(*a)
                v, _ = objective(rows, mode)
                log.append(f"  [{family}] 权重方案 {vname}: 目标={v}")
                if v > best_val:
                    best_variant, best_val = vname, v
                    best_pack = (w, z, mu, sd, rows)
            w, z, mu, sd, rows = best_pack
            all_rows += rows
            _, r = objective(rows, mode)
            used = ([pool_names[c] for c in chosen]
                    if best_variant == "greedy_equal"
                    else [pool_names[i] for i in keep])
            wt = ([1.0] * len(chosen) if best_variant == "greedy_equal"
                  else [float(x) for x in w[keep]])
            frozen[mode]["families"][family] = {
                "weight_scheme": best_variant, "channels": used, "weights": wt,
                "norm_mu": mu.tolist(), "norm_sd": sd.tolist(),
                "operating_point": plain(r), "objective_value": int(best_val),
            }
            if family == "cusum":
                # textbook constraint k >= 0, so the drift's share is visible
                _, r0 = objective([x for x in rows if x["k"] >= 0], mode)
                frozen[mode]["families"]["cusum_k>=0"] = {
                    "weight_scheme": best_variant, "channels": used,
                    "weights": wt, "norm_mu": mu.tolist(),
                    "norm_sd": sd.tolist(), "operating_point": plain(r0)}
            if r:
                log.append("  [%s] TP %3d FP %4d 超出基线 %+4d 中位提前 %2.0f  %s"
                           % (family, r[f"tp_lead{LEAD}"], r[f"fp_lead{LEAD}"],
                              r["excess_tp"], r["median_lead"],
                              {k: round(r[k], 4) if isinstance(r[k], float)
                               else r[k]
                               for k in ("k", "q_min", "K", "M", "thr")
                               if k in r}))
        pd.DataFrame(all_rows).to_csv(
            C.RESULTS / f"development_frontier_{mode}.csv", index=False)

    # ---- selection stability: task-disjoint halves of development ----
    tasks = np.unique(f["task"])
    rs = np.random.default_rng(C.SEED)
    perm = rs.permutation(len(tasks))
    halves = [np.isin(f["task"], tasks[perm[:len(tasks) // 2]]),
              np.isin(f["task"], tasks[perm[len(tasks) // 2:]])]
    stab = {}
    for mode in ("window", "v7budget"):
        cfg = frozen[mode]["families"]["cusum"]
        w = np.zeros(len(pool_names), np.float32)
        if cfg["weight_scheme"] == "greedy_equal":
            for nm in cfg["channels"]:
                w[pool_names.index(nm)] = 1.0
        else:
            w[keep] = np.array(cfg["weights"], np.float32)
        z, _, _ = composite(Zs, w, valid, np.array(cfg["norm_mu"]),
                            np.array(cfg["norm_sd"]))
        cf = cfg["operating_point"]
        S = D.cusum_path(z, valid, cf["k"])
        first = D.first_crossing(S, valid, cf["thr"], int(cf["q_min"]))
        per = []
        for hmask in halves:
            b = C.fixed_chunk_baseline(risk[hmask], length[hmask])
            ba = C.baseline_frontier(b, LEAD)
            s = C.score(first[hmask], risk[hmask], length[hmask])
            per.append({"n": int(hmask.sum()), "tp": s[f"tp_lead{LEAD}"],
                        "fp": s[f"fp_lead{LEAD}"],
                        "excess": int(s[f"tp_lead{LEAD}"] - ba(s[f"fp_lead{LEAD}"])),
                        "median_lead": s["median_lead"]})
        stab[mode] = per
        log.append(f"[{mode}] 任务不相交二折稳定性: " + str(per))
    frozen["_stability_task_halves"] = stab

    np.savez_compressed(C.RESULTS / "frozen_pool.npz",
                        pool_names=np.array(pool_names),
                        keep=np.array(keep))
    (C.RESULTS / "frozen_config.json").write_text(json.dumps({
        "cohort_fit": COHORT, "lead": LEAD, "k_grid": list(K_GRID),
        "greedy_k_grid": list(GREEDY_K),
        "qmin_grid": list(QMIN_GRID), "corr_max": CORR_MAX,
        "pool_top": POOL_TOP, "v7_dev_fp": V7_DEV_FP,
        "v7_ext_fp_scaled": V7_EXT_FP_SCALED,
        "selections": frozen,
    }, indent=2))
    print("\n".join(log))
    (C.RESULTS / "fit_log.txt").write_text("\n".join(log))
    print(f"\n完成 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
