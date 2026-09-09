"""A consensus head: many weak signals at a moderate quantile, fire on m-of-k.

Every head so far thresholds one quantity in its extreme tail.  That is why
quantities with genuinely replicated discriminative power keep buying nothing:
libero_object scores 0.846-0.892 AUC at chunk 9 across all eight layers, and a
chunk-gated head built on it over 5,712 configurations bought +8 TP; the
spectral centroid replicates at 0.601/0.602 against a null floor of 0.048 and
its detector exchange rate is 0.26 TP per FP.  **High AUC is a statement about
the whole ranking; a q99 threshold uses only the top percent, and the
enrichment there is not enough.**

The joint tail of k signals each at a moderate quantile is a different object.
If the signals are close to conditionally independent given success but
co-elevated given failure, requiring m of k simultaneously is far more enriched
than any single q99 tail, at a comparable alarm rate.  That mechanism is
already documented on this corpus in the opposite direction - voting across
published detectors works from base rates rather than from false-alarm
independence - so it is worth testing directly rather than assumed.

Nothing is trained.  Each signal gets an order statistic of the unlabeled
development scores; the rule is "at least m of k are at or above their own
threshold at the same chunk", held for K consecutive chunks.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V8DIR = ROOT / "moe-v8-0906"
sys.path.insert(0, str(V8DIR / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V  # noqa: E402
import freeze_v82 as F  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
LEADS = (0, 4, 8, 12, 16, 20)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
V83 = {"baseline": 2, "width": 6, "confirm": 2, "slope": -0.0015}
STEPS = ROOT / "moe-flow-semantics-0906/results/step_profiles"
SEED = 20260906
# Declared grid.
MEMBER_QUANTILES = (0.75, 0.80, 0.85, 0.90)
M_OF_K = (3, 4, 5, 6)
CONFIRMS = (1, 2)
EARLIEST = 6


def signals(cohort: str) -> dict[str, np.ndarray]:
    """Eight weak signals, each already shown to replicate across cohorts.

    All are causal and use only chunk index plus routing.  Direction is
    normalised here so that *larger means riskier* for every one of them,
    which is what lets a single m-of-k rule combine them.
    """
    mob = np.load(OUT / f"{cohort}_step_mobility.npz")["mobility"]   # [n,52,8,10]
    speed = np.load(V8DIR / "results" / f"{cohort}_flow_speed.npz")["flow_speed"]
    out: dict[str, np.ndarray] = {}

    # per-step mobility at the layers and steps that replicated
    for tag, layers, step in (("mob_back_d9", slice(4, 8), 9),
                              ("mob_front_d3", slice(0, 4), 3),
                              ("mob_back_d6", slice(4, 8), 6)):
        out[tag] = np.nanmean(mob[:, :, layers, step], axis=2)

    # layer-profile shape: curvature across the ordered layer axis
    prof = np.nanmean(mob, axis=3)
    out["layer_curv"] = prof[:, :, 0] - 2 * prof[:, :, 3] + prof[:, :, 7]

    # oscillation: causal spectral centroid of the back-layer sequence
    series = np.nanmean(mob[:, :, 4:, 9], axis=2)
    centroid = np.full_like(series, np.nan, dtype=np.float64)
    window = 8
    freqs = np.arange(window // 2 + 1)
    for q in range(window - 1, series.shape[1]):
        seg = series[:, q - window + 1:q + 1]
        centred = np.nan_to_num(seg - np.nanmean(seg, axis=1, keepdims=True))
        power = np.abs(np.fft.rfft(centred, axis=1)) ** 2
        total = power.sum(axis=1)
        val = np.full(series.shape[0], np.nan)
        ok = total > 1e-12
        val[ok] = (power[ok] * freqs).sum(axis=1) / total[ok]
        val[np.isnan(seg).any(axis=1)] = np.nan
        centroid[:, q] = val
    out["spec_centroid"] = centroid

    # v8's two heads, sign-normalised so larger is riskier
    heads = F.build_heads(speed, V83["baseline"], V83["width"])
    out["frontback_inv"] = -heads["frontback_flowpath"]   # head fires on `low`
    out["curv3"] = heads["curvature_3step"]

    # state channel: v7 and v8 never read token 0
    state = np.asarray(np.load(STEPS / f"{cohort}_state_mobility.npy",
                               mmap_mode="r")[:, :, :, 9]) \
        if (STEPS / f"{cohort}_state_mobility.npy").exists() else None
    if state is not None:
        out["state_low"] = -np.nanmean(state[:, :, 4:], axis=2)
    return out


def consensus_first(sigs, thresholds, m, confirm, earliest=EARLIEST):
    """At least `m` signals at or above their own threshold at the same chunk,
    held for `confirm` consecutive chunks.  `>=`, never strict."""
    votes = None
    for name, series in sigs.items():
        hit = np.isfinite(series) & (series >= thresholds[name])
        votes = hit.astype(np.int16) if votes is None else votes + hit
    fire = votes >= m
    fire[:, :earliest] = False
    held = np.zeros_like(fire)
    run = np.zeros(fire.shape[0], dtype=int)
    for q in range(fire.shape[1]):
        run = np.where(fire[:, q], run + 1, 0)
        held[:, q] = run >= confirm
    return np.where(held.any(axis=1), held.argmax(axis=1), -1)


def main() -> None:
    data = {c: V.load_cohort(c) for c in COHORTS}
    sig = {c: signals(c) for c in COHORTS}
    names = sorted(set.intersection(*(set(s) for s in sig.values())))
    print("共识头的成员信号 (%d 个): %s" % (len(names), ", ".join(names)))
    sig = {c: {n: s[n] for n in names} for c, s in sig.items()}

    heads = {c: F.build_heads(d["speed"], V83["baseline"], V83["width"])
             for c, d in data.items()}
    thr83 = F.thresholds_from(heads["development_main"])
    base = {}
    for cohort, d in data.items():
        firsts = [F.moving_first(heads[cohort][h], thr83[h], V.DIRECTION[h],
                                 V83["slope"], V83["confirm"], V83["width"])
                  for h in heads[cohort]]
        base[cohort] = V.union(d["v7"], *firsts)

    def profile(alarms):
        prof = {lead: [0, 0] for lead in LEADS}
        for cohort, first in alarms.items():
            d = data[cohort]
            for lead in LEADS:
                s = V.score(first, d["risk"], d["length"], lead)
                prof[lead][0] += s["tp"]
                prof[lead][1] += s["fp"]
        return prof

    p83 = profile(base)
    dev = data["development_main"]
    d0 = V.score(base["development_main"], dev["risk"], dev["length"], 0)
    d12 = V.score(base["development_main"], dev["risk"], dev["length"], 12)
    print("v8.3 全量 " + "  ".join("L%d %d/%d" % (b, *p83[b]) for b in LEADS))
    print("  development: L0 %d/487 %dFP | L12 %d/487" % (d0["tp"], d0["fp"], d12["tp"]))

    rows = []
    for mq, m, confirm in itertools.product(MEMBER_QUANTILES, M_OF_K, CONFIRMS):
        if m > len(names):
            continue
        thresholds = {}
        for name in names:
            pool = sig["development_main"][name]
            pool = pool[np.isfinite(pool)]
            thresholds[name] = float(np.quantile(pool, mq, method="lower"))
        head = consensus_first(sig["development_main"], thresholds, m, confirm)
        u = V.union(base["development_main"], head)
        s0 = V.score(u, dev["risk"], dev["length"], 0)
        s12 = V.score(u, dev["risk"], dev["length"], 12)
        rows.append({"member_q": mq, "m": m, "confirm": confirm,
                     "dev0_tp": s0["tp"], "dev0_fp": s0["fp"],
                     "dev12_tp": s12["tp"],
                     "d_tp": s0["tp"] - d0["tp"], "d_fp": s0["fp"] - d0["fp"]})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "consensus_grid.csv", index=False)
    print("\ndevelopment 网格 (成员分位 x m x 确认):")
    print(grid.to_string(index=False))

    # Declared rule: exchange rate at lead >= 0 must be at least 2.0 and the
    # development false alarms may rise by at most 8; among those, maximise the
    # lead >= 12 count.  The lead >= 0 constraint is what stops a long-lead
    # objective from degenerating into a length filter.
    ok = grid[(grid.d_fp <= 8) & (grid.d_tp > 0) &
              (grid.d_tp >= 2.0 * grid.d_fp.clip(lower=1))]
    if not len(ok):
        ok = grid[(grid.d_fp <= 8) & (grid.d_tp > 0)]
        print("\n没有配置达到 2.0 的兑换率；放宽到仅要求 新增FP<=8 且 TP 有增")
    if not len(ok):
        print("\n没有配置通过 —— 共识头不成立，v8.3 保持不变")
        return
    pick = ok.sort_values(["dev12_tp", "d_fp"], ascending=[False, True]).iloc[0]
    print("\n选中: 成员分位 %.2f, m=%d/%d, 确认 K=%d"
          % (pick.member_q, pick.m, len(names), pick.confirm))
    print("  development 净增 (lead>=0): +%d TP / +%d FP" % (pick.d_tp, pick.d_fp))

    thresholds = {}
    for name in names:
        pool = sig["development_main"][name]
        pool = pool[np.isfinite(pool)]
        thresholds[name] = float(np.quantile(pool, float(pick.member_q),
                                             method="lower"))
    alarms = {}
    for cohort in COHORTS:
        head = consensus_first(sig[cohort], thresholds, int(pick.m),
                               int(pick.confirm))
        alarms[cohort] = V.union(base[cohort], head)
    p9 = profile(alarms)
    np.savez_compressed(OUT / "consensus_alarms.npz",
                        **{f"{c}|v9c": v for c, v in alarms.items()})

    print("\n===== 全量 32,960 episodes / 1,358 失败 =====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%-2d" % b for b in LEADS)))
    print("%-6s %s" % ("v8.3", "  ".join("%4d/%-4d" % tuple(p83[b]) for b in LEADS)))
    print("%-6s %s" % ("v9c", "  ".join("%4d/%-4d" % tuple(p9[b]) for b in LEADS)))

    print("\n===== 分 cohort (lead>=4) =====")
    for cohort in COHORTS:
        d = data[cohort]
        a = V.score(base[cohort], d["risk"], d["length"], 4)
        b = V.score(alarms[cohort], d["risk"], d["length"], 4)
        print("  %-18s %3d/%-4d %3dFP -> %3d/%-4d %3dFP  (%+d/%+d)"
              % (cohort, a["tp"], int(d["risk"].sum()), a["fp"], b["tp"],
                 int(d["risk"].sum()), b["fp"], b["tp"] - a["tp"], b["fp"] - a["fp"]))

    print("\n===== 分 suite =====")
    for suite in SUITES:
        line, n = [], 0
        for lead in (4, 12, 16):
            agg = {"v8.3": 0, "v9c": 0}
            fp = 0
            for cohort, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                if lead == 4:
                    n += int((m & d["risk"]).sum())
                for arm, first in (("v8.3", base[cohort]), ("v9c", alarms[cohort])):
                    t = (first >= 0) & ((d["length"] - first) >= lead)
                    agg[arm] += int((t & m & d["risk"]).sum())
                    if arm == "v9c":
                        fp += int((t & m & ~d["risk"]).sum())
            line.append((lead, agg["v8.3"], agg["v9c"], fp))
        print("  %-16s /%3d  " % (suite, n) +
              "  ".join("L%d %3d->%3d(%dFP)" % c for c in line))

    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for cohort in COHORTS:
        shuffled = {}
        for name, series in sig[cohort].items():
            s = series.copy()
            for q in range(s.shape[1]):
                col = s[:, q]
                idx = np.flatnonzero(np.isfinite(col))
                col[idx] = col[rng.permutation(idx)]
            shuffled[name] = s
        head = consensus_first(shuffled, thresholds, int(pick.m), int(pick.confirm))
        s = V.score(V.union(base[cohort], head), data[cohort]["risk"],
                    data[cohort]["length"], 4)
        ntp += s["tp"]
        nfp += s["fp"]
    print("\n零对照(每个成员信号逐 chunk 打乱): %d/1358 TP %d FP  —— v9c 是 %d/1358 %d FP"
          % (ntp, nfp, *p9[4]))

    (OUT / "consensus_summary.json").write_text(json.dumps({
        "members": names, "member_quantile": float(pick.member_q),
        "m": int(pick.m), "confirm": int(pick.confirm),
        "thresholds": thresholds,
        "v83_profile": {str(b): list(p83[b]) for b in LEADS},
        "v9c_profile": {str(b): list(p9[b]) for b in LEADS},
        "null": {"tp": ntp, "fp": nfp},
    }, indent=2))


if __name__ == "__main__":
    main()
