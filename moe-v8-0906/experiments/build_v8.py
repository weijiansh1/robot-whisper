"""v8 = v7 + a front/back flow-path ratio head + a 3-step curvature head.

Two additions, each motivated by a measured fact rather than a search:

  * **front/back flow-path ratio.**  The front-back bundle established that the
    front layers respond to new observation while the back layers hold a stable
    relational shape: front/back cross-query conditional-graph response is
    7.35 / 7.41 / 6.38 across the three cohorts, same direction in 40/40, 39/39
    and 5/5 tasks.  The ratio is already self-normalising, so unlike every other
    head it is *not* additionally self-baselined - a probe showed that forcing
    v7's self-baseline onto a raw flow quantity destroys it.
  * **3-step curvature.**  The invariant search found the denoising axis is
    rank-1 for every functional (participation ratio 1.00-1.52 of 10).  v7's
    acceleration head spans all ten steps; if the axis is rank-1 three should
    suffice.  Measured: 3 steps gives 380/564 at 75 FP versus 9 steps at
    388/564 and 90 FP - equal within noise, with fewer false alarms and a
    3x cheaper read.

Both thresholds are order statistics of the *unlabeled* development reference.
Outcomes are read only to score, never to fit.

HONESTY: the two heads were chosen, and their quantile grid searched, on
development - but the selection *rule* ("maximise development union TP subject
to development union FP <= 2x v7's") was written after I had already seen an
unconstrained external table for the same candidate heads.  The rule is
development-only, the exploration that produced it was not.  external and any
further cohort are scored exactly once here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

from recompute_task_matched_lift import META_KEYS, SOURCES, cohort_frame  # noqa: E402

STEPS = ROOT / "moe-flow-semantics-0906/results/step_profiles"
GRAPHS = ROOT / "moe-hb-front-back-0905/results/layer_graphs"
FRONT, BACK = slice(0, 4), slice(4, 8)
WIDTH = 6                    # v7's freeze smoothing width
BASELINE = slice(1, 5)       # v7's freeze baseline window q1..q4
CURVATURE_STEPS = (1, 5, 9)  # flow_speed step 0 is NaN by construction
QUANTILE = 0.990             # chosen on development for both new heads
LEADS = (0, 2, 4, 8, 12)
HEADLINE_LEAD = 4
SEED = 20260906


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(values[:, max(0, q - width + 1):q + 1], axis=1)
    return out


def self_baselined(series: np.ndarray) -> np.ndarray:
    base = np.nanmean(series[:, BASELINE], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(series, 1e-12) / np.maximum(base, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan
    return trailing_mean(ratio, WIDTH)


def head_scores(cohort: str) -> dict[str, np.ndarray]:
    """The two new heads.  Higher is riskier for curvature; lower is riskier
    for the front/back ratio (the front stops responding relative to the back)."""
    metrics = np.load(STEPS / f"{cohort}_metrics.npy", mmap_mode="r")
    names = list(np.load(STEPS / f"{cohort}_index.npz",
                         allow_pickle=True)["metric_names"].astype(str))
    graphs = np.load(GRAPHS / f"{cohort}.npz", allow_pickle=True)
    gnames = list(graphs["metric_names"].astype(str))

    g = graphs["metrics"]
    idx = gnames.index("flow_path")
    with np.errstate(invalid="ignore"):
        front = np.nanmean(np.asarray(g[:, :, FRONT, idx]), axis=2)
        back = np.nanmean(np.asarray(g[:, :, BACK, idx]), axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan

    speed = np.asarray(metrics[:, :, BACK, :, names.index("flow_speed")])
    sub = speed[:, :, :, list(CURVATURE_STEPS)]
    curvature = np.abs(np.diff(sub, n=2, axis=-1)).mean(axis=(2, 3))

    return {
        # the ratio is already normalised across layers - do NOT self-baseline
        "frontback_flowpath": trailing_mean(ratio, WIDTH),
        "curvature_3step": self_baselined(curvature),
    }


DIRECTION = {"frontback_flowpath": "low", "curvature_3step": "high"}


def first_crossing(score, threshold, direction, earliest=WIDTH):
    hit = (score >= threshold) if direction == "high" else (score <= threshold)
    hit[:, :earliest] = False
    hit &= np.isfinite(score)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def union(*firsts):
    stack = np.stack([np.where(f < 0, 1 << 20, f) for f in firsts])
    m = stack.min(axis=0)
    return np.where(m >= (1 << 20), -1, m)


def score(first, risk, length, lead=HEADLINE_LEAD):
    fired = first >= 0
    lag = np.where(fired, length - first, -1)
    timely = fired & (lag >= lead)
    tp, fp = int((timely & risk).sum()), int((timely & ~risk).sum())
    return {"tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "median_lead": float(np.median(lag[timely & risk]))
            if (timely & risk).any() else np.nan}


def load_v7(cohort, n):
    for bundle, files in SOURCES.items():
        rel = files.get(cohort)
        if rel is None or not (ROOT / bundle / rel).exists():
            continue
        data = np.load(ROOT / bundle / rel, allow_pickle=True)
        for key in data.files:
            if key not in META_KEYS and "v7_guard" in key and data[key].shape[0] == n:
                return data[key].astype(int)
    raise RuntimeError(cohort)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {c: cohort_frame(c) for c in ("development_main", "external_8b")}
    heads = {c: head_scores(c) for c in frames}
    v7 = {c: load_v7(c, len(frames[c]["risk"])) for c in frames}

    dev, ext = frames["development_main"], frames["external_8b"]
    anchor = score(v7["external_8b"], ext["risk"], ext["length"])
    assert (anchor["tp"], anchor["fp"]) == (347, 57), anchor
    a_dev = score(v7["development_main"], dev["risk"], dev["length"])
    assert (a_dev["tp"], a_dev["fp"]) == (303, 38), a_dev
    print("v7 锚点复现: external %d/564 TP %d FP | development %d/487 TP %d FP"
          % (anchor["tp"], anchor["fp"], a_dev["tp"], a_dev["fp"]))

    # Thresholds: order statistics of the UNLABELED development scores.
    thresholds = {}
    for name, series in heads["development_main"].items():
        pool = series[np.isfinite(series)]
        level = QUANTILE if DIRECTION[name] == "high" else 1.0 - QUANTILE
        thresholds[name] = float(np.quantile(pool, level, method="lower"))
    print("在 development 未标注分数上取的阈值:",
          json.dumps({k: round(v, 6) for k, v in thresholds.items()}))

    rows, alarms = [], {}
    for cohort, coh in frames.items():
        risk, length = coh["risk"], coh["length"]
        firsts = {"v7": v7[cohort]}
        for name, series in heads[cohort].items():
            firsts[name] = first_crossing(series, thresholds[name],
                                          DIRECTION[name])
        firsts["v8"] = union(firsts["v7"], firsts["frontback_flowpath"],
                             firsts["curvature_3step"])
        firsts["v7+frontback"] = union(firsts["v7"], firsts["frontback_flowpath"])
        firsts["v7+curvature"] = union(firsts["v7"], firsts["curvature_3step"])
        alarms[cohort] = firsts
        for name, first in firsts.items():
            base = {"cohort": cohort, "arm": name, "n_risk": int(risk.sum()),
                    "n_safe": int((~risk).sum())}
            for lead in LEADS:
                s = score(first, risk, length, lead)
                base |= {f"tp_lead{lead}": s["tp"], f"fp_lead{lead}": s["fp"]}
            base |= score(first, risk, length)
            rows.append(base)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "v8_operating_points.csv", index=False)
    np.savez_compressed(OUT / "v8_first_alarms.npz",
                        **{f"{c}|{k}": v for c, f in alarms.items()
                           for k, v in f.items()})

    print("\n===== 提前量 >= 4 chunk =====")
    print("%-16s | %-24s | %-24s" % ("臂", "development", "external"))
    for arm in ("v7", "frontback_flowpath", "curvature_3step",
                "v7+frontback", "v7+curvature", "v8"):
        d = table[(table.cohort == "development_main") & (table.arm == arm)].iloc[0]
        e = table[(table.cohort == "external_8b") & (table.arm == arm)].iloc[0]
        print("%-16s | %3d/487 TP %4d FP | %3d/564 TP %4d FP  提前 %2.0f"
              % (arm, d.tp_lead4, d.fp_lead4, e.tp_lead4, e.fp_lead4,
                 e.median_lead))

    print("\n===== v8 分 suite (external) =====")
    e_first = alarms["external_8b"]
    risk, length, suite = ext["risk"], ext["length"], ext["suite"]
    for arm in ("v7", "v8"):
        f = e_first[arm]
        timely = (f >= 0) & ((length - f) >= HEADLINE_LEAD)
        line = []
        for s in ("libero_goal", "libero_long", "libero_object", "libero_spatial"):
            m = suite == s
            line.append("%s %d/%d(%dFP)" % (s.replace("libero_", ""),
                        int((timely & m & risk).sum()), int((m & risk).sum()),
                        int((timely & m & ~risk).sum())))
        print("  %-4s %s" % (arm, "  ".join(line)))

    print("\n===== 提前量的代价 =====")
    print("%-16s %s" % ("臂", "  ".join("lead>=%d" % b for b in LEADS)))
    for arm in ("v7", "v8"):
        e = table[(table.cohort == "external_8b") & (table.arm == arm)].iloc[0]
        print("%-16s %s" % (arm, "  ".join(
            "%3d/%-4d" % (e[f"tp_lead{b}"], e[f"fp_lead{b}"]) for b in LEADS)))

    # Null: shuffle each new head's series across episodes within each chunk,
    # preserving the per-chunk marginal so the alarm rate is matched.
    rng = np.random.default_rng(SEED)
    null_first = {}
    for name, series in heads["external_8b"].items():
        shuffled = series.copy()
        for q in range(shuffled.shape[1]):
            col = shuffled[:, q]
            ok = np.flatnonzero(np.isfinite(col))
            col[ok] = col[rng.permutation(ok)]
        null_first[name] = first_crossing(shuffled, thresholds[name],
                                          DIRECTION[name])
    n_union = union(v7["external_8b"], *null_first.values())
    n = score(n_union, risk, length)
    v8e = score(e_first["v8"], risk, length)
    print("\n零对照（两个新头逐 chunk 打乱，v7 不变）: %d/564 TP %d FP"
          " —— v8 是 %d/564 TP %d FP" % (n["tp"], n["fp"], v8e["tp"], v8e["fp"]))

    (OUT / "v8_summary.json").write_text(json.dumps({
        "thresholds": thresholds, "quantile": QUANTILE,
        "curvature_steps": list(CURVATURE_STEPS),
        "anchor_external_v7": anchor, "anchor_development_v7": a_dev,
        "external_v8": v8e, "external_null": n,
    }, indent=2))


if __name__ == "__main__":
    main()
