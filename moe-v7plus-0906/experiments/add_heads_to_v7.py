"""Add heads to v7 and measure what each one buys, cap-free.

Three candidate modifications, each calibrated on development only and scored
on external exactly once:

  A. front/back structural head - the front-back bundle's strongest replicated
     ratio (front cross-query conditional-graph response over back, 7.35 /
     7.41 / 6.38 across the three cohorts, same direction in 40/40, 39/39 and
     5/5 tasks).  A probe already put this whole family at only +20 of v7's
     217 misses, so this is a confirmation, not a hope.
  B. flow_path head - the *first-order* quantity on the denoising axis.  v7's
     acceleration head already uses that axis but only its *second* order
     (curvature).  A probe put `flow_path|global|0.95` at +117 of 217.
  C. step-reduced v7 - the invariant search found the denoising axis is rank-1
     for every functional (participation ratio 1.00-1.52 of 10).  Two of v7's
     three heads already read step 9 only; only acceleration spans all ten.
     If the axis really is rank-1 the curvature should survive subsampling.

Each head is built in v7's own idiom so the comparison is about the quantity,
not the machinery: self-baseline over q1..q4, causal trailing mean, and a
threshold that is an order statistic of the *unlabeled* reference corpus.
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
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
FRONT, BACK = slice(0, 4), slice(4, 8)
BASELINE = slice(1, 5)     # v7's freeze baseline window, q1..q4
WIDTH = 6                  # v7's freeze smoothing
LEAD = 4
# Calibrated on development only; a grid, not a fit.
QUANTILES = (0.90, 0.95, 0.975, 0.99)


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Causal width-`width` mean, ignoring NaN, matching v7's convention."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    n = values.shape[1]
    for q in range(n):
        lo = max(0, q - width + 1)
        block = values[:, lo:q + 1]
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(block, axis=1)
    return out


def self_baselined(series: np.ndarray) -> np.ndarray:
    """log(value / this episode's own q1..q4 mean), then smooth.

    This is exactly what makes v7 task-agnostic: the scale of any routing
    quantity is task-dependent, but its ratio to the same episode's opening
    regime is not.
    """
    base = np.nanmean(series[:, BASELINE], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(series, 1e-12) / np.maximum(base, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan
    return trailing_mean(ratio, WIDTH)


def first_crossing(score: np.ndarray, threshold: float, direction: str,
                   earliest: int) -> np.ndarray:
    """First chunk at or beyond the threshold.  `>=`, never `>`: strict `>`
    silently drops the whole tie group at the threshold."""
    hit = (score >= threshold) if direction == "high" else (score <= threshold)
    hit[:, :earliest] = False
    hit &= np.isfinite(score)
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1), -1)


def score_capfree(first: np.ndarray, risk: np.ndarray,
                  length: np.ndarray) -> dict:
    fired = first >= 0
    lead = np.where(fired, length - first, -1)
    timely = fired & (lead >= LEAD)
    tp, fp = int((timely & risk).sum()), int((timely & ~risk).sum())
    return {
        "tp": tp, "fp": fp,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "median_lead": float(np.median(lead[timely & risk]))
        if (timely & risk).any() else np.nan,
    }


def load_v7(cohort: str, n: int) -> np.ndarray:
    for bundle, files in SOURCES.items():
        rel = files.get(cohort)
        if rel is None or not (ROOT / bundle / rel).exists():
            continue
        data = np.load(ROOT / bundle / rel, allow_pickle=True)
        for key in data.files:
            if key in META_KEYS:
                continue
            if "v7_guard" in key and data[key].shape[0] == n:
                return data[key].astype(int)
    raise RuntimeError(f"no v7_guard vector for {cohort}")


def build_candidates(cohort: str) -> dict[str, tuple[np.ndarray, str, int]]:
    """(score series, direction that means risk, earliest admissible chunk)."""
    metrics = np.load(STEPS / f"{cohort}_metrics.npy", mmap_mode="r")
    names = list(np.load(STEPS / f"{cohort}_index.npz",
                         allow_pickle=True)["metric_names"].astype(str))
    graphs = np.load(GRAPHS / f"{cohort}.npz", allow_pickle=True)
    gnames = list(graphs["metric_names"].astype(str))

    out: dict[str, tuple[np.ndarray, str, int]] = {}

    # A. front/back structural ratio, in the log domain so it is a difference.
    g = graphs["metrics"]
    for name in ("conditional_query_d1", "conditional_energy", "flow_path"):
        if name not in gnames:
            continue
        idx = gnames.index(name)
        front = np.nanmean(np.asarray(g[:, :, FRONT, idx]), axis=2)
        back = np.nanmean(np.asarray(g[:, :, BACK, idx]), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
        ratio[~np.isfinite(ratio)] = np.nan
        out[f"A_frontback_{name}"] = (
            trailing_mean(ratio, WIDTH), "low", WIDTH)

    # B. flow_path: the first-order quantity on the axis whose second order
    #    (curvature) v7 already uses.
    fp_idx = names.index("flow_speed")
    for tag, layers in (("front", FRONT), ("back", BACK)):
        speed = np.asarray(metrics[:, :, layers, :, fp_idx])
        path = np.nansum(speed[:, :, :, 1:], axis=3).mean(axis=2)  # 9 valid steps
        out[f"B_flowpath_{tag}"] = (self_baselined(path), "high", WIDTH)

    # C. step-reduced curvature: v7's acceleration head on a subsample of the
    #    denoising axis.  If the axis is rank-1 this should barely move.
    for tag, keep in (("all9", range(1, 10)), ("s4", (1, 4, 7, 9)),
                      ("s3", (1, 5, 9)), ("s2", (1, 9))):
        speed = np.asarray(metrics[:, :, BACK, :, fp_idx])[:, :, :, list(keep)]
        if speed.shape[-1] < 3:
            # second difference needs three points; fall back to first order
            curv = np.abs(np.diff(speed, axis=-1)).mean(axis=(2, 3))
        else:
            curv = np.abs(np.diff(speed, n=2, axis=-1)).mean(axis=(2, 3))
        out[f"C_curvature_{tag}"] = (self_baselined(curv), "high", WIDTH)

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, cands, v7 = {}, {}, {}
    for cohort in ("development_main", "external_8b"):
        frames[cohort] = cohort_frame(cohort)
        cands[cohort] = build_candidates(cohort)
        v7[cohort] = load_v7(cohort, len(frames[cohort]["risk"]))

    dev, ext = frames["development_main"], frames["external_8b"]
    anchor = score_capfree(v7["external_8b"], ext["risk"], ext["length"])
    print("v7 锚点 external 提前量>=%d: %d/%d TP, %d FP, 中位提前 %.0f"
          % (LEAD, anchor["tp"], int(ext["risk"].sum()), anchor["fp"],
             anchor["median_lead"]))
    assert anchor["tp"] == 347 and anchor["fp"] == 57, anchor

    rows = []
    for name in cands["development_main"]:
        dscore, direction, earliest = cands["development_main"][name]
        escore, _, _ = cands["external_8b"][name]
        for q in QUANTILES:
            # Threshold is an order statistic of the UNLABELED development
            # reference - outcomes are not read to set it.
            pool = dscore[np.isfinite(dscore)]
            if pool.size == 0:
                print(f"  skip {name}: development score is all NaN")
                break
            level = q if direction == "high" else 1.0 - q
            threshold = float(np.quantile(pool, level, method="lower"))

            d_first = first_crossing(dscore, threshold, direction, earliest)
            e_first = first_crossing(escore, threshold, direction, earliest)
            d_alone = score_capfree(d_first, dev["risk"], dev["length"])
            e_alone = score_capfree(e_first, ext["risk"], ext["length"])

            # OR with v7, keeping the earlier of the two alarms.
            def union(a, b):
                a2 = np.where(a < 0, 1 << 20, a)
                b2 = np.where(b < 0, 1 << 20, b)
                m = np.minimum(a2, b2)
                return np.where(m >= (1 << 20), -1, m)

            d_or = score_capfree(union(d_first, v7["development_main"]),
                                 dev["risk"], dev["length"])
            e_or = score_capfree(union(e_first, v7["external_8b"]),
                                 ext["risk"], ext["length"])
            rows.append({
                "head_name": name, "quantile": q, "threshold": threshold,
                "dev_alone_tp": d_alone["tp"], "dev_alone_fp": d_alone["fp"],
                "ext_alone_tp": e_alone["tp"], "ext_alone_fp": e_alone["fp"],
                "dev_or_tp": d_or["tp"], "dev_or_fp": d_or["fp"],
                "ext_or_tp": e_or["tp"], "ext_or_fp": e_or["fp"],
                "ext_gain_tp": e_or["tp"] - anchor["tp"],
                "ext_gain_fp": e_or["fp"] - anchor["fp"],
                "ext_or_median_lead": e_or["median_lead"],
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "v7_plus_heads.csv", index=False)

    print("\n===== 每个候选头,与 v7 取并集 (external 一次性评估) =====")
    print("%-26s %5s | %-16s | %-22s | %s"
          % ("头", "分位", "单独 (ext)", "v7 OR 它 (ext)", "净增"))
    for name in sorted(table["head_name"].unique()):
        sub = table[table.head_name == name]
        for _, r in sub.iterrows():
            print("%-26s %5.3f | %4d/%-4d TP %5d FP | %4d/%-4d TP %5d FP | %+4d TP / %+5d FP"
                  % (name, r["quantile"], r.ext_alone_tp, int(ext["risk"].sum()),
                     r.ext_alone_fp, r.ext_or_tp, int(ext["risk"].sum()),
                     r.ext_or_fp, r.ext_gain_tp, r.ext_gain_fp))

    # Pick, on development only, the single best quantile per head.
    best = []
    for name in sorted(table["head_name"].unique()):
        sub = table[table.head_name == name].copy()
        sub["dev_gain"] = sub.dev_or_tp - score_capfree(
            v7["development_main"], dev["risk"], dev["length"])["tp"]
        pick = sub.loc[sub.dev_gain.idxmax()]
        best.append(pick)
    chosen = pd.DataFrame(best)
    chosen.to_csv(OUT / "v7_plus_chosen.csv", index=False)
    print("\n===== 分位在 development 上选定后的 external 结果 =====")
    print("%-26s %6s | %-24s | %s" % ("头", "分位", "v7 OR 它 (ext)", "净增"))
    for _, r in chosen.iterrows():
        print("%-26s %6.3f | %4d/%-4d TP %5d FP  提前 %2.0f | %+4d TP / %+5d FP"
              % (r.head_name, r["quantile"], r.ext_or_tp, int(ext["risk"].sum()),
                 r.ext_or_fp, r.ext_or_median_lead, r.ext_gain_tp, r.ext_gain_fp))
    (OUT / "anchor.json").write_text(json.dumps(anchor, indent=2))


if __name__ == "__main__":
    main()
