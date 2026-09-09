"""Stage 1 - channel screening, DEVELOPMENT ONLY.

For every candidate channel:
  * within-episode information (a channel that is bit-constant inside an
    episode carries no temporal signal for CUSUM to integrate),
  * the single-chunk-threshold detector under the cap-free protocol - this is
    the floor CUSUM has to improve on, and it doubles as the channel ranking.

External is never opened here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402
import capfree_common as C  # noqa: E402

COHORT = "development_main"
N_THR = 61


def first_alarm_from_cummax(cm: np.ndarray, thr: float) -> np.ndarray:
    """First q with z_q >= thr, using the running max.  `>=`, never `>`."""
    idx = (cm < thr).sum(axis=1)
    return np.where(idx < C.N_CHUNKS, idx, -1)


def within_episode_info(x: np.ndarray, valid: np.ndarray) -> dict:
    xf = np.where(valid, x, np.nan)
    with np.errstate(invalid="ignore", all="ignore"):
        hi = np.nanmax(xf, axis=1)
        lo = np.nanmin(xf, axis=1)
    const = ~np.isfinite(hi - lo) | (hi == lo)
    d = np.diff(xf, axis=1)
    both = valid[:, 1:] & valid[:, :-1]
    changed = np.where(both, np.abs(d) > 0, False).sum(axis=1)
    denom = np.maximum(1, both.sum(axis=1))
    return {
        "const_frac": float(const.mean()),
        "switch_rate": float((changed / denom).mean()),
        "n_distinct": int(min(1_000_000, len(np.unique(xf[np.isfinite(xf)])))),
    }


def main() -> None:
    out = C.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    f = C.frame(COHORT)
    risk, length, valid = f["risk"], f["length"], f["valid"]
    base = C.fixed_chunk_baseline(risk, length)
    base_at = C.baseline_frontier(base, C.HEADLINE_LEAD)

    names, X = bank.build(COHORT)
    specs = names
    print(f"候选通道 {len(specs)} 个，仅在 {COHORT} 上筛选")
    rows = []
    for i, (name, x) in enumerate(zip(names, X)):
        info = within_episode_info(x, valid)
        mu, sd = C.chunk_stats(x, valid)
        z = C.normalise(x, valid, mu, sd, "raw")
        rec = {"channel": name, **info}
        if info["const_frac"] >= 0.999 or info["n_distinct"] < 2:
            rec.update({"best_excess": -10_000, "sign": 0, "dead": True})
            rows.append(rec)
            continue
        rec["dead"] = False
        pool = z[valid]
        grid = np.unique(np.quantile(pool, np.linspace(0.02, 0.999, N_THR)))
        best = None
        for sign in (+1, -1):
            zz = sign * z
            cm = np.where(valid, zz, -np.inf)
            cm = np.maximum.accumulate(cm, axis=1)
            g = np.unique(np.quantile(sign * pool, np.linspace(0.02, 0.999, N_THR)))
            for thr in g:
                first = first_alarm_from_cummax(cm, thr)
                if (first >= 0).sum() == 0:
                    continue
                s = C.score(first, risk, length)
                fp = s[f"fp_lead{C.HEADLINE_LEAD}"]
                if not (C.FP_LO <= fp <= C.FP_HI):
                    continue
                ex = C.excess_over_baseline(s, base_at)
                if best is None or ex > best[0]:
                    best = (ex, sign, float(thr), s)
        if best is None:
            rec.update({"best_excess": -10_000, "sign": 0})
        else:
            ex, sign, thr, s = best
            rec.update({
                "best_excess": int(ex), "sign": int(sign), "thr": thr,
                "tp_lead4": s["tp_lead4"], "fp_lead4": s["fp_lead4"],
                "tp_lead0": s["tp_lead0"], "fp_lead0": s["fp_lead0"],
                "median_lead": s["median_lead"],
            })
        rows.append(rec)
        del grid
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{len(specs)}")

    df = pd.DataFrame(rows).sort_values("best_excess", ascending=False)
    df.to_csv(out / "channel_screen_development.csv", index=False)
    live = df[~df.dead]
    print("\n=== 单点阈值（floor）在 development 上的前 25 通道（提前量>=4，FP∈[80,320]）===")
    print(live.head(25)[["channel", "sign", "tp_lead4", "fp_lead4",
                         "best_excess", "median_lead", "const_frac",
                         "switch_rate"]].to_string(index=False))
    dead = df[df.dead]
    print(f"\n剔除（episode 内逐位常数）: {len(dead)} 个")
    if len(dead):
        print("  例:", list(dead.channel.head(10)))
    hi_const = live[live.const_frac > 0.5]
    print(f"\n高 episode 内常数率（>50%，不得作为 headline）: {len(hi_const)} 个")
    if len(hi_const):
        print(hi_const[["channel", "const_frac", "n_distinct",
                        "best_excess"]].head(12).to_string(index=False))
    (out / "screen_meta.json").write_text(json.dumps({
        "cohort": COHORT, "n_specs": len(specs), "n_thresholds": N_THR,
        "fp_window": [C.FP_LO, C.FP_HI], "headline_lead": C.HEADLINE_LEAD,
        "baseline_best_fp80_lead4": int(
            base[base.fp_lead4 <= 80].tp_lead4.max()),
    }, indent=2))


if __name__ == "__main__":
    main()
