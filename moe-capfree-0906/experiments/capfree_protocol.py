"""Cap-free evaluation: no detector and no metric may use the horizon cap.

Every result in this project so far defined its window as ``first_alarm <
0.65 x cap`` and divided lift by a survival prior estimated per suite and
chunk.  Both need the cap.  For an unseen task the cap is unknown - and the
risk label is *defined* relative to it - so any conclusion that depends on it
does not transfer.  In particular the finding "nothing beats waiting until 65%
of the horizon" is an artefact of being allowed to know the horizon.

Here nothing may use it:

  * detector input  = chunk index + routing only
  * baseline        = "alarm if still running at a fixed chunk q0", swept over
                      q0.  This is the cap-free form of the survival rule: it
                      needs a counter, not a horizon.
  * metric          = recall, false alarms, and **lead time in absolute
                      chunks** (length - alarm chunk), replacing in-window
                      recall.
  * reference       = excess true positives over the best fixed-chunk baseline
                      admissible at the same false-alarm count, replacing lift.

The cap is used in exactly one place - reproducing the published anchors, to
show the harness reads the same vectors the old protocol did - and that block
is clearly marked.
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

from recompute_task_matched_lift import (  # noqa: E402
    META_KEYS,
    SOURCES,
    cohort_frame,
)

# Lead-time budgets, in chunks.  B is how many chunks of warning remain after
# the alarm; B = 0 accepts an alarm on the final chunk, which is useless
# operationally, so the headline uses B >= 4.
LEADS = (0, 2, 4, 8, 12)
HEADLINE_LEAD = 4
Q0_GRID = tuple(range(2, 40))
SEED = 20260906
# For the anchor block only.
CAPS = {"libero_goal": 30, "libero_long": 52, "libero_object": 28,
        "libero_spatial": 22}
ANCHOR_WINDOW = {"libero_goal": 18, "libero_long": 33, "libero_object": 17,
                 "libero_spatial": 13}


def load_detectors(cohort: str, n: int) -> dict[str, np.ndarray]:
    """Task-agnostic first-alarm vectors.  `per_task` thresholds are excluded:
    they need the task label, which an unseen task does not have."""
    out: dict[str, np.ndarray] = {}
    for bundle, files in SOURCES.items():
        rel = files.get(cohort)
        if rel is None or not (ROOT / bundle / rel).exists():
            continue
        data = np.load(ROOT / bundle / rel, allow_pickle=True)
        for key in data.files:
            if key in META_KEYS or "|per_task" in key:
                continue
            vec = data[key]
            if vec.ndim != 1 or vec.shape[0] != n:
                continue
            out[f"{bundle.split('-')[1]}::{key}"] = vec.astype(int)
    return out


def score(first: np.ndarray, risk: np.ndarray, length: np.ndarray) -> dict:
    """Cap-free scoring.  `lead` is chunks of warning before the episode ends."""
    fired = first >= 0
    lead = np.where(fired, length - first, -1)
    row = {
        "alarms": int(fired.sum()),
        "tp": int((fired & risk).sum()),
        "fp": int((fired & ~risk).sum()),
    }
    row["recall"] = row["tp"] / int(risk.sum())
    row["precision"] = row["tp"] / row["alarms"] if row["alarms"] else np.nan
    hit = fired & risk
    row["median_lead"] = float(np.median(lead[hit])) if hit.any() else np.nan
    for b in LEADS:
        timely = fired & (lead >= b)
        row[f"tp_lead{b}"] = int((timely & risk).sum())
        row[f"fp_lead{b}"] = int((timely & ~risk).sum())
        row[f"recall_lead{b}"] = row[f"tp_lead{b}"] / int(risk.sum())
    return row


def fixed_chunk_baseline(risk: np.ndarray, length: np.ndarray) -> pd.DataFrame:
    """"Still running at chunk q0" - the cap-free survival rule."""
    rows = []
    for q0 in Q0_GRID:
        first = np.where(length > q0, q0, -1)
        rows.append({"q0": q0, **score(first, risk, length)})
    return pd.DataFrame(rows)


def baseline_frontier(base: pd.DataFrame, lead: int) -> callable:
    """Best baseline TP achievable at or below a given false-alarm count."""
    tp_col, fp_col = f"tp_lead{lead}", f"fp_lead{lead}"
    pts = base[[fp_col, tp_col]].sort_values(fp_col).to_numpy()
    fps, best = pts[:, 0], np.maximum.accumulate(pts[:, 1])

    def at(fp_budget: float) -> int:
        take = fps <= fp_budget
        return int(best[take].max()) if take.any() else 0

    return at


def rate_matched_null(first: np.ndarray, rng: np.random.Generator,
                      length: np.ndarray) -> np.ndarray:
    """Same number of alarms and same alarm-chunk distribution, random targets.

    Alarm chunks are resampled from the detector's own alarm-chunk histogram
    and only placed on episodes long enough to reach them, so the null inherits
    the detector's exposure to survival without inheriting its information."""
    fired = first >= 0
    chunks = first[fired]
    out = np.full(len(first), -1, dtype=int)
    if not fired.any():
        return out
    order = rng.permutation(len(first))
    drawn = rng.choice(chunks, size=len(first), replace=True)
    placed = 0
    for idx in order:
        if placed >= fired.sum():
            break
        if length[idx] > drawn[idx]:
            out[idx] = drawn[idx]
            placed += 1
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frames, dets, bases = {}, {}, {}

    for cohort in ("development_main", "external_8b"):
        coh = cohort_frame(cohort)
        frames[cohort] = coh
        dets[cohort] = load_detectors(cohort, len(coh["risk"]))
        bases[cohort] = fixed_chunk_baseline(coh["risk"], coh["length"])

    # ---- anchor block: the ONLY place the cap appears ----
    ext = frames["external_8b"]
    win = np.array([ANCHOR_WINDOW[s] for s in ext["suite"]])
    guard_key = next((k for k in dets["external_8b"] if "v7_guard" in k), None)
    anchors = {}
    if guard_key:
        g = dets["external_8b"][guard_key]
        anchors["v7_guard_all"] = [int(((g >= 0) & ext["risk"]).sum()),
                                   int(((g >= 0) & ~ext["risk"]).sum())]
        inw = (g >= 0) & (g <= win)
        anchors["v7_guard_in_window_capped"] = [
            int((inw & ext["risk"]).sum()), int((inw & ~ext["risk"]).sum())]
    (OUT / "anchors.json").write_text(json.dumps(anchors, indent=2))
    print("锚点（此处唯一使用 cap，用于确认读的是同一批向量）:", anchors)

    # ---- cap-free census ----
    rows = []
    for cohort, coh in frames.items():
        risk, length = coh["risk"], coh["length"]
        for name, first in dets[cohort].items():
            rows.append({"cohort": cohort, "detector": name, "arm": "routing",
                         **score(first, risk, length)})
            rows.append({"cohort": cohort, "detector": name, "arm": "null",
                         **score(rate_matched_null(first, rng, length), risk,
                                 length)})
        for _, b in bases[cohort].iterrows():
            rows.append({"cohort": cohort, "detector": f"still_running_q{int(b.q0)}",
                         "arm": "baseline", **{k: b[k] for k in b.index if k != "q0"}})
    census = pd.DataFrame(rows)
    census.to_csv(OUT / "capfree_census.csv", index=False)

    # ---- excess over the cap-free baseline, at matched false alarms ----
    lead = HEADLINE_LEAD
    tp_c, fp_c = f"tp_lead{lead}", f"fp_lead{lead}"
    census["baseline_tp_at_same_fp"] = [
        baseline_frontier(bases[r.cohort], lead)(r[fp_c])
        for _, r in census.iterrows()
    ]
    census["excess_tp"] = census[tp_c] - census.baseline_tp_at_same_fp
    census.to_csv(OUT / "capfree_census.csv", index=False)

    print(f"\n===== 无 cap 基线：跑到第 q0 步还在跑就报警（提前量 >= {lead}）=====")
    for cohort in frames:
        b = bases[cohort]
        n_risk = int(frames[cohort]["risk"].sum())
        n_safe = int((~frames[cohort]["risk"]).sum())
        print(f"-- {cohort} ({n_risk} risks, {n_safe} safe)")
        show = b[b.q0.isin([8, 12, 16, 20, 24, 28, 32])]
        for _, r in show.iterrows():
            print("   q0=%2d  TP %4d  FP %6d  召回 %.3f  精度 %.3f  中位提前 %2.0f"
                  % (r.q0, r[tp_c], r[fp_c], r[tp_c] / n_risk,
                     r[tp_c] / max(1, r[tp_c] + r[fp_c]), r.median_lead))

    print(f"\n===== 纯 MoE 检测器，按 超出基线的 TP 排序（提前量 >= {lead}）=====")
    for cohort in frames:
        sub = census[(census.cohort == cohort) & (census.arm == "routing")]
        n_risk = int(frames[cohort]["risk"].sum())
        print(f"-- {cohort}")
        print("   %-40s %6s %7s %8s %8s %7s %8s"
              % ("检测器", "TP", "FP", "召回", "精度", "提前量", "超出基线"))
        for _, r in sub.nlargest(8, "excess_tp").iterrows():
            print("   %-40s %6d %7d %8.3f %8.3f %7.0f %+8d"
                  % (r.detector.split("::")[1][:40], r[tp_c], r[fp_c],
                     r[tp_c] / n_risk,
                     r[tp_c] / max(1, r[tp_c] + r[fp_c]), r.median_lead,
                     int(r.excess_tp)))
        nul = census[(census.cohort == cohort) & (census.arm == "null")]
        print("   零对照（同发射率、同报警时刻分布）最好 超出基线 = %+d"
              % int(nul.excess_tp.max()))

    # ---- honest arm: pick on development, score external once ----
    dev, ex = frames["development_main"], frames["external_8b"]
    shared = sorted(set(dets["development_main"]) & set(dets["external_8b"]))
    dev_sub = (census[(census.cohort == "development_main") &
                      (census.arm == "routing") &
                      census.detector.isin(shared)]
               .drop_duplicates("detector").set_index("detector"))
    pick = dev_sub.excess_tp.idxmax()
    ext_row = census[(census.cohort == "external_8b") & (census.arm == "routing") &
                     (census.detector == pick)].iloc[0]
    ext_base = baseline_frontier(bases["external_8b"], lead)(ext_row[fp_c])
    summary = {
        "headline_lead": lead,
        "selected_on_development": pick,
        "dev_excess_tp": int(dev_sub.loc[pick, "excess_tp"]),
        "external": {
            "tp": int(ext_row[tp_c]), "fp": int(ext_row[fp_c]),
            "recall": float(ext_row[tp_c] / int(ex["risk"].sum())),
            "precision": float(ext_row[tp_c] / max(1, ext_row[tp_c] + ext_row[fp_c])),
            "median_lead": float(ext_row.median_lead),
            "baseline_tp_at_same_fp": int(ext_base),
            "excess_tp": int(ext_row[tp_c] - ext_base),
        },
    }
    (OUT / "capfree_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== 诚实臂：在 development 上选，external 只评一次 =====")
    print("   选中: %s" % pick)
    e = summary["external"]
    print("   external: TP %d / FP %d  召回 %.3f  精度 %.3f  中位提前 %.0f chunk"
          % (e["tp"], e["fp"], e["recall"], e["precision"], e["median_lead"]))
    print("   同 FP 下无 cap 基线只能拿到 %d 个 TP  →  **超出 %+d**"
          % (e["baseline_tp_at_same_fp"], e["excess_tp"]))

    print(f"\n===== 提前量的代价（external，{pick.split('::')[-1][:40]}）=====")
    row = census[(census.cohort == "external_8b") & (census.arm == "routing") &
                 (census.detector == pick)].iloc[0]
    n_risk = int(ex["risk"].sum())
    print("   %8s %7s %7s %9s %10s" % ("提前量>=", "TP", "FP", "召回", "超出基线"))
    for b in LEADS:
        base_at = baseline_frontier(bases["external_8b"], b)(row[f"fp_lead{b}"])
        print("   %8d %7d %7d %9.3f %+10d"
              % (b, row[f"tp_lead{b}"], row[f"fp_lead{b}"],
                 row[f"tp_lead{b}"] / n_risk, int(row[f"tp_lead{b}"] - base_at)))


if __name__ == "__main__":
    main()
