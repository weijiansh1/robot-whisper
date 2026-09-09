"""Shared loading and scoring for the boundary-head study.

Everything that scores an alarm vector is imported from v8 rather than
reimplemented, so any difference in the numbers is a difference in the head and
not in the metric:

    from evaluate_full_corpus import score, union, confirmed_first, load_cohort

`load_cohort` gives risk / length / suite / the v7 alarm vector; this module
adds the boundary cells, the per-episode task name, and the frozen v8 alarm
vector read from `moe-v8-0906/results/v8_full_corpus_alarms.npz`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V8 = ROOT / "moe-v8-0906"
V4 = ROOT / "moe-v4-0904/results/layerwise_mobility"

sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))
sys.path.insert(0, str(V8 / "experiments"))

from evaluate_full_corpus import (  # noqa: E402,F401
    confirmed_first,
    load_cohort,
    score,
    union,
)

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
BASE_CELLS = ("front_state", "front_action", "back_state", "back_action")
# `wc_` = adjacent-query mobility at step 9 (v7's freeze axis, both tokens);
# `chord_` = within-query denoising displacement (v8's flow axis).
CELLS = BASE_CELLS
WC_CELLS = tuple(f"wc_{c}" for c in BASE_CELLS)
CHORD_CELLS = tuple(f"chord_{c}" for c in BASE_CELLS)
ALL_CELLS = CELLS + WC_CELLS + CHORD_CELLS
LEADS = (0, 2, 4, 8, 12)
HEADLINE_LEAD = 4
SUITE_OF = {"libero_goal": "goal", "libero_long": "long",
            "libero_object": "object", "libero_spatial": "spatial"}

# The frozen v8 anchors, cap-free, lead >= 4.  Every run asserts these.
V8_ANCHOR = {"development_main": (331, 44), "external_8b": (377, 66),
             "legacy_main16x32": (224, 16)}
V7_ANCHOR = {"development_main": (303, 38), "external_8b": (347, 57),
             "legacy_main16x32": (220, 16)}


def _task_names(cohort: str) -> np.ndarray:
    cache = {"development_main": V4 / "main_reference.npz",
             "external_8b": V4 / "external_8b.npz",
             "legacy_main16x32":
                 ROOT / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
             }[cohort]
    with np.load(cache, allow_pickle=False) as a:
        return a["task_names"].astype(str)[a["task_index"].astype(int)]


def load(cohort: str) -> dict:
    """risk / length / suite / task / v7 / v8 / the four boundary cells."""
    d = load_cohort(cohort)
    d.pop("speed", None)
    d["task"] = _task_names(cohort)
    with np.load(OUT / f"{cohort}_boundary.npz", allow_pickle=False) as a:
        for cell in ALL_CELLS:
            d[cell] = np.asarray(a[cell], dtype=np.float64)
    alarms = np.load(V8 / "results/v8_full_corpus_alarms.npz")
    d["v8"] = alarms[f"{cohort}|v8"].astype(int)
    n = len(d["risk"])
    for key in ("v7", "v8", "length", "suite", "task"):
        assert len(d[key]) == n, (cohort, key)
    for cell in ALL_CELLS:
        assert d[cell].shape[0] == n, (cohort, cell)
    return d


def load_all() -> dict[str, dict]:
    data = {c: load(c) for c in COHORTS}
    for c, d in data.items():
        for arm, anchor in (("v7", V7_ANCHOR), ("v8", V8_ANCHOR)):
            s = score(d[arm], d["risk"], d["length"], HEADLINE_LEAD)
            assert (s["tp"], s["fp"]) == anchor[c], (c, arm, s, anchor[c])
    return data


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """v7/v8's smoothing idiom, NaN-tolerant, strictly causal."""
    if width <= 1:
        return values.astype(np.float64, copy=True)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(values[:, max(0, q - width + 1):q + 1], axis=1)
    return out


def chunk_rank(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Per-chunk rank of each episode's value inside that chunk's marginal.

    Returns a value in [0, 1].  Ranks are always taken against the *reference*
    cohort's per-chunk distribution (development, unlabeled) so that the
    transform is frozen along with the threshold and carries no test-cohort
    information.  Comparing two cells needs a common scale: their raw
    magnitudes differ by 3.9x, so a raw difference is just the state cell.
    """
    ref = values if reference is None else reference
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        base = ref[:, q] if q < ref.shape[1] else np.array([np.nan])
        base = base[np.isfinite(base)]
        col = values[:, q]
        ok = np.isfinite(col)
        if base.size == 0 or not ok.any():
            continue
        base = np.sort(base)
        # mid-rank on the reference ECDF; ties get the same value, as required
        lo = np.searchsorted(base, col[ok], side="left")
        hi = np.searchsorted(base, col[ok], side="right")
        out[ok, q] = 0.5 * (lo + hi) / base.size
    return out


def score_all_leads(first, risk, length, leads=LEADS) -> dict:
    """TP/FP at every lead in one pass.

    Selecting at a *long* lead is banned because failures are defined as running
    to the cap, so risk episodes are systematically the long ones.  On this
    corpus the minimum risk length is 22 chunks against a safe median of 13, and
    length alone has AUC 0.94-0.96.  A head that only fires late therefore has
    its false alarms deleted by the lead filter rather than by being right:
    `sweep_head.py`'s first pass found a configuration at 39 TP / 0 FP at
    lead >= 4 which is 82 TP / 73 FP at lead >= 0.  Selection happens at
    lead >= 0; everything else is reported.
    """
    fired = first >= 0
    lag = np.where(fired, length - first, -(1 << 20))
    out = {}
    for lead in leads:
        timely = fired & (lag >= lead)
        out[f"tp{lead}"] = int(np.count_nonzero(timely & risk))
        out[f"fp{lead}"] = int(np.count_nonzero(timely & ~risk))
    return out


def per_suite(first, d, lead=HEADLINE_LEAD):
    timely = (first >= 0) & ((d["length"] - first) >= lead)
    rows = {}
    for suite, short in SUITE_OF.items():
        m = d["suite"] == suite
        if not m.any():
            continue
        rows[short] = (int((timely & m & d["risk"]).sum()),
                       int((m & d["risk"]).sum()),
                       int((timely & m & ~d["risk"]).sum()))
    return rows


def per_task_table(firsts: dict, data: dict, lead=HEADLINE_LEAD) -> pd.DataFrame:
    """Pooled over cohorts, one row per task, recall and FP for each arm."""
    rows = {}
    for cohort, d in data.items():
        for task in np.unique(d["task"]):
            m = d["task"] == task
            key = task
            r = rows.setdefault(key, {"task": task, "n_risk": 0, "n_safe": 0})
            r["n_risk"] += int((m & d["risk"]).sum())
            r["n_safe"] += int((m & ~d["risk"]).sum())
            for arm, first in firsts[cohort].items():
                timely = (first >= 0) & ((d["length"] - first) >= lead)
                r[f"{arm}_tp"] = r.get(f"{arm}_tp", 0) + int((timely & m & d["risk"]).sum())
                r[f"{arm}_fp"] = r.get(f"{arm}_fp", 0) + int((timely & m & ~d["risk"]).sum())
    table = pd.DataFrame(list(rows.values()))
    return table.sort_values("task").reset_index(drop=True)


def total(firsts: dict, data: dict, lead=HEADLINE_LEAD) -> tuple[int, int, int]:
    tp = fp = risk = 0
    for cohort, d in data.items():
        s = score(firsts[cohort], d["risk"], d["length"], lead)
        tp += s["tp"]
        fp += s["fp"]
        risk += int(d["risk"].sum())
    return tp, fp, risk
