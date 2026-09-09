"""Shared, read-only harness for the long-lead search.

Everything here either reads published artefacts or re-implements the frozen
v8.3 scoring rules verbatim.  Nothing in this bundle writes outside
``moe-longlead-search-0906/``.

Protocol constants that are *not* negotiable and are asserted in tests:
  * detector input is (chunk index, routing) only -- no horizon, no phase,
    no task identity, no per-task thresholds;
  * every comparison is ``>=`` / ``<=``, never strict;
  * thresholds are order statistics of the *unlabeled* development pool;
  * lead is measured in absolute chunks: ``lead = length - first_alarm``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
RESULTS = BUNDLE / "results"

for extra in ("moe-prior-correction-0906/experiments",
              "moe-v7-0905/experiments", "moe-v7-0905/method"):
    sys.path.insert(0, str(ROOT / extra))

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
LEADS = (0, 4, 8, 12, 16, 20)
# Suite caps: risk episodes run to exactly the cap, so lead >= L needs an
# alarm no later than chunk cap - L.
CAP = {"libero_goal": 30, "libero_long": 52, "libero_object": 28,
       "libero_spatial": 22}

# --- frozen v8.3 configuration (moe-v8-0906/results/v83_summary.json) --------
V83 = {"baseline": 2, "width": 6, "confirm": 2, "slope": -0.0015}
V83_DIRECTION = {"frontback_flowpath": "low", "curvature_3step": "high"}
QUANTILE = 0.995
FRONT, BACK = slice(0, 4), slice(4, 8)
CURVATURE_IDX = (0, 4, 8)

ANCHOR_LEAD4 = {"development_main": (344, 487, 47),
                "external_8b": (398, 564, 70),
                "legacy_main16x32": (225, 307, 17)}
ANCHOR_PROFILE = {0: (1167, 209), 4: (967, 134), 8: (711, 104),
                  12: (617, 66), 16: (533, 30), 20: (352, 14)}


# ---------------------------------------------------------------- primitives
def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(values[:, max(0, q - width + 1):q + 1],
                                   axis=1)
    return out


def order_threshold(pool: np.ndarray, quantile: float,
                    direction: str) -> float:
    """Order statistic of an unlabeled pool.  `method='lower'` keeps the value
    on the data grid so `>=` cannot straddle a synthetic interpolant."""
    good = pool[np.isfinite(pool)]
    level = quantile if direction == "high" else 1.0 - quantile
    return float(np.quantile(good, level, method="lower"))


def confirmed_first(score_series: np.ndarray, threshold, direction: str,
                    confirm: int, earliest: int) -> np.ndarray:
    """First chunk where a `>=`/`<=` crossing has held `confirm` chunks."""
    thr = threshold
    if np.isscalar(thr):
        thr = np.full(score_series.shape[1], float(thr))
    thr = np.asarray(thr, dtype=np.float64)[None, :]
    hit = (score_series >= thr) if direction == "high" else (score_series <= thr)
    hit &= np.isfinite(score_series)
    hit[:, :earliest] = False
    if confirm > 1:
        held = np.zeros_like(hit)
        run = np.zeros(hit.shape[0], dtype=int)
        for q in range(hit.shape[1]):
            run = np.where(hit[:, q], run + 1, 0)
            held[:, q] = run >= confirm
        hit = held
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def union(*firsts) -> np.ndarray:
    stack = np.stack([np.where(np.asarray(f) < 0, 1 << 20, f) for f in firsts])
    m = stack.min(axis=0)
    return np.where(m >= (1 << 20), -1, m)


def score(first, risk, length, lead=4) -> dict:
    first = np.asarray(first)
    fired = first >= 0
    lag = np.where(fired, length - first, -1)
    timely = fired & (lag >= lead)
    tp, fp = int((timely & risk).sum()), int((timely & ~risk).sum())
    return {"tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "median_lead": float(np.median(lag[timely & risk]))
            if (timely & risk).any() else float("nan")}


def profile(alarms: dict, data: dict, leads=LEADS) -> dict:
    out = {}
    for lead in leads:
        tp = fp = 0
        for c, d in data.items():
            s = score(alarms[c], d["risk"], d["length"], lead)
            tp += s["tp"]
            fp += s["fp"]
        out[lead] = (tp, fp)
    return out


# ------------------------------------------------------------------- cohorts
_CACHE: dict = {}


def load_cohort(name: str) -> dict:
    if name in _CACHE:
        return _CACHE[name]
    from recompute_task_matched_lift import META_KEYS, SOURCES, cohort_frame

    if name == "legacy_main16x32":
        table = pd.read_csv(
            ROOT / "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv")
        risk = table.failure.to_numpy(bool)
        length = table.length.to_numpy(int)
        task = table.task.astype(str).to_numpy()
        suite = np.array([t.split("/")[0] for t in task])
        task = np.array([t.split("/", 1)[1] if "/" in t else t for t in task])
        v7 = np.load(ROOT / "moe-v7-legacy16x32-0906/results/legacy16x32"
                     "/legacy16x32_first_alarms.npz")["legacy_guard"].astype(int)
    else:
        coh = cohort_frame(name)
        risk, length, suite, task = (coh["risk"], coh["length"],
                                     coh["suite"], coh["task"])
        task = np.array([t.split("/", 1)[1] if "/" in str(t) else str(t)
                         for t in task])
        v7 = None
        for bundle, files in SOURCES.items():
            rel = files.get(name)
            if rel is None or not (ROOT / bundle / rel).exists():
                continue
            data = np.load(ROOT / bundle / rel, allow_pickle=True)
            for key in data.files:
                if key not in META_KEYS and "v7_guard" in key:
                    v7 = data[key].astype(int)
        assert v7 is not None, name

    speed = np.load(ROOT / f"moe-v8-0906/results/{name}_flow_speed.npz"
                    )["flow_speed"]
    v83 = np.load(ROOT / "moe-v8-0906/results/v83_alarms.npz"
                  )[f"{name}|v8.3"].astype(int)
    assert len(risk) == speed.shape[0] == len(v7) == len(v83), name
    out = {"name": name, "risk": risk, "length": length, "suite": suite,
           "task": task, "v7": v7, "v83": v83, "speed": speed,
           "n_chunk": speed.shape[1]}
    _CACHE[name] = out
    return out


def load_all(with_speed: bool = True) -> dict:
    data = {c: load_cohort(c) for c in COHORTS}
    if not with_speed:
        for d in data.values():
            d.pop("speed", None)
    return data


def mobility(name: str) -> np.ndarray:
    """[n, 52, 8, 10] per-denoising-step routing mobility, already extracted
    and verified to 2.0e-6 against the published cache.  Read, never rebuild."""
    return np.load(ROOT / f"moe-v9-0906/results/{name}_step_mobility.npz"
                   )["mobility"]


def flow_speed(name: str) -> np.ndarray:
    return np.load(ROOT / f"moe-v8-0906/results/{name}_flow_speed.npz"
                   )["flow_speed"]


# ------------------------------------------------------------ v8.3 rebuilt
def build_v83_heads(speed, baseline=V83["baseline"], width=V83["width"]):
    with np.errstate(invalid="ignore"):
        path = np.nansum(speed, axis=3)
        front = np.nanmean(path[:, :, FRONT], axis=2)
        back = np.nanmean(path[:, :, BACK], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan

    sub = speed[:, :, BACK][:, :, :, list(CURVATURE_IDX)]
    curvature = np.abs(np.diff(sub, n=2, axis=-1)).mean(axis=(2, 3))
    base = np.nanmean(curvature[:, 1:1 + baseline], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(np.maximum(curvature, 1e-12) / np.maximum(base, 1e-12))
    rel[~np.isfinite(rel)] = np.nan
    return {"frontback_flowpath": trailing_mean(ratio, width),
            "curvature_3step": trailing_mean(rel, width)}


def rebuild_v83(data: dict) -> dict:
    """Reproduce the frozen v8.3 alarm vectors from flow_speed."""
    heads = {c: build_v83_heads(d["speed"]) for c, d in data.items()}
    thr = {h: order_threshold(s, QUANTILE, V83_DIRECTION[h])
           for h, s in heads["development_main"].items()}
    out = {}
    for c, d in data.items():
        firsts = []
        for h, series in heads[c].items():
            q = np.arange(series.shape[1])
            direction = V83_DIRECTION[h]
            moving = thr[h] + V83["slope"] * q * (1 if direction == "high" else -1)
            firsts.append(confirmed_first(series, moving, direction,
                                          V83["confirm"], V83["width"]))
        out[c] = union(d["v7"], *firsts)
    return out


# --------------------------------------------------------------- reporting
def per_suite(alarms: dict, data: dict, lead: int) -> pd.DataFrame:
    rows = []
    for suite in SUITES:
        tp = fp = n = 0
        for c, d in data.items():
            m = d["suite"] == suite
            if not m.any():
                continue
            f = np.asarray(alarms[c])
            timely = (f >= 0) & ((d["length"] - f) >= lead)
            n += int((m & d["risk"]).sum())
            tp += int((timely & m & d["risk"]).sum())
            fp += int((timely & m & ~d["risk"]).sum())
        rows.append({"suite": suite, "lead": lead, "n_risk": n,
                     "tp": tp, "fp": fp})
    return pd.DataFrame(rows)


def per_task(alarms: dict, data: dict, lead: int) -> pd.DataFrame:
    rows = {}
    for c, d in data.items():
        f = np.asarray(alarms[c])
        timely = (f >= 0) & ((d["length"] - f) >= lead)
        for suite, task in zip(np.unique(d["suite"]), [None]):
            pass
        keys = np.array([f"{s}/{t}" for s, t in zip(d["suite"], d["task"])])
        for k in np.unique(keys):
            m = keys == k
            r = rows.setdefault(k, {"task": k, "lead": lead,
                                    "n_risk": 0, "tp": 0, "fp": 0})
            r["n_risk"] += int((m & d["risk"]).sum())
            r["tp"] += int((timely & m & d["risk"]).sum())
            r["fp"] += int((timely & m & ~d["risk"]).sum())
    return pd.DataFrame(sorted(rows.values(), key=lambda r: -r["n_risk"]))


def fmt_profile(prof: dict) -> str:
    return "  ".join("L%-2d %4d/%-4d" % (k, v[0], v[1])
                     for k, v in sorted(prof.items()))
