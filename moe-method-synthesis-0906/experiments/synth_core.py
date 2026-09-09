"""Shared loading / alignment for the method-synthesis accounting.

Nothing here computes a new detector or re-derives a feature.  Every alarm
vector is read exactly as it was saved by the bundle that published it; this
module only aligns them to a common episode ordering, attaches the canonical
risk label, the per-suite in-window deadline, and the physical failure mode,
and exposes the per-task survival prior that the prior-correction bundle
established as the only honest basis.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
RESULTS = BUNDLE / "results"

sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))

from recompute_task_matched_lift import (  # noqa: E402
    META_KEYS,
    SOURCES,
    cohort_frame,
    prior_of,
    survival_prior,
)

COHORTS = ("external_8b", "development_main")

# Sealed v7 streams live in their own npz with cohort-prefixed keys, so they
# are not in SOURCES.  They are the honest arm's own detectors and belong in
# the census.
V7_PATH = ROOT / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
V7_PREFIX = {"external_8b": "external_", "development_main": "main_"}

# Original horizon caps.  Risk == did not finish before the cap.
CAP = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}
WINDOW_FRACTION = 0.65

RUN_ID = {"external_8b": "right-50x8b-20260903", "development_main": "right-50x8-20260903"}
PHYS_CSV = ROOT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"


@dataclass
class Cohort:
    name: str
    risk: np.ndarray            # bool  (n,)
    length: np.ndarray          # int   (n,)
    suite: np.ndarray           # str   (n,)
    task: np.ndarray            # str   (n,)
    episode: np.ndarray         # int   (n,)
    init_state: np.ndarray      # int   (n,)
    deadline: np.ndarray        # float (n,)  0.65 * cap(suite)
    mode: np.ndarray            # str   (n,)  physical failure mode or ''
    alarms: np.ndarray          # int16 (n_det, n)  first-alarm chunk, -1 = never
    detectors: list[str]        # (n_det,) "bundle::detector"
    priors_task: dict
    priors_suite: dict

    @property
    def n(self) -> int:
        return len(self.risk)

    def fired(self) -> np.ndarray:
        return self.alarms >= 0

    def in_window(self) -> np.ndarray:
        """Alarm exists and lands strictly before 0.65 x cap."""
        return (self.alarms >= 0) & (self.alarms < self.deadline[None, :])


def _episode_ids(cohort_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Row-aligned episode index and init state id, straight from the layer cache."""
    sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
    from evaluate_intrinsic_guard_v7 import EXTERNAL_LAYER, MAIN_LAYER, load_npz

    layer = MAIN_LAYER if cohort_name == "development_main" else EXTERNAL_LAYER
    cache = load_npz(layer)
    return cache["episode"].astype(int), cache["init_state_id"].astype(int)


def _physical_modes(cohort_name: str, task: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """Join the replay-derived physical failure mode onto the cohort rows.

    The labels come from a separate 36,098-trajectory replay keyed by
    (run_id, suite, task_name, episode_index); the cohort task string is
    "suite/task_name".  Rows with no failure carry ''.
    """
    phys = pd.read_csv(PHYS_CSV)
    phys = phys[phys.run_id == RUN_ID[cohort_name]].copy()
    phys["task"] = phys.suite + "/" + phys.task_name
    key = phys.set_index(["task", "episode_index"])["primary_failure_reason"]
    out = np.array(
        [str(key.get((t, int(e)), "")) for t, e in zip(task, episode)], dtype=object
    )
    out[out == "nan"] = ""
    return out.astype(str)


def load_cohort(name: str) -> Cohort:
    coh = cohort_frame(name)
    suite = coh["suite"]
    deadline = np.array([WINDOW_FRACTION * CAP[s] for s in suite], dtype=float)
    episode, init_state = _episode_ids(name)
    mode = _physical_modes(name, coh["task"], episode)

    vectors: list[np.ndarray] = []
    names: list[str] = []
    n = len(coh["risk"])
    for bundle, files in SOURCES.items():
        rel = files.get(name)
        if rel is None:
            continue
        path = ROOT / bundle / rel
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        for key in sorted(data.files):
            if key in META_KEYS:
                continue
            vec = data[key]
            if vec.ndim != 1 or vec.shape[0] != n:
                continue
            vectors.append(vec.astype(np.int32))
            names.append(f"{bundle}::{key}")

    v7 = np.load(V7_PATH, allow_pickle=False)
    pre = V7_PREFIX[name]
    for key in sorted(v7.files):
        if not key.startswith(pre):
            continue
        vec = v7[key]
        if vec.ndim != 1 or vec.shape[0] != n:
            continue
        vectors.append(vec.astype(np.int32))
        names.append(f"moe-v7-0905::{key[len(pre):]}")

    alarms = np.stack(vectors).astype(np.int32)
    return Cohort(
        name=name,
        risk=coh["risk"],
        length=coh["length"],
        suite=suite,
        task=coh["task"],
        episode=episode,
        init_state=init_state,
        deadline=deadline,
        mode=mode,
        alarms=alarms,
        detectors=names,
        priors_task=survival_prior(coh["task"], coh["length"], coh["risk"]),
        priors_suite=survival_prior(suite, coh["length"], coh["risk"]),
    )


def shared_detectors(a: Cohort, b: Cohort) -> list[str]:
    return sorted(set(a.detectors) & set(b.detectors))


def restrict(c: Cohort, keep: list[str]) -> Cohort:
    idx = [c.detectors.index(k) for k in keep]
    c.alarms = c.alarms[idx]
    c.detectors = list(keep)
    return c


def negative_control_length(c: Cohort) -> np.ndarray:
    """Length is a negative control, never a baseline.

    Risk is *defined* as not finishing before the cap, so alarming on
    "length has reached the suite cap" recalls every risk by construction and
    can never appear in a ranking.  Returned as a first-alarm vector for
    bookkeeping only.
    """
    cap = np.array([CAP[s] for s in c.suite])
    return np.where(c.length >= cap, 0, -1).astype(np.int32)


def rate_matched_null(
    fire: np.ndarray, rng: np.random.Generator, strata: np.ndarray | None = None
) -> np.ndarray:
    """Random alarm vectors matched to each detector's firing rate.

    ``fire`` is (n_det, n) boolean.  If ``strata`` is given the rate is matched
    within each stratum (used for the per-suite arms, so the null cannot get
    complementarity for free by moving alarms between suites).
    """
    out = np.zeros_like(fire)
    groups = [np.arange(fire.shape[1])] if strata is None else [
        np.flatnonzero(strata == s) for s in np.unique(strata)
    ]
    for g in groups:
        for d in range(fire.shape[0]):
            k = int(fire[d, g].sum())
            if k:
                out[d, rng.choice(g, size=k, replace=False)] = True
    return out
