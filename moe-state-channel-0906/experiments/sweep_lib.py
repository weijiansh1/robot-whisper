#!/usr/bin/env python3
"""Cohort loading and the declared per-query quantities of this bundle.

The detector protocol itself is *not* re-implemented here.  It is imported
verbatim from `moe-flow-semantics-0906/experiments/protocol.py` (which is
itself a verbatim copy of the frozen v4 / front-back protocol) together with
the sweep and external-replay drivers from `detect_step_alarm.py`, with
`sys.dont_write_bytecode = True` so that importing writes nothing into those
directories.

Every quantity below is a function of `hb_router_probs` at query `q` only.
All ten denoising steps of query `q` run before the action of query `q` is
emitted, so reading the step axis is causal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
FLOW = PROJECT / "moe-flow-semantics-0906"
STEP_PROFILES = FLOW / "results/step_profiles"
ARC = BUNDLE / "results/arc"

sys.path.insert(0, str(FLOW / "experiments"))
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402,F401  (frozen protocol, imported verbatim)
import detect_step_alarm as D  # noqa: E402  (frozen sweep / external replay drivers)
from arc_lib import ARC_NAMES, ols_slope  # noqa: E402

COHORTS = ("development_main", "development_extra", "external_8b")
EPSILON = 1e-6

# ---------------------------------------------------------------- declared set
# id -> (family, human description).  See results/PREREG.md.
PRIMARY_A = ("td_slope", "td_late_early", "pc1corr_slope", "pc1share_slope",
             "stretch_slope", "centred_slope")
SECONDARY_A = ("pc1_index_corr_s9", "arc_stretch_s9", "fiedler_index_rho_s9",
               "along_state_energy_s9", "along_state_fraction_s9", "pc1_state_cos_s9")
CONTROL = ("saa_slope",)
STATE_B = ("state_mobility_s9", "state_mobility_mean",
           "state_minus_action_mobility", "state_over_action_logratio")
QUANTITIES = PRIMARY_A + SECONDARY_A + CONTROL + STATE_B
FAMILY = (
    {n: "A_primary_formation_slope" for n in PRIMARY_A}
    | {n: "A_secondary_endpoint" for n in SECONDARY_A}
    | {n: "A_negative_control" for n in CONTROL}
    | {n: "B_state_channel" for n in STATE_B}
)
# Reference heads re-derived with the same driver, as anchors.
ANCHORS = ("mobility_s9", "load_entropy_s9")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class Cohort:
    """Row-aligned view of one cohort's step profiles and arc geometry."""

    def __init__(self, name: str, need_arc: bool = True) -> None:
        index = load_npz(STEP_PROFILES / f"{name}_index.npz")
        self.name = name
        self.task_names = index["task_names"].astype(str)
        self.task_index = index["task_index"].astype(int)
        self.task = self.task_names[self.task_index]
        self.init_state = index["init_state_id"].astype(int)
        self.length = index["length"].astype(int)
        self.valid = index["valid"].astype(bool)
        self.episode = index["episode"].astype(int)
        self.metric_names = index["metric_names"].astype(str).tolist()
        self._metrics = np.load(STEP_PROFILES / f"{name}_metrics.npy", mmap_mode="r")
        self._mobility = np.load(STEP_PROFILES / f"{name}_mobility.npy", mmap_mode="r")
        self._state_mobility = np.load(
            STEP_PROFILES / f"{name}_state_mobility.npy", mmap_mode="r"
        )
        self._arc_path = ARC / f"{name}_arc.npy"
        self._arc = None
        if need_arc:
            arc_index = load_npz(ARC / f"{name}_arc_index.npz")
            if not np.array_equal(arc_index["episode"].astype(int), self.episode):
                raise ValueError(f"{name}: arc rows are not aligned with the step profile")
            if list(arc_index["arc_names"].astype(str)) != list(ARC_NAMES):
                raise ValueError(f"{name}: arc feature list changed")
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- step planes
    def metric_plane(self, metric: str) -> np.ndarray:
        key = f"metric::{metric}"
        if key not in self._cache:
            position = self.metric_names.index(metric)
            self._cache[key] = np.ascontiguousarray(
                self._metrics[:, :, :, :, position], dtype=np.float32
            )
        return self._cache[key]

    def arc_plane(self, feature: str) -> np.ndarray:
        key = f"arc::{feature}"
        if key not in self._cache:
            if self._arc is None:
                self._arc = np.load(self._arc_path, mmap_mode="r")
            position = list(ARC_NAMES).index(feature)
            self._cache[key] = np.ascontiguousarray(
                self._arc[:, :, :, :, position], dtype=np.float32
            )
        return self._cache[key]

    def mobility_plane(self) -> np.ndarray:
        if "mobility" not in self._cache:
            self._cache["mobility"] = np.asarray(self._mobility, dtype=np.float32)
        return self._cache["mobility"]

    def state_mobility_plane(self) -> np.ndarray:
        if "state_mobility" not in self._cache:
            self._cache["state_mobility"] = np.asarray(
                self._state_mobility, dtype=np.float32
            )
        return self._cache["state_mobility"]

    # --------------------------------------------------------------- quantities
    def quantity(self, name: str) -> np.ndarray:
        """[episode, query, 8 layers] for one declared quantity."""
        if name == "td_slope":
            return ols_slope(self.metric_plane("token_differentiation"))
        if name == "td_late_early":
            block = self.metric_plane("token_differentiation")
            return (block[..., 7:10].mean(-1) - block[..., 0:3].mean(-1)).astype(np.float32)
        if name == "saa_slope":
            return ols_slope(self.metric_plane("state_action_alignment"))
        if name == "pc1corr_slope":
            return ols_slope(self.arc_plane("pc1_index_corr"))
        if name == "pc1share_slope":
            return ols_slope(self.arc_plane("pc1_share"))
        if name == "stretch_slope":
            return ols_slope(self.arc_plane("arc_stretch"))
        if name == "centred_slope":
            return ols_slope(self.arc_plane("centred_energy"))
        if name == "along_state_fraction_s9":
            along = self.arc_plane("along_state_energy")[..., 9]
            centred = self.arc_plane("centred_energy")[..., 9]
            return (along / np.maximum(centred, 1e-12)).astype(np.float32)
        if name == "state_mobility_s9":
            return self.state_mobility_plane()[..., 9]
        if name == "state_mobility_mean":
            return self.state_mobility_plane().mean(-1).astype(np.float32)
        if name == "state_minus_action_mobility":
            return (
                self.state_mobility_plane()[..., 9] - self.mobility_plane()[..., 9]
            ).astype(np.float32)
        if name == "state_over_action_logratio":
            return (
                np.log(self.state_mobility_plane()[..., 9] + EPSILON)
                - np.log(self.mobility_plane()[..., 9] + EPSILON)
            ).astype(np.float32)
        if "_s" in name and name.rsplit("_s", 1)[1].isdigit():
            base, step = name.rsplit("_s", 1)
            step = int(step)
            if base == "mobility":
                return self.mobility_plane()[..., step]
            if base == "state_mobility":
                return self.state_mobility_plane()[..., step]
            if base in ARC_NAMES:
                return self.arc_plane(base)[..., step]
            return self.metric_plane(base)[..., step]
        raise KeyError(name)

    def release(self) -> None:
        self._cache.clear()
        self._arc = None
