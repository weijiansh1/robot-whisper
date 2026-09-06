#!/usr/bin/env python3
"""Shared loading, joining and scoring helpers for the failure-mode study.

Everything a detector reads comes from `hb_router_probs`-derived routing caches.
The simulator-backed physical labels are loaded here too, but they are used for
ANALYSIS AND VALIDATION ONLY: no function in this module lets a physical label
reach an alarm decision.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))
import evaluate_layerwise_alarm_development as dev  # noqa: E402

# --- frozen upstream constants (copied, not re-derived) ----------------------
WIDTH, CONFIRMATIONS = 4, 4  # moe-hb-front-back-0905/experiments/compare_layers_early.py
LOW_PRIOR = 0.25  # moe-hb-front-back-0905/experiments/select_early_lock.py

GRAPH_ROOT = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
MOBILITY_ROOT = PROJECT / "moe-v4-0904/results/layerwise_mobility"
LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
PHYSICAL = PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"
EXTERNAL_ALARMS = (
    PROJECT / "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz"
)
EXTERNAL_DETECTORS = (
    PROJECT / "moe-hb-front-back-0905/results/frame_survey/external_detectors.csv"
)
V7_ALARMS = PROJECT / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"

MOBILITY_PATHS = {
    "development_main": MOBILITY_ROOT / "main_reference.npz",
    "development_extra": MOBILITY_ROOT / "extra_reference.npz",
    "external_8b": MOBILITY_ROOT / "external_8b.npz",
}
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
RUN_IDS = {
    "development_main": "right-50x8-20260903",
    "development_extra": "right-50x8-20260903",
    "external_8b": "right-50x8b-20260903",
}

HORIZON_CAP = {
    "libero_goal": 30,
    "libero_long": 52,
    "libero_object": 28,
    "libero_spatial": 22,
}

# Physical failure modes, ordered by external frequency. Shortened names are the
# only thing that appears in plots/tables; the long name is the simulator label.
MODES = (
    "object_released_or_dropped_before_goal",
    "stable_grasp_not_observed",
    "object_moved_but_goal_unmet",
    "goal_predicate_regressed",
    "object_released_outside_goal",
    "timeout_while_holding_target",
    "approached_target_without_observed_contact",
    "mechanism_threshold_not_reached",
)
MODE_SHORT = {
    "object_released_or_dropped_before_goal": "dropped",
    "stable_grasp_not_observed": "no_grasp",
    "object_moved_but_goal_unmet": "moved_unmet",
    "goal_predicate_regressed": "regressed",
    "object_released_outside_goal": "released_outside",
    "timeout_while_holding_target": "timeout_holding",
    "approached_target_without_observed_contact": "no_contact",
    "mechanism_threshold_not_reached": "mechanism",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_graph(cohort: str) -> dict[str, np.ndarray]:
    return load_npz(GRAPH_ROOT / f"{cohort}.npz")


def suite_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    return np.asarray([name.split("/", 1)[0] for name in task])


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def survival_prior(
    suite: np.ndarray, length: np.ndarray, risk: np.ndarray
) -> dict[str, dict[int, float]]:
    """P(risk | still running at chunk q), per suite. Same definition as v4/v7."""
    priors: dict[str, dict[int, float]] = {}
    for name in np.unique(suite):
        take = suite == name
        priors[str(name)] = {
            chunk: float(risk[take][length[take] > chunk].mean())
            for chunk in range(int(length[take].max()))
        }
    return priors


def prior_of(
    first: np.ndarray, suite: np.ndarray, priors: dict[str, dict[int, float]]
) -> np.ndarray:
    out = np.full(len(first), np.nan)
    fired = first >= 0
    out[fired] = [
        priors[s][int(c)] for s, c in zip(suite[fired], first[fired], strict=True)
    ]
    return out


def score_candidate(
    first: np.ndarray, risk: np.ndarray, prior: np.ndarray
) -> dict[str, float | int]:
    """Identical to moe-hb-front-back-0905/experiments/select_early_lock.py."""
    fired = first >= 0
    early = fired & (prior < LOW_PRIOR)
    tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
    etp, efp = int((early & risk).sum()), int((early & ~risk).sum())
    mean_prior = float(np.nanmean(prior[fired])) if (tp + fp) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / int(risk.sum()),
        "timely_fpr": fp / int((~risk).sum()),
        "precision": tp / max(tp + fp, 1),
        "low_prior_tp": etp,
        "low_prior_fp": efp,
        "low_prior_precision": etp / max(etp + efp, 1),
        "low_prior_share": etp / max(tp, 1),
        "mean_alarm_prior": mean_prior,
        "lift": (tp / max(tp + fp, 1)) / mean_prior
        if (tp + fp) and np.isfinite(mean_prior)
        else float("nan"),
    }


# --- the join ---------------------------------------------------------------


def physical_table() -> pd.DataFrame:
    frame = pd.read_csv(PHYSICAL)
    frame["task_key"] = frame["suite"] + "/" + frame["task_name"]
    return frame


def cohort_index(cohort: str) -> pd.DataFrame:
    """Row-aligned index for one routing cache, with outcome + physical labels.

    The routing cache row order is authoritative: every returned frame has the
    cache's own row order, so alarm arrays index straight into it.
    """
    graph = load_graph(cohort)
    task = task_of(graph)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task_key": task,
            "suite": suite_of(graph),
            "episode": graph["episode"].astype(int),
            "init_state_id": graph["init_state_id"].astype(int),
            "flow_noise_seed": graph["flow_noise_seed"].astype(int),
            "length": graph["length"].astype(int),
        }
    )
    index["task_name"] = index["task_key"].str.split("/", n=1).str[1]

    label_key = "external_8b" if cohort == "external_8b" else "development_main"
    labels = pd.read_csv(LABEL_PATHS[label_key])[
        ["task", "episode", "original_failure", "failure",
         "late_success_plus10_queries", "episode_length"]
    ].rename(columns={"task": "task_key"})
    merged = index.merge(labels, on=["task_key", "episode"], how="left",
                         validate="one_to_one")
    if merged["original_failure"].isna().any():
        raise ValueError(f"{cohort}: outcome label join incomplete")
    if not (merged["episode_length"].to_numpy(int) == merged["length"].to_numpy(int)).all():
        raise ValueError(f"{cohort}: episode length disagreement between cache and labels")

    physical = physical_table()
    physical = physical[physical["run_id"] == RUN_IDS[cohort]]
    physical = physical[
        ["task_name", "episode_index", "init_state_id", "flow_noise_seed",
         "recorded_success", "primary_failure_reason", "failure_reason_confidence",
         "physics_validation_status"]
    ].rename(columns={"episode_index": "episode",
                      "init_state_id": "phys_init_state_id",
                      "flow_noise_seed": "phys_flow_noise_seed"})
    merged = merged.merge(physical, on=["task_name", "episode"], how="left",
                          validate="one_to_one")
    merged["physical_matched"] = merged["recorded_success"].notna()
    merged["risk"] = merged["original_failure"].astype(bool)
    merged["mode"] = merged["primary_failure_reason"].fillna("")
    if not (merged["row"].to_numpy() == np.arange(len(merged))).all():
        raise ValueError(f"{cohort}: merge reordered rows")
    return merged


# --- detector reconstruction -------------------------------------------------


def quantity_cache(
    graph: dict[str, np.ndarray], mobility: dict[str, np.ndarray], quantity: str
) -> dict[str, np.ndarray]:
    """Exactly survey_reference_frames.quantity_cache."""
    if quantity == "mobility":
        values = np.asarray(mobility["mobility"], dtype=np.float32)
    else:
        names = graph["metric_names"].astype(str).tolist()
        values = np.asarray(
            graph["metrics"][:, :, :, names.index(quantity)], dtype=np.float32
        )
    return {
        "mobility": values,
        "valid": graph["valid"].astype(bool),
        "layer_names": graph["layer_names"],
        "task_names": graph["task_names"],
        "task_index": graph["task_index"],
        "episode": graph["episode"],
        "init_state_id": graph["init_state_id"],
        "length": graph["length"],
    }


def oriented(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = dev.trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed
