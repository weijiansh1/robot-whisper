"""Frozen, training-free comparison of vector modes and scalar candidate risk."""

import hashlib
import json

import numpy as np


PROTOCOL = "moe_control.native_long_modes.v1"
SIGNALS = ("freeze", "acceleration", "periodicity", "frontback_inversion", "curvature")
LETTERS = ("F", "A", "P", "I", "C")
FAMILIES = ("resample", "random", "scalar", "mode", "short", "hold", "withdraw")
ARMS = ("native",) + tuple(family+"_r"+str(repeat) for family in FAMILIES for repeat in range(2))
SETTINGS = dict(trigger="v82_frozen", slope=-.0015, horizon=520, chunk=10,
    repeats=2, candidates=16, search_queries=5, short_chunk=2, short_queries=5,
    physical_steps=16, training=False, hidden_capture=False, threshold_fitting=False,
    entry_probe="one original-noise forward before intervention; no history or RNG commit",
    mode="current F/A/P strictly above threshold; I/C at or above threshold; five finite scores required",
    target="minus one margin for active default-candidate components, zero for inactive components",
    objective="sum of squared positive target violations across all five components",
    abstain="mode selects candidate zero if the default has no active components or any score is missing",
    scalar="max(rF, min(rA, rP), rI, rC)",
    selection="whole generated action chunk; first minimum wins ties; no action averaging or gate edits",
    noise="common per-parent/repeat/query/candidate SHA256 streams across all non-native families",
    physical="existing fixed 8-step lift plus 8-step retreat, or 16-step hold; last gripper sign",
    stop="success or original total 520 environment steps; candidate search ends after five queries",
    analysis="pre-intervention entry mode; two repeats reported separately and averaged, never best-of-two")


def decode_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown mode experiment arm")
    return ("native", -1) if arm == "native" else (arm[:-3], int(arm[-1]))


def stream(main_id, repeat, query, candidate, name):
    if repeat not in (0, 1) or query < 0 or candidate < 0:
        raise ValueError("Invalid paired stream coordinates")
    key = json.dumps([PROTOCOL, main_id, repeat, query, candidate, name], separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16], "little")
    return np.random.default_rng(seed)


def noise_for(main_id, repeat, query, candidate=0):
    return stream(main_id, repeat, query, candidate, "flow").standard_normal((10, 24)).astype(np.float32)


def thresholds_at(query, base):
    if query < 0 or int(query) != query:
        raise ValueError("Expected an absolute executed-query index")
    values = np.asarray(base, float).copy()
    if values.shape != (5,):
        raise ValueError("Five thresholds required")
    values[3:] += SETTINGS["slope"]*query
    return values


def normalized(scores, thresholds, margins):
    values, scale = np.asarray(scores, float), np.asarray(margins, float)
    if values.shape[-1] != 5 or scale.shape != (5,) or not np.all(scale > 0):
        raise ValueError("Five scores and positive margins required")
    return (values-np.asarray(thresholds, float))/scale


def active_mask(scores, thresholds):
    values, tau = np.asarray(scores, float), np.asarray(thresholds, float)
    if values.shape != (5,) or tau.shape != (5,):
        raise ValueError("One five-component state required")
    return np.isfinite(values) & np.r_[values[:3] > tau[:3], values[3:] >= tau[3:]]


def mode_name(scores, thresholds):
    if not np.isfinite(scores).all():
        return "unavailable"
    mask = active_mask(scores, thresholds)
    return "+".join(letter for letter, enabled in zip(LETTERS, mask) if enabled) or "none"


def scalar_cost(values):
    r = np.asarray(values, float)
    result = np.maximum.reduce([r[..., 0], np.minimum(r[..., 1], r[..., 2]), r[..., 3], r[..., 4]])
    return np.where(np.isfinite(r).all(axis=-1), result, np.inf)


def vector_cost(values, active):
    r, active = np.asarray(values, float), np.asarray(active, bool)
    if r.shape[-1] != 5 or active.shape != (5,):
        raise ValueError("Five-dimensional mode required")
    target = np.where(active, -1., 0.)
    cost = np.square(np.maximum(r-target, 0.)).sum(axis=-1)
    return np.where(np.isfinite(r).all(axis=-1), cost, np.inf)


def select(family, scores, thresholds, margins, main_id, repeat, query):
    values = np.asarray(scores, float)
    if family not in ("random", "scalar", "mode") or values.shape != (SETTINGS["candidates"], 5):
        raise ValueError("A complete paired pool is required")
    r = normalized(values, thresholds, margins)
    active = active_mask(values[0], thresholds)
    scalar, vector = scalar_cost(r), vector_cost(r, active)
    if family == "random":
        picked = int(stream(main_id, repeat, query, 0, "choice").integers(len(values)))
    elif family == "mode" and (not active.any() or not np.isfinite(values[0]).all()):
        picked = 0
    else:
        cost = scalar if family == "scalar" else vector
        picked = int(np.argmin(cost)) if np.isfinite(cost).any() else 0
    return picked, scalar, vector


class ModeTrace:
    """Record trends and duration on executed history, independently of alarm latches."""

    def __init__(self):
        self.previous_scores = np.full(5, np.nan)
        self.previous_thresholds = np.full(5, np.nan)
        self.previous_normalized = np.full(5, np.nan)
        self.streaks = np.zeros(5, np.int32)

    def commit(self, scores, thresholds, margins):
        scores, thresholds = np.asarray(scores, float), np.asarray(thresholds, float)
        active, r = active_mask(scores, thresholds), normalized(scores, thresholds, margins)
        self.streaks = np.where(active, self.streaks+1, 0).astype(np.int32)
        row = dict(component_scores=scores.copy(), selector_thresholds=thresholds.copy(),
            normalized_scores=r.copy(), active_components=active, active_streaks=self.streaks.copy(),
            mode=np.asarray(mode_name(scores, thresholds), dtype="S32"),
            raw_score_delta=scores-self.previous_scores,
            threshold_delta=thresholds-self.previous_thresholds,
            normalized_delta=r-self.previous_normalized)
        self.previous_scores, self.previous_thresholds, self.previous_normalized = scores.copy(), thresholds.copy(), r.copy()
        return row
