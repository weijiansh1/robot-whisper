"""Use the existing frozen detector and share batch/online feature semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


DETECTOR_PATH = (Path(__file__).resolve().parents[2] / "VLA_MUI_HUB"
                 / "moe-history-only" / "monitor.py")
spec = importlib.util.spec_from_file_location("probability_history_detector", DETECTOR_PATH)
detector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = detector
spec.loader.exec_module(detector)

BUDGET_NAMES = ["remaining_action_steps", "elapsed_action_steps"]
SIGNAL_NAMES = ["mobility", "displacement4", "token_spread"]
FEATURE_NAMES = BUDGET_NAMES + [
    f"{group}_{signal}_{stat}"
    for group in ("front", "back")
    for signal in SIGNAL_NAMES
    for stat in ("current", "mean4", "change3", "baseline_ratio")
] + ["freeze_score", "recurrence_score", "alarm_now", "alarm_streak",
     "ever_alarm", "queries_since_first_alarm"]


def history_features(features: np.ndarray, max_steps: int, replan_steps: int = 10):
    """Rows are after inference q, before executing its already sampled chunk.

    No terminal length, endpoint, future window, task, or physical state is used.
    The number of input rows only allocates output; every row is prefix invariant.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (8, 3) or not len(x):
        raise ValueError("expected nonempty [query,8,3] causal route features")
    if max_steps <= 0 or replan_steps <= 0 or (len(x) - 1) * replan_steps >= max_steps:
        raise ValueError("queries must precede the execution deadline")
    if np.isinf(x).any():
        raise ValueError("infinite features")
    q = np.arange(len(x))
    output = [max_steps - q * replan_steps, q * replan_steps]
    for selected in (slice(0, 4), slice(4, 8)):
        for k in range(3):
            signal = np.median(x[:, selected, k], axis=1)
            mean4 = np.full(len(x), np.nan)
            change3 = np.full(len(x), np.nan)
            ratio = np.full(len(x), np.nan)
            for t in range(len(x)):
                observed = signal[max(0, t - 3):t + 1]
                observed = observed[np.isfinite(observed)]
                if len(observed):
                    mean4[t] = observed.mean()
                if t >= 3:
                    change3[t] = signal[t] - signal[t - 3]
                # Baseline is available only after q4, never filled backwards.
                if t >= 4:
                    base = signal[1:5]
                    base = base[np.isfinite(base)]
                    if len(base) and base.mean() > detector.EPS:
                        ratio[t] = signal[t] / base.mean()
            output.extend([signal, mean4, change3, ratio])
    score = detector.score_stream(x[None], detector.PRIMARY)[0]
    recurrence = detector.score_stream(
        x[None], detector.Rule("recurrence", "recurrence", threshold=0.25)
    )[0]
    now = np.isfinite(score) & (score <= detector.PRIMARY.threshold)
    streak = np.zeros(len(x), dtype=int)
    first = np.flatnonzero(now)
    ever = np.zeros(len(x), dtype=bool)
    age = np.zeros(len(x), dtype=int)
    for t in range(len(x)):
        streak[t] = (streak[t - 1] + 1 if t else 1) if now[t] else 0
    if len(first):
        ever[first[0]:] = True
        age[first[0]:] = q[first[0]:] - first[0]
    output.extend([score, recurrence, now, streak, ever, age])
    result = np.column_stack(output).astype(np.float32)
    if result.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("feature schema mismatch")
    return result, dict(score=score, alarm_now=now, ever_alarm=ever,
                        first_alarm=int(first[0]) if len(first) else -1)


class PrefixFeatures:
    """Streaming adapter accepting one raw router tensor per inference."""

    def __init__(self, max_steps: int, replan_steps: int = 10):
        self.max_steps = max_steps
        self.replan_steps = replan_steps
        self.roots = []
        self.history = []

    def update(self, hb_router_probs):
        root = detector.root_action_routes(hb_router_probs)
        if root.shape != (8, 10, 32):
            raise ValueError("expected one [8,10,11,32] or [8,10,32] tensor")
        feature = np.full((8, 3), np.nan)
        if self.roots:
            feature[:, 0] = detector.distance(root, self.roots[-1])
        if len(self.roots) == 4:
            feature[:, 1] = detector.distance(root, self.roots[0])
        centered = root - root.mean(axis=-2, keepdims=True)
        feature[:, 2] = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=-1) / 2)
        self.roots.append(root)
        self.roots = self.roots[-4:]
        # Match the audited float32 feature cache used during training.
        self.history.append(feature.astype(np.float32))
        x, info = history_features(np.asarray(self.history), self.max_steps, self.replan_steps)
        return x[-1:], {k: v[-1].item() if isinstance(v, np.ndarray) else v
                       for k, v in info.items()}

