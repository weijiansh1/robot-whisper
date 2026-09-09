"""Fixed, prefix-only MoE diagnostics. No fitted parameters or reference corpus."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
GROUPS = {"front": slice(0, 4), "back": slice(4, 8), "all": slice(0, 8)}
FEATURES = ("mobility", "displacement4", "token_spread")
EPS = 1e-10


def root_action_routes(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape[-4:] == (8, 10, 11, 32):
        p = p[..., :, 9, 1:, :]
    elif p.shape[-3:] != (8, 10, 32):
        raise ValueError("expected HB probabilities [...,8,10,11,32] or final action routes [...,8,10,32]")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=-1) <= 0).any():
        raise ValueError("probabilities must be finite, nonnegative, with positive row mass")
    return np.sqrt(p / p.sum(axis=-1, keepdims=True))


def distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # The norm form retains small distances that 1 - affinity loses to rounding.
    return (np.linalg.norm(left - right, axis=-1) / np.sqrt(2)).mean(axis=-1)


def features_from_roots(roots: np.ndarray) -> np.ndarray:
    if roots.ndim != 4 or roots.shape[1:] != (8, 10, 32):
        raise ValueError("expected one episode with shape [query,8,10,32]")
    out = np.full((len(roots), 8, len(FEATURES)), np.nan)
    out[1:, :, 0] = distance(roots[1:], roots[:-1])
    out[4:, :, 1] = distance(roots[4:], roots[:-4])
    centered = roots - roots.mean(axis=-2, keepdims=True)
    out[:, :, 2] = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=-1) / 2)
    return out


@dataclass(frozen=True)
class Rule:
    name: str
    signal: str
    group: str = "back"
    threshold: float = 0.25
    baseline: int = 4
    width: int = 2
    confirmations: int = 2

    def __post_init__(self) -> None:
        if self.signal not in ("freeze", "recurrence", "collapse", "joint"):
            raise ValueError("unknown signal")
        if self.group not in GROUPS or min(self.baseline, self.width, self.confirmations) < 1:
            raise ValueError("invalid rule")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")


PRIMARY = Rule("freeze_back_quarter", "freeze")


def fixed_rules() -> list[Rule]:
    rules = [PRIMARY]
    for group in ("back", "all", "front"):
        for threshold, label in ((0.5, "half"), (0.25, "quarter"), (0.125, "eighth")):
            rule = Rule(f"freeze_{group}_{label}", "freeze", group, threshold)
            if rule != PRIMARY:
                rules.append(rule)
    for confirmations in (1, 4):
        rules.append(Rule(f"freeze_back_quarter_k{confirmations}", "freeze", confirmations=confirmations))
    for baseline in (2, 8):
        rules.append(Rule(f"freeze_back_quarter_b{baseline}", "freeze", baseline=baseline))
    for threshold, label in ((0.15, "015"), (0.25, "025"), (0.35, "035")):
        rules.append(Rule(f"recurrence_back_{label}", "recurrence", threshold=threshold, width=1))
    for threshold, label in ((0.5, "half"), (0.25, "quarter")):
        rules.append(Rule(f"collapse_back_{label}", "collapse", threshold=threshold))
        rules.append(Rule(f"joint_back_{label}", "joint", threshold=threshold))
    # Follow-up sensitivity after transient false alarms in the initial audit.
    for confirmations in (1, 4, 8):
        rules.append(Rule(f"freeze_back_half_k{confirmations}", "freeze", threshold=0.5, confirmations=confirmations))
    return rules


def score_stream(features: np.ndarray, rule: Rule) -> np.ndarray:
    """[episode,query,layer,feature] -> persistent low score at each query.

    No length, task, labels, or values from other episodes enter this function.
    Undefined history stays NaN; zero baselines abstain instead of inventing risk.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 4 or x.shape[2:] != (8, len(FEATURES)):
        raise ValueError("expected [episode,query,8,3]")
    n, qmax = x.shape[:2]
    scalar = np.full((n, qmax), np.nan)
    selected = GROUPS[rule.group]
    if rule.signal == "recurrence":
        for q in range(4, qmax):
            path = x[:, q - 3:q + 1, :, 0].sum(axis=1)
            ratio = np.divide(x[:, q, :, 1], path, out=np.full_like(path, np.nan), where=path > EPS)
            scalar[:, q] = np.median(ratio[:, selected], axis=-1)
    elif qmax > rule.baseline:
        base = x[:, 1:rule.baseline + 1, :, :].mean(axis=1)
        mobility = np.divide(x[..., 0], base[:, None, :, 0], out=np.full_like(x[..., 0], np.nan), where=base[:, None, :, 0] > EPS)
        spread = np.divide(x[..., 2], base[:, None, :, 2], out=np.full_like(x[..., 2], np.nan), where=base[:, None, :, 2] > EPS)
        ratios = mobility if rule.signal == "freeze" else spread
        if rule.signal == "joint":
            ratios = np.maximum(mobility, spread)
        scalar = np.median(ratios[:, :, selected], axis=-1)
        scalar[:, :rule.baseline + 1] = np.nan
    smooth = np.full_like(scalar, np.nan)
    for q in range(rule.width - 1, qmax):
        smooth[:, q] = scalar[:, q - rule.width + 1:q + 1].mean(axis=1)
    persistent = np.full_like(scalar, np.nan)
    for q in range(rule.confirmations - 1, qmax):
        persistent[:, q] = smooth[:, q - rule.confirmations + 1:q + 1].max(axis=1)
    return persistent


def first_alarm(scores: np.ndarray, threshold: float) -> np.ndarray:
    hit = np.isfinite(scores) & (scores <= threshold)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


class HistoryOnlyMonitor:
    """Call update after each inference, before executing its action chunk."""

    def __init__(self, rule: Rule = PRIMARY):
        self.rule = rule
        self._roots: deque[np.ndarray] = deque(maxlen=4)
        self._features: list[np.ndarray] = []
        self.first_alarm_query = -1

    def update(self, hb_router_probs: np.ndarray) -> dict:
        root = root_action_routes(hb_router_probs)
        if root.shape != (8, 10, 32):
            raise ValueError("update accepts exactly one inference")
        features = np.full((8, len(FEATURES)), np.nan)
        if self._roots:
            features[:, 0] = distance(root, self._roots[-1])
        if len(self._roots) == 4:
            features[:, 1] = distance(root, self._roots[0])
        centered = root - root.mean(axis=-2, keepdims=True)
        features[:, 2] = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=-1) / 2)
        self._roots.append(root)
        self._features.append(features)
        q = len(self._features) - 1
        score = float(score_stream(np.asarray(self._features)[None], self.rule)[0, -1])
        trigger = np.isfinite(score) and score <= self.rule.threshold
        if trigger and self.first_alarm_query < 0:
            self.first_alarm_query = q
        return {"query": q, "score": score, "trigger_now": bool(trigger),
                "alarm": self.first_alarm_query >= 0, "first_alarm_query": self.first_alarm_query,
                "rule": asdict(self.rule)}
