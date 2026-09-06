"""Fourth layer: outcome and intervention semantics attached to grammar contexts.

`idea.md` separates two things that the existing audits collapsed together:

* the **grammar** is the conditional transition probability of healthy sentences;
* the **semantics** is what a sentence implies about the future -- loop, static,
  return, or completion -- and what response it calls for.

Every previous result scored a grammar surprisal and regressed it directly against the
final episode outcome. That is not what the design asked for. This module implements the
missing layer: a suffix trie whose nodes carry Dirichlet outcome counts over a fixed
horizon, with backoff from long to short contexts, plus an arm-level intervention table.

Contexts are word tuples ordered most-recent-first, so a node's suffix parent is simply
the tuple with its last element dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Outcome vocabulary. `idea.md` lists loop/static/return/success/other; this corpus has
# no reliable loop label, and `static` exists only in the physically annotated anchor, so
# the shared alphabet keeps the four that are defined everywhere and records `static`
# separately where it is available.
OUTCOMES = ("return", "persist", "end_success", "end_failure")


@dataclass
class SemanticSuffixTrie:
    """Dirichlet outcome counts per grammar context, with suffix backoff.

    ``alpha`` is the symmetric Dirichlet prior. ``min_support`` is the count below which
    a node defers to its suffix parent, which is how a long context with three
    observations avoids reporting a confident semantics.
    """

    max_order: int = 3
    alpha: float = 1.0
    min_support: int = 30
    outcomes: tuple[str, ...] = OUTCOMES

    def __post_init__(self) -> None:
        if self.max_order < 0 or self.alpha <= 0.0 or self.min_support < 1:
            raise ValueError("invalid semantic trie hyperparameters")
        self.counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.outcomes), dtype=np.int64)
        )

    def _contexts(self, history: np.ndarray) -> list[tuple[int, ...]]:
        """Return every context from order 0 up to ``max_order``, most recent first."""
        recent = [int(value) for value in history[::-1][: self.max_order]]
        return [tuple(recent[:order]) for order in range(len(recent) + 1)]

    def add(self, history: np.ndarray, outcome: str) -> None:
        index = self.outcomes.index(outcome)
        for context in self._contexts(history):
            self.counts[context][index] += 1

    def fit(self, observations: list[tuple[np.ndarray, str]]) -> "SemanticSuffixTrie":
        for history, outcome in observations:
            self.add(history, outcome)
        return self

    def posterior(self, history: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return ``(mean Dirichlet posterior, backoff order, support)``.

        Walks from the longest available context down to the root and keeps the longest
        one that clears ``min_support``.
        """
        chosen = self.counts.get((), np.zeros(len(self.outcomes), dtype=np.int64))
        order = 0
        for candidate_order, context in enumerate(self._contexts(history)):
            counts = self.counts.get(context)
            if counts is None or counts.sum() < self.min_support:
                continue
            chosen = counts
            order = candidate_order
        posterior = chosen + self.alpha
        return posterior / posterior.sum(), order, int(chosen.sum())

    def probability(self, history: np.ndarray, outcome: str) -> float:
        return float(self.posterior(history)[0][self.outcomes.index(outcome)])


@dataclass
class InterventionTable:
    """Counts ``n[arm][outcome]`` for snapshot-fork arms, blocked by trunk.

    The design wants ``P(Y | c, do(u))`` conditioned on the routing context. With only a
    handful of independent trunks that is not estimable, so this deliberately stops at
    the arm level and keeps the trunk identity so inference can resample trunks rather
    than candidates, which are not independent within a fork.
    """

    alpha: float = 1.0
    trunk_arm_counts: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    def add(self, trunk: str, arm: str, success: bool) -> None:
        key = (trunk, arm)
        if key not in self.trunk_arm_counts:
            self.trunk_arm_counts[key] = [0, 0]
        self.trunk_arm_counts[key][0] += int(success)
        self.trunk_arm_counts[key][1] += 1

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted({arm for _, arm in self.trunk_arm_counts}))

    @property
    def trunks(self) -> tuple[str, ...]:
        return tuple(sorted({trunk for trunk, _ in self.trunk_arm_counts}))

    def posterior_mean(self, arm: str) -> float:
        successes = sum(v[0] for (_, a), v in self.trunk_arm_counts.items() if a == arm)
        total = sum(v[1] for (_, a), v in self.trunk_arm_counts.items() if a == arm)
        return float((successes + self.alpha) / (total + 2.0 * self.alpha))

    def trunk_blocked_interval(
        self, arm: str, draws: int = 5000, seed: int = 20260905
    ) -> tuple[float, float] | None:
        """Resample whole trunks, since candidates inside one fork share a prefix."""
        trunks = [
            t
            for t in self.trunks
            if (t, arm) in self.trunk_arm_counts and self.trunk_arm_counts[(t, arm)][1] > 0
        ]
        if len(trunks) < 2:
            return None
        successes = np.asarray([self.trunk_arm_counts[(t, arm)][0] for t in trunks], dtype=float)
        totals = np.asarray([self.trunk_arm_counts[(t, arm)][1] for t in trunks], dtype=float)
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(trunks), size=(draws, len(trunks)))
        rates = successes[index].sum(axis=1) / np.maximum(totals[index].sum(axis=1), 1.0)
        return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))

    def contrast_interval(
        self, arm: str, baseline: str, draws: int = 5000, seed: int = 20260905
    ) -> tuple[float, tuple[float, float]] | None:
        """Paired trunk-blocked contrast, keeping only trunks that ran both arms."""
        shared = [
            t
            for t in self.trunks
            if (t, arm) in self.trunk_arm_counts and (t, baseline) in self.trunk_arm_counts
        ]
        if len(shared) < 2:
            return None
        arm_successes = np.asarray([self.trunk_arm_counts[(t, arm)][0] for t in shared], float)
        arm_totals = np.asarray([self.trunk_arm_counts[(t, arm)][1] for t in shared], float)
        base_successes = np.asarray(
            [self.trunk_arm_counts[(t, baseline)][0] for t in shared], float
        )
        base_totals = np.asarray([self.trunk_arm_counts[(t, baseline)][1] for t in shared], float)
        observed = arm_successes.sum() / arm_totals.sum() - base_successes.sum() / base_totals.sum()
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(shared), size=(draws, len(shared)))
        differences = arm_successes[index].sum(axis=1) / np.maximum(
            arm_totals[index].sum(axis=1), 1.0
        ) - base_successes[index].sum(axis=1) / np.maximum(base_totals[index].sum(axis=1), 1.0)
        return float(observed), (
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        )


def label_outcomes(
    deviation: np.ndarray,
    start: int,
    length: int,
    horizon: int,
    success: bool,
    threshold: float = 0.8,
) -> list[tuple[int, str]]:
    """Label what happens in the ``horizon`` queries after each deviating position.

    Only positions that are currently deviating get a label: asking "does this sentence
    return" is meaningless while nothing has gone off-manifold. ``end_success`` and
    ``end_failure`` take precedence, because an episode that finishes inside the horizon
    has no later return to observe.
    """
    output: list[tuple[int, str]] = []
    for offset in range(length):
        row = start + offset
        if deviation[row] < threshold:
            continue
        remaining = length - offset - 1
        future = deviation[row + 1 : row + 1 + min(horizon, remaining)]
        if remaining <= horizon:
            output.append((offset, "end_success" if success else "end_failure"))
        elif np.any(future < threshold):
            output.append((offset, "return"))
        else:
            output.append((offset, "persist"))
    return output
