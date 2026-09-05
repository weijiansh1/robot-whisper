"""Typed boundary between internal evidence, mode belief, and outcome assurance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutcomeEstimate:
    intervention: str
    horizon: str
    outcome: str
    posterior_mean: float
    ci95_low: float
    ci95_high: float
    support: int
    posterior: str


@dataclass(frozen=True)
class MoEAssuranceState:
    corpus: str
    suite: str
    task: str
    episode_id: int
    query: int

    # These are internal evidence coordinates, not outcome probabilities.
    commitment: float
    flow_coherence: float
    token_consensus: float
    effective_rank: float
    temporal_recurrence: list[float | None]
    healthy_energy: float
    static_lockin_evidence: float | None
    layer_evidence: dict[str, dict[str, float]]

    # These are posterior beliefs of a fitted mode model, not physical committors.
    mode_belief: dict[str, float]

    # Entries exist only where real repeated rollout/fork counts support them.
    outcome_assurance: list[OutcomeEstimate] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    schema: str = "himoe.moe_assurance_state.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
