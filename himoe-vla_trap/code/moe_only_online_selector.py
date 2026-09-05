#!/usr/bin/env python3
"""Train-free MoE-only online phase matching and persistent alarm logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from trainfree_belief_selector import calibrate_healthy, evaluate_stale_belief


EXPECTED_ROUTE_SHAPE = (8, 10, 11, 32)
FRONT_LAYERS = slice(0, 4)
ACTION_TOKENS = slice(1, 11)


def normalize(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float32)
    mass = value.sum(axis=-1, keepdims=True)
    if value.shape[-1] != 32 or not np.all(np.isfinite(value)) or np.any(mass <= 0):
        raise ValueError("routes must be finite positive-mass 32-expert probabilities")
    return np.maximum(value, 0.0) / np.maximum(mass, 1e-12)


def front_action_distance_to_sequence(
    query_route: np.ndarray, reference_routes: np.ndarray
) -> np.ndarray:
    """Hellinger distance from one route to every route in a sequence."""

    query = normalize(query_route)[FRONT_LAYERS, :, ACTION_TOKENS]
    reference = normalize(reference_routes)[:, FRONT_LAYERS, :, ACTION_TOKENS]
    coefficient = (
        np.sqrt(query)[None, ...] * np.sqrt(reference)
    ).sum(axis=-1)
    distance = np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))
    return distance.mean(axis=(1, 2, 3))


@dataclass(frozen=True)
class MoeOnlyDecision:
    query: int
    raw_reject: bool
    alarm: bool
    consecutive_raw_rejects: int
    matched_reference_queries: tuple[int, ...]
    mean_local_match_distance: float
    layer5_gap: float
    back_chunk_jump: float
    front_action_distance: float
    layer5_gap_ratio: float
    back_chunk_jump_ratio: float
    front_action_distance_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HealthySequenceBank:
    def __init__(self, routes: Sequence[np.ndarray], identities: Sequence[int]) -> None:
        if len(routes) < 3 or len(routes) != len(identities):
            raise ValueError("at least three aligned route sequences are required")
        self.routes = []
        for sequence in routes:
            value = normalize(sequence)
            if value.ndim != 5 or tuple(value.shape[1:]) != EXPECTED_ROUTE_SHAPE:
                raise ValueError("each sequence must be [Q,8,10,11,32]")
            if len(value) < 2:
                raise ValueError("each sequence needs at least two queries")
            self.routes.append(value)
        self.identities = tuple(int(value) for value in identities)

    @classmethod
    def load(
        cls, path: Path, exclude_identity: Optional[int] = None
    ) -> "HealthySequenceBank":
        with np.load(str(path), allow_pickle=False) as archive:
            if archive["schema"].item() != "himoe.moe_only_healthy_sequences.v1":
                raise RuntimeError("unsupported healthy sequence archive")
            if bool(archive["training"].item()):
                raise RuntimeError("healthy sequence archive declares training")
            if bool(archive["failure_labels_used"].item()):
                raise RuntimeError("healthy sequence archive used failure labels")
            if bool(archive["physical_alignment_used"].item()):
                raise RuntimeError("healthy sequence archive used physical alignment")
            padded = np.asarray(archive["routes"], dtype=np.float32)
            lengths = np.asarray(archive["lengths"], dtype=np.int64)
            identities = np.asarray(archive["candidates"], dtype=np.int64)
        keep = [
            index
            for index, identity in enumerate(identities)
            if exclude_identity is None or int(identity) != int(exclude_identity)
        ]
        return cls(
            [padded[index, : lengths[index]] for index in keep],
            [int(identities[index]) for index in keep],
        )


class OnlineMonotoneMatcher:
    """Closed-begin online DTW with stay, +1, or +2 reference transitions."""

    def __init__(self, bank: HealthySequenceBank, max_advance: int = 2) -> None:
        if max_advance not in (1, 2):
            raise ValueError("max_advance must be 1 or 2")
        self.bank = bank
        self.max_advance = max_advance
        self.cumulative = [
            np.full(len(sequence), np.inf, dtype=np.float64)
            for sequence in bank.routes
        ]
        self.transition_count = 0

    def update(
        self, current: np.ndarray, previous: np.ndarray
    ) -> tuple[np.ndarray, float]:
        positions = []
        selected_local_costs = []
        for index, reference in enumerate(self.bank.routes):
            current_distance = front_action_distance_to_sequence(
                current, reference[1:]
            )
            previous_distance = front_action_distance_to_sequence(
                previous, reference[:-1]
            )
            local = 0.5 * (current_distance + previous_distance)
            updated = np.full(len(reference), np.inf, dtype=np.float64)
            if self.transition_count == 0:
                updated[1] = float(local[0])
            else:
                old = self.cumulative[index]
                for position in range(1, len(reference)):
                    predecessors = [old[position]]
                    predecessors.append(old[position - 1])
                    if self.max_advance == 2 and position >= 2:
                        predecessors.append(old[position - 2])
                    updated[position] = float(local[position - 1]) + min(predecessors)
            best = int(np.argmin(updated))
            if not np.isfinite(updated[best]):
                raise RuntimeError("online DTW has no reachable healthy phase")
            self.cumulative[index] = updated
            positions.append(best)
            selected_local_costs.append(float(local[best - 1]))
        self.transition_count += 1
        return np.asarray(positions, dtype=np.int16), float(
            np.mean(selected_local_costs)
        )


class MoeOnlyOnlineAlarm:
    def __init__(
        self,
        bank: HealthySequenceBank,
        persistence: int = 2,
        max_advance: int = 2,
    ) -> None:
        if persistence < 1:
            raise ValueError("persistence must be positive")
        self.bank = bank
        self.matcher = OnlineMonotoneMatcher(bank, max_advance=max_advance)
        self.persistence = int(persistence)
        self.previous: Optional[np.ndarray] = None
        self.query = -1
        self.consecutive = 0

    def update(self, route: np.ndarray) -> Optional[MoeOnlyDecision]:
        current = normalize(route)
        self.query += 1
        if self.previous is None:
            self.previous = current
            return None
        positions, match_distance = self.matcher.update(current, self.previous)
        reference_current = np.stack([
            sequence[int(position)]
            for sequence, position in zip(self.bank.routes, positions)
        ])
        reference_previous = np.stack([
            sequence[int(position) - 1]
            for sequence, position in zip(self.bank.routes, positions)
        ])
        thresholds = calibrate_healthy(reference_current, reference_previous)
        route_decision = evaluate_stale_belief(
            current, self.previous, reference_current, thresholds
        )
        self.consecutive = self.consecutive + 1 if route_decision.reject_stale_chunk else 0
        alarm = self.consecutive >= self.persistence
        self.previous = current
        return MoeOnlyDecision(
            query=self.query,
            raw_reject=route_decision.reject_stale_chunk,
            alarm=alarm,
            consecutive_raw_rejects=self.consecutive,
            matched_reference_queries=tuple(int(value) for value in positions),
            mean_local_match_distance=match_distance,
            layer5_gap=route_decision.layer5_gap,
            back_chunk_jump=route_decision.back_chunk_jump,
            front_action_distance=route_decision.front_action_distance,
            layer5_gap_ratio=route_decision.layer5_gap_ratio,
            back_chunk_jump_ratio=route_decision.back_chunk_jump_ratio,
            front_action_distance_ratio=route_decision.front_action_distance_ratio,
        )
