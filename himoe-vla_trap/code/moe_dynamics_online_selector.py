#!/usr/bin/env python3
"""Train-free online alarm based on back/front MoE route acceleration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from moe_only_online_selector import (
    EXPECTED_ROUTE_SHAPE,
    HealthySequenceBank,
    OnlineMonotoneMatcher,
    normalize,
)


CALIBRATION_SCHEMA = "himoe.moe_dynamics_calibration.v1"
SELECTOR_VERSION = "back_front_route_acceleration_v2"
FRONT_LAYERS = slice(0, 4)
BACK_LAYERS = slice(4, 8)
ACTION_TOKENS = slice(1, 11)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_acceleration(route: np.ndarray, layers: slice) -> float:
    """Mean Hellinger-chord second difference over flow and action tokens."""

    value = normalize(route)
    if tuple(value.shape) != EXPECTED_ROUTE_SHAPE:
        raise ValueError(f"route must have shape {EXPECTED_ROUTE_SHAPE}")
    root = np.sqrt(value[layers, :, ACTION_TOKENS])
    second_difference = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
    magnitude = np.linalg.norm(second_difference, axis=-1) / np.sqrt(2.0)
    return float(magnitude.mean())


def acceleration_metrics(route: np.ndarray) -> tuple[float, float, float]:
    front = route_acceleration(route, FRONT_LAYERS)
    back = route_acceleration(route, BACK_LAYERS)
    return front, back, back / max(front, 1e-12)


def normalized_acceleration_excess(
    current: np.ndarray, reference_current: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Contrast current back/front acceleration with phase-matched healthy routes."""

    if reference_current.ndim != 5 or len(reference_current) < 3:
        raise ValueError("reference_current must be [N,8,10,11,32], N >= 3")
    front, back, contrast = acceleration_metrics(current)
    healthy_contrasts = np.asarray(
        [acceleration_metrics(route)[2] for route in reference_current],
        dtype=np.float64,
    )
    healthy_max = float(healthy_contrasts.max())
    return front, back, contrast, healthy_max, contrast / max(healthy_max, 1e-12)


@dataclass(frozen=True)
class DynamicsCalibration:
    threshold: float
    persistence: int
    max_reference_advance: int
    healthy_reference_sha256: str
    healthy_reference_identities: tuple[int, ...]

    @classmethod
    def load(
        cls,
        path: Path,
        healthy_reference: Path,
        bank: HealthySequenceBank,
    ) -> "DynamicsCalibration":
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise RuntimeError("unsupported dynamics calibration schema")
        if payload.get("training") is not False:
            raise RuntimeError("dynamics calibration declares training")
        if payload.get("failure_labels_used_to_set_threshold") is not False:
            raise RuntimeError("dynamics threshold used failure labels")
        observed_digest = sha256_file(healthy_reference)
        if payload.get("healthy_reference_sha256") != observed_digest:
            raise RuntimeError("healthy reference digest does not match calibration")
        identities = tuple(int(value) for value in payload["healthy_reference_identities"])
        if identities != bank.identities:
            raise RuntimeError("healthy reference identities do not match calibration")
        return cls(
            threshold=float(payload["normalized_excess_threshold"]),
            persistence=int(payload["persistence"]),
            max_reference_advance=int(payload["max_reference_advance"]),
            healthy_reference_sha256=observed_digest,
            healthy_reference_identities=identities,
        )


@dataclass(frozen=True)
class MoeDynamicsDecision:
    query: int
    raw_reject: bool
    alarm: bool
    consecutive_raw_rejects: int
    matched_reference_queries: tuple[int, ...]
    mean_local_match_distance: float
    front_route_acceleration: float
    back_route_acceleration: float
    back_front_acceleration_ratio: float
    matched_healthy_ratio_max: float
    normalized_acceleration_excess: float
    normalized_excess_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MoeDynamicsOnlineAlarm:
    """Frozen v2 alarm; it consumes routing and a healthy reference bank only."""

    metric_names = (
        "mean_local_match_distance",
        "front_route_acceleration",
        "back_route_acceleration",
        "back_front_acceleration_ratio",
        "matched_healthy_ratio_max",
        "normalized_acceleration_excess",
        "normalized_excess_threshold",
    )

    def __init__(
        self,
        bank: HealthySequenceBank,
        calibration: DynamicsCalibration,
    ) -> None:
        self.bank = bank
        self.calibration = calibration
        self.matcher = OnlineMonotoneMatcher(
            bank, max_advance=calibration.max_reference_advance
        )
        self.previous: np.ndarray | None = None
        self.query = -1
        self.consecutive = 0

    def update(self, route: np.ndarray) -> MoeDynamicsDecision | None:
        current = normalize(route)
        self.query += 1
        if self.previous is None:
            self.previous = current
            return None
        positions, match_distance = self.matcher.update(current, self.previous)
        reference_current = np.stack(
            [
                sequence[int(position)]
                for sequence, position in zip(self.bank.routes, positions)
            ]
        )
        front, back, contrast, healthy_max, excess = normalized_acceleration_excess(
            current, reference_current
        )
        raw_reject = bool(excess > self.calibration.threshold)
        self.consecutive = self.consecutive + 1 if raw_reject else 0
        alarm = self.consecutive >= self.calibration.persistence
        self.previous = current
        return MoeDynamicsDecision(
            query=self.query,
            raw_reject=raw_reject,
            alarm=alarm,
            consecutive_raw_rejects=self.consecutive,
            matched_reference_queries=tuple(int(value) for value in positions),
            mean_local_match_distance=match_distance,
            front_route_acceleration=front,
            back_route_acceleration=back,
            back_front_acceleration_ratio=contrast,
            matched_healthy_ratio_max=healthy_max,
            normalized_acceleration_excess=excess,
            normalized_excess_threshold=self.calibration.threshold,
        )

