#!/usr/bin/env python3
"""Runtime monitor for the task-conditional long-suite guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dual_regime_monitor import ConfigurableDualRegimeMonitor


PROFILE_SCHEMAS = {
    "himoe.long_guard_v5.profile.v1",
    "himoe.lead_guard_v5.profile.v1",
}


@dataclass(frozen=True)
class LongGuardTaskProfile:
    task: str
    lock_threshold: float
    instability_threshold: float
    lock_representation: str
    lock_width: int
    lock_confirmations: int
    instability_layer: str
    instability_width: int
    instability_confirmations: int

    @classmethod
    def load(cls, path: Path, task: str) -> "LongGuardTaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) not in PROFILE_SCHEMAS:
                raise ValueError("unknown task-conditional profile schema")
            names = archive["task_names"].astype(str).tolist()
            if task not in names:
                raise KeyError(f"task not present in profile: {task}")
            position = names.index(task)
            return cls(
                task=task,
                lock_threshold=float(archive["lock_thresholds"][position]),
                instability_threshold=float(
                    archive["instability_thresholds"][position]
                ),
                lock_representation=str(
                    archive["lock_representations"][position]
                ),
                lock_width=int(archive["lock_widths"][position]),
                lock_confirmations=int(
                    archive["lock_confirmations"][position]
                ),
                instability_layer=str(
                    archive["instability_layers"][position]
                ),
                instability_width=int(
                    archive["instability_widths"][position]
                ),
                instability_confirmations=int(
                    archive["instability_confirmations"][position]
                ),
            )


class LongGuardMonitor(ConfigurableDualRegimeMonitor):
    """Apply the selected L12 guard on long tasks and frozen v4 elsewhere."""

    def __init__(self, profile: LongGuardTaskProfile) -> None:
        super().__init__(
            profile,
            lock_representation=profile.lock_representation,
            lock_width=profile.lock_width,
            lock_confirmations=profile.lock_confirmations,
            instability_layer=profile.instability_layer,
            instability_width=profile.instability_width,
            instability_confirmations=profile.instability_confirmations,
        )
