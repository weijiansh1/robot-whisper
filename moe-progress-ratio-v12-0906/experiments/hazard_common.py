#!/usr/bin/env python3
"""Survival-prior helpers shared by the v12 hazard-aware evaluators.

Risk is "the rollout failed to finish before its horizon cap", so being still
alive at chunk q is already evidence of risk.  Any detector that simply alarms
late inherits that base rate for free.  These helpers make the base rate
explicit so a detector can be scored against it instead of against raw
precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def survival_prior(labels: pd.DataFrame) -> dict[str, dict[int, float]]:
    """P(risk | still running at chunk q)，按 suite 估计。"""
    priors: dict[str, dict[int, float]] = {}
    for suite, block in labels.groupby("suite"):
        risk = block["original_failure"].to_numpy(bool)
        length = block["length"].to_numpy(int)
        priors[str(suite)] = {
            chunk: float(risk[length > chunk].mean())
            for chunk in range(int(length.max()))
        }
    return priors


def matched_prior(alarms: pd.DataFrame, prior: dict[str, dict[int, float]]) -> np.ndarray:
    values = np.empty(len(alarms), dtype=np.float64)
    for position, (suite, chunk) in enumerate(
        zip(alarms["suite"], alarms["alarm_chunk"])
    ):
        table = prior.get(str(suite))
        if table is None or int(chunk) not in table:
            raise KeyError(f"no survival prior for suite {suite} at chunk {chunk}")
        values[position] = table[int(chunk)]
    return values
