#!/usr/bin/env python3
"""Detector arms: the real routing channels and the two mandated null arms.

The null arms are *not* a permutation of the labels.  They are a full re-run of
the identical pipeline on channels that carry no information by construction:

  ``white``          iid N(0,1) at every (episode, chunk, layer, channel);
  ``episode_const``  one N(0,1) per (episode, layer, channel), held constant for
                     the whole episode.

Both arms are given exactly as many columns as the real arm they shadow, so the
selection stage searches a search space of the same size.  Anything the real arm
achieves that the null arm also achieves is a search artefact.
"""

from __future__ import annotations

import numpy as np

import protocol as P

ARMS = ("real", "white", "episode_const")


def arm_values(frame: dict, arm: str, columns: list[str], seed: int) -> np.ndarray:
    """The (n, chunk, len(columns)) block this arm presents to the pipeline."""
    idx = [frame["columns"].index(c) for c in columns]
    real = frame["values"][:, :, idx]
    if arm == "real":
        return real
    rng = np.random.default_rng(seed)
    n, n_chunk, n_col = real.shape
    if arm == "white":
        out = rng.standard_normal((n, n_chunk, n_col), dtype=np.float32)
    elif arm == "episode_const":
        out = np.repeat(
            rng.standard_normal((n, 1, n_col), dtype=np.float32), n_chunk, axis=1
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    # the null channels are observable exactly where the real ones are
    out[~np.isfinite(real)] = np.nan
    out[~frame["alive"]] = np.nan
    return out


def arm_seed(arm: str, cohort: str) -> int:
    return P.SEED + 1009 * ARMS.index(arm) + 7 * P.COHORTS.index(cohort)
