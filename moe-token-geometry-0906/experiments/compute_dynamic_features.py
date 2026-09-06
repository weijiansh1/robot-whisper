#!/usr/bin/env python3
"""Causal shape-change features: how the token configuration moves within a run.

Two reference frames, both scale-invariant and both causal:

    procrustes_prev  full Procrustes disparity between the configuration at
                     query q and the configuration at query q - 1.
    procrustes_self  full Procrustes disparity between the configuration at
                     query q and the mean configuration of queries 0 .. q - 1
                     of the same rollout.

``procrustes_self`` is the only quantity in this bundle whose reference is the
rollout's own history rather than a corpus template, so a detector built on it
needs no task identity and no reference cohort at all.  Both are undefined at
q = 0 and are stored as NaN there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import geometry as G


DEFAULT_OUTPUT = G.FEATURE_ROOT
COHORTS = ("development_main", "development_extra", "external_8b")
DYNAMIC_NAMES = ("procrustes_prev", "procrustes_self")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def disparity(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Element-wise full Procrustes disparity between two kernel stacks."""
    a_values, a_coordinates = G.configuration(G.double_centre(reference))
    b_values, b_coordinates = G.configuration(G.double_centre(current))
    cross = np.einsum("...ji,...jk->...ik", a_coordinates, b_coordinates, optimize=True)
    nuclear = np.linalg.svd(cross, compute_uv=False).sum(axis=-1)
    denominator = np.maximum(a_values.sum(axis=-1) * b_values.sum(axis=-1), G.EPSILON)
    return np.clip(1.0 - nuclear**2 / denominator, 0.0, 1.0)


def build(cohort: str, output: Path) -> dict[str, int]:
    index = G.load_index(cohort)
    valid = index["valid"].astype(bool)
    lengths = index["length"].astype(int)
    packed = np.asarray(G.load_packed(cohort, mmap=False))
    dense = np.zeros(valid.shape + (8, 55), dtype=np.float32)
    dense[valid] = packed
    del packed

    running = np.cumsum(dense, axis=1, dtype=np.float64)
    counts = np.arange(1, valid.shape[1] + 1, dtype=np.float64)
    running /= counts[None, :, None, None]

    out = np.full(valid.shape + (8, 2), np.nan, dtype=np.float32)
    step = 2000
    for start in range(0, len(dense), step):
        stop = min(start + step, len(dense))
        current = G.symmetrise(dense[start:stop, 1:])
        previous = G.symmetrise(dense[start:stop, :-1])
        history = G.symmetrise(running[start:stop, :-1].astype(np.float32))
        out[start:stop, 1:, :, 0] = disparity(previous, current)
        out[start:stop, 1:, :, 1] = disparity(history, current)
    del dense, running

    expected = valid.copy()
    expected[:, 0] = False
    if not np.isfinite(out[expected]).all():
        raise ValueError(f"{cohort}: non-finite dynamic feature on a defined query")
    out[~expected] = np.nan
    if int(expected.sum()) != int(valid.sum()) - int((lengths > 0).sum()):
        raise ValueError(f"{cohort}: unexpected defined-query count")

    packed_out = np.ascontiguousarray(out[valid])
    np.save(output / f"{cohort}_dynamic.npy", packed_out)
    return {"cohort": cohort, "valid_queries": int(valid.sum())}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [build(cohort, args.output) for cohort in COHORTS]
    (args.output / "dynamic_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.dynamic.v1",
                "names": list(DYNAMIC_NAMES),
                "causal": True,
                "first_defined_query": 1,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
