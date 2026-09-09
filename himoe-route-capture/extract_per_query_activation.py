#!/usr/bin/env python3
"""Cache per-query MoE activation concentration for every captured route row.

Only permutation-invariant concentration statistics are kept: normalized
entropy and top-4 probability mass.  Both are unchanged when tied experts swap
rank, so they are immune to the p4=p5 boundary ambiguity that makes hard
expert identity unidentifiable in this cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

import analyze_rolling_star_experiment as rolling

BLOCK = 256


def concentration(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized entropy and top-4 mass over the final expert axis."""
    total = probability.sum(axis=-1, keepdims=True)
    normalized = probability / np.maximum(total, 1e-12)
    entropy = -np.sum(
        normalized * np.log(np.maximum(normalized, 1e-12)), axis=-1
    ) / np.log(normalized.shape[-1])
    partition = np.partition(normalized, -4, axis=-1)[..., -4:]
    return entropy, partition.sum(axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_path = args.out or run_root / "analysis_trap_onset" / "per_query_activation.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    probs = store["hb_router_probs"]
    n_row = probs.shape[0]
    groups = list(rolling.LAYER_GROUPS.items())
    n_denoise = probs.shape[2]

    # feature axis: token_family x group x denoise x {entropy, top4}
    columns: list[str] = []
    for family in ("state", "action"):
        for group_name, _ in groups:
            for denoise in range(n_denoise):
                for metric in ("entropy", "top4"):
                    columns.append(f"{family}|{group_name}|d{denoise}|{metric}")
    values = np.zeros((n_row, len(columns)), dtype=np.float32)

    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        block = probs[start:stop].astype(np.float32)
        state = block[:, :, :, 0, :]
        action = block[:, :, :, 1:, :]
        state_entropy, state_top4 = concentration(state)
        action_entropy, action_top4 = concentration(action)
        # average over action tokens
        action_entropy = action_entropy.mean(axis=-1)
        action_top4 = action_top4.mean(axis=-1)
        column = 0
        for family_entropy, family_top4 in (
            (state_entropy, state_top4),
            (action_entropy, action_top4),
        ):
            for _, layer_axes in groups:
                for denoise in range(n_denoise):
                    values[start:stop, column] = family_entropy[
                        :, layer_axes, denoise
                    ].mean(axis=1)
                    values[start:stop, column + 1] = family_top4[
                        :, layer_axes, denoise
                    ].mean(axis=1)
                    column += 2
        if start % (BLOCK * 16) == 0:
            print(f"  {stop}/{n_row}", flush=True)

    np.savez_compressed(
        out_path,
        values=values,
        columns=np.asarray(columns),
        episode_id=store["episode_id"][:],
        control_step=store["control_step"][:],
    )
    print(f"wrote {out_path}  shape={values.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
