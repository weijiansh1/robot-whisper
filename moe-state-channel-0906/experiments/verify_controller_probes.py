#!/usr/bin/env python3
"""Independent check of the four controller probes this study was handed.

Nothing downstream depends on the controller's numbers; this script exists so
that the report can say which of them replicate and which do not, with the
disagreements stated rather than smoothed over.

  probe 1  the state token's route is step-invariant within a query but highly
           variable across queries, 1.6x-4.9x the action mobility.
  probe 2  state-action alignment is flat across the ten denoising steps while
           token differentiation rises in the back layers and falls in L2.
  probe 3  at the final step the ten action tokens form a one-dimensional
           ordered arc: PC1 correlates with the token index at 0.958-0.998 and
           the normalised-Laplacian Fiedler vector has median |Spearman| 1.0000
           in all eight layers.
  probe 4  is checked in `analyze_increment.py` (prefix-causal coupling).

Probe 3 is checked against four Fiedler constructions, because the published
circuit work builds its Laplacian from the Schur complement while the natural
reading of "the token graph" is the raw Gram; if the claim only holds for one
of them, that is worth knowing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arc_lib import (  # noqa: E402
    LAYER_NAMES, N_TOKENS, _abs_corr_with_index, _abs_spearman_with_index, normalize,
)
from sweep_lib import Cohort, PROJECT, load_npz  # noqa: E402

DEFAULT_OUTPUT = HERE.parent / "results/controller_probes"
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
CACHE = PROJECT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz"
RUN_ID = "right-50x8b-20260903"
ROWS_PER_TASK = 400
RNG_SEED = 20260906


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", type=int, default=8)
    return parser.parse_args()


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(1.0 - (left * right).sum(-1), 0.0, None))


def fiedler_variants(root: np.ndarray) -> dict[str, np.ndarray]:
    """|Spearman(Fiedler vector, token index)| under four graph constructions."""
    flat = root.reshape(-1, root.shape[-2], root.shape[-1]).astype(np.float64)
    action, state = flat[:, 1:, :], flat[:, 0, :]
    gram = action @ np.swapaxes(action, -1, -2)
    c = np.einsum("ne,nte->nt", state, action, optimize=True)
    conditional = gram - c[:, :, None] * c[:, None, :]
    eye = np.eye(N_TOKENS)
    idx = np.arange(N_TOKENS)
    out: dict[str, np.ndarray] = {}
    for label, matrix in (("gram", gram), ("schur", conditional)):
        weight = np.clip(matrix, 0.0, None).copy()
        weight[:, idx, idx] = 0.0
        degree = weight.sum(-1)
        # normalised Laplacian
        inv = np.where(degree > 0, 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0)), 0.0)
        lnorm = eye - weight * inv[:, :, None] * inv[:, None, :]
        out[f"{label}_normalised"] = _abs_spearman_with_index(
            np.linalg.eigh(lnorm)[1][:, :, 1]
        )
        # combinatorial Laplacian
        lap = degree[:, :, None] * eye - weight
        out[f"{label}_combinatorial"] = _abs_spearman_with_index(
            np.linalg.eigh(lap)[1][:, :, 1]
        )
    # PC1 of the centred Gram, for comparison
    h = eye - np.full((N_TOKENS, N_TOKENS), 1.0 / N_TOKENS)
    centred = h @ gram @ h
    v1 = np.linalg.eigh(centred)[1][:, :, -1]
    out["pc1_index_pearson"] = _abs_corr_with_index(v1)
    out["pc1_index_spearman"] = _abs_spearman_with_index(v1)
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # ---------------------------------------------------- probe 1, cross-query
    external = Cohort("external_8b", need_arc=False)
    action = external.mobility_plane()[..., 9]
    state = external.state_mobility_plane()[..., 9]
    finite = np.isfinite(action[:, :, 0]) & np.isfinite(state[:, :, 0])
    cross_rows = []
    for position, layer in enumerate(LAYER_NAMES):
        a = action[:, :, position][finite]
        s = state[:, :, position][finite]
        cross_rows.append(
            {
                "layer": layer,
                "n": int(finite.sum()),
                "median_action_mobility": float(np.median(a)),
                "median_state_mobility": float(np.median(s)),
                "ratio_state_over_action": float(np.median(s) / np.median(a)),
                "median_of_per_query_ratio": float(np.median(s / np.maximum(a, 1e-12))),
            }
        )
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(args.output / "probe1_cross_query.csv", index=False)

    # step-invariance of the state route needs the raw tensor
    cache = load_npz(CACHE)
    tasks = cache["task_names"].astype(str)
    picked = rng.choice(len(tasks), size=min(args.tasks, len(tasks)), replace=False)
    within_rows = []
    fiedler_blocks: list[dict[str, np.ndarray]] = []
    for position in picked:
        task = tasks[position]
        group = zarr.open_group(
            str(ROUTE_ROOT / task / RUN_ID / "server/routes.zarr"), mode="r"
        )
        source = group["hb_router_probs"]
        rows = min(ROWS_PER_TASK, int(source.shape[0]))
        offset = int(rng.integers(0, max(int(source.shape[0]) - rows, 1)))
        root = np.sqrt(normalize(np.asarray(source[offset : offset + rows])))
        state_route = root[:, :, :, 0, :]
        action_route = root[:, :, :, 1:, :]
        for layer_position, layer in enumerate(LAYER_NAMES):
            s = state_route[:, layer_position]
            a = action_route[:, layer_position]
            within_rows.append(
                {
                    "task": task, "layer": layer, "n_queries": rows,
                    "state_step0_to_step9": float(np.median(hellinger(s[:, 0], s[:, 9]))),
                    "state_max_pairwise_step": float(
                        np.median(
                            np.max(
                                [hellinger(s[:, i], s[:, j])
                                 for i in range(10) for j in range(i + 1, 10)],
                                axis=0,
                            )
                        )
                    ),
                    "state_consecutive_mean": float(
                        np.median(hellinger(s[:, 1:], s[:, :-1]).mean(-1))
                    ),
                    "action_step0_to_step9": float(
                        np.median(hellinger(a[:, 0], a[:, 9]).mean(-1))
                    ),
                    "action_consecutive_mean": float(
                        np.median(hellinger(a[:, 1:], a[:, :-1]).mean((-1, -2)))
                    ),
                }
            )
        # the arc claim is about the FINAL denoising step, so step 9 only
        fiedler_blocks.append(
            {k: v.reshape(rows, len(LAYER_NAMES))
             for k, v in fiedler_variants(root[:, :, 9]).items()}
        )
    within = pd.DataFrame(within_rows)
    within.to_csv(args.output / "probe1_within_query.csv", index=False)

    # ------------------------------------------------------------- probe 3
    merged = {
        key: np.concatenate([block[key] for block in fiedler_blocks])
        for key in fiedler_blocks[0]
    }
    arc_rows = []
    for key, values in merged.items():
        for layer_position, layer in enumerate(LAYER_NAMES):
            column = values[:, layer_position]
            arc_rows.append(
                {
                    "statistic": key, "layer": layer, "n": int(column.size),
                    "median": float(np.median(column)),
                    "mean": float(column.mean()),
                    "q10": float(np.quantile(column, 0.10)),
                    "share_equal_1": float((column > 0.99999).mean()),
                }
            )
    arc = pd.DataFrame(arc_rows)
    arc.to_csv(args.output / "probe3_arc_statistics.csv", index=False)

    # ------------------------------------------------------------- probe 2
    profile = pd.read_csv(HERE.parent / "results/formation/step_profile.csv")
    probe2 = profile[
        profile["quantity"].isin(["state_action_alignment", "token_differentiation"])
        & profile["step"].isin([0, 9])
    ].pivot_table(
        index=["cohort", "quantity", "layer"], columns="step", values="mean"
    ).reset_index()
    probe2["relative_change"] = probe2[9] / probe2[0] - 1.0
    probe2.to_csv(args.output / "probe2_step_change.csv", index=False)

    summary = {
        "schema": "himoe.state_channel.controller_probes.v1",
        "probe1_cross_query_ratio_range": [
            float(cross["ratio_state_over_action"].min()),
            float(cross["ratio_state_over_action"].max()),
        ],
        "probe1_claimed_ratio_range": [1.6, 4.9],
        "probe1_state_step0_to_step9_range": [
            float(within["state_step0_to_step9"].min()),
            float(within["state_step0_to_step9"].max()),
        ],
        "probe1_claimed_state_step_drift": [2e-5, 4e-4],
        "probe1_action_step0_to_step9_range": [
            float(within["action_step0_to_step9"].min()),
            float(within["action_step0_to_step9"].max()),
        ],
        "probe3_claimed": {
            "pc1_index_corr": [0.958, 0.998],
            "fiedler_index_abs_spearman_median": 1.0000,
        },
        "probe3_observed_median_by_statistic": {
            key: {
                row["layer"]: row["median"]
                for _, row in arc[arc["statistic"] == key].iterrows()
            }
            for key in merged
        },
        "rng_seed": RNG_SEED,
        "raw_sampled": {
            "cohort": "external_8b", "tasks": int(len(picked)),
            "rows_per_task": ROWS_PER_TASK,
            "task_names": [str(tasks[p]) for p in picked],
        },
    }
    (args.output / "probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 240)
    print("=== probe 1, cross-query mobility (external_8b, all valid queries) ===")
    print(cross.to_string(index=False, float_format="%.4f"))
    print("\n=== probe 1, within-query step drift (raw sample) ===")
    print(
        within.groupby("layer", sort=False)[
            ["state_step0_to_step9", "state_max_pairwise_step", "state_consecutive_mean",
             "action_step0_to_step9", "action_consecutive_mean"]
        ].median().reindex(list(LAYER_NAMES)).to_string(float_format="%.6f")
    )
    print("\n=== probe 3, arc statistics at the final step (raw sample) ===")
    print(
        arc.pivot(index="statistic", columns="layer", values="median")[list(LAYER_NAMES)]
        .to_string(float_format="%.4f")
    )
    print("\nshare of queries with the statistic exactly 1:")
    print(
        arc.pivot(index="statistic", columns="layer", values="share_equal_1")[list(LAYER_NAMES)]
        .to_string(float_format="%.4f")
    )
    print("\n=== probe 2, mean change from step 0 to step 9 ===")
    print(probe2.to_string(index=False, float_format="%.5f"))


if __name__ == "__main__":
    main()
