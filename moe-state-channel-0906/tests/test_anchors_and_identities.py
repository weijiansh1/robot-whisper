#!/usr/bin/env python3
"""Checks that must pass before any number in this bundle is believed.

1. The three published detector anchors reproduce exactly through the frozen
   protocol, as recorded by `experiments/run_detectors.py`.
2. The algebraic identities behind the state/action decomposition hold against
   raw Zarr to float precision.
3. The extracted arc geometry is row-aligned with the frozen v4 caches, and its
   validity mask matches.
4. The published survival priors (horizon caps and the chunk at which the prior
   crosses 0.25) reproduce.

Run: python tests/test_anchors_and_identities.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "experiments"))

from arc_lib import arc_features, normalize  # noqa: E402

RESULTS = BUNDLE / "results"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
PUBLISHED = {
    ("mobility_s9", "global"): ("L12", "low", 0.975, 195, 17),
    ("mobility_s9", "per_task"): ("L2", "low", 0.700, 272, 57),
    ("load_entropy_s9", "per_task"): ("L3", "low", 0.850, 370, 93),
}
HORIZON = {"libero_goal": (30, 18), "libero_long": (52, 26),
           "libero_object": (28, 17), "libero_spatial": (22, 13)}


def test_anchors() -> None:
    external = pd.read_csv(RESULTS / "detectors/external_detectors.csv")
    for (quantity, mode), (rep, direction, quantile, tp, fp) in PUBLISHED.items():
        row = external[(external["quantity"] == quantity) & (external["mode"] == mode)].iloc[0]
        assert row["representation"] == rep, (quantity, mode, row["representation"])
        assert row["direction"] == direction, (quantity, mode, row["direction"])
        assert abs(float(row["quantile"]) - quantile) < 1e-9, row["quantile"]
        assert int(row["tp"]) == tp and int(row["fp"]) == fp, (row["tp"], row["fp"])
        print(f"  {quantity} {mode}: {rep} {direction} q{quantile} -> {tp}/{fp}")
    summary = json.loads((RESULTS / "detectors/summary.json").read_text())
    assert all(entry["matches"] for entry in summary["anchors"].values())
    anchor = summary["anchors"]["mobility_s9|global"]["reproduced"]
    assert abs(anchor["precision"] - 0.9198) < 5e-5, anchor["precision"]
    assert abs(anchor["lift"] - 1.7641) < 5e-5, anchor["lift"]
    load = summary["anchors"]["load_entropy_s9|per_task"]["reproduced"]
    assert abs(load["lift"] - 1.5479) < 5e-5, load["lift"]
    print("  precision 0.9198 and lifts 1.7641 / 1.5479 reproduced")


def test_identities(rows: int = 256) -> None:
    import zarr

    with np.load(V4 / "main_reference.npz", allow_pickle=False) as cache:
        tasks = cache["task_names"].astype(str)
        run_id = str(cache["run_id"])
    worst = 0.0
    for task in (tasks[0], tasks[-1]):
        group = zarr.open_group(str(ROUTE_ROOT / task / run_id / "server/routes.zarr"), mode="r")
        root = np.sqrt(normalize(np.asarray(group["hb_router_probs"][:rows]))).astype(np.float64)
        action, state = root[:, :, :, 1:, :], root[:, :, :, 0, :]
        gram = action @ np.swapaxes(action, -1, -2)
        c = np.einsum("blse,blste->blst", state, action, optimize=True)
        upper = np.triu_indices(10, 1)
        consensus = gram[..., upper[0], upper[1]].mean(-1)
        conditional_energy = np.clip(
            np.diagonal(gram - c[..., :, None] * c[..., None, :], axis1=-2, axis2=-1), 0, None
        ).mean(-1)
        features = arc_features(root).astype(np.float64)
        worst = max(
            worst,
            float(np.abs(conditional_energy - (1.0 - (c ** 2).mean(-1))).max()),
            float(np.abs(features[..., 0] - 0.9 * (1.0 - consensus)).max()),
            float(np.abs(features[..., 1] - (1.0 - conditional_energy - c.mean(-1) ** 2)).max()),
        )
    assert worst < 1e-6, worst
    print(f"  conditional_energy / centred_energy / along_state_energy identities: max |err| {worst:.2e}")


def test_arc_alignment() -> None:
    for cohort, cache_name in (
        ("development_main", "main_reference.npz"),
        ("development_extra", "extra_reference.npz"),
        ("external_8b", "external_8b.npz"),
    ):
        with np.load(V4 / cache_name, allow_pickle=False) as cache:
            valid = cache["valid"].astype(bool)
            episode = cache["episode"].astype(int)
        with np.load(RESULTS / f"arc/{cohort}_arc_index.npz", allow_pickle=False) as index:
            assert np.array_equal(index["episode"].astype(int), episode)
        block = np.load(RESULTS / f"arc/{cohort}_arc.npy", mmap_mode="r")
        observed = np.isfinite(np.asarray(block[:, :, 0, 9, 0]))
        assert np.array_equal(observed, valid), cohort
        print(f"  {cohort}: {int(valid.sum()):,} valid queries, mask matches v4")


def test_priors() -> None:
    audit = json.loads((RESULTS / "detectors/summary.json").read_text())["survival_prior_audit"]
    for suite, (cap, crossing) in HORIZON.items():
        entry = audit["external_8b"][suite]
        assert entry["horizon_cap_observed"] == cap, (suite, entry)
        assert entry["prior_crosses_0.25_at"] == crossing, (suite, entry)
    print("  horizon caps 30/52/28/22 and prior crossings 18/26/17/13 reproduced")


def main() -> int:
    print("published detector anchors")
    test_anchors()
    print("algebraic identities against raw Zarr")
    test_identities()
    print("arc geometry row alignment")
    test_arc_alignment()
    print("survival priors")
    test_priors()
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
