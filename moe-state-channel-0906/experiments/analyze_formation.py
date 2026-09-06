#!/usr/bin/env python3
"""Question A, descriptive: how the action-token arc forms across the ten steps.

Three blocks, none of which uses a threshold:

A.0  the algebraic decomposition, verified against raw Zarr.  The published
     ``conditional_energy`` turns out to be an exact function of the state
     alignment vector alone, and the centred configuration size an exact
     function of ``action_consensus`` alone, which settles the reported
     "lower conditional energy but larger centred configuration" contradiction
     without fitting anything.

A.1  the step profile of the twelve arc quantities, per layer, on both cohorts.
     Includes a permutation null for the arc-ordering statistics, so "at which
     denoising step does the ordering become detectable" has a fixed answer.

A.2  threshold-free information: survival-conditioned AUC of the declared A
     quantities on development_main.  This is selection information and is
     computed on development only; external is opened once, in
     ``run_detectors.py``.

`survival_auc`/`auc` are copied verbatim from
``moe-circuit-analogy-0906/experiments/summarise_results.py`` and validated
against its published table before use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import zarr  # noqa: E402

from arc_lib import ARC_NAMES, LAYER_NAMES, arc_features, normalize  # noqa: E402
from sweep_lib import Cohort, PROJECT, load_npz  # noqa: E402
import protocol as P  # noqa: E402

DEFAULT_OUTPUT = HERE.parent / "results/formation"
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
MIN_STRATUM = 20
FRONT, BACK = slice(0, 4), slice(4, 8)
RNG_SEED = 20260906
# Published survival AUCs used as an implementation check, from
# moe-circuit-analogy-0906/results/detectors/survival_conditioned_auc.csv
SURVIVAL_AUC_CHECKS = {
    ("mobility", "development_main", "L12"): 0.429194,
    ("mobility", "external_8b", "L12"): 0.385871,
    ("conditional_energy", "external_8b", "L3"): 0.451608,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--identity-rows", type=int, default=512)
    return parser.parse_args()


# --------------------------------------------------------------- survival AUC
def auc(values: np.ndarray, positive: np.ndarray) -> float:
    ranks = stats.rankdata(values)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def survival_auc(
    dense: np.ndarray,
    valid: np.ndarray,
    suite: np.ndarray,
    risk: np.ndarray,
    early_cut: dict[str, int],
) -> tuple[float, float]:
    total, weight = 0.0, 0.0
    early_total, early_weight = 0.0, 0.0
    for name in np.unique(suite):
        take = suite == name
        cut = early_cut.get(str(name))
        for query in range(valid.shape[1]):
            running = take & valid[:, query]
            if not running.any():
                continue
            positive = risk[running]
            n_pos = int(positive.sum())
            n_neg = int(len(positive) - n_pos)
            if n_pos < MIN_STRATUM or n_neg < MIN_STRATUM:
                continue
            column = dense[running, query]
            if not np.isfinite(column).all():
                continue
            value = auc(column, positive)
            here = n_pos * n_neg
            total += value * here
            weight += here
            if cut is not None and query < cut:
                early_total += value * here
                early_weight += here
    return (
        total / weight if weight else float("nan"),
        early_total / early_weight if early_weight else float("nan"),
    )


# ------------------------------------------------------------------ A.0 checks
def verify_identities(rows: int) -> dict:
    """Recompute the three algebraic identities directly from raw Zarr."""
    cache = load_npz(PROJECT / "moe-v4-0904/results/layerwise_mobility/main_reference.npz")
    tasks = cache["task_names"].astype(str)
    run_id = str(cache["run_id"])
    picked = [tasks[0], tasks[len(tasks) // 2], tasks[-1]]
    worst = {"conditional_energy": 0.0, "centred_energy": 0.0, "along_state_energy": 0.0}
    fractions = []
    centroid_cos = []
    for task in picked:
        group = zarr.open_group(
            str(ROUTE_ROOT / task / run_id / "server/routes.zarr"), mode="r"
        )
        raw = np.asarray(group["hb_router_probs"][:rows])
        root = np.sqrt(normalize(raw)).astype(np.float64)
        action, state = root[:, :, :, 1:, :], root[:, :, :, 0, :]
        gram = action @ np.swapaxes(action, -1, -2)
        c = np.einsum("blse,blste->blst", state, action, optimize=True)
        upper = np.triu_indices(10, 1)
        consensus = gram[..., upper[0], upper[1]].mean(-1)
        conditional_energy = np.clip(
            np.diagonal(gram - c[..., :, None] * c[..., None, :], axis1=-2, axis2=-1), 0, None
        ).mean(-1)
        features = arc_features(root).astype(np.float64)
        names = list(ARC_NAMES)
        worst["conditional_energy"] = max(
            worst["conditional_energy"],
            float(np.abs(conditional_energy - (1.0 - (c ** 2).mean(-1))).max()),
        )
        worst["centred_energy"] = max(
            worst["centred_energy"],
            float(
                np.abs(
                    features[..., names.index("centred_energy")] - 0.9 * (1.0 - consensus)
                ).max()
            ),
        )
        worst["along_state_energy"] = max(
            worst["along_state_energy"],
            float(
                np.abs(
                    features[..., names.index("along_state_energy")]
                    - (1.0 - conditional_energy - c.mean(-1) ** 2)
                ).max()
            ),
        )
        along = features[..., names.index("along_state_energy")]
        centred = features[..., names.index("centred_energy")]
        fractions.append(along / np.maximum(centred, 1e-15))
        # cosine between the action centroid direction and the state token
        centroid_cos.append(c.mean(-1) / np.sqrt(0.1 + 0.9 * consensus))
    fraction = np.concatenate([f.reshape(-1, 8, 10) for f in fractions])
    cosine = np.concatenate([f.reshape(-1, 8, 10) for f in centroid_cos])
    return {
        "tasks_sampled": picked,
        "rows_per_task": rows,
        "max_abs_error": worst,
        "identities": {
            "conditional_energy": "1 - mean_t(c_t^2)",
            "centred_energy": "0.9 * (1 - action_consensus)",
            "along_state_energy": "var_t(c_t) = 1 - conditional_energy - state_action_alignment^2",
        },
        "along_state_fraction_median_step9": {
            layer: float(np.median(fraction[:, i, 9])) for i, layer in enumerate(LAYER_NAMES)
        },
        "centroid_state_cosine_median_step9": {
            layer: float(np.median(cosine[:, i, 9])) for i, layer in enumerate(LAYER_NAMES)
        },
        "centroid_state_angle_deg_step9": {
            layer: float(np.degrees(np.arccos(np.clip(np.median(cosine[:, i, 9]), -1, 1))))
            for i, layer in enumerate(LAYER_NAMES)
        },
    }


# ------------------------------------------------------------- A.1 step profile
def permutation_null(cohort: Cohort, rng: np.random.Generator, draws: int = 200_000) -> dict:
    """Null distribution of the ordering statistics when the token order is random.

    |Pearson(v, index)| and |Spearman(v, index)| for a length-10 vector v whose
    coordinates carry no information about the token index.  The null does not
    depend on the data, so it is generated directly.
    """
    values = rng.standard_normal((draws, 10))
    index = np.arange(10, dtype=np.float64)
    centred = values - values.mean(1, keepdims=True)
    index_c = index - index.mean()
    pearson = np.abs(
        (centred * index_c).sum(1)
        / np.maximum(np.sqrt((centred ** 2).sum(1)) * np.sqrt((index_c ** 2).sum()), 1e-12)
    )
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(10), order.shape), axis=1)
    spearman = np.abs(1.0 - 6.0 * ((ranks - index) ** 2).sum(1) / (10 * 99))
    return {
        "draws": draws,
        "abs_pearson_median": float(np.median(pearson)),
        "abs_pearson_p95": float(np.quantile(pearson, 0.95)),
        "abs_pearson_p99": float(np.quantile(pearson, 0.99)),
        "abs_spearman_median": float(np.median(spearman)),
        "abs_spearman_p95": float(np.quantile(spearman, 0.95)),
        "abs_spearman_p99": float(np.quantile(spearman, 0.99)),
    }


def step_profile(cohort: Cohort) -> pd.DataFrame:
    valid = cohort.valid
    rows = []
    planes: dict[str, np.ndarray] = {name: cohort.arc_plane(name) for name in ARC_NAMES}
    for metric in (
        "token_differentiation",
        "state_action_alignment",
        "action_consensus",
        "conditional_energy",
        "conditional_effective_rank",
    ):
        planes[metric] = cohort.metric_plane(metric)
    planes["along_state_fraction"] = planes["along_state_energy"] / np.maximum(
        planes["centred_energy"], 1e-15
    )
    planes["orthogonal_energy"] = planes["centred_energy"] - planes["along_state_energy"]
    for name, plane in planes.items():
        for layer_position, layer in enumerate(LAYER_NAMES):
            for step in range(10):
                column = plane[valid, layer_position, step]
                column = column[np.isfinite(column)]
                if column.size == 0:
                    continue
                q = np.quantile(column, (0.25, 0.5, 0.75))
                rows.append(
                    {
                        "cohort": cohort.name,
                        "quantity": name,
                        "layer": layer,
                        "block": "front" if layer_position < 4 else "back",
                        "step": step,
                        "n": int(column.size),
                        "mean": float(column.mean()),
                        "q25": float(q[0]),
                        "median": float(q[1]),
                        "q75": float(q[2]),
                    }
                )
    return pd.DataFrame(rows)


def emergence_table(profile: pd.DataFrame, null: dict) -> pd.DataFrame:
    rows = []
    thresholds = {
        "pc1_index_corr": null["abs_pearson_p95"],
        "fiedler_index_rho": null["abs_spearman_p95"],
    }
    for (cohort, quantity, layer), block in profile[
        profile["quantity"].isin(thresholds)
    ].groupby(["cohort", "quantity", "layer"], sort=False):
        block = block.sort_values("step")
        line = thresholds[quantity]
        above = block[block["median"] > line]["step"].to_numpy()
        rows.append(
            {
                "cohort": cohort,
                "quantity": quantity,
                "layer": layer,
                "null_p95": line,
                "median_step0": float(block[block["step"] == 0]["median"].iloc[0]),
                "median_step9": float(block[block["step"] == 9]["median"].iloc[0]),
                "delta": float(
                    block[block["step"] == 9]["median"].iloc[0]
                    - block[block["step"] == 0]["median"].iloc[0]
                ),
                "first_step_above_null": int(above[0]) if above.size else -1,
                "above_null_at_step0": bool(
                    block[block["step"] == 0]["median"].iloc[0] > line
                ),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- A.2 information
def development_information(cohort: Cohort, quantities: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(P.LABEL_PATHS["development_main"])
    if not np.array_equal(labels["task"].to_numpy(str), cohort.task):
        raise ValueError("labels not row aligned with the cohort")
    risk = labels["original_failure"].to_numpy(bool)
    suite = P.suite_of(cohort.task_names, cohort.task_index)
    priors = P.survival_prior(suite, cohort.length, risk)
    early_cut = {
        s: next((q for q, p in sorted(t.items()) if p >= P.LOW_PRIOR), None)
        for s, t in priors.items()
    }
    rows = []
    for name in quantities:
        values = cohort.quantity(name)
        for position, layer in enumerate(LAYER_NAMES):
            dense = np.where(cohort.valid, values[:, :, position], np.nan)
            overall, early = survival_auc(dense, cohort.valid, suite, risk, early_cut)
            rows.append(
                {
                    "quantity": name,
                    "layer": layer,
                    "survival_auc": overall,
                    "survival_auc_early": early,
                    "abs_auc_gap": abs(overall - 0.5),
                }
            )
    return pd.DataFrame(rows)


def check_survival_auc_implementation() -> dict:
    """Reproduce three published survival AUCs before any new one is believed."""
    observed = {}
    for (quantity, cohort_name, layer), expected in SURVIVAL_AUC_CHECKS.items():
        cohort = Cohort(cohort_name, need_arc=False)
        labels = pd.read_csv(P.LABEL_PATHS[cohort_name])
        risk = labels["original_failure"].to_numpy(bool)
        suite = P.suite_of(cohort.task_names, cohort.task_index)
        priors = P.survival_prior(suite, cohort.length, risk)
        early_cut = {
            s: next((q for q, p in sorted(t.items()) if p >= P.LOW_PRIOR), None)
            for s, t in priors.items()
        }
        position = LAYER_NAMES.index(layer)
        plane = (
            cohort.mobility_plane()[..., 9]
            if quantity == "mobility"
            else cohort.metric_plane(quantity)[..., 9]
        )
        dense = np.where(cohort.valid, plane[:, :, position], np.nan)
        value = survival_auc(dense, cohort.valid, suite, risk, early_cut)[0]
        observed[f"{quantity}|{cohort_name}|{layer}"] = {
            "expected": expected,
            "observed": float(value),
            "abs_error": abs(float(value) - expected),
        }
        cohort.release()
    worst = max(v["abs_error"] for v in observed.values())
    if worst > 2e-4:
        raise RuntimeError(f"survival AUC re-implementation disagrees by {worst}")
    return observed


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("A.0  algebraic identities against raw Zarr", flush=True)
    identities = verify_identities(args.identity_rows)
    print(json.dumps(identities["max_abs_error"], indent=2), flush=True)

    print("checking the survival-AUC re-implementation against the published table", flush=True)
    auc_check = check_survival_auc_implementation()
    print(json.dumps(auc_check, indent=2), flush=True)

    print("A.1  step profiles", flush=True)
    null = permutation_null(None, rng)
    profiles = []
    for name in ("development_main", "external_8b"):
        cohort = Cohort(name)
        profiles.append(step_profile(cohort))
        cohort.release()
        print(f"  {name} done", flush=True)
    profile = pd.concat(profiles, ignore_index=True)
    profile.to_csv(args.output / "step_profile.csv", index=False)
    emergence = emergence_table(profile, null)
    emergence.to_csv(args.output / "arc_emergence.csv", index=False)

    print("A.2  development survival-conditioned AUC of the declared quantities", flush=True)
    from sweep_lib import QUANTITIES

    cohort = Cohort("development_main")
    information = development_information(
        cohort, list(QUANTITIES) + ["mobility_s9", "conditional_energy_s9", "action_consensus_s9"]
    )
    cohort.release()
    information.to_csv(args.output / "development_survival_auc.csv", index=False)

    (args.output / "formation_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.state_channel.formation.v1",
                "identity_check": identities,
                "survival_auc_implementation_check": auc_check,
                "permutation_null": null,
                "rng_seed": RNG_SEED,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    print("\n=== arc emergence (median over all valid queries) ===")
    print(emergence.to_string(index=False, float_format="%.4f"))
    print("\n=== development survival AUC, |gap| ranked ===")
    print(
        information.sort_values("abs_auc_gap", ascending=False)
        .head(30)
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
