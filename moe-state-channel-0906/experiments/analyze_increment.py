#!/usr/bin/env python3
"""Post-hoc follow-up, run after `run_detectors.py` opened external once.

Everything here is labelled post-hoc in the report.  Six blocks:

1. survival-conditioned AUC on both cohorts for the decomposition quantities.
   This is what decides which component of the token configuration actually
   shrinks near failure, and therefore settles the reported
   "lower conditional_energy but larger centred configuration" contradiction.
2. slope versus level: the identical protocol applied to the *endpoint* levels
   `token_differentiation_s9`, `action_consensus_s9`, `centred_energy_s9`, so
   the formation slope is compared with the thing it is nearly a monotone
   transform of, instead of with nothing.
3. redundancy: Spearman between the formation slope and every published
   neighbour, on a fixed subsample of both cohorts.
4. prefix-causal state / action mobility coupling at K = 12 chunks, the check
   the controller asked for.  Whole-episode coupling is computed alongside so
   the two can be compared directly.
5. `libero_object` broken out.
6. physical failure modes of what each head catches (analysis only, never a
   score input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_formation import survival_auc, auc  # noqa: E402
from sweep_lib import COHORTS, Cohort, D, P, QUANTITIES, PROJECT  # noqa: E402
from arc_lib import LAYER_NAMES  # noqa: E402

DEFAULT_OUTPUT = HERE.parent / "results/increment"
PHYSICAL = PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"
RUN_ID = {"development_main": "right-50x8-20260903", "external_8b": "right-50x8b-20260903"}
COUPLING_K = 12
SUBSAMPLE = 150_000
RNG_SEED = 20260906

DECOMPOSITION = (
    "conditional_energy_s9",        # = 1 - mean_t(c_t^2)              : pure state channel
    "state_action_alignment_s9",    # = mean_t(c_t)
    "along_state_energy_s9",        # = var_t(c_t)                     : discarded by the Schur complement
    "centred_energy_s9",            # = 0.9 (1 - action_consensus)     : pure action consensus
    "action_consensus_s9",
    "token_differentiation_s9",
    "conditional_effective_rank_s9",
)
LEVEL_HEADS = ("token_differentiation_s9", "action_consensus_s9", "centred_energy_s9")
NEIGHBOURS = (
    "token_differentiation_s9", "action_consensus_s9", "token_differentiation_s0",
    "mobility_s9", "state_mobility_s9", "flow_speed_s9", "conditional_effective_rank_s9",
    "pc1corr_slope", "stretch_slope", "pc1share_slope", "saa_slope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def context_of(cohort: Cohort, name: str) -> dict[str, Any]:
    labels = pd.read_csv(P.LABEL_PATHS[name])
    risk = labels["original_failure"].to_numpy(bool)
    suite = P.suite_of(cohort.task_names, cohort.task_index)
    priors = P.survival_prior(suite, cohort.length, risk)
    return {
        "risk": risk,
        "suite": suite,
        "priors": priors,
        "early_cut": {
            s: next((q for q, p in sorted(t.items()) if p >= P.LOW_PRIOR), None)
            for s, t in priors.items()
        },
    }


def quantity_of(cohort: Cohort, name: str) -> np.ndarray:
    if name == "centred_energy_s9":
        return cohort.arc_plane("centred_energy")[..., 9]
    if name == "orthogonal_energy_s9":
        return (
            cohort.arc_plane("centred_energy")[..., 9]
            - cohort.arc_plane("along_state_energy")[..., 9]
        )
    return cohort.quantity(name)


# ------------------------------------------------------------------- block 1+3
def information_and_redundancy(
    cohorts: dict[str, Cohort], contexts: dict[str, dict], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = sorted(set(DECOMPOSITION) | set(QUANTITIES) | set(NEIGHBOURS) | {"orthogonal_energy_s9"})
    info_rows, redundancy_rows = [], []
    for cohort_name in ("development_main", "external_8b"):
        cohort = cohorts[cohort_name]
        context = contexts[cohort_name]
        values = {name: quantity_of(cohort, name) for name in names}
        for name, block in values.items():
            for position, layer in enumerate(LAYER_NAMES):
                dense = np.where(cohort.valid, block[:, :, position], np.nan)
                overall, early = survival_auc(
                    dense, cohort.valid, context["suite"], context["risk"], context["early_cut"]
                )
                info_rows.append(
                    {
                        "cohort": cohort_name, "quantity": name, "layer": layer,
                        "survival_auc": overall, "survival_auc_early": early,
                        "abs_auc_gap": abs(overall - 0.5),
                    }
                )
        flat = np.flatnonzero(cohort.valid.ravel())
        take = rng.choice(flat, min(SUBSAMPLE, len(flat)), replace=False)
        for layer in ("L2", "L12", "L14", "L15"):
            position = LAYER_NAMES.index(layer)
            columns = {n: np.asarray(v[:, :, position]).ravel()[take] for n, v in values.items()}
            good = np.all([np.isfinite(c) for c in columns.values()], axis=0)
            reference = columns["td_slope"][good]
            for name in NEIGHBOURS:
                redundancy_rows.append(
                    {
                        "cohort": cohort_name, "layer": layer, "reference": "td_slope",
                        "other": name, "n": int(good.sum()),
                        "spearman": float(stats.spearmanr(reference, columns[name][good]).statistic),
                    }
                )
        cohort.release()
    return pd.DataFrame(info_rows), pd.DataFrame(redundancy_rows)


# --------------------------------------------------------------------- block 2
def level_versus_slope(
    cohorts: dict[str, Cohort], contexts: dict[str, dict]
) -> pd.DataFrame:
    main, extra, external = (
        cohorts["development_main"], cohorts["development_extra"], cohorts["external_8b"]
    )
    dev, ext = contexts["development_main"], contexts["external_8b"]
    rows: list[dict[str, Any]] = []
    for quantity in LEVEL_HEADS:
        reprs = {
            name: P.representations(quantity_of(cohorts[name], quantity), cohorts[name].valid)
            for name in COHORTS
        }
        grid = pd.DataFrame(
            D.sweep(main, extra, reprs["development_main"], reprs["development_extra"],
                    dev["risk"], dev["suite"], dev["priors"])
        )
        for mode in P.MODES:
            eligible = grid[
                (grid["mode"] == mode)
                & (grid["timely_fpr"] <= P.MAX_TIMELY_FPR)
                & (grid["low_prior_precision"] >= P.MIN_LOW_PRIOR_PRECISION)
            ]
            if eligible.empty:
                rows.append({"quantity": quantity, "mode": mode, "feasible": False})
                continue
            best = eligible.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]
            first = D.external_alarm(
                external, [main, extra], reprs, str(best["representation"]),
                str(best["direction"]), float(best["quantile"]), mode,
            )
            rows.append(
                {
                    "quantity": quantity, "mode": mode, "feasible": True,
                    "representation": str(best["representation"]),
                    "direction": str(best["direction"]), "quantile": float(best["quantile"]),
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    **P.score_candidate(
                        first, ext["risk"], P.prior_of(first, ext["suite"], ext["priors"])
                    ),
                }
            )
        for cohort in cohorts.values():
            cohort.release()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- block 4
def coupling(cohorts: dict[str, Cohort], contexts: dict[str, dict]) -> pd.DataFrame:
    """Per-episode correlation between state and action mobility across queries."""
    rows: list[dict[str, Any]] = []
    for cohort_name in ("development_main", "external_8b"):
        cohort = cohorts[cohort_name]
        context = contexts[cohort_name]
        action = cohort.mobility_plane()[..., 9]
        state = cohort.state_mobility_plane()[..., 9]
        risk, suite, length = context["risk"], context["suite"], cohort.length
        for position, layer in enumerate(LAYER_NAMES):
            a, s = action[:, :, position], state[:, :, position]
            for window, label in ((COUPLING_K, f"prefix_K{COUPLING_K}"), (None, "whole_episode")):
                limit = a.shape[1] if window is None else window
                block_a, block_s = a[:, :limit], s[:, :limit]
                good = np.isfinite(block_a) & np.isfinite(block_s) & cohort.valid[:, :limit]
                count = good.sum(1)
                usable = count >= 6
                value = np.full(len(a), np.nan)
                for row in np.flatnonzero(usable):
                    mask = good[row]
                    value[row] = stats.spearmanr(block_a[row, mask], block_s[row, mask]).statistic
                for suite_name in np.unique(suite):
                    if window is None:
                        take = usable & (suite == suite_name) & np.isfinite(value)
                    else:
                        take = usable & (suite == suite_name) & np.isfinite(value) & (length > window)
                    positive = risk[take]
                    if positive.sum() < 10 or (~positive).sum() < 10:
                        continue
                    rows.append(
                        {
                            "cohort": cohort_name, "layer": layer, "window": label,
                            "suite": str(suite_name), "n": int(take.sum()),
                            "n_risk": int(positive.sum()),
                            "mean_coupling_risk": float(value[take][positive].mean()),
                            "mean_coupling_ok": float(value[take][~positive].mean()),
                            "delta": float(
                                value[take][positive].mean() - value[take][~positive].mean()
                            ),
                            "auc": auc(value[take], positive),
                        }
                    )
        cohort.release()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- block 6
def failure_modes(external: Cohort, context: dict, alarms: dict[str, np.ndarray]) -> pd.DataFrame:
    physical = pd.read_csv(PHYSICAL)
    physical = physical[physical["run_id"] == RUN_ID["external_8b"]]
    key = pd.DataFrame(
        {
            "suite": [t.split("/", 1)[0] for t in external.task],
            "task_name": [t.split("/", 1)[1] for t in external.task],
            "episode_index": external.episode,
        }
    )
    merged = key.merge(
        physical[["suite", "task_name", "episode_index", "primary_failure_reason"]],
        on=["suite", "task_name", "episode_index"], how="left", validate="one_to_one",
    )
    reason = merged["primary_failure_reason"].fillna("(no_failure)").to_numpy(str)
    risk = context["risk"]
    rows: list[dict[str, Any]] = []
    for name, first in alarms.items():
        fired = first >= 0
        for value in np.unique(reason[risk]):
            take = risk & (reason == value)
            rows.append(
                {
                    "head": name, "primary_failure_reason": value,
                    "n_risk": int(take.sum()), "caught": int((take & fired).sum()),
                    "recall": float((take & fired).sum() / max(take.sum(), 1)),
                }
            )
    coverage = {
        "risk_episodes": int(risk.sum()),
        "risk_episodes_with_a_physical_reason": int((risk & (reason != "(no_failure)")).sum()),
    }
    frame = pd.DataFrame(rows)
    frame.attrs["coverage"] = coverage
    return frame


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    cohorts = {name: Cohort(name) for name in COHORTS}
    contexts = {
        name: context_of(cohorts[name], name)
        for name in ("development_main", "external_8b")
    }

    print("block 1+3: survival AUC and redundancy", flush=True)
    information, redundancy = information_and_redundancy(cohorts, contexts, rng)
    information.to_csv(args.output / "survival_auc.csv", index=False)
    redundancy.to_csv(args.output / "redundancy.csv", index=False)

    print("block 2: level versus slope under the identical protocol", flush=True)
    levels = level_versus_slope(cohorts, contexts)
    levels.to_csv(args.output / "level_versus_slope.csv", index=False)

    print("block 4: prefix-causal coupling", flush=True)
    couple = coupling(cohorts, contexts)
    couple.to_csv(args.output / "coupling.csv", index=False)

    print("block 5+6: suites and physical failure modes", flush=True)
    with np.load(
        HERE.parent / "results/detectors/external_first_alarms.npz", allow_pickle=False
    ) as archive:
        alarms = {
            k: np.asarray(archive[k])
            for k in archive.files
            if k not in ("schema", "risk", "suite", "length", "task", "episode")
        }
    modes = failure_modes(cohorts["external_8b"], contexts["external_8b"], alarms)
    modes.to_csv(args.output / "failure_modes.csv", index=False)

    (args.output / "increment_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.state_channel.increment.v1",
                "status": "post-hoc; external outcomes were already opened by run_detectors.py",
                "coupling_prefix_chunks": COUPLING_K,
                "subsample": SUBSAMPLE,
                "rng_seed": RNG_SEED,
                "physical_label_coverage": modes.attrs["coverage"],
            },
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 240)
    print("\n=== external survival AUC, decomposition ===")
    block = information[
        (information["cohort"] == "external_8b")
        & (information["quantity"].isin(list(DECOMPOSITION) + ["orthogonal_energy_s9"]))
    ]
    print(block.pivot(index="quantity", columns="layer", values="survival_auc")[
        list(LAYER_NAMES)
    ].to_string(float_format="%.4f"))
    print("\n=== level versus slope on external ===")
    print(levels.to_string(index=False, float_format="%.3f"))
    print("\n=== prefix-causal coupling, external ===")
    print(
        couple[(couple["cohort"] == "external_8b") & (couple["layer"].isin(["L2", "L12", "L15"]))]
        .to_string(index=False, float_format="%.3f")
    )


if __name__ == "__main__":
    main()
