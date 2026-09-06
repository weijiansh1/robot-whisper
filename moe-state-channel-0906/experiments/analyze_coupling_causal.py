#!/usr/bin/env python3
"""Post-hoc: make the state/action coupling a per-query causal score and test it.

`analyze_increment.py` found that the whole-episode coupling and the prefix
coupling have *opposite* signs: over a whole episode risk goes with higher
state/action mobility coupling (survival AUC up to 0.90), but among the
episodes still running at chunk 12 the same statistic is strongly reversed
(AUC 0.23-0.31 in three of four suites).  A whole-episode statistic is not a
detector, so the only way to settle this is to make coupling causal and run it
through the frozen protocol.

  running_coupling      Pearson correlation between the action-token mobility
                        and the state-token mobility over queries 1..q of the
                        same episode, evaluated at every q.  Needs at least three
                        finite pairs (the shortest episode in the cohort is 7
                        chunks long and the frozen protocol needs a complete
                        width-4 trailing mean by chunk 6, so three is the
                        largest value the protocol admits).  Uses only routing
                        tensors and only the past.
  running_coupling_log  the same on log mobility, as a heavy-tail control.

Both are scored with the identical frozen protocol.  Two controls are reported
next to them so the increment is visible rather than asserted:
the survival-conditioned AUC at every chunk against `mobility` and
`state_mobility`, and the same coupling restricted to a *fixed* window so the
"more chunks seen" confound is removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_formation import auc, survival_auc  # noqa: E402
from analyze_increment import context_of  # noqa: E402
from arc_lib import LAYER_NAMES  # noqa: E402
from run_detectors import wilson  # noqa: E402
from sweep_lib import COHORTS, Cohort, D, P  # noqa: E402

DEFAULT_OUTPUT = HERE.parent / "results/coupling"
MIN_POINTS = 3
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def running_pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Expanding-window Pearson along axis 1.  x, y, mask are [n, q, L]."""
    weight = mask.astype(np.float64)
    xs = np.where(mask, x, 0.0).astype(np.float64)
    ys = np.where(mask, y, 0.0).astype(np.float64)
    count = np.cumsum(weight, axis=1)
    sx = np.cumsum(xs, axis=1)
    sy = np.cumsum(ys, axis=1)
    sxx = np.cumsum(xs * xs, axis=1)
    syy = np.cumsum(ys * ys, axis=1)
    sxy = np.cumsum(xs * ys, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(count, 1.0)
        cov = sxy / n - (sx / n) * (sy / n)
        vx = np.maximum(sxx / n - (sx / n) ** 2, 0.0)
        vy = np.maximum(syy / n - (sy / n) ** 2, 0.0)
        r = cov / np.sqrt(np.maximum(vx * vy, 1e-30))
    r = np.clip(r, -1.0, 1.0)
    r[(count < MIN_POINTS) | (vx <= 0) | (vy <= 0)] = np.nan
    return r.astype(np.float32)


def coupling_quantity(cohort: Cohort, name: str) -> np.ndarray:
    action = cohort.mobility_plane()[..., 9]
    state = cohort.state_mobility_plane()[..., 9]
    mask = np.isfinite(action) & np.isfinite(state) & cohort.valid[:, :, None]
    if name == "running_coupling":
        return running_pearson(action, state, mask)
    if name == "running_coupling_log":
        return running_pearson(
            np.log(np.maximum(action, 0.0) + EPSILON),
            np.log(np.maximum(state, 0.0) + EPSILON),
            mask,
        )
    raise KeyError(name)


def fixed_window_coupling(cohort: Cohort, width: int) -> np.ndarray:
    """Coupling over the last `width` queries only, so window length is constant."""
    action = cohort.mobility_plane()[..., 9]
    state = cohort.state_mobility_plane()[..., 9]
    mask = np.isfinite(action) & np.isfinite(state) & cohort.valid[:, :, None]
    out = np.full(action.shape, np.nan, dtype=np.float32)
    for query in range(width - 1, action.shape[1]):
        block = slice(query - width + 1, query + 1)
        a, s, m = action[:, block], state[:, block], mask[:, block]
        complete = m.all(axis=1)
        ac = a - a.mean(axis=1, keepdims=True)
        sc = s - s.mean(axis=1, keepdims=True)
        num = (ac * sc).sum(axis=1)
        den = np.sqrt((ac ** 2).sum(axis=1) * (sc ** 2).sum(axis=1))
        value = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        out[:, query] = np.where(complete, value, np.nan)
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohorts = {name: Cohort(name, need_arc=False) for name in COHORTS}
    contexts = {n: context_of(cohorts[n], n) for n in ("development_main", "external_8b")}
    main_cohort, extra_cohort, external_cohort = (
        cohorts["development_main"], cohorts["development_extra"], cohorts["external_8b"]
    )
    dev, ext = contexts["development_main"], contexts["external_8b"]

    values = {
        quantity: {name: coupling_quantity(cohorts[name], quantity) for name in COHORTS}
        for quantity in ("running_coupling", "running_coupling_log")
    }
    values["fixed4_coupling"] = {
        name: fixed_window_coupling(cohorts[name], 4) for name in COHORTS
    }
    values["mobility_s9"] = {name: cohorts[name].mobility_plane()[..., 9] for name in COHORTS}
    values["state_mobility_s9"] = {
        name: cohorts[name].state_mobility_plane()[..., 9] for name in COHORTS
    }

    # ------------------------------------------------ threshold-free information
    info_rows: list[dict[str, Any]] = []
    for cohort_name in ("development_main", "external_8b"):
        cohort, context = cohorts[cohort_name], contexts[cohort_name]
        for quantity, block in values.items():
            for position, layer in enumerate(LAYER_NAMES):
                dense = np.where(cohort.valid, block[cohort_name][:, :, position], np.nan)
                overall, early = survival_auc(
                    dense, cohort.valid, context["suite"], context["risk"], context["early_cut"]
                )
                info_rows.append(
                    {
                        "cohort": cohort_name, "quantity": quantity, "layer": layer,
                        "survival_auc": overall, "survival_auc_early": early,
                        "abs_auc_gap": abs(overall - 0.5),
                    }
                )
    information = pd.DataFrame(info_rows)
    information.to_csv(args.output / "survival_auc.csv", index=False)

    # ------------------------------------------- per-chunk AUC, to see the flip
    chunk_rows: list[dict[str, Any]] = []
    cohort, context = external_cohort, ext
    for quantity in ("running_coupling", "mobility_s9", "state_mobility_s9"):
        for position, layer in enumerate(("L2", "L12", "L15")):
            index = LAYER_NAMES.index(layer)
            dense = values[quantity]["external_8b"][:, :, index]
            for suite_name in np.unique(context["suite"]):
                take = context["suite"] == suite_name
                for query in range(cohort.valid.shape[1]):
                    running = take & cohort.valid[:, query] & np.isfinite(dense[:, query])
                    positive = context["risk"][running]
                    if positive.sum() < 20 or (~positive).sum() < 20:
                        continue
                    chunk_rows.append(
                        {
                            "quantity": quantity, "layer": layer, "suite": str(suite_name),
                            "chunk": query, "n": int(running.sum()),
                            "n_risk": int(positive.sum()),
                            "auc": auc(dense[running, query], positive),
                        }
                    )
    pd.DataFrame(chunk_rows).to_csv(args.output / "per_chunk_auc.csv", index=False)

    # ---------------------------------------------------- the frozen protocol
    external_rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    for quantity in ("running_coupling", "running_coupling_log", "fixed4_coupling"):
        reprs = {
            name: P.representations(values[quantity][name], cohorts[name].valid)
            for name in COHORTS
        }
        grid = pd.DataFrame(
            D.sweep(main_cohort, extra_cohort, reprs["development_main"],
                    reprs["development_extra"], dev["risk"], dev["suite"], dev["priors"])
        )
        grid["quantity"] = quantity
        grid.to_csv(args.output / f"development_grid_{quantity}.csv", index=False)
        for mode in P.MODES:
            eligible = grid[
                (grid["mode"] == mode)
                & (grid["timely_fpr"] <= P.MAX_TIMELY_FPR)
                & (grid["low_prior_precision"] >= P.MIN_LOW_PRIOR_PRECISION)
            ]
            if eligible.empty:
                external_rows.append({"quantity": quantity, "mode": mode, "feasible": False})
                continue
            best = eligible.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]
            first = D.external_alarm(
                external_cohort, [main_cohort, extra_cohort], reprs,
                str(best["representation"]), str(best["direction"]),
                float(best["quantile"]), mode,
            )
            alarms[f"{quantity}|{mode}"] = first
            prior = P.prior_of(first, ext["suite"], ext["priors"])
            scored = P.score_candidate(first, ext["risk"], prior)
            low, high = wilson(scored["tp"], scored["tp"] + scored["fp"])
            row = {
                "quantity": quantity, "mode": mode, "feasible": True,
                "representation": str(best["representation"]),
                "direction": str(best["direction"]), "quantile": float(best["quantile"]),
                "dev_low_prior_tp": int(best["low_prior_tp"]),
                "dev_low_prior_fp": int(best["low_prior_fp"]),
                **scored,
                "lift_ci_low": low / scored["mean_alarm_prior"] if scored["tp"] + scored["fp"] else np.nan,
                "lift_ci_high": high / scored["mean_alarm_prior"] if scored["tp"] + scored["fp"] else np.nan,
                "median_alarm_chunk": float(np.median(first[first >= 0])) if (first >= 0).any() else np.nan,
            }
            for suite_name in np.unique(ext["suite"]):
                take = ext["suite"] == suite_name
                block = P.score_candidate(first[take], ext["risk"][take], prior[take])
                row[f"{suite_name}__tp"] = block["tp"]
                row[f"{suite_name}__fp"] = block["fp"]
                row[f"{suite_name}__low_prior_tp"] = block["low_prior_tp"]
                row[f"{suite_name}__low_prior_fp"] = block["low_prior_fp"]
            external_rows.append(row)
    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_detectors.csv", index=False)

    # ------------------------------------------------ dependence vs mobility
    with np.load(
        HERE.parent / "results/detectors/external_first_alarms.npz", allow_pickle=False
    ) as archive:
        baselines = {
            mode: np.asarray(archive[f"mobility_s9|{mode}"]) >= 0 for mode in P.MODES
        }
    timely = ~ext["risk"]
    dependence_rows = []
    for key, first in alarms.items():
        quantity, mode = key.split("|", 1)
        baseline = baselines[mode]
        head = first >= 0
        expected = head[timely].sum() * baseline[timely].sum() / int(timely.sum())
        dependence_rows.append(
            {
                "quantity": quantity, "mode": mode,
                "head_fp": int((head & timely).sum()),
                "baseline_fp": int((baseline & timely).sum()),
                "both_fp": int((head & baseline & timely).sum()),
                "expected_both_if_independent": float(expected),
                "false_alarm_dependence": float((head & baseline & timely).sum() / expected)
                if expected > 0 else float("nan"),
                "tp_missed_by_mobility": int((head & ext["risk"] & ~baseline).sum()),
                "or_tp": int(((head | baseline) & ext["risk"]).sum()),
                "or_fp": int(((head | baseline) & timely).sum()),
            }
        )
    pd.DataFrame(dependence_rows).to_csv(args.output / "false_alarm_dependence.csv", index=False)
    np.savez_compressed(args.output / "external_first_alarms.npz", **alarms)

    (args.output / "coupling_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.state_channel.causal_coupling.v1",
                "status": "post-hoc; declared after the prefix/whole-episode sign flip was seen",
                "min_points": MIN_POINTS,
                "protocol": "frozen; width 4, confirmations 4, same selection rule",
            },
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 250)
    print("=== survival-conditioned AUC ===")
    print(
        information.pivot_table(
            index=["cohort", "quantity"], columns="layer", values="survival_auc"
        )[list(LAYER_NAMES)].to_string(float_format="%.4f")
    )
    print("\n=== frozen protocol on external ===")
    columns = ["quantity", "mode", "representation", "direction", "quantile", "tp", "fp",
               "precision", "risk_recall", "low_prior_tp", "low_prior_fp",
               "mean_alarm_prior", "lift", "lift_ci_low", "lift_ci_high", "median_alarm_chunk"]
    print(external[[c for c in columns if c in external.columns]].to_string(index=False, float_format="%.3f"))
    print("\n=== false-alarm dependence against mobility ===")
    print(pd.DataFrame(dependence_rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
