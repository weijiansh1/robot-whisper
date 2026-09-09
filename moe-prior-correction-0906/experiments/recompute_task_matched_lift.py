"""Recompute every published lift on a task-matched survival prior.

Nine bundles import ``survival_prior`` from ``moe-hb-front-back-0905`` which
groups by *suite*.  An audit of 264 threshold-free cells found suite
stratification to be anti-conservative, not merely weak: 52 cells survived a
suite-stratified permutation but only 1 survived a task-stratified one.  So
every lift those bundles published is on the inflated basis.

This recomputes each saved first-alarm vector against both bases and reports
the inflation.  Nothing is re-selected and no threshold is re-fit; the alarm
vectors are read exactly as published.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_LAYER,
    LABEL_ROOT,
    MAIN_LAYER,
    aligned_labels,
    load_npz,
)
import loso_folds  # noqa: E402

COHORTS = {
    "external_8b": (EXTERNAL_LAYER, "external_8b_clean_labels.csv"),
    "development_main": (MAIN_LAYER, "development_main_clean_labels.csv"),
}

# Saved alarm vectors, keyed by the bundle that published them.
SOURCES = {
    "moe-circuit-analogy-0906": {
        "external_8b": "results/detectors/external_first_alarms.npz",
    },
    "moe-flow-semantics-0906": {
        "external_8b": "results/step_alarm/external_first_alarms.npz",
    },
    "moe-state-channel-0906": {
        "external_8b": "results/detectors/external_first_alarms.npz",
    },
    "moe-token-geometry-0906": {
        "external_8b": "results/detection/external_first_alarms.npz",
    },
    "moe-two-tier-0906": {
        "external_8b": "results/external_first_alarms.npz",
        "development_main": "results/development_first_alarms.npz",
    },
    "moe-failure-modes-0906": {
        "external_8b": "results/first_alarms_external.npz",
        "development_main": "results/first_alarms_development.npz",
    },
    "moe-combination-rules-0906": {
        "external_8b": "results/alarms/external_8b_alarms.npz",
        "development_main": "results/alarms/development_main_alarms.npz",
    },
}

# Non-alarm metadata columns stored alongside the vectors in some bundles.
META_KEYS = {"schema", "risk", "suite", "length", "task", "episode", "physical_mode"}


def cohort_frame(name: str) -> dict:
    layer, label_csv = COHORTS[name]
    cache = load_npz(layer)
    labels = aligned_labels(cache, LABEL_ROOT / label_csv, name)
    task_names = cache["task_names"].astype(str)
    return {
        "risk": labels.original_failure.to_numpy(bool),
        "length": cache["length"].astype(int),
        "suite": loso_folds.suite_of(cache),
        "task": task_names[cache["task_index"].astype(int)],
    }


def survival_prior(group: np.ndarray, length: np.ndarray, risk: np.ndarray) -> dict:
    """P(risk | group, still running at chunk q).

    Identical in form to the published ``survival_prior`` but with ``group``
    left free, so the same code produces both the suite-matched and the
    task-matched basis.  Any difference between the two is therefore a
    difference of stratification, not of implementation.
    """
    priors: dict[str, dict[int, float]] = {}
    for name in np.unique(group):
        take = group == name
        priors[str(name)] = {
            chunk: float(risk[take][length[take] > chunk].mean())
            for chunk in range(int(length[take].max()))
        }
    return priors


def prior_of(first: np.ndarray, group: np.ndarray, priors: dict) -> np.ndarray:
    out = np.full(len(first), np.nan)
    fired = first >= 0
    out[fired] = [priors[str(g)][int(c)] for g, c in zip(group[fired], first[fired])]
    return out


LOW_PRIOR = 0.25


def score(first: np.ndarray, coh: dict, priors_suite: dict, priors_task: dict) -> dict:
    fired = first >= 0
    if not fired.any():
        return {}
    risk = coh["risk"]
    tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
    precision = tp / (tp + fp)
    p_suite = prior_of(first, coh["suite"], priors_suite)
    p_task = prior_of(first, coh["task"], priors_task)
    mean_suite = float(np.nanmean(p_suite[fired]))
    mean_task = float(np.nanmean(p_task[fired]))

    # The published headline is usually the *low-prior* subset: alarms fired
    # while the survival prior is still below 0.25.  The basis choice moves the
    # subset as well as the lift, so report both a consistently-suite-matched
    # figure (what was published) and a consistently-task-matched one.
    out = {
        "alarms": tp + fp,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "suite_prior": mean_suite,
        "suite_lift": precision / mean_suite if mean_suite else np.nan,
        "task_prior": mean_task,
        "task_lift": precision / mean_task if mean_task else np.nan,
    }
    for tag, p_sel, p_val in (
        ("lp_suite", p_suite, p_suite),
        ("lp_task", p_task, p_task),
    ):
        early = fired & (p_sel < LOW_PRIOR)
        etp, efp = int((early & risk).sum()), int((early & ~risk).sum())
        prec = etp / (etp + efp) if (etp + efp) else np.nan
        base = float(np.nanmean(p_val[early])) if early.any() else np.nan
        out[f"{tag}_tp"] = etp
        out[f"{tag}_fp"] = efp
        out[f"{tag}_precision"] = prec
        out[f"{tag}_lift"] = prec / base if base else np.nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort_name in COHORTS:
        coh = cohort_frame(cohort_name)
        ps = survival_prior(coh["suite"], coh["length"], coh["risk"])
        pt = survival_prior(coh["task"], coh["length"], coh["risk"])

        # The prior mechanism must be exact: alarming on *everything still
        # running* at a fixed chunk is, by definition, the survival prior, so
        # its lift is 1 under both bases.  If this drifts the metric is broken.
        for chunk in (0, 4, 8, 12):
            ctrl = np.where(coh["length"] > chunk, chunk, -1).astype(np.int16)
            got = score(ctrl, coh, ps, pt)
            for basis in ("suite_lift", "task_lift"):
                assert abs(got[basis] - 1.0) < 1e-9, (chunk, basis, got[basis])
            rows.append(
                {
                    "bundle": "_control",
                    "cohort": cohort_name,
                    "detector": f"all_running_at_chunk{chunk}",
                    **got,
                }
            )

        # Length is not a baseline: because risk is defined as failing to
        # finish before the cap, any sub-cap length threshold recovers every
        # risk by construction.  Kept only as a negative control.
        cap = pd.Series(coh["length"]).groupby(coh["suite"]).transform("max").to_numpy()
        length_alarm = np.where(coh["length"] >= cap, 0, -1).astype(np.int16)
        rows.append(
            {
                "bundle": "_negative_control",
                "cohort": cohort_name,
                "detector": "length_at_cap (NOT a baseline)",
                **score(length_alarm, coh, ps, pt),
            }
        )

        for bundle, files in SOURCES.items():
            rel = files.get(cohort_name)
            if rel is None:
                continue
            path = ROOT / bundle / rel
            if not path.exists():
                print(f"  missing {path}", file=sys.stderr)
                continue
            data = np.load(path, allow_pickle=True)
            # Where a bundle stored its own labels, they must agree with the
            # canonical ones or the vectors are not comparably aligned.
            if "risk" in data.files:
                assert np.array_equal(data["risk"].astype(bool), coh["risk"]), path
            if "length" in data.files:
                assert np.array_equal(data["length"].astype(int), coh["length"]), path
            for key in data.files:
                if key in META_KEYS:
                    continue
                first = data[key]
                if first.ndim != 1 or first.shape[0] != len(coh["risk"]):
                    continue
                got = score(first.astype(int), coh, ps, pt)
                if got:
                    rows.append(
                        {"bundle": bundle, "cohort": cohort_name, "detector": key, **got}
                    )

    df = pd.DataFrame(rows)
    df["inflation"] = df["task_prior"] / df["suite_prior"]
    df = df.sort_values(["cohort", "bundle", "detector"]).reset_index(drop=True)
    df.to_csv(OUT / "task_matched_lift.csv", index=False)

    published = df[~df.bundle.str.startswith("_")]
    summary = {
        "n_detectors": int(len(published)),
        "n_bundles": int(published.bundle.nunique()),
        "median_suite_lift": float(published.suite_lift.median()),
        "median_task_lift": float(published.task_lift.median()),
        "n_suite_lift_above_1": int((published.suite_lift > 1).sum()),
        "n_task_lift_above_1": int((published.task_lift > 1).sum()),
        "n_crossing_below_1": int(
            ((published.suite_lift > 1) & (published.task_lift <= 1)).sum()
        ),
        "median_shrinkage_pct": float(
            100
            * (
                1
                - (published.task_lift - 1).clip(lower=0)
                / (published.suite_lift - 1).clip(lower=1e-9)
            )
            .clip(0, 100)
            .median()
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Lifts computed on a handful of alarms carry no information either way;
    # a detector that fires 5 times can show any lift at all.  Report the
    # substantive population separately from the full table.
    MIN_ALARMS = 20
    subst = published[published.alarms >= MIN_ALARMS]
    summary["n_substantive"] = int(len(subst))
    summary["min_alarms"] = MIN_ALARMS
    summary["substantive_median_suite_lift"] = float(subst.suite_lift.median())
    summary["substantive_median_task_lift"] = float(subst.task_lift.median())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{len(published)} published detectors over {summary['n_bundles']} bundles")
    print(f"{len(subst)} of them fire at least {MIN_ALARMS} times\n")
    print("suite-matched lift > 1: %d   task-matched lift > 1: %d   fall below 1: %d"
          % (summary["n_suite_lift_above_1"], summary["n_task_lift_above_1"],
             summary["n_crossing_below_1"]))
    print("median lift, all detectors   suite %.4f -> task %.4f"
          % (summary["median_suite_lift"], summary["median_task_lift"]))
    print("median lift, >=%d alarms      suite %.4f -> task %.4f"
          % (MIN_ALARMS, summary["substantive_median_suite_lift"],
             summary["substantive_median_task_lift"]))
    print("\n-- controls (lift must equal 1 exactly under both bases) --")
    ctl = df[df.bundle == "_control"]
    print(ctl[["cohort", "detector", "alarms", "suite_lift", "task_lift"]]
          .to_string(index=False))
    print("\n-- negative control: length is NOT a baseline --")
    neg = df[df.bundle == "_negative_control"]
    print(neg[["cohort", "detector", "tp", "fp", "precision", "task_lift"]]
          .to_string(index=False))

    cols = ["cohort", "bundle", "detector", "alarms", "precision",
            "suite_lift", "task_lift"]
    lp = ["cohort", "detector", "lp_suite_tp", "lp_suite_fp", "lp_suite_lift",
          "lp_task_tp", "lp_task_fp", "lp_task_lift"]
    print(f"\n-- published as beating the survival prior, but does not (>={MIN_ALARMS} alarms) --")
    crossed = subst[(subst.suite_lift > 1) & (subst.task_lift <= 1)]
    print(crossed.sort_values("suite_lift", ascending=False)[cols].to_string(index=False)
          if len(crossed) else "   none")
    print(f"\n-- largest suite->task shrinkage (>={MIN_ALARMS} alarms) --")
    worst = subst.assign(drop=subst.suite_lift - subst.task_lift)
    print(worst.nlargest(15, "drop")[cols].to_string(index=False))
    print(f"\n-- survives task matching best (>={MIN_ALARMS} alarms) --")
    print(subst.nlargest(15, "task_lift")[cols].to_string(index=False))
    print("\n-- low-prior subset, both bases, for the strongest survivors --")
    print(subst.nlargest(12, "task_lift")[lp].to_string(index=False))


if __name__ == "__main__":
    main()
