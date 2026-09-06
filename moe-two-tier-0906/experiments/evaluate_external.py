#!/usr/bin/env python3
"""Score the frozen development selection on external_8b, once.

Nothing is chosen here. Every rule evaluated is read back from
results/development_rules.csv, which was written by select_on_development.py
before external combination scoring began, and every objective is the one in
results/PREREG.md.

Also runs, on external:
  * the three published anchors, as hard assertions;
  * the two-tier joint accounting (WATCH-only / ACT / neither, per-tier false
    alarm budget, ACT-inside-WATCH containment, alarm-time ordering);
  * the OR increment's mode-concentration permutation test, stratified by
    suite and separately by task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import twotier_lib as lib


SINGLE_HEAD_LIFT_ANCHOR = 1.7640674271437853
ANCHORS = {
    "mobility|global": {
        "representation": "L12",
        "direction": "low",
        "quantile": 0.975,
        "tp": 195,
        "fp": 17,
        "precision": 0.9198113207547169,
        "lift": SINGLE_HEAD_LIFT_ANCHOR,
    },
    "mobility|per_task": {
        "representation": "L2",
        "direction": "low",
        "quantile": 0.70,
        "tp": 272,
        "fp": 57,
    },
    "expert_load_effective_rank|per_task": {
        "representation": "L3",
        "direction": "low",
        "quantile": 0.85,
        "tp": 370,
        "fp": 93,
        "lift": 1.5478999999999999,
    },
}
PERMUTATIONS = 10_000
SEED = 20260906
SKIP = {"schema", "risk", "suite", "length", "physical_mode"}

# The frozen pairing: which development-selected rule plays each tier.
TIERS = {
    "watch": ("watch_a3", "watch_a1", "watch_a2", "watch_b", "watch_b4"),
    "act": ("act_b", "act_a"),
}
HEADLINE = {"watch": "watch_a3", "act": "act_b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lib.RESULTS)
    return parser.parse_args()


def assert_anchors(alarms: dict[str, np.ndarray], cohort: dict[str, Any]) -> dict[str, Any]:
    inventory = pd.read_csv(lib.RESULTS / "head_inventory.csv").set_index("head")
    out: dict[str, Any] = {}
    from select_early_lock import prior_of, score_candidate  # noqa: E402

    for head, want in ANCHORS.items():
        first = alarms[head]
        got = score_candidate(
            first, cohort["risk"], prior_of(first, cohort["suite"], cohort["priors"])
        )
        row = inventory.loc[head]
        checks = {
            "representation": (str(row["representation"]), want["representation"]),
            "direction": (str(row["direction"]), want["direction"]),
            "quantile": (float(row["quantile"]), float(want["quantile"])),
            "tp": (int(got["tp"]), int(want["tp"])),
            "fp": (int(got["fp"]), int(want["fp"])),
        }
        for key in ("precision", "lift"):
            if key in want:
                checks[key] = (float(got[key]), float(want[key]))
        failures = [
            k
            for k, (a, b) in checks.items()
            if (abs(a - b) > 1e-3 if isinstance(a, float) else a != b)
        ]
        if failures:
            raise AssertionError(f"anchor {head} failed on {failures}: {checks}")
        out[head] = {k: v[0] for k, v in checks.items()}
    return out


def mode_concentration_test(
    increment: np.ndarray,
    risk: np.ndarray,
    modes: np.ndarray,
    scored: Sequence[str],
    strata: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Is the OR's increment concentrated on particular physical modes?

    Null permutes the increment indicator among risk episodes within stratum,
    preserving both the per-stratum increment count and each stratum's mode
    composition. Statistic is the CV of the per-mode increment rate; larger
    means more concentrated.
    """
    take = np.flatnonzero(risk)
    flag = increment[take]
    mode_of = modes[take]
    stratum_of = strata[take]
    blocks = [np.flatnonzero(stratum_of == s) for s in np.unique(stratum_of)]
    membership = [mode_of == name for name in scored]
    sizes = np.asarray([int(m.sum()) for m in membership], dtype=float)

    def statistic(vector: np.ndarray) -> float:
        rates = np.asarray(
            [float(vector[m].sum()) / n for m, n in zip(membership, sizes, strict=True)]
        )
        mean = rates.mean()
        return float(rates.std() / mean) if mean > 0 else float("nan")

    observed = statistic(flag)
    null = np.empty(PERMUTATIONS)
    for draw in range(PERMUTATIONS):
        shuffled = np.empty_like(flag)
        for block in blocks:
            shuffled[block] = rng.permutation(flag[block])
        null[draw] = statistic(shuffled)
    finite = np.isfinite(null)
    return {
        "observed_cv": observed,
        "null_mean_cv": float(null[finite].mean()),
        "p_value": float((null[finite] >= observed).sum() + 1) / float(finite.sum() + 1),
        "permutations": int(finite.sum()),
        "increment_n": int(flag.sum()),
        "per_mode_increment": {
            name: {"n": int(n), "increment": int(m[flag].sum() if flag.any() else 0)}
            for name, m, n in zip(scored, membership, sizes, strict=True)
        },
    }


def main() -> None:
    args = parse_args()
    cohort = lib.load_cohort("external_8b")
    data = np.load(args.output / "external_first_alarms.npz", allow_pickle=True)
    if not np.array_equal(data["risk"].astype(bool), cohort["risk"]):
        raise ValueError("external alarm cache is not aligned with the labels")
    alarms = {k: np.asarray(data[k], np.int16) for k in data.files if k not in SKIP}
    modes = data["physical_mode"].astype(str)
    counts = lib.mode_counts(modes, cohort["risk"])
    scored = lib.scored_modes(counts)
    fp_cap = 0.005 * int((~cohort["risk"]).sum())

    anchors = assert_anchors(alarms, cohort)
    print("anchors reproduce:", json.dumps(anchors, indent=None, sort_keys=True), flush=True)

    rules = pd.read_csv(args.output / "development_rules.csv")
    rows: list[dict[str, Any]] = []
    built: dict[tuple[str, str, str], np.ndarray] = {}

    for _, rule in rules.iterrows():
        keys = str(rule["heads"]).split("+")
        first = lib.combine([alarms[key] for key in keys], int(rule["k"]))
        record = lib.evaluate(first, cohort, modes, scored, counts)
        record.update(
            {
                "pool": rule["pool"],
                "mode": rule["mode"],
                "role": rule["role"],
                "heads": rule["heads"],
                "n_heads": int(rule["n_heads"]),
                "k": int(rule["k"]),
                "frames": rule["frames"],
                "n_frames": int(rule["n_frames"]),
                "dev_tp": int(rule["tp"]),
                "dev_fp": int(rule["fp"]),
                "dev_worst_mode_coverage": float(rule["worst_mode_coverage"]),
                "dev_mode_cv": float(rule["mode_cv"]),
                "dev_fits_fp_cap": bool(rule["fits_dev_fp_cap"]),
                "fits_ext_fp_cap": bool(record["fp"] <= fp_cap),
                "lift_vs_single_head_anchor": float(record["lift"]) - SINGLE_HEAD_LIFT_ANCHOR,
            }
        )
        record.update(lib.suite_breakdown(first, cohort))
        rows.append(record)
        if rule["role"] in TIERS["watch"] or rule["role"] in TIERS["act"]:
            built[(str(rule["pool"]), str(rule["mode"]), str(rule["role"]))] = first

    external = pd.DataFrame(rows)
    external.to_csv(args.output / "external_rules.csv", index=False)

    # ---------------- (a) approximates (b)? ------------------------------
    verdicts: dict[str, Any] = {}
    for (pool, mode), block in external.groupby(["pool", "mode"]):
        indexed = block.set_index("role")
        if "watch_b" not in indexed.index:
            continue
        entry: dict[str, Any] = {}
        for name in ("watch_a1", "watch_a2", "watch_a3", "watch_a3_free"):
            if name not in indexed.index:
                continue
            left = indexed.loc[name]
            for comparator in ("watch_b", "watch_b4"):
                if comparator not in indexed.index:
                    continue
                right = indexed.loc[comparator]
                lset = set(str(left["heads"]).split("+"))
                rset = set(str(right["heads"]).split("+"))
                entry[f"{name}_vs_{comparator}"] = {
                    "identical_head_set": lset == rset,
                    "jaccard": len(lset & rset) / len(lset | rset),
                    "external_worst_mode_gap": float(
                        left["worst_mode_coverage"] - right["worst_mode_coverage"]
                    ),
                    "external_mode_cv_gap": float(left["mode_cv"] - right["mode_cv"]),
                    "approximates_within_0.05": bool(
                        left["worst_mode_coverage"] >= right["worst_mode_coverage"] - 0.05
                    ),
                    "left_fits_dev_fp_cap": bool(left["dev_fits_fp_cap"]),
                }
        verdicts[f"{pool}|{mode}"] = entry

    # ---------------- two-tier joint accounting --------------------------
    joint: dict[str, Any] = {}
    for pool, mode in sorted({(p, m) for p, m, _ in built}):
        entry: dict[str, Any] = {}
        for watch_role in TIERS["watch"]:
            if (pool, mode, watch_role) not in built:
                continue
            watch = built[(pool, mode, watch_role)]
            for act_role in TIERS["act"]:
                if (pool, mode, act_role) not in built:
                    continue
                act = built[(pool, mode, act_role)]
                w, a = watch >= 0, act >= 0
                risk = cohort["risk"]
                shared = w & a
                entry[f"{watch_role}+{act_role}"] = {
                    "watch_alarms": int(w.sum()),
                    "act_alarms": int(a.sum()),
                    "watch_only_alarms": int((w & ~a).sum()),
                    "act_outside_watch": int((a & ~w).sum()),
                    "act_containment_in_watch": float(shared.sum() / max(int(a.sum()), 1)),
                    "no_alarm": int((~w & ~a).sum()),
                    "watch_tp": int((w & risk).sum()),
                    "watch_fp": int((w & ~risk).sum()),
                    "act_tp": int((a & risk).sum()),
                    "act_fp": int((a & ~risk).sum()),
                    "watch_only_tp": int((w & ~a & risk).sum()),
                    "watch_only_fp": int((w & ~a & ~risk).sum()),
                    "act_escalation_rate_of_watch_tp": float(
                        (w & a & risk).sum() / max(int((w & risk).sum()), 1)
                    ),
                    "act_escalation_rate_of_watch_fp": float(
                        (w & a & ~risk).sum() / max(int((w & ~risk).sum()), 1)
                    ),
                    "act_after_watch_on_shared": int(
                        (act[shared] >= watch[shared]).sum()
                    ),
                    "shared_episodes": int(shared.sum()),
                    "median_delay_act_minus_watch": float(
                        np.median((act[shared] - watch[shared])) if shared.any() else np.nan
                    ),
                }
        joint[f"{pool}|{mode}"] = entry

    # ---------------- increment concentration ----------------------------
    increment: dict[str, Any] = {}
    for pool, mode in sorted({(p, m) for p, m, _ in built}):
        role = HEADLINE["watch"]
        if (pool, mode, role) not in built:
            continue
        rule = external[
            (external["pool"] == pool)
            & (external["mode"] == mode)
            & (external["role"] == role)
        ].iloc[0]
        keys = str(rule["heads"]).split("+")
        solo = {key: int(((alarms[key] >= 0) & cohort["risk"]).sum()) for key in keys}
        best = max(solo, key=lambda key: (solo[key], key))
        combined = built[(pool, mode, role)] >= 0
        flag = combined & (alarms[best] < 0)
        rng = np.random.default_rng(SEED)
        entry = {
            "watch_role": role,
            "heads": rule["heads"],
            "best_single_member": best,
            "best_single_member_tp": solo[best],
            "or_tp": int(rule["tp"]),
            "increment_tp": int((flag & cohort["risk"]).sum()),
            "increment_fp": int((flag & ~cohort["risk"]).sum()),
        }
        for name, strata in (
            ("suite_stratified", cohort["suite"]),
            ("task_stratified", cohort["task"]),
        ):
            entry[name] = mode_concentration_test(
                flag, cohort["risk"], modes, scored, strata, np.random.default_rng(SEED)
            )
        increment[f"{pool}|{mode}"] = entry

    summary = {
        "schema": "himoe.two_tier.external_evaluation.v1",
        "evaluated_once": True,
        "anchors": anchors,
        "single_head_lift_anchor": SINGLE_HEAD_LIFT_ANCHOR,
        "external_mode_counts": counts,
        "modes_scored": scored,
        "external_fp_cap": fp_cap,
        "watch_a_vs_b": verdicts,
        "two_tier_joint": joint,
        "or_increment_concentration": increment,
        "permutations": PERMUTATIONS,
        "seed": SEED,
    }
    (args.output / "external_evaluation.json").write_text(
        json.dumps(lib.plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    show = external[
        external["role"].isin(
            list(TIERS["watch"]) + list(TIERS["act"]) + ["reference_controller_trio"]
        )
    ]
    print("\n=== external, frozen rules ===")
    print(
        show[
            [
                "pool",
                "mode",
                "role",
                "k",
                "n_frames",
                "tp",
                "fp",
                "precision",
                "lift",
                "low_prior_tp",
                "low_prior_fp",
                "worst_mode_coverage",
                "mode_cv",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )
    print("\nwrote external_rules.csv and external_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
