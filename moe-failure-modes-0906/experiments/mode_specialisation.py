#!/usr/bin/env python3
"""Q1 + Q2: is recall-by-physical-mode real after controlling suite and timing,
and is the quantity -> mode mapping stable from development to external?

Design (see results/PREREG.md):

  statistic     pooled recall of detector d on risk episodes of mode m
  null          permute the physical mode among risk episodes WITHIN a stratum
  strata        `suite` (the named confound) and `task` (strictly finer; it also
                absorbs the per_task threshold channel)
  effect        SEL(d, m) = log2( r_dm / E_stratum[r_dm] )
                E_stratum[r_dm] = sum_s (n_ms / n_m) * r_ds is exactly the
                detector's OWN recall re-weighted to mode m's stratum mix, so the
                effect is relative to the detector, as required.
  multiplicity  Benjamini-Hochberg, q = 0.05, within each stratification.

Timing. Every risk episode runs to its suite horizon cap (verified in
results/join_audit.json), so within a suite no mode gets more chunks of exposure
than another. Timing is therefore reported as (a) alarm chunk / survival prior at
alarm by mode, and (b) a prior-matched recall in which every alarm later than the
chunk where the suite prior reaches 0.25 is discarded.

Nothing here feeds a runtime decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"
B_PERM = 20000
SEED = 20260906
SMALL_CELL = 30
BH_Q = 0.05


def build(cohort: str):
    frame = C.cohort_index(cohort)
    key = "development" if cohort == "development_main" else "external"
    alarms = C.load_npz(OUT / f"first_alarms_{key}.npz")
    alarms.pop("schema", None)
    suite = frame["suite"].to_numpy(str)
    risk = frame["risk"].to_numpy(bool)
    length = frame["length"].to_numpy(int)
    priors = C.survival_prior(suite, length, risk)
    return frame, alarms, suite, risk, priors


def permutations(stratum: np.ndarray, rng: np.random.Generator, b: int) -> np.ndarray:
    n = len(stratum)
    out = np.tile(np.arange(n, dtype=np.int32), (b, 1))
    for value in np.unique(stratum):
        take = np.flatnonzero(stratum == value)
        if len(take) < 2:
            continue
        block = np.argsort(rng.random((b, len(take))), axis=1)
        out[:, take] = take[block]
    return out


def bh(pvalues: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order]
    n = len(pvalues)
    thresh = q * np.arange(1, n + 1) / n
    passed = ranked <= thresh
    cutoff = np.max(np.flatnonzero(passed)) if passed.any() else -1
    keep = np.zeros(n, bool)
    if cutoff >= 0:
        keep[order[: cutoff + 1]] = True
    return keep


def analyse(cohort: str, population: str = "all_risks") -> pd.DataFrame:
    frame, alarms, suite, risk, priors = build(cohort)
    if population == "persistent":
        take_risk = risk & ~frame["late_success_plus10_queries"].astype(bool).to_numpy()
    else:
        take_risk = risk
    idx = np.flatnonzero(take_risk)
    mode = frame["primary_failure_reason"].fillna("unknown").to_numpy(str)[idx]
    suite_r = suite[idx]
    task_r = frame["task_key"].to_numpy(str)[idx]

    modes = [m for m in pd.Series(mode).value_counts().index]
    onehot = np.stack([(mode == m).astype(np.float32) for m in modes], axis=1)
    n_m = onehot.sum(axis=0)

    rng = np.random.default_rng(SEED)
    perms = {
        "suite": permutations(suite_r, rng, B_PERM),
        "task": permutations(task_r, rng, B_PERM),
    }

    # per-suite chunk where the survival prior first reaches 0.25; alarms strictly
    # before it are "early". Verified to be goal 18 / long 26 / object 17 / spatial 13.
    early_cut = {}
    for s, table in priors.items():
        chunks = sorted(table)
        above = [c for c in chunks if table[c] >= C.LOW_PRIOR]
        early_cut[s] = int(min(above)) if above else int(max(chunks)) + 1

    rows = []
    for name, first in sorted(alarms.items()):
        first_r = np.asarray(first, dtype=int)[idx]
        fired = (first_r >= 0).astype(np.float32)
        prior_at = C.prior_of(first_r, suite_r, priors)
        early = ((first_r >= 0) & (prior_at < C.LOW_PRIOR)).astype(np.float32)
        cut = np.asarray([early_cut[s] for s in suite_r])
        matched = ((first_r >= 0) & (first_r < cut)).astype(np.float32)

        caught_m = fired @ onehot
        early_m = early @ onehot
        matched_m = matched @ onehot
        overall = float(fired.mean())

        # stratum-matched expectation and permutation null
        stats = {}
        for stratum_name, perm in perms.items():
            null = (fired[perm] @ onehot) / n_m  # (B, n_modes)
            observed = caught_m / n_m
            expected = null.mean(axis=0)
            centred = np.abs(observed - expected)
            p_two = (np.abs(null - expected) >= centred[None, :] - 1e-12).mean(axis=0)
            # omnibus: sum over modes of standardised squared deviation
            var = null.var(axis=0)
            safe = np.where(var > 0, var, np.inf)
            t_obs = float((((observed - expected) ** 2) / safe).sum())
            t_null = (((null - expected[None, :]) ** 2) / safe[None, :]).sum(axis=1)
            stats[stratum_name] = {
                "expected": expected,
                "null_sd": np.sqrt(var),
                "p": p_two,
                "omnibus_p": float((t_null >= t_obs - 1e-12).mean()),
                "omnibus_t": t_obs,
            }

        for position, m in enumerate(modes):
            rec = float(caught_m[position] / n_m[position])
            take_mode = mode == m
            chunks = first_r[take_mode & (first_r >= 0)]
            row = {
                "cohort": cohort,
                "population": population,
                "detector": name,
                "mode": m,
                "mode_short": C.MODE_SHORT.get(m, m),
                "n_mode": int(n_m[position]),
                "caught": int(caught_m[position]),
                "recall": rec,
                "overall_recall": overall,
                "early_caught": int(early_m[position]),
                "early_recall": float(early_m[position] / n_m[position]),
                "prior_matched_caught": int(matched_m[position]),
                "prior_matched_recall": float(matched_m[position] / n_m[position]),
                "mean_alarm_chunk": float(chunks.mean()) if len(chunks) else np.nan,
                "mean_alarm_prior": float(np.nanmean(prior_at[take_mode]))
                if np.isfinite(prior_at[take_mode]).any() else np.nan,
                "small_cell": bool(n_m[position] < SMALL_CELL),
                "omnibus_p_suite": stats["suite"]["omnibus_p"],
                "omnibus_p_task": stats["task"]["omnibus_p"],
            }
            for stratum_name in ("suite", "task"):
                exp = float(stats[stratum_name]["expected"][position])
                sd = float(stats[stratum_name]["null_sd"][position])
                row[f"expected_recall_{stratum_name}"] = exp
                row[f"sel_{stratum_name}"] = (
                    float(np.log2(rec / exp)) if rec > 0 and exp > 0
                    else (-np.inf if exp > 0 else np.nan)
                )
                row[f"p_{stratum_name}"] = float(stats[stratum_name]["p"][position])
                row[f"null_sd_{stratum_name}"] = sd
                row[f"z_{stratum_name}"] = (rec - exp) / sd if sd > 0 else np.nan
                # smallest |recall - expected| a two-sided 0.05 test could see
                row[f"mde_{stratum_name}"] = 1.96 * sd
            # how much of the suite-controlled effect is still there within task
            eff_s = rec - float(stats["suite"]["expected"][position])
            eff_t = rec - float(stats["task"]["expected"][position])
            row["effect_suite"] = eff_s
            row["effect_task"] = eff_t
            row["within_task_share"] = eff_t / eff_s if abs(eff_s) > 1e-9 else np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    for stratum_name in ("suite", "task"):
        keep = bh(table[f"p_{stratum_name}"].to_numpy(), BH_Q)
        table[f"bh_{stratum_name}"] = keep
    return table


def stability(dev: pd.DataFrame, ext: pd.DataFrame) -> dict:
    merged = dev.merge(
        ext, on=["detector", "mode"], suffixes=("_dev", "_ext"), how="inner"
    )
    out = {}
    for stratum in ("suite", "task"):
        cells = merged[
            (merged["n_mode_dev"] >= SMALL_CELL) & (merged["n_mode_ext"] >= SMALL_CELL)
        ]
        a = cells[f"sel_{stratum}_dev"].to_numpy()
        b = cells[f"sel_{stratum}_ext"].to_numpy()
        good = np.isfinite(a) & np.isfinite(b)
        a, b = a[good], b[good]
        if len(a) > 2:
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            rho = float(np.corrcoef(ra, rb)[0, 1])
            pearson = float(np.corrcoef(a, b)[0, 1])
            sign = float((np.sign(a) == np.sign(b)).mean())
        else:
            rho = pearson = sign = float("nan")
        # argmax-mode agreement, restricted to non-small cells
        agree, total = 0, 0
        for detector, block in cells.groupby("detector"):
            block = block[np.isfinite(block[f"sel_{stratum}_dev"])
                          & np.isfinite(block[f"sel_{stratum}_ext"])]
            if len(block) < 2:
                continue
            total += 1
            top_dev = block.loc[block[f"sel_{stratum}_dev"].idxmax(), "mode"]
            top_ext = block.loc[block[f"sel_{stratum}_ext"].idxmax(), "mode"]
            agree += int(top_dev == top_ext)
        out[stratum] = {
            "cells": int(len(a)),
            "spearman": rho,
            "pearson": pearson,
            "sign_agreement": sign,
            "argmax_agreement": agree / total if total else float("nan"),
            "detectors_compared": total,
        }
    merged.to_csv(OUT / "selectivity_dev_vs_external.csv", index=False)
    return out


def main() -> None:
    tables = []
    for cohort in ("development_main", "external_8b"):
        for population in ("all_risks", "persistent"):
            table = analyse(cohort, population)
            tables.append(table)
            print(f"{cohort}/{population}: {len(table)} cells", flush=True)
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(OUT / "mode_recall.csv", index=False)

    dev = full[(full["cohort"] == "development_main") & (full["population"] == "all_risks")]
    ext = full[(full["cohort"] == "external_8b") & (full["population"] == "all_risks")]
    report = {
        "schema": "himoe.failure_modes_0906.mode_specialisation.v1",
        "permutations": B_PERM,
        "seed": SEED,
        "bh_q": BH_Q,
        "small_cell_threshold": SMALL_CELL,
        "stability_all_risks": stability(dev, ext),
        "omnibus_external": {
            d: {"suite": float(b["omnibus_p_suite"].iloc[0]),
                "task": float(b["omnibus_p_task"].iloc[0])}
            for d, b in ext.groupby("detector")
        },
        "omnibus_development": {
            d: {"suite": float(b["omnibus_p_suite"].iloc[0]),
                "task": float(b["omnibus_p_task"].iloc[0])}
            for d, b in dev.groupby("detector")
        },
        "significant_cells_external": {
            "suite": int(ext["bh_suite"].sum()),
            "task": int(ext["bh_task"].sum()),
            "total": int(len(ext)),
        },
        "significant_cells_development": {
            "suite": int(dev["bh_suite"].sum()),
            "task": int(dev["bh_task"].sum()),
            "total": int(len(dev)),
        },
    }
    (OUT / "mode_specialisation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["stability_all_risks"], indent=2))
    print(json.dumps(report["significant_cells_external"], indent=2))


if __name__ == "__main__":
    main()
