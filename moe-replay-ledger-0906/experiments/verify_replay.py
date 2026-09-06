#!/usr/bin/env python3
"""Test the causality claim behind every stored first-alarm array.

The claim is: at chunk q a detector saw chunks 0..q only, so truncating an
episode at or after its alarm chunk cannot move the alarm.

The test is run on the head that everything else is anchored to,
``mobility|L12|low|q0.975|global``, rebuilt from the raw per-chunk mobility
array rather than read from the bundle:

  step 1  rebuild the head from the scores and check it is bit-identical to the
          sealed array published by moe-hb-front-back-0905;
  step 2  for every episode that alarmed, truncate the mobility array after the
          alarm chunk (all later chunks set to NaN and marked invalid), rerun
          the identical pipeline, and require the alarm chunk to be unchanged;
  step 3  as a falsification control, truncate one chunk *before* the alarm and
          require that the alarm disappears -- if it survived, the alarm chunk
          would not be the first chunk at which the evidence existed;
  step 4  repeat step 2 for a per_task head and for a rule with an 8-chunk
          confirmation window, which is the longest look-back in the corpus.

A pipeline that peeked at the future would fail step 2 for at least some rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_core import (  # noqa: E402
    CONFIRMATIONS,
    HB,
    PROFILE_ROOT,
    Cohort,
    _npz,
    dev,
    load_npz,
    MOBILITY_PATHS,
    oriented,
    quantity_values,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _reprs(quantity: str, cohorts: dict[str, tuple[dict, dict]]) -> dict[str, dict]:
    return {
        name: {
            k: v
            for k, (v, _) in dev.representations(
                quantity_values(graph, mob, quantity)
            ).items()
        }
        for name, (graph, mob) in cohorts.items()
    }


def _global_line(reprs: dict, rep: str, direction: str, quantile: float) -> float:
    pooled = np.concatenate(
        [
            dev.row_max(oriented(reprs[n][rep], direction))
            for n in ("development_main", "development_extra")
        ]
    )
    return dev.quantile_higher(pooled, quantile)


def _first_from_scores(
    values: np.ndarray, valid: np.ndarray, direction: str, line, confirmations: int
) -> np.ndarray:
    persistent = dev.persistent_score(oriented(values, direction), confirmations)
    line_arr = line if np.ndim(line) else np.full((len(values), 1), float(line))
    return dev.first_query(np.isfinite(persistent) & (persistent > line_arr) & valid)


def truncation_test(
    values: np.ndarray,
    valid: np.ndarray,
    direction: str,
    line,
    confirmations: int,
    reference: np.ndarray,
    offset: int,
) -> dict:
    """Truncate every alarming episode `offset` chunks after its alarm."""
    fired = np.flatnonzero(reference >= 0)
    trunc_values = values.copy()
    trunc_valid = valid.copy()
    for row in fired:
        cut = int(reference[row]) + offset + 1
        trunc_values[row, cut:] = np.nan
        trunc_valid[row, cut:] = False
    replayed = _first_from_scores(
        trunc_values, trunc_valid, direction, line, confirmations
    )
    same = replayed[fired] == reference[fired]
    return {
        "episodes_tested": int(len(fired)),
        "alarm_chunk_unchanged": int(same.sum()),
        "alarm_chunk_moved": int((~same).sum()),
        "alarm_lost": int((replayed[fired] < 0).sum()),
    }


def falsification_test(
    values: np.ndarray,
    valid: np.ndarray,
    direction: str,
    line,
    confirmations: int,
    reference: np.ndarray,
) -> dict:
    """Truncate one chunk *before* the alarm; the alarm must not survive."""
    fired = np.flatnonzero(reference > 0)
    trunc_values = values.copy()
    trunc_valid = valid.copy()
    for row in fired:
        cut = int(reference[row])
        trunc_values[row, cut:] = np.nan
        trunc_valid[row, cut:] = False
    replayed = _first_from_scores(
        trunc_values, trunc_valid, direction, line, confirmations
    )
    return {
        "episodes_tested": int(len(fired)),
        "alarm_correctly_absent": int((replayed[fired] < 0).sum()),
        "alarm_survived_truncation": int((replayed[fired] >= 0).sum()),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cohort = Cohort("external_8b")
    graphs = {
        n: load_npz(PROFILE_ROOT / f"{n}.npz")
        for n in ("development_main", "development_extra")
    }
    mobs = {n: load_npz(MOBILITY_PATHS[n]) for n in graphs}
    graphs["external_8b"] = cohort.graph_cache
    mobs["external_8b"] = cohort.mobility_cache
    pack = {n: (graphs[n], mobs[n]) for n in graphs}

    sealed = _npz(HB / "results/frame_survey/external_first_alarms.npz")
    report: dict = {
        "schema": "himoe.replay_ledger.causality_check.v1",
        "definition": (
            "at chunk q the detector may read chunks 0..q only; truncating an "
            "episode at or after its alarm chunk must leave the alarm chunk fixed"
        ),
        "checks": [],
    }

    # ---- head 1: the global anchor, rebuilt from raw scores -----------------
    reprs = _reprs("mobility", pack)
    rep, direction, quantile = "L12", "low", 0.975
    line = _global_line(reprs, rep, direction, quantile)
    values = reprs["external_8b"][rep]
    rebuilt = _first_from_scores(
        values, cohort.valid, direction, line, CONFIRMATIONS
    )
    stored = sealed["mobility|global"].astype(np.int16)
    tp = int(((rebuilt >= 0) & cohort.risk).sum())
    fp = int(((rebuilt >= 0) & ~cohort.risk).sum())
    check = {
        "head": "mobility|L12|low|q0.975|global",
        "rebuilt_from": "moe-v4-0904/results/layerwise_mobility/external_8b.npz "
        "(raw per-chunk mobility), threshold from the pooled development peaks",
        "bit_identical_to_sealed_array": bool(np.array_equal(rebuilt, stored)),
        "rows_differing_from_sealed": int((rebuilt != stored).sum()),
        "rebuilt_tp": tp,
        "rebuilt_fp": fp,
        "published_tp": 195,
        "published_fp": 17,
        "global_threshold": float(line),
        "truncate_at_alarm": truncation_test(
            values, cohort.valid, direction, line, CONFIRMATIONS, rebuilt, 0
        ),
        "truncate_one_chunk_after_alarm": truncation_test(
            values, cohort.valid, direction, line, CONFIRMATIONS, rebuilt, 1
        ),
        "truncate_one_chunk_before_alarm": falsification_test(
            values, cohort.valid, direction, line, CONFIRMATIONS, rebuilt
        ),
    }
    report["checks"].append(check)
    print(json.dumps(check, indent=1), flush=True)

    # ---- head 2: a per_task head (task-side thresholds) ---------------------
    survey = pd.read_csv(HB / "results/frame_survey/external_detectors.csv")
    row = survey[(survey["quantity"] == "mobility") & (survey["mode"] == "per_task")]
    rep2 = str(row.iloc[0]["representation"])
    dir2 = str(row.iloc[0]["direction"])
    q2 = float(row.iloc[0]["quantile"])
    ref_task = {
        n: graphs[n]["task_names"].astype(str)[graphs[n]["task_index"].astype(int)]
        for n in ("development_main", "development_extra")
    }
    lines = np.full((cohort.n, 1), np.nan)
    for task in np.unique(cohort.task):
        take = np.flatnonzero(cohort.task == task)
        peaks = [
            dev.row_max(oriented(reprs[n][rep2][ref_task[n] == task], dir2))
            for n in ("development_main", "development_extra")
            if (ref_task[n] == task).any()
        ]
        lines[take, 0] = dev.quantile_higher(np.concatenate(peaks), q2)
    values2 = reprs["external_8b"][rep2]
    rebuilt2 = _first_from_scores(values2, cohort.valid, dir2, lines, CONFIRMATIONS)
    stored2 = sealed["mobility|per_task"].astype(np.int16)
    check2 = {
        "head": f"mobility|{rep2}|{dir2}|q{q2:g}|per_task",
        "bit_identical_to_sealed_array": bool(np.array_equal(rebuilt2, stored2)),
        "rows_differing_from_sealed": int((rebuilt2 != stored2).sum()),
        "rebuilt_tp": int(((rebuilt2 >= 0) & cohort.risk).sum()),
        "rebuilt_fp": int(((rebuilt2 >= 0) & ~cohort.risk).sum()),
        "published_tp": 272,
        "published_fp": 57,
        "truncate_at_alarm": truncation_test(
            values2, cohort.valid, dir2, lines, CONFIRMATIONS, rebuilt2, 0
        ),
        "truncate_one_chunk_before_alarm": falsification_test(
            values2, cohort.valid, dir2, lines, CONFIRMATIONS, rebuilt2
        ),
    }
    report["checks"].append(check2)
    print(json.dumps(check2, indent=1), flush=True)

    # ---- head 3: the longest look-back in the corpus, v4's L5 instability ---
    from analyze_front_back import (  # noqa: E402
        calibrated_external_alarm,
        mobility_representations,
        profile_task_map,
    )

    main_cache, extra_cache = mobs["development_main"], mobs["development_extra"]
    reference_map = profile_task_map([main_cache, extra_cache])
    reference_representations = {
        id(main_cache): mobility_representations(main_cache),
        id(extra_cache): mobility_representations(extra_cache),
    }
    ext_repr = mobility_representations(cohort.mobility_cache)
    l5 = calibrated_external_alarm(
        cohort.mobility_cache, ext_repr["L5"], reference_map,
        reference_representations, "L5", "high", 4, 8, 0.80,
    )
    l5_lines = np.full((cohort.n, 1), np.nan)
    for task in np.unique(cohort.task):
        take = np.flatnonzero(cohort.task == task)
        reference, reference_take = reference_map[task]
        ref_vals = reference_representations[id(reference)]["L5"]
        l5_lines[take, 0] = dev.quantile_higher(
            dev.row_max(oriented(ref_vals[reference_take], "high", 4)), 0.80
        )
    check3 = {
        "head": "L5|high|w4|k8|q0.80|per_task  (v4 instability branch, 8-chunk confirmation)",
        "rebuilt_tp": int(((l5 >= 0) & cohort.risk).sum()),
        "rebuilt_fp": int(((l5 >= 0) & ~cohort.risk).sum()),
        "published_tp": 34,
        "published_fp": 5,
        "truncate_at_alarm": truncation_test(
            ext_repr["L5"], cohort.valid, "high", l5_lines, 8, l5, 0
        ),
        "truncate_one_chunk_before_alarm": falsification_test(
            ext_repr["L5"], cohort.valid, "high", l5_lines, 8, l5
        ),
    }
    report["checks"].append(check3)
    print(json.dumps(check3, indent=1), flush=True)

    # ---- structural check across every collected array ----------------------
    report["structural"] = {
        "note": "no stored alarm may land on a chunk the episode never reached",
        "checked_in": "collect_methods.add()",
    }
    (RESULTS / "causality_check.json").write_text(json.dumps(report, indent=1))
    print("wrote results/causality_check.json", flush=True)


if __name__ == "__main__":
    main()
