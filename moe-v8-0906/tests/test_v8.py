"""Invariants that must hold for the v8 guard.

These guard the claims in `method/ONLINE_INTRINSIC_GUARD_V8_PROTOCOL.md`.
Anything that would silently weaken the protocol - a strict `>` creeping back
in, the horizon cap leaking into a head, a threshold drifting - fails here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "moe-v8-0906"
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V8  # noqa: E402

SUMMARY = json.loads((BUNDLE / "results" / "v8_full_corpus_summary.json").read_text())
ALARMS = np.load(BUNDLE / "results" / "v8_full_corpus_alarms.npz")

V7_ANCHOR = {"development_main": (303, 38), "external_8b": (347, 57),
             "legacy_main16x32": (220, 16)}
V8_EXPECTED = {"development_main": (331, 44), "external_8b": (377, 66),
               "legacy_main16x32": (224, 16)}


@pytest.fixture(scope="module")
def cohorts():
    return {n: V8.load_cohort(n) for n in V7_ANCHOR}


def test_v7_anchors_reproduce(cohorts):
    """v8 must not disturb v7. If these move, the comparison is meaningless."""
    for name, (tp, fp) in V7_ANCHOR.items():
        d = cohorts[name]
        got = V8.score(d["v7"], d["risk"], d["length"])
        assert (got["tp"], got["fp"]) == (tp, fp), (name, got)


def test_v8_full_corpus_totals(cohorts):
    total_tp = total_fp = total_risk = 0
    for name, (tp, fp) in V8_EXPECTED.items():
        d = cohorts[name]
        got = V8.score(ALARMS[f"{name}|v8"], d["risk"], d["length"])
        assert (got["tp"], got["fp"]) == (tp, fp), (name, got)
        total_tp += got["tp"]
        total_fp += got["fp"]
        total_risk += int(d["risk"].sum())
    assert (total_tp, total_fp, total_risk) == (932, 126, 1358)
    assert SUMMARY["v8"] == {"tp": 932, "fp": 126}
    assert SUMMARY["risks"] == 1358 and SUMMARY["episodes"] == 32960


def test_v8_contains_v7(cohorts):
    """v8 is a union with v7, so it can never alarm later or catch less."""
    for name in V7_ANCHOR:
        v7, v8 = ALARMS[f"{name}|v7"], ALARMS[f"{name}|v8"]
        fired7, fired8 = v7 >= 0, v8 >= 0
        assert (fired7 <= fired8).all()
        assert (v8[fired7] <= v7[fired7]).all()


def test_thresholds_match_protocol():
    assert SUMMARY["quantile"] == 0.995 and SUMMARY["confirm"] == 2
    assert SUMMARY["thresholds"]["frontback_flowpath"] == pytest.approx(0.0, abs=1e-9)
    assert SUMMARY["thresholds"]["curvature_3step"] == pytest.approx(0.389, abs=5e-4)


def test_tie_group_fires():
    """`>=`, never `>`. A value exactly at the threshold must alarm; under
    strict `>` the whole tie group at the threshold is silently dropped."""
    score = np.full((3, 12), np.nan)
    score[0, 6:] = 1.0          # exactly at the threshold
    score[1, 6:] = 1.5          # above
    score[2, 6:] = 0.5          # below
    first = V8.confirmed_first(score, 1.0, "high", confirm=2)
    assert first[0] == 7, "value equal to the threshold must fire"
    assert first[1] == 7
    assert first[2] == -1


def test_confirmation_requires_consecutive():
    """K=2 means two *consecutive* crossings; alternating must not fire."""
    score = np.full((2, 14), 0.0)
    score[0, 6::2] = 2.0        # every other chunk - never two in a row
    score[1, 8:10] = 2.0        # two consecutive
    first = V8.confirmed_first(score, 1.0, "high", confirm=2)
    assert first[0] == -1
    assert first[1] == 9


def test_earliest_alarm_is_q6():
    """The width-6 causal mean makes q6 the earliest admissible chunk."""
    score = np.full((1, 20), 5.0)
    first = V8.confirmed_first(score, 1.0, "high", confirm=1)
    assert first[0] == V8.WIDTH == 6


def test_heads_use_only_the_flow_speed_array(cohorts):
    """The entire v8 addition is a function of one [n, chunk, 8, 9] array."""
    d = cohorts["external_8b"]
    assert d["speed"].ndim == 4 and d["speed"].shape[2:] == (8, 9)
    heads = V8.heads_from_flow_speed(d["speed"])
    assert set(heads) == {"frontback_flowpath", "curvature_3step"}
    for series in heads.values():
        assert series.shape == d["speed"].shape[:2]


def test_no_horizon_and_no_task_identity():
    """No cap, no suite table, no per-task threshold anywhere in the head code."""
    source = (BUNDLE / "experiments" / "evaluate_full_corpus.py").read_text()
    head_block = source[source.index("def heads_from_flow_speed"):
                        source.index("def confirmed_first")]
    for banned in ("cap", "CAPS", "per_task", "task_index", "0.65", "phase"):
        assert banned not in head_block, banned


def test_curvature_reads_three_steps():
    assert V8.CURVATURE_IDX == (0, 4, 8)          # stored steps 1, 5, 9
    assert len(V8.CURVATURE_IDX) == 3


def test_null_control_is_far_worse():
    """Shuffling the new heads within each chunk must cost far more alarms."""
    null, v8 = SUMMARY["null"], SUMMARY["v8"]
    assert null["fp"] > 20 * v8["fp"]
    assert null["fp"] == 4418 and null["tp"] == 1210


def test_frontback_head_is_not_self_baselined():
    """The ratio is already cross-layer normalised; self-baselining destroys it
    (+169 TP / +1417 FP versus +93 TP / +40 FP).  Guard the exception."""
    source = (BUNDLE / "experiments" / "evaluate_full_corpus.py").read_text()
    block = source[source.index("def heads_from_flow_speed"):
                   source.index("def confirmed_first")]
    ratio_part = block[:block.index("sub = speed")]
    assert "BASELINE" not in ratio_part
    assert "BASELINE" in block[block.index("sub = speed"):]


# ---------------------------------------------------------------- v8.2 ------
V82_SUMMARY = json.loads((BUNDLE / "results" / "v82_summary.json").read_text())
V82_ALARMS = np.load(BUNDLE / "results" / "v82_alarms.npz")
V82_EXPECTED = {"development_main": (345, 47), "external_8b": (396, 73),
                "legacy_main16x32": (225, 17)}


def test_v82_totals(cohorts):
    total_tp = total_fp = 0
    for name, (tp, fp) in V82_EXPECTED.items():
        got = V8.score(V82_ALARMS[f"{name}|v8.2"], cohorts[name]["risk"],
                       cohorts[name]["length"])
        assert (got["tp"], got["fp"]) == (tp, fp), (name, got)
        total_tp += got["tp"]
        total_fp += got["fp"]
    assert (total_tp, total_fp) == (966, 137)
    assert V82_SUMMARY["profile"]["4"] == [966, 137]


def test_v82_dominates_v8_at_every_lead():
    """The selection required this; if it ever fails, aggregate recall is being
    bought by giving up early detections."""
    for lead, (tp, _) in V82_SUMMARY["profile"].items():
        v8_tp, _ = V82_SUMMARY["v8_profile"][lead]
        assert tp >= v8_tp, (lead, tp, v8_tp)


def test_v82_contains_v7(cohorts):
    for name in V82_EXPECTED:
        v7, v82 = ALARMS[f"{name}|v7"], V82_ALARMS[f"{name}|v8.2"]
        fired7 = v7 >= 0
        assert (fired7 <= (v82 >= 0)).all()
        assert (v82[fired7] <= v7[fired7]).all()


def test_v82_config_matches_protocol():
    cfg = V82_SUMMARY["config"]
    assert cfg == {"baseline": 4, "width": 6, "confirm": 2, "slope": -0.0015}
    assert V82_SUMMARY["earliest_chunk"] == 6
    assert V82_SUMMARY["curvature_steps"] == [0, 4, 8]      # stored steps 1,5,9


def test_v82_slope_relaxes_both_directions():
    """A negative slope must make BOTH heads more permissive over time, not one
    of each - the sign flip for `low` heads is what guarantees that."""
    import freeze_v82 as F
    high = np.full((1, 30), 0.5)
    low = np.full((1, 30), -0.5)
    late_high = F.moving_first(high, 0.6, "high", -0.02, 1, 0)
    late_low = F.moving_first(low, -0.6, "low", -0.02, 1, 0)
    assert late_high[0] > 0 and late_low[0] > 0, (late_high, late_low)
    assert F.moving_first(high, 0.6, "high", 0.0, 1, 0)[0] == -1
    assert F.moving_first(low, -0.6, "low", 0.0, 1, 0)[0] == -1


def test_v82_null_needs_far_more_false_alarms():
    null = V82_SUMMARY["null"]
    assert null["fp"] == 4565 and null["tp"] == 1225
    assert null["fp"] > 30 * V82_SUMMARY["profile"]["4"][1]
