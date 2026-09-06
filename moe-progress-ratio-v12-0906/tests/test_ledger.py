"""Unit tests for the complementarity ledger's pure decision functions.

The whole point of the ledger is the four-way bucket split, so the classifier is
pinned before it is ever pointed at a cohort.  The boundary cases carry the
interpretation: a gain of exactly ``lead - 1`` chunks must be called redundant
and a gain of exactly ``lead`` chunks must be called a real lead, otherwise
"these are mostly the same detector" and "this one is meaningfully earlier"
trade places for free.  Everything here is synthetic; no cohort is opened.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import build_complementarity_ledger as ledger  # noqa: E402


NEVER = -1


def bucket_of(base: int, candidate: int, lead: int = ledger.LEAD_CHUNKS) -> str:
    """The single bucket one (base, candidate) pair falls into."""
    masks = ledger.bucket_masks(
        np.asarray([base], dtype=np.int16), np.asarray([candidate], dtype=np.int16), lead
    )
    hit = [name for name, mask in masks.items() if bool(mask[0])]
    assert len(hit) == 1, f"({base}, {candidate}) landed in {hit}"
    return hit[0]


# --------------------------------------------------------------------------- #
# The lead boundary: exactly 1 chunk versus exactly 2 chunks
# --------------------------------------------------------------------------- #


def test_a_gain_of_exactly_one_chunk_is_redundant_not_earlier():
    assert bucket_of(base=11, candidate=10) == "redundant"


def test_a_gain_of_exactly_two_chunks_is_earlier():
    assert bucket_of(base=12, candidate=10) == "earlier"


def test_a_loss_of_exactly_one_chunk_is_redundant_not_later():
    assert bucket_of(base=10, candidate=11) == "redundant"


def test_a_loss_of_exactly_two_chunks_is_later():
    assert bucket_of(base=10, candidate=12) == "later"


def test_identical_first_alarms_are_redundant():
    assert bucket_of(base=7, candidate=7) == "redundant"


def test_a_large_gain_is_still_earlier():
    assert bucket_of(base=40, candidate=0) == "earlier"


@pytest.mark.parametrize("lead", [1, 2, 3, 5])
def test_the_boundary_moves_with_the_lead_parameter(lead):
    assert bucket_of(base=20, candidate=20 - lead, lead=lead) == "earlier"
    if lead > 1:
        assert bucket_of(base=20, candidate=20 - (lead - 1), lead=lead) == "redundant"


def test_lead_one_leaves_no_redundancy_band_except_ties():
    assert bucket_of(base=10, candidate=10, lead=1) == "redundant"
    assert bucket_of(base=10, candidate=9, lead=1) == "earlier"
    assert bucket_of(base=10, candidate=11, lead=1) == "later"


# --------------------------------------------------------------------------- #
# Never-alarmed (-1) on either side
# --------------------------------------------------------------------------- #


def test_candidate_only_when_the_base_never_alarms():
    assert bucket_of(base=NEVER, candidate=0) == "candidate_only"
    assert bucket_of(base=NEVER, candidate=51) == "candidate_only"


def test_base_only_when_the_candidate_never_alarms():
    assert bucket_of(base=0, candidate=NEVER) == "base_only"
    assert bucket_of(base=51, candidate=NEVER) == "base_only"


def test_neither_when_both_are_silent():
    assert bucket_of(base=NEVER, candidate=NEVER) == "neither"


def test_a_silent_base_is_never_counted_as_a_lead():
    """``-1`` must not be read as a numerically small, i.e. very early, chunk."""
    masks = ledger.bucket_masks(
        np.asarray([NEVER, NEVER], dtype=np.int16), np.asarray([0, 30], dtype=np.int16)
    )
    assert masks["earlier"].sum() == 0
    assert masks["later"].sum() == 0
    assert masks["redundant"].sum() == 0
    assert masks["candidate_only"].sum() == 2


def test_chunk_zero_is_a_real_alarm_not_a_missing_value():
    assert bucket_of(base=0, candidate=0) == "redundant"
    assert bucket_of(base=2, candidate=0) == "earlier"
    assert bucket_of(base=0, candidate=NEVER) == "base_only"


# --------------------------------------------------------------------------- #
# Structural guarantees the pair table relies on
# --------------------------------------------------------------------------- #


def test_the_buckets_partition_every_episode_exactly_once():
    rng = np.random.default_rng(0)
    base = rng.integers(-1, 52, size=5000).astype(np.int16)
    candidate = rng.integers(-1, 52, size=5000).astype(np.int16)
    masks = ledger.bucket_masks(base, candidate)
    stacked = np.stack([masks[name] for name in masks])
    assert np.array_equal(stacked.sum(axis=0), np.ones(len(base), dtype=int))
    assert sum(int(mask.sum()) for mask in masks.values()) == len(base)


def test_the_three_both_detect_buckets_sum_to_the_both_detect_count():
    rng = np.random.default_rng(1)
    base = rng.integers(-1, 52, size=5000).astype(np.int16)
    candidate = rng.integers(-1, 52, size=5000).astype(np.int16)
    masks = ledger.bucket_masks(base, candidate)
    both = (base >= 0) & (candidate >= 0)
    assert (
        int(masks["earlier"].sum())
        + int(masks["redundant"].sum())
        + int(masks["later"].sum())
    ) == int(both.sum())


def test_swapping_base_and_candidate_swaps_earlier_with_later():
    rng = np.random.default_rng(2)
    base = rng.integers(-1, 52, size=2000).astype(np.int16)
    candidate = rng.integers(-1, 52, size=2000).astype(np.int16)
    forward = ledger.bucket_masks(base, candidate)
    reverse = ledger.bucket_masks(candidate, base)
    assert np.array_equal(forward["earlier"], reverse["later"])
    assert np.array_equal(forward["redundant"], reverse["redundant"])
    assert np.array_equal(forward["candidate_only"], reverse["base_only"])
    assert np.array_equal(forward["neither"], reverse["neither"])


def test_a_channel_compared_with_itself_is_entirely_redundant_or_neither():
    rng = np.random.default_rng(3)
    first = rng.integers(-1, 52, size=2000).astype(np.int16)
    masks = ledger.bucket_masks(first, first)
    assert int(masks["candidate_only"].sum()) == 0
    assert int(masks["base_only"].sum()) == 0
    assert int(masks["earlier"].sum()) == 0
    assert int(masks["later"].sum()) == 0
    assert int(masks["redundant"].sum()) == int((first >= 0).sum())


def test_a_strict_superset_channel_never_produces_base_only():
    """``first_or(base, extra)`` can only ever tie or beat ``base``."""
    rng = np.random.default_rng(4)
    base = rng.integers(-1, 52, size=2000).astype(np.int16)
    extra = rng.integers(-1, 52, size=2000).astype(np.int16)
    union = ledger.first_or(base, extra)
    masks = ledger.bucket_masks(base, union)
    assert int(masks["base_only"].sum()) == 0
    assert int(masks["later"].sum()) == 0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_misaligned_arrays_are_rejected():
    with pytest.raises(ValueError):
        ledger.bucket_masks(np.zeros(3, np.int16), np.zeros(4, np.int16))


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        ledger.bucket_masks(np.zeros((2, 3), np.int16), np.zeros((2, 3), np.int16))


def test_a_non_positive_lead_is_rejected():
    with pytest.raises(ValueError):
        ledger.bucket_masks(np.zeros(3, np.int16), np.zeros(3, np.int16), lead=0)


# --------------------------------------------------------------------------- #
# first_or
# --------------------------------------------------------------------------- #


def test_first_or_takes_the_earlier_alarm_and_keeps_minus_one_only_when_both_are_silent():
    left = np.asarray([-1, -1, 5, 9], dtype=np.int16)
    right = np.asarray([-1, 3, 9, 5], dtype=np.int16)
    assert ledger.first_or(left, right).tolist() == [-1, 3, 5, 5]


def test_first_or_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        ledger.first_or(np.zeros(3, np.int16), np.zeros(4, np.int16))


# --------------------------------------------------------------------------- #
# structurally_nested
# --------------------------------------------------------------------------- #


def test_a_union_and_its_own_branches_are_flagged_as_nested_in_both_directions():
    assert ledger.structurally_nested("v7_guard", "v7_freeze")
    assert ledger.structurally_nested("v7_freeze", "v7_guard")
    assert ledger.structurally_nested("v7_guard", "v7_turbulence")
    assert ledger.structurally_nested("v7_guard_or_progress_ratio", "progress_ratio")
    assert ledger.structurally_nested("v7_freeze", "v7_guard_or_progress_ratio")


def test_independent_channel_pairs_are_not_flagged_as_nested():
    assert not ledger.structurally_nested("v4", "v7_freeze")
    assert not ledger.structurally_nested("v4", "v7_guard")
    assert not ledger.structurally_nested("v4", "progress_ratio")
    assert not ledger.structurally_nested("v7_freeze", "v7_turbulence")
    assert not ledger.structurally_nested("v7_guard", "progress_ratio")


def test_every_nested_pair_really_is_a_superset_on_synthetic_alarms():
    """The flag must describe the arithmetic, not just a naming convention."""
    rng = np.random.default_rng(7)
    parts = {
        "v7_freeze": rng.integers(-1, 52, size=3000).astype(np.int16),
        "v7_turbulence": rng.integers(-1, 52, size=3000).astype(np.int16),
        "progress_ratio": rng.integers(-1, 52, size=3000).astype(np.int16),
    }
    parts["v7_guard"] = ledger.first_or(parts["v7_freeze"], parts["v7_turbulence"])
    parts["v7_guard_or_progress_ratio"] = ledger.first_or(
        parts["v7_guard"], parts["progress_ratio"]
    )
    for union, components in ledger.NESTED_COMPONENTS.items():
        for component in components:
            masks = ledger.bucket_masks(parts[component], parts[union])
            assert int(masks["base_only"].sum()) == 0
            assert int(masks["later"].sum()) == 0


# --------------------------------------------------------------------------- #
# clustered_interval
# --------------------------------------------------------------------------- #


def test_the_clustered_interval_brackets_a_constant_rate_exactly():
    """Every task has the same rate, so every resample reproduces it."""
    task_code = np.repeat(np.arange(10), 20)
    numerator = np.tile(np.r_[np.ones(5, bool), np.zeros(15, bool)], 10)
    denominator = np.ones(200, dtype=bool)
    draws = np.random.default_rng(5).integers(0, 10, size=(500, 10))
    low, high = ledger.clustered_interval(task_code, 10, numerator, denominator, draws)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_the_clustered_interval_is_nan_when_the_denominator_is_empty():
    task_code = np.zeros(10, dtype=int)
    empty = np.zeros(10, dtype=bool)
    draws = np.zeros((10, 1), dtype=int)
    low, high = ledger.clustered_interval(task_code, 1, empty, empty, draws)
    assert np.isnan(low) and np.isnan(high)


def test_the_clustered_interval_widens_when_the_rate_is_task_dependent():
    """Half the tasks are all-positive, half all-negative: resampling must vary."""
    task_code = np.repeat(np.arange(10), 20)
    numerator = np.repeat(np.r_[np.ones(5, bool), np.zeros(5, bool)], 20)
    denominator = np.ones(200, dtype=bool)
    draws = np.random.default_rng(6).integers(0, 10, size=(2000, 10))
    low, high = ledger.clustered_interval(task_code, 10, numerator, denominator, draws)
    assert 0.0 <= low < 0.5 < high <= 1.0


def test_the_clustered_interval_rejects_misaligned_indicators():
    with pytest.raises(ValueError):
        ledger.clustered_interval(
            np.zeros(4, int), 1, np.zeros(3, bool), np.zeros(4, bool),
            np.zeros((2, 1), int),
        )
