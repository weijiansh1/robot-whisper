from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_moe_connectivity_graph import (
    auk8,
    balanced_mst_cut,
    common_seed_permute,
    hellinger_distance,
    pairwise_rms,
    resolve_normalization_stats_path,
    tie_aware_auc,
)


def test_hellinger_distance_has_known_endpoints() -> None:
    # [candidate, aligned token site, expert]
    probabilities = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            # Per-site normalization makes this identical to candidate 2.
            [[2.0, 0.0], [0.0, 3.0]],
        ]
    )
    distance = hellinger_distance(probabilities)

    expected = np.asarray(
        [
            [0.0, 1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(distance, expected, atol=1e-12, rtol=0.0)

    with pytest.raises(ValueError, match="positive probability mass"):
        hellinger_distance(np.zeros((2, 3)))


def test_pairwise_rms_uses_all_non_candidate_dimensions() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    distance = pairwise_rms(values)

    assert np.isclose(distance[0, 1], np.sqrt(2.0))
    assert np.isclose(distance[0, 2], np.sqrt(2.0))
    assert np.isclose(distance[1, 2], 2.0)
    np.testing.assert_allclose(distance, distance.T, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(np.diag(distance), 0.0)


def test_auk8_is_perfect_for_identical_geometry_and_near_chance_when_random() -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=(96, 8))
    slightly_noisy = source + 0.03 * rng.normal(size=source.shape)
    unrelated = rng.normal(size=source.shape)

    source_distance = pairwise_rms(source)
    perfect = auk8(source_distance, source_distance, max_k=8)
    noisy = auk8(source_distance, pairwise_rms(slightly_noisy), max_k=8)
    random = auk8(source_distance, pairwise_rms(unrelated), max_k=8)

    # Even perfect geometry averages finite neighbor ranks k=1,...,8. With
    # u=(rank-1)/(n-2), its exact gain is 1-(max_k-1)/(2*(n-2)).
    expected_perfect = 1.0 - 7.0 / (2.0 * (len(source) - 2))
    assert np.isclose(perfect, expected_perfect, atol=1e-12, rtol=0.0)
    assert perfect > noisy > 0.5
    assert abs(random) < 0.20

    # Every peer tied in the target geometry has normalized midrank 0.5.
    assert np.isclose(auk8(source_distance, np.zeros_like(source_distance)), 0.0)


def test_tie_aware_auc_gives_half_credit_for_equal_scores() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.0, 1.0, 1.0, 2.0])

    # Positive-negative comparisons contain three wins and one tie.
    assert np.isclose(tie_aware_auc(labels, scores), 0.875)
    assert np.isclose(tie_aware_auc(labels, np.ones_like(scores)), 0.5)
    assert np.isclose(tie_aware_auc(labels, labels), 1.0)
    assert np.isclose(tie_aware_auc(labels, 1 - labels), 0.0)
    with pytest.raises(ValueError, match="both label classes"):
        tie_aware_auc(np.ones(4), np.arange(4))
    with pytest.raises(ValueError, match="non-finite"):
        tie_aware_auc(labels, np.asarray([0.0, 1.0, np.nan, 2.0]))


def test_balanced_mst_cut_rejects_the_largest_unbalanced_edge() -> None:
    # The largest MST edge isolates the first two points, which min_size forbids.
    # With eight points and min_size four, the only valid cut is four versus four.
    points = np.asarray([0.0, 1.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    labels = np.asarray(balanced_mst_cut(pairwise_rms(points[:, None]), min_size=4))

    unique, counts = np.unique(labels, return_counts=True)
    np.testing.assert_array_equal(np.sort(counts), np.asarray([4, 4]))
    assert len(unique) == 2
    assert np.all(labels[:4] == labels[0])
    assert np.all(labels[4:] == labels[4])
    assert labels[0] != labels[4]


def test_common_seed_permutation_is_shared_across_aligned_arrays() -> None:
    first = np.arange(2 * 4, dtype=np.int64).reshape(2, 4)
    second = (100 + np.arange(2 * 4 * 3, dtype=np.int64)).reshape(2, 4, 3)
    originals = [first.copy(), second.copy()]
    permutation = np.asarray([2, 0, 3, 1])

    permuted = common_seed_permute([first, second], permutation)

    np.testing.assert_array_equal(permuted[0], first[:, permutation])
    np.testing.assert_array_equal(permuted[1], second[:, permutation, :])
    np.testing.assert_array_equal(first, originals[0])
    np.testing.assert_array_equal(second, originals[1])

    with pytest.raises(ValueError, match="every seed-column index once"):
        common_seed_permute([first], np.asarray([0, 0, 2, 3]))


def test_normalization_stats_uses_hash_checked_bundle_fallback(tmp_path: Path) -> None:
    metadata = tmp_path / "server_metadata.json"
    metadata.write_text("{}")
    portable = tmp_path / "normalization_stats.json"
    portable.write_text('{"actions": {"std": [1, 1, 1, 1, 1, 1, 1]}}')

    assert resolve_normalization_stats_path(metadata, "/definitely/missing/stats.json") == portable
    recorded = tmp_path / "recorded.json"
    recorded.write_text("{}")
    assert resolve_normalization_stats_path(metadata, str(recorded)) == recorded
