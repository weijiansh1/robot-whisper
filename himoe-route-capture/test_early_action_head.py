from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_early_action_head import (
    block_krr_cv_residuals,
    center_scale_features,
    common_random_subsets,
    exact_random_coverage,
    exact_stratified_random_coverage,
    pairwise_rms,
    pam,
    seed_disjoint_predictions,
    vote_metrics,
)


class EarlyActionHeadTest(unittest.TestCase):
    def test_block_krr_identity_matches_explicit_refit(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(12, 5))
        y = rng.normal(size=(12, 3))
        kernel = x @ x.T / x.shape[1]
        alpha = 0.1
        valid = np.asarray([1, 4, 8])
        train = np.setdiff1d(np.arange(len(x)), valid)
        explicit = y[valid] - kernel[np.ix_(valid, train)] @ np.linalg.solve(
            kernel[np.ix_(train, train)] + alpha * np.eye(len(train)),
            y[train],
        )
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        identity = block_krr_cv_residuals(
            eigenvalues,
            eigenvectors,
            eigenvectors.T @ y,
            valid,
            alpha,
        )
        np.testing.assert_allclose(identity, explicit, atol=1e-10, rtol=1e-10)

    def test_exact_random_coverage_matches_enumeration(self) -> None:
        points = np.random.default_rng(3).normal(size=(6, 4))
        distance = pairwise_rms(points)
        expected = np.mean(
            [
                np.min(distance[:, subset], axis=1).mean()
                for subset in itertools.combinations(range(6), 2)
            ]
        )
        self.assertAlmostEqual(exact_random_coverage(distance, 2), expected, places=12)

    def test_exact_stratified_coverage_matches_enumeration(self) -> None:
        points = np.random.default_rng(5).normal(size=(6, 3))
        distance = pairwise_rms(points)
        folds = [np.asarray([0, 1, 2]), np.asarray([3, 4, 5])]
        subsets = [
            left + right
            for left in itertools.combinations(folds[0], 1)
            for right in itertools.combinations(folds[1], 1)
        ]
        expected = np.mean(
            [np.min(distance[:, subset], axis=1).mean() for subset in subsets]
        )
        self.assertAlmostEqual(
            exact_stratified_random_coverage(distance, folds, 1),
            expected,
            places=12,
        )

    def test_pam_is_unique_deterministic_and_swap_stable(self) -> None:
        points = np.random.default_rng(7).normal(size=(32, 5))
        distance = pairwise_rms(points)
        selected = pam(distance, 8)
        np.testing.assert_array_equal(selected, pam(distance, 8))
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(np.unique(selected)), 8)
        base = np.min(distance[:, selected], axis=1).sum()
        for old in selected:
            for candidate in np.setdiff1d(np.arange(32), selected):
                proposed = selected.copy()
                proposed[np.flatnonzero(proposed == old)[0]] = candidate
                self.assertLessEqual(
                    base,
                    np.min(distance[:, proposed], axis=1).sum() + 1e-12,
                )

    def test_vote_and_common_random_subset_contract(self) -> None:
        points = np.arange(32, dtype=np.float64)[:, None]
        distance = pairwise_rms(points)
        result = vote_metrics(distance, np.asarray([12, 13, 14, 15, 16, 17, 18, 19]))
        self.assertEqual(result["voted_candidate"], 15)
        self.assertEqual(result["global_medoid_candidate"], 15)
        self.assertEqual(result["exact_global_medoid_reproduction"], 1.0)
        self.assertEqual(result["normalized_global_medoid_regret"], 0.0)

        folds = [np.arange(32)[fold::4] for fold in range(4)]
        first = common_random_subsets(folds)
        second = common_random_subsets(folds)
        np.testing.assert_array_equal(first["uniform"], second["uniform"])
        np.testing.assert_array_equal(first["stratified"], second["stratified"])
        self.assertTrue(np.all(np.diff(first["uniform"], axis=1) > 0))
        for fold in folds:
            self.assertTrue(
                np.all(np.isin(first["stratified"], fold).sum(axis=1) == 2)
            )

    def test_held_seed_targets_do_not_enter_their_prediction_geometry(self) -> None:
        rng = np.random.default_rng(11)
        features = center_scale_features(rng.normal(size=(4, 32, 5)))
        targets = rng.normal(size=(4, 32, 3))
        folds = [np.arange(32)[fold::4] for fold in range(4)]
        original, _ = seed_disjoint_predictions(features, targets, folds, 2)

        changed = targets.copy()
        changed[:, folds[0]] += rng.normal(size=changed[:, folds[0]].shape) * 1000.0
        perturbed, _ = seed_disjoint_predictions(features, changed, folds, 2)

        for group in range(4):
            np.testing.assert_allclose(
                pairwise_rms(original[group, folds[0]]),
                pairwise_rms(perturbed[group, folds[0]]),
                atol=1e-9,
                rtol=1e-9,
            )


if __name__ == "__main__":
    unittest.main()
