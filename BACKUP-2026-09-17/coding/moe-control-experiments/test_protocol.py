import unittest

import numpy as np

from protocol import (PAIRS, candidate_noise, distances, fit_distance, medoid,
                      metrics, normalize_routes, parent_weights, pair_rms,
                      permute_state_blocks, predict_distance, split_for)


class ProtocolTests(unittest.TestCase):
    def test_noise_is_identity_stable(self):
        original = np.zeros((10, 24), np.float32)
        np.testing.assert_array_equal(candidate_noise("pro", 2, 8, 0, original), original)
        a = candidate_noise("pro", 2, 8, 1, original)
        np.testing.assert_array_equal(a, candidate_noise("pro", 2, 8, 1, original))
        self.assertFalse(np.array_equal(a, candidate_noise("plus", 2, 8, 1, original)))
        self.assertFalse(np.array_equal(a, candidate_noise("pro", 2, 8, 2, original)))

    def test_split_is_by_base_task(self):
        self.assertEqual([split_for(x) for x in (2, 5, 8)], ["holdout"] * 3)
        self.assertEqual(split_for(0), "train")

    def test_invalid_routes_fail(self):
        with self.assertRaises(ValueError):
            normalize_routes(np.zeros((8, 8, 10, 11, 32)))
        with self.assertRaises(ValueError):
            normalize_routes(np.ones((8, 8, 10, 11, 4)))
        p = np.ones((8, 8, 10, 11, 32))
        p[0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalize_routes(p)

    def test_site_identity_and_zero_distances(self):
        p = np.ones((8, 8, 10, 11, 32))
        ids = np.broadcast_to(np.arange(4), (8, 8, 10, 11, 4)).copy()
        result = distances(p, ids)
        self.assertEqual(result["structured"].shape, (28, 64))
        np.testing.assert_array_equal(result["structured"], 0)
        p[1, 0, :3, 0, 0] = 20
        result = distances(p, ids)
        self.assertGreater(result["structured"][0, 0], 0)
        np.testing.assert_array_equal(result["structured"][0, 1:], 0)
        ids[0, 0, 0, 0] = [0, 0, 1, 2]
        with self.assertRaises(ValueError):
            distances(p, ids)

    def test_old_center_matches_embedding(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(.01, 1, (8, 8, 10, 11, 32))
        ids = np.broadcast_to(np.arange(4), (8, 8, 10, 11, 4))
        result = distances(p, ids)
        root = np.sqrt(normalize_routes(p)[:, 4:, :3, 1:]).reshape(8, -1)
        expected = np.linalg.norm(root[PAIRS[:, 0]] - root[PAIRS[:, 1]], axis=1)
        expected /= np.sqrt(2 * 4 * 3 * 10)
        np.testing.assert_allclose(result["old_center"][:, 0], expected)

    def test_medoid_stable_ties(self):
        index, scores = medoid(np.ones(28))
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(scores, 1)
        with self.assertRaises(ValueError):
            medoid(np.full(28, -1))

    def test_parent_balance(self):
        weights = parent_weights(["a", "a", "b"])
        np.testing.assert_allclose(weights, [.25, .25, .5])
        self.assertEqual(metrics([0, 0, 0], [1, 1, 3], ["a", "a", "b"])["mae"], 2)

    def test_model_fit_and_frozen_scaling(self):
        x = np.arange(1, 9, dtype=float)[:, None]
        model = fit_distance(x, x[:, 0] * 2, ["a"] * 4 + ["b"] * 4)
        before = dict(model)
        a = predict_distance(model, np.array([[10.]]))[0]
        b = predict_distance(model, np.array([[10.], [1e9]]))[0]
        self.assertEqual(a, b)
        self.assertEqual(model, before)
        self.assertLess(abs(a - 20), .3)
        self.assertGreaterEqual(min(model["coefficient"]), 0)

    def test_pairs_and_block_permutation(self):
        self.assertEqual(len(set(map(tuple, PAIRS))), 28)
        d = pair_rms(np.arange(8)[:, None])
        self.assertEqual(d[0], 1)
        x = np.arange(4 * 28 * 2).reshape(4 * 28, 2)
        y = permute_state_blocks(x, 3).reshape(4, 28, 2)
        originals = {tuple(b.ravel()) for b in x.reshape(4, 28, 2)}
        self.assertEqual({tuple(b.ravel()) for b in y}, originals)


if __name__ == "__main__":
    unittest.main()
