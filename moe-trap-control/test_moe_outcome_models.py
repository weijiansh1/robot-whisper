import copy
import unittest

import numpy as np

from moe_outcome_models import FeatureHistory, MODELS, OPERATORS, choose, derangement
from fit_moe_outcomes import fit


class OutcomeModelTests(unittest.TestCase):
    def test_probe_preserves_executed_history(self):
        rng = np.random.default_rng(11)
        history = FeatureHistory()
        for q in range(12):
            p = rng.random((8,10,11,32)).astype(np.float32)
            actual = history.update(p,10*q)
            self.assertEqual(actual.shape,(77,))
            self.assertTrue(np.isfinite(actual).all())
        before = copy.deepcopy(history)
        candidate = history.probe(p,120)
        self.assertEqual(history.monitor.v7.query,before.monitor.v7.query)
        np.testing.assert_array_equal(history.previous_root,before.previous_root)
        np.testing.assert_array_equal(candidate,before.update(p,120))

    def test_matched_shuffle_is_a_derangement(self):
        parents = [str(i) for i in range(21)]
        for method in MODELS+("clock_knn",):
            for repeat in (0,1):
                mapping = derangement(parents,method,repeat)
                self.assertEqual(set(mapping),set(mapping.values()))
                self.assertTrue(all(left != right for left,right in mapping.items()))
                self.assertEqual(mapping,derangement(parents,method,repeat))

    def test_fit_respects_parent_exclusion_and_valid_probabilities(self):
        rng = np.random.default_rng(19)
        n,frames = 12,8
        data = dict(frame_x=rng.normal(size=(n*frames,77)),frame_parent=np.repeat(np.arange(n),frames),
            frame_y=np.repeat(np.arange(n)%2,frames),event_x=rng.normal(size=(n,77)),
            event_parent=np.arange(n),outcomes=np.zeros((n,len(OPERATORS),2)),
            responses=rng.normal(size=(n,len(OPERATORS),2,77)),
            response_valid=np.ones((n,len(OPERATORS),2),bool),terminal=np.zeros((n,len(OPERATORS),2),int))
        data["outcomes"][:,0,:] = (np.arange(n)%2)[:,None]
        data["outcomes"][:,1:,:] = data["outcomes"][:,:1,:]
        model = fit(data,np.arange(10))
        self.assertNotIn(10,model["event_parent_indices"])
        self.assertNotIn(11,model["event_parent_indices"])
        np.testing.assert_allclose(np.asarray(model["transition"]).sum(1),1.)
        self.assertTrue(np.all((np.asarray(model["committor"]) >= 0)&(np.asarray(model["committor"]) <= 1)))
        for method in MODELS+("clock_knn",):
            operator,scores = choose(model,method,data["event_x"][10])
            self.assertIn(operator,OPERATORS)
            self.assertTrue(np.isfinite(scores).all())
            self.assertEqual(scores[0],0.)
            if method in ("knn_uplift","ridge_uplift","clock_knn"):
                self.assertEqual(operator,"resample")


if __name__ == "__main__":
    unittest.main()
