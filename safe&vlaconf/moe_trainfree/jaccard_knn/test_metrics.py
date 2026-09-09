"""Check optimized routing distances against direct set and min/max formulas."""

import numpy as np
import pandas as pd
import pytest

try:
    import numba
except ImportError as error:
    pytest.skip(f"routing kNN needs a compatible Numba/NumPy environment: {error}", allow_module_level=True)

from metrics import METHODS, JaccardScorer, calibrate, distance_matrix, prepare


def samples(count, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((count, 80, 32), dtype=np.float32)
    ids = np.argsort(p, axis=-1)[..., -4:].astype(np.uint8)
    return p, ids


def oracle(p, ids, bank, bank_ids):
    p, root, _, _ = prepare(p, ids)
    bank, bank_root, _, _ = prepare(bank, bank_ids)
    output = np.empty((len(p), len(bank), 4))
    for i in range(len(p)):
        for j in range(len(bank)):
            inter = np.array([len(set(ids[i,s]) & set(bank_ids[j,s])) for s in range(80)])
            low = np.minimum(p[i].astype(float), bank[j].astype(float)).sum(-1)
            high = np.maximum(p[i].astype(float), bank[j].astype(float)).sum(-1)
            output[i,j] = [np.mean(1-inter/(8-inter)), np.mean(1-low/high),
                            1-low.sum()/high.sum(), np.sqrt(np.square(root[i].astype(float)-bank_root[j]).sum()/160)]
    return output


def test_distances_and_exact_neighbors_match_independent_formulas():
    p, ids = samples(4, 73)
    bank, bank_ids = samples(31, 16)
    expected = oracle(p, ids, bank, bank_ids)
    actual = distance_matrix(*prepare(p, ids), *prepare(bank, bank_ids))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    expected_score = np.sort(expected, axis=1)[:, :20].mean(1).T
    np.testing.assert_allclose(JaccardScorer(bank, bank_ids, batch_size=3).score(p, ids), expected_score, rtol=2e-6, atol=2e-7)


def test_identity_disjoint_sets_and_expert_31():
    p = np.zeros((2,80,32),np.float32)
    p[0,:,:4], p[1,:,28:] = .25,.25
    ids = np.broadcast_to(np.array([[0,1,2,3],[28,29,30,31]],np.uint8)[:,None],(2,80,4)).copy()
    distance = distance_matrix(*prepare(p,ids),*prepare(p,ids))
    np.testing.assert_allclose(distance[0,0],0,atol=1e-7)
    np.testing.assert_allclose(distance[0,1],1,atol=1e-7)
    np.testing.assert_allclose(distance[1,0],distance[0,1],atol=1e-7)


def test_position_alignment_is_not_token_pooling():
    p, ids = samples(1,9)
    q, other = p[:,::-1].copy(), ids[:,::-1].copy()
    np.testing.assert_allclose(p.mean(1),q.mean(1),rtol=2e-6)
    assert (distance_matrix(*prepare(p,ids),*prepare(q,other)) > .1).all()


def test_failed_calibration_trajectories_do_not_change_threshold():
    rng=np.random.default_rng(11)
    scores=rng.random((len(METHODS),100,12)).astype(np.float32)
    labels=np.repeat([0,1],[90,10])
    frame=pd.DataFrame({"task":["a"]*100,"init_state_id":np.arange(100)})
    first,_,_=calibrate(scores,labels,frame,scores[:,:3])
    scores[:,labels==1] += 100
    second,_,_=calibrate(scores,labels,frame,scores[:,:3])
    np.testing.assert_array_equal(first,second)


def test_invalid_probabilities_and_duplicate_ids_are_rejected():
    p,ids=samples(1,17)
    p[0,0,0]=-1
    with pytest.raises(ValueError,match="probabilities"):
        prepare(p,ids)
    p[0,0,0]=1
    ids[0,0,0]=ids[0,0,1]
    with pytest.raises(ValueError,match="distinct"):
        prepare(p,ids)
