import argparse

import numpy as np
import pytest

from run_runtime_vector_grid import (
    QUERY_STATE_STRIDE,
    QUERY_TASK_STRIDE,
    expected_identity,
    parse_states,
    query_base,
)


def test_query_identity_is_unique_across_tasks_states_and_candidates():
    query, candidate = expected_identity((0, 2, 4), n_states=8, candidates=16)
    assert query.shape == candidate.shape == (3 * 8 * 16,)
    assert len(np.unique(query)) == 3 * 8
    for start in range(0, len(query), 16):
        np.testing.assert_array_equal(candidate[start : start + 16], np.arange(16))
        assert len(np.unique(query[start : start + 16])) == 1
    assert query_base(1, 0) - query_base(0, 0) == QUERY_TASK_STRIDE
    assert query_base(0, 1) - query_base(0, 0) == QUERY_STATE_STRIDE


def test_state_parser_rejects_duplicates_and_out_of_range():
    assert parse_states("1, 5,11") == (1, 5, 11)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_states("1,1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_states("50")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_states("")


def test_identity_builder_rejects_empty_axes():
    with pytest.raises(ValueError):
        expected_identity((0,), n_states=0, candidates=16)
    with pytest.raises(ValueError):
        expected_identity((0,), n_states=8, candidates=0)
    with pytest.raises(ValueError):
        query_base(-1, 0)
