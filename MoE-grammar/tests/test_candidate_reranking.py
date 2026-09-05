import numpy as np
from pathlib import Path

from moe_grammar.candidate_reranking import (
    CandidateTrace,
    RouteArchive,
    RouteStore,
    cluster_bootstrap_mean,
    selector_indexes,
    stratified_pair_auc,
)


def test_candidate_selectors_keep_grammar_and_medoid_semantics_separate() -> None:
    surprise = np.asarray([2.0, 0.5, 1.0])
    actions = np.asarray([[[0.0]], [[1.0]], [[2.0]]])
    routing = np.asarray([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
    selected = selector_indexes(surprise, actions, routing, diversity_beta=0.0)
    assert selected["action_medoid"] == 1
    assert selected["grammar"] == 1
    assert selected["anti_grammar"] == 0
    assert selected["grammar_action_diverse"] == selected["grammar"]
    assert selected["grammar_route_diverse"] == selected["grammar"]


def test_stratified_auc_never_compares_different_snapshots() -> None:
    score = np.asarray([0.0, 1.0, 100.0, 99.0])
    success = np.asarray([True, False, True, False])
    group = np.asarray([0, 0, 1, 1])
    assert stratified_pair_auc(score, success, group) == 0.5


def test_cluster_bootstrap_resamples_whole_trunks() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    cluster = np.asarray(["a", "a", "b", "b"])
    low, high = cluster_bootstrap_mean(values, cluster, draws=2000, seed=7)
    assert low == 0.0
    assert high == 1.0


def test_route_matching_allows_concurrent_episode_rows() -> None:
    expert_ids = np.arange(5 * 8 * 10 * 11 * 4, dtype=np.int64).reshape(5, 8, 10, 11, 4)
    expert_ids = np.asarray(expert_ids % 32, dtype=np.uint8)
    store = RouteStore(
        path=Path("routes.zarr"),
        group=None,
        episode_id=np.asarray([7, 99, 7, 99, 99], dtype=np.int32),
        expert_ids=expert_ids,
    )
    route_ids = store.response_ids_rows([0, 2])
    trace = CandidateTrace(
        arm="triggered",
        candidate=0,
        path=Path("candidate_0.npz"),
        episode_id=7,
        success=False,
        fork_query=3,
        budget=2,
        route_ids=route_ids,
    )
    assert RouteArchive._trace_matches(store, trace) == [(0, 2)]
