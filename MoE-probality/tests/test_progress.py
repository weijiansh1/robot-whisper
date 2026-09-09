import inspect

import numpy as np
import pandas as pd
import pytest

from probability.progress_labels import episode_labels
from probability.progress_model import MoEProgressMonitor
from probability.progress_statistics import matched_auc


def physical(length=22, goals=2):
    return dict(goal=np.zeros((length, goals), bool), grasp=np.zeros((length, 1), bool),
                position=np.zeros((length, 1, 3)), distance=np.ones((length, 1)),
                subject_goal=np.array([[True] + [False]*(goals-1)]))


def labels(data, success=False, action_steps=220, max_steps=220):
    return episode_labels(**data, success=success, action_steps=action_steps, max_steps=max_steps)


def test_progress_requires_two_future_observations_and_ignores_after_horizon():
    data = physical()
    data["goal"][9, 0] = True
    assert not labels(data)["progress"][7]
    data["goal"][10, 0] = True
    assert labels(data)["progress"][7]
    assert labels(data)["first_event_chunks"][7] == 3
    data["goal"][9:11] = False
    data["goal"][12:14, 0] = True
    assert not labels(data)["progress"][7]
    data["goal"][11, 0] = True
    assert labels(data)["progress"][7]


def test_early_success_is_observed_positive_not_censored():
    result = labels(physical(9), success=True, action_steps=85)
    assert result["eligible"][7] and result["progress"][7]
    assert result["first_event_chunks"][7] == 1.5
    assert labels(physical(12), success=True, action_steps=120)["success_h5"][7]
    assert not labels(physical(13), success=True, action_steps=121)["success_h5"][7]


def test_schedule_excludes_exact_cap_endpoint_independently_of_outcome():
    for success, steps in ((False, 220), (True, 219)):
        result = labels(physical(), success=success, action_steps=steps)
        assert result["eligible"][16]
        assert not result["eligible"][17]
        assert not result["eligible"][:7].any()
    with pytest.raises(ValueError, match="incomplete"):
        labels(physical(10), action_steps=100)


def test_goal_count_is_net_and_grasp_lift_is_new_and_goal_relevant():
    data = physical(goals=3)
    data["goal"][:, 1] = True
    data["goal"][8:13, 1] = False
    data["goal"][8:13, 2] = True
    assert not labels(data)["progress"][7]
    data["grasp"][8:13] = True
    data["position"][8:13, 0, 2] = 0.03
    assert labels(data)["grasp_lift_h5"][7]
    data["grasp"][7] = True
    data["position"][7, 0, 2] = 0.03
    assert not labels(data)["grasp_lift_h5"][7]
    data["grasp"][7] = False
    data["goal"][8:13, 2] = False
    assert not labels(data)["grasp_lift_h5"][7]


def test_labels_support_articulation_tasks_without_mobile_subjects():
    data = dict(goal=np.zeros((22, 1), bool), grasp=np.zeros((22, 0), bool),
                position=np.zeros((22, 0, 3)), distance=np.zeros((22, 0)),
                subject_goal=np.zeros((0, 1), bool))
    result = labels(data)
    assert not result["progress"].any()
    assert np.isfinite(result["physical"]).all()


def test_progress_monitor_rejects_success_bundle_and_has_no_metadata_input():
    with pytest.raises(ValueError, match="physical-progress"):
        MoEProgressMonitor({})
    assert list(inspect.signature(MoEProgressMonitor.update).parameters) == ["self", "hb_router_probs"]


def test_current_restored_success_is_excluded_without_dropping_earlier_prefixes():
    data = physical()
    data["goal"][12] = True
    result = labels(data)
    assert result["eligible"][7]
    assert not result["eligible"][12]
    assert result["already_complete"][12]


def test_matched_auc_counts_only_same_stage_pairs_and_ties():
    frame = pd.DataFrame(dict(task=["t"]*6, query=[7]*6, stage=["a"]*4+["b"]*2,
                              cluster=["i0", "i0", "i1", "i1", "i0", "i1"],
                              progress=[1, 0, 1, 0, 1, 0],
                              moe=[0.9, 0.7, 0.5, 0.1, 0.2, 0.8], prior=[0.5]*6))
    scores, support, tasks = matched_auc(frame, ["moe", "prior"], ["task", "query", "stage"], frame, repeats=100)
    result = scores.set_index("model")
    assert result.loc["moe", "pairs"] == 5
    assert result.loc["moe", "matched_auroc"] == 3/5
    assert result.loc["prior", "matched_auroc"] == 0.5
    assert result.loc["prior", "low"] == result.loc["prior", "high"] == 0.5
    assert support.pairs.sum() == tasks[tasks.model == "moe"].pairs.sum() == 5
