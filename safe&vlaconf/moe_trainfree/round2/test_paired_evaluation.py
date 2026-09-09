import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("paired_evaluation", HERE / "evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_initial_state_difficulty_does_not_count_as_within_state_prediction():
    frame = pd.DataFrame({"task": ["task"] * 10, "init_state_id": [0] * 5 + [1] * 5,
                          "failure": [False] * 4 + [True] + [False] + [True] * 4})
    result = evaluation.metrics(frame, np.repeat([0.0, 1.0], 5))
    assert result["auc"] > 0.5
    assert result["within_init_macro_auc"] == 0.5


def test_completed_object_release_is_excluded_from_failed_goal_timing():
    record = {
        "goal_predicates": [{"id": "a", "expression": ["in", "placed", "goal"]},
                            {"id": "b", "expression": ["in", "dropped", "goal"]}],
        "goal_failure_labels": [{"goal_id": "a", "reason": "goal_satisfied_at_last_checkpoint"},
                                {"goal_id": "b", "reason": "object_released_outside_goal"}],
        "goal_subject_physics": {"placed": {"first_release_snapshot": 2},
                                 "dropped": {"first_release_snapshot": 13}},
    }
    assert evaluation.releases(record, "any_goal") == 2
    assert evaluation.releases(record, "failed_goal") == 13
    assert evaluation.releases(record, "drop_goal") == 13
