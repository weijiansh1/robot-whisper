import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics import subject_release  # noqa: E402


def test_release_uses_failed_goal_subject_not_earlier_successful_object():
    record = {
        "goal_predicates": [{"id": "a", "expression": ["in", "first", "target"]},
                            {"id": "b", "expression": ["in", "second", "target"]}],
        "goal_failure_labels": [{"goal_id": "a", "reason": "goal_satisfied_at_last_checkpoint"},
                                {"goal_id": "b", "reason": "object_released_outside_goal"}],
        "goal_subject_physics": {"first": {"first_release_snapshot": 3},
                                 "second": {"first_release_snapshot": 14}},
    }
    assert subject_release(record) == 14
    record["goal_subject_physics"]["second"]["first_release_snapshot"] = None
    assert subject_release(record) is None
