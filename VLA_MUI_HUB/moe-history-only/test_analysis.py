import numpy as np
import pandas as pd

from analyze import auc, episode_metrics, landmark_metrics


def test_matched_landmark_removes_task_identity():
    frame = pd.DataFrame({
        "row": np.arange(8), "regime": "libero_natural", "run_id": "run",
        "suite": "suite", "source_run": ["a"]*4+["b"]*4,
        "task": ["a"]*4+["b"]*4, "init_state": 0, "length": 8,
        "success": [False, False, False, True, False, True, True, True],
    })
    scores = np.repeat(np.array([0.1]*4+[0.9]*4)[:,None], 32, axis=1)
    rows = landmark_metrics(frame, scores, 0.5, "task_only")
    assert rows and all(row["matched_auc"] == 0.5 for row in rows)
    assert all(row["pooled_auc"] > 0.5 for row in rows)
    assert all(row["query"] < 8 for row in rows)


def test_auc_ties_and_sign():
    labels = np.array([True, False, True, False])
    assert auc(labels, np.ones(4))[0] == 0.5
    assert auc(labels, labels.astype(float))[0] == 1
    assert auc(labels, -labels.astype(float))[0] == 0


def test_endpoint_censoring_and_recall_denominator():
    frame = pd.DataFrame({"success": [False, False, True, True], "length": [10]*4})
    result = episode_metrics(frame, np.array([4, -1, 5, 10]))
    assert result["tp"] == result["fp"] == 1
    assert result["recall"] == result["fpr"] == result["precision"] == 0.5
    assert result["recall_by_half"] == 0.5
    assert result["lead_median"] == 5
