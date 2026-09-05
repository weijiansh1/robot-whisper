import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_conformal_protocol_has_disjoint_task_split():
    summary = json.loads((ROOT / "results/route_9/summary.json").read_text())
    calibration = set(summary["calibration_tasks"])
    test = set(summary["heldout_test_tasks"])
    assert calibration
    assert test
    assert calibration.isdisjoint(test)
    assert summary["semantics"].startswith("risk-controlled alarm")


def test_static_lockin_is_retained_as_its_own_channel():
    summary = json.loads((ROOT / "results/route_9/summary.json").read_text())
    rows = [row for row in summary["conformal"] if row["channel"] == "r3_static_lockin"]
    assert {row["unit"] for row in rows} == {"episode_maximum", "scene_cluster_maximum"}
    selected = [row for row in summary["learn_then_test_style"] if row["event"] == "static"][0]
    assert selected["selected"]["channel"] == "r3_static_lockin"
