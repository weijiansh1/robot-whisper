import json
from functools import cache
from html.parser import HTMLParser

import numpy as np

from build_moe_rollout_demo_data import RESULT_DIR, build_payload


class _DemoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.script_sources = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "script" and "src" in attributes:
            self.script_sources.add(attributes["src"])


@cache
def _payload():
    return build_payload(RESULT_DIR)


def test_demo_payload_matches_fixed_cohort_and_curves():
    payload = _payload()
    assert payload["meta"]["rollouts"] == 2048
    assert payload["meta"]["failure"] == 307
    assert payload["meta"]["success"] == 1741
    assert len(payload["meta"]["tasks"]) == 4
    assert len(payload["curves"]) == 3 * 9 * 2
    assert len(payload["gaps"]) == 3 * 9
    assert len(payload["by_task"]) == 4 * 3 * 9
    assert payload["meta"]["frozen_anchor_max_abs_diff"] == 0.0


def test_demo_payload_keeps_both_evaluations_and_controls():
    payload = _payload()
    evaluations = {row["evaluation"] for row in payload["classifier_macro"]}
    families = {row["family"] for row in payload["classifier_macro"]}
    assert evaluations == {"seed_heldout", "scene_loso"}
    assert {"routed_full", "hidden_identity", "base", "base_routed"} <= families


def test_demo_payload_contains_real_chunk_internal_dynamics():
    payload = _payload()
    dynamics = payload["dynamics"]
    assert dynamics["layers"] == [2, 5, 12, 15]
    assert dynamics["denoise_rounds"] == 10
    assert set(dynamics["metrics"]) == {
        "routed_rms",
        "cancellation",
        "shared_conflict",
        "commitment",
    }
    assert set(dynamics["by_scope"]) == {
        "aggregate",
        *(task["id"] for task in payload["meta"]["tasks"]),
    }
    for group in ("success", "failure", "gap"):
        values = np.asarray(
            dynamics["by_scope"]["aggregate"]["routed_rms"][group]
        )
        assert values.shape == (9, 10, 4)
        assert np.all(np.isfinite(values))


def test_demo_payload_contains_same_state_physical_routes():
    payload = _payload()
    atlas = payload["trajectory_routes"]
    assert atlas["version"] == 3
    assert atlas["fixed_contribution_chunks"] == 9
    assert len(atlas["tasks"]) == 4
    assert sum(len(task["scenes"]) for task in atlas["tasks"]) == 64
    for task in atlas["tasks"]:
        assert len(task["scenes"]) == 16
        assert task["min_chunks"] >= 9
        assert task["max_chunks"] >= task["min_chunks"]
        for scene in task["scenes"]:
            assert len(scene["routes"]) == 32
            assert scene["success"] + scene["failure"] == 32
            assert len(scene["physical_centroid_gap_m"]) == scene["max_chunks"]
            assert len(scene["routing_gap"]) == scene["max_chunks"]
            assert len(scene["active_counts"]) == scene["max_chunks"]
            starts = np.asarray([route["points"][0] for route in scene["routes"]])
            assert np.allclose(starts, starts[0])
            for route in scene["routes"]:
                assert len(route["points"]) == route["chunks"]
                assert len(route["routing_percentile"]) == route["chunks"]
                assert len(route["routing_churn_percentile"]) == route["chunks"]
                assert len(route["routing_revision_percentile"]) == route["chunks"]
                assert all(
                    0 < value < 1
                    for value in route["routing_churn_percentile"]
                )
                assert all(
                    0 < value < 1
                    for value in route["routing_revision_percentile"]
                )
                assert len(route["contribution_percentile"]) == min(
                    atlas["fixed_contribution_chunks"], route["chunks"]
                )


def test_full_route_atlas_exposes_long_rollouts_and_risk_set_limit():
    atlas = _payload()["trajectory_routes"]
    task = next(task for task in atlas["tasks"] if "moka_pots" in task["task"])
    assert task["max_chunks"] == 52
    scene = next(scene for scene in task["scenes"] if scene["scene"] == 42)
    assert (scene["success"], scene["failure"]) == (14, 18)
    assert max(route["chunks"] for route in scene["routes"]) == 52
    assert any(route["chunks"] == 52 and not route["success"] for route in scene["routes"])
    assert scene["active_counts"][51] == [0, 18]
    assert scene["routing_gap"][51] is None


def test_generated_demo_data_matches_latest_payload():
    payload = _payload()
    expected = (
        "window.HIMOE_ROLLOUT_DEMO = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    assert (RESULT_DIR / "demo-data.js").read_text() == expected


def test_demo_html_keeps_interactive_analysis_contract():
    html = (RESULT_DIR / "demo.html").read_text()
    parser = _DemoHTMLParser()
    parser.feed(html)
    assert parser.script_sources == {"./demo-data.js"}
    assert {
        "signal-controls",
        "task-select",
        "chunk-range",
        "basin-canvas",
        "moe-canvas",
        "dynamic-metric-controls",
        "motion-timeline",
        "main-chart",
        "gap-chart",
        "task-table-body",
        "evaluation-controls",
        "auc-bars",
    } <= parser.ids
    assert all(
        signal in html
        for signal in (
            "per_query_success_manifold_percentile",
            "per_query_risk_percentile",
            "frozen_k8_failure_axis",
            "routed_rms",
            "shared_conflict",
            "commitment",
        )
    )


def test_trajectory_demo_keeps_route_animation_contract():
    html = (RESULT_DIR / "trajectory-demo.html").read_text()
    javascript = (RESULT_DIR / "trajectory-demo.js").read_text()
    parser = _DemoHTMLParser()
    parser.feed(html)
    assert parser.script_sources == {"./demo-data.js", "./trajectory-demo.js"}
    assert {
        "route-canvas",
        "task-select",
        "scene-select",
        "route-select",
        "play",
        "replay",
        "route-range",
        "nodes",
        "physical-gap",
        "task-gap",
        "cohort-gap",
    } <= parser.ids
    assert "trajectory_routes" in html
    assert "s<sub>k+1</sub> = F" in html
    assert "fixed_contribution_chunks" in javascript
    assert "routing_percentile" in javascript
    assert "maxIndex" in javascript


def test_moe_state_demo_keeps_computation_state_contract():
    html = (RESULT_DIR / "moe-state-demo.html").read_text()
    javascript = (RESULT_DIR / "moe-state-demo.js").read_text()
    parser = _DemoHTMLParser()
    parser.feed(html)
    assert parser.script_sources == {"./demo-data.js", "./moe-state-demo.js"}
    assert {
        "state-canvas",
        "denoise-canvas",
        "task-select",
        "scene-select",
        "route-select",
        "metric-control",
        "chunk-range",
        "distance-gap",
        "dynamic-gap",
        "contribution-gap",
    } <= parser.ids
    assert "routing_churn_percentile" in javascript
    assert "routing_revision_percentile" in javascript
    assert "active-success routing centroid" in javascript
    assert "trajectory-demo.html" in html
