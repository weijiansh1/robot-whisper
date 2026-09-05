import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_requested_routes_have_results():
    required = [
        "results/routes_1_4/summary.json",
        "results/routes_5_6/summary.json",
        "results/route_7/summary.json",
        "results/route_8/summary.json",
        "results/route_9/summary.json",
        "results/full_moe_axes/summary.json",
        "results/example_assurance_q0.json",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_full_axis_inventory_and_external_lock():
    summary = json.loads((ROOT / "results/full_moe_axes/summary.json").read_text())
    assert summary["axes"] == 109
    assert summary["fitted_weights"] is False
    assert "grid50x8 is untouched confirmation" in summary["label_use"]


def test_example_separates_belief_from_outcome():
    state = json.loads((ROOT / "results/example_assurance_q0.json").read_text())
    assert abs(sum(state["mode_belief"].values()) - 1.0) < 2e-6
    assert abs(sum(row["posterior_mean"] for row in state["outcome_assurance"]) - 1.0) < 1e-10
    assert state["schema"] == "himoe.moe_assurance_state.v1"


def test_sealed_summary_contains_all_routes_and_probability_boundary():
    summary = json.loads((ROOT / "results/final_summary.json").read_text())
    manifest = json.loads((ROOT / "results/sealed_manifest.json").read_text())
    assert len(summary["routes"]) == 9
    assert summary["inventory"]["total_axes"] == 109
    assert summary["probability_artifact"] == "results/route_8/assurance_tensor.csv"
    assert manifest["artifact_count"] == len(manifest["artifacts"])
    assert any(row["path"] == "results/final_summary.json" for row in manifest["artifacts"])
