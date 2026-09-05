import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_q0_dirichlet_probabilities_form_a_simplex():
    tensor = pd.read_csv(ROOT / "results/route_8/assurance_tensor.csv")
    q0 = tensor[tensor.source == "q0_committor"]
    grouped = q0.groupby(["corpus", "suite", "task", "scene"], dropna=False)
    assert len(grouped) == 2080
    for _key, frame in grouped:
        assert set(frame.outcome) == {"success", "loop", "static", "other_failure"}
        assert abs(frame.posterior_mean.sum() - 1.0) < 1e-10
        assert frame["count"].sum() == frame.total.iloc[0]
        assert ((frame.ci95_low <= frame.posterior_mean) & (frame.posterior_mean <= frame.ci95_high)).all()


def test_recovery_collection_is_not_overstated():
    summary = json.loads((ROOT / "results/route_8/summary.json").read_text())
    recovery = summary["recovery_collection"]
    assert recovery["status"] == "incomplete"
    assert recovery["budget20_complete_trunks"] == 1
    assert "multi-snapshot two-query physical no-loop committor" in summary["unavailable"]
