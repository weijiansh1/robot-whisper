import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import evaluate_phenotype_2x2 as ph  # noqa: E402


def test_cell_assignment_uses_the_development_median():
    mobility = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    ratio = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)
    cells = ph.assign_cells(mobility, ratio, mobility_median=0.5, ratio_threshold=0.5)
    assert list(cells) == ["low_mob_low_R", "low_mob_high_R", "high_mob_low_R", "high_mob_high_R"]


def test_k1_gate_counts_only_risk_episodes_in_the_high_mobility_cell():
    cells = np.array(["high_mob_low_R"] * 25 + ["low_mob_low_R"] * 40)
    risk = np.array([True] * 19 + [False] * 6 + [True] * 40)
    assert ph.k1_gate(cells, risk) == (19, False)
    risk[19] = True
    assert ph.k1_gate(cells, risk) == (20, True)


def test_k3_gate_requires_ten_percent_novel_detections():
    ratio_first = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=np.int16)
    freeze_first = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=np.int16)
    risk = np.ones(10, dtype=bool)
    assert ph.k3_gate(ratio_first, freeze_first, risk) == (0.0, False)
    freeze_first[:2] = -1
    share, passed = ph.k3_gate(ratio_first, freeze_first, risk)
    assert share == pytest.approx(0.2)
    assert passed is True


def test_k3_gate_ignores_non_risk_episodes():
    ratio_first = np.array([3, 3], dtype=np.int16)
    freeze_first = np.array([-1, -1], dtype=np.int16)
    risk = np.array([False, False])
    assert ph.k3_gate(ratio_first, freeze_first, risk) == (0.0, False)
