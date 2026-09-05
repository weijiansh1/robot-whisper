from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "analysis/results"


def test_overall_alarm_phase_counts_and_medians() -> None:
    frame = pd.read_csv(RESULT / "alarm_timing_overall.csv").set_index("cohort")
    expected = {
        "development_main": (439, 372, 67, 72.549020),
        "external_8b": (491, 410, 81, 71.428571),
        "cache_right16x32": (242, 229, 13, 70.588235),
    }
    for cohort, (alarms, tp, fp, median_phase) in expected.items():
        row = frame.loc[cohort]
        assert int(row["alarm_n"]) == alarms
        assert int(row["tp"]) == tp
        assert int(row["fp"]) == fp
        assert np.isclose(row["alarm_phase_pct_median"], median_phase)
        bucket_count = sum(
            int(row[f"phase_{bucket}_n"])
            for bucket in ("00_25", "25_50", "50_75", "75_100")
        )
        assert bucket_count == alarms


def test_task_percent_table_uses_zero_to_one_hundred_units() -> None:
    frame = pd.read_csv(RESULT / "alarm_timing_by_task_percent.csv")
    percent_columns = [column for column in frame if column.endswith("_pct")]
    finite = frame[percent_columns].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    assert (finite >= 0.0).all()
    assert (finite <= 100.0).all()
