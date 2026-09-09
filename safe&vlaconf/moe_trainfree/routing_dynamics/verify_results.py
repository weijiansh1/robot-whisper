"""Seal documentation, inspect plots, and verify the numerical-only correction."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from run_analysis import BASE, digest, write_json


def main():
    output = BASE / "routing_dynamics_20260908"
    initial = BASE / "routing_dynamics_20260908_before_roundoff_fix"
    manifest = json.loads((output / "verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    tables = ("query_comparisons.csv", "same_initial_comparisons.csv", "task_summary.csv", "summary.csv")
    for name in tables:
        pd.testing.assert_frame_equal(pd.read_csv(initial / name), pd.read_csv(output / name))
    unchanged = []
    with np.load(initial / "features.npz", allow_pickle=False) as old, np.load(output / "features.npz", allow_pickle=False) as current:
        for name in ("features", "per_layer", "valid", "global_rows", "names", "layer_names", "layer_feature_names"):
            np.testing.assert_array_equal(old[name], current[name])
            unchanged.append(name)
    figures = []
    for path in output.glob("*.png"):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        nonwhite = float((pixels.min(-1) < 245).mean())
        assert pixels.shape[0] > 700 and pixels.shape[1] > 1500 and nonwhite > .025
        figures.append(dict(name=path.name, shape=pixels.shape, nonwhite_share=nonwhite))
    links = re.findall(r"\]\(([^)]+)\)", (output / "REPORT_ZH.md").read_text())
    for link in links:
        assert (output / link).exists(), link
    write_json(output / "final_verification.json", dict(
        all_checks_passed=True, unchanged_after_roundoff_fix=unchanged,
        unchanged_statistical_tables=tables, documentation_links_checked=len(links), figures=figures,
        implementation_sources={p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        artifacts={p.name: digest(p) for p in sorted(output.iterdir())
                   if p.is_file() and p.name != "final_verification.json"}))
    print("VERIFIED: unchanged corpus encodings and statistics; plot pixels and report links valid")


if __name__ == "__main__":
    main()
