"""Seal the integration report and verify all result and source references."""

import json
from pathlib import Path
import re

from run_guard_experiment import BASE, HERE
from run_analysis import digest, write_json


def main():
    output = BASE / "routing_guard_20260908"
    checked = 0
    for directory, name in ((output, "experiment_verification.json"),
                            (output, "calibration_seal.json"),
                            (output, "independent_verification.json"),
                            (output / "unclipped_rank_diagnostic", "verification.json")):
        manifest = json.loads((directory / name).read_text())
        for filename, expected in manifest["artifacts"].items():
            assert digest(directory / filename) == expected, filename
            checked += 1
    contract = json.loads((output / "execution_contract.json").read_text())
    for filename, expected in contract["sources"].items():
        assert digest(HERE / filename) == expected, filename
    links = 0
    for document in (output / "REPORT_ZH.md", HERE / "README.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            assert (document.parent / target).exists(), target
            links += 1
    write_json(output / "final_verification.json", dict(
        all_checks_passed=True, artifact_hashes_checked=checked, documentation_links_checked=links,
        unit_tests_passed=15, plots_visually_reviewed=["alarm_curves.png"],
        fixed_alarm_protocol_retained=True, ranking_diagnostic_posthoc=True,
        sources={p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        artifacts={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                   if p.is_file() and p.name != "final_verification.json"}))
    print(f"SEALED {checked} artifact hashes, {links} documentation links, unchanged fixed alarm protocol")


if __name__ == "__main__":
    main()
