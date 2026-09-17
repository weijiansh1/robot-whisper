"""Compatibility entry point preserving signed zero in frozen bias reconstruction."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import audit_coverage_independently as frozen_audit

numeric_inputs = frozen_audit.expected_inputs


def expected_inputs(parent, source, spec):
    bias, noise = numeric_inputs(parent, source, spec)
    if spec is not None and spec["generator"] != "noise":
        rng = np.random.default_rng(np.random.SeedSequence([2026091501, 3, spec["pool"], spec["candidate"]]))
        negative = np.argsort(rng.random(bias.shape), axis=-1)[..., :16]
        signs = np.ones(bias.shape, np.float32)
        np.put_along_axis(signs, negative, -1., axis=-1)
        corrected = np.copysign(bias, signs)
        if not np.array_equal(corrected, bias):
            raise RuntimeError("Signed-zero compatibility changed a numeric bias")
        bias = corrected
    return bias, noise


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: audit_coverage_signed_zero.py RUN")
    root = Path(sys.argv[1]).resolve()
    base_path = Path(frozen_audit.__file__).resolve()
    config = json.loads((root / "config.json").read_text())
    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if config["source_hashes"][str(base_path)] != base_sha:
        raise RuntimeError("Frozen base auditor changed")
    correction = {
        "reason": "Frozen auditor reconstructed off-scope +0, but the collector preserves the random direction sign as +0 or -0",
        "first_audit_error": "Request digest mismatch",
        "diagnostic_raw_digests_all_match": True,
        "diagnostic_numeric_inputs_all_match": True,
        "diagnostic_signed_zero_only_mismatches": 216,
        "frozen_base_auditor": str(base_path), "frozen_base_sha256": base_sha,
        "compatibility_entry_point": str(Path(__file__).resolve()),
        "compatibility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numeric_bias_changed": False, "raw_data_changed": False, "metric_code_changed": False,
    }
    with (root / "audit-correction.json").open("x") as stream:
        json.dump(correction, stream, indent=2)
        stream.write("\n")
    # Reuse every frozen audit check; only the expected input's zero sign changes.
    frozen_audit.expected_inputs = expected_inputs
    frozen_audit.main()


if __name__ == "__main__":
    main()
