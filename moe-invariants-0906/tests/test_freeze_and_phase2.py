"""Phase 2 may only test what phase 1 froze, and may not use the horizon cap."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1]
OUT = BUNDLE / "results"
EXP = BUNDLE / "experiments"
sys.path.insert(0, str(EXP))


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p.name} not produced yet")
    return json.loads(p.read_text())


def test_freeze_hash_matches():
    meta = _need(OUT / "phase1_freeze.json")
    p = OUT / "phase1_frozen_candidates.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]


def test_freeze_covers_its_inputs():
    meta = _need(OUT / "phase1_freeze.json")
    for name, h in meta["phase1_inputs"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == h, name


def test_phase2_tests_only_frozen_relations():
    frozen = _need(OUT / "phase1_frozen_candidates.json")
    ph2 = _need(OUT / "phase2_breach_summary.json")
    known = {(c["target"], tuple(c["preds"])) for c in frozen["candidates"]}
    for r in ph2["relations_tested"]:
        if r["role"] in ("empirical_invariant", "positive_control_exact_identity"):
            assert (r["target"], tuple(r["preds"])) in known, r["target"]


def test_phase2_never_uses_the_cap():
    """No cap, no phase, no per-task threshold, no task identity."""
    src = (EXP / "phase2_breach.py").read_text()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc.add(ln)
    banned = ("CAPS", "ANCHOR_WINDOW", "per_task", "survival_prior", "prior_of",
              "0.65")
    bad = []
    for i, line in enumerate(src.splitlines(), start=1):
        code = line.split("#", 1)[0]
        if i in doc or not code.strip():
            continue
        for b in banned:
            if b in code:
                bad.append((i, b, line.strip()))
    assert not bad, bad


def test_phase2_imports_the_shared_protocol():
    src = (EXP / "phase2_breach.py").read_text()
    assert "import capfree_protocol as cp" in src
    assert "cp.score(" in src and "cp.fixed_chunk_baseline(" in src
    assert "cp.baseline_frontier(" in src and "cp.rate_matched_null(" in src


def test_phase2_headline_is_lead_four_and_counts_are_absolute():
    ph2 = _need(OUT / "phase2_breach_summary.json")
    assert ph2["headline_lead"] == 4
    e = ph2["external_once"]
    assert isinstance(e["tp"], int) and isinstance(e["fp"], int)
    assert e["n_risk"] == 564
    assert 0 <= e["tp"] <= e["n_risk"]


def test_exact_identity_has_no_breach_series():
    """The known identity is exact, so it cannot alarm at any threshold."""
    ph2 = _need(OUT / "phase2_breach_summary.json")
    pc = ph2["per_role_best"]["positive_control_exact_identity"]
    assert pc["unbreakable"] is True
    assert pc["ext_alarms"] == 0 and pc["ext_tp"] == 0


def test_alignment_between_bank_and_labels_was_audited():
    ph2 = _need(OUT / "phase2_breach_summary.json")
    for c, a in ph2["alignment_audit"].items():
        assert a["valid_chunks_equal_length"] is True, c
    assert ph2["alignment_audit"]["external_8b"]["n_episodes"] == 15600
    assert ph2["alignment_audit"]["development_main"]["n_episodes"] == 14800


def test_manifest_hashes_every_result():
    man = _need(OUT / "manifest.json")
    for rec in man["results"]:
        p = BUNDLE / rec["path"]
        assert p.exists(), rec["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"], rec["path"]
