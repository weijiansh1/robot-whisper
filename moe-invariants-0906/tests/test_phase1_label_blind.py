"""Phase 1 must be blind to outcomes.  These tests enforce it two ways.

Static: no phase-1 source may name a label, a label-bearing module, or an
alarm vector.  Runtime: every array phase 1 actually reads is intercepted, and
the test asserts that no forbidden cache file is opened and that the key
`length` -- against which the risk label is defined -- is never pulled out of
any index file.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
EXP = BUNDLE / "experiments"
ROOT = BUNDLE.parent
sys.path.insert(0, str(EXP))

PHASE1 = sorted(EXP.glob("phase1_*.py"))

# Anything that names, or is a route to, an outcome.
FORBIDDEN_TOKENS = (
    "risk", "original_failure", "failure_mode", "physical_failure",
    "cohort_frame", "recompute_task_matched_lift", "capfree_protocol",
    "first_alarm", "alarm", "success", "timeout", "outcome", "label",
    "moe-cusum-capfree-0906", "moe-failure-mode-detectors-0906",
)
# Words that legitimately appear in prose about what is NOT done.
DOC_EXEMPT = re.compile(r"^$")   # no exemptions: executable code must be clean

ALLOWED_CACHES = (
    ROOT / "moe-flow-semantics-0906" / "results" / "step_profiles",
    ROOT / "moe-unused-channels-0906" / "results" / "channels",
    ROOT / "moe-hb-front-back-0905" / "results" / "layer_graphs",
    BUNDLE / "results",
)


def test_phase1_files_exist():
    assert len(PHASE1) >= 6, [p.name for p in PHASE1]


@pytest.mark.parametrize("path", PHASE1, ids=lambda p: p.name)
def test_no_forbidden_token_in_code(path: Path):
    """Docstrings and comments may say the word; executable code may not."""
    tree = ast.parse(path.read_text())
    doc_ranges = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_ranges.add(ln)
    bad = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]
        if not code.strip() or i in doc_ranges:
            continue
        for tok in FORBIDDEN_TOKENS:
            if tok in code.lower() and not DOC_EXEMPT.search(line):
                bad.append((i, tok, line.strip()))
    assert not bad, bad


@pytest.mark.parametrize("path", PHASE1, ids=lambda p: p.name)
def test_no_label_bearing_import(path: Path):
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert not m.startswith(("recompute", "capfree", "pandas")), (path.name, m)


def test_phase1_reads_only_routing_caches(monkeypatch, tmp_path):
    """Intercept every np.load and every npz key phase 1 asks for."""
    import phase1_build_bank as pb

    opened: list[str] = []
    keys: list[str] = []
    real_load = np.load
    real_getitem = np.lib.npyio.NpzFile.__getitem__

    def spy_load(file, *a, **k):
        opened.append(str(file))
        return real_load(file, *a, **k)

    def spy_getitem(self, key):
        keys.append(str(key))
        return real_getitem(self, key)

    monkeypatch.setattr(np, "load", spy_load)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy_getitem)
    pb.build("development_extra")

    assert opened, "nothing was read"
    for p in opened:
        assert any(str(a) in p for a in ALLOWED_CACHES), p
        assert "cusum-capfree" not in p and "failure-mode-detectors" not in p, p
    assert "length" not in keys, "phase 1 read the episode length"
    for forbidden in ("risk", "original_failure", "failure_mode"):
        assert forbidden not in keys


def test_bank_carries_no_length_or_label():
    bank = np.load(BUNDLE / "results" / "bank" / "development_main_bank.npz",
                   allow_pickle=True)
    for k in bank.files:
        assert k not in ("length", "risk", "original_failure"), k


def test_frozen_list_mentions_no_label():
    p = BUNDLE / "results" / "phase1_frozen_candidates.json"
    if not p.exists():
        pytest.skip("phase 1 not frozen yet")
    text = p.read_text().lower()
    for tok in ("risk", "original_failure", "failure_mode", "alarm",
                "\"length\"", "tp_lead", "recall"):
        assert tok not in text, tok
