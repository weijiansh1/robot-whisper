#!/usr/bin/env bash
# Deterministic end-to-end rerun. CPU only; no GPU is touched.
set -euo pipefail
cd "$(dirname "$0")/experiments"

python3 verify_join.py                 # simulator labels <-> routing caches
python3 rebuild_development_alarms.py  # 24 frozen detectors on development; external re-checked
python3 mode_specialisation.py         # Q1: stratified permutation nulls
python3 confound_structure.py          # confounding + power + timing diagnostics
python3 replication.py                 # Q2: development -> external replication
python3 multihead.py                   # Q3/Q4: pre-registered greedy bundles
python3 frontier.py                    # Q4: external TP-vs-FP frontier
python3 interpretable_heads.py         # Q2/Q3: threshold-mode interaction, named heads
python3 bundle_mode_gain.py            # Q3: what the extra heads actually catch
python3 partner_control.py             # Q3: label-derived prediction, outcome-only test
python3 headline.py                    # consolidated operating points
python3 make_manifest.py
