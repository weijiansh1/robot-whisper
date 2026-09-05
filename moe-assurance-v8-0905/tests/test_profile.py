from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILES = sorted((ROOT / "results/profiles/main16x32").glob("*/*.npz"))
PROFILE = PROFILES[0] if PROFILES else None
pytestmark = pytest.mark.skipif(PROFILE is None, reason="optional dense profiles not present")


def test_profile_is_outcome_free_and_graph_rich():
    with np.load(PROFILE, allow_pickle=False) as archive:
        assert "success" not in archive.files
        assert "outcome" not in archive.files
        assert archive["features"].shape[1] == 40
        assert archive["layer_features"].shape[1:] == (4, 13)


def test_graph_identities_and_bounds():
    with np.load(PROFILE, allow_pickle=False) as archive:
        names = archive["layer_feature_names"].astype(str).tolist()
        layer = archive["layer_features"].astype(np.float64)
    commitment = layer[:, :, names.index("commitment")]
    entropy = layer[:, :, names.index("entropy_normalized")]
    top1 = layer[:, :, names.index("top1_mass")]
    top4 = layer[:, :, names.index("top4_mass")]
    effective_rank = layer[:, :, names.index("effective_rank")]
    mutual_information = layer[:, :, names.index("token_expert_mi")]
    assert np.max(np.abs(commitment + entropy - 1.0)) < 1e-3
    assert np.all(top4 >= top1)
    assert np.all((effective_rank >= 1.0) & (effective_rank <= 10.01))
    assert np.min(mutual_information) >= -1e-4
