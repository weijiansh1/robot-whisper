import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assurance"))

from hsmm import HSMMModel  # noqa: E402


def load_model():
    with np.load(ROOT / "results/route_7/model.npz", allow_pickle=False) as archive:
        return HSMMModel(
            states=tuple(archive["state_names"].astype(str)),
            means=archive["means"],
            variances=archive["variances"],
            initial=archive["initial"],
            survival=archive["survival"],
            exit_probability=archive["exit_probability"],
            max_age=int(archive["max_age"]),
        )


def test_transition_and_forecast_normalize():
    model = load_model()
    belief = np.zeros((len(model.states), model.max_age))
    belief[:, 0] = model.initial
    for _ in range(20):
        belief = model.transition(belief)
        assert np.isclose(belief.sum(), 1.0)
        assert np.min(belief) >= 0
    for value in model.forecast(belief, (1, 2, 4)).values():
        assert np.isclose(value.sum(), 1.0)


def test_filter_is_prefix_causal():
    model = load_model()
    rng = np.random.default_rng(7)
    observation = rng.normal(size=(12, model.means.shape[1]))
    full, _ = model.filter(observation)
    prefix, _ = model.filter(observation[:7])
    assert np.allclose(full[:7], prefix)
    assert np.allclose(full.sum(axis=1), 1.0)


def test_saved_posteriors_normalize():
    path = next((ROOT / "results/route_7/posteriors/grid50x8").glob("*/*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        posterior = archive["posterior"].astype(np.float64)
        forecast = archive["forecast"].astype(np.float64)
    assert np.max(np.abs(posterior.sum(axis=1) - 1.0)) < 2e-6
    assert np.max(np.abs(forecast.sum(axis=2) - 1.0)) < 2e-6
