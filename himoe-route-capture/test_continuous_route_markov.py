import numpy as np

from analyze_intraquery_markov import (
    HB_LAYERS,
    N_ACTION_TOKENS,
    N_DENOISE,
    N_EXPERTS,
)
from continuous_route_markov import (
    continuous_route_states,
    continuous_sequence_log_likelihoods,
    fit_continuous_route_models,
)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def test_continuous_route_state_keeps_every_layer_token_expert_coordinate():
    routes = np.zeros(
        (2, len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS),
        dtype=np.float32,
    )
    routes[0, ..., 0] = 1.0
    routes[1, ..., 1] = 1.0

    states = continuous_route_states(routes)

    assert states.shape == (
        2,
        N_DENOISE,
        len(HB_LAYERS) * N_ACTION_TOKENS * N_EXPERTS,
    )
    assert np.all(states[0].reshape(N_DENOISE, -1, N_EXPERTS)[..., 0] == 1.0)
    assert np.all(states[1].reshape(N_DENOISE, -1, N_EXPERTS)[..., 1] == 1.0)


def test_continuous_markov_detects_dynamics_with_matched_occupancy():
    rng = np.random.default_rng(11)
    labels = np.tile([False, True], 200)
    states = np.empty((len(labels), N_DENOISE, 4), dtype=np.float64)
    states[:, 0] = rng.standard_normal((len(labels), 4))
    innovation = np.sqrt(1.0 - 0.85**2)
    for tau in range(1, N_DENOISE):
        slope = np.where(labels, 0.85, -0.85)[:, None]
        states[:, tau] = slope * states[:, tau - 1] + innovation * rng.standard_normal(
            (len(labels), 4)
        )

    models = fit_continuous_route_models(states[:300], labels[:300])
    logp = {
        label: continuous_sequence_log_likelihoods(models[label], states[300:])
        for label in (False, True)
    }
    markov_score = (
        logp[True]["markov_periodic"][:, -1] - logp[False]["markov_periodic"][:, -1]
    )
    occupancy_score = (
        logp[True]["occupancy_periodic"][:, -1]
        - logp[False]["occupancy_periodic"][:, -1]
    )

    assert _auc(labels[300:], markov_score) > 0.95
    assert _auc(labels[300:], markov_score) > _auc(labels[300:], occupancy_score) + 0.3


def test_outcome_models_share_all_variance_parameters():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(40, N_DENOISE, 5))
    labels = np.repeat([False, True], 20)

    models = fit_continuous_route_models(states, labels)

    assert np.array_equal(
        models[False].occupancy_periodic_variance,
        models[True].occupancy_periodic_variance,
    )
    assert np.array_equal(
        models[False].transition_periodic_variance,
        models[True].transition_periodic_variance,
    )
    assert np.array_equal(
        models[False].transition_position_variance,
        models[True].transition_position_variance,
    )
