import numpy as np

from analyze_intraquery_markov import (
    HB_LAYERS,
    N_ACTION_TOKENS,
    N_DENOISE,
    N_EXPERTS,
    assign_layer_codebooks,
    double_holdout_mask,
    fit_route_markov_model,
    flatten_microstates,
    sequence_log_likelihoods,
    topk_tie_weights,
)


def test_microstate_order_is_denoise_then_layer():
    codes = np.empty((1, N_DENOISE, len(HB_LAYERS)), dtype=np.int64)
    for denoise in range(N_DENOISE):
        for layer in range(len(HB_LAYERS)):
            codes[0, denoise, layer] = denoise * 10 + layer

    flat = flatten_microstates(codes)

    assert np.array_equal(flat[0, :9], [0, 1, 2, 3, 4, 5, 6, 7, 10])
    assert flat[0, -1] == 97


def test_wrap_transition_does_not_cross_query_boundaries():
    codes = np.empty((2, N_DENOISE, len(HB_LAYERS)), dtype=np.int64)
    codes[0] = 0
    codes[1] = 1

    model = fit_route_markov_model(codes, clusters=2, alpha=1.0)
    wrap = model.transition_periodic[-1]

    assert wrap[0, 0] > wrap[0, 1]
    assert wrap[1, 1] > wrap[1, 0]


def test_periodic_markov_beats_occupancy_on_deterministic_microchain():
    codes = np.empty((40, N_DENOISE, len(HB_LAYERS)), dtype=np.int64)
    for episode in range(len(codes)):
        offset = episode % 2
        for denoise in range(N_DENOISE):
            for layer in range(len(HB_LAYERS)):
                codes[episode, denoise, layer] = (offset + denoise + layer) % 2

    model = fit_route_markov_model(codes[:30], clusters=2, alpha=1.0)
    logp = sequence_log_likelihoods(model, codes[30:])

    assert logp["markov_periodic"][:, -1].mean() > (
        logp["occupancy_periodic"][:, -1].mean() + 30.0
    )


def test_double_holdout_removes_scene_and_seed_fold():
    scenes = np.repeat([0, 1, 2], 4)
    folds = np.tile([0, 1, 2, 3], 3)

    mask = double_holdout_mask(scenes, folds, held_scene=1, held_fold=2)

    assert not np.any(mask[scenes == 1])
    assert not np.any(mask[folds == 2])
    assert int(mask.sum()) == 6


def test_topk_ties_receive_fractional_weights():
    weights = topk_tie_weights(np.array([3.0, 2.0, 2.0, 0.0]), k=2)

    assert np.allclose(weights, [1.0, 0.5, 0.5, 0.0])
    assert weights.sum() == 2.0


def test_frozen_codebooks_assign_new_route_fingerprints():
    routes = np.zeros(
        (2, len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS),
        dtype=np.float32,
    )
    routes[0, ..., 0] = 1.0
    routes[1, ..., 1] = 1.0
    centers = np.zeros(
        (len(HB_LAYERS), 2, N_ACTION_TOKENS * N_EXPERTS), dtype=np.float32
    )
    center_view = centers.reshape(
        len(HB_LAYERS), 2, N_ACTION_TOKENS, N_EXPERTS
    )
    center_view[:, 0, :, 0] = 1.0
    center_view[:, 1, :, 1] = 1.0

    codes = assign_layer_codebooks(routes, centers)

    assert np.all(codes[0] == 0)
    assert np.all(codes[1] == 1)
