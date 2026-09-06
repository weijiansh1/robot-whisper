import numpy as np
import pytest
import torch
from sklearn.mixture import GaussianMixture

from moe_grammar.gpu_tokenizer import BatchedDiagonalGMM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _sample(seed: int = 3, rows: int = 4000, dimensions: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(4, dimensions))
    assignments = rng.integers(0, 4, size=rows)
    return (centers[assignments] + rng.normal(scale=0.6, size=(rows, dimensions))).astype(
        np.float32
    )


def test_scoring_matches_sklearn_for_identical_parameters() -> None:
    values = _sample()
    reference = GaussianMixture(
        n_components=4, covariance_type="diag", reg_covar=1e-4, max_iter=30, n_init=1,
        random_state=0,
    ).fit(values)

    model = BatchedDiagonalGMM(n_components=4, n_init=1, device="cuda:0")
    model.weights_ = reference.weights_
    model.means_ = reference.means_
    model.covariances_ = reference.covariances_

    np.testing.assert_allclose(
        model.score_samples(values), reference.score_samples(values), rtol=1e-4, atol=1e-3
    )
    np.testing.assert_array_equal(model.predict(values), reference.predict(values))


def test_one_em_step_matches_sklearn_from_the_same_start() -> None:
    values = _sample(seed=11)
    start = GaussianMixture(
        n_components=4, covariance_type="diag", reg_covar=1e-4, max_iter=5, n_init=1,
        random_state=1,
    ).fit(values)

    reference = GaussianMixture(
        n_components=4, covariance_type="diag", reg_covar=1e-4, max_iter=1, n_init=1,
        weights_init=start.weights_, means_init=start.means_,
        precisions_init=1.0 / start.covariances_, random_state=1,
    ).fit(values)

    model = BatchedDiagonalGMM(
        n_components=4, n_init=1, max_iter=1, reg_covar=1e-4, device="cuda:0"
    )
    device = torch.device("cuda:0")
    data = torch.as_tensor(values, device=device)
    weights = torch.as_tensor(start.weights_, dtype=torch.float32, device=device).unsqueeze(0)
    means = torch.as_tensor(start.means_, dtype=torch.float32, device=device).unsqueeze(0)
    covariances = torch.as_tensor(
        start.covariances_, dtype=torch.float32, device=device
    ).unsqueeze(0)
    log_responsibility, _ = model._estep(data, weights, means, covariances)
    responsibility = torch.exp(log_responsibility)
    counts = responsibility.sum(dim=-2) + 10.0 * torch.finfo(data.dtype).eps
    new_means = torch.einsum("irc,rd->icd", responsibility, data) / counts.unsqueeze(-1)
    squared = torch.einsum("irc,rd->icd", responsibility, data * data) / counts.unsqueeze(-1)
    new_covariances = squared - new_means * new_means + model.reg_covar

    np.testing.assert_allclose(
        new_means[0].cpu().numpy(), reference.means_, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        new_covariances[0].cpu().numpy(), reference.covariances_, rtol=1e-2, atol=1e-3
    )
    np.testing.assert_allclose(
        (counts[0] / len(values)).cpu().numpy(), reference.weights_, rtol=1e-3, atol=1e-4
    )


def test_batched_fit_reaches_a_comparable_optimum() -> None:
    values = _sample(seed=5)
    reference = GaussianMixture(
        n_components=4, covariance_type="diag", reg_covar=1e-4, max_iter=100, n_init=10,
        init_params="k-means++", random_state=0,
    ).fit(values)
    model = BatchedDiagonalGMM(
        n_components=4, n_init=10, max_iter=100, reg_covar=1e-4, device="cuda:0"
    ).fit(values)
    # Both search the same objective; the batched GPU run must not be materially worse.
    assert model.lower_bound_ > reference.lower_bound_ - 0.05
    assert model.all_lower_bounds_.shape == (10,)


def test_more_components_never_reduce_the_training_bound() -> None:
    values = _sample(seed=7)
    small = BatchedDiagonalGMM(n_components=2, n_init=4, device="cuda:0").fit(values)
    large = BatchedDiagonalGMM(n_components=8, n_init=4, device="cuda:0").fit(values)
    assert large.lower_bound_ > small.lower_bound_
