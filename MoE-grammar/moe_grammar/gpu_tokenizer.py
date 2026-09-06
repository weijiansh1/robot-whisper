"""Batched diagonal-covariance GMM on the GPU.

The audit asks for a vocabulary sweep over ``K`` with at least ten initializations and a
cross-fold word-stability check. On CPU that is hundreds of sklearn fits over 508k
queries and is not tractable. This module runs all initializations of one ``K`` as a
leading batch dimension so the E-step is a single large batched matmul, which is what
makes the sweep both fast and dense enough to actually load the device.

Semantics follow ``sklearn.mixture.GaussianMixture(covariance_type="diag")``:
the same ``reg_covar`` floor, the same mean-weighted M-step, and the same
``score_samples`` definition. ``tests/test_gpu_tokenizer.py`` checks agreement against
sklearn to tight tolerance for both scoring and a full EM step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def _kmeans_plusplus(
    values: torch.Tensor, n_components: int, generator: torch.Generator
) -> torch.Tensor:
    """Standard k-means++ seeding for one initialization."""
    count = values.shape[0]
    first = int(torch.randint(count, (1,), generator=generator, device=values.device).item())
    centers = [values[first]]
    closest = torch.sum((values - centers[0]) ** 2, dim=1)
    for _ in range(1, n_components):
        total = float(closest.sum())
        if not math.isfinite(total) or total <= 0.0:
            index = int(
                torch.randint(count, (1,), generator=generator, device=values.device).item()
            )
        else:
            index = int(torch.multinomial(closest, 1, generator=generator).item())
        centers.append(values[index])
        closest = torch.minimum(closest, torch.sum((values - centers[-1]) ** 2, dim=1))
    return torch.stack(centers)


@dataclass
class BatchedDiagonalGMM:
    """Fit ``n_init`` diagonal GMMs of the same size simultaneously.

    Attributes after :meth:`fit` mirror sklearn for the best initialization:
    ``weights_``, ``means_``, ``covariances_``, ``lower_bound_``, ``n_iter_``.
    """

    n_components: int
    n_init: int = 10
    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-4
    seed: int = 20260905
    device: str = "cuda:0"

    def _log_prob(
        self, values: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[init, rows, component]`` Gaussian log-densities."""
        precision = 1.0 / covariances
        # -0.5 * (D log 2pi + sum log var + sum (x-mu)^2 / var), expanded so the
        # data-dependent term is a single [rows, dim] x [dim, component] matmul.
        constant = values.shape[1] * math.log(2.0 * math.pi) + torch.sum(
            torch.log(covariances), dim=-1
        )
        squared = values * values
        quadratic = (
            squared @ precision.transpose(-1, -2)
            - 2.0 * (values @ (means * precision).transpose(-1, -2))
            + torch.sum(means * means * precision, dim=-1).unsqueeze(-2)
        )
        return -0.5 * (constant.unsqueeze(-2) + quadratic)

    def _estep(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = self._log_prob(values, means, covariances) + torch.log(weights).unsqueeze(-2)
        total = torch.logsumexp(joint, dim=-1)
        return joint - total.unsqueeze(-1), total

    def fit(self, values: np.ndarray) -> "BatchedDiagonalGMM":
        device = torch.device(self.device)
        data = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
        rows, dimensions = data.shape
        if rows < self.n_components * 10:
            raise ValueError("too few training rows for the requested vocabulary")

        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        means = torch.stack(
            [_kmeans_plusplus(data, self.n_components, generator) for _ in range(self.n_init)]
        )
        variance = data.var(dim=0, unbiased=False) + self.reg_covar
        covariances = variance.expand(self.n_init, self.n_components, dimensions).clone()
        weights = torch.full(
            (self.n_init, self.n_components), 1.0 / self.n_components, device=device
        )

        previous = torch.full((self.n_init,), -math.inf, device=device)
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            log_responsibility, total = self._estep(data, weights, means, covariances)
            bound = total.mean(dim=-1)
            responsibility = torch.exp(log_responsibility)
            counts = responsibility.sum(dim=-2) + 10.0 * torch.finfo(data.dtype).eps
            means = torch.einsum("irc,rd->icd", responsibility, data) / counts.unsqueeze(-1)
            squared = torch.einsum("irc,rd->icd", responsibility, data * data) / counts.unsqueeze(-1)
            covariances = squared - means * means + self.reg_covar
            weights = counts / rows
            if torch.all(torch.abs(bound - previous) < self.tol):
                previous = bound
                break
            previous = bound

        best = int(torch.argmax(previous).item())
        self.weights_ = weights[best].detach().cpu().numpy().astype(np.float64)
        self.means_ = means[best].detach().cpu().numpy().astype(np.float64)
        self.covariances_ = covariances[best].detach().cpu().numpy().astype(np.float64)
        self.lower_bound_ = float(previous[best])
        self.n_iter_ = iterations
        self.all_lower_bounds_ = previous.detach().cpu().numpy().astype(np.float64)
        return self

    def component_log_likelihood(self, values: np.ndarray, batch_rows: int = 262144) -> np.ndarray:
        """Return ``log p(x | component)`` without mixture weights, as ``[rows, K]``."""
        device = torch.device(self.device)
        means = torch.as_tensor(self.means_, dtype=torch.float32, device=device)
        covariances = torch.as_tensor(self.covariances_, dtype=torch.float32, device=device)
        output = np.empty((len(values), self.n_components), dtype=np.float32)
        for start in range(0, len(values), batch_rows):
            chunk = torch.as_tensor(
                np.asarray(values[start : start + batch_rows], dtype=np.float32), device=device
            )
            output[start : start + batch_rows] = (
                self._log_prob(chunk, means, covariances).cpu().numpy()
            )
        return output

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        joint = self.component_log_likelihood(values) + np.log(
            np.maximum(self.weights_, 1e-300)
        )[None, :]
        maximum = joint.max(axis=1, keepdims=True)
        return (maximum[:, 0] + np.log(np.exp(joint - maximum).sum(axis=1))).astype(np.float64)

    def predict(self, values: np.ndarray) -> np.ndarray:
        joint = self.component_log_likelihood(values) + np.log(
            np.maximum(self.weights_, 1e-300)
        )[None, :]
        return joint.argmax(axis=1).astype(np.int16)
