from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


@dataclass
class Preprocessor:
    pca_dim: int = 24
    seed: int = 20260905
    clip: float = 12.0

    def fit(self, values: np.ndarray) -> "Preprocessor":
        values = np.asarray(values, dtype=np.float32)
        self.scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
            unit_variance=False,
        )
        scaled = self.scaler.fit_transform(values).astype(np.float32, copy=False)
        np.clip(scaled, -self.clip, self.clip, out=scaled)
        dimensions = min(self.pca_dim, len(scaled) - 1, scaled.shape[1])
        if dimensions < 2:
            raise ValueError("not enough training data for PCA")
        self.pca = PCA(
            n_components=dimensions,
            whiten=True,
            svd_solver="randomized",
            random_state=self.seed,
        )
        self.pca.fit(scaled)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(values, dtype=np.float32)).astype(
            np.float32, copy=False
        )
        np.clip(scaled, -self.clip, self.clip, out=scaled)
        return self.pca.transform(scaled).astype(np.float32, copy=False)


@dataclass
class GMMTokenizer:
    n_words: int
    seed: int = 20260905
    max_iter: int = 150
    n_init: int = 2
    reg_covar: float = 1e-4
    unknown_weight: float = 1e-3
    unknown_quantile: float = 0.001

    def fit(self, values: np.ndarray) -> "GMMTokenizer":
        values = np.asarray(values, dtype=np.float32)
        if len(values) < self.n_words * 10:
            raise ValueError("too few training queries for requested vocabulary")
        if not 0.0 < self.unknown_weight < 1.0:
            raise ValueError("unknown_weight must lie in (0, 1)")
        self.model = GaussianMixture(
            n_components=self.n_words,
            covariance_type="diag",
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params="k-means++",
            random_state=self.seed,
        )
        self.model.fit(values)
        # A flat background component so a query far from every healthy word can be
        # reported as out-of-vocabulary instead of being forced onto its nearest word.
        _, _, lexical = self.transform(values)
        self.unknown_log_density_ = float(np.quantile(lexical, self.unknown_quantile))
        return self

    def unknown_posterior(self, lexical_log_likelihood: np.ndarray) -> np.ndarray:
        """Return ``P(out-of-vocabulary | g)`` against the flat background component."""
        if not hasattr(self, "unknown_log_density_"):
            raise ValueError("tokenizer must be fitted before scoring")
        known = np.log1p(-self.unknown_weight) + np.asarray(
            lexical_log_likelihood, dtype=np.float64
        )
        unknown = np.log(self.unknown_weight) + self.unknown_log_density_
        return np.exp(unknown - np.logaddexp(known, unknown))

    def component_log_likelihood(self, values: np.ndarray) -> np.ndarray:
        """Return log p(z | word), excluding mixture weights."""
        values = np.asarray(values, dtype=np.float64)
        mean = np.asarray(self.model.means_, dtype=np.float64)
        covariance = np.asarray(self.model.covariances_, dtype=np.float64)
        dimensions = values.shape[1]
        constant = dimensions * np.log(2.0 * np.pi) + np.log(covariance).sum(axis=1)
        quadratic = np.square(values[:, None, :] - mean[None, :, :]) / covariance[None, :, :]
        return -0.5 * (constant[None, :] + quadratic.sum(axis=2))

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        component = self.component_log_likelihood(values)
        log_joint = component + np.log(np.maximum(self.model.weights_, 1e-300))[None, :]
        words = np.argmax(log_joint, axis=1).astype(np.int16)
        lexical_log_likelihood = logsumexp(log_joint, axis=1)
        return words, component, lexical_log_likelihood

    def diagnostics(self, train_values: np.ndarray) -> dict[str, object]:
        words, _, lexical = self.transform(train_values)
        counts = np.bincount(words, minlength=self.n_words)
        return {
            "converged": bool(self.model.converged_),
            "iterations": int(self.model.n_iter_),
            "lower_bound": float(self.model.lower_bound_),
            "train_mean_nll": float(-lexical.mean()),
            "cluster_counts": counts.tolist(),
            "minimum_cluster_count": int(counts.min()),
            "effective_words": float(
                np.exp(
                    -np.sum(
                        (counts / counts.sum()) * np.log(np.maximum(counts / counts.sum(), 1e-12))
                    )
                )
            ),
        }
