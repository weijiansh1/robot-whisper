from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge


def episode_balanced_positions(
    starts: np.ndarray,
    lengths: np.ndarray,
    episode_indexes: np.ndarray,
    samples_per_episode: int,
) -> np.ndarray:
    """Choose roughly phase-uniform positions with equal episode contribution."""

    rows: list[np.ndarray] = []
    for episode_index in np.asarray(episode_indexes, dtype=np.int64):
        start = int(starts[episode_index])
        length = int(lengths[episode_index])
        if length <= 1:
            continue
        count = min(samples_per_episode, length - 1)
        positions = np.unique(np.rint(np.linspace(1, length - 1, count)).astype(np.int64))
        rows.append(start + positions)
    if not rows:
        raise ValueError("no episode positions are available")
    return np.concatenate(rows)


def causal_context_design(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    order: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    dimensions = values.shape[1]
    design = np.zeros((len(values), order * dimensions + order), dtype=np.float32)
    for episode_start, episode_length in zip(starts, lengths):
        start = int(episode_start)
        length = int(episode_length)
        for lag in range(1, order + 1):
            if length <= lag:
                continue
            target = slice(start + lag, start + length)
            source = slice(start, start + length - lag)
            left = (lag - 1) * dimensions
            design[target, left : left + dimensions] = values[source]
            design[target, order * dimensions + lag - 1] = 1.0
    return design


def _gaussian_parameters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    model = LedoitWolf().fit(np.asarray(values, dtype=np.float64))
    sign, precision_logdet = np.linalg.slogdet(model.precision_)
    if sign <= 0:
        raise ValueError("Gaussian precision is not positive definite")
    normalizer = 0.5 * (
        values.shape[1] * np.log(2.0 * np.pi) - precision_logdet
    )
    return np.asarray(model.location_), np.asarray(model.precision_), float(normalizer)


def _gaussian_nll(
    values: np.ndarray,
    center: np.ndarray,
    precision: np.ndarray,
    normalizer: float,
) -> np.ndarray:
    difference = np.asarray(values, dtype=np.float64) - center
    distance = np.einsum(
        "ij,jk,ik->i", difference, precision, difference, optimize=True
    )
    return ((0.5 * distance + normalizer) / difference.shape[1]).astype(np.float32)


@dataclass
class ContinuousVARGrammar:
    """Episode-balanced continuous autoregression with Gaussian innovations."""

    order: int = 4
    alpha: float = 20.0
    samples_per_episode: int = 8

    def fit(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episode_indexes: np.ndarray,
        *,
        device: str = "cpu",
    ) -> "ContinuousVARGrammar":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or self.order <= 0:
            raise ValueError("VAR values must be a matrix and order must be positive")
        rows = episode_balanced_positions(
            starts, lengths, episode_indexes, self.samples_per_episode
        )
        design = causal_context_design(values, starts, lengths, self.order)
        if device == "cpu":
            self.model_ = Ridge(alpha=self.alpha).fit(design[rows], values[rows])
            self.coef_ = np.asarray(self.model_.coef_, dtype=np.float64)
            self.intercept_ = np.asarray(self.model_.intercept_, dtype=np.float64)
        else:
            self.coef_, self.intercept_ = self._fit_ridge_torch(
                design[rows], values[rows], device
            )
            self.model_ = None
        residual = values[rows] - self._predict_numpy(design[rows])
        (
            self.residual_center_,
            self.residual_precision_,
            self.residual_normalizer_,
        ) = _gaussian_parameters(residual)
        (
            self.marginal_center_,
            self.marginal_precision_,
            self.marginal_normalizer_,
        ) = _gaussian_parameters(values[rows])
        self.fit_rows_ = rows
        return self

    def _fit_ridge_torch(
        self, design: np.ndarray, target: np.ndarray, device: str
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        torch_device = torch.device(device)
        with torch.inference_mode():
            x = torch.as_tensor(design, dtype=torch.float64, device=torch_device)
            y = torch.as_tensor(target, dtype=torch.float64, device=torch_device)
            x_mean = x.mean(dim=0)
            y_mean = y.mean(dim=0)
            centered_x = x - x_mean
            centered_y = y - y_mean
            gram = centered_x.T @ centered_x
            gram.diagonal().add_(self.alpha)
            coefficients = torch.linalg.solve(gram, centered_x.T @ centered_y)
            intercept = y_mean - x_mean @ coefficients
        return coefficients.T.cpu().numpy(), intercept.cpu().numpy()

    def _coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "coef_") and hasattr(self, "intercept_"):
            return np.asarray(self.coef_), np.asarray(self.intercept_)
        return np.asarray(self.model_.coef_), np.asarray(self.model_.intercept_)

    def _predict_numpy(self, design: np.ndarray) -> np.ndarray:
        coefficient, intercept = self._coefficients()
        return np.asarray(design, dtype=np.float64) @ coefficient.T + intercept

    def score(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        design = causal_context_design(values, starts, lengths, self.order)
        if device == "cpu":
            residual = values - self._predict_numpy(design)
            output = _gaussian_nll(
                residual,
                self.residual_center_,
                self.residual_precision_,
                self.residual_normalizer_,
            )
        else:
            output = self._score_torch(values, design, device)
        for start in np.asarray(starts, dtype=np.int64):
            output[start] = self.marginal_score(
                values[start : start + 1], device=device
            )[0]
        return output

    def _score_torch(
        self, values: np.ndarray, design: np.ndarray, device: str
    ) -> np.ndarray:
        import torch

        coefficient, intercept = self._coefficients()
        torch_device = torch.device(device)
        with torch.inference_mode():
            x = torch.as_tensor(design, dtype=torch.float32, device=torch_device)
            y = torch.as_tensor(values, dtype=torch.float32, device=torch_device)
            weight = torch.as_tensor(
                coefficient.T, dtype=torch.float32, device=torch_device
            )
            bias = torch.as_tensor(intercept, dtype=torch.float32, device=torch_device)
            residual = y - (x @ weight + bias)
            output = self._gaussian_nll_torch(
                residual,
                self.residual_center_,
                self.residual_precision_,
                self.residual_normalizer_,
            )
        return output.cpu().numpy()

    @staticmethod
    def _gaussian_nll_torch(
        values: object,
        center: np.ndarray,
        precision: np.ndarray,
        normalizer: float,
    ) -> object:
        import torch

        difference = values - torch.as_tensor(
            center, dtype=values.dtype, device=values.device
        )
        precision_tensor = torch.as_tensor(
            precision, dtype=values.dtype, device=values.device
        )
        distance = torch.einsum(
            "ij,jk,ik->i", difference, precision_tensor, difference
        )
        return (0.5 * distance + normalizer) / difference.shape[1]

    def marginal_score(
        self, values: np.ndarray, *, device: str = "cpu"
    ) -> np.ndarray:
        if device != "cpu":
            import torch

            torch_device = torch.device(device)
            with torch.inference_mode():
                tensor = torch.as_tensor(
                    values, dtype=torch.float32, device=torch_device
                )
                output = self._gaussian_nll_torch(
                    tensor,
                    self.marginal_center_,
                    self.marginal_precision_,
                    self.marginal_normalizer_,
                )
            return output.cpu().numpy()
        return _gaussian_nll(
            values,
            self.marginal_center_,
            self.marginal_precision_,
            self.marginal_normalizer_,
        )


def _defined_min(distances: np.ndarray) -> np.ndarray:
    """Row-wise minimum over defined lags, NaN where no lag is defined."""
    defined = np.isfinite(distances)
    filled = np.where(defined, distances, np.inf)
    return np.where(defined.any(axis=1), filled.min(axis=1), np.nan).astype(np.float32)


def causal_recurrence_features(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    max_lag: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lag-1 speed, lag-2..K recurrence, nearest prior distance, and validity.

    Early positions have no prior state at the required lag. Filling them with a large
    sentinel placed 8.3% of the calibration rows at an extreme value and compressed the
    usable percentile range, so those positions are masked instead. ``valid`` has one
    boolean column per returned statistic, in the order they are returned.
    """

    values = np.asarray(values, dtype=np.float32)
    missing = np.float32(np.nan)
    speed = np.full(len(values), missing, dtype=np.float32)
    periodic = np.full(len(values), missing, dtype=np.float32)
    nearest = np.full(len(values), missing, dtype=np.float32)
    for episode_start, episode_length in zip(starts, lengths):
        start = int(episode_start)
        length = int(episode_length)
        distances = np.full((length, max_lag), missing, dtype=np.float32)
        sequence = values[start : start + length]
        for lag in range(1, max_lag + 1):
            if length <= lag:
                continue
            distances[lag:, lag - 1] = np.sqrt(
                np.square(sequence[lag:] - sequence[:-lag]).mean(axis=1)
            )
        speed[start : start + length] = distances[:, 0]
        if max_lag > 1:
            periodic[start : start + length] = _defined_min(distances[:, 1:])
        nearest[start : start + length] = _defined_min(distances)
    valid = np.column_stack(
        [np.isfinite(speed), np.isfinite(periodic), np.isfinite(nearest)]
    )
    return speed, periodic, nearest, valid


def combine_overregularity(
    freeze_percentile: np.ndarray,
    recurrence_percentile: np.ndarray,
    low_surprise_percentile: np.ndarray,
) -> np.ndarray:
    """Require two agreeing symptoms so one naturally slow query is not a lock-in."""

    components = np.column_stack(
        [freeze_percentile, recurrence_percentile, low_surprise_percentile]
    )
    components.sort(axis=1)
    return components[:, -2:].mean(axis=1, dtype=np.float32)
