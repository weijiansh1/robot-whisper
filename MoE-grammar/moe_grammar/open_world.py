from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

from moe_grammar.statistics import empirical_percentile


PHENOTYPE_NAMES = (
    "entropy_mean",
    "entropy_terminal_late_layers",
    "entropy_late_flow_late_layers",
    "entropy_flow_slope_late_layers",
    "margin_mean",
    "margin_terminal_late_layers",
    "margin_flow_slope_late_layers",
    "top1_terminal_late_layers",
    "top4_terminal_late_layers",
    "soft_consensus_mean",
    "soft_consensus_terminal_late_layers",
    "top4_consensus_terminal_late_layers",
    "effective_rank_mean",
    "effective_rank_terminal_late_layers",
    "velocity_mean",
    "velocity_late_flow_late_layers",
    "support_switch_mean",
    "support_switch_late_flow_late_layers",
    "acceleration_mean",
    "acceleration_late_flow_late_layers",
    "layer_profile_disagreement_mean",
    "layer_profile_disagreement_terminal",
)

AXIS_NAMES = (
    "switching",
    "lock_in",
    "flattening",
    "synchronization",
    "recurrence",
    "phase_stall",
    "off_manifold",
)


def _metric_columns(feature_names: tuple[str, ...], metric: str) -> np.ndarray:
    columns = [
        index
        for index, name in enumerate(feature_names)
        if name == metric or name.startswith(f"{metric}|layer_")
    ]
    if not columns:
        raise ValueError(f"missing routing metric {metric}")
    return np.asarray(columns, dtype=np.int64)


def _mean(
    features: np.ndarray,
    columns: np.ndarray,
    flows: slice | np.ndarray | None = None,
) -> np.ndarray:
    selected = features[:, :, columns]
    if flows is not None:
        selected = selected[:, flows]
    return selected.mean(axis=(1, 2), dtype=np.float32)


def _flow_slope(features: np.ndarray, columns: np.ndarray) -> np.ndarray:
    values = features[:, :, columns].mean(axis=2, dtype=np.float32)
    time = np.arange(values.shape[1], dtype=np.float32)
    time -= time.mean()
    return (values * time[None, :]).sum(axis=1) / np.square(time).sum()


def _layer_profile_disagreement(
    features: np.ndarray, feature_names: tuple[str, ...]
) -> np.ndarray:
    metrics = (
        "entropy",
        "margin",
        "top1_mass",
        "top4_mass",
        "soft_token_consensus",
        "top4_token_consensus",
        "effective_rank",
    )
    profile = np.stack(
        [features[:, :, _metric_columns(feature_names, metric)] for metric in metrics],
        axis=-1,
    )
    return profile.std(axis=2, dtype=np.float32).mean(axis=-1, dtype=np.float32)


def build_query_phenotypes(features: np.ndarray, feature_names: tuple[str, ...]) -> np.ndarray:
    """Map a [query,flow,feature] chord to interpretable MoE-only coordinates."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[1] != 10:
        raise ValueError("expected [query,10,feature] routing chords")
    late_layers = np.asarray([6, 7], dtype=np.int64)
    late_flow = slice(7, 10)
    terminal = slice(9, 10)

    entropy = _metric_columns(feature_names, "entropy")
    margin = _metric_columns(feature_names, "margin")
    top1 = _metric_columns(feature_names, "top1_mass")
    top4 = _metric_columns(feature_names, "top4_mass")
    soft_consensus = _metric_columns(feature_names, "soft_token_consensus")
    top4_consensus = _metric_columns(feature_names, "top4_token_consensus")
    effective_rank = _metric_columns(feature_names, "effective_rank")
    velocity = _metric_columns(feature_names, "flow_velocity")
    support_switch = _metric_columns(feature_names, "flow_top4_switch")
    acceleration = _metric_columns(feature_names, "flow_acceleration")
    disagreement = _layer_profile_disagreement(features, feature_names)

    values = (
        _mean(features, entropy),
        _mean(features, entropy[late_layers], terminal),
        _mean(features, entropy[late_layers], late_flow),
        _flow_slope(features, entropy[late_layers]),
        _mean(features, margin),
        _mean(features, margin[late_layers], terminal),
        _flow_slope(features, margin[late_layers]),
        _mean(features, top1[late_layers], terminal),
        _mean(features, top4[late_layers], terminal),
        _mean(features, soft_consensus),
        _mean(features, soft_consensus[late_layers], terminal),
        _mean(features, top4_consensus[late_layers], terminal),
        _mean(features, effective_rank),
        _mean(features, effective_rank[late_layers], terminal),
        _mean(features, velocity),
        _mean(features, velocity[late_layers], late_flow),
        _mean(features, support_switch),
        _mean(features, support_switch[late_layers], late_flow),
        _mean(features, acceleration),
        _mean(features, acceleration[late_layers], late_flow),
        disagreement.mean(axis=1, dtype=np.float32),
        disagreement[:, terminal].mean(axis=1, dtype=np.float32),
    )
    output = np.column_stack(values).astype(np.float32)
    if output.shape[1] != len(PHENOTYPE_NAMES) or not np.isfinite(output).all():
        raise AssertionError("invalid phenotype construction")
    return output


def phenotypes_in_batches(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    batch_size: int = 8192,
) -> np.ndarray:
    output = np.empty((len(features), len(PHENOTYPE_NAMES)), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        output[start:stop] = build_query_phenotypes(features[start:stop], feature_names)
    return output


@dataclass
class TaskRobustScaler:
    clip: float = 10.0

    def fit(
        self,
        values: np.ndarray,
        task: np.ndarray,
        rows: np.ndarray,
        task_count: int,
    ) -> "TaskRobustScaler":
        values = np.asarray(values, dtype=np.float64)
        task = np.asarray(task)
        rows = np.asarray(rows, dtype=np.int64)
        self.center_ = np.empty((task_count, values.shape[1]), dtype=np.float32)
        self.scale_ = np.empty_like(self.center_)
        global_center = np.median(values[rows], axis=0)
        global_low, global_high = np.quantile(values[rows], [0.25, 0.75], axis=0)
        global_scale = np.maximum(global_high - global_low, 1e-5)
        for task_index in range(task_count):
            selected = rows[task[rows] == task_index]
            if len(selected) < 64:
                center, scale = global_center, global_scale
            else:
                center = np.median(values[selected], axis=0)
                low, high = np.quantile(values[selected], [0.25, 0.75], axis=0)
                scale = np.maximum(high - low, global_scale * 0.02)
            self.center_[task_index] = center
            self.scale_[task_index] = scale
        return self

    def transform(self, values: np.ndarray, task: np.ndarray) -> np.ndarray:
        output = (
            np.asarray(values, dtype=np.float32) - self.center_[np.asarray(task)]
        ) / self.scale_[np.asarray(task)]
        return np.clip(output, -self.clip, self.clip).astype(np.float32)


@dataclass
class RoutingPhaseModel:
    alpha: float = 10.0

    def fit(
        self,
        values: np.ndarray,
        task: np.ndarray,
        progress: np.ndarray,
        rows: np.ndarray,
        task_count: int,
    ) -> "RoutingPhaseModel":
        values = np.asarray(values, dtype=np.float32)
        task = np.asarray(task)
        progress = np.asarray(progress, dtype=np.float32)
        rows = np.asarray(rows, dtype=np.int64)
        self.coefficient_ = np.zeros((task_count, values.shape[1]), dtype=np.float32)
        self.intercept_ = np.zeros(task_count, dtype=np.float32)
        global_model = Ridge(alpha=self.alpha).fit(values[rows], progress[rows])
        for task_index in range(task_count):
            selected = rows[task[rows] == task_index]
            model = (
                Ridge(alpha=self.alpha).fit(values[selected], progress[selected])
                if len(selected) >= 64
                else global_model
            )
            self.coefficient_[task_index] = model.coef_
            self.intercept_[task_index] = model.intercept_
        return self

    def predict(self, values: np.ndarray, task: np.ndarray) -> np.ndarray:
        task = np.asarray(task)
        prediction = (np.asarray(values, dtype=np.float32) * self.coefficient_[task]).sum(
            axis=1
        ) + self.intercept_[task]
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)


@dataclass
class HealthyPrefixGrammar:
    """Shared healthy phase grammar with exact causal forward filtering.

    Progress is an offline supervision target for naming the latent phase states.
    At inference, the forward belief and predictive density consume routing
    phenotypes only. The belief at q therefore summarizes every observed word
    before q rather than a fixed local suffix.
    """

    phase_states: int = 8
    min_support: int = 128
    transition_alpha: float = 0.05
    samples_per_episode: int | None = None
    equal_episode_transitions: bool = False

    def fit(
        self,
        values: np.ndarray,
        progress: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episode_indexes: np.ndarray,
    ) -> "HealthyPrefixGrammar":
        values = np.asarray(values, dtype=np.float64)
        progress = np.asarray(progress, dtype=np.float64)
        episode_indexes = np.asarray(episode_indexes, dtype=np.int64)
        if self.samples_per_episode is None:
            rows = np.concatenate(
                [
                    np.arange(starts[index], starts[index] + lengths[index])
                    for index in episode_indexes
                ]
            ).astype(np.int64)
        else:
            sampled_rows = []
            for index in episode_indexes:
                start = int(starts[index])
                length = int(lengths[index])
                count = min(self.samples_per_episode, length)
                offsets = np.unique(
                    np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
                )
                sampled_rows.append(start + offsets)
            rows = np.concatenate(sampled_rows).astype(np.int64)
        labels = self.phase_bin(progress)
        fallback = LedoitWolf().fit(values[rows])
        self.centers_ = np.empty((self.phase_states, values.shape[1]), dtype=np.float64)
        self.precisions_ = np.empty(
            (self.phase_states, values.shape[1], values.shape[1]), dtype=np.float64
        )
        self.log_normalizers_ = np.empty(self.phase_states, dtype=np.float64)
        for state in range(self.phase_states):
            selected = rows[labels[rows] == state]
            model = (
                LedoitWolf().fit(values[selected])
                if len(selected) >= self.min_support
                else fallback
            )
            self.centers_[state] = model.location_
            self.precisions_[state] = model.precision_
            sign, precision_logdet = np.linalg.slogdet(model.precision_)
            if sign <= 0:
                raise ValueError("healthy prefix emission precision is not positive")
            self.log_normalizers_[state] = 0.5 * (
                values.shape[1] * np.log(2.0 * np.pi) - precision_logdet
            )

        initial = np.full(self.phase_states, self.transition_alpha, dtype=np.float64)
        transition = np.full(
            (self.phase_states, self.phase_states),
            self.transition_alpha,
            dtype=np.float64,
        )
        for episode_index in episode_indexes:
            start = int(starts[episode_index])
            length = int(lengths[episode_index])
            sequence = labels[start : start + length]
            if not len(sequence):
                continue
            initial[sequence[0]] += 1.0
            if len(sequence) > 1:
                weight = 1.0 / (len(sequence) - 1) if self.equal_episode_transitions else 1.0
                np.add.at(transition, (sequence[:-1], sequence[1:]), weight)
        self.initial_ = initial / initial.sum()
        self.transition_ = transition / transition.sum(axis=1, keepdims=True)
        self.phase_centers_ = (
            np.arange(self.phase_states, dtype=np.float64) + 0.5
        ) / self.phase_states
        return self

    def phase_bin(self, progress: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(progress) * self.phase_states).astype(np.int16),
            0,
            self.phase_states - 1,
        )

    def emission_log_likelihood(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        output = np.empty((len(values), self.phase_states), dtype=np.float64)
        for state in range(self.phase_states):
            difference = values - self.centers_[state]
            distance = np.einsum(
                "ij,jk,ik->i",
                difference,
                self.precisions_[state],
                difference,
                optimize=True,
            )
            output[:, state] = -0.5 * distance - self.log_normalizers_[state]
        return output

    def step(
        self,
        value: np.ndarray,
        previous_belief: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, float, float]:
        """Update one online prefix and return surprise, belief, phase, entropy."""

        observation = np.asarray(value, dtype=np.float64)
        if observation.shape != self.centers_.shape[1:]:
            raise ValueError("online routing word has the wrong dimension")
        if previous_belief is None:
            predictive = self.initial_
        else:
            belief = np.asarray(previous_belief, dtype=np.float64)
            if belief.shape != self.initial_.shape:
                raise ValueError("online phase belief has the wrong dimension")
            predictive = belief @ self.transition_
        emission = self.emission_log_likelihood(observation[None, :])[0]
        log_joint = np.log(np.maximum(predictive, 1e-300)) + emission
        log_density = logsumexp(log_joint)
        posterior = np.exp(log_joint - log_density)
        surprise = float(-log_density / observation.size)
        phase = float(posterior @ self.phase_centers_)
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-300))))
        return surprise, posterior.astype(np.float32), phase, entropy

    def score_candidates(
        self,
        values: np.ndarray,
        previous_belief: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score alternative current words against one frozen causal prefix.

        Every row in ``values`` is treated as a mutually exclusive continuation of
        the same prefix.  Candidate observations therefore do not update one another.
        """

        observations = np.asarray(values, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1:] != self.centers_.shape[1:]:
            raise ValueError("candidate routing words have the wrong shape")
        if previous_belief is None:
            predictive = self.initial_
        else:
            belief = np.asarray(previous_belief, dtype=np.float64)
            if belief.shape != self.initial_.shape:
                raise ValueError("online phase belief has the wrong dimension")
            predictive = belief @ self.transition_
        emission = self.emission_log_likelihood(observations)
        log_joint = np.log(np.maximum(predictive, 1e-300))[None, :] + emission
        log_density = logsumexp(log_joint, axis=1)
        posterior = np.exp(log_joint - log_density[:, None])
        surprise = -log_density / observations.shape[1]
        phase = posterior @ self.phase_centers_
        entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1e-300)), axis=1)
        return (
            surprise.astype(np.float32),
            posterior.astype(np.float32),
            phase.astype(np.float32),
            entropy.astype(np.float32),
        )

    def score(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        update_with_observations: bool = True,
        device: str = "cpu",
        emission_batch_size: int = 262_144,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictive surprise, phase belief, and belief entropy at every q."""

        if device != "cpu":
            return self._score_torch(
                values,
                starts,
                lengths,
                update_with_observations=update_with_observations,
                device=device,
                emission_batch_size=emission_batch_size,
            )

        emissions = self.emission_log_likelihood(values)
        surprise = np.empty(len(values), dtype=np.float32)
        filtered_phase = np.empty(len(values), dtype=np.float32)
        belief_entropy = np.empty(len(values), dtype=np.float32)
        dimensions = np.asarray(values).shape[1]
        for episode_start, length in zip(starts, lengths):
            start = int(episode_start)
            stop = start + int(length)
            belief = self.initial_.copy()
            for row in range(start, stop):
                predictive = belief if row == start else belief @ self.transition_
                log_joint = np.log(np.maximum(predictive, 1e-300)) + emissions[row]
                log_density = logsumexp(log_joint)
                posterior = np.exp(log_joint - log_density)
                surprise[row] = -log_density / dimensions
                filtered_phase[row] = posterior @ self.phase_centers_
                belief_entropy[row] = -np.sum(posterior * np.log(np.maximum(posterior, 1e-300)))
                belief = posterior if update_with_observations else predictive
        return surprise, filtered_phase, belief_entropy

    def _score_torch(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        update_with_observations: bool,
        device: str,
        emission_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        torch_device = torch.device(device)
        values = np.asarray(values, dtype=np.float32)
        starts = np.asarray(starts, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        dimensions = values.shape[1]
        with torch.inference_mode():
            centers = torch.as_tensor(self.centers_, dtype=torch.float32, device=torch_device)
            precisions = torch.as_tensor(self.precisions_, dtype=torch.float32, device=torch_device)
            normalizers = torch.as_tensor(
                self.log_normalizers_, dtype=torch.float32, device=torch_device
            )
            emissions = torch.empty(
                (len(values), self.phase_states),
                dtype=torch.float32,
                device=torch_device,
            )
            for batch_start in range(0, len(values), emission_batch_size):
                batch_stop = min(batch_start + emission_batch_size, len(values))
                batch = torch.as_tensor(
                    values[batch_start:batch_stop],
                    dtype=torch.float32,
                    device=torch_device,
                )
                difference = batch[:, None, :] - centers[None, :, :]
                distance = torch.einsum("bsi,sij,bsj->bs", difference, precisions, difference)
                emissions[batch_start:batch_stop] = -0.5 * distance - normalizers[None, :]

            starts_tensor = torch.as_tensor(starts, device=torch_device)
            lengths_tensor = torch.as_tensor(lengths, device=torch_device)
            initial = torch.as_tensor(self.initial_, dtype=torch.float32, device=torch_device)
            transition = torch.as_tensor(self.transition_, dtype=torch.float32, device=torch_device)
            phase_centers = torch.as_tensor(
                self.phase_centers_, dtype=torch.float32, device=torch_device
            )
            belief = initial[None, :].expand(len(starts), -1).clone()
            surprise = torch.empty(len(values), dtype=torch.float32, device=torch_device)
            filtered_phase = torch.empty_like(surprise)
            belief_entropy = torch.empty_like(surprise)
            for position in range(int(lengths.max())):
                active = lengths_tensor > position
                rows = starts_tensor[active] + position
                active_belief = belief[active]
                predictive = active_belief if position == 0 else active_belief @ transition
                log_joint = torch.log(predictive.clamp_min(1e-30)) + emissions[rows]
                log_density = torch.logsumexp(log_joint, dim=1)
                posterior = torch.exp(log_joint - log_density[:, None])
                surprise[rows] = -log_density / dimensions
                filtered_phase[rows] = posterior @ phase_centers
                belief_entropy[rows] = -torch.sum(
                    posterior * torch.log(posterior.clamp_min(1e-30)), dim=1
                )
                belief[active] = posterior if update_with_observations else predictive
        return (
            surprise.cpu().numpy(),
            filtered_phase.cpu().numpy(),
            belief_entropy.cpu().numpy(),
        )


def build_dynamic_features(
    scaled: np.ndarray,
    phase: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Build causal state/transition/recurrence features from complete observed prefixes."""

    scaled = np.asarray(scaled, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    dimensions = scaled.shape[1]
    output = np.zeros((len(scaled), 2 * dimensions + 9), dtype=np.float32)
    output[:, :dimensions] = scaled
    for episode_start, length in zip(starts, lengths):
        start = int(episode_start)
        stop = start + int(length)
        values = scaled[start:stop]
        current_phase = phase[start:stop]
        if len(values) > 1:
            output[start + 1 : stop, dimensions : 2 * dimensions] = np.diff(values, axis=0)
        recurrence_start = 2 * dimensions
        for lag in (1, 2, 3):
            if len(values) > lag:
                distance = np.sqrt(np.square(values[lag:] - values[:-lag]).mean(axis=1))
                output[start + lag : stop, recurrence_start + lag - 1] = distance
                output[start + lag : stop, recurrence_start + 3 + lag - 1] = 1.0
        phase_column = recurrence_start + 6
        output[start:stop, phase_column] = current_phase
        if len(values) > 1:
            velocity = np.diff(current_phase)
            output[start + 1 : stop, phase_column + 1] = velocity
        if len(values) > 2:
            output[start + 2 : stop, phase_column + 2] = np.diff(current_phase, n=2)
    return np.clip(output, -12.0, 12.0)


@dataclass
class HealthyDynamicsModel:
    phase_bins: int = 5
    min_support: int = 64

    def fit(
        self,
        values: np.ndarray,
        task: np.ndarray,
        phase: np.ndarray,
        rows: np.ndarray,
        task_count: int,
    ) -> "HealthyDynamicsModel":
        values = np.asarray(values, dtype=np.float64)
        task = np.asarray(task)
        phase_bin = self.phase_bin(phase)
        rows = np.asarray(rows, dtype=np.int64)
        self.models_: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.task_models_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        global_model = LedoitWolf().fit(values[rows])
        self.global_model_ = (global_model.location_, global_model.precision_)
        for task_index in range(task_count):
            task_rows = rows[task[rows] == task_index]
            task_model = (
                LedoitWolf().fit(values[task_rows])
                if len(task_rows) >= self.min_support
                else global_model
            )
            self.task_models_[task_index] = (
                np.asarray(task_model.location_),
                np.asarray(task_model.precision_),
            )
            for bin_index in range(self.phase_bins):
                selected = task_rows[phase_bin[task_rows] == bin_index]
                if len(selected) >= self.min_support:
                    model = LedoitWolf().fit(values[selected])
                    self.models_[(task_index, bin_index)] = (
                        np.asarray(model.location_),
                        np.asarray(model.precision_),
                    )
        return self

    def phase_bin(self, phase: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(phase) * self.phase_bins).astype(np.int16),
            0,
            self.phase_bins - 1,
        )

    def score(self, values: np.ndarray, task: np.ndarray, phase: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        task = np.asarray(task)
        phase_bin = self.phase_bin(phase)
        output = np.empty(len(values), dtype=np.float64)
        for task_index in np.unique(task):
            for bin_index in np.unique(phase_bin[task == task_index]):
                selected = (task == task_index) & (phase_bin == bin_index)
                center, precision = self.models_.get(
                    (int(task_index), int(bin_index)),
                    self.task_models_.get(int(task_index), self.global_model_),
                )
                difference = values[selected] - center
                output[selected] = (
                    np.einsum("ij,jk,ik->i", difference, precision, difference, optimize=True)
                    / values.shape[1]
                )
        return output.astype(np.float32)


@dataclass
class PhaseConditionalCDF:
    phase_bins: int = 5
    min_support: int = 64

    def fit(
        self,
        values: np.ndarray,
        task: np.ndarray,
        phase: np.ndarray,
        rows: np.ndarray,
    ) -> "PhaseConditionalCDF":
        values = np.asarray(values, dtype=np.float64)
        task = np.asarray(task)
        phase_bin = self.phase_bin(phase)
        rows = np.asarray(rows, dtype=np.int64)
        self.global_reference_ = np.sort(values[rows])
        self.task_references_: dict[int, np.ndarray] = {}
        self.references_: dict[tuple[int, int], np.ndarray] = {}
        for task_index in np.unique(task[rows]):
            task_rows = rows[task[rows] == task_index]
            self.task_references_[int(task_index)] = np.sort(values[task_rows])
            for bin_index in range(self.phase_bins):
                selected = task_rows[phase_bin[task_rows] == bin_index]
                if len(selected) >= self.min_support:
                    self.references_[(int(task_index), bin_index)] = np.sort(values[selected])
        return self

    def phase_bin(self, phase: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(phase) * self.phase_bins).astype(np.int16),
            0,
            self.phase_bins - 1,
        )

    def transform(self, values: np.ndarray, task: np.ndarray, phase: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        task = np.asarray(task)
        phase_bin = self.phase_bin(phase)
        output = np.empty(len(values), dtype=np.float64)
        for task_index in np.unique(task):
            task_reference = self.task_references_.get(int(task_index), self.global_reference_)
            for bin_index in np.unique(phase_bin[task == task_index]):
                selected = (task == task_index) & (phase_bin == bin_index)
                reference = self.references_.get((int(task_index), int(bin_index)), task_reference)
                output[selected] = empirical_percentile(reference, values[selected])
        return output.astype(np.float32)


def causal_dwell(
    percentile: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    output = np.zeros(len(percentile), dtype=np.int16)
    for episode_start, length in zip(starts, lengths):
        start = int(episode_start)
        stop = start + int(length)
        current = 0
        for row in range(start, stop):
            current = current + 1 if percentile[row] >= threshold else 0
            output[row] = current
    return output


@dataclass
class HealthyReturnTable:
    horizon: int = 3
    return_threshold: float = 0.8
    dwell_threshold: float = 0.9
    min_support: int = 32
    alpha: float = 2.0

    level_edges = np.asarray([0.0, 0.8, 0.9, 0.95, 0.98, 1.01])

    def _states(
        self,
        percentile: np.ndarray,
        phase_velocity: np.ndarray,
        dwell: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        level = np.clip(
            np.digitize(percentile, self.level_edges) - 1,
            0,
            len(self.level_edges) - 2,
        )
        dwell_bin = np.clip(dwell, 0, 3)
        phase_bin = np.digitize(phase_velocity, [-0.02, 0.02])
        return level.astype(np.int8), dwell_bin.astype(np.int8), phase_bin.astype(np.int8)

    def fit(
        self,
        percentile: np.ndarray,
        phase_velocity: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episode_indexes: np.ndarray,
    ) -> "HealthyReturnTable":
        dwell = causal_dwell(percentile, starts, lengths, threshold=self.dwell_threshold)
        level, dwell_bin, phase_bin = self._states(percentile, phase_velocity, dwell)
        shape = (len(self.level_edges) - 1, 4, 3)
        self.total_ = np.zeros(shape, dtype=np.int64)
        self.returned_ = np.zeros(shape, dtype=np.int64)
        self.backoff_total_ = np.zeros(shape[:2], dtype=np.int64)
        self.backoff_returned_ = np.zeros(shape[:2], dtype=np.int64)
        self.level_total_ = np.zeros(shape[0], dtype=np.int64)
        self.level_returned_ = np.zeros(shape[0], dtype=np.int64)
        for episode_index in np.asarray(episode_indexes, dtype=np.int64):
            start = int(starts[episode_index])
            length = int(lengths[episode_index])
            for offset in range(max(0, length - self.horizon)):
                row = start + offset
                future = percentile[row + 1 : row + self.horizon + 1]
                returned = bool(np.any(future < self.return_threshold))
                key = (level[row], dwell_bin[row], phase_bin[row])
                self.total_[key] += 1
                self.returned_[key] += int(returned)
                short_key = key[:2]
                self.backoff_total_[short_key] += 1
                self.backoff_returned_[short_key] += int(returned)
                self.level_total_[key[0]] += 1
                self.level_returned_[key[0]] += int(returned)
        self.global_probability_ = float(
            (self.level_returned_.sum() + self.alpha) / (self.level_total_.sum() + 2.0 * self.alpha)
        )
        return self

    def predict(
        self,
        percentile: np.ndarray,
        phase_velocity: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dwell = causal_dwell(percentile, starts, lengths, threshold=self.dwell_threshold)
        level, dwell_bin, phase_bin = self._states(percentile, phase_velocity, dwell)
        output = np.empty(len(percentile), dtype=np.float32)
        for row, key in enumerate(zip(level, dwell_bin, phase_bin)):
            if self.total_[key] >= self.min_support:
                returned, total = self.returned_[key], self.total_[key]
            elif self.backoff_total_[key[:2]] >= self.min_support:
                returned = self.backoff_returned_[key[:2]]
                total = self.backoff_total_[key[:2]]
            elif self.level_total_[key[0]] >= self.min_support:
                returned = self.level_returned_[key[0]]
                total = self.level_total_[key[0]]
            else:
                output[row] = self.global_probability_
                continue
            output[row] = (returned + self.alpha) / (total + 2.0 * self.alpha)
        return output, dwell


def open_set_scores(
    percentile: np.ndarray,
    return_probability: np.ndarray,
    dwell: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    deviation = np.clip((np.asarray(percentile) - 0.8) / 0.2, 0.0, 1.0)
    learned = deviation * (1.0 - np.asarray(return_probability))
    dwell_score = deviation * np.minimum(np.asarray(dwell) / 3.0, 1.0)
    ewma = np.zeros(len(percentile), dtype=np.float32)
    for episode_start, length in zip(starts, lengths):
        start = int(episode_start)
        stop = start + int(length)
        current = 0.0
        for row in range(start, stop):
            current = 0.4 * float(deviation[row]) + 0.6 * current
            ewma[row] = current
    return {
        "instant": np.asarray(percentile, dtype=np.float32),
        "learned_persistent": learned.astype(np.float32),
        "dwell_persistent": dwell_score.astype(np.float32),
        "ewma_persistent": ewma,
    }


def phenotype_axes(
    scaled_base: np.ndarray,
    dynamic: np.ndarray,
    off_manifold: np.ndarray,
) -> np.ndarray:
    index = {name: position for position, name in enumerate(PHENOTYPE_NAMES)}
    values = np.asarray(scaled_base, dtype=np.float32)
    dimensions = values.shape[1]
    recurrence = dynamic[:, 2 * dimensions : 2 * dimensions + 3]
    phase_velocity = dynamic[:, 2 * dimensions + 7]
    switching = np.mean(
        values[
            :,
            [
                index["velocity_late_flow_late_layers"],
                index["support_switch_late_flow_late_layers"],
                index["acceleration_late_flow_late_layers"],
            ],
        ],
        axis=1,
    )
    lock_in = np.mean(
        np.column_stack(
            [
                values[:, index["margin_terminal_late_layers"]],
                values[:, index["top1_terminal_late_layers"]],
                values[:, index["top4_terminal_late_layers"]],
                -values[:, index["velocity_late_flow_late_layers"]],
                -values[:, index["acceleration_late_flow_late_layers"]],
            ]
        ),
        axis=1,
    )
    flattening = np.mean(
        np.column_stack(
            [
                values[:, index["entropy_terminal_late_layers"]],
                -values[:, index["margin_terminal_late_layers"]],
                -values[:, index["top1_terminal_late_layers"]],
            ]
        ),
        axis=1,
    )
    synchronization = np.mean(
        np.column_stack(
            [
                values[:, index["soft_consensus_terminal_late_layers"]],
                values[:, index["top4_consensus_terminal_late_layers"]],
                -values[:, index["effective_rank_terminal_late_layers"]],
            ]
        ),
        axis=1,
    )
    recurrence_axis = recurrence[:, 0] - np.minimum(recurrence[:, 1], recurrence[:, 2])
    return np.column_stack(
        [
            switching,
            lock_in,
            flattening,
            synchronization,
            recurrence_axis,
            -phase_velocity,
            off_manifold,
        ]
    ).astype(np.float32)


def causal_window_signatures(
    axes: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """Current/mean/slope signature using only the trailing phenotype window."""

    output = np.empty((len(axes), axes.shape[1] * 3), dtype=np.float32)
    for episode_start, length in zip(starts, lengths):
        start = int(episode_start)
        stop = start + int(length)
        for row in range(start, stop):
            left = max(start, row - window + 1)
            sequence = axes[left : row + 1]
            output[row, : axes.shape[1]] = axes[row]
            output[row, axes.shape[1] : 2 * axes.shape[1]] = sequence.mean(axis=0)
            if len(sequence) > 1:
                time = np.arange(len(sequence), dtype=np.float32)
                time -= time.mean()
                slope = (sequence * time[:, None]).sum(axis=0) / np.square(time).sum()
            else:
                slope = 0.0
            output[row, 2 * axes.shape[1] :] = slope
    return output


@dataclass
class AnchorTemplate:
    offsets: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)

    def fit(
        self,
        signatures: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        onsets: np.ndarray,
        episode_indexes: np.ndarray,
    ) -> "AnchorTemplate":
        by_offset: dict[int, list[np.ndarray]] = {offset: [] for offset in self.offsets}
        for episode_index in np.asarray(episode_indexes, dtype=np.int64):
            onset = int(onsets[episode_index])
            if onset < 0:
                continue
            for offset in self.offsets:
                position = onset + offset
                if 0 <= position < lengths[episode_index]:
                    by_offset[offset].append(signatures[starts[episode_index] + position])
        prototypes = []
        residuals = []
        for offset in self.offsets:
            values = np.asarray(by_offset[offset], dtype=np.float64)
            if len(values) < 4:
                continue
            prototype = values.mean(axis=0)
            prototypes.append(prototype)
            residuals.append(values - prototype)
        if not prototypes:
            raise ValueError("no anchor windows available")
        self.prototypes_ = np.asarray(prototypes)
        residual = np.concatenate(residuals)
        covariance = LedoitWolf(assume_centered=True).fit(residual)
        self.precision_ = covariance.precision_
        return self

    def score(self, signatures: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        output = np.empty(len(signatures), dtype=np.float32)
        for start in range(0, len(signatures), batch_size):
            stop = min(start + batch_size, len(signatures))
            difference = (
                np.asarray(signatures[start:stop], dtype=np.float64)[:, None, :]
                - self.prototypes_[None, :, :]
            )
            distance = (
                np.einsum("bki,ij,bkj->bk", difference, self.precision_, difference, optimize=True)
                / signatures.shape[1]
            )
            output[start:stop] = -distance.min(axis=1)
        return output
