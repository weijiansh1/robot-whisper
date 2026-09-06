"""Interpretable multitrack symbols and causal categorical grammars.

The symbols in this module are deliberately stateless by default.  That keeps
sequence structure out of the tokenizer so that any measured order gain must
come from the grammar rather than from hysteresis in the discretizer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .open_world import PHENOTYPE_NAMES


TRACK_NAMES = (
    "sharpness",
    "synchronization",
    "rank",
    "switching",
    "flow_sharpening",
)

TRACK_SYMBOLS = (
    ("Fl", "", "Sh"),
    ("Ds", "", "Sy"),
    ("LR", "", "HR"),
    ("Lk", "", "Sw"),
    ("Ff", "", "Fs"),
)


def _column(values: np.ndarray, name: str) -> np.ndarray:
    return values[:, PHENOTYPE_NAMES.index(name)]


def phenotype_track_scores(scaled_phenotypes: np.ndarray) -> np.ndarray:
    """Collapse the clean 22-D phenotype into five signed, interpretable tracks.

    Input features must already be robustly scaled using healthy training data.
    Positive values mean the high-end symbol in ``TRACK_SYMBOLS``.
    """

    values = np.asarray(scaled_phenotypes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(PHENOTYPE_NAMES):
        raise ValueError(
            f"expected [query, {len(PHENOTYPE_NAMES)}] phenotypes, got {values.shape}"
        )

    sharpness = np.mean(
        np.column_stack(
            [
                -_column(values, "entropy_terminal_late_layers"),
                _column(values, "margin_terminal_late_layers"),
                _column(values, "top1_terminal_late_layers"),
                _column(values, "top4_terminal_late_layers"),
            ]
        ),
        axis=1,
    )
    synchronization = np.mean(
        np.column_stack(
            [
                _column(values, "soft_consensus_terminal_late_layers"),
                _column(values, "top4_consensus_terminal_late_layers"),
            ]
        ),
        axis=1,
    )
    rank = np.mean(
        np.column_stack(
            [
                _column(values, "effective_rank_mean"),
                _column(values, "effective_rank_terminal_late_layers"),
            ]
        ),
        axis=1,
    )
    switching = np.mean(
        np.column_stack(
            [
                _column(values, "velocity_mean"),
                _column(values, "velocity_late_flow_late_layers"),
                _column(values, "support_switch_mean"),
                _column(values, "support_switch_late_flow_late_layers"),
                _column(values, "acceleration_mean"),
                _column(values, "acceleration_late_flow_late_layers"),
            ]
        ),
        axis=1,
    )
    flow_sharpening = np.mean(
        np.column_stack(
            [
                -_column(values, "entropy_flow_slope_late_layers"),
                _column(values, "margin_flow_slope_late_layers"),
            ]
        ),
        axis=1,
    )
    return np.column_stack(
        [sharpness, synchronization, rank, switching, flow_sharpening]
    ).astype(np.float32)


def episode_uniform_rows(
    starts: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
    samples_per_episode: int = 12,
) -> np.ndarray:
    """Sample positions uniformly so every episode has comparable influence."""

    rows: list[np.ndarray] = []
    for episode in np.asarray(episodes, dtype=np.int64):
        length = int(lengths[episode])
        if length <= 0:
            continue
        count = min(int(samples_per_episode), length)
        offsets = np.unique(np.rint(np.linspace(0, length - 1, count)).astype(np.int64))
        rows.append(int(starts[episode]) + offsets)
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)


@dataclass
class TernaryChordTokenizer:
    """Healthy-quantile tokenizer for simultaneous, non-exclusive phenotypes."""

    low_quantile: float = 0.20
    high_quantile: float = 0.80
    low_release_quantile: float = 0.35
    high_release_quantile: float = 0.65
    samples_per_episode: int = 12
    low_: np.ndarray | None = field(default=None, init=False, repr=False)
    high_: np.ndarray | None = field(default=None, init=False, repr=False)
    low_release_: np.ndarray | None = field(default=None, init=False, repr=False)
    high_release_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        tracks: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episodes: np.ndarray,
    ) -> "TernaryChordTokenizer":
        rows = episode_uniform_rows(
            starts, lengths, episodes, samples_per_episode=self.samples_per_episode
        )
        if rows.size == 0:
            raise ValueError("cannot fit tokenizer without healthy training rows")
        sample = np.asarray(tracks, dtype=np.float64)[rows]
        self.low_ = np.quantile(sample, self.low_quantile, axis=0)
        self.high_ = np.quantile(sample, self.high_quantile, axis=0)
        self.low_release_ = np.quantile(sample, self.low_release_quantile, axis=0)
        self.high_release_ = np.quantile(sample, self.high_release_quantile, axis=0)
        return self

    def transform(
        self,
        tracks: np.ndarray,
        starts: np.ndarray | None = None,
        lengths: np.ndarray | None = None,
        *,
        hysteresis: bool = False,
    ) -> np.ndarray:
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("tokenizer has not been fitted")
        values = np.asarray(tracks, dtype=np.float64)
        chords = np.ones(values.shape, dtype=np.int8)
        chords[values <= self.low_[None, :]] = 0
        chords[values >= self.high_[None, :]] = 2
        if not hysteresis:
            return chords
        if starts is None or lengths is None:
            raise ValueError("starts and lengths are required for hysteresis")
        if self.low_release_ is None or self.high_release_ is None:
            raise RuntimeError("tokenizer release thresholds are unavailable")

        result = chords.copy()
        for start, length in zip(starts, lengths, strict=True):
            start = int(start)
            stop = start + int(length)
            if stop <= start:
                continue
            state = chords[start].copy()
            result[start] = state
            for row in range(start + 1, stop):
                for track in range(values.shape[1]):
                    value = values[row, track]
                    if state[track] == 0 and value <= self.low_release_[track]:
                        result[row, track] = 0
                    elif state[track] == 2 and value >= self.high_release_[track]:
                        result[row, track] = 2
                    else:
                        result[row, track] = chords[row, track]
                state = result[row].copy()
        return result


def chord_codes(chords: np.ndarray) -> np.ndarray:
    values = np.asarray(chords, dtype=np.int64)
    powers = 3 ** np.arange(values.shape[1], dtype=np.int64)
    return values @ powers


def chord_label(chord: np.ndarray) -> str:
    pieces = [TRACK_SYMBOLS[k][int(value)] for k, value in enumerate(chord)]
    active = [piece for piece in pieces if piece]
    return "+".join(active) if active else "Neutral"


@dataclass
class FactorialCategoricalHMM:
    """Progress-initialized categorical HMM with a full-prefix filter."""

    phase_states: int = 12
    alpha: float = 0.25
    transition_alpha: float = 0.25
    samples_per_episode: int = 12
    initial_: np.ndarray | None = field(default=None, init=False, repr=False)
    transition_: np.ndarray | None = field(default=None, init=False, repr=False)
    emission_: np.ndarray | None = field(default=None, init=False, repr=False)
    global_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        chords: np.ndarray,
        progress: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episodes: np.ndarray,
    ) -> "FactorialCategoricalHMM":
        symbols = np.asarray(chords, dtype=np.int64)
        phase = np.minimum(
            (np.asarray(progress, dtype=np.float64) * self.phase_states).astype(np.int64),
            self.phase_states - 1,
        )
        tracks = symbols.shape[1]
        initial = np.full(self.phase_states, self.transition_alpha, dtype=np.float64)
        transition = np.full(
            (self.phase_states, self.phase_states), self.transition_alpha, dtype=np.float64
        )
        emission = np.full(
            (self.phase_states, tracks, 3), self.alpha, dtype=np.float64
        )
        global_counts = np.full((tracks, 3), self.alpha, dtype=np.float64)

        rows = episode_uniform_rows(
            starts, lengths, episodes, samples_per_episode=self.samples_per_episode
        )
        for row in rows:
            state = phase[row]
            for track in range(tracks):
                emission[state, track, symbols[row, track]] += 1.0
                global_counts[track, symbols[row, track]] += 1.0

        for episode in np.asarray(episodes, dtype=np.int64):
            start = int(starts[episode])
            length = int(lengths[episode])
            if length <= 0:
                continue
            initial[phase[start]] += 1.0
            weight = 1.0 / max(length - 1, 1)
            for row in range(start + 1, start + length):
                transition[phase[row - 1], phase[row]] += weight

        self.initial_ = initial / initial.sum()
        self.transition_ = transition / transition.sum(axis=1, keepdims=True)
        self.emission_ = emission / emission.sum(axis=2, keepdims=True)
        self.global_ = global_counts / global_counts.sum(axis=1, keepdims=True)
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (self.initial_, self.transition_, self.emission_, self.global_)
        ):
            raise RuntimeError("categorical HMM has not been fitted")

    def score(
        self,
        chords: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        update_with_observations: bool = True,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return causal NLL/track, filtered phase, and predictive entropy."""

        self._check_fitted()
        symbols = np.asarray(chords, dtype=np.int64)
        if device.startswith("cuda"):
            try:
                return self._score_torch(
                    symbols,
                    starts,
                    lengths,
                    update_with_observations=update_with_observations,
                    device=device,
                )
            except (ImportError, RuntimeError):
                pass
        return self._score_numpy(
            symbols,
            starts,
            lengths,
            update_with_observations=update_with_observations,
        )

    def _emission_log_likelihood(self, symbols: np.ndarray) -> np.ndarray:
        assert self.emission_ is not None
        output = np.zeros((len(symbols), self.phase_states), dtype=np.float64)
        for track in range(symbols.shape[1]):
            output += np.log(self.emission_[:, track, symbols[:, track]].T + 1e-300)
        return output

    def _score_numpy(
        self,
        symbols: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        update_with_observations: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.initial_ is not None and self.transition_ is not None
        emission = self._emission_log_likelihood(symbols)
        nll = np.empty(len(symbols), dtype=np.float64)
        phase = np.empty(len(symbols), dtype=np.float32)
        entropy = np.empty(len(symbols), dtype=np.float64)
        log_initial = np.log(self.initial_ + 1e-300)
        log_transition = np.log(self.transition_ + 1e-300)
        starts_array = np.asarray(starts, dtype=np.int64)
        lengths_array = np.asarray(lengths, dtype=np.int64)
        beliefs = np.repeat(log_initial[None, :], len(starts_array), axis=0)
        for offset in range(int(lengths_array.max(initial=0))):
            active = np.flatnonzero(lengths_array > offset)
            rows = starts_array[active] + offset
            predictive = beliefs[active]
            predictive -= logsumexp(predictive, axis=1, keepdims=True)
            nll[rows] = -logsumexp(predictive + emission[rows], axis=1) / symbols.shape[1]
            probabilities = np.exp(predictive)
            phase[rows] = np.argmax(probabilities, axis=1) / max(self.phase_states - 1, 1)
            entropy[rows] = -np.sum(probabilities * predictive, axis=1)
            posterior = (
                predictive + emission[rows] if update_with_observations else predictive
            )
            posterior -= logsumexp(posterior, axis=1, keepdims=True)
            beliefs[active] = logsumexp(
                posterior[:, :, None] + log_transition[None, :, :], axis=1
            )
        return nll.astype(np.float32), phase, entropy.astype(np.float32)

    def _score_torch(
        self,
        symbols: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        update_with_observations: bool,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        assert self.initial_ is not None
        assert self.transition_ is not None
        assert self.emission_ is not None
        symbol_tensor = torch.as_tensor(symbols, dtype=torch.long, device=device)
        log_emission_table = torch.log(
            torch.as_tensor(self.emission_, dtype=torch.float64, device=device) + 1e-300
        )
        emission = torch.zeros(
            (len(symbols), self.phase_states), dtype=torch.float64, device=device
        )
        for track in range(symbols.shape[1]):
            emission += log_emission_table[:, track, symbol_tensor[:, track]].T
        log_initial = torch.log(
            torch.as_tensor(self.initial_, dtype=torch.float64, device=device) + 1e-300
        )
        log_transition = torch.log(
            torch.as_tensor(self.transition_, dtype=torch.float64, device=device) + 1e-300
        )
        nll = torch.empty(len(symbols), dtype=torch.float64, device=device)
        phase = torch.empty(len(symbols), dtype=torch.float32, device=device)
        entropy = torch.empty(len(symbols), dtype=torch.float64, device=device)
        starts_tensor = torch.as_tensor(starts, dtype=torch.long, device=device)
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
        beliefs = log_initial[None, :].repeat(len(starts_tensor), 1)
        maximum_length = int(lengths_tensor.max().item()) if len(lengths_tensor) else 0
        for offset in range(maximum_length):
            active = torch.nonzero(lengths_tensor > offset, as_tuple=False).flatten()
            rows = starts_tensor[active] + offset
            predictive = beliefs[active]
            predictive -= torch.logsumexp(predictive, dim=1, keepdim=True)
            nll[rows] = -torch.logsumexp(
                predictive + emission[rows], dim=1
            ) / symbols.shape[1]
            probabilities = torch.exp(predictive)
            phase[rows] = torch.argmax(probabilities, dim=1).to(torch.float32) / max(
                self.phase_states - 1, 1
            )
            entropy[rows] = -torch.sum(probabilities * predictive, dim=1)
            posterior = (
                predictive + emission[rows] if update_with_observations else predictive
            )
            posterior -= torch.logsumexp(posterior, dim=1, keepdim=True)
            beliefs[active] = torch.logsumexp(
                posterior[:, :, None] + log_transition[None, :, :], dim=1
            )
        return (
            nll.cpu().numpy().astype(np.float32),
            phase.cpu().numpy(),
            entropy.cpu().numpy().astype(np.float32),
        )

    def global_nll(self, chords: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.global_ is not None
        symbols = np.asarray(chords, dtype=np.int64)
        nll = np.zeros(len(symbols), dtype=np.float64)
        for track in range(symbols.shape[1]):
            nll -= np.log(self.global_[track, symbols[:, track]] + 1e-300)
        return (nll / symbols.shape[1]).astype(np.float32)

    def score_last_observation(
        self,
        chords: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        *,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score q from the latent clock and q-1 only, discarding older observations."""

        self._check_fitted()
        symbols = np.asarray(chords, dtype=np.int64)
        if device.startswith("cuda"):
            try:
                return self._score_last_torch(symbols, starts, lengths, device)
            except (ImportError, RuntimeError):
                pass
        return self._score_last_numpy(symbols, starts, lengths)

    def _score_last_numpy(
        self, symbols: np.ndarray, starts: np.ndarray, lengths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.initial_ is not None and self.transition_ is not None
        emission = self._emission_log_likelihood(symbols)
        starts_array = np.asarray(starts, dtype=np.int64)
        lengths_array = np.asarray(lengths, dtype=np.int64)
        maximum_length = int(lengths_array.max(initial=0))
        log_transition = np.log(self.transition_ + 1e-300)
        clock = np.log(self.initial_ + 1e-300)
        clock -= logsumexp(clock)
        clock_priors = [clock]
        for _ in range(1, maximum_length):
            clock = logsumexp(clock[:, None] + log_transition, axis=0)
            clock -= logsumexp(clock)
            clock_priors.append(clock)

        nll = np.empty(len(symbols), dtype=np.float64)
        phase = np.empty(len(symbols), dtype=np.float32)
        entropy = np.empty(len(symbols), dtype=np.float64)
        for offset in range(maximum_length):
            active = np.flatnonzero(lengths_array > offset)
            rows = starts_array[active] + offset
            if offset == 0:
                predictive = np.repeat(clock_priors[0][None, :], len(active), axis=0)
            else:
                previous_rows = rows - 1
                posterior = clock_priors[offset - 1][None, :] + emission[previous_rows]
                posterior -= logsumexp(posterior, axis=1, keepdims=True)
                predictive = logsumexp(
                    posterior[:, :, None] + log_transition[None, :, :], axis=1
                )
                predictive -= logsumexp(predictive, axis=1, keepdims=True)
            nll[rows] = -logsumexp(predictive + emission[rows], axis=1) / symbols.shape[1]
            probabilities = np.exp(predictive)
            phase[rows] = np.argmax(probabilities, axis=1) / max(self.phase_states - 1, 1)
            entropy[rows] = -np.sum(probabilities * predictive, axis=1)
        return nll.astype(np.float32), phase, entropy.astype(np.float32)

    def _score_last_torch(
        self,
        symbols: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        assert self.initial_ is not None
        assert self.transition_ is not None
        assert self.emission_ is not None
        symbol_tensor = torch.as_tensor(symbols, dtype=torch.long, device=device)
        log_emission_table = torch.log(
            torch.as_tensor(self.emission_, dtype=torch.float64, device=device) + 1e-300
        )
        emission = torch.zeros(
            (len(symbols), self.phase_states), dtype=torch.float64, device=device
        )
        for track in range(symbols.shape[1]):
            emission += log_emission_table[:, track, symbol_tensor[:, track]].T
        starts_tensor = torch.as_tensor(starts, dtype=torch.long, device=device)
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
        maximum_length = int(lengths_tensor.max().item()) if len(lengths_tensor) else 0
        log_transition = torch.log(
            torch.as_tensor(self.transition_, dtype=torch.float64, device=device) + 1e-300
        )
        clock = torch.log(
            torch.as_tensor(self.initial_, dtype=torch.float64, device=device) + 1e-300
        )
        clock -= torch.logsumexp(clock, dim=0)
        clock_priors = [clock]
        for _ in range(1, maximum_length):
            clock = torch.logsumexp(clock[:, None] + log_transition, dim=0)
            clock -= torch.logsumexp(clock, dim=0)
            clock_priors.append(clock)

        nll = torch.empty(len(symbols), dtype=torch.float64, device=device)
        phase = torch.empty(len(symbols), dtype=torch.float32, device=device)
        entropy = torch.empty(len(symbols), dtype=torch.float64, device=device)
        for offset in range(maximum_length):
            active = torch.nonzero(lengths_tensor > offset, as_tuple=False).flatten()
            rows = starts_tensor[active] + offset
            if offset == 0:
                predictive = clock_priors[0][None, :].repeat(len(active), 1)
            else:
                posterior = clock_priors[offset - 1][None, :] + emission[rows - 1]
                posterior -= torch.logsumexp(posterior, dim=1, keepdim=True)
                predictive = torch.logsumexp(
                    posterior[:, :, None] + log_transition[None, :, :], dim=1
                )
                predictive -= torch.logsumexp(predictive, dim=1, keepdim=True)
            nll[rows] = -torch.logsumexp(
                predictive + emission[rows], dim=1
            ) / symbols.shape[1]
            probabilities = torch.exp(predictive)
            phase[rows] = torch.argmax(probabilities, dim=1).to(torch.float32) / max(
                self.phase_states - 1, 1
            )
            entropy[rows] = -torch.sum(probabilities * predictive, dim=1)
        return (
            nll.cpu().numpy().astype(np.float32),
            phase.cpu().numpy(),
            entropy.cpu().numpy().astype(np.float32),
        )


@dataclass
class FactorialContextGrammar:
    """Backoff categorical grammar over ordered or bagged chord contexts."""

    max_order: int = 4
    alpha: float = 0.05
    min_support: float = 2.0
    ordered: bool = True
    counts_: list[dict[tuple[int, ...], np.ndarray]] = field(
        default_factory=list, init=False, repr=False
    )

    def _context(self, history: list[int], order: int) -> tuple[int, ...]:
        context = tuple(history[-order:]) if order else ()
        return context if self.ordered else tuple(sorted(context))

    def fit(
        self,
        chords: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episodes: np.ndarray,
    ) -> "FactorialContextGrammar":
        symbols = np.asarray(chords, dtype=np.int64)
        codes = chord_codes(symbols)
        tracks = symbols.shape[1]
        tables: list[defaultdict[tuple[int, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros((tracks, 3), dtype=np.float64))
            for _ in range(self.max_order + 1)
        ]
        for episode in np.asarray(episodes, dtype=np.int64):
            start = int(starts[episode])
            length = int(lengths[episode])
            if length <= 0:
                continue
            weight = 1.0 / length
            history: list[int] = []
            for row in range(start, start + length):
                for order in range(min(self.max_order, len(history)) + 1):
                    counts = tables[order][self._context(history, order)]
                    for track in range(tracks):
                        counts[track, symbols[row, track]] += weight
                history.append(int(codes[row]))
        self.counts_ = [dict(table) for table in tables]
        return self

    def distribution(
        self, history: list[int], max_order: int | None = None
    ) -> tuple[np.ndarray, int]:
        if not self.counts_:
            raise RuntimeError("context grammar has not been fitted")
        selected = self.counts_[0][()]
        selected_order = 0
        effective_order = self.max_order if max_order is None else min(max_order, self.max_order)
        for order in range(1, min(effective_order, len(history)) + 1):
            counts = self.counts_[order].get(self._context(history, order))
            if counts is not None and float(counts[0].sum()) >= self.min_support:
                selected = counts
                selected_order = order
        probability = (selected + self.alpha) / (
            selected.sum(axis=1, keepdims=True) + 3.0 * self.alpha
        )
        return probability, selected_order

    def score(
        self,
        chords: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episodes: np.ndarray | None = None,
        max_order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        symbols = np.asarray(chords, dtype=np.int64)
        codes = chord_codes(symbols)
        nll = np.full(len(symbols), np.nan, dtype=np.float32)
        depth = np.full(len(symbols), -1, dtype=np.int8)
        selected_episodes = (
            np.arange(len(starts), dtype=np.int64)
            if episodes is None
            else np.asarray(episodes, dtype=np.int64)
        )
        for episode in selected_episodes:
            start = int(starts[episode])
            length = int(lengths[episode])
            start = int(start)
            history: list[int] = []
            for row in range(start, start + int(length)):
                probability, order = self.distribution(history, max_order=max_order)
                log_probability = 0.0
                for track in range(symbols.shape[1]):
                    log_probability += np.log(
                        probability[track, symbols[row, track]] + 1e-300
                    )
                nll[row] = -log_probability / symbols.shape[1]
                depth[row] = order
                history.append(int(codes[row]))
        return nll, depth


def causal_track_run_lengths(
    chords: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    symbols = np.asarray(chords, dtype=np.int64)
    output = np.ones(symbols.shape, dtype=np.int16)
    for start, length in zip(starts, lengths, strict=True):
        start = int(start)
        stop = start + int(length)
        for row in range(start + 1, stop):
            same = symbols[row] == symbols[row - 1]
            output[row] = np.where(same, output[row - 1] + 1, 1)
    return output


@dataclass
class FactorialDurationModel:
    """Episode-balanced healthy reference for ongoing per-track dwell."""

    samples_per_episode: int = 12
    reference_: list[list[np.ndarray]] = field(default_factory=list, init=False, repr=False)

    def fit(
        self,
        chords: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
        episodes: np.ndarray,
    ) -> "FactorialDurationModel":
        symbols = np.asarray(chords, dtype=np.int64)
        run = causal_track_run_lengths(symbols, starts, lengths)
        rows = episode_uniform_rows(
            starts, lengths, episodes, samples_per_episode=self.samples_per_episode
        )
        self.reference_ = []
        for track in range(symbols.shape[1]):
            per_state: list[np.ndarray] = []
            for state in range(3):
                values = run[rows, track][symbols[rows, track] == state]
                per_state.append(np.sort(values.astype(np.float64)))
            self.reference_.append(per_state)
        return self

    def score(
        self, chords: np.ndarray, starts: np.ndarray, lengths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.reference_:
            raise RuntimeError("duration model has not been fitted")
        symbols = np.asarray(chords, dtype=np.int64)
        run = causal_track_run_lengths(symbols, starts, lengths)
        track_risk = np.empty(symbols.shape, dtype=np.float32)
        for track in range(symbols.shape[1]):
            for state in range(3):
                mask = symbols[:, track] == state
                reference = self.reference_[track][state]
                if reference.size == 0:
                    track_risk[mask, track] = 0.5
                    continue
                rank = np.searchsorted(reference, run[mask, track], side="right")
                track_risk[mask, track] = (rank + 0.5) / (reference.size + 1.0)
        sorted_risk = np.sort(track_risk, axis=1)
        combined = sorted_risk[:, -2:].mean(axis=1)
        return combined.astype(np.float32), track_risk


def categorical_recurrence(
    chords: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    max_lag: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return freeze similarity, periodic similarity, and the winning lag."""

    symbols = np.asarray(chords, dtype=np.int64)
    freeze = np.zeros(len(symbols), dtype=np.float32)
    periodic = np.zeros(len(symbols), dtype=np.float32)
    winning_lag = np.zeros(len(symbols), dtype=np.int8)
    for start, length in zip(starts, lengths, strict=True):
        start = int(start)
        stop = start + int(length)
        if stop - start > 1:
            freeze[start + 1 : stop] = np.mean(
                symbols[start + 1 : stop] == symbols[start : stop - 1], axis=1
            )
        for lag in range(2, min(max_lag, stop - start - 1) + 1):
            target = slice(start + lag, stop)
            similarity = np.mean(
                symbols[start + lag : stop] == symbols[start : stop - lag], axis=1
            ).astype(np.float32)
            improved = similarity > periodic[target]
            periodic[target] = np.where(improved, similarity, periodic[target])
            winning_lag[target] = np.where(improved, lag, winning_lag[target])
    return freeze, periodic, winning_lag
