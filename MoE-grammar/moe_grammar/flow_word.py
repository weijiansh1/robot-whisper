"""Flow-step chords and query words that retain all ten denoising steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .discrete_phenotype import TRACK_NAMES, chord_codes, chord_label


FLOW_TRACK_NAMES = TRACK_NAMES


@dataclass
class FlowTrackProjector:
    """Robustly normalize primitive metrics before forming semantic tracks."""

    clip: float = 10.0
    center_: np.ndarray | None = field(default=None, init=False, repr=False)
    scale_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(
        self, primitives: np.ndarray, query_rows: np.ndarray
    ) -> "FlowTrackProjector":
        values = np.asarray(primitives, dtype=np.float32)[query_rows].reshape(-1, 10)
        self.center_ = np.median(values, axis=0)
        low, high = np.quantile(values, [0.25, 0.75], axis=0)
        self.scale_ = np.maximum(high - low, 1e-6)
        return self

    def transform(self, primitives: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("flow track projector has not been fitted")
        values = np.asarray(primitives, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (10, 10):
            raise ValueError(f"expected [query,10,10] primitives, got {values.shape}")
        scaled = np.clip(
            (values - self.center_[None, None, :]) / self.scale_[None, None, :],
            -self.clip,
            self.clip,
        )
        sharpness = np.mean(
            np.stack([-scaled[:, :, 0], scaled[:, :, 1], scaled[:, :, 2], scaled[:, :, 3]]),
            axis=0,
        )
        synchronization = np.mean(scaled[:, :, 4:6], axis=2)
        rank = scaled[:, :, 6]
        switching = np.mean(scaled[:, :, 7:10], axis=2)
        flow_sharpening = np.zeros_like(sharpness)
        flow_sharpening[:, 1:] = np.diff(sharpness, axis=1)
        return np.stack(
            [sharpness, synchronization, rank, switching, flow_sharpening], axis=2
        ).astype(np.float32)


@dataclass
class FlowChordTokenizer:
    low_quantile: float = 0.20
    high_quantile: float = 0.80
    low_: np.ndarray | None = field(default=None, init=False, repr=False)
    high_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, tracks: np.ndarray, query_rows: np.ndarray) -> "FlowChordTokenizer":
        selected = np.asarray(tracks, dtype=np.float32)[query_rows]
        self.low_ = np.empty(selected.shape[2], dtype=np.float32)
        self.high_ = np.empty(selected.shape[2], dtype=np.float32)
        for track in range(selected.shape[2]):
            values = selected[:, 1:, track] if track == 4 else selected[:, :, track]
            self.low_[track], self.high_[track] = np.quantile(
                values, [self.low_quantile, self.high_quantile]
            )
        return self

    def transform(self, tracks: np.ndarray) -> np.ndarray:
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("flow chord tokenizer has not been fitted")
        values = np.asarray(tracks, dtype=np.float32)
        output = np.ones(values.shape, dtype=np.int8)
        output[values <= self.low_[None, None, :]] = 0
        output[values >= self.high_[None, None, :]] = 2
        # Switching and flow sharpening compare adjacent flow steps. They are
        # undefined at f0, where their extracted primitive values are structural zeros.
        output[:, 0, 3] = 1
        output[:, 0, 4] = 1
        return output


def flow_query_words(flow_chords: np.ndarray) -> np.ndarray:
    chords = np.asarray(flow_chords, dtype=np.int8)
    if chords.ndim != 3 or chords.shape[1:] != (10, len(FLOW_TRACK_NAMES)):
        raise ValueError(f"expected [query,10,{len(FLOW_TRACK_NAMES)}], got {chords.shape}")
    return chords.reshape(len(chords), -1)


@dataclass
class PositionFactorialContextGrammar:
    """Predict each flow chord from its position and preceding flow chords."""

    max_order: int = 4
    alpha: float = 0.25
    min_support: float = 32.0
    ordered: bool = True
    keys_: list[list[np.ndarray]] = field(
        default_factory=list, init=False, repr=False
    )
    counts_: list[list[np.ndarray]] = field(default_factory=list, init=False, repr=False)

    def _encode_context(self, context: np.ndarray) -> np.ndarray:
        values = np.asarray(context, dtype=np.int64)
        if not self.ordered:
            values = np.sort(values, axis=-1)
        if values.shape[-1] == 0:
            return np.zeros(values.shape[:-1], dtype=np.int64)
        powers = 243 ** np.arange(values.shape[-1], dtype=np.int64)
        return values @ powers

    def fit(
        self, flow_chords: np.ndarray, query_rows: np.ndarray
    ) -> "PositionFactorialContextGrammar":
        chords = np.asarray(flow_chords, dtype=np.int8)
        codes = chord_codes(chords.reshape(-1, chords.shape[2])).reshape(chords.shape[:2])
        query_rows = np.asarray(query_rows, dtype=np.int64)
        selected_chords = chords[query_rows]
        selected_codes = codes[query_rows]
        tracks = chords.shape[2]
        self.keys_ = [
            [np.empty(0, dtype=np.int64) for _ in range(chords.shape[1])]
            for _ in range(self.max_order + 1)
        ]
        self.counts_ = [
            [np.empty((0, tracks, 3), dtype=np.float64) for _ in range(chords.shape[1])]
            for _ in range(self.max_order + 1)
        ]
        track_offsets = np.arange(tracks, dtype=np.int64) * 3
        for order in range(self.max_order + 1):
            for position in range(order, chords.shape[1]):
                context = selected_codes[:, position - order : position]
                encoded = self._encode_context(context)
                keys, inverse = np.unique(encoded, return_inverse=True)
                targets = selected_chords[:, position]
                flat_indexes = (
                    inverse[:, None] * (tracks * 3)
                    + track_offsets[None, :]
                    + targets
                )
                counts = np.bincount(
                    flat_indexes.ravel(), minlength=len(keys) * tracks * 3
                ).reshape(len(keys), tracks, 3)
                self.keys_[order][position] = keys
                self.counts_[order][position] = counts.astype(np.float64)
        return self

    def distribution(
        self,
        position: int,
        history: list[int],
        max_order: int | None = None,
    ) -> tuple[np.ndarray, int]:
        if not self.counts_:
            raise RuntimeError("flow context grammar has not been fitted")
        selected = self.counts_[0][position][0]
        selected_order = 0
        effective_order = self.max_order if max_order is None else min(max_order, self.max_order)
        for order in range(1, min(effective_order, len(history)) + 1):
            encoded = int(
                self._encode_context(
                    np.asarray(history[-order:], dtype=np.int64)[None, :]
                )[0]
            )
            keys = self.keys_[order][position]
            index = int(np.searchsorted(keys, encoded))
            counts = (
                self.counts_[order][position][index]
                if index < len(keys) and keys[index] == encoded
                else None
            )
            if counts is not None and float(counts[0].sum()) >= self.min_support:
                selected = counts
                selected_order = order
        probability = (selected + self.alpha) / (
            selected.sum(axis=1, keepdims=True) + 3.0 * self.alpha
        )
        return probability, selected_order

    def score_words(
        self, flow_chords: np.ndarray, max_order: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        nll, depth = self.score_order_curve(flow_chords)
        effective_order = self.max_order if max_order is None else min(max_order, self.max_order)
        return nll[:, effective_order], depth[:, effective_order]

    def score_order_curve(self, flow_chords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score orders 0..max_order together without repeating context lookups."""

        chords = np.asarray(flow_chords, dtype=np.int8)
        codes = chord_codes(chords.reshape(-1, chords.shape[2])).reshape(chords.shape[:2])
        nll = np.zeros((len(chords), self.max_order + 1), dtype=np.float64)
        mean_depth = np.zeros_like(nll)
        track_index = np.arange(chords.shape[2])
        for position in range(chords.shape[1]):
            root = self.counts_[0][position][0]
            root_probability = (root + self.alpha) / (
                root.sum(axis=1, keepdims=True) + 3.0 * self.alpha
            )
            probability = np.repeat(root_probability[None, :, :], len(chords), axis=0)
            used_order = np.zeros(len(chords), dtype=np.int8)
            for order in range(self.max_order + 1):
                if 0 < order <= position:
                    encoded = self._encode_context(codes[:, position - order : position])
                    keys = self.keys_[order][position]
                    indexes = np.searchsorted(keys, encoded)
                    safe_indexes = np.minimum(indexes, len(keys) - 1)
                    found = (indexes < len(keys)) & (keys[safe_indexes] == encoded)
                    candidate_counts = self.counts_[order][position][safe_indexes]
                    supported = found & (
                        candidate_counts[:, 0, :].sum(axis=1) >= self.min_support
                    )
                    candidate_probability = (candidate_counts + self.alpha) / (
                        candidate_counts.sum(axis=2, keepdims=True) + 3.0 * self.alpha
                    )
                    probability[supported] = candidate_probability[supported]
                    used_order[supported] = order
                target_probability = probability[
                    np.arange(len(chords))[:, None],
                    track_index[None, :],
                    chords[:, position],
                ]
                nll[:, order] -= np.log(target_probability + 1e-300).sum(axis=1)
                mean_depth[:, order] += used_order
        denominator = chords.shape[1] * chords.shape[2]
        return (
            (nll / denominator).astype(np.float32),
            (mean_depth / chords.shape[1]).astype(np.float32),
        )


def flow_motif_indicators(flow_chords: np.ndarray) -> dict[str, np.ndarray]:
    chords = np.asarray(flow_chords, dtype=np.int8)
    flat = chords[:, :, 0] == 0
    sync = chords[:, :, 1] == 2
    any_motif = np.zeros(len(chords), dtype=np.float32)
    for start in range(7):
        any_motif = np.maximum(
            any_motif,
            (
                flat[:, start]
                & flat[:, start + 1]
                & sync[:, start + 2]
                & sync[:, start + 3]
            ).astype(np.float32),
        )
    return {
        "flow_fl_fl_sy_sy_any": any_motif,
        "flow_fl_fl_sy_sy_terminal": (
            flat[:, 6] & flat[:, 7] & sync[:, 8] & sync[:, 9]
        ).astype(np.float32),
        "flow_fl_parallel_sy_terminal": (
            (flat[:, 6:10].sum(axis=1) >= 3) & sync[:, 8] & sync[:, 9]
        ).astype(np.float32),
        "flow_terminal_fl_sy": (flat[:, 9] & sync[:, 9]).astype(np.float32),
    }


def format_flow_word(word: np.ndarray) -> str:
    labels = [chord_label(chord) for chord in np.asarray(word)]
    pieces: list[str] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            length = index - start
            pieces.append(labels[start] if length == 1 else f"{labels[start]}^{length}")
            start = index
    return " -> ".join(pieces)
