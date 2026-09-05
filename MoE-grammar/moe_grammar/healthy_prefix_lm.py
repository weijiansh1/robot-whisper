from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class HealthyPrefixTransformer(nn.Module):
    """Causal language model over the complete routing prefix.

    Input position 0 is BOS and predicts word 0. Input position q > 0 contains
    word/descriptor q-1 and predicts word q. The output vocabulary includes END,
    while the input vocabulary uses the same numeric ID for BOS through a separate
    embedding table.
    """

    def __init__(
        self,
        descriptor_dim: int,
        n_words: int,
        task_count: int,
        max_positions: int,
        model_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.descriptor_dim = descriptor_dim
        self.n_words = n_words
        self.end_token = n_words
        self.bos_token = n_words
        self.max_positions = max_positions
        self.model_dim = model_dim
        self.heads = heads
        self.layers = layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.word_embedding = nn.Embedding(n_words + 1, model_dim)
        self.descriptor_projection = nn.Linear(descriptor_dim, model_dim, bias=False)
        self.task_embedding = nn.Embedding(task_count, model_dim)
        self.position_embedding = nn.Embedding(max_positions, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_dim, n_words + 1)

    def forward(
        self,
        input_words: torch.Tensor,
        input_descriptors: torch.Tensor,
        task: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_words.ndim != 2 or input_descriptors.ndim != 3:
            raise ValueError("expected [batch,time] words and [batch,time,dim] descriptors")
        if input_words.shape != input_descriptors.shape[:2]:
            raise ValueError("word and descriptor axes differ")
        if input_words.shape[1] > self.max_positions:
            raise ValueError("sequence is longer than max_positions")

        positions = torch.arange(input_words.shape[1], device=input_words.device)
        values = (
            self.word_embedding(input_words)
            + self.descriptor_projection(input_descriptors.clamp(-12.0, 12.0))
            + self.task_embedding(task)[:, None, :]
            + self.position_embedding(positions)[None, :, :]
        )
        values = self.input_norm(values)
        causal_mask = torch.triu(
            torch.ones(
                input_words.shape[1],
                input_words.shape[1],
                dtype=torch.bool,
                device=input_words.device,
            ),
            diagonal=1,
        )
        padding_mask = None if valid is None else ~valid
        hidden = self.transformer(
            values,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output(hidden), hidden


@dataclass(frozen=True)
class PrefixBatch:
    input_words: np.ndarray
    input_descriptors: np.ndarray
    targets: np.ndarray
    target_descriptors: np.ndarray
    valid: np.ndarray
    word_target: np.ndarray


def build_shifted_batch(
    projected: np.ndarray,
    words: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    episode_indexes: np.ndarray,
    n_words: int,
    *,
    include_end: bool = True,
    maximum_words: int | None = None,
) -> PrefixBatch:
    """Build BOS-shifted sequences without exposing a target to its predictor."""

    episode_indexes = np.asarray(episode_indexes, dtype=np.int64)
    if not len(episode_indexes):
        raise ValueError("cannot build an empty batch")
    counts = np.asarray(lengths[episode_indexes], dtype=np.int64)
    if maximum_words is not None:
        counts = np.minimum(counts, maximum_words)
    target_counts = counts + int(include_end)
    width = int(target_counts.max())
    descriptor_dim = projected.shape[1]

    input_words = np.full((len(episode_indexes), width), n_words, dtype=np.int64)
    input_descriptors = np.zeros(
        (len(episode_indexes), width, descriptor_dim), dtype=np.float32
    )
    targets = np.full((len(episode_indexes), width), -100, dtype=np.int64)
    target_descriptors = np.zeros_like(input_descriptors)
    valid = np.zeros((len(episode_indexes), width), dtype=np.bool_)
    word_target = np.zeros((len(episode_indexes), width), dtype=np.bool_)

    for row, (episode_index, count) in enumerate(zip(episode_indexes, counts)):
        start = int(starts[episode_index])
        count = int(count)
        episode_words = np.asarray(words[start : start + count], dtype=np.int64)
        episode_descriptors = np.asarray(
            projected[start : start + count], dtype=np.float32
        )
        if count:
            targets[row, :count] = episode_words
            target_descriptors[row, :count] = episode_descriptors
            word_target[row, :count] = True
            if count > 1:
                input_words[row, 1:count] = episode_words[:-1]
                input_descriptors[row, 1:count] = episode_descriptors[:-1]
        if include_end:
            targets[row, count] = n_words
            if count:
                input_words[row, count] = episode_words[-1]
                input_descriptors[row, count] = episode_descriptors[-1]
        valid[row, : target_counts[row]] = True
    return PrefixBatch(
        input_words=input_words,
        input_descriptors=input_descriptors,
        targets=targets,
        target_descriptors=target_descriptors,
        valid=valid,
        word_target=word_target,
    )


def component_log_likelihood_torch(
    values: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
) -> torch.Tensor:
    dimensions = values.shape[-1]
    constant = dimensions * np.log(2.0 * np.pi) + torch.log(covariances).sum(dim=1)
    quadratic = (
        torch.square(values[:, :, None, :] - means[None, None, :, :])
        / covariances[None, None, :, :]
    ).sum(dim=3)
    return -0.5 * (constant[None, None, :] + quadratic)


def soft_word_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_descriptors: torch.Tensor,
    valid: torch.Tensor,
    word_target: torch.Tensor,
    component_means: torch.Tensor,
    component_covariances: torch.Tensor,
    component_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against soft GMM word membership plus an exact END target."""

    log_probability = torch.log_softmax(logits, dim=2)
    component = component_log_likelihood_torch(
        target_descriptors, component_means, component_covariances
    )
    responsibility = torch.softmax(
        component + torch.log(component_weights)[None, None, :], dim=2
    )
    word_loss = -(responsibility * log_probability[:, :, :-1]).sum(dim=2)
    losses = torch.where(word_target, word_loss, -log_probability[:, :, -1])
    return losses[valid].mean()


def reverse_remote_history(batch: PrefixBatch, target_position: int) -> PrefixBatch:
    """Reverse history older than the last two words for one prediction position.

    The target and the two most recent history items stay fixed, so any score
    change specifically requires use of remote order rather than local context.
    """

    if target_position < 0 or target_position >= batch.input_words.shape[1]:
        raise ValueError("target_position outside batch")
    words = batch.input_words.copy()
    descriptors = batch.input_descriptors.copy()
    remote_stop = max(1, target_position - 1)
    if remote_stop > 1:
        words[:, 1:remote_stop] = words[:, 1:remote_stop][:, ::-1]
        descriptors[:, 1:remote_stop] = descriptors[:, 1:remote_stop][:, ::-1]
    return PrefixBatch(
        input_words=words,
        input_descriptors=descriptors,
        targets=batch.targets,
        target_descriptors=batch.target_descriptors,
        valid=batch.valid,
        word_target=batch.word_target,
    )
