import numpy as np
import torch

from moe_grammar.healthy_prefix_lm import (
    HealthyPrefixTransformer,
    build_shifted_batch,
    reverse_remote_history,
)
from moe_grammar.evaluate_global_prefix import (
    conditional_percentiles,
    cumulative_mean,
    running_max,
)


def test_shifted_batch_never_places_target_in_its_input() -> None:
    projected = np.arange(15, dtype=np.float32).reshape(5, 3)
    words = np.asarray([1, 2, 3, 4, 5], dtype=np.int16)
    batch = build_shifted_batch(
        projected,
        words,
        starts=np.asarray([0, 3]),
        lengths=np.asarray([3, 2]),
        episode_indexes=np.asarray([0, 1]),
        n_words=8,
    )

    np.testing.assert_array_equal(batch.input_words[0], [8, 1, 2, 3])
    np.testing.assert_array_equal(batch.targets[0], [1, 2, 3, 8])
    np.testing.assert_array_equal(batch.input_words[1], [8, 4, 5, 8])
    np.testing.assert_array_equal(batch.targets[1], [4, 5, 8, -100])
    np.testing.assert_array_equal(batch.input_descriptors[0, 1], projected[0])
    np.testing.assert_array_equal(batch.target_descriptors[0, 1], projected[1])


def test_prefix_transformer_cannot_read_future_inputs() -> None:
    torch.manual_seed(7)
    model = HealthyPrefixTransformer(
        descriptor_dim=3,
        n_words=6,
        task_count=2,
        max_positions=8,
        model_dim=16,
        heads=4,
        layers=2,
        feedforward_dim=32,
        dropout=0.0,
    ).eval()
    input_words = torch.randint(0, 7, (2, 7))
    descriptors = torch.randn(2, 7, 3)
    changed_words = input_words.clone()
    changed_descriptors = descriptors.clone()
    changed_words[:, 4:] = torch.randint(0, 7, (2, 3))
    changed_descriptors[:, 4:] += 100.0
    task = torch.tensor([0, 1])

    with torch.no_grad():
        original, _ = model(input_words, descriptors, task)
        changed, _ = model(changed_words, changed_descriptors, task)

    torch.testing.assert_close(original[:, :4], changed[:, :4])


def test_remote_reversal_preserves_last_two_history_words() -> None:
    projected = np.arange(21, dtype=np.float32).reshape(7, 3)
    words = np.arange(7, dtype=np.int16)
    batch = build_shifted_batch(
        projected,
        words,
        starts=np.asarray([0]),
        lengths=np.asarray([7]),
        episode_indexes=np.asarray([0]),
        n_words=8,
        include_end=False,
    )

    reversed_batch = reverse_remote_history(batch, target_position=6)

    np.testing.assert_array_equal(reversed_batch.input_words[0, 1:5], [3, 2, 1, 0])
    np.testing.assert_array_equal(reversed_batch.input_words[0, 5:7], [4, 5])
    assert reversed_batch.targets[0, 6] == batch.targets[0, 6]


def test_prefix_energy_and_position_calibration_are_causal() -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.5, 30.0, np.nan]],
        dtype=np.float64,
    )
    lengths = np.asarray([3, 3, 2])
    energy = cumulative_mean(values, lengths)
    np.testing.assert_allclose(energy[0], [1.0, 1.5, 2.0])
    assert np.isnan(energy[2, 2])

    percentiles = conditional_percentiles(
        energy,
        reference_indexes=np.asarray([0, 1]),
        task=np.zeros(3, dtype=np.int16),
        lengths=lengths,
        task_count=1,
    )
    alarm = running_max(percentiles, lengths)
    assert alarm[2, 1] >= alarm[2, 0]
    assert np.isnan(alarm[2, 2])
