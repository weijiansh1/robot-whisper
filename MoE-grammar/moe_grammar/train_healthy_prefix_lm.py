from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from moe_grammar.healthy_prefix_lm import (
    HealthyPrefixTransformer,
    PrefixBatch,
    build_shifted_batch,
    component_log_likelihood_torch,
    reverse_remote_history,
    soft_word_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-score-queries", type=int, default=13)
    parser.add_argument("--order-control-max-query", type=int, default=12)
    return parser.parse_args()


def tensor_batch(batch: PrefixBatch, task: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_words": torch.as_tensor(batch.input_words, device=device),
        "input_descriptors": torch.as_tensor(batch.input_descriptors, device=device),
        "targets": torch.as_tensor(batch.targets, device=device),
        "target_descriptors": torch.as_tensor(batch.target_descriptors, device=device),
        "valid": torch.as_tensor(batch.valid, device=device),
        "word_target": torch.as_tensor(batch.word_target, device=device),
        "task": torch.as_tensor(task, dtype=torch.long, device=device),
    }


@torch.inference_mode()
def validation_loss(
    model: HealthyPrefixTransformer,
    projected: np.ndarray,
    words: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    task: np.ndarray,
    indexes: np.ndarray,
    component_means: torch.Tensor,
    component_covariances: torch.Tensor,
    component_weights: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    weighted_loss = 0.0
    target_count = 0
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        batch = build_shifted_batch(
            projected, words, starts, lengths, selected, model.n_words
        )
        tensors = tensor_batch(batch, task[selected], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(
                tensors["input_words"],
                tensors["input_descriptors"],
                tensors["task"],
                tensors["valid"],
            )
        loss = soft_word_loss(
            logits.float(),
            tensors["targets"],
            tensors["target_descriptors"].float(),
            tensors["valid"],
            tensors["word_target"],
            component_means,
            component_covariances,
            component_weights,
        )
        count = int(batch.valid.sum())
        weighted_loss += float(loss.cpu()) * count
        target_count += count
    return weighted_loss / target_count


def fit_hidden_reference(
    hidden: np.ndarray,
    task: np.ndarray,
    lengths: np.ndarray,
    train_success: np.ndarray,
    task_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = hidden.shape[1]
    dimensions = hidden.shape[2]
    centers = np.zeros((task_count, positions, dimensions), dtype=np.float32)
    variances = np.ones_like(centers)
    global_values = np.asarray(hidden[train_success], dtype=np.float32)
    global_valid = np.arange(positions)[None, :] < lengths[train_success, None]
    fallback = global_values[global_valid]
    fallback_center = fallback.mean(axis=0)
    fallback_variance = fallback.var(axis=0)
    for task_index in range(task_count):
        for position in range(positions):
            selected = train_success[
                (task[train_success] == task_index) & (lengths[train_success] > position)
            ]
            values = (
                np.asarray(hidden[selected, position], dtype=np.float32)
                if len(selected) >= 32
                else fallback
            )
            center = values.mean(axis=0) if len(values) else fallback_center
            variance = values.var(axis=0) if len(values) else fallback_variance
            floor = max(float(np.median(variance)) * 0.05, 1e-5)
            centers[task_index, position] = center
            variances[task_index, position] = np.maximum(variance, floor)
    return centers, variances


@torch.inference_mode()
def predict_all(
    model: HealthyPrefixTransformer,
    projected: np.ndarray,
    words: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    task: np.ndarray,
    component_means: torch.Tensor,
    component_covariances: torch.Tensor,
    device: torch.device,
    batch_size: int,
    hidden_score_queries: int,
) -> dict[str, np.ndarray]:
    model.eval()
    episode_count = len(lengths)
    maximum_length = int(lengths.max())
    shape = (episode_count, maximum_length)
    output = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in ("global_hard", "global_soft", "prototype_distance", "next_entropy")
    }
    hidden_queries = min(hidden_score_queries, maximum_length)
    hidden_output = np.full(
        (episode_count, hidden_queries, model.model_dim), np.nan, dtype=np.float16
    )
    indexes = np.arange(episode_count, dtype=np.int64)

    for batch_start in range(0, episode_count, batch_size):
        selected = indexes[batch_start : batch_start + batch_size]
        batch = build_shifted_batch(
            projected, words, starts, lengths, selected, model.n_words
        )
        tensors = tensor_batch(batch, task[selected], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, hidden = model(
                tensors["input_words"],
                tensors["input_descriptors"],
                tensors["task"],
                tensors["valid"],
            )
        logits = logits.float()
        hidden = hidden.float()
        full_log_probability = torch.log_softmax(logits, dim=2)
        word_log_probability = torch.log_softmax(logits[:, :, :-1], dim=2)
        component = component_log_likelihood_torch(
            tensors["target_descriptors"].float(),
            component_means,
            component_covariances,
        )
        soft = -torch.logsumexp(word_log_probability + component, dim=2)
        safe_targets = tensors["targets"].clamp(0, model.n_words)
        hard = -torch.gather(full_log_probability, 2, safe_targets[:, :, None]).squeeze(2)
        probability = torch.softmax(logits[:, :, :-1], dim=2)
        predicted_mean = probability @ component_means
        prototype_distance = torch.sqrt(
            torch.square(predicted_mean - tensors["target_descriptors"]).mean(dim=2)
        )
        entropy = -(torch.softmax(logits, dim=2) * full_log_probability).sum(dim=2)

        hard_numpy = hard.cpu().numpy()
        soft_numpy = soft.cpu().numpy()
        distance_numpy = prototype_distance.cpu().numpy()
        entropy_numpy = entropy.cpu().numpy()
        hidden_numpy = hidden.cpu().numpy()
        for row, episode_index in enumerate(selected):
            count = int(lengths[episode_index])
            output["global_hard"][episode_index, :count] = hard_numpy[row, :count]
            output["global_soft"][episode_index, :count] = soft_numpy[row, :count]
            output["prototype_distance"][episode_index, :count] = distance_numpy[row, :count]
            output["next_entropy"][episode_index, :count] = entropy_numpy[row, 1 : count + 1]
            hidden_count = min(count, hidden_queries)
            hidden_output[episode_index, :hidden_count] = hidden_numpy[
                row, 1 : hidden_count + 1
            ].astype(np.float16)
        if (batch_start // batch_size + 1) % 8 == 0 or batch_start + batch_size >= episode_count:
            print(
                f"  inference {min(batch_start + batch_size, episode_count):,}/"
                f"{episode_count:,} episodes",
                flush=True,
            )
    output["hidden"] = hidden_output
    return output


@torch.inference_mode()
def predict_remote_reversed(
    model: HealthyPrefixTransformer,
    projected: np.ndarray,
    words: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    task: np.ndarray,
    test_success: np.ndarray,
    component_means: torch.Tensor,
    component_covariances: torch.Tensor,
    device: torch.device,
    batch_size: int,
    maximum_query: int,
) -> np.ndarray:
    model.eval()
    output = np.full((len(lengths), int(lengths.max())), np.nan, dtype=np.float32)
    for query in range(4, min(maximum_query, int(lengths.max()) - 1) + 1):
        eligible = test_success[lengths[test_success] > query]
        for start in range(0, len(eligible), batch_size):
            selected = eligible[start : start + batch_size]
            batch = build_shifted_batch(
                projected,
                words,
                starts,
                lengths,
                selected,
                model.n_words,
                include_end=False,
                maximum_words=query + 1,
            )
            batch = reverse_remote_history(batch, query)
            tensors = tensor_batch(batch, task[selected], device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(
                    tensors["input_words"],
                    tensors["input_descriptors"],
                    tensors["task"],
                    tensors["valid"],
                )
            word_log_probability = torch.log_softmax(
                logits[:, query, :-1].float(), dim=1
            )
            component = component_log_likelihood_torch(
                tensors["target_descriptors"][:, query : query + 1].float(),
                component_means,
                component_covariances,
            )[:, 0]
            value = -torch.logsumexp(word_log_probability + component, dim=1)
            output[selected, query] = value.cpu().numpy()
        print(f"  remote-order control q{query}: {len(eligible):,} successes", flush=True)
    return output


def model_configuration(model: HealthyPrefixTransformer) -> dict[str, Any]:
    return {
        "descriptor_dim": model.descriptor_dim,
        "n_words": model.n_words,
        "task_count": model.task_embedding.num_embeddings,
        "max_positions": model.max_positions,
        "model_dim": model.model_dim,
        "heads": model.heads,
        "layers": model.layers,
        "feedforward_dim": model.feedforward_dim,
        "dropout": model.dropout,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading {args.dataset}...", flush=True)
    with np.load(args.dataset, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        projected = np.asarray(payload["projected"])
        words = np.asarray(payload["words"], dtype=np.int16)
        starts = np.asarray(payload["starts"], dtype=np.int32)
        lengths = np.asarray(payload["lengths"], dtype=np.int16)
        task = np.asarray(payload["task"], dtype=np.int16)
        state = np.asarray(payload["state"], dtype=np.int16)
        success = np.asarray(payload["success"], dtype=np.bool_)
        means_numpy = np.asarray(payload["component_means"], dtype=np.float32)
        covariances_numpy = np.asarray(payload["component_covariances"], dtype=np.float32)
        weights_numpy = np.asarray(payload["component_weights"], dtype=np.float32)

    split = {name: np.asarray(value, dtype=np.int16) for name, value in metadata["split"].items()}
    train_success = np.flatnonzero(np.isin(state, split["train"]) & success)
    calibration_success = np.flatnonzero(np.isin(state, split["calibration"]) & success)
    test_success = np.flatnonzero(np.isin(state, split["test"]) & success)
    if not np.all(success[train_success]) or not np.all(success[calibration_success]):
        raise AssertionError("healthy LM must never train/select on failed episodes")
    print(
        f"Healthy-only split: train={len(train_success):,}, "
        f"validation={len(calibration_success):,}, test={len(test_success):,}",
        flush=True,
    )

    component_means = torch.as_tensor(means_numpy, device=device)
    component_covariances = torch.as_tensor(covariances_numpy, device=device)
    component_weights = torch.as_tensor(weights_numpy, device=device)
    model = HealthyPrefixTransformer(
        descriptor_dim=projected.shape[1],
        n_words=means_numpy.shape[0],
        task_count=len(metadata["tasks"]),
        max_positions=int(lengths.max()) + 1,
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = generator.permutation(train_success)
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start : start + args.batch_size]
            batch = build_shifted_batch(
                projected, words, starts, lengths, selected, model.n_words
            )
            tensors = tensor_batch(batch, task[selected], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(
                    tensors["input_words"],
                    tensors["input_descriptors"],
                    tensors["task"],
                    tensors["valid"],
                )
            loss = soft_word_loss(
                logits.float(),
                tensors["targets"],
                tensors["target_descriptors"].float(),
                tensors["valid"],
                tensors["word_target"],
                component_means,
                component_covariances,
                component_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        current_validation = validation_loss(
            model,
            projected,
            words,
            starts,
            lengths,
            task,
            calibration_success,
            component_means,
            component_covariances,
            component_weights,
            device,
            args.batch_size,
        )
        train_loss = float(np.mean(losses))
        history.append(
            {"epoch": epoch, "train_soft_ce": train_loss, "healthy_validation_soft_ce": current_validation}
        )
        print(
            f"epoch={epoch:02d} train_soft_ce={train_loss:.4f} "
            f"healthy_val_soft_ce={current_validation:.4f}",
            flush=True,
        )
        if current_validation < best_loss - 1e-4:
            best_loss = current_validation
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} stale epochs", flush=True)
                break

    if best_state is None:
        raise AssertionError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    print(f"Best healthy validation epoch={best_epoch}, loss={best_loss:.4f}", flush=True)

    predictions = predict_all(
        model,
        projected,
        words,
        starts,
        lengths,
        task,
        component_means,
        component_covariances,
        device,
        args.batch_size,
        args.hidden_score_queries,
    )
    hidden = predictions.pop("hidden")
    centers, variances = fit_hidden_reference(
        hidden,
        task,
        lengths,
        train_success,
        len(metadata["tasks"]),
    )
    hidden_mahalanobis = np.full(
        (len(lengths), int(lengths.max())), np.nan, dtype=np.float32
    )
    for episode_index in range(len(lengths)):
        count = min(int(lengths[episode_index]), hidden.shape[1])
        difference = (
            np.asarray(hidden[episode_index, :count], dtype=np.float32)
            - centers[task[episode_index], :count]
        )
        hidden_mahalanobis[episode_index, :count] = np.mean(
            np.square(difference) / variances[task[episode_index], :count], axis=1
        )
    del hidden
    predictions["hidden_mahalanobis"] = hidden_mahalanobis
    predictions["remote_reversed_soft"] = predict_remote_reversed(
        model,
        projected,
        words,
        starts,
        lengths,
        task,
        test_success,
        component_means,
        component_covariances,
        device,
        args.batch_size,
        args.order_control_max_query,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "healthy_prefix_lm.pt"
    predictions_path = args.output_dir / "predictions.npz"
    checkpoint_metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "training": "train-state successful episodes only",
        "selection": "minimum calibration-state successful soft-word cross entropy",
        "failed_episodes_used_for_training_or_selection": False,
        "best_epoch": best_epoch,
        "best_healthy_validation_soft_ce": best_loss,
        "history": history,
        "split": metadata["split"],
        "tasks": metadata["tasks"],
        "causality": (
            "output q sees BOS plus words/descriptors 0..q-1; hidden q+1 sees prefix 0..q"
        ),
        "similarity_score": "-log sum_k p_LM(k|prefix) p_GMM(z_q|k)",
        "model": model_configuration(model),
    }
    torch.save(
        {
            "state_dict": {name: value.cpu() for name, value in best_state.items()},
            "metadata": checkpoint_metadata,
            "component_means": means_numpy,
            "component_covariances": covariances_numpy,
            "component_weights": weights_numpy,
            "hidden_centers": centers,
            "hidden_variances": variances,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        predictions_path,
        **predictions,
        lengths=lengths,
        task=task,
        state=state,
        success=success,
        metadata_json=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
    )
    print(
        f"Wrote {checkpoint_path} and {predictions_path} "
        f"({predictions_path.stat().st_size / 2**20:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
