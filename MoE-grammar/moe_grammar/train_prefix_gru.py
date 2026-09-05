from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from moe_grammar.statistics import stratified_pair_auc


class PrefixGRU(nn.Module):
    def __init__(self, input_dim: int, task_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(task_count, 16)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim + 16, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values)
        embedding = self.task_embedding(task)[:, None].expand(-1, values.shape[1], -1)
        hidden, _ = self.gru(torch.cat([encoded, embedding], dim=2))
        return self.head(hidden).squeeze(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/full40-prefix-q12.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("routing", "behavior", "combined"), default="routing")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def feature_moments(
    features: np.ndarray, indexes: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[2], dtype=np.float64)
    squared = np.zeros(features.shape[2], dtype=np.float64)
    count = 0
    positions = np.arange(features.shape[1])[None, :]
    for start in range(0, len(indexes), 1024):
        selected = indexes[start : start + 1024]
        values = np.asarray(features[selected], dtype=np.float32)
        mask = positions < np.minimum(lengths[selected, None], features.shape[1])
        total += (values * mask[:, :, None]).sum(axis=(0, 1), dtype=np.float64)
        squared += (np.square(values) * mask[:, :, None]).sum(axis=(0, 1), dtype=np.float64)
        count += int(mask.sum())
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def prediction_auc(
    logits: np.ndarray,
    success: np.ndarray,
    task: np.ndarray,
    state: np.ndarray,
    lengths: np.ndarray,
    indexes: np.ndarray,
    horizon: int,
) -> float:
    selected = indexes[lengths[indexes] > horizon]
    labels = ~success[selected]
    strata = np.asarray(
        [f"{item_task}|{item_state}" for item_task, item_state in zip(task[selected], state[selected])],
        dtype=object,
    )
    return stratified_pair_auc(labels, logits[selected, horizon], strata)[0]


@torch.inference_mode()
def predict(
    model: PrefixGRU,
    features: np.ndarray,
    task: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty(features.shape[:2], dtype=np.float32)
    center_tensor = torch.as_tensor(center, device=device)
    scale_tensor = torch.as_tensor(scale, device=device)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        values = torch.as_tensor(
            np.asarray(features[start:stop], dtype=np.float32), device=device
        )
        values = (values - center_tensor) / scale_tensor
        tasks = torch.as_tensor(task[start:stop], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output[start:stop] = model(values, tasks).float().cpu().numpy()
    return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Loading {args.dataset} ({args.modality})...", flush=True)
    with np.load(args.dataset, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if args.modality == "combined":
            features = np.concatenate([payload["routing"], payload["behavior"]], axis=2)
        else:
            features = np.asarray(payload[args.modality])
        lengths = np.asarray(payload["lengths"], dtype=np.int16)
        task = np.asarray(payload["task"], dtype=np.int16)
        state = np.asarray(payload["state"], dtype=np.int16)
        success = np.asarray(payload["success"], dtype=np.bool_)

    split = {name: np.asarray(values, dtype=np.int16) for name, values in metadata["split"].items()}
    train_index = np.flatnonzero(np.isin(state, split["train"]))
    calibration_index = np.flatnonzero(np.isin(state, split["calibration"]))
    center, scale = feature_moments(features, train_index, lengths)
    task_count = len(metadata["tasks"])
    model = PrefixGRU(features.shape[2], task_count, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positive = int(np.sum(~success[train_index]))
    negative = len(train_index) - positive
    positive_weight = torch.tensor(negative / positive, device=device)
    generator = np.random.default_rng(args.seed)
    positions = torch.arange(features.shape[1], device=device)[None, :]
    center_tensor = torch.as_tensor(center, device=device)
    scale_tensor = torch.as_tensor(scale, device=device)
    best_auc = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        permutation = generator.permutation(train_index)
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start : start + args.batch_size]
            values = torch.as_tensor(
                np.asarray(features[selected], dtype=np.float32), device=device
            )
            values = (values - center_tensor) / scale_tensor
            tasks = torch.as_tensor(task[selected], dtype=torch.long, device=device)
            labels = torch.as_tensor(
                (~success[selected]).astype(np.float32), device=device
            )[:, None]
            batch_lengths = torch.as_tensor(
                np.minimum(lengths[selected], features.shape[1]), device=device
            )
            mask = positions < batch_lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(values, tasks)
                loss_by_query = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels.expand_as(logits),
                    reduction="none",
                    pos_weight=positive_weight,
                )
                loss = (
                    (loss_by_query * mask).sum(dim=1) / mask.sum(dim=1)
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        calibration_logits = predict(
            model,
            features[calibration_index],
            task[calibration_index],
            center,
            scale,
            device,
            args.batch_size,
        )
        full_calibration_logits = np.full(features.shape[:2], np.nan, dtype=np.float32)
        full_calibration_logits[calibration_index] = calibration_logits
        auc7 = prediction_auc(
            full_calibration_logits,
            success,
            task,
            state,
            lengths,
            calibration_index,
            7,
        )
        auc12 = prediction_auc(
            full_calibration_logits,
            success,
            task,
            state,
            lengths,
            calibration_index,
            12,
        )
        selection_auc = float(np.nanmean([auc7, auc12]))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "calibration_auc_q7": auc7,
                "calibration_auc_q12": auc12,
                "selection_auc": selection_auc,
            }
        )
        print(
            f"epoch={epoch + 1:02d} loss={np.mean(losses):.4f} "
            f"cal_auc(q7/q12)={auc7:.3f}/{auc12:.3f}",
            flush=True,
        )
        if selection_auc > best_auc:
            best_auc = selection_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    logits = predict(
        model, features, task, center, scale, device, args.batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.modality}_seed{args.seed}"
    np.savez_compressed(
        args.output_dir / f"{stem}_predictions.npz",
        logits=logits,
        lengths=lengths,
        task=task,
        state=state,
        success=success,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "modality": args.modality,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_calibration_mean_auc_q7_q12": best_auc,
                    "history": history,
                    "split": metadata["split"],
                    "tasks": metadata["tasks"],
                    "causality": "unidirectional GRU; logit q depends only on inputs 0..q",
                },
                sort_keys=True,
            )
        ),
    )
    torch.save(
        {
            "state_dict": {name: value.cpu() for name, value in best_state.items()},
            "center": center,
            "scale": scale,
            "input_dim": features.shape[2],
            "task_count": task_count,
            "hidden_dim": args.hidden_dim,
            "metadata": metadata,
        },
        args.output_dir / f"{stem}.pt",
    )
    print(f"Wrote {stem}; best epoch={best_epoch}, calibration AUC={best_auc:.3f}", flush=True)


if __name__ == "__main__":
    main()
