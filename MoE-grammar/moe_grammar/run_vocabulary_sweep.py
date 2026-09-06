"""Vocabulary-size sweep with many initializations and a word-stability check.

The design audit asked for `K in {8, 16, 32, 64}` with at least ten initializations and an
AMI/stability check on word assignments across folds and seeds. The shipped tokenizer
instead hard-codes one `K` with `n_init=2` and neither check was ever run. Both are done
here on the GPU, where all initializations of one `K` fit as a single batched EM.

Selection uses held-out successful queries only. Stability is the adjusted mutual
information between word assignments produced by independent seeds on the same fold, and
between folds on their shared queries, so a codebook that merely reshuffles labels is not
counted as unstable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from moe_grammar.gpu_tokenizer import BatchedDiagonalGMM

VOCABULARY_SIZES = (8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/vocabulary-sweep.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-vocabulary-sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--stability-seeds", type=int, default=3)
    parser.add_argument(
        "--vocabulary-sizes",
        type=int,
        nargs="+",
        default=list(VOCABULARY_SIZES),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def held_out_nll(model: BatchedDiagonalGMM, values: np.ndarray) -> float:
    """Mean negative log-likelihood per held-out query, in bits."""
    return float(-model.score_samples(values).mean() / np.log(2.0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset)
    started = time.time()

    per_size: dict[int, dict[str, Any]] = {}
    assignments: dict[tuple[int, int, int], np.ndarray] = {}
    shared_rows: dict[int, np.ndarray] = {}

    for size in args.vocabulary_sizes:
        fold_nll: list[float] = []
        fold_spread: list[float] = []
        seed_stability: list[float] = []
        for fold in args.folds:
            projected = data[f"projected_{fold}"]
            train = projected[data[f"train_rows_{fold}"]]
            test = projected[data[f"test_rows_{fold}"]]
            models = []
            for offset in range(args.stability_seeds):
                model = BatchedDiagonalGMM(
                    n_components=size,
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    seed=args.seed + 1000 * offset + fold,
                    device=args.device,
                ).fit(train)
                models.append(model)
            fold_nll.append(held_out_nll(models[0], test))
            fold_spread.append(float(np.ptp(models[0].all_lower_bounds_)))

            # Same fold, independent seeds: does the codebook reproduce?
            probe = test[:: max(1, len(test) // 20000)]
            labels = [model.predict(probe) for model in models]
            seed_stability.extend(
                adjusted_mutual_info_score(labels[i], labels[j])
                for i in range(len(labels))
                for j in range(i + 1, len(labels))
            )
            assignments[(size, fold, 0)] = models[0].predict(projected[:: 97])
            shared_rows[fold] = np.arange(0, len(projected), 97)
            print(
                f"K={size:4d} fold {fold}: held-out {fold_nll[-1]:.4f} bits, "
                f"init spread {fold_spread[-1]:.4f}",
                flush=True,
            )

        cross_fold = [
            adjusted_mutual_info_score(assignments[(size, a, 0)], assignments[(size, b, 0)])
            for a in args.folds
            for b in args.folds
            if a < b
        ]
        per_size[size] = {
            "held_out_bits_per_query_mean": float(np.mean(fold_nll)),
            "held_out_bits_per_query_folds": fold_nll,
            "initialization_bound_spread_mean": float(np.mean(fold_spread)),
            "same_fold_seed_ami_mean": float(np.mean(seed_stability)),
            "same_fold_seed_ami_min": float(np.min(seed_stability)),
            "cross_fold_ami_mean": float(np.mean(cross_fold)),
            "cross_fold_ami_min": float(np.min(cross_fold)),
        }

    best = min(per_size, key=lambda k: per_size[k]["held_out_bits_per_query_mean"])
    summary = {
        "schema_version": 1,
        "protocol": {
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "stability_seeds": args.stability_seeds,
            "folds": args.folds,
            "selection": "held-out successful queries only",
        },
        "by_vocabulary_size": {str(k): v for k, v in per_size.items()},
        "selected_vocabulary_size": best,
        "shipped_vocabulary_size": 64,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# 词表规模扫描与词稳定性",
        "",
        f"每个 K 用 {args.n_init} 次初始化批量 EM，{args.stability_seeds} 个独立 seed 重复；",
        "选择只用 held-out 成功 query 的 NLL。AMI 对标签重排不敏感，因此只惩罚真正的划分变化。",
        "",
        "| K | held-out bits/query | 初始化 bound 极差 | 同折跨 seed AMI | 跨折 AMI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size in args.vocabulary_sizes:
        row = per_size[size]
        mark = " **<- 选中**" if size == best else ""
        lines.append(
            f"| {size} | {row['held_out_bits_per_query_mean']:.4f} | "
            f"{row['initialization_bound_spread_mean']:.4f} | "
            f"{row['same_fold_seed_ami_mean']:.3f} | "
            f"{row['cross_fold_ami_mean']:.3f} |{mark}"
        )
    lines += [
        "",
        f"当前实现固定使用 K=64、`n_init=2`；本扫描选出的是 K={best}。",
        "",
        "初始化 bound 极差量化了 `n_init=2` 的风险：极差越大，两次初始化越可能停在明显更差的解上。",
        "AMI 量化审计关心的“词是否稳定”：同折跨 seed 高、跨折低，说明词汇表依赖于具体训练 state。",
    ]
    (args.output_dir / "REPORT.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output_dir} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
