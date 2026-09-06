from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layer_graph_shapes_and_identity_geometry() -> None:
    module = load_module("extract_layer_graphs_gpu", "experiments/extract_layer_graphs_gpu.py")
    probability = np.full((2, 8, 10, 11, 32), 1.0 / 32.0, dtype=np.float32)
    base, conditional, partial, conditional_edges, partial_edges = module.compute_batch(
        probability, torch.device("cpu")
    )
    assert base.shape == (2, 8, 9)
    assert conditional.shape == (2, 8, 55)
    assert partial.shape == (2, 8, 45)
    assert conditional_edges.shape == partial_edges.shape == (2, 8, 45)
    assert np.isfinite(base).all()
    assert np.allclose(base[..., 0], 0.0, atol=1e-6)
    assert np.allclose(base[..., 2], 0.0, atol=1e-6)


def test_empirical_rank_columns() -> None:
    module = load_module("analyze_front_back", "experiments/analyze_front_back.py")
    reference = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)[:, None, None]
    current = np.asarray([0.0, 1.5, 2.0], dtype=np.float32)[:, None, None]
    ranked = module.empirical_rank_columns(reference, current)
    assert np.allclose(ranked[:, 0, 0], [1 / 6, 2 / 3, 5 / 6])
