"""Capture one pooled frozen-PaliGemma prefix feature per policy query."""

from __future__ import annotations

import numpy as np


class VLMFeatureTracer:
    """Pool valid final-layer prefix tokens without changing model outputs."""

    def __init__(self, model) -> None:
        self.model = model
        self.enabled = False
        self._mask = None
        self._features = []
        self._inner_embed_prefix = model.embed_prefix

        def traced_embed_prefix(*args, **kwargs):
            result = self._inner_embed_prefix(*args, **kwargs)
            if self.enabled:
                self._mask = result[1].detach()
            return result

        model.embed_prefix = traced_embed_prefix
        language_model = model.paligemma_with_expert.paligemma.language_model.model
        self.feature_dim = int(language_model.layers[-1].input_layernorm.weight.numel())
        self._handle = language_model.layers[-1].register_forward_hook(
            self._capture, with_kwargs=True
        )

    def _capture(self, _module, _args, _kwargs, output) -> None:
        if not self.enabled:
            return
        if self._mask is None:
            raise RuntimeError("VLM prefix mask was not observed before the final layer")
        import torch

        hidden = (output[0] if isinstance(output, (tuple, list)) else output).float()
        mask = self._mask.to(device=hidden.device, dtype=torch.float32)[..., None]
        denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denominator
        self._features.append(pooled.detach().to("cpu", dtype=torch.float16).numpy())

    def begin(self) -> None:
        self.enabled = True
        self._mask = None
        self._features = []

    def cancel(self) -> None:
        self.enabled = False
        self._mask = None
        self._features = []

    def finish(self) -> np.ndarray:
        try:
            if not self.enabled or len(self._features) != 1:
                raise RuntimeError(
                    "expected exactly one frozen-VLM prefix feature, observed %d"
                    % len(self._features)
                )
            feature = np.asarray(self._features[0], dtype=np.float16)
            if feature.ndim != 2 or feature.shape[-1] != self.feature_dim:
                raise RuntimeError("pooled VLM feature has an unexpected shape")
            if not np.all(np.isfinite(feature)):
                raise RuntimeError("pooled VLM feature contains NaN or infinity")
            return feature
        finally:
            self.cancel()

    def close(self) -> None:
        self.cancel()
        self._handle.remove()
        self.model.embed_prefix = self._inner_embed_prefix
