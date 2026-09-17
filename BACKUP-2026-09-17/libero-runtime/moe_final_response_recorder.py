"""Read-only final-flow full-vector capture around the original MoE dispatch."""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, '/data/coding/robot-whisper-0909/himoe-route-capture')
from himoe_state_recorder import discover_blocks


class FinalResponseRecorder:
    def __init__(self, core, expected_flow=10):
        self.blocks = [b for b in discover_blocks(core) if b.kind == 'HB']
        if len(self.blocks) != 8 or any(not b.has_shared for b in self.blocks):
            raise ValueError('expected eight shared HB blocks')
        if any(b.module.training for b in self.blocks):
            raise ValueError('capture is for evaluation mode only')
        self.expected_flow = expected_flow
        self.counts = {b.layer_idx: 0 for b in self.blocks}
        self.rows = {}
        self.experts = {}
        self.handles = []
        self.original = {}
        self.had_instance = {}
        self.attached = False

    def _pre(self, block):
        def hook(_module, args):
            layer = block.layer_idx
            self.counts[layer] += 1
            if self.counts[layer] > self.expected_flow:
                raise ValueError('more flow iterations than expected')
            if self.counts[layer] == self.expected_flow:
                self.rows[layer] = {'input': args[0].detach().float().clone()}
                self.experts[layer] = {}
        return hook

    def _expert(self, block, expert_id):
        def hook(_module, args, output):
            layer = block.layer_idx
            if self.counts[layer] == self.expected_flow:
                if expert_id in self.experts[layer]:
                    raise ValueError('expert evaluated more than once')
                self.experts[layer][expert_id] = (args[0].detach().clone(), output.detach().float().clone())
        return hook

    def _shared(self, block):
        def hook(_module, _args, output):
            if self.counts[block.layer_idx] == self.expected_flow:
                self.rows[block.layer_idx]['shared'] = output.detach().float().clone()
        return hook

    def _block(self, block):
        def hook(_module, _args, output):
            if self.counts[block.layer_idx] == self.expected_flow:
                self.rows[block.layer_idx]['total'] = output.detach().float().clone()
        return hook

    def _dispatch(self, block, original):
        def call(x, flat_expert_indices, flat_expert_weights):
            result = original(x, flat_expert_indices, flat_expert_weights)
            layer = block.layer_idx
            if self.counts[layer] != self.expected_flow:
                return result
            row = self.rows[layer]
            batch, tokens, hidden = row['input'].shape
            if batch != 1 or tokens != 11:
                raise ValueError('expected one state token and ten action tokens')
            order = flat_expert_indices.argsort()
            ends = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
            raw = torch.empty((flat_expert_indices.numel(), hidden), dtype=torch.float32, device=x.device)
            for expert_id, end in enumerate(ends):
                start = 0 if expert_id == 0 else ends[expert_id-1]
                if start == end:
                    continue
                positions = order[start:end]
                input_tokens, outputs = self.experts[layer][expert_id]
                if not torch.equal(input_tokens, x[positions // block.top_k]):
                    raise ValueError('captured expert dispatch order mismatch')
                raw[positions] = outputs
            row['ids'] = flat_expert_indices.detach().reshape(tokens, block.top_k).clone()
            row['weights'] = flat_expert_weights.detach().float().reshape(tokens, block.top_k).clone()
            row['expert_output'] = raw.reshape(tokens, block.top_k, hidden)
            row['routed'] = result.detach().float().reshape(1, tokens, hidden).clone()
            self.experts[layer].clear()
            return result
        return call

    def attach(self):
        if self.attached:
            raise ValueError('recorder already attached')
        for block in self.blocks:
            layer, module = block.layer_idx, block.module
            self.had_instance[layer] = 'moe_infer' in module.__dict__
            self.original[layer] = module.moe_infer
            module.moe_infer = self._dispatch(block, module.moe_infer)
            self.handles.append(module.register_forward_pre_hook(self._pre(block)))
            for expert_id, expert in enumerate(module.experts):
                self.handles.append(expert.register_forward_hook(self._expert(block, expert_id)))
            self.handles.append(module.shared_experts.register_forward_hook(self._shared(block)))
            self.handles.append(module.register_forward_hook(self._block(block)))
        self.attached = True
        return self

    def end(self):
        if not self.attached or any(c != self.expected_flow for c in self.counts.values()):
            raise ValueError('incomplete flow capture')
        keys = ('input', 'ids', 'weights', 'expert_output', 'routed', 'shared', 'total')
        result = {'hb_layers': np.asarray([b.layer_idx for b in self.blocks], np.int16)}
        for key in keys:
            values = []
            for block in self.blocks:
                value = self.rows[block.layer_idx][key]
                if key in ('input', 'routed', 'shared', 'total'):
                    value = value[0]
                values.append(value)
            result[key] = torch.stack(values).to(device='cpu').numpy().copy()
        result['ids'] = result['ids'].astype(np.int16)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('nonfinite captured value')
        reconstructed = (result['expert_output'] * result['weights'][..., None]).sum(axis=-2)
        relative_error = np.linalg.norm(reconstructed-result['routed'], axis=-1) / np.maximum(
            np.linalg.norm(result['routed'], axis=-1), 1e-12)
        result['routed_reconstruction_relative'] = relative_error
        if relative_error.max() > 1e-5:
            raise ValueError('expert outputs do not reconstruct original routed output')
        if not np.array_equal(result['routed'] + result['shared'], result['total']):
            raise ValueError('routed and shared outputs do not reproduce block output exactly')
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block in self.blocks:
            layer, module = block.layer_idx, block.module
            if layer not in self.original:
                continue
            if self.had_instance[layer]:
                module.moe_infer = self.original[layer]
            elif 'moe_infer' in module.__dict__:
                delattr(module, 'moe_infer')
        self.rows.clear()
        self.experts.clear()
        self.attached = False
