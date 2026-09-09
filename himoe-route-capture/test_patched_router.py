"""RoutePatcher unit tests against mock gates -- no checkpoint, no GPU.

What has to hold before this is worth running on the real model:

  * a recorded slot replayed into a later call reproduces the routing bit-exactly
    (this is the self-patch identity check the experiment depends on);
  * the denoise counter advances once per round, not once per gate, so slot keys
    line up across rollouts;
  * a donor whose suffix length differs is refused rather than broadcast.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from serve_patched_router import RoutePatcher

N_EXPERTS = 32
TOP_K = 4
N_TOKEN = 11
HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
N_DENOISE = 10


class MoEGate_load_bal(nn.Module):  # noqa: N801 - discover_gates matches on this name
    """Returns a fresh random top-4 per call, so record and apply cannot coincide."""

    def __init__(self) -> None:
        super().__init__()
        self.n_routed_experts = N_EXPERTS
        self.top_k = TOP_K
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        n_token = hidden_states.shape[0]
        scores = torch.rand(n_token, N_EXPERTS)
        weight, idx = scores.topk(TOP_K, dim=-1, sorted=False)
        return idx, weight / weight.sum(-1, keepdim=True), None


class _Mlp(nn.Module):
    def __init__(self, gate: nn.Module | None) -> None:
        super().__init__()
        if gate is not None:
            self.gate = gate


class _Layer(nn.Module):
    def __init__(self, gate: nn.Module | None) -> None:
        super().__init__()
        self.mlp = _Mlp(gate)


class _Core(nn.Module):
    def __init__(self, hb_layers=tuple(HB_LAYERS), n_layers: int = 18) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _Layer(MoEGate_load_bal() if i in hb_layers else None) for i in range(n_layers)
        )


def _run_call(core: _Core, n_token: int = N_TOKEN, n_denoise: int = N_DENOISE):
    """Drive the gates in execution order: every HB layer, once per denoise round."""
    outputs = []
    for _ in range(n_denoise):
        for i in HB_LAYERS:
            gate = core.layers[i].mlp.gate
            outputs.append(gate(torch.randn(n_token, 1024)))
    return outputs


@pytest.fixture()
def patcher():
    core = _Core()
    p = RoutePatcher(core).attach()
    yield core, p
    p.close()


def test_discovers_only_hb_gates(patcher):
    _, p = patcher
    assert [g.layer_idx for g in p.gates] == HB_LAYERS


def test_hooks_are_inert_until_armed(patcher):
    core, p = patcher
    _run_call(core)
    assert p.recorded_sites == 0
    assert p.applied_sites == 0
    assert p.slots == {}


def test_records_one_entry_per_layer_and_denoise_round(patcher):
    core, p = patcher
    p.begin_call(record="a", apply=None, randomise=False)
    _run_call(core)
    recorded, applied = p.end_call()
    assert recorded == len(HB_LAYERS) * N_DENOISE
    assert applied == 0
    assert sorted(p.slots["a"]) == sorted(
        (layer, d) for d in range(N_DENOISE) for layer in HB_LAYERS
    )


def test_replaying_a_slot_reproduces_it_bit_exactly(patcher):
    core, p = patcher
    p.begin_call(record="a", apply=None, randomise=False)
    donor = _run_call(core)
    p.end_call()

    p.begin_call(record=None, apply="a", randomise=False)
    replayed = _run_call(core)
    _, applied = p.end_call()

    assert applied == len(HB_LAYERS) * N_DENOISE
    for (d_idx, d_weight, _), (r_idx, r_weight, _) in zip(donor, replayed):
        assert torch.equal(d_idx, r_idx)
        assert torch.equal(d_weight, r_weight)


def test_recording_while_applying_stores_the_donor_not_the_gate(patcher):
    """The self-patch arm records and applies in one call; the slot must not drift."""
    core, p = patcher
    p.begin_call(record="a", apply=None, randomise=False)
    _run_call(core)
    p.end_call()
    before = {k: (v[0].clone(), v[1].clone()) for k, v in p.slots["a"].items()}

    p.begin_call(record="b", apply="a", randomise=False)
    _run_call(core)
    p.end_call()

    # Slot b holds what the gate produced, which is unrelated to the donor; the
    # hook records before it overrides, so a self-patch chain stays anchored to
    # the original rather than to the patched output.
    for key, (idx, weight) in before.items():
        assert torch.equal(p.slots["a"][key][0], idx)
        assert torch.equal(p.slots["a"][key][1], weight)


def test_random_mode_changes_experts_but_keeps_router_weights(patcher):
    core, p = patcher
    p.begin_call(record=None, apply=None, randomise=True)
    outputs = _run_call(core)
    _, applied = p.end_call()

    assert applied == len(HB_LAYERS) * N_DENOISE
    for idx, weight, _ in outputs:
        assert idx.shape == (N_TOKEN, TOP_K)
        assert idx.max() < N_EXPERTS
        assert torch.allclose(weight.sum(-1), torch.ones(N_TOKEN), atol=1e-5)


def test_unknown_slot_is_refused(patcher):
    _, p = patcher
    with pytest.raises(KeyError):
        p.begin_call(record=None, apply="missing", randomise=False)


def test_apply_and_random_are_mutually_exclusive(patcher):
    _, p = patcher
    with pytest.raises(ValueError):
        p.begin_call(record=None, apply="a", randomise=True)


def test_donor_with_a_different_suffix_length_is_refused(patcher):
    core, p = patcher
    p.begin_call(record="a", apply=None, randomise=False)
    _run_call(core, n_token=N_TOKEN)
    p.end_call()

    p.begin_call(record=None, apply="a", randomise=False)
    with pytest.raises(RuntimeError, match="suffix length"):
        _run_call(core, n_token=N_TOKEN + 1)
    p.end_call()


def test_donor_with_fewer_denoise_rounds_is_refused(patcher):
    core, p = patcher
    p.begin_call(record="a", apply=None, randomise=False)
    _run_call(core, n_denoise=N_DENOISE - 1)
    p.end_call()

    p.begin_call(record=None, apply="a", randomise=False)
    with pytest.raises(KeyError, match="different number of denoising rounds"):
        _run_call(core, n_denoise=N_DENOISE)
    p.end_call()


def test_self_patch_keeps_the_donor_and_reproduces_it(patcher):
    """record == apply must read the stored slot, not the value just written.

    begin_call used to clear self.slots[record], so the self-patch identity check
    -- the one thing that validates the whole transplant experiment -- read back
    whatever the gate had produced microseconds earlier.  It passed by
    construction, reported a full patch that was a no-op, and destroyed the
    phase-A recording for that seed.  The other tests all use two distinct slot
    names and so never touch this path.
    """
    core, p = patcher
    p.begin_call(record="s", apply=None, randomise=False)
    donor = _run_call(core)
    p.end_call()
    stored = {k: (i.clone(), w.clone()) for k, (i, w) in p.slots["s"].items()}
    assert stored, "phase A recorded nothing"

    p.begin_call(record="s", apply="s", randomise=False)
    replayed = _run_call(core)
    _, applied = p.end_call()

    assert applied == len(HB_LAYERS) * N_DENOISE
    for (d_idx, d_weight, _), (r_idx, r_weight, _) in zip(donor, replayed):
        assert torch.equal(d_idx, r_idx), "self-patch did not return the stored donor"
        assert torch.equal(d_weight, r_weight)
    assert set(p.slots["s"]) == set(stored), "the donor slot lost keys"
