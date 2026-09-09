import json

import numpy as np
import zarr

import analyze_runtime_scalar_pilot as pilot


def _moving_slot_fixture():
    ids = np.empty((4, 10, 1, 2), dtype=np.int64)
    raw = np.full(ids.shape, 99.0, dtype=np.float64)
    weight = np.full(ids.shape, 0.5, dtype=np.float64)
    candidate_ids = np.asarray([[7, 0], [1, 7], [7, 2], [3, 7]])
    candidate_raw = np.asarray([[1.0, 99.0], [99.0, 2.0], [3.0, 99.0], [99.0, 4.0]])
    for denoise in range(10):
        ids[:, denoise, 0] = candidate_ids
        raw[:, denoise, 0] = candidate_raw + denoise
    return ids, raw, weight


def test_fixed_expert_round_concordance_tracks_identity_across_slots():
    ids, raw, _ = _moving_slot_fixture()
    target = np.arange(4.0)
    result = pilot.fixed_expert_round_concordance(ids, raw, target, 6, 4)
    assert result["pair_concordance"] == 1.0
    assert result["valid_token_expert_strata"] == 1
    assert result["median_candidates_per_stratum"] == 4.0


def test_persistent_rebound_requires_all_rounds_and_uses_weighted_curve():
    ids, raw, weight = _moving_slot_fixture()
    # Expert 7 has weighted curves whose rebound grows with candidate ID.
    weight[:, :, 0] = 0.25
    for candidate in range(4):
        hit = ids[candidate, :, 0] == 7
        value = np.linspace(0.1, 0.1 + candidate, 10)
        weight[candidate, :, 0][hit] = value[:, None].repeat(2, axis=1)[hit]
    target = np.arange(4.0)
    result = pilot.persistent_weighted_rebound_concordance(
        ids, raw, weight, target, min_selected=4
    )
    assert result["pair_concordance"] == 1.0
    assert result["valid_persistent_token_expert_strata"] == 1

    ids[3, 5, 0] = np.asarray([3, 4])
    missing = pilot.persistent_weighted_rebound_concordance(
        ids, raw, weight, target, min_selected=4
    )
    assert missing["pair_concordance"] is None


def test_flow_targets_align_round_input_update_and_endpoint():
    slope = np.asarray([1.0, 2.0])
    time = np.arange(11, dtype=np.float64)
    x = slope[:, None, None, None] * time[None, :, None, None]
    x = np.broadcast_to(x, (2, 11, 10, 7)).copy()
    targets = pilot.flow_targets(x)
    np.testing.assert_allclose(
        targets["immediate_xd_xd1"],
        np.broadcast_to(slope[:, None], (2, 10)),
    )
    np.testing.assert_allclose(
        targets["remaining_xd_x10"],
        slope[:, None] * (10 - time[:10]),
    )
    np.testing.assert_allclose(targets["total_x0_x10"], slope * 10)


def test_runtime_absolute_tension_uses_captured_gate_values():
    weight = np.asarray([[[0.25, 0.75]]])
    raw = np.asarray([[[3.0, 1.0]]])
    routed = np.asarray([[1.5]])
    result = pilot.runtime_absolute_tension(weight, raw, routed)
    np.testing.assert_allclose(result["D_abs"], [[np.sqrt(0.75)]])
    np.testing.assert_allclose(result["C_abs"], [[0.0]])


def test_task_loader_accepts_v2_rms_store(tmp_path):
    run = tmp_path / "v2-run"
    run.mkdir()
    root = zarr.create_group(store=str(run / "activation_flow.zarr"), overwrite=True)
    root.attrs.update(
        {
            "format": "himoe_hb_activation_flow_v2",
            "hb_layers": [5],
        }
    )
    ids = np.broadcast_to(
        np.arange(4, dtype=np.uint8), (16, 1, 10, 10, 4)
    ).copy()
    candidate_scale = np.arange(1, 17, dtype=np.float32).reshape(16, 1, 1, 1, 1)
    raw = np.broadcast_to(candidate_scale, ids.shape).copy()
    weight = np.full(ids.shape, 0.25, dtype=np.float32)
    routed = np.broadcast_to(
        0.5 * candidate_scale[..., 0], (16, 1, 10, 10)
    ).copy()
    trajectory = np.zeros((16, 11, 10, 7), dtype=np.float32)
    trajectory[:, :, :, 0] = np.arange(16, dtype=np.float32)[:, None, None]
    for name, value in {
        "hb_selected_expert_id": ids,
        "hb_selected_expert_weight": weight,
        "hb_selected_expert_raw_rms": raw,
        "hb_routed_rms": routed,
        "x_traj": trajectory,
        "control_step": np.arange(16, dtype=np.int32),
    }.items():
        root.create_array(name, data=value)
    (run / "experiment_config.json").write_text(
        json.dumps({"noise_seed_base": 1000})
    )
    (run / "capture_summary.json").write_text(json.dumps({"queries": 16}))
    (run / "query_records.json").write_text(json.dumps([{"state": 0}]))

    loaded = pilot._load_task("toy", run, layer=5, min_selected=2)
    assert loaded["attrs"]["format"] == "himoe_hb_activation_flow_v2"
    assert loaded["validation"]["candidate_count"] == 16
