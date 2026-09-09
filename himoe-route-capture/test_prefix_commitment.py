import numpy as np

from analyze_prefix_commitment import PrefixCell, _integrity
from rollout_prefix_commitment import _prefix_grid_plan, _prefix_noises


def test_prefix_grid_pilot_covers_every_branch_on_the_same_two_by_two_block():
    branches = [4, 8, 10, 12, 14]
    pilot = _prefix_grid_plan(branches, 8, 8)[:20]
    assert {branch for branch, _row, _col in pilot} == set(branches)
    for branch in branches:
        assert [cell[1:] for cell in pilot if cell[0] == branch] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
    full = _prefix_grid_plan(branches, 8, 8)
    assert len(full) == 320
    assert len(set(full)) == 320


def test_prefix_noise_stream_is_exactly_nested_across_branch_points():
    short = _prefix_noises(2000, 4)
    long = _prefix_noises(2000, 14)
    assert np.array_equal(short, long[:4])


def _cell(branch, row, col):
    prefix_noise = _prefix_noises(100 + row, branch)
    future_rng = np.random.default_rng(900 + col)
    future = np.stack(
        [future_rng.standard_normal((10, 24)).astype(np.float32) for _ in range(2)]
    )
    controls = branch + 2
    step = np.arange(controls, dtype=np.float32)
    as_routes = np.broadcast_to(
        (row + step[:branch, None, None] / 100.0), (branch, 4, 3)
    ).copy()
    hb_routes = np.broadcast_to(
        (row + step[:branch, None, None, None, None] / 100.0),
        (branch, 2, 2, 3, 4),
    ).copy()
    states = np.broadcast_to(row + step[:, None] / 10.0, (controls, 8)).copy()
    actions = np.broadcast_to(
        row + step[:, None, None] / 10.0, (controls, 10, 7)
    ).copy()
    return PrefixCell(
        summary={
            "branch_controls": branch,
            "grid_row": row,
            "grid_col": col,
            "success": False,
            "success_before_branch": False,
            "initial_observation_sha256": "same-observation",
        },
        as_prefix=as_routes,
        hb_prefix=hb_routes,
        flow_noises=np.concatenate([prefix_noise, future]),
        states=states,
        action_chunks=actions,
    )


def test_integrity_checks_fixed_prefix_common_future_and_nested_branches():
    branches = [2, 4]
    cells = [_cell(branch, row, col) for branch in branches for row in range(2) for col in range(2)]
    result = _integrity(cells, branches)
    assert result["valid"] is True
    assert result["same_prefix_max_abs_as_route_difference"] == 0.0
    assert result["same_prefix_max_abs_hb_route_difference"] == 0.0
    assert result["same_prefix_max_abs_state_difference"] == 0.0
    assert result["same_prefix_max_abs_action_difference"] == 0.0
    assert result["same_future_noise_mismatches"] == 0
    assert result["nested_prefix_noise_mismatches"] == 0
    assert result["nested_max_abs_hb_route_difference"] == 0.0
