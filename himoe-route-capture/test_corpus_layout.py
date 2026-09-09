"""Unit tests for the routing-corpus layout.

The interesting cases are the ones that would silently mislabel or silently
truncate the corpus: task ids whose alphabetical order disagrees with the
benchmark order, lexical vs numeric directory sort, and an episode index built
from a zarr whose control-step count does not match the client's.
"""

import csv
import json
import pathlib

import pytest

import corpus_layout as cl


FROZEN = pathlib.Path(__file__).parent / "corpus" / "libero30-right-v1" / "_tasks.json"


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #

def test_task_dir_name_is_zero_padded():
    assert cl.task_dir_name(0, "open_the_middle_drawer_of_the_cabinet") == \
        "t00__open_the_middle_drawer_of_the_cabinet"
    assert cl.task_dir_name(9, "x").startswith("t09__")


def test_lexical_order_equals_numeric_order():
    names = [cl.task_dir_name(i, "task_%d" % i) for i in range(11)]
    assert sorted(names) == names, "zero padding must make ls order == task_id order"


def test_parse_round_trip():
    for task_id, name in ((0, "a_b"), (7, "turn_on_the_stove"), (10, "x")):
        assert cl.parse_task_dir_name(cl.task_dir_name(task_id, name)) == (task_id, name)


@pytest.mark.parametrize("bad", ["t0__x", "tXX__x", "no_prefix", "t00_x", ""])
def test_parse_rejects_foreign_names(bad):
    with pytest.raises(ValueError):
        cl.parse_task_dir_name(bad)


@pytest.mark.parametrize("task_id,name", [(-1, "x"), (100, "x"), (0, ""), (0, "a/b"), (0, " x")])
def test_task_dir_name_rejects_bad_input(task_id, name):
    with pytest.raises(ValueError):
        cl.task_dir_name(task_id, name)


def test_task_dir_name_is_idempotent():
    once = cl.task_dir_name(3, "open_the_top_drawer_and_put_the_bowl_inside")
    assert once == cl.task_dir_name(3, "open_the_top_drawer_and_put_the_bowl_inside")


# --------------------------------------------------------------------------- #
# the actual trap this module exists to prevent
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not FROZEN.is_file(), reason="frozen task table not present")
def test_task_order_is_not_alphabetical():
    """`ls bddl_files/libero_goal` would put open_the_top_drawer... at index 1.

    The benchmark puts it at 3 and gives index 1 to put_the_bowl_on_the_stove.
    If this ever passes with the alphabetical answer, the table was regenerated
    from the filesystem and every task directory in the corpus is mislabelled.
    """
    table = json.loads(FROZEN.read_text())["suites"]["libero_goal"]["tasks"]
    by_id = {item["task_id"]: item["name"] for item in table}
    assert by_id[1] == "put_the_bowl_on_the_stove"
    assert by_id[3] == "open_the_top_drawer_and_put_the_bowl_inside"
    alphabetical = sorted(by_id.values())
    assert [by_id[i] for i in range(len(by_id))] != alphabetical


@pytest.mark.skipif(not FROZEN.is_file(), reason="frozen task table not present")
def test_frozen_table_covers_thirty_tasks():
    """This corpus was frozen with three suites, before libero_10 was usable.

    It must stay a subset of BENCHMARKS rather than equal to it, otherwise adding
    a fourth suite silently invalidates every table frozen before that point.
    """
    suites = json.loads(FROZEN.read_text())["suites"]
    assert set(suites) < set(cl.BENCHMARKS)
    assert sum(suite["n_tasks"] for suite in suites.values()) == 30
    for suite in suites.values():
        assert all(task["n_init_states"] == 50 for task in suite["tasks"])


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_corpus(root, episodes_per_task=2, control_steps=None, status=cl.STATUS_COMPLETE):
    """Minimal two-task corpus; control_steps defaults to the consistent value."""
    root = pathlib.Path(root)
    table = {"suites": {
        benchmark: {"n_tasks": 1, "tasks": [
            {"task_id": 0, "name": "task_zero", "language": "do the thing",
             "bddl_file": "x.bddl", "n_init_states": 50}]}
        for benchmark in cl.BENCHMARKS}}
    (root / "_tasks.json").write_text(json.dumps(table))

    for benchmark in cl.BENCHMARKS:
        path = cl.task_dir(root, benchmark, 0, "task_zero")
        cl.server_dir(path).mkdir(parents=True)
        cl.client_dir(path).mkdir(parents=True)
        (cl.server_dir(path) / "routes.zarr").mkdir()
        summaries = [
            {"episode_index": i, "init_state_id": i, "flow_noise_seed": 1000 + i,
             "success": i % 2 == 0, "action_steps": 30, "inference_calls": 3 + i,
             "wall_s": 12.0, "prompt": "do the thing"}
            for i in range(episodes_per_task)
        ]
        (cl.client_dir(path) / "summaries.json").write_text(json.dumps(summaries))
        total = sum(item["inference_calls"] for item in summaries)
        (cl.server_dir(path) / "capture_summary.json").write_text(json.dumps(
            {"control_steps": total if control_steps is None else control_steps}))
        cl.write_task_meta(path, status=status, suite=cl.SUITE_KEY[benchmark],
                           task_id=0, mig_uuid="MIG-test", mig_profile="2g.35gb")
    return root


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def test_write_task_meta_requires_valid_status(tmp_path):
    with pytest.raises(ValueError):
        cl.write_task_meta(tmp_path, task_id=0)
    with pytest.raises(ValueError):
        cl.write_task_meta(tmp_path, status="donezo")


def test_write_task_meta_merges_and_stamps_schema(tmp_path):
    cl.write_task_meta(tmp_path, status=cl.STATUS_RUNNING, task_id=4)
    cl.write_task_meta(tmp_path, status=cl.STATUS_COMPLETE, wall_s=9.5)
    meta = cl.read_task_meta(tmp_path)
    assert meta["task_id"] == 4, "earlier fields must survive the merge"
    assert meta["status"] == cl.STATUS_COMPLETE
    assert meta["schema_version"] == cl.SCHEMA_VERSION
    assert cl.is_complete(tmp_path)


def test_suite_meta_carries_official_horizon(tmp_path):
    cl.write_suite_meta(tmp_path, "libero_spatial", checkpoint_sha256="abc")
    payload = json.loads((cl.suite_dir(tmp_path, "libero_spatial") / "_suite.json").read_text())
    assert payload["max_steps"] == 220
    assert payload["suite"] == "spatial"


def test_unknown_benchmark_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        cl.suite_dir(tmp_path, "libero_90")


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #

def test_verify_task_accepts_consistent_task(tmp_path):
    _make_corpus(tmp_path)
    report = cl.verify_task(cl.task_dir(tmp_path, "libero_goal", 0, "task_zero"), expect_episodes=2)
    assert report["ok"], report["problems"]
    assert report["episodes"] == 2 and report["control_steps"] == 7


def test_verify_task_catches_control_step_mismatch(tmp_path):
    _make_corpus(tmp_path, control_steps=999)
    report = cl.verify_task(cl.task_dir(tmp_path, "libero_goal", 0, "task_zero"))
    assert not report["ok"]
    assert any("control_steps" in problem for problem in report["problems"])


def test_verify_task_catches_wrong_episode_count(tmp_path):
    _make_corpus(tmp_path, episodes_per_task=2)
    report = cl.verify_task(cl.task_dir(tmp_path, "libero_goal", 0, "task_zero"), expect_episodes=50)
    assert not report["ok"]
    assert any("expected 50" in problem for problem in report["problems"])


def test_verify_task_catches_over_horizon_episode(tmp_path):
    _make_corpus(tmp_path)
    path = cl.task_dir(tmp_path, "libero_spatial", 0, "task_zero")
    summaries = cl.load_summaries(path)
    summaries[0]["action_steps"] = 221  # spatial horizon is 220
    (cl.client_dir(path) / "summaries.json").write_text(json.dumps(summaries))
    report = cl.verify_task(path)
    assert not report["ok"]
    assert any("horizon" in problem for problem in report["problems"])


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #

def test_index_offsets_are_cumulative_within_task(tmp_path):
    _make_corpus(tmp_path, episodes_per_task=3)
    rows = [row for row in cl.index_rows(tmp_path) if row["suite"] == "goal"]
    assert [row["inference_calls"] for row in rows] == [3, 4, 5]
    assert [row["control_step_offset"] for row in rows] == [0, 3, 7]


def test_index_offset_resets_per_task(tmp_path):
    _make_corpus(tmp_path, episodes_per_task=2)
    first_of_each = {}
    for row in cl.index_rows(tmp_path):
        first_of_each.setdefault(row["suite"], row["control_step_offset"])
    assert set(first_of_each.values()) == {0}, "each task has its own zarr"


def test_index_skips_incomplete_tasks(tmp_path):
    _make_corpus(tmp_path, status=cl.STATUS_RUNNING)
    assert list(cl.index_rows(tmp_path)) == []


def test_build_index_writes_all_columns(tmp_path):
    _make_corpus(tmp_path, episodes_per_task=2)
    path, count = cl.build_index(tmp_path)
    assert count == 8  # 4 suites x 1 task x 2 episodes
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(cl.INDEX_COLUMNS)
    assert rows[0]["task_dir"] == "t00__task_zero"
    assert rows[0]["mig_profile"] == "2g.35gb"
    assert {row["success"] for row in rows} == {"0", "1"}


def test_iter_planned_tasks_is_ordered(tmp_path):
    _make_corpus(tmp_path)
    planned = list(cl.iter_planned_tasks(tmp_path))
    assert [item["benchmark"] for item in planned] == list(cl.BENCHMARKS)
    assert all(item["dir_name"] == "t00__task_zero" for item in planned)


def test_load_task_table_missing_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        cl.load_task_table(tmp_path)


# --------------------------------------------------------------------------- #
# hub layout
# --------------------------------------------------------------------------- #

def test_hub_task_dir_shape(tmp_path):
    path = cl.hub_task_dir("turn_on_the_stove", "libero_goal", "tag_20260813_113806",
                           hub_root=tmp_path, model="HiMoE-VLA")
    assert path == (tmp_path / "cache" / "HiMoE-VLA" / "libero_goal"
                    / "turn_on_the_stove" / "tag_20260813_113806")


@pytest.mark.parametrize("run_id", ["", "a/b"])
def test_hub_task_dir_rejects_bad_run_id(run_id):
    with pytest.raises(ValueError):
        cl.hub_task_dir("t", "libero_goal", run_id)


def test_index_task_dir_does_not_collapse_under_hub_layout(tmp_path):
    """Regression: the hub leaf directory is the run_id, identical for all tasks.

    Taking the index key from ``path.name`` made all 1500 rows share one
    task_dir value, which silently destroys the index as a lookup key.
    """
    _make_corpus(tmp_path)
    hub = tmp_path / "hub"
    for benchmark in cl.BENCHMARKS:
        source = cl.task_dir(tmp_path, benchmark, 0, "task_zero")
        destination = cl.hub_task_dir("task_zero", benchmark, "run_1", hub, "M")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    rows = list(cl.index_rows(tmp_path, resolve=cl.hub_resolver("run_1", hub, "M")))
    assert rows, "nothing indexed"
    assert {row["task_dir"] for row in rows} == {"t00__task_zero"}
    assert all(row["path"].endswith("task_zero/run_1") for row in rows)
    assert all("run_1" != row["task_dir"] for row in rows)


# --------------------------------------------------------------------------- #
# hub run ids and sampling provenance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("run_id", ["right-50x1", "left-1x64", "a", "v2.1_pilot"])
def test_validate_run_id_accepts_slugs(run_id):
    assert cl.validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["Right-50x1", "with space", "a/b", "", "-lead", None])
def test_validate_run_id_rejects(run_id):
    with pytest.raises(ValueError):
        cl.validate_run_id(run_id)


def test_sampling_block_counts_instead_of_asserting():
    """The whole point: a run named 64x8 that holds 28 episodes must say so."""
    summaries = [{"init_state_id": 24, "flow_noise_seed": 3000 + i // 8} for i in range(28)]
    block = cl.sampling_block("first-step x future grid", 512, summaries,
                              held_fixed=["init_state"], varies=["flow_noise"])
    assert block["actual_episodes"] == 28
    assert block["designed_episodes"] == 512
    assert block["complete"] is False
    assert block["coverage"] == pytest.approx(0.0547, abs=1e-4)
    assert block["unique_init_states"] == 1


def test_sampling_block_complete_when_counts_match():
    summaries = [{"init_state_id": i, "flow_noise_seed": 1000} for i in range(50)]
    block = cl.sampling_block("init-states x draws", 50, summaries)
    assert block["complete"] is True and block["unique_init_states"] == 50


def test_capturable_is_a_subset_of_hub_benchmarks():
    assert set(cl.CAPTURABLE) < set(cl.HUB_BENCHMARKS)
    assert "libero_10" in cl.CAPTURABLE       # weights verified 2026-08-14
    assert "calvin_d_d" not in cl.CAPTURABLE  # no capture path at all


def test_long_suite_is_wired_end_to_end():
    """LIBERO-Long joined late; every table that keys on suite must know it."""
    assert cl.SUITE_KEY["libero_10"] == "long"
    assert cl.MAX_STEPS["long"] == 520   # upstream examples/libero/main.py:67
    assert "libero_10" in cl.BENCHMARKS
    import run_corpus_capture as rcc
    assert "long" in rcc.CHECKPOINT_DIR
    import validate_corpus as vc
    assert "long" in vc.EXPECTED_SHA and "long" in vc.SECTION_41


def test_iter_planned_tasks_skips_suites_absent_from_the_table(tmp_path):
    """A corpus frozen with three suites must keep loading after a fourth was added."""
    _make_corpus(tmp_path)
    table = json.loads((tmp_path / "_tasks.json").read_text())
    table["suites"].pop("libero_10", None)
    (tmp_path / "_tasks.json").write_text(json.dumps(table))
    benchmarks = [t["benchmark"] for t in cl.iter_planned_tasks(tmp_path)]
    assert "libero_10" not in benchmarks and len(benchmarks) == 3


def test_hub_dir_maps_libero_10_to_libero_long(tmp_path):
    """The hub directory is `libero_long`; the LIBERO benchmark id stays `libero_10`.

    LIBERO numbers that suite by task count, which reads as a version and files
    next to the other three suites that also have ten tasks each.  The client's
    --benchmark must still be libero_10 -- it is the benchmark-dict key.
    """
    assert cl.HUB_DIR["libero_10"] == "libero_long"
    path = cl.hub_task_dir("some_task", "libero_10", "r", hub_root=tmp_path, model="M")
    assert path == tmp_path / "cache" / "M" / "libero_long" / "some_task" / "r"
    for benchmark in ("libero_goal", "libero_spatial", "libero_object"):
        assert cl.HUB_DIR[benchmark] == benchmark, "only libero_10 is renamed"
    libero = [d for b, d in cl.HUB_DIR.items() if b.startswith("libero_")]
    assert all(d.startswith("libero_") for d in libero), "keep the libero_ prefix aligned"


def test_hub_dir_covers_every_hub_benchmark():
    assert set(cl.HUB_DIR) == set(cl.HUB_BENCHMARKS)
