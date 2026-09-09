"""Directory layout, naming and provenance for the LIBERO-30 routing corpus.

Single source of truth for where a (suite, task, episode) lives on disk.  Both the
capture driver and every analysis script must go through here -- two hand-rolled
path joins drift apart eventually, and the corpus is keyed by directory name.

Naming rules that are NOT negotiable:

* Task directories are ``t{id:02d}__{task_name}``.  Zero padding makes lexical
  order equal numeric order; the double underscore separates the id from a name
  that itself contains single underscores.
* ``task_name`` comes from ``benchmark.get_task(i).name``, frozen once into
  ``_tasks.json``.  It is **not** the alphabetical bddl order: in ``libero_goal``
  task 1 is ``put_the_bowl_on_the_stove`` while the alphabetically-second bddl
  file is ``open_the_top_drawer_and_put_the_bowl_inside`` (which is task 3).
  Regenerating names with ``ls`` silently mislabels the corpus.
* A task directory is never renamed after capture: INDEX.csv and every analysis
  artefact key on it.

Kept import-light and Python 3.8 compatible on purpose: the capture client runs
in the LIBERO env (3.8), the server in the model env (3.11), analysis in neither.
LIBERO itself is *not* imported here -- consumers read the frozen ``_tasks.json``.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCHEMA_VERSION = "route-corpus/1"

#: benchmark name (client ``--benchmark``) -> suite key (server ``--suite``)
SUITE_KEY = {
    "libero_goal": "goal",
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_10": "long",          # bridge --suite value; the paper's LIBERO-Long
}
#: fixed iteration order for the whole corpus
BENCHMARKS = ("libero_goal", "libero_spatial", "libero_object", "libero_10")

#: official per-suite rollout horizon.  goal/spatial/object mirror
#: himoe_libero_bridge.batch.SUITE_MAX_STEPS; ``long`` is missing there, so 520
#: comes from the upstream evaluation script itself
#: (HiMoE-VLA/examples/libero/main.py:67, "longest training demo has 505 steps"),
#: whose other three values match the bridge exactly.
MAX_STEPS = {"goal": 300, "spatial": 220, "object": 280, "long": 520}

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

INDEX_COLUMNS = (
    "suite",
    "task_id",
    "task_dir",
    "task_name",
    "prompt",
    "episode_index",
    "init_state_id",
    "flow_noise_seed",
    "success",
    "action_steps",
    "inference_calls",
    "control_step_offset",
    "path",
    "mig_uuid",
    "mig_profile",
    "wall_s",
)


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #

def task_dir_name(task_id: int, task_name: str) -> str:
    """``t00__open_the_middle_drawer_of_the_cabinet``."""
    if not 0 <= int(task_id) < 100:
        raise ValueError("task_id %r outside the two-digit range" % (task_id,))
    if not task_name or "/" in task_name or task_name != task_name.strip():
        raise ValueError("unusable task_name %r" % (task_name,))
    return "t%02d__%s" % (int(task_id), task_name)


def parse_task_dir_name(name: str) -> Tuple[int, str]:
    """Inverse of :func:`task_dir_name`; raises on anything else."""
    head, sep, task_name = name.partition("__")
    if not sep or not head.startswith("t") or not head[1:].isdigit() or len(head) != 3:
        raise ValueError("not a task directory name: %r" % (name,))
    return int(head[1:]), task_name


def suite_dir(root, benchmark: str) -> pathlib.Path:
    if benchmark not in SUITE_KEY:
        raise ValueError("unknown benchmark %r; expected one of %s" % (benchmark, BENCHMARKS))
    return pathlib.Path(root) / benchmark


def task_dir(root, benchmark: str, task_id: int, task_name: str) -> pathlib.Path:
    return suite_dir(root, benchmark) / task_dir_name(task_id, task_name)


# --------------------------------------------------------------------------- #
# VLA_MUI_HUB layout
# --------------------------------------------------------------------------- #
#
# The hub mirrors /home/jovyan/public-ro/MUI_HUB and its layout is
# ``cache/<model>/<benchmark>/<task>/<run_id>/``.  Two differences from the
# capture layout above, both deliberate on the hub's side:
#
#   * the task directory is the bare task name, with no ``t{NN}__`` prefix, so
#     the id lives only in meta.json -- do not recover it by sorting;
#   * one logical capture run is split across 30 task directories that share a
#     run_id string, which is therefore the join key.

HUB_ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
HUB_MODEL = "HiMoE-VLA"

#: hub directory name per benchmark.  Only libero_10 differs: LIBERO numbers that
#: suite by its task count (LIBERO-100 splits into LIBERO-90 + LIBERO-10), which
#: reads as a version and is easy to misfile -- the other three suites also have
#: ten tasks each.  The hub directory therefore says what the suite *is*, keeping
#: the ``libero_`` prefix the other three use: ``libero_long``.  The LIBERO
#: benchmark id stays ``libero_10`` everywhere it is an argument: the client's
#: --benchmark, the bddl directory and benchmark.get_benchmark_dict() require it.
HUB_DIR = {
    "libero_goal": "libero_goal",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_10": "libero_long",
    "calvin_d_d": "calvin_d_d",
}

#: benchmarks the hub has directories for, vs the ones this capture code can drive.
#: calvin_d_d has no capture path in this toolchain at all.  Failing loudly on it
#: beats a confusing runtime error.
HUB_BENCHMARKS = ("libero_goal", "libero_spatial", "libero_object", "libero_10", "calvin_d_d")
CAPTURABLE = ("libero_goal", "libero_spatial", "libero_object", "libero_10")

#: run_id carries no state and no design numbers -- those live in meta.json and are
#: verified there.  Its only job is to be unique inside a task directory, so the
#: rule is just "a safe slug"; no timestamp is required.
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def validate_run_id(run_id: str) -> str:
    """Reject run ids that need quoting or that would hide mutable state in a name.

    A run id like ``preaction64x8`` is legal but a bad idea: the design numbers it
    claims are not checked against the data, and one such directory in this repo
    holds 28 of the 512 episodes its name implies.  Put the numbers in meta.json.
    """
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ValueError(
            "run_id %r must match %s (lowercase slug, no spaces or slashes)"
            % (run_id, RUN_ID_RE.pattern))
    return run_id


def load_hub_manifest(hub_root=HUB_ROOT) -> Dict[str, Any]:
    path = pathlib.Path(hub_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("hub manifest missing: %s" % path)
    return json.loads(path.read_text())


def iter_hub_tasks(hub_root=HUB_ROOT, benchmarks=None) -> Iterator[Dict[str, Any]]:
    """Planned tasks straight from the hub manifest -- the hub is self-describing.

    Yields the same shape as :func:`iter_planned_tasks` so the two are drop-in
    interchangeable for the capture driver and the validators.
    """
    manifest = load_hub_manifest(hub_root)
    for benchmark in (benchmarks or CAPTURABLE):
        entry = manifest["benchmarks"][benchmark]
        for task in entry["tasks"]:
            record = dict(task)
            record["benchmark"] = benchmark
            record["suite"] = SUITE_KEY.get(benchmark)
            record["dir_name"] = task_dir_name(task["task_id"], task["name"])
            yield record


def sampling_block(design: str, designed_episodes: int, summaries,
                   held_fixed=(), varies=()) -> Dict[str, Any]:
    """Designed vs actual, counted from the data rather than asserted.

    ``complete`` is what consumers should gate on; a partial pilot must never be
    read as if it were the grid its name suggests.
    """
    init_states = {item.get("init_state_id") for item in summaries}
    seeds = {item.get("flow_noise_seed") for item in summaries}
    actual = len(summaries)
    return {
        "design": design,
        "designed_episodes": int(designed_episodes),
        "actual_episodes": actual,
        "coverage": round(actual / designed_episodes, 4) if designed_episodes else None,
        "complete": actual == int(designed_episodes),
        "unique_init_states": len(init_states),
        "unique_flow_noise_seeds": len(seeds),
        "held_fixed": list(held_fixed),
        "varies": list(varies),
    }


def hub_task_dir(task_name: str, benchmark: str, run_id: str,
                 hub_root=HUB_ROOT, model: str = HUB_MODEL) -> pathlib.Path:
    if benchmark not in SUITE_KEY:
        raise ValueError("unknown benchmark %r" % (benchmark,))
    if not run_id or "/" in run_id:
        raise ValueError("unusable run_id %r" % (run_id,))
    return (pathlib.Path(hub_root) / "cache" / model
            / HUB_DIR.get(benchmark, benchmark) / task_name / run_id)


def hub_resolver(run_id: str, hub_root=HUB_ROOT, model: str = HUB_MODEL):
    """Resolver for :func:`index_rows` / :func:`verify_task` over the hub tree."""
    def resolve(planned):
        return hub_task_dir(planned["name"], planned["benchmark"], run_id, hub_root, model)
    return resolve


def corpus_resolver(root):
    def resolve(planned):
        return task_dir(root, planned["benchmark"], planned["task_id"], planned["name"])
    return resolve


def server_dir(task_path) -> pathlib.Path:
    return pathlib.Path(task_path) / "server"


def client_dir(task_path) -> pathlib.Path:
    return pathlib.Path(task_path) / "client"


def episode_filename(episode_index: int) -> str:
    return "episode_%04d.npz" % int(episode_index)


def log_paths(root, benchmark: str, task_id: int) -> Tuple[pathlib.Path, pathlib.Path]:
    logs = pathlib.Path(root) / "_logs"
    stem = "%s-t%02d" % (benchmark, int(task_id))
    return logs / ("srv-%s.log" % stem), logs / ("cli-%s.log" % stem)


# --------------------------------------------------------------------------- #
# frozen task table
# --------------------------------------------------------------------------- #

def load_task_table(root) -> Dict[str, Any]:
    """Read the frozen ``_tasks.json``.  Never re-derive this from the filesystem."""
    path = pathlib.Path(root) / "_tasks.json"
    if not path.is_file():
        raise FileNotFoundError("missing frozen task table: %s" % path)
    return json.loads(path.read_text())


def iter_planned_tasks(root) -> Iterator[Dict[str, Any]]:
    """Yield every (benchmark, task) the corpus is supposed to contain, in order."""
    table = load_task_table(root)["suites"]
    # skip suites the table does not carry: older corpora were frozen with three
    # suites and must keep loading after libero_10 joined BENCHMARKS
    for benchmark in (b for b in BENCHMARKS if b in table):
        for task in table[benchmark]["tasks"]:
            record = dict(task)
            record["benchmark"] = benchmark
            record["suite"] = SUITE_KEY[benchmark]
            record["dir_name"] = task_dir_name(task["task_id"], task["name"])
            yield record


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def _write_json(path, payload: Dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False))
    return path


def write_task_meta(task_path, **fields: Any) -> pathlib.Path:
    """Write/merge ``meta.json``.  ``status`` must be one of the STATUS_* values."""
    path = pathlib.Path(task_path) / "meta.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(fields)
    status = payload.get("status")
    if status not in (STATUS_RUNNING, STATUS_COMPLETE, STATUS_FAILED):
        raise ValueError("meta.json needs a valid status, got %r" % (status,))
    payload["schema_version"] = SCHEMA_VERSION
    return _write_json(path, payload)


def read_task_meta(task_path) -> Optional[Dict[str, Any]]:
    path = pathlib.Path(task_path) / "meta.json"
    return json.loads(path.read_text()) if path.is_file() else None


def is_complete(task_path) -> bool:
    meta = read_task_meta(task_path)
    return bool(meta) and meta.get("status") == STATUS_COMPLETE


def write_suite_meta(root, benchmark: str, **fields: Any) -> pathlib.Path:
    payload = dict(fields)
    payload.update({
        "benchmark": benchmark,
        "suite": SUITE_KEY[benchmark],
        "max_steps": MAX_STEPS[SUITE_KEY[benchmark]],
        "schema_version": SCHEMA_VERSION,
    })
    return _write_json(suite_dir(root, benchmark) / "_suite.json", payload)


def write_manifest(root, **fields: Any) -> pathlib.Path:
    payload = dict(fields)
    payload["schema_version"] = SCHEMA_VERSION
    return _write_json(pathlib.Path(root) / "MANIFEST.json", payload)


def update_manifest(root, **fields: Any) -> pathlib.Path:
    path = pathlib.Path(root) / "MANIFEST.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(fields)
    return _write_json(path, payload)


# --------------------------------------------------------------------------- #
# integrity + index
# --------------------------------------------------------------------------- #

def load_summaries(task_path) -> List[Dict[str, Any]]:
    path = client_dir(task_path) / "summaries.json"
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("episodes", [])
    return list(payload)


def captured_control_steps(task_path) -> Optional[int]:
    """Control steps the *server* says it recorded, from ``capture_summary.json``."""
    path = server_dir(task_path) / "capture_summary.json"
    if not path.is_file():
        return None
    return int(json.loads(path.read_text()).get("control_steps", 0))


def verify_task(task_path, expect_episodes: Optional[int] = None,
                expect_routes: bool = True) -> Dict[str, Any]:
    """Cheap structural checks.  ``ok`` false means do not put this task in INDEX."""
    task_path = pathlib.Path(task_path)
    problems = []  # type: List[str]
    meta = read_task_meta(task_path)
    if meta is None:
        problems.append("missing meta.json")
    elif meta.get("status") != STATUS_COMPLETE:
        problems.append("status=%s" % meta.get("status"))

    summaries = []  # type: List[Dict[str, Any]]
    try:
        summaries = load_summaries(task_path)
    except (OSError, ValueError) as error:
        problems.append("summaries unreadable: %s" % error)

    if expect_episodes is not None and len(summaries) != expect_episodes:
        problems.append("episodes %d != expected %d" % (len(summaries), expect_episodes))

    client_calls = sum(int(item.get("inference_calls", 0)) for item in summaries)
    server_steps = captured_control_steps(task_path)
    if server_steps is None:
        problems.append("missing capture_summary.json")
    elif server_steps != client_calls:
        # the server writes one flat zarr with no episode boundaries; slicing is
        # only valid when these two agree exactly
        problems.append("control_steps %d != sum(inference_calls) %d" % (server_steps, client_calls))

    if expect_routes and not (server_dir(task_path) / "routes.zarr").exists():
        problems.append("missing routes.zarr")

    horizon = MAX_STEPS.get((meta or {}).get("suite", ""))
    if horizon is not None:
        over = [item.get("episode_index") for item in summaries
                if int(item.get("action_steps", 0)) > horizon]
        if over:
            problems.append("action_steps over horizon in episodes %s" % over[:5])

    return {
        "task_dir": task_path.name,
        "ok": not problems,
        "problems": problems,
        "episodes": len(summaries),
        "successes": sum(1 for item in summaries if item.get("success")),
        "control_steps": server_steps,
    }


def index_rows(root, resolve=None) -> Iterator[Dict[str, Any]]:
    """One row per episode, with the zarr offset needed to slice that episode out.

    ``resolve`` maps a planned task to its directory; pass :func:`hub_resolver`
    to index the same corpus after it has been moved under VLA_MUI_HUB.
    """
    resolve = resolve or corpus_resolver(root)
    for planned in iter_planned_tasks(root):
        path = resolve(planned)
        if not is_complete(path):
            continue
        meta = read_task_meta(path) or {}
        offset = 0
        for item in load_summaries(path):
            calls = int(item.get("inference_calls", 0))
            yield {
                "suite": planned["suite"],
                "task_id": planned["task_id"],
                # canonical key, deliberately NOT path.name: under the hub layout
                # the leaf directory is the run_id and is identical for all tasks
                "task_dir": task_dir_name(planned["task_id"], planned["name"]),
                "task_name": planned["name"],
                "prompt": item.get("prompt", planned.get("language", "")),
                "episode_index": item.get("episode_index"),
                "init_state_id": item.get("init_state_id"),
                "flow_noise_seed": item.get("flow_noise_seed"),
                "success": int(bool(item.get("success"))),
                "action_steps": item.get("action_steps"),
                "inference_calls": calls,
                "control_step_offset": offset,
                "path": str(path),
                "mig_uuid": meta.get("mig_uuid", ""),
                "mig_profile": meta.get("mig_profile", ""),
                "wall_s": item.get("wall_s"),
            }
            offset += calls


def build_index(root, resolve=None, out=None) -> Tuple[pathlib.Path, int]:
    path = pathlib.Path(out) if out else pathlib.Path(root) / "INDEX.csv"
    count = 0
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        for row in index_rows(root, resolve):
            writer.writerow(row)
            count += 1
    return path, count
