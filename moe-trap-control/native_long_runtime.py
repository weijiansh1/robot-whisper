"""Original LIBERO-Long identity and isolated simulator environment."""

import os
from pathlib import Path
import subprocess

from benchmarks.run_benchmarks import BASE, BRIDGE, CACHE
from collection_storage import atomic_json, digest

ROOT = (BASE / "upstream/LIBERO").resolve()
COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CONFIG = Path("/tmp/himoe-native-long-config-20260909")


def verify_source():
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    if actual != COMMIT or dirty:
        raise ValueError("Original LIBERO source differs from the clean frozen commit")
    return dict(root=str(ROOT), commit=actual, working_tree_clean=True)


def environment(render_gpu, initialize=False):
    if render_gpu not in (0, 3):
        raise ValueError("Native render device must be physical GPU 0 or 3")
    data = ROOT / "libero/libero"
    if initialize:
        atomic_json(CONFIG / "config.yaml", dict(benchmark_root=str(data),
            bddl_files=str(data / "bddl_files"), init_states=str(data / "init_files"),
            assets=str(data / "assets"), datasets=str(ROOT / "datasets")))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MUJOCO_EGL_DEVICE_ID=str(render_gpu),
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LIBERO_CONFIG_PATH=str(CONFIG),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, CACHE / "python-extras", BRIDGE))),
        LD_LIBRARY_PATH=os.pathsep.join(map(str, (CACHE / "system-libs/usr/lib/x86_64-linux-gnu",
            BASE / "system-libs/usr/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"))))
    return env


def inventory():
    import numpy as np
    import libero.libero
    from benchmarks.run_benchmarks import load_suite
    from libero.libero import get_libero_path

    if ROOT not in Path(libero.libero.__file__).resolve().parents:
        raise ValueError("A LIBERO extension was imported instead of original LIBERO")
    suite = load_suite("libero_10", {})
    if suite.n_tasks != 10:
        raise ValueError("Original Long must contain exactly ten tasks")
    rows = []
    for index in range(10):
        task = suite.get_task(index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        initial = np.asarray(suite.get_task_init_states(index))
        if initial.ndim != 2 or len(initial) != 50 or not np.isfinite(initial).all():
            raise ValueError("Invalid official initial states")
        if ROOT not in bddl.resolve().parents or ROOT not in init.resolve().parents:
            raise ValueError("Original task assets point outside original LIBERO")
        rows.append(dict(variant_id="native_long_task%02d" % index, benchmark="native_long",
            category="Original", analysis_role="native", suite="libero_10", registry="libero_10",
            registry_index=index, task_name=task.name, base_task=task.name, horizon_steps=520,
            bddl_path=str(bddl), bddl_sha256=digest(bddl), init_file=str(init),
            init_file_sha256=digest(init), initial_state_count=len(initial), language=task.language))
    return rows
