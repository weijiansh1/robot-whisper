# LIBERO Benchmarks and HiMoE Evaluation

Downloaded on 2026-09-14 for evaluation with existing model weights.

| Component | Local path | Pinned revision |
| --- | --- | --- |
| LIBERO-Plus source | `/data/libero-runtime/upstream/LIBERO-plus` | `4976dc30028e805ff8094b55501d532c48fec182` |
| LIBERO-Plus assets | `/data/libero-runtime/downloads/LIBERO-plus/assets.zip` | `dd2bd61b7d9a6fef1abc52d606e983b41886a149` |
| LIBERO-Pro source | `/data/libero-runtime/upstream/LIBERO-PRO` | `eafdb809426b13153aa1e4c42d6601844217dfec` |
| LIBERO-Pro evaluation data | `/data/libero-runtime/datasets/LIBERO-Pro` | `c86fc3b8293185a6f373677018ff3e37f8391602` |

## Official Sources

- [LIBERO-Plus code](https://github.com/sylvestf/LIBERO-plus)
- [LIBERO-Plus evaluation assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus)
- [LIBERO-Pro code](https://github.com/Zxy-MLlab/LIBERO-PRO)
- [LIBERO-Pro evaluation data](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)

GitHub repositories were cloned directly. Hugging Face files were downloaded
through `https://hf-mirror.com` because direct connections timed out.

## Installed Data

LIBERO-Plus assets are extracted to
`upstream/LIBERO-plus/libero/libero/assets`. The published ZIP includes a nested
`inspire/hdd/.../LIBERO-plus-0/assets` prefix; the extracted assets directory was
moved to the documented location. The original archive is retained.

LIBERO-Pro's downloaded `bddl_files` and `init_files` directories are copied into
`upstream/LIBERO-PRO/libero/libero`. The complete dataset repository, including
its metadata and checksum list, is retained under `datasets/LIBERO-Pro`.
Its two Git LFS initialization files were downloaded as actual binary files.

## Verification

- LIBERO-Plus archive size: `6395849578` bytes.
- LIBERO-Plus archive SHA-256:
  `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`.
- ZIP extraction completed successfully, including CRC verification.
- LIBERO-Pro's published checksum list passes after normalizing CRLF line endings
  in the verification stream. The source checksum file is preserved.

Repeat the Pro dataset checksum verification with:

```bash
cd /data/libero-runtime/datasets/LIBERO-Pro
sed 's/\r$//' SHA256SUMS.txt | sha256sum --check --quiet
```

## Evaluation Configuration

Use a separate `LIBERO_CONFIG_PATH` for each simulator process:

| Benchmark | `LIBERO_CONFIG_PATH` | Source path for `PYTHONPATH` |
| --- | --- | --- |
| Plus | `/data/libero-runtime/configs/libero-plus` | `/data/libero-runtime/upstream/LIBERO-plus` |
| Pro | `/data/libero-runtime/configs/libero-pro` | `/data/libero-runtime/upstream/LIBERO-PRO` |

The runner selects these paths automatically, imports one benchmark per process,
and uses EGL rendering. No fine-tuning was performed, and no training
demonstration datasets or additional model weights were fetched.

## Run an Episode

The existing HiMoE policy server is at `ws://127.0.0.1:9500`. Its loaded
checkpoint is `HiMoE-VLA-Libero-10`, SHA-256
`cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256`.
It runs in the Python 3.11 model environment on an A100 40 GB GPU. Keep this
server running when using the commands below.

```bash
/data/libero-runtime/envs/libero/bin/python /data/libero-runtime/run_benchmark.py --benchmark plus
/data/libero-runtime/envs/libero/bin/python /data/libero-runtime/run_benchmark.py --benchmark pro
```

Both commands use the instruction "put both the alphabet soup and the tomato
sauce in the basket", environment seed 7, initial state 0, flow-noise seed 42,
10 actions per model call, a 520-action limit, and 256 x 256 rendering.

- Plus defaults to task index 688, camera viewpoint difficulty 1:
  `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_1_15_100_0_0_initstate_0`.
  This is classification ID 689; Python task indices start at zero.
- Pro defaults to task index 0 in `libero_10_swap`, which changes object
  positions. Its initial state differs from the original task in 29 components.
- `--task-id`, `--task-name`, `--init-state-id`, `--seed`, `--flow-noise-seed`,
  `--max-steps`, `--port`, and `--output-root` select other run settings.
  Other base suites require their matching HiMoE checkpoint on the server.

Each episode saves `summary.json`, `events.jsonl`, `episode.mp4`, and a
hash-checked `episode-trace.json` / `episode-trace.npz` pair under
`/data/libero-runtime/simulations/{plus,pro}`. The summary records the actual
benchmark source revision, task, model metadata, and task success. A
`status` of `completed` means the episode ran without an execution error;
check the separate `success` field for whether the robot achieved its goal.

## Runtime Dependencies and Corrections

Both simulators use `/data/libero-runtime/envs/libero/bin/python` (Python 3.8),
NumPy 1.22.4, robosuite 1.4.0, and MuJoCo 3.2.3. Plus additionally needs
Wand and scikit-image. Their pinned wheels and supporting packages are installed
in `/data/libero-runtime/dependencies/libero-plus`; the runner adds this directory
only for Plus. The existing NumPy and simulator packages were preserved.

The system library `libmagickwand-6.q16-6` is installed for Wand. To reconstruct
the additional Python dependencies using the Python 3.11 environment's pip:

```bash
/data/venv311/bin/python -m pip install \
  --target /data/libero-runtime/dependencies/libero-plus \
  --python-version 3.8 --implementation cp --abi cp38 \
  --platform manylinux2014_x86_64 --only-binary=:all: --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  -r /data/libero-runtime/plus-requirements.txt
```

Two client-side compatibility corrections were needed:

- The bridge accepts the five explicit Pro perturbation suffixes with their
  corresponding base-suite weights and normalization. It still rejects
  mismatched base suites and unknown suffixes.
- Plus task registration now reads the instruction from the parsed BDDL.
  Upstream's filename-derived instruction included camera/initial-state
  parameters. Virtual camera filenames resolve to their underlying BDDL,
  while language variants retain their own instructions. This local source
  change is retained in `libero-plus-bddl-prompt.patch`, and the runner records
  the tracked source diff hash in new summaries.

The interrupted Plus diagnostic episode ending in `20260914T111403Z` used the
unfixed prompt and is explicitly excluded from results in its summary.

## Verified Episode Results

Both complete runs finished on 2026-09-14 with `status: completed` and
`success: false`. They reached the 520-action limit without achieving the
environment's success condition.

| Benchmark | Perturbation | Actions | Model calls | Wall time | Mean request latency |
| --- | --- | --- | --- | --- | --- |
| Plus | Camera viewpoint, difficulty 1 | 520 | 52 | 137.91 s | 878.21 ms |
| Pro | Object positions, `swap` | 520 | 52 | 138.44 s | 875.77 ms |

Artifacts:

- Plus: [summary](simulations/plus/episode-libero-10-task688-seed7-20260914T111709Z/summary.json),
  [video](simulations/plus/episode-libero-10-task688-seed7-20260914T111709Z/episode.mp4).
- Pro: [summary](simulations/pro/episode-libero-10-swap-task00-seed7-20260914T111107Z/summary.json),
  [video](simulations/pro/episode-libero-10-swap-task00-seed7-20260914T111107Z/episode.mp4).

Both videos decode all 531 frames at 256 x 256, including the initial frame
and 10 settling steps. Start/middle/end frames were visually inspected;
the scenes and robot motion render correctly. Complete trace manifests,
array hashes, array shapes, finite action values, and event counts passed
validation. Each episode directory also contains `frames-start-middle-end.png`.
The corrected Plus source diff SHA-256 is
`4e3dde7b5baa5d668682fef2528cee4f4b42ef4661eb5f8f05546e70d61597dc`.

## Coverage Limits

These commands run one representative episode each, not the full benchmarks.
Success rates require evaluating more tasks and initial states. Other Plus
perturbation categories and Pro perturbation suites have not been run here.

A subsequent [60-episode sampling study](samples/STUDY.md) evaluated all
10 LIBERO-10 base tasks under Plus camera and Pro position perturbations.
Plus succeeded in 9/30 episodes and Pro in 0/30, with no runtime errors.
See the [full sampling report](samples/report-20260914/RESULTS.md) for failure
rates, confidence intervals, and the successful unperturbed Pro control.

The Pro dataset also contains a newer seven-case extension. Its dataset card
requires a parser and runtime that support `:perturbation_config`; the pinned
upstream Python source contains no implementation of that field. That extension
needs integration before evaluation. Preserve the pinned versions when comparing
results.
