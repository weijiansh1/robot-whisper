# Kolmogorov-inspired train-free routing experiments

This directory tests the concrete ideas against the established HiMoE-VLA
`d9 + cross-chunk Hellinger + W8` baseline:

- a Markov/metastable-state view;
- finite-window symbolic entropy rate;
- normalized compression length;
- routing quantization and natural layer/token measurement budgets;
- a B-only hidden-state effective-dimension control;
- a layer-lag cascade check;
- leave-one-group-out online alarms and the evidence boundary for noise rescue.

The new features do not train an outcome classifier. Labels are used only for
grouped AUCs, permutation tests, and leave-one-group-out alarm-threshold
calibration. The feature definitions use only soft router probabilities; hard
expert IDs are not used.

Run from the workspace root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python analysis_kolmogorov_trainfree/run.py
```

Use `--skip-hidden` for the route-only pass. The B-only hidden control reads a
fixed 0.75 GB slice from the existing 33 GB `hidden.zarr` capture.

The main output is `report.zh.md`; all reported values, including per-group
robustness checks, are also emitted as CSV.
