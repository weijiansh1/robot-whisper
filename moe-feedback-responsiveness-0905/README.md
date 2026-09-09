# MoE feedback responsiveness Stage 2

This bundle tests a train-free, MoE-only interpretation of the frozen v4 alarm:

1. v4 marks original-horizon deadline risk;
2. a causal Stage 2 monitors whether routing structure restarts after the alarm;
3. at the budget boundary, the restart evidence ranks `+10` late success versus
   persistent failure.

No event-relative `q-2` anchor is used. The only anchors are the frozen first v4
alarm, causal offsets after that alarm, and the original task horizon.

## Reproduce

The lightweight mobility analysis reuses existing caches:

```bash
python experiments/analyze_stage2.py --bootstrap 5000
```

The graph profile build reads every value in the original external
`hb_router_probs[8,10,11,32]` tensors and uses GPUs 6 and 7:

```bash
CUDA_VISIBLE_DEVICES=6,7 python experiments/build_external_graph_profiles_gpu.py --resume
python experiments/analyze_graph_stage2.py --bootstrap 5000
python experiments/analyze_fusion.py --bootstrap 5000
pytest -q
```

See `REPORT_ZH.md` for interpretation and validity limits. Machine-readable
results are under `results/`; the 456 MiB bundle does not duplicate raw Zarr
routes.
