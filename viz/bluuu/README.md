# HiMoE-VLA Manim explainer

This directory contains a three-scene Manim animation explaining the current
evidence for an early MoE-related stasis signal.

## Files

- `plan.md`: narrative, evidence boundary, and review checklist.
- `script.py`: all three independently renderable Manim scenes.
- `concat.txt`: production clips in narrative order.
- `final.mp4`: stitched 1080p result after rendering.
- `preview.gif`: compact browser-friendly preview.
- `media/videos/script/1080p60/*.srt`: generated English subtitle tracks.

## Data sources

The script reads existing repository artifacts at render time:

- `rerun-2026-08-27/replay.json`
- `rerun-2026-08-27/failure_modes.json`
- `VLA_MUI_HUB/moe-token-dynamics/results/early_seedblock_summary.json`

The expert lattices use episode 22 (success) and episode 5 (partial/stasis),
HB layer 15, token 5, denoise step 9. The state-space paths in Scene 1 are an
explicitly labeled conceptual diagram, not measured physical x/y coordinates.

## Render

```bash
/home/jovyan/.cache/manim-video-conda/bin/manim -ql script.py \
  Scene1_StateBranches Scene2_CurrentChunkLens Scene3_EvidenceCurve

/home/jovyan/.cache/manim-video-conda/bin/manim -qh script.py \
  Scene1_StateBranches Scene2_CurrentChunkLens Scene3_EvidenceCurve

ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```
