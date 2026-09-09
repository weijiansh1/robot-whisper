# When Does MoE Know the Robot Is Stuck?

## Overview

- **Topic**: Current-chunk HiMoE-VLA signals before and after rollout stasis.
- **Hook**: The visible loop is late. Does one current chunk already contain a warning?
- **Aha moment**: The evidence changes meaning over time: a weak pre-expert hidden clue appears before the earliest measured stasis onset, while the strong later signal behaves primarily as a no-progress detector.
- **Audience**: Robotics and VLA researchers.
- **Length**: About 80 seconds, no narration; generated per-scene English SRT files.
- **Resolution**: 854x480 draft, 1920x1080 final, 16:9.
- **Arc**: Discovery.

## Evidence Boundary

- The abstract state-space paths in Scene 1 are explanatory geometry, not measured robot x/y trajectories.
- Current expert selections in Scene 2 come from `rerun-2026-08-27/replay.json`:
  - episode 22: success;
  - episode 5: partial/stasis failure;
  - HB layer 15, token 5, denoise step 9.
- Each displayed router lattice is one current chunk only. Previous router states are replaced, not accumulated.
- Early held-out AUCs come from `early_seedblock_summary.json`:
  - current pre-expert hidden identity: k7 = 0.476, k12 = 0.624;
  - current routing: k7 = 0.464, k12 = 0.578;
  - hidden k12 family-wise maxT p = 0.019.
- Stasis onset quantiles are min k15, median k19, q90 k30.
- The later curve is explicitly labeled as a different, seven-chunk-history monitor:
  k7/k12/k20/k27/k34 AUC = 0.475/0.540/0.628/0.829/0.925.
- The final statement is association-only: no routed-expert causal claim.

## Color Palette

- Background `#0D1117`: neutral field.
- Success `#69D18B`: successful progress.
- Stasis `#FF6B6B`: no-progress basin.
- Event `#FFB84D`: event-like failure.
- MoE `#58C4DD`: internal computation and measured MoE signal.
- Current chunk `#FFE66D`: the only observation window.
- Structure `#7D8590`: axes, context, inactive experts.

## Scene 1: One Start, Three Futures (~14 s)

**Purpose**: Establish that failure is a trajectory mode, not a single terminal label.

**Layout**: Full-frame abstract state landscape, with no physical-coordinate axes.

### Visual beats

1. A single state node appears with the question: "Same state. Different futures."
2. Three paths grow from the node:
   - success reaches a goal ring;
   - stasis advances, loses speed, and settles into a basin;
   - event failure advances, then breaks away abruptly.
3. Outcome counts appear as a compact legend: 1741 success, 275 stasis, 32 event-like failures.
4. The camera focus moves to the stasis path because that is where the current evidence is strongest.

### Subtitle

"A rollout can fail by stopping, or by a discrete event. The internal signal is not equally strong for both."

## Scene 2: A Lens on One Chunk (~20 s)

**Purpose**: Make the single-current-chunk constraint visually unavoidable.

**Layout**: Success and stasis lanes on the left; one 8x4 expert lattice on the right; ten action cells below.

### Visual beats

1. A yellow lens selects k0. The label states `one chunk = 10 actions`.
2. The lens jumps through k7, k12, k15, and k20.
3. At every jump, the expert lattice is replaced by the selected experts from that current chunk only.
4. At k7, the readout shows `hidden AUC 0.476: no clue`.
5. At k12, it transforms to `hidden AUC 0.624: weak clue`, with `maxT p=0.019`.
6. A red onset marker appears at k15 and a median marker at k19.

### Subtitle

"At k12, before the earliest measured stasis onset, the current pre-expert hidden state contains a weak clue. Current routing does not survive correction."

## Scene 3: Precursor or Detector? (~21 s)

**Purpose**: Separate the early single-chunk result from the later history-based detector.

**Layout**: Evidence plane with chunk index on x and held-out AUC on y.

### Visual beats

1. Plot the two current-only series at k7 and k12: hidden and routing.
2. Reveal the stasis-onset band from k15 to k30, with median k19.
3. Draw the seven-chunk-history monitor through k34 as a dashed cyan curve.
4. Emphasize that the strong rise occurs after stasis formation.
5. Transform the chart into the final three-part statement:
   - `early: weak hidden precursor`;
   - `late: strong no-progress detector`;
   - `not shown: routed-expert cause`.

### Subtitle

"The early clue and the late detector are different claims. The present data do not establish a routed-expert cause."

## Review Checklist

- Every plotted number matches the frozen JSON sources.
- Scene 1 is labeled as abstract state space, never physical x/y.
- Scene 2 never displays accumulated routing history.
- The history curve in Scene 3 is visually dashed and explicitly labeled.
- At most six active visual groups are bright at once.
- All text is at least 18 pt and uses DejaVu Sans Mono.
- Preview frames are checked at the opening, k12 reveal, onset reveal, and final verdict.
