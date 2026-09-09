---
layout: ../../layouts/Post.astro
title: The state the tokens never showed
date: 2026-08-30
blurb: Something first seen in language models — a neuron's internal state holds information the tokens never show — carried over to a cooking robot, and grew a full sequel. The failure becomes visible, and rescuable.
figure: himoe_fig_rescue.png
figureAlt: >-
  Routing-change ratio of one robot rollout. A white line carries the shared history;
  a red ring marks the alarm at control step 32. After it the orange line, continuing
  on the original sampling noise, sinks and fails at the 52-step horizon, while the
  blue line, resampled at the alarm from the same saved state, climbs back above the
  criterion and finishes the task at step 40. A dashed line marks the threshold 0.95.
figureCaption: >-
  The whole story in one plot: a rollout runs online, the routing alarm fires at step
  32, and the run forks right there. Original noise — 52 steps, failure. One fresh
  noise stream from the same state — done in 8. All three curves are measured.
  <span class="figtag" style="margin-left:.6em">HiMoE-VLA, LIBERO</span>
---

This story starts with a different kind of model.

On reasoning LLMs we built a thing called **NAD** (Neuron Activation Distribution):
while the model answers, record the activation distribution of tens of thousands of
neurons — never reading what it wrote, only these internal states. The result: among
many sampled answers to the same question, **the neuron activity alone picks out the one
more likely to be right**. Confidence and hesitation are not written into the tokens,
but the internal state carries them.

It left one question open: is this a quirk of language models, or a property of large
models in general?

Carry the same idea into embodied AI: would the inside of a VLA model hold a similar
signal? We took a cooking robot to find out. Its "words" are not text but twenty joint
actions per second; it too can "think wrong" — not a wrong answer, but an arm frozen
mid-air, or shuttling between two pots until time runs out. The two clips
below start from a **bit-identical simulator state**, run the same Vision-Language-Action
model, and differ in exactly one thing: the noise stream used to sample actions. One
puts both moka pots on the stove in 19 seconds. The other struggles for the full 26,
times out, and fails.

<div class="vidpair" style="position:relative;display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:1.4rem 0">
<figure style="margin:0"><video src="/assets/himoe_success.mp4" muted playsinline preload="metadata" style="width:100%;border-radius:8px"></video><figcaption><b>Success</b> — 38 control steps, both pots placed.</figcaption></figure>
<figure style="margin:0"><video src="/assets/himoe_failure.mp4" muted playsinline preload="metadata" style="width:100%;border-radius:8px"></video><figcaption><b>Failure</b> — all 52 steps spent; retry after retry until the horizon.</figcaption></figure>
</div>

Whatever separates these two runs never appears in any token. The action stream keeps
flowing, the trajectory keeps extending, every frame looks like work being done. To tell
them apart *while they are still running*, the output is the wrong place to look.

## Part one — what a trap looks like in the physical world

First, the failure itself. Sample 32 rollouts from one initial state — again, only the
noise differs — and 14 succeed while 18 fall into a trap, on a 52-step budget.

We call this whole family of failures a **trap**: the rollout keeps running, but its
progress is already dead. It wears two main faces — an arm hanging mid-air, nearly
motionless, consecutive frames identical; or an arm shuttling between the two pots,
returning to the same spot again and again, plenty of path and zero progress.
**Two looks, one condition: caught in a trap.**

<figure><img src="/assets/himoe_fig_trap.png" alt="Distance-to-goal curves for 32 rollouts from one initial state: 14 blue lines descend through the 5-centimetre goal radius; the 18 trapped runs' orange lines flatten halfway down for all 52 steps." loading="lazy"><figcaption>The physical definition of a trap, in one look: 32 same-state rollouts drawn as distance to the goal pose. 14 blue lines descend through 5 cm; the 18 trapped runs' orange lines flatten halfway down — the robot keeps moving but never gets closer. Simulator poses only; no MoE involved.</figcaption></figure>

<figure><img src="/assets/himoe_fig_heat.png" alt="Four heatmaps: the top shallow-layer row flickers on both sides; in the deep-layer row the success keeps flickering while the trap develops horizontal stripes after its onset." loading="lazy"><figcaption>What the router is choosing, moment by moment. Each cell counts how many of the layer's 10 action tokens picked that expert (top-4/32); top row shallow L4, bottom row deep L14; left a success, right a trap. Watch the bottom-right: after the onset (red dashes) it develops horizontal stripes — the same few experts picked step after step. The success keeps flickering.</figcaption></figure>

The rest is a grab-bag of *still moving, but doing the wrong thing*: pots lifted and
dropped, placed and then dragged away (all 32 trajectories,
[frame by frame against the deep-layer routing](/himoe-trap-animation.html)).

From the token side, three limits you cannot get around:

- **The action stream won't tell you.** A stuck model still emits a full, well-formed
  action chunk every step — ten numbers, right format, normal magnitudes. No error, no
  silence. The failure happens in the form of perfectly normal output.
- **Length has two readings.** A long trajectory might mean the problem is hard, or
  that the run is already stuck. The two call for opposite responses — wait it out
  versus cut it off — but the trace shows one symptom: long.
- **The verdict comes at the finish line.** The success predicate stamps only when the
  episode ends. By the time failure is "confirmed", the 26 seconds are spent.

## Part two — the neurons knew all along

This VLA is a Mixture-of-Experts model: inside certain layers, a small group of neurons
acts as a dispatcher, scoring 32 experts at every control step and lighting up the top
4 to handle that step. Which four are lit — call that roster the **pattern**.

While the arm is working, the pattern reshuffles step after step. The moment the arm
stalls, the pattern freezes: the same four pathways stay lit until the clock runs out.
**The trap is written right there, inside the neurons** — while the action stream shows
nothing at all.

So the detector can afford to be one sentence long: count how much the pattern changes
per step, divide by this rollout's **own opening rate** to get a dimensionless ratio
`r(t)`, and fire when it spends **three consecutive steps below θ = 0.95**. No
learning, no calibration, four constants, all frozen in advance.

<figure><img src="/assets/himoe_fig_fan.png" alt="A fan of 352 routing-ratio curves, blue for success and orange for failure; a dot marks each branch's alarm step, red dots crowding the failure side." loading="lazy"><figcaption>r(t) for 352 branches (22 saved states × 16 noise streams). Orange = failure (235), blue = success (117); each dot is that branch's alarm. Alarms land at steps 18–36, median 28 — just past halfway.</figcaption></figure>

On that corpus the rule calls **80.1%** of rollouts correctly against a 66.8%
majority-guess baseline. On a second, independently collected corpus of 512 rollouts it
calls **82.0%** against 57.8% — *with the same threshold*. Because the ratio divides by
the rollout's own opening, θ is a dimensionless number, and the false-alarm rates the
two corpora produce at any given θ agree within two points, bracket by bracket.

<figure><img src="/assets/himoe_fig_transfer.png" alt="False-alarm rate versus one shared threshold; the two corpora's curves nearly coincide across five brackets." loading="lazy"><figcaption>One threshold, two corpora: the false-alarm rates line up (gaps 0–2 points). Continuous probability distances fail this test; the counting ratio's discrete grid pins the scale down.</figcaption></figure>

The false alarms deserve their own sentence. Of 203 alarms, 19 land on runs that
ultimately succeed — and those 19 are the *slow* successes, median length 48 steps
against 39 overall. They really did stall, then walked out of it. So the alarm's
semantics are **"this run is stuck right now"**, not "this run is doomed". That
distinction pays off below.

## Part three — an honest word about why it works

The first time the deep-layer pattern froze on screen, it looked like magic — until the
control experiment made it soberingly plain:

> Robot stuck → observation unchanged → router input unchanged → of course the experts
> don't change. The routing's stickiness correlates with the input's own stability at
> r = +0.93; regress the input out, and the routing metric collapses to chance.

The neurons are not an oracle. They are a **free state sensor**: they faithfully write
"the physical world has stopped moving" into a log the model produces anyway. Mundane
is precisely why it is dependable — nothing mystical, so nothing fails mystically. And
*free* is meant literally: expert selection is a by-product of the MoE forward pass, so
reading it adds no inference and no probe.

One counter-intuitive detail. *Which* neurons are lit carries almost no outcome
information — the static roster identifies the initial scene at nearly 100% accuracy
yet is nearly useless about success once scenes are held out. **Identity is a
fingerprint of the scene; change is the state.** The ratio above is invariant to
renumbering the experts, so it takes only the latter.

## One more thinking — can it be rescued, live?

An alarm that only keeps score is a dashboard nobody reads. So the last experiment used
it as a trigger: a fresh rollout runs online — threshold and all constants frozen
beforehand — and the alarm fires at step 32. **At that exact step** the full simulator
state is saved, and eight fresh noise streams are run out of it.

<div class="vidpair" style="position:relative;display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:1.4rem 0">
<figure style="margin:0"><video src="/assets/himoe_rescue_trunk.mp4" muted playsinline preload="metadata" style="width:100%;border-radius:8px"></video><figcaption><b>Original noise, continued</b> — 26.0 s, 52 steps, fails at the horizon.</figcaption></figure>
<figure style="margin:0"><video src="/assets/himoe_rescue_ok.mp4" muted playsinline preload="metadata" style="width:100%;border-radius:8px"></video><figcaption><b>Resampled at the alarm</b> — same state, one new noise stream, done in 3.9 s.</figcaption></figure>
</div>

Three of the eight resamples succeed. A rollout headed for certain failure swapped its
noise at the step the alarm pointed to, and finished in eight. **Detection closed into
intervention.**

The limits were measured the same way, and none are hidden:

- **Not every state is rescuable.** Another initial state, same procedure: 0 of 8.
  Nothing in the alarm reading tells the two apart.
- **The moment is not special.** A control arm forked at a random earlier step also
  rescued 3 of 8. The alarm's value is knowing *that* it is time to act — not having
  found the perfect instant.
- **The pattern cannot pick the winner.** Among the eight candidates, the one with the
  largest routing change failed; the rescue happened to be the smallest (n = 1, and the
  direction ran against intuition).
- **No early warning.** The median alarm lands at 54% of the horizon; before that,
  single-step readings are coin flips. It reads *already stuck*, it does not prophesy
  *about to be*.

<figure><img src="/assets/himoe_fig_candidates.png" alt="Bar chart of routing change for the eight candidates sampled at the alarm: all above the stuck trunk, yet the largest change failed and the executed rescue was the smallest." loading="lazy"><figcaption>The eight candidates at the alarm step: any fresh noise moves the pattern more than the stuck trunk (0.42) — corroborating the mechanism — but magnitude does not pick winners.</figcaption></figure>

## The takeaway

A trajectory is what the model did. The pattern is what state it was in. The first gets
stamped only at the finish line; the second says its piece halfway through — in a log
the model writes at every step, that nobody had read. The gauge was on the whole time.
We just finally looked at it.

*A 50-second animated version of this post lives at
[/himoe-pattern-film.html](/himoe-pattern-film.html). 中文版:
[/blog/the-state-the-tokens-never-showed-zh/](../the-state-the-tokens-never-showed-zh/).
For the neighbouring observation on language models — that routing carries what the
answer text does not — see
[Why the answer hides in the routes](../why-the-answer-hides-in-the-routes/).*

<style>
.vidpair{position:relative}
.vidpair video{pointer-events:none}
.pairplay{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:3;
  width:64px;height:64px;border-radius:50%;border:1px solid rgba(255,255,255,.35);
  background:rgba(10,14,21,.55);color:#fff;cursor:pointer;font-size:22px;
  display:grid;place-items:center;transition:opacity .25s}
.vidpair.playing .pairplay{opacity:0}
.vidpair.playing:hover .pairplay{opacity:1}
</style>
<script>
/* one click plays both clips of a pair in sync; native controls are hidden so
   the two cannot be scrubbed apart */
document.querySelectorAll('.vidpair').forEach(function(pair){
  var vids=[].slice.call(pair.querySelectorAll('video'));
  vids.forEach(function(v){v.removeAttribute('controls');});
  var btn=document.createElement('button');
  btn.className='pairplay';btn.textContent='▶';btn.setAttribute('aria-label','play both');
  pair.appendChild(btn);
  var playing=false,ended=0;
  function sync(){
    if(playing){vids.forEach(function(v){v.pause();});playing=false;btn.textContent='▶';}
    else{vids.forEach(function(v){v.currentTime=0;v.play();});playing=true;btn.textContent='❚❚';}
    pair.classList.toggle('playing',playing);
  }
  btn.addEventListener('click',sync);
  pair.addEventListener('click',function(e){if(e.target!==btn)sync();});
  vids.forEach(function(v){v.addEventListener('ended',function(){
    ended++;if(ended>=vids.length){ended=0;playing=false;btn.textContent='↺';
      pair.classList.remove('playing');}});});
});
</script>
