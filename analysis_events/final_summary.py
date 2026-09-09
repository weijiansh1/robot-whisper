"""Assemble the final markdown summary from the persisted result tables."""
import glob
import shutil

import numpy as np
import pandas as pd

OUT = "/home/jovyan/work/himoe-vla/analysis_events"
ORDER = ["phase", "tier1", "routing", "routing_shufchunk", "tier1+routing",
         "tier1+routing_perm", "tier2", "tier2+routing"]

L = []
A = L.append

A("# Does MoE routing signal discrete physical events?")
A("")
A("Corpus A = rolling-star branch capture (352 branches, 16158 chunks, 4 initial")
A("states). Corpus B = SCENE8 replication (512 episodes, 22883 chunks, 16 initial")
A("states). Task in both: KITCHEN_SCENE8 put both moka pots on the stove.")
A("")
A("## 0. Data inventory (sim_layout.json + control_sim_state, verified numerically)")
A("")
A("`sim_state` is 47-dim = [time(1)] + qpos(24) + qvel(22). Field map:")
A("")
A("| index | contents | is_robot |")
A("|---|---|---|")
A("| 0 | sim time | - |")
A("| 1:8 | **robot0_joint1..7 - the 7 arm joint angles** | yes |")
A("| 8:10 | gripper0_finger_joint1,2 | yes |")
A("| 10:13 / 13:17 | moka_pot_1 free-joint position / quaternion | no |")
A("| 17:20 / 20:24 | moka_pot_2 free-joint position / quaternion | no |")
A("| 24 | flat_stove_1_button (1 dof) | no |")
A("| 25:32 | arm qvel (7) | yes |")
A("| 32:34 | finger qvel (2) | yes |")
A("| 34:40 / 40:46 | moka_pot_1 / moka_pot_2 qvel (6 each) | no |")
A("| 46 | stove button qvel | no |")
A("")
A("The 7 arm joint angles ARE present, so the joint-configuration ('twisted joint')")
A("target is possible and was run. They are absent from the 8-dim proprio the")
A("policy actually receives ([eef pos 3, axis-angle 3, finger joints 2]) - that")
A("asymmetry is used below as the one provable blindness case.")
A("The CALVIN cache under VLA_MUI_HUB/cache/HiMoE-VLA/calvin_d_d/ is empty")
A("(.gitkeep only), so no CALVIN fallback was needed or possible.")
A("")

A("## 1. The structural fact that kills the top-priority target")
A("")
A("Contactless object motion does not occur in this task.")
A("")
A("| corpus | chunk transitions | pot moved >1cm | of those, eef never within 15 cm | median eef-object dist during motion |")
A("|---|---|---|---|---|")
A("| A pot_1 | 15806 | 1044 | 4 (0.38%) | 0.068 m |")
A("| A pot_2 | 15806 | 1866 | 19 (1.02%) | 0.065 m |")
A("| B pot_1 | 22371 | 2537 | 1 (0.04%) | 0.069 m |")
A("| B pot_2 | 22371 | 3566 | 0 (0.00%) | 0.066 m |")
A("")
A("9013 object-motion events across both corpora; 24 (0.27%) without the")
A("end-effector ever coming within 15 cm. A moka pot only moves when it is held.")
A("Consequently every 'gripper open and far away' event has <=5 positives and is")
A("dropped. This is a property of the task, not of the sample size.")
A("")

A("## 2. Sentinels (run before any headline)")
A("")
A("**Shuffle sentinel** - routing rows permuted within each branch, labels fixed.")
A("Across all 74 event cells in both corpora the acausal control lands at")
A("0.410-0.620 (median 0.516). Chunk-level events are NOT recoverable from")
A("branch identity, unlike the branch-level tasks where an earlier agent found")
A("0.93-0.96. The chunk-level design is clean.")
A("")
A("**Clock sentinel** - t, T, t/T, T-t alone. This is the dominant already-known")
A("baseline: phase reaches 0.998-1.000 on 'which pot moves' in corpus B, and")
A("0.63-0.99 on most other events. Any raw AUC must be read against it.")
A("")

for tag, name in [("A2", "A"), ("B2", "B")]:
    f = f"{OUT}/auc_{tag}.csv"
    df = pd.read_csv(f)
    df["ev"] = df.event + "_m" + df.horizon.astype(str)
    p = df.pivot_table(index=["ev", "n_pos", "n_neg"], columns="feature_block",
                       values="value")
    cols = [c for c in ORDER if c in p.columns]
    p = p[cols]
    p["gap_t2_minus_t1"] = p["tier2"] - p["tier1"]
    p["routing_increment"] = p["tier1+routing"] - p["tier1"]
    A(f"## 3.{'AB'.index(name)+1} Corpus {name}: per-event grouped AUC "
      f"(sorted by tier-1 blindness gap)")
    A("")
    A("```")
    A(p.sort_values("gap_t2_minus_t1", ascending=False).round(3).to_string())
    A("```")
    A("")

for tag, name in [("A2", "A"), ("B2", "B")]:
    inc = pd.read_csv(f"{OUT}/increments_{tag}.csv")
    inc["pair"] = inc.base + " -> " + inc.combined
    for pr, lab in [("tier1 -> tier1+routing", "routing increment over the deployable control"),
                    ("tier1+routing_perm -> tier1+routing", "vs capacity-matched permuted-routing null")]:
        s = inc[inc.pair == pr][["event", "n_pos", "delta", "lo", "hi", "sig"]]
        A(f"## 4. Corpus {name}: {lab} (grouped bootstrap 95% CI)")
        A("")
        A("```")
        A(s.sort_values("delta", ascending=False).round(4).to_string(index=False))
        A("```")
        A("")

A("## 5. Mirror-selection symmetry check")
A("")
A("Restricting to rows where tier-1's out-of-fold probability sits near the base")
A("rate makes routing look far better than tier-1. The mirror selection (restrict")
A("on routing's own score, read tier-1) produces an equal or larger advantage for")
A("tier-1. The effect is symmetric, i.e. a range-restriction artifact of")
A("conditioning on a model's own score, not evidence of routing-specific")
A("information. Both directions are in stratified_A2.csv / stratified_B2.csv.")
A("")

A("## 6. Cells dropped for <30 positives (named, per the protocol)")
A("")
A("| event family | corpus A positives (m=1/2/4) | corpus B | verdict |")
A("|---|---|---|---|")
A("| leavegoal1 (pot_1 leaves goal) | 0 / 0 / 0 (pop 214/90/44) | 4 / 9 / 14 | dropped both |")
A("| leavegoal2 (pot_2 leaves goal) | 4 / 7 / 15 (pop 6058) | 0 / 0 / 0 (pop 13126) | dropped both |")
A("| dispfar1/2 (motion, no contact anywhere in window) | 1 / 1 / 1 and 5 / 5 / 5 | 1 / 1 / 0 and 0 / 2 / 4 | dropped both |")
A("| dispnc1/2 m=1,2 (motion, gripper open, eef >15 cm) | 1 / 2 and 5 / 5 | 1 / 15 and 0 / 2 | dropped; only m=4 survives |")
A("| goalregnc1/2 (goal regression, no contact) | 0-2 | 0 | dropped both |")
A("| whichpotnc (which pot, no contact) | 3 / 4 / 38 | 0 / 0 / 18 | dropped both |")
A("| jlim (within 10% of a joint limit) | 0 / 0 / 0 | 59 / 89 / 149 | dropped in A, kept in B |")
A("| twist (large dq, small d-eef) | 133 / 43 / 78 | 24 / 5 / 74 | kept in A; only m=4 in B |")
A("| twistq (percentile version) | 30 / 5 / 4 | 3 / 1 / 0 | dropped both |")
A("| goalreg1_m4 | 8 | 23 | dropped both |")
A("| dropfail2_m4 | 28 | 3 | dropped both |")
A("| tilt1_m1 (A) | 33 | 59 | kept, flagged as marginal |")
A("")

txt = "\n".join(L)
open(f"{OUT}/moe_events.md", "w").write(txt)
open("/tmp/moe_events.md", "w").write(txt)

frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/auc_*.csv"))
          if "B2main" not in f and "B2sup" not in f]
df = pd.concat(frames, ignore_index=True)
df["corpus"] = df.corpus.str.rstrip("2")
inc = pd.concat([pd.read_csv(f"{OUT}/increments_{t}.csv") for t in ("A2", "B2")],
                ignore_index=True)
inc["corpus"] = inc.corpus.str.rstrip("2")
long = pd.concat([
    df[["corpus", "event", "horizon", "feature_block", "metric", "value",
        "n_pos", "n_neg"]],
    inc.assign(event_=inc.event.str.split("_m").str[0],
               horizon=inc.event.str.split("_m").str[1].astype(int),
               feature_block=inc.base + "->" + inc.combined,
               metric="delta_auc_grouped_boot95",
               value=inc.delta, n_neg=np.nan)
    .rename(columns={"event_": "ev2"})
    .assign(event=lambda x: x.ev2)[["corpus", "event", "horizon", "feature_block",
                                    "metric", "value", "n_pos", "n_neg"]],
], ignore_index=True)
long.to_csv(f"{OUT}/moe_events.csv", index=False)
shutil.copy(f"{OUT}/moe_events.csv", "/tmp/moe_events.csv")
print("rows:", len(long), "| md:", len(txt), "chars")
