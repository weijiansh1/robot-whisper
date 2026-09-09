"""Write the long-format result table and the running markdown summary,
to /tmp and mirrored into analysis_events/."""
import os
import glob
import shutil

import numpy as np
import pandas as pd

OUT = "/home/jovyan/work/himoe-vla/analysis_events"
ORDER = ["phase", "tier1", "routing", "routing_shufchunk", "tier1+routing",
         "tier1+routing_perm", "tier2", "tier2+routing"]
NICE = {"A2": "A (rolling-star branches, 4 init states)",
        "B2": "B (SCENE8 replication, 16 init states)"}


def main():
    frames = []
    for f in sorted(glob.glob(f"{OUT}/auc_*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df["corpus"] = df.corpus.map(lambda c: c.rstrip("2"))
    df.to_csv("/tmp/moe_events.csv", index=False)
    shutil.copy("/tmp/moe_events.csv", f"{OUT}/moe_events.csv")

    lines = ["# MoE routing vs discrete physical events - running results", ""]
    lines.append("Metric: within-CV-group AUC, pooled by pair count. "
                 "Groups: worker/init_state (leave-one-group-out). "
                 "Every block: standardise -> PCA (train fold only) -> "
                 "HistGradientBoosting, identical hyper-parameters.")
    lines.append("")
    for tag in ["A", "B"]:
        sub = df[df.corpus == tag]
        if not len(sub):
            continue
        lines.append(f"## Corpus {NICE.get(tag + '2', tag)}")
        lines.append("")
        p = sub.pivot_table(index=["event", "horizon", "n_pos", "n_neg"],
                            columns="feature_block", values="value")
        cols = [c for c in ORDER if c in p.columns]
        lines.append(p[cols].round(3).to_string())
        lines.append("")
    for tag in ["A2", "B2"]:
        f = f"{OUT}/increments_{tag}.csv"
        if os.path.exists(f):
            inc = pd.read_csv(f)
            lines.append(f"## Increments with grouped bootstrap 95% CI - corpus {tag[0]}")
            lines.append("")
            lines.append(inc.round(4).to_string(index=False))
            lines.append("")
    for tag in ["A2", "B2"]:
        f = f"{OUT}/stratified_{tag}.csv"
        if os.path.exists(f) and os.path.getsize(f) > 60:
            lines.append(f"## Routing AUC where tier-1 sits near chance - corpus {tag[0]}")
            lines.append("")
            lines.append(pd.read_csv(f).round(4).to_string(index=False))
            lines.append("")
    txt = "\n".join(lines)
    open("/tmp/moe_events.md", "w").write(txt)
    open(f"{OUT}/moe_events.md", "w").write(txt)
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
