#!/bin/bash
set -uo pipefail
REPO=/home/jovyan/work/himoe-vla; M=/home/jovyan/work/robot-whisper-preparation-2026-09-09/manifest; BR=backup/2026-09-09
export GIT_INDEX_FILE=$M/backup.index
cd $REPO; rm -f "$GIT_INDEX_FILE"; git read-tree HEAD; parent=$(git rev-parse HEAD)
commit_push() { tree=$(git write-tree); c=$(git commit-tree "$tree" -p "$parent" -m "$1"); git update-ref refs/heads/$BR "$c"; parent=$c; echo "[$(date +%T)] commit $c: $1"
  for i in 1 2 3; do git push origin $BR:$BR 2>&1 | tr '\r' '\n' | tail -3 && [ "${PIPESTATUS[0]}" = 0 ] && { echo "[$(date +%T)] pushed"; return 0; }; echo "push failed, retry $i"; sleep 30; done; return 1; }
grep -zvE '^(moe-trap-control|safe&vlaconf|himoe-route-capture)/' $M/git-include.lst0 | git update-index --add -z --stdin
printf '%s\0' .gitignore MoE-grammar/moe_grammar/run_channel_comparison_audit.py MoE-grammar/results-channel-comparison/summary.json analysis_moe_execution_signals/README.md | git update-index --add -z --stdin
zstd -19 -T4 -q -f $M/manifest-raw-data.tsv -o $M/manifest-raw-data.tsv.zst
for f in README.md manifest-raw-data.tsv.zst git-exclude.tsv nested-repos.tsv skipped-ignored.lst pack-order.lst build_manifest.py select_git_set.py pack_upload.sh part_filter.sh build_backup_branch.sh; do blob=$(git hash-object -w $M/$f); git update-index --add --cacheinfo 100644,$blob,BACKUP-2026-09-09/$f; done
commit_push "backup 2026-09-09 (1/4): untracked sources, reports, small results, manifests" || exit 1
grep -zE '^himoe-route-capture/' $M/git-include.lst0 | git update-index --add -z --stdin; commit_push "backup 2026-09-09 (2/4): himoe-route-capture" || exit 1
grep -zE '^safe&vlaconf/' $M/git-include.lst0 | git update-index --add -z --stdin; commit_push "backup 2026-09-09 (3/4): safe&vlaconf" || exit 1
grep -zE '^moe-trap-control/' $M/git-include.lst0 | git update-index --add -z --stdin; commit_push "backup 2026-09-09 (4/4): moe-trap-control" || exit 1
echo "[$(date +%T)] BRANCH DONE $parent"
