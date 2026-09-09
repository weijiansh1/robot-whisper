#!/bin/bash
set -uo pipefail
M=/home/jovyan/work/robot-whisper-preparation-2026-09-09/manifest
f=$2; path=$M/tmp/$f
cat > "$path"
sha=$(sha256sum "$path" | cut -d' ' -f1); bytes=$(stat -c %s "$path")
ok=0; for i in 1 2 3 4; do if gh release upload backup-2026-09-09 "$path" --clobber >>$M/upload.log 2>&1; then ok=1; break; fi; echo "[$(date +%T)] upload retry $i for $f" >> $M/upload.log; sleep 60; done
st=$([ $ok = 1 ] && echo uploaded || echo FAILED)
printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$bytes" "$sha" "$st" "$(date -u +%FT%TZ)" >> $M/parts-manifest.tsv
rm -f "$path"; [ $ok = 1 ]
