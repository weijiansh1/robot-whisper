#!/bin/bash
set -uo pipefail
REPO=/home/jovyan/work/himoe-vla; M=/home/jovyan/work/robot-whisper-preparation-2026-09-09/manifest; TAG=backup-2026-09-09
mkdir -p $M/tmp; touch $M/groups-done.lst $M/parts-manifest.tsv; cd $REPO
gh release view $TAG >/dev/null 2>&1 || gh release create $TAG --target 8e37b8db1a35115a22c0787083b11b2485f47c15 --prerelease --latest=false --title "Emergency backup 2026-09-09: raw data packs" --notes "Raw experiment data packed per directory as <group>.tar.zst.partNNN (1 GiB parts). Restore: cat <group>.tar.zst.part* | zstd -d | tar -x -C himoe-vla. Per-file SHA-256 is in branch backup/2026-09-09 under BACKUP-2026-09-09/. Not a curated release." || { echo "release create failed"; exit 1; }
while IFS= read -r name; do
  [ -z "$name" ] && continue
  grep -qxF -- "$name" $M/groups-done.lst && continue
  safe=$(printf '%s' "$name" | sed 's#[/&]#_#g; s#^\.#_#')
  awk -F'\t' -v g="$name" '$1==g{print $3}' $M/pack-groups.tsv | tr '\n' '\0' > $M/tmp/list.lst0
  n=$(tr -cd '\0' < $M/tmp/list.lst0 | wc -c)
  echo "[$(date +%T)] START $name ($n files) -> $safe.tar.zst.partNNN"
  tar --null -T $M/tmp/list.lst0 --warning=no-file-changed -cf - 2>>$M/tar.err | zstd -T4 -3 -q | split -b 1024m -d -a 3 --filter="$M/part_filter.sh $safe \$FILE" - "$safe.tar.zst.part"
  rc=(${PIPESTATUS[@]})
  if [ "${rc[2]}" = 0 ] && [ "${rc[1]}" = 0 ] && [ "${rc[0]}" -le 1 ]; then printf '%s\n' "$name" >> $M/groups-done.lst; echo "[$(date +%T)] DONE  $name (tar rc=${rc[0]})"; else echo "[$(date +%T)] FAILED $name rc=${rc[*]}"; fi
done < $M/pack-order.lst
echo "[$(date +%T)] ALL GROUPS PROCESSED"
