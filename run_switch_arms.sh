#!/bin/bash
# Three paired arms on libero_spatial t05, to test whether the state-token
# gripper gate at HB layers 2-5 changes the outcome.
#
# The base arm is rerun here rather than reused from right-16x32 on purpose:
# a byte-identical recapture agrees on only 58/64 outcomes because MuJoCo's
# solver is not reproducible in contact, so the control has to share this run's
# noise floor.  Scene ids and flow-noise seeds match right-16x32 exactly, so the
# comparison is paired per (initial state, noise draw).
set -u
cd /home/jovyan/work/himoe-vla/himoe-route-capture

PIN=/home/jovyan/work/himoe-vla/pin_libero_spatial_pick_up_the_black_bowl_on_th.json
SCENES=0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49
A=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a      # 2g.35gb
B=MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3      # 1g.35gb
LOG=/home/jovyan/work/himoe-vla/logs-switch
mkdir -p "$LOG"

arm () {  # run-id gpu port [pin-regime]
  local id=$1 gpu=$2 port=$3 regime=${4:-}
  local extra=()
  [ -n "$regime" ] && extra=(--pin "$PIN" --pin-regime "$regime")
  echo "[$(date +%H:%M:%S)] start $id ${regime:-none}"
  python3 -u run_corpus_capture.py \
    --run-id "$id" --gpu "$gpu" --port "$port" \
    --benchmarks libero_spatial --tasks 5 \
    --scene-ids "$SCENES" --draws 32 \
    "${extra[@]}" > "$LOG/$id.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $id rc=$?"
}

# lane A takes two arms, lane B one; lane B is the smaller slice
( arm pin-base "$A" 8430 ; arm pin-on "$A" 8431 on ) &
( arm pin-off "$B" 8432 off ) &
wait
echo "[$(date +%H:%M:%S)] all arms finished"
