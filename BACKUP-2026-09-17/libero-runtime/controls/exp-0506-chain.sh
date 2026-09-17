#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
while [ ! -f controls/exp-04c-fixed-q24-all/t1 ]; do sleep 20; done
/home/swj/bin/himoe-batch stop all; sleep 10; /home/swj/bin/himoe-batch start-all topo-20260916c 8 9570 32 > /tmp/batch-restart2.log 2>&1
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 16 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-06a-online-diam110-kick --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick --extra-args "--diam-threshold 1.10"
run exp-06b-online-diam115-kick --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick --extra-args "--diam-threshold 1.15"
run exp-06c-online-diam110-or-v82-kick --parents controls/parents-topo-all-i.json --branch online --trigger diam_or_v82 --include-successes --control-queries 1 --seeds 0,1 --strategies kick --extra-args "--diam-threshold 1.10"
run exp-05c-online-v82-kick-retrigger --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 1 --seeds 0,1 --strategies kick --extra-args "--retrigger-cooldown 6"
run exp-06d-online-diam110-kick-retrigger --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick --extra-args "--diam-threshold 1.10 --retrigger-cooldown 8"
run exp-05a-online-routestep-kick --parents controls/parents-topo-all-i.json --branch online --trigger route_step --include-successes --control-queries 1 --seeds 0,1 --strategies kick
