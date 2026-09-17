#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
while ! grep -q "ports:" /tmp/batch-restart5.log 2>/dev/null; do sleep 10; done
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 12 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-13a-idle-alarm-8 --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 3 --seeds 0,1 --strategies idle_hold,idle_open,kick
run exp-13b-idle-alarm-16 --parents controls/parents-topo-all-i.json --branch alarm-16 --control-queries 3 --seeds 0,1 --strategies idle_hold,idle_open
run exp-13c-idle-alarm-8-c6 --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 6 --seeds 0,1 --strategies idle_hold
