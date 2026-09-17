#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
while [ ! -f controls/exp-02b-alarm-n25-c3/t1 ]; do sleep 15; done
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 16 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-03a-kickvariants-alarm-8-c3 --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 3 --seeds 0,1 --strategies kick_open,kick_lift_closed,kick_high,kick_retreat,kick_wiggle,kick_random,kick_then_motion,kick_then_random
run exp-03b-kick-duration-alarm-8 --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 1 --seeds 0,1 --strategies kick
run exp-03b2-kick-duration6-alarm-8 --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 6 --seeds 0,1 --strategies kick
run exp-03c-kick-timing-alarm-16 --parents controls/parents-topo-all-i.json --branch alarm-16 --control-queries 3 --seeds 0,1 --strategies native,kick
run exp-03c2-kick-timing-alarm-4 --parents controls/parents-topo-all-i.json --branch alarm-4 --control-queries 3 --seeds 0,1 --strategies native,kick
run exp-03d-online-v82-kick --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 --strategies native,kick
run exp-03e-proswap-alarm-8 --parents controls/parents-proswap.json --benchmarks proswap --branch alarm-8 --control-queries 3 --seeds 0,1 --strategies native,random,kick,kick_then_motion
