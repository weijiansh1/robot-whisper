#!/bin/bash
# Pause (SIGSTOP) biggest loading servers when MemAvailable dips; resume on recovery.
STOPPED=""
while :; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt 250 ]; then
        pid=$(ps -wweo pid,rss,cmd | grep "[s]erve_with_recorder" | sort -k2 -rn | head -1 | awk '{print $1}')
        if [ -n "$pid" ] && ! echo "$STOPPED" | grep -qw "$pid"; then
            kill -STOP "$pid" && STOPPED="$STOPPED $pid" && echo "[$(date +%H:%M:%S)] avail=${avail}G STOP $pid"
        fi
    elif [ "$avail" -gt 400 ] && [ -n "${STOPPED## }" ]; then
        pid=$(echo $STOPPED | awk '{print $1}')
        kill -CONT "$pid" 2>/dev/null && echo "[$(date +%H:%M:%S)] avail=${avail}G CONT $pid"
        STOPPED=$(echo $STOPPED | sed "s/\b$pid\b//")
    fi
    # exit once all servers are past loading (ports open = ready lines in logs)
    ready=$(grep -h "server ready" logs/v*.log 2>/dev/null | wc -l)
    [ "$ready" -ge 39 ] && { echo "all ready"; for p in $STOPPED; do kill -CONT $p 2>/dev/null; done; exit 0; }
    sleep 20
done
