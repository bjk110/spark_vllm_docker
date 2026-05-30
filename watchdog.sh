#!/bin/bash
# RAM watchdog: kill vllm-spark containers before host-wide page-cache thrash
# locks sshd. Runs high-priority (nice -20 / ionice rt) so it stays schedulable
# under memory pressure. Deployed for the dsv4-d568-5d64798 safeguarded retry.
THRESH_MB=8000
LOG="/home/bjk110/docker/vllm-spark/watchdog.log"
: > "$LOG"
echo "$(date '+%T') watchdog start thresh=${THRESH_MB}MB host=$(hostname)" >> "$LOG"
while true; do
  avail=$(free -m | awk '/Mem:/{print $7}')
  echo "$(date '+%T') avail=${avail}MB" >> "$LOG"
  if [ "${avail:-999999}" -lt "$THRESH_MB" ]; then
    echo "$(date '+%T') !!! WATCHDOG ABORT avail=${avail}MB -> killing vllm containers" >> "$LOG"
    docker ps -q --filter name=vllm-spark | xargs -r docker kill >> "$LOG" 2>&1
    echo "$(date '+%T') killed. watchdog exit." >> "$LOG"
    break
  fi
  sleep 1
done
