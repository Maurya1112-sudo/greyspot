#!/usr/bin/env bash
# Stop the work queue and VERIFY it stopped (rule R16).
PIDFILE=/tmp/greyspot_queue.pids
[ -f "$PIDFILE" ] || { echo "no queue pidfile"; exit 0; }
while read -r kind pid rest; do
  [ -n "${pid:-}" ] || continue
  kill -9 "$pid" 2>/dev/null
  powershell -NoProfile -Command "taskkill /PID $pid /T /F" >/dev/null 2>&1
  echo "killed $kind $pid $rest"
done < "$PIDFILE"
sleep 2
echo "--- verification ---"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { '{0}|{1}' -f \$_.ProcessId, \$_.CommandLine.Substring(0,[Math]::Min(70,\$_.CommandLine.Length)) }"
echo "(empty above = nothing left running)"
rm -f "$PIDFILE"
