#!/usr/bin/env bash
# Stops the three processes run_all.sh starts, by the PIDs it saved to
# .run_all.pids (Linux has no window titles to match on, unlike
# stop_all.bat). Kills each PID's whole process group so a child
# python.exe under the shell wrapper (if any) dies too.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .run_all.pids ]; then
  echo "no .run_all.pids -- nothing to stop (was run_all.sh run from here?)"
  exit 0
fi

while read -r pid; do
  [ -z "$pid" ] && continue
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "stopped $pid"
  fi
done < .run_all.pids

rm -f .run_all.pids
