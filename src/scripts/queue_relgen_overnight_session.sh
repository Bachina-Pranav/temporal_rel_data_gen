#!/usr/bin/env bash

# Preflight first so an unattended launch cannot degrade into eight hours of
# heartbeat because scientific entry points are absent.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

CONFIG="${RELGEN_OVERNIGHT_CONFIG:-configs/experiments/relgen_overnight_8h_session.yaml}"
OUT="${RELGEN_OVERNIGHT_OUTPUT:-outputs/overnight_8h_session}"
DEVICE="${RELGEN_OVERNIGHT_DEVICE:-cuda}"

mkdir -p "$OUT/logs"

python -u src/scripts/run_relgen_overnight_session.py \
  --config "$CONFIG" \
  --stage preflight \
  --device "$DEVICE" \
  --output-dir "$OUT" \
  > "$OUT/logs/preflight.log" 2>&1
PREFLIGHT_STATUS=$?

if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
  echo "Overnight launch blocked by preflight (exit $PREFLIGHT_STATUS)."
  echo "Inspect $OUT/preflight.md and $OUT/logs/preflight.log"
  exit "$PREFLIGHT_STATUS"
fi

nohup python -u src/scripts/run_relgen_overnight_session.py \
  --config "$CONFIG" \
  --stage run \
  --device "$DEVICE" \
  --output-dir "$OUT" \
  > "$OUT/logs/watchdog.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$OUT/session.pid"
echo "RelGen overnight watchdog started with PID $PID"
echo "Log: $OUT/logs/watchdog.log"
echo "Status: $OUT/job_status.json"
