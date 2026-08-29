#!/usr/bin/env bash
set -euo pipefail

PID_FILE="outputs/qwen_text_decoder_06b/logs/experiment.pid"
LOG_FILE="outputs/qwen_text_decoder_06b/followup/queued_followup.log"
DEVICE="cuda"

while (($#)); do
  case "$1" in
    --pid-file) PID_FILE="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] Follow-up queue started"
echo "PID file: $PID_FILE"

if [[ ! -s "$PID_FILE" ]]; then
  echo "Missing or empty PID file: $PID_FILE" >&2
  exit 1
fi

WAIT_PID=$(tr -dc '0-9' <"$PID_FILE")
if [[ -z "$WAIT_PID" ]]; then
  echo "PID file does not contain a valid PID: $PID_FILE" >&2
  exit 1
fi

if kill -0 "$WAIT_PID" 2>/dev/null; then
  START_TIME=$(awk '{print $22}' "/proc/$WAIT_PID/stat" 2>/dev/null || true)
  COMMAND=$(tr '\0' ' ' <"/proc/$WAIT_PID/cmdline" 2>/dev/null || true)
  echo "Waiting for PID $WAIT_PID: $COMMAND"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    CURRENT_START=$(awk '{print $22}' "/proc/$WAIT_PID/stat" 2>/dev/null || true)
    [[ -n "$START_TIME" && "$CURRENT_START" != "$START_TIME" ]] && break
    sleep 5
  done
fi

echo "[$(date --iso-8601=seconds)] Previous PID $WAIT_PID has exited"

REQUIRED=(
  outputs/qwen_text_decoder_06b/training/best_adapter/adapter_config.json
  outputs/qwen_text_decoder_06b/oracle_structured/synthetic_text.csv
  outputs/qwen_text_decoder_06b/oracle_structured/canonical_text_c2st.json
  outputs/qwen_text_decoder_06b/experiment_report.md
)
for artifact in "${REQUIRED[@]}"; do
  if [[ ! -s "$artifact" ]]; then
    echo "Main Qwen experiment did not complete required artifact: $artifact" >&2
    exit 1
  fi
done

echo "[$(date --iso-8601=seconds)] Required main-experiment artifacts verified"
set +e
python -u src/scripts/run_qwen_text_decoder_followup.py \
  --stage all \
  --device "$DEVICE"
STATUS=$?
set -e
echo "[$(date --iso-8601=seconds)] Follow-up exit status: $STATUS"
exit "$STATUS"
