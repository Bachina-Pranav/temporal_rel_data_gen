#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/attribute_generation/lstm_hm_10k_customers.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/evaluation/single_event_table_paper_metrics_hm_10k_customers.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/hm-10k-customers/lstm_v53_three_seed}"
PRETOKENIZED_DIR="${PRETOKENIZED_DIR:-data/processed/interaction_benchmarks/hm_10k_customers/pretokenized_lstm_explicit_split}"
NEIGHBOR_CACHE_DIR="${NEIGHBOR_CACHE_DIR:-data/processed/interaction_benchmarks/hm_10k_customers/neighbor_cache}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8192}"
MODE="${1:-full}"

EXTRA_ARGS=()
case "$MODE" in
  full)
    ;;
  smoke)
    EXTRA_ARGS+=(--smoke-only)
    ;;
  resume)
    EXTRA_ARGS+=(--skip-smoke --skip-existing)
    ;;
  *)
    echo "Usage: $0 [full|smoke|resume]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT/logs"

python -u src/scripts/run_lstm_multiseed_experiment.py \
  --config "$CONFIG" \
  --evaluation-config "$EVAL_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --pretokenized-dir "$PRETOKENIZED_DIR" \
  --neighbor-cache-dir "$NEIGHBOR_CACHE_DIR" \
  --seeds 17 42 73 \
  --device "$DEVICE" \
  --sample-batch-size "$SAMPLE_BATCH_SIZE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/logs/launcher.log"
