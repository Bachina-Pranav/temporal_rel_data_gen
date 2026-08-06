#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/hm-10k-customers/lstm_v53_three_seed}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/diagnostics/posthoc_v1}"
MAX_C2ST_ROWS="${MAX_C2ST_ROWS:-10000}"

mkdir -p "$OUTPUT_DIR/logs"

python -u src/scripts/run_lstm_posthoc_diagnostics.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --model-config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --evaluation-config configs/evaluation/single_event_table_paper_metrics_hm_10k_customers.yaml \
  --seeds 17 42 73 \
  --classifier-seeds 11 23 37 53 71 \
  --output-dir "$OUTPUT_DIR" \
  --phases all \
  --max-c2st-rows "$MAX_C2ST_ROWS" \
  --chance-tolerance 0.15 \
  --projection-classifier-seed 42 \
  --stochastic-neighbors 8 \
  --stochastic-temperature 1.0 \
  --min-entity-rows 5 \
  --permutation-repeats 5 \
  2>&1 | tee "$OUTPUT_DIR/logs/posthoc_diagnostics.log"
