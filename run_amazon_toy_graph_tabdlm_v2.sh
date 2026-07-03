#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run ConditionalTABDLM v2 graph-conditioned attribute generation on Amazon-toy.

This runs, in order:
  1. prepare graph-conditioned TABDLM data
  2. train graph-conditioned TABDLM
  3. sample attributes on the synthetic event spine
  4. evaluate generated attributes
  5. compare against the v1.2 baseline metrics, if present

Usage:
  ./run_amazon_toy_graph_tabdlm_v2.sh

Useful server run:
  nohup ./run_amazon_toy_graph_tabdlm_v2.sh > graph_tabdlm_v2.log 2>&1 &
  tail -f graph_tabdlm_v2.log

Common environment overrides:
  DEVICE, NUM_ROWS, CONFIG, CHECKPOINT, SYNTHETIC_SPINE, REAL_REVIEWS
  OUTPUT_DIR, OUTPUT_ATTRS, BASELINE_METRICS

Resume/skip flags:
  SKIP_PREPARE=1
  SKIP_TRAIN=1
  SKIP_SAMPLE=1
  SKIP_EVALUATE=1
  SKIP_COMPARE=1

Examples:
  DEVICE=cuda NUM_ROWS=50000 ./run_amazon_toy_graph_tabdlm_v2.sh
  SKIP_TRAIN=1 CHECKPOINT=outputs/amazon-toy/conditional_tabdlm_exp2_structure_graph/checkpoints/best.pt ./run_amazon_toy_graph_tabdlm_v2.sh
EOF
  exit 0
fi

CONFIG="${CONFIG:-configs/attribute_generation/conditional_tabdlm_amazon_toy_exp2_graph_structure.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/amazon-toy/conditional_tabdlm_exp2_structure_graph}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best.pt}"
SYNTHETIC_SPINE="${SYNTHETIC_SPINE:-outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv}"
REAL_REVIEWS="${REAL_REVIEWS:-data/original/rel-amazon-toy/review.csv}"
OUTPUT_ATTRS="${OUTPUT_ATTRS:-$OUTPUT_DIR/synthetic_review_attrs.csv}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$OUTPUT_DIR/evaluation/eval_metrics.json}"
BASELINE_METRICS="${BASELINE_METRICS:-outputs/amazon-toy/conditional_tabdlm_exp1_2_length_calibrated/evaluation/eval_metrics.json}"

DEVICE="${DEVICE:-cuda}"
NUM_ROWS="${NUM_ROWS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
SEED="${SEED:-}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_EVALUATE="${SKIP_EVALUATE:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_step() {
  echo
  echo "================================================================================"
  echo "$*"
  echo "================================================================================"
  "$@"
}

optional_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$flag" "$value"
  fi
}

require_file "$CONFIG"
require_file "$REAL_REVIEWS"
require_file "$SYNTHETIC_SPINE"
mkdir -p "$OUTPUT_DIR"

echo "Running ConditionalTABDLM v2 graph-conditioned Amazon-toy experiment"
echo "  config:          $CONFIG"
echo "  real reviews:    $REAL_REVIEWS"
echo "  synthetic spine: $SYNTHETIC_SPINE"
echo "  output dir:      $OUTPUT_DIR"
echo "  checkpoint:      $CHECKPOINT"
echo "  output attrs:    $OUTPUT_ATTRS"
echo "  eval output:     $EVAL_OUTPUT"
echo "  device:          $DEVICE"
echo "  num rows:        $NUM_ROWS"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  run_step python -u src/scripts/prepare_graph_conditioned_tabdlm_amazon_toy.py \
    --config "$CONFIG"
else
  echo "Skipping prepare because SKIP_PREPARE=1"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  run_step python -u src/scripts/train_graph_conditioned_tabdlm.py \
    --config "$CONFIG" \
    --device "$DEVICE"
else
  echo "Skipping train because SKIP_TRAIN=1"
fi

require_file "$CHECKPOINT"

if [[ "$SKIP_SAMPLE" != "1" ]]; then
  sample_cmd=(
    python -u src/scripts/sample_graph_conditioned_tabdlm.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --synthetic-spine "$SYNTHETIC_SPINE"
    --num-rows "$NUM_ROWS"
    --output "$OUTPUT_ATTRS"
    --device "$DEVICE"
  )
  while IFS= read -r item; do
    sample_cmd+=("$item")
  done < <(optional_arg --batch-size "$BATCH_SIZE")
  while IFS= read -r item; do
    sample_cmd+=("$item")
  done < <(optional_arg --temperature "$TEMPERATURE")
  while IFS= read -r item; do
    sample_cmd+=("$item")
  done < <(optional_arg --top-p "$TOP_P")
  while IFS= read -r item; do
    sample_cmd+=("$item")
  done < <(optional_arg --seed "$SEED")
  run_step "${sample_cmd[@]}"
else
  echo "Skipping sample because SKIP_SAMPLE=1"
fi

require_file "$OUTPUT_ATTRS"

if [[ "$SKIP_EVALUATE" != "1" ]]; then
  run_step python -u src/scripts/evaluate_graph_conditioned_tabdlm.py \
    --config "$CONFIG" \
    --real-reviews "$REAL_REVIEWS" \
    --synthetic-reviews "$OUTPUT_ATTRS" \
    --output "$EVAL_OUTPUT"
else
  echo "Skipping evaluate because SKIP_EVALUATE=1"
fi

if [[ "$SKIP_COMPARE" != "1" ]]; then
  if [[ -f "$BASELINE_METRICS" && -f "$EVAL_OUTPUT" ]]; then
    run_step python -u src/scripts/compare_conditional_tabdlm_v1_v2.py \
      --baseline "$BASELINE_METRICS" \
      --graph "$EVAL_OUTPUT" \
      --output-dir "$OUTPUT_DIR"
  else
    echo "Skipping compare because one of these files is missing:"
    echo "  baseline: $BASELINE_METRICS"
    echo "  graph:    $EVAL_OUTPUT"
  fi
else
  echo "Skipping compare because SKIP_COMPARE=1"
fi

echo
echo "Done."
echo "Generated attributes:"
echo "  $OUTPUT_ATTRS"
echo "Evaluation metrics:"
echo "  $EVAL_OUTPUT"
echo "Sample metadata:"
echo "  $OUTPUT_DIR/sample_metadata.json"
