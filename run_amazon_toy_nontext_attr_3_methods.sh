#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run V1, V2, and V3 non-text temporal attribute generators on rel-amazon-toy.

Usage:
  ./run_amazon_toy_nontext_attr_3_methods.sh [epochs]

Examples:
  ./run_amazon_toy_nontext_attr_3_methods.sh 100
  ./run_amazon_toy_nontext_attr_3_methods.sh 200
  EPOCHS=200 SEED=7 DEVICE=cuda ./run_amazon_toy_nontext_attr_3_methods.sh

Common environment overrides:
  EPOCHS, SEED, DEVICE, TRAIN_BATCH_SIZE, SAMPLE_STEPS, OUTPUT_ROOT
  REAL_REVIEWS, SYNTHETIC_SPINE, STRUCTURE_DEBUG_DIR
EOF
  exit 0
fi

EPOCHS="${1:-${EPOCHS:-100}}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
CAT_SAMPLING_STRATEGY="${CAT_SAMPLING_STRATEGY:-sample}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TEMPORAL_CALIBRATION_STRENGTH="${TEMPORAL_CALIBRATION_STRENGTH:-0.75}"

REAL_REVIEWS="${REAL_REVIEWS:-data/original/rel-amazon-toy/review.csv}"
SYNTHETIC_SPINE="${SYNTHETIC_SPINE:-outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/synthetic_review.csv}"
STRUCTURE_DEBUG_DIR="${STRUCTURE_DEBUG_DIR:-outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/debug}"
RUN_TAG="${RUN_TAG:-${EPOCHS}ep_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/amazon-toy/nontext_attr_3_methods_${RUN_TAG}}"

CUSTOMER_ID_COL="${CUSTOMER_ID_COL:-customer_id}"
PRODUCT_ID_COL="${PRODUCT_ID_COL:-product_id}"
TIMESTAMP_COL="${TIMESTAMP_COL:-review_time}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
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

require_file "$REAL_REVIEWS"
require_file "$SYNTHETIC_SPINE"
require_dir "$STRUCTURE_DEBUG_DIR"
mkdir -p "$OUTPUT_ROOT"

echo "Running serious non-text attribute comparison"
echo "  epochs:              $EPOCHS"
echo "  seed:                $SEED"
echo "  device:              $DEVICE"
echo "  real reviews:        $REAL_REVIEWS"
echo "  synthetic spine:     $SYNTHETIC_SPINE"
echo "  structure debug dir: $STRUCTURE_DEBUG_DIR"
echo "  output root:         $OUTPUT_ROOT"

V1_DIR="$OUTPUT_ROOT/v1_temporal_nontext_attr_diffusion"
V2_PRIOR_DIR="$OUTPUT_ROOT/v2_entity_effect_priors"
V2_DIR="$OUTPUT_ROOT/v2_temporal_nontext_attr_diffusion"
V3_DIR="$OUTPUT_ROOT/v3_temporal_nontext_attr_diffusion"

V1_SYNTHETIC="$OUTPUT_ROOT/synthetic_review_nontext_v1.csv"
V2_SYNTHETIC="$OUTPUT_ROOT/synthetic_review_nontext_v2.csv"
V3_SYNTHETIC="$OUTPUT_ROOT/synthetic_review_nontext_v3.csv"

run_step python -u src/scripts/train_temporal_nontext_attr_diffusion.py \
  --real-reviews "$REAL_REVIEWS" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --output-dir "$V1_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/sample_temporal_nontext_attr_diffusion.py \
  --synthetic-spine "$SYNTHETIC_SPINE" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --checkpoint "$V1_DIR/checkpoints/best.pt" \
  --output "$V1_SYNTHETIC" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --num-diffusion-steps "$SAMPLE_STEPS" \
  --cat-sampling-strategy "$CAT_SAMPLING_STRATEGY" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/evaluate_temporal_nontext_attrs.py \
  --real-reviews "$REAL_REVIEWS" \
  --synthetic-reviews "$V1_SYNTHETIC" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --temporal-bucket-level year_month \
  --diagnostics-dir "$V1_DIR/eval_diagnostics" \
  --output "$V1_DIR/metrics.json"

run_step python -u src/scripts/train_entity_effect_priors.py \
  --real-reviews "$REAL_REVIEWS" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --rating-col rating \
  --verified-col verified \
  --output-dir "$V2_PRIOR_DIR" \
  --seed "$SEED"

run_step python -u src/scripts/train_temporal_nontext_attr_diffusion_v2.py \
  --real-reviews "$REAL_REVIEWS" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --entity-prior-dir "$V2_PRIOR_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --output-dir "$V2_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/sample_temporal_nontext_attr_diffusion_v2.py \
  --synthetic-spine "$SYNTHETIC_SPINE" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --entity-prior-dir "$V2_PRIOR_DIR" \
  --checkpoint "$V2_DIR/checkpoints/best.pt" \
  --output "$V2_SYNTHETIC" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --num-diffusion-steps "$SAMPLE_STEPS" \
  --cat-sampling-strategy "$CAT_SAMPLING_STRATEGY" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/evaluate_temporal_nontext_attrs.py \
  --real-reviews "$REAL_REVIEWS" \
  --synthetic-reviews "$V2_SYNTHETIC" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --temporal-bucket-level year_month \
  --diagnostics-dir "$V2_DIR/eval_diagnostics" \
  --output "$V2_DIR/metrics.json"

run_step python -u src/scripts/train_temporal_nontext_attr_diffusion_v3.py \
  --real-reviews "$REAL_REVIEWS" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --output-dir "$V3_DIR" \
  --temporal-prior-level year_month \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/sample_temporal_nontext_attr_diffusion_v3.py \
  --synthetic-spine "$SYNTHETIC_SPINE" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --checkpoint "$V3_DIR/checkpoints/best.pt" \
  --output "$V3_SYNTHETIC" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --num-diffusion-steps "$SAMPLE_STEPS" \
  --cat-sampling-strategy "$CAT_SAMPLING_STRATEGY" \
  --temperature "$TEMPERATURE" \
  --use-temporal-calibration \
  --temporal-calibration-strength "$TEMPORAL_CALIBRATION_STRENGTH" \
  --diagnostics-dir "$V3_DIR/sample_diagnostics" \
  --device "$DEVICE" \
  --seed "$SEED"

run_step python -u src/scripts/evaluate_temporal_nontext_attrs.py \
  --real-reviews "$REAL_REVIEWS" \
  --synthetic-reviews "$V3_SYNTHETIC" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --customer-id-col "$CUSTOMER_ID_COL" \
  --product-id-col "$PRODUCT_ID_COL" \
  --timestamp-col "$TIMESTAMP_COL" \
  --cat-cols rating verified \
  --temporal-bucket-level year_month \
  --diagnostics-dir "$V3_DIR/eval_diagnostics" \
  --output "$V3_DIR/metrics.json"

run_step python -u src/scripts/compare_nontext_attr_versions.py \
  --metrics \
  "v1=$V1_DIR/metrics.json" \
  "v2=$V2_DIR/metrics.json" \
  "v3=$V3_DIR/metrics.json" \
  --output "$OUTPUT_ROOT/nontext_attr_versions_comparison.csv"

echo
echo "Done."
echo "Outputs:"
echo "  $OUTPUT_ROOT"
echo "Comparison CSV:"
echo "  $OUTPUT_ROOT/nontext_attr_versions_comparison.csv"
