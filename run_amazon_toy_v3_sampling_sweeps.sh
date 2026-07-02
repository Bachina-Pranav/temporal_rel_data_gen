#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run sampling-only V3 sweeps from one trained checkpoint.

This does not retrain. It resamples/evaluates the grid:
  calibration_strength: 0.5, 0.75, 1.0, 1.25
  customer_effect_scale: 1.0, 1.25, 1.5, 2.0
  lambda_customer_effect: 0.7, 1.0, 1.25, 1.5

Usage:
  ./run_amazon_toy_v3_sampling_sweeps.sh

Useful server run:
  nohup ./run_amazon_toy_v3_sampling_sweeps.sh > v3_sampling_sweeps.log 2>&1 &
  tail -f v3_sampling_sweeps.log

Common environment overrides:
  DEVICE, SEED, SAMPLE_STEPS, OUTPUT_ROOT, CHECKPOINT
  REAL_REVIEWS, SYNTHETIC_SPINE, STRUCTURE_DEBUG_DIR
EOF
  exit 0
fi

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
CAT_SAMPLING_STRATEGY="${CAT_SAMPLING_STRATEGY:-sample}"
TEMPERATURE="${TEMPERATURE:-1.0}"

REAL_REVIEWS="${REAL_REVIEWS:-data/original/rel-amazon-toy/review.csv}"
SYNTHETIC_SPINE="${SYNTHETIC_SPINE:-outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/synthetic_review.csv}"
STRUCTURE_DEBUG_DIR="${STRUCTURE_DEBUG_DIR:-outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/debug}"
CHECKPOINT="${CHECKPOINT:-outputs/amazon-toy/nontext_attr_3_methods_200ep_seed42/v3_temporal_nontext_attr_diffusion/checkpoints/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/amazon-toy/v3_sampling_sweeps_200ep_seed42}"

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

require_file "$REAL_REVIEWS"
require_file "$SYNTHETIC_SPINE"
require_file "$CHECKPOINT"
require_dir "$STRUCTURE_DEBUG_DIR"

python -u src/scripts/sweep_temporal_nontext_v3_grid.py \
  --real-reviews "$REAL_REVIEWS" \
  --synthetic-spine "$SYNTHETIC_SPINE" \
  --structure-debug-dir "$STRUCTURE_DEBUG_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --calibration-strengths 0.5 0.75 1.0 1.25 \
  --customer-effect-scales 1.0 1.25 1.5 2.0 \
  --lambda-customer-effects 0.7 1.0 1.25 1.5 \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --cat-cols rating verified \
  --num-diffusion-steps "$SAMPLE_STEPS" \
  --cat-sampling-strategy "$CAT_SAMPLING_STRATEGY" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  "$@"

echo
echo "Sweep complete."
echo "CSV:"
echo "  $OUTPUT_ROOT/v3_sampling_sweep_summary.csv"
echo "HTML:"
echo "  $OUTPUT_ROOT/v3_sampling_sweep_comparison.html"
