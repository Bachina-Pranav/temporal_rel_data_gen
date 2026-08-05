# Hierarchical Diffusion Diagnostics

## What This Framework Answers

The diagnostic matrix separates five possible sources of end-to-end error:

| Mode | Structured values | Text lengths | Graph history |
| --- | --- | --- | --- |
| O1 | real | exact real | real strict-past prefix |
| O2 | generated | exact real | real strict-past prefix |
| O3 | generated | generated | real strict-past prefix |
| O4 | generated | generated | evaluation-spine history |
| O5 | generated | generated | none |

O1--O3 are diagnostic upper bounds, not valid generative baselines. O4 and O5
are valid generated pipelines. All modes use the same test events, tokenizer,
checkpoint, reverse schedule, rows, and seeds.

## First Run

Run from the repository root:

```bash
set -euo pipefail

MODEL_CONFIG=configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml
EVAL_CONFIG=configs/evaluation/single_event_table_paper_metrics_amazon_toy.yaml
BENCHMARK=outputs/amazon-toy/hierarchical_diffusion_benchmark

python src/scripts/prepare_hierarchical_diffusion_benchmark.py \
  --config "$MODEL_CONFIG" \
  --output-dir "$BENCHMARK" \
  --num-evaluation-rows all \
  --seed 42

python src/scripts/audit_c2st_integrity.py \
  --config "$EVAL_CONFIG" \
  --real-table "$BENCHMARK/evaluation_real.csv" \
  --output "$BENCHMARK/c2st_integrity_audit.json" \
  --max-rows-per-side 5000 \
  --seed 42

python src/scripts/run_hierarchical_diffusion_diagnostics.py \
  --experiment-config configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml \
  --matrices progressive_conditioning \
  --device cuda
```

Every invocation creates a timestamped root under
`outputs/amazon-toy/hierarchical_diffusion_diagnostics`. It writes:

- `resolved_experiment.json`;
- one directory per mode and seed;
- sampling runtime and GPU-memory metadata;
- paper metrics and legacy semantic-consistency metrics;
- special-token, length, diversity, repetition, and duplication diagnostics;
- `consolidated_results.csv`;
- `aggregate_mean_std.csv`;
- `results_report.md`;
- `diagnosis_and_recommendation.json`.

The benchmark runner verifies every frozen CSV hash before sampling and aborts
if any benchmark row has changed.

## Graph And Decoding Matrices

Run both without retraining:

```bash
python src/scripts/run_hierarchical_diffusion_diagnostics.py \
  --experiment-config configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml \
  --matrices graph_context decoding_policy \
  --device cuda
```

The graph matrix contains no-history, user-only, product-only, both histories,
both plus coverage metadata, shuffled histories, and zeroed context. Customer
and product contexts are shuffled independently. If output change and metric
change are negligible, the result report flags ignored graph conditioning.
Benchmark preparation also freezes a per-event history coverage table. Every
run reports user-side, product-side, and overall coverage, history-count
distributions, and separate C2ST results for cold, partially covered, and warm
events.

The decoding matrix contains greedy, two top-k settings, two nucleus settings,
two temperatures, and the current constrained policy. It always excludes
control tokens from content positions.

## Condition-Robustness Training

This runs clean, corrupted, and mixed-condition variants. It reuses the exact
prepared split and tokenizer from the base run and stores a resolved YAML for
every seed:

```bash
python src/scripts/run_hierarchical_condition_training_ablation.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml \
  --output-root outputs/amazon-toy/hierarchical_condition_training_ablation \
  --variants clean corrupted mixed \
  --seeds 17 42 73 \
  --diagnostic-experiment-config configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml \
  --device cuda
```

The mixed variant uses 50% clean, 25% schema-valid corrupted, and 25%
one-forward generated structured conditions. The final category is explicitly
not a full reverse-process sample; that remaining mismatch is documented in
the architecture report.

Condition corruption is configured under
`training.text_conditioning.corruption`: `replacement_probability`,
`mask_probability`, `length_bucket_step_probability`,
`allow_missing_replacement`, and optional `per_field` overrides. Loss weights
are configured with `loss_group_weights` and
`loss_groups.field_groups`; the standard groups are `structured`, `summary`,
`review`, and `auxiliary`. Set
`training.modality_gradient_audit_interval` to control the additional
structured/text gradient-norm audit.

## Loss-Weight Training

Run the independently normalized loss-group variants and evaluate both the O1
oracle ceiling and O4 generated pipeline:

```bash
python src/scripts/run_hierarchical_loss_weight_ablation.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml \
  --variants-config configs/experiments/hierarchical_diffusion_loss_weight_variants.yaml \
  --output-root outputs/amazon-toy/hierarchical_loss_weight_ablation \
  --seeds 17 42 73 \
  --diagnostic-experiment-config configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml \
  --device cuda
```

Both training-ablation runners preserve each resolved training configuration,
checkpoint hash, training time, per-seed diagnostic result, consolidated CSV,
mean and standard deviation table, and Markdown report.

## Structured-Only Comparison

Pass the diffusion and LSTM outputs produced on the frozen evaluation spine:

```bash
python src/scripts/compare_structured_attribute_generators.py \
  --model-config configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml \
  --evaluation-config configs/evaluation/single_event_table_paper_metrics_amazon_toy.yaml \
  --benchmark-manifest outputs/amazon-toy/hierarchical_diffusion_benchmark/benchmark_manifest.json \
  --model-output diffusion=/path/to/diffusion_on_benchmark_spine.csv \
  --model-output lstm=/path/to/lstm_on_benchmark_spine.csv \
  --runtime-metadata diffusion=/path/to/diffusion_runtime.json \
  --runtime-metadata lstm=/path/to/lstm_runtime.json \
  --output-root outputs/amazon-toy/structured_attribute_comparison \
  --seed 42
```

The script also creates a joint empirical conditional baseline and an
independent-column baseline. It reports validity, marginal shape, pairwise
association, structured C2ST, per-categorical TV distance, temporal trend,
and target error conditioned on every foreign-key entity. Runtime JSON files
are optional; when supplied, training and sampling times are extracted into
the comparison. Baseline generation time is measured directly.

## Debug Runs

Use one seed and one mode before a full matrix:

```bash
python src/scripts/run_hierarchical_diffusion_diagnostics.py \
  --experiment-config configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml \
  --matrices progressive_conditioning \
  --seeds 42 \
  --max-runs 1 \
  --device cuda
```

Use `--dry-run` to validate paths and print the resolved matrix without
sampling. Missing checkpoints, oracle rows, graph prefixes, or changed
benchmark files cause a clear failure; there is no silent fallback.

## Current Recommendation Status

No new O1--O5 values were fabricated in this checkout because the Amazon-toy
data and trained checkpoint are not present locally. The framework therefore
marks the recommendation as `not_yet_determined` until the fixed benchmark is
executed. Once complete, the generated report bases its recommendation on O1,
the adjacent O1--O5 degradations, and the fixed LSTM reference rather than
assuming that masked discrete diffusion is preferable.
