# MovieLens LSTM Diagnosis

This document records the leakage-safe diagnostic protocol for the MovieLens
single interaction-table LSTM attribute generator.

## Current Run Map

| Item | Path / Command |
| --- | --- |
| Dataset config | `configs/datasets/movielens_100k.yaml` |
| Model config | `configs/attribute_generation/lstm_movielens_100k.yaml` |
| Ordinal ablation config | `configs/attribute_generation/lstm_movielens_100k_ordinal_0p1.yaml` |
| Training entry point | `src/scripts/train_lstm_joint_full_review_text.py` |
| Sampling entry point | `src/scripts/sample_lstm_joint_full_review_text.py` |
| Paper evaluation entry point | `src/scripts/evaluate_single_event_table_paper_metrics.py` |
| Normal rating diagnostics | `src/scripts/evaluate_interaction_rating_diagnostics.py` |
| Current checkpoint | `outputs/movielens-100k/lstm_v53/checkpoints/best.pt` |
| Split files | `outputs/movielens-100k/lstm_v53/spines/{train,validation,test}_{real,spine}.csv` |
| Rating vocab metadata | `outputs/movielens-100k/lstm_v53/data/vocab_rating.json` |
| Event spine for real-spine diagnostic | `outputs/movielens-100k/lstm_v53/spines/test_spine.csv` |
| Real rows for fixed evaluation | `outputs/movielens-100k/lstm_v53/spines/test_real.csv` |

The MovieLens LSTM generates only `rating` from `user_id`, `movie_id`, and
`event_time`. No text modules should be instantiated for this config.

## Known Baseline Snapshot

The previously reported real-spine diagnostic used 15,000 real rows and 15,000
synthetic rows.

Paper-grade summary:

```json
{
  "constraint_violation_rate": 0.0,
  "fk_cardinality_similarity": 1.0,
  "shape_error": 0.12319999999999998,
  "single_table_c2st_error": 0.45349980888888886,
  "temporal_event_distance": 0.0,
  "text_embedding_c2st_error": null,
  "trend_error": 0.025434736891931566
}
```

Legacy diagnostics:

```json
{
  "invalid_rating_rate": 0.0018666666666666666,
  "rating_distribution_js": 0.02872827158650329,
  "rating_distribution_l1": 0.33416572226990665,
  "rating_ks": 0.1670828611349533,
  "rating_total_variation": 0.16708286113495333,
  "monthly_rating_mean_corr": -0.11741135964558688,
  "monthly_rating_mean_mae": 0.41142468191257925,
  "customer_rating_top_1000_coverage": 1.0,
  "customer_rating_top_1000_mae": 0.49708505130507535,
  "product_rating_top_1000_coverage": 1.0,
  "product_rating_top_1000_mae": 0.5979236386327712
}
```

The evaluator disagreement was caused by legacy normalization defaulting to the
integer Amazon rating support `[1, 2, 3, 4, 5]`. The MovieLens config declares
the half-star ordinal domain. The legacy evaluator now derives the valid rating
domain from config when available and reports raw/canonicalized invalid counts.

## Rating-Domain Audit

```bash
python src/scripts/audit_interaction_rating_domain.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --evaluation-config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
  --processed-table data/processed/interaction_benchmarks/movielens_100k/interactions.csv \
  --train-table outputs/movielens-100k/lstm_v53/data/train.parquet \
  --validation-table outputs/movielens-100k/lstm_v53/data/valid.parquet \
  --test-table outputs/movielens-100k/lstm_v53/data/test.parquet \
  --sampled-output outputs/movielens-100k/lstm_v53/samples/test_real_spine_synthetic_interactions.csv \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/rating_domain
```

Outputs:

```text
outputs/movielens-100k/lstm_v53/diagnostics/rating_domain/rating_domain_audit.json
outputs/movielens-100k/lstm_v53/diagnostics/rating_domain/rating_domain_audit.csv
outputs/movielens-100k/lstm_v53/diagnostics/rating_domain/rating_domain_audit.md
```

Expected domain for MovieLens should be derived from the config and observed
data, normally:

```text
0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
```

## Reproduce Baseline

Do not overwrite the current run. Resample from the existing checkpoint into a
diagnostic directory:

```bash
mkdir -p outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline

python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/synthetic_interactions.csv \
  --num-rows all \
  --device cuda \
  --seed 42
```

Paper-grade evaluation:

```bash
python src/scripts/evaluate_single_event_table_paper_metrics.py \
  --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
  --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/synthetic_interactions.csv \
  --legacy-config configs/attribute_generation/lstm_movielens_100k.yaml \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/evaluation \
  --seed 42
```

Normal rating diagnostics:

```bash
python src/scripts/evaluate_interaction_rating_diagnostics.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/synthetic_interactions.csv \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/evaluation \
  --c2st \
  --seed 42
```

## Empirical Baselines

All baselines fit from the train split only.

```bash
python src/scripts/run_interaction_rating_baselines.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines \
  --smoothing 5.0 \
  --min-group-count 2 \
  --include-time-baseline \
  --time-bin year \
  --seed 42
```

Evaluate each baseline on the same fixed test rows:

```bash
for name in global_empirical user_empirical movie_empirical user_movie_mixture time_empirical; do
  python src/scripts/evaluate_single_event_table_paper_metrics.py \
    --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
    --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
    --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/${name}/synthetic_interactions.csv \
    --legacy-config configs/attribute_generation/lstm_movielens_100k.yaml \
    --output-dir outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/${name}/evaluation \
    --seed 42

  python src/scripts/evaluate_interaction_rating_diagnostics.py \
    --config configs/attribute_generation/lstm_movielens_100k.yaml \
    --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
    --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/${name}/synthetic_interactions.csv \
    --output-dir outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/${name}/evaluation \
    --c2st \
    --seed 42
done
```

## Graph-Context Diagnostic

```bash
python src/scripts/diagnose_lstm_graph_context.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --reference-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/graph_context \
  --num-rows 15000 \
  --device cuda \
  --seed 42
```

This writes correct, zero, and shuffled graph-context samples plus logit-level
diagnostics:

```text
mean_abs_logit_difference
mean_kl_divergence
argmax_changed_fraction
mean_abs_expected_rating_change
```

Evaluate each graph mode:

```bash
for mode in correct zero shuffled; do
  python src/scripts/evaluate_interaction_rating_diagnostics.py \
    --config configs/attribute_generation/lstm_movielens_100k.yaml \
    --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
    --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/graph_context/${mode}/synthetic_interactions.csv \
    --output-dir outputs/movielens-100k/lstm_v53/diagnostics/graph_context/${mode}/evaluation \
    --c2st \
    --seed 42
done
```

## Conditioning-Input Diagnostic

```bash
python src/scripts/diagnose_lstm_conditioning_inputs.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output-dir outputs/movielens-100k/lstm_v53/diagnostics/conditioning_inputs \
  --num-rows 15000 \
  --device cuda \
  --seed 42
```

Evaluate modes:

```bash
for mode in correct shuffled_user shuffled_movie shuffled_time zero_graph; do
  python src/scripts/evaluate_interaction_rating_diagnostics.py \
    --config configs/attribute_generation/lstm_movielens_100k.yaml \
    --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
    --synthetic-table outputs/movielens-100k/lstm_v53/diagnostics/conditioning_inputs/${mode}/synthetic_interactions.csv \
    --output-dir outputs/movielens-100k/lstm_v53/diagnostics/conditioning_inputs/${mode}/evaluation \
    --c2st \
    --seed 42
done
```

## Ordinal-Loss Training

Run this only after the rating-domain audit is clean and empirical baselines are
evaluated.

```bash
python src/scripts/train_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k_ordinal_0p1.yaml \
  --output-dir outputs/movielens-100k/lstm_v53_ordinal_0p1 \
  --device cuda \
  --mixed-precision
```

Sample and evaluate the ordinal run on the same fixed rows:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k_ordinal_0p1.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53_ordinal_0p1/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output outputs/movielens-100k/lstm_v53_ordinal_0p1/samples/test_real_spine_synthetic_interactions.csv \
  --num-rows all \
  --device cuda \
  --seed 42

python src/scripts/evaluate_single_event_table_paper_metrics.py \
  --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
  --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53_ordinal_0p1/samples/test_real_spine_synthetic_interactions.csv \
  --legacy-config configs/attribute_generation/lstm_movielens_100k_ordinal_0p1.yaml \
  --output-dir outputs/movielens-100k/lstm_v53_ordinal_0p1/evaluation/test_real_spine \
  --seed 42

python src/scripts/evaluate_interaction_rating_diagnostics.py \
  --config configs/attribute_generation/lstm_movielens_100k_ordinal_0p1.yaml \
  --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53_ordinal_0p1/samples/test_real_spine_synthetic_interactions.csv \
  --output-dir outputs/movielens-100k/lstm_v53_ordinal_0p1/evaluation/test_real_spine \
  --c2st \
  --seed 42
```

## Temperature Calibration

Fit on validation only:

```bash
python src/scripts/calibrate_lstm_rating_temperature.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --output outputs/movielens-100k/lstm_v53/diagnostics/temperature_calibration.json \
  --temperatures 0.5 0.75 1.0 1.25 1.5 2.0 \
  --device cuda \
  --seed 42
```

Sample with the selected temperature, replacing `BEST_TEMP`:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output outputs/movielens-100k/lstm_v53/diagnostics/calibrated_temperature/synthetic_interactions.csv \
  --num-rows all \
  --temperature BEST_TEMP \
  --device cuda \
  --seed 42
```

## Synthetic-Spine Primary Result

After real-spine diagnostics, evaluate end-to-end only under a separate label:

```bash
python src/scripts/run_time_biased_block_stub_matching_generator.py \
  --real-reviews data/processed/interaction_benchmarks/movielens_100k/interactions.csv \
  --customer-id-col user_id \
  --product-id-col movie_id \
  --timestamp-col event_time \
  --dataset movielens_100k \
  --output-dir outputs/movielens-100k/event_spine_time_biased_stub_matching \
  --time-granularity day \
  --time-gate-granularity month \
  --rank 32 \
  --alpha-customer-time auto \
  --alpha-product-time auto \
  --alpha-time-gate auto \
  --pairing-mode dynamic_exact_penalized \
  --max-exact-affinity-cell-size 128 \
  --seed 42

python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/event_spine_time_biased_stub_matching/synthetic_review.csv \
  --output outputs/movielens-100k/lstm_v53/samples/full_synthetic_interactions.csv \
  --num-rows all \
  --device cuda \
  --seed 42
```

## Build Comparison Table

Put paper and normal diagnostics in the same evaluation directories, then run:

```bash
python src/scripts/make_interaction_lstm_model_comparison.py \
  --metric "Global empirical" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/global_empirical/evaluation/paper_metrics.json \
  --metric "User empirical" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/user_empirical/evaluation/paper_metrics.json \
  --metric "Movie empirical" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/movie_empirical/evaluation/paper_metrics.json \
  --metric "User-movie mixture" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53/diagnostics/empirical_baselines/user_movie_mixture/evaluation/paper_metrics.json \
  --metric "Original LSTM" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53/diagnostics/reproduced_baseline/evaluation/paper_metrics.json \
  --metric "Ordinal LSTM 0.1" "REAL-SPINE ATTRIBUTE DIAGNOSTIC" outputs/movielens-100k/lstm_v53_ordinal_0p1/evaluation/test_real_spine/paper_metrics.json \
  --output outputs/movielens-100k/lstm_v53/diagnostics/model_comparison.csv
```

Required final table fields:

```text
Model, Rating domain, Rating TV, Rating JS, Ordinal Wasserstein,
User MAE, Movie MAE, Monthly corr, Monthly MAE, C2ST AUC/error,
Invalid rate
```

## Tests

```bash
pytest -q \
  tests/test_movielens_lstm_compatibility.py \
  tests/test_lstm_joint_model_outputs.py \
  tests/test_evaluator_rating_normalization.py \
  tests/test_paper_metrics_schema_validation.py
```

## Decision Gate

Use the real-spine diagnostic first.

- If the domain audit finds mismatch, fix/retrain the categorical baseline before interpreting other metrics.
- If empirical baselines beat LSTM on user/movie MAE, inspect conditioning and do not assume ordinal loss solves it.
- If correct graph context does not beat zero/shuffled graph, graph conditioning is ignored or ineffective.
- If LSTM conditional metrics are good but rating marginal is bad, prefer validation temperature calibration before architecture changes.
- If most errors are nearby half-star categories, use the ordinal auxiliary ablation.

