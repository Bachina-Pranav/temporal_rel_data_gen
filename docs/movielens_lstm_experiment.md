# MovieLens-100K-Induced LSTM Attribute Experiment

This experiment reuses the Amazon v5.3 joint LSTM attribute generator on the induced MovieLens interaction subset:

```text
event_id, user_id, movie_id, event_time -> rating
```

The model generates only the temporal interaction-table attribute `rating`. It does not generate `users.csv` or `movies.csv`; those are fixed support tables for FK validation.

## Dataset

Expected processed directory:

```text
data/processed/interaction_benchmarks/movielens_100k/
```

Expected files:

```text
interactions.csv
users.csv
movies.csv
schema.yaml
subset_manifest.json
validation_report.json
statistics.json
```

The preprocessing selects complete user histories from MovieLens 25M, then assigns chronological splits:

```text
train: earliest 70%
validation: next 15%
test: latest 15%
```

The LSTM data loader now honors the explicit `split` column when present. It falls back to the legacy 90/5/5 time split only for older datasets without a split column.

## Architecture Reused From Amazon V5.3

Reused entry points:

```text
src/scripts/train_lstm_joint_full_review_text.py
src/scripts/sample_lstm_joint_full_review_text.py
src/scripts/evaluate_single_event_table_paper_metrics.py
```

Reused implementation:

```text
src/attribute_generation/conditional_tabdlm/lstm_joint.py
src/attribute_generation/conditional_tabdlm/dataset.py
src/attribute_generation/conditional_tabdlm/graph_dataset.py
src/attribute_generation/conditional_tabdlm/neighbor_sampling.py
src/attribute_generation/conditional_tabdlm/graph_encoder.py
```

The MovieLens model path is:

```text
user_id, movie_id, event_time, past-only graph context
  -> shared event-context encoder
  -> stochastic row latent
  -> categorical rating head
  -> sampled rating
```

The categorical head is fit over the observed training rating vocabulary. Half-star ratings are valid.

## Disabled Components

MovieLens has no text or numerical generated attributes, so these are disabled:

```text
summary decoder
review_text decoder
token-level text generation
text-length auxiliary heads
text losses
numerical heads
```

The implementation now instantiates no text embedding, no text decoders, and no text heads when `columns.target.text` is empty.

## Leakage Safety

The graph context is structure-only and past-only. For an event at `event_time = t`, the graph history index may use only events with:

```text
event_time < t
```

The graph contains source/destination identities, past interaction structure, and timestamps. It does not contain the current rating, future ratings, summary text, review text, or any generated target attribute.

Validation graph context is built from train plus validation rows, but each validation event still receives only strict-past neighbors. Test labels are not used for training or model selection.

## Configs

Dataset config:

```text
configs/datasets/movielens_100k.yaml
```

Model config:

```text
configs/attribute_generation/lstm_movielens_100k.yaml
```

Paper metrics config:

```text
configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml
```

Output directory:

```text
outputs/movielens-100k/lstm_v53/
```

## Commands

Validate the processed subset:

```bash
python src/scripts/validate_interaction_subsets.py \
  --datasets movielens_100k \
  --processed-root data/processed/interaction_benchmarks
```

Smoke-test the schema-driven LSTM loader/model without text modules:

```bash
python src/scripts/validate_lstm_dataset_configs.py \
  --datasets movielens_100k \
  --run-forward-pass \
  --run-loss-pass \
  --run-sampling-smoke-test \
  --device cpu
```

Materialize split-specific real tables and spines:

```bash
python src/scripts/materialize_interaction_lstm_splits.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --output-dir outputs/movielens-100k/lstm_v53/spines
```

Train one seed:

```bash
python src/scripts/train_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --output-dir outputs/movielens-100k/lstm_v53 \
  --device cuda \
  --mixed-precision \
  --save-best
```

Resume epoch-mode training:

```bash
python src/scripts/train_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --output-dir outputs/movielens-100k/lstm_v53 \
  --device cuda \
  --mixed-precision \
  --resume-from outputs/movielens-100k/lstm_v53/checkpoints/last.pt
```

Generate a 10,000-row real-spine attribute diagnostic:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output outputs/movielens-100k/lstm_v53/samples/real_spine_diagnostic_10k.csv \
  --num-rows 10000 \
  --device cuda \
  --seed 42
```

This is a real-spine diagnostic, not an end-to-end synthetic database result.

Generate a MovieLens synthetic event spine:

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
```

If no MovieLens block files are supplied, this generator falls back to one user block and one movie block. That is valid for a first end-to-end baseline but should be improved later.

Generate the full end-to-end synthetic interaction table:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/movielens-100k/event_spine_time_biased_stub_matching/synthetic_review.csv \
  --output outputs/movielens-100k/lstm_v53/samples/full_synthetic_interactions.csv \
  --num-rows all \
  --device cuda \
  --seed 42
```

Run train-only empirical baselines on the test real spine:

```bash
python src/scripts/run_interaction_rating_baselines.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --output-dir outputs/movielens-100k/lstm_v53/baselines/real_spine_test \
  --seed 42
```

Evaluate those baselines:

```bash
for name in global_empirical user_empirical movie_empirical; do
  python src/scripts/evaluate_single_event_table_paper_metrics.py \
    --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
    --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
    --synthetic-table outputs/movielens-100k/lstm_v53/baselines/real_spine_test/${name}/synthetic_interactions.csv \
    --output-dir outputs/movielens-100k/lstm_v53/evaluation/${name} \
    --seed 42
done
```

Evaluate the real-spine LSTM diagnostic:

```bash
python src/scripts/evaluate_single_event_table_paper_metrics.py \
  --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
  --real-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53/samples/real_spine_diagnostic_10k.csv \
  --output-dir outputs/movielens-100k/lstm_v53/evaluation/real_spine_diagnostic_10k \
  --sample-size 10000 \
  --seed 42
```

Evaluate the full synthetic result:

```bash
python src/scripts/evaluate_single_event_table_paper_metrics.py \
  --config configs/evaluation/single_event_table_paper_metrics_movielens_100k.yaml \
  --real-table data/processed/interaction_benchmarks/movielens_100k/interactions.csv \
  --synthetic-table outputs/movielens-100k/lstm_v53/samples/full_synthetic_interactions.csv \
  --output-dir outputs/movielens-100k/lstm_v53/evaluation/synthetic_spine_primary \
  --seed 42
```

Run graph-context diagnostics:

```bash
python src/scripts/diagnose_lstm_graph_context.py \
  --config configs/attribute_generation/lstm_movielens_100k.yaml \
  --checkpoint outputs/movielens-100k/lstm_v53/checkpoints/best.pt \
  --spine outputs/movielens-100k/lstm_v53/spines/test_spine.csv \
  --reference-table outputs/movielens-100k/lstm_v53/spines/test_real.csv \
  --output-dir outputs/movielens-100k/lstm_v53/evaluation/graph_context_diagnostic \
  --num-rows 10000 \
  --device cuda \
  --seed 42
```

Build a compact comparison CSV after evaluations are available:

```bash
python src/scripts/make_interaction_lstm_model_comparison.py \
  --metric "global empirical" "real-spine diagnostic" outputs/movielens-100k/lstm_v53/evaluation/global_empirical/paper_metrics.json \
  --metric "user empirical" "real-spine diagnostic" outputs/movielens-100k/lstm_v53/evaluation/user_empirical/paper_metrics.json \
  --metric "movie empirical" "real-spine diagnostic" outputs/movielens-100k/lstm_v53/evaluation/movie_empirical/paper_metrics.json \
  --metric "LSTM" "real-spine diagnostic" outputs/movielens-100k/lstm_v53/evaluation/real_spine_diagnostic_10k/paper_metrics.json \
  --metric "LSTM" "synthetic-spine primary" outputs/movielens-100k/lstm_v53/evaluation/synthetic_spine_primary/paper_metrics.json \
  --output outputs/movielens-100k/lstm_v53/evaluation/model_comparison.csv
```

Run fast tests:

```bash
pytest -q \
  tests/test_movielens_lstm_compatibility.py \
  tests/test_interaction_benchmarks.py \
  tests/test_lstm_joint_model_outputs.py \
  tests/test_paper_metrics_schema_validation.py
```

## Reporting Template

When the VM runs finish, report:

| Model | Spine | Rating TV ↓ | Rating Wasserstein ↓ | C2ST AUC ↓ | Trend ↓ | Validity violations ↓ | Runtime |
| ----- | ----- | ----------: | -------------------: | ---------: | ------: | --------------------: | ------: |
| global empirical | real-spine diagnostic | | | | | | |
| user empirical | real-spine diagnostic | | | | | | |
| movie empirical | real-spine diagnostic | | | | | | |
| LSTM | real-spine diagnostic | | | | | | |
| LSTM | synthetic-spine primary | | | | | | |

Keep real-spine diagnostics clearly separate from end-to-end synthetic-spine results.
