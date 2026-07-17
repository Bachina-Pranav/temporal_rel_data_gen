# Rel-H&M LSTM Attribute Generator Preparation

## Goal

Prepare an Amazon-toy-style induced subdatabase for Rel-H&M and train the schema-driven v5.3 LSTM attribute generator to synthesize only the temporal `transactions` table.

The fixed support/entity tables are:

- `customers.csv`
- `articles.csv`

The generated event table is:

- `interactions.csv`, mapped from RelBench `transactions.csv`

Event spine:

- `customer_id`
- `article_id`
- `event_time`

Generated attributes:

- `price`
- `sales_channel_id`

There are no text fields for this experiment.

## Raw Data

Expected RelBench export files under `data/original/rel-hm/`:

- `customer.csv`
- `article.csv`
- `transactions.csv`

If those files are missing, the builder calls:

```python
relbench.datasets.get_dataset("rel-hm")
```

and exports the RelBench database with `upto_test_timestamp=True`.

The local checkout used to prepare this code did not contain those RelBench CSV files, so raw row counts and schema values are populated by the builder at run time in:

- `data/processed/interaction_benchmarks/hm_10k_customers/subset_manifest.json`
- `data/processed/interaction_benchmarks/hm_10k_customers/statistics.json`
- `data/processed/interaction_benchmarks/hm_10k_customers/validation_report.json`

The builder records:

- raw paths
- raw schemas
- customer/article/transaction row counts
- active customer count
- transaction time span
- required-field missing counts
- `price` dtype, quantiles, zero rate, and range
- `sales_channel_id` domain and counts
- duplicate-looking transaction rows observed within raw chunks

## Induced Subdatabase Rule

The subset is source-entity induced:

1. Scan RelBench `transactions.csv`.
2. Identify customers with at least one transaction.
3. Sort active customer IDs deterministically.
4. Uniformly sample exactly 10,000 active customers with seed 42.
5. Retain every transaction for those customers.
6. Retain every referenced article.
7. Retain exactly those 10,000 customer rows.
8. Preserve duplicate-looking transaction rows.

This intentionally does not target 100,000 interactions. The final transaction count is whatever complete histories produce.

## Canonical Processed Schema

Output directory:

```text
data/processed/interaction_benchmarks/hm_10k_customers/
```

Files:

- `customers.csv`
- `articles.csv`
- `interactions.csv`
- `schema.yaml`
- `subset_manifest.json`
- `selected_customer_ids.txt`
- `statistics.json`
- `statistics.md`
- `validation_report.json`
- `README.md`

`interactions.csv` columns:

```text
event_id
customer_id
article_id
event_time
price
sales_channel_id
split
```

`event_time` is parsed from raw `t_dat`. Rel-H&M dates have day-level granularity, so the builder preserves dates and does not invent times of day.

`event_id` is deterministic: `hm-transaction-<raw_row_number>`. It remains unique even when all transaction attribute values are identical.

Splits are chronological 70/15/15, sorted by `event_time` then `event_id`.

## Model Preparation

Config:

```text
configs/attribute_generation/lstm_hm_10k_customers.yaml
```

The model reuses the optimized v5.3 LSTM path:

- fixed-step training mode
- temporal-stratified sampling
- pretokenized array path
- cached past-only neighbor graph path
- bulk graph-context batching
- mixed precision
- gradient accumulation
- validation batch cap
- checkpoint resume support
- no text decoders, no tokenizer vocabulary, no text-length heads

Architecture:

```text
customer_id, article_id, event_time, past-only graph context
  -> shared event-context encoder
  -> shared stochastic row latent
  -> categorical head: sales_channel_id
  -> Gaussian numerical head: price
```

`price` uses the existing stochastic Gaussian numerical head. The numerical transformation is fitted from the training split only and saved in `numerical_metadata.json` and checkpoint metadata.

`sales_channel_id` uses a train-derived categorical vocabulary and decodes exact observed values.

Graph conditioning is leakage-safe:

- only past transactions are used
- current `price` and `sales_channel_id` are forbidden
- future transactions are forbidden
- target attributes are not graph inputs

## Commands

Build the subset:

```bash
python src/scripts/build_interaction_subsets.py \
  --dataset hm \
  --raw-root data/raw \
  --relbench-root data/original \
  --processed-root data/processed/interaction_benchmarks \
  --num-source-entities 10000 \
  --seed 42 \
  --chunk-size 500000
```

If `data/original/rel-hm/customer.csv`, `article.csv`, and `transactions.csv` are not present, this command now attempts the RelBench download/cache automatically.

The `--archive` fallback exists only for the legacy Kaggle H&M CSV layout. It is not the primary path for this experiment:

```bash
python src/scripts/build_interaction_subsets.py \
  --dataset hm \
  --raw-root data/raw \
  --relbench-root data/original \
  --processed-root data/processed/interaction_benchmarks \
  --num-source-entities 10000 \
  --seed 42 \
  --chunk-size 500000 \
  --archive /path/to/h-and-m-personalized-fashion-recommendations.zip
```

Validate the subset:

```bash
python src/scripts/validate_interaction_subsets.py \
  --datasets hm_10k_customers \
  --processed-root data/processed/interaction_benchmarks
```

Precompute optimized LSTM inputs:

```bash
python src/scripts/pretokenize_single_event_table_text_fields.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --real-table data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --output-dir data/processed/interaction_benchmarks/hm_10k_customers/pretokenized_lstm \
  --chunk-size 500000

python src/scripts/precompute_temporal_neighbor_cache.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --real-table data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --output-dir data/processed/interaction_benchmarks/hm_10k_customers/neighbor_cache \
  --chunk-size 500000
```

Smoke-test the loader, forward pass, loss pass, optimizer step, numerical inverse transform, graph context, and sampling:

```bash
python src/scripts/validate_lstm_dataset_configs.py \
  --datasets hm_10k_customers \
  --run-forward-pass \
  --run-loss-pass \
  --run-optimizer-step \
  --run-sampling-smoke-test \
  --device cuda
```

Materialize real split spines:

```bash
python src/scripts/materialize_interaction_lstm_splits.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --output-dir outputs/hm-10k-customers/lstm_v53/spines
```

Profile a short optimized run:

```bash
python src/scripts/profile_lstm_joint_training_step.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --pretokenized-dir data/processed/interaction_benchmarks/hm_10k_customers/pretokenized_lstm \
  --neighbor-cache-dir data/processed/interaction_benchmarks/hm_10k_customers/neighbor_cache \
  --output-dir outputs/hm-10k-customers/lstm_v53/profile \
  --physical-batch-size 512 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 5 \
  --profile-steps 20 \
  --validation-max-batches 5 \
  --device cuda \
  --mixed-precision
```

Full training:

```bash
python src/scripts/train_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --pretokenized-dir data/processed/interaction_benchmarks/hm_10k_customers/pretokenized_lstm \
  --neighbor-cache-dir data/processed/interaction_benchmarks/hm_10k_customers/neighbor_cache \
  --output-dir outputs/hm-10k-customers/lstm_v53 \
  --physical-batch-size 512 \
  --gradient-accumulation-steps 4 \
  --validation-max-batches 50 \
  --device cuda \
  --mixed-precision
```

Resume training:

```bash
python src/scripts/train_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --pretokenized-dir data/processed/interaction_benchmarks/hm_10k_customers/pretokenized_lstm \
  --neighbor-cache-dir data/processed/interaction_benchmarks/hm_10k_customers/neighbor_cache \
  --output-dir outputs/hm-10k-customers/lstm_v53 \
  --resume-from outputs/hm-10k-customers/lstm_v53/checkpoints/last.pt \
  --device cuda \
  --mixed-precision
```

REAL-SPINE ATTRIBUTE DIAGNOSTIC:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --checkpoint outputs/hm-10k-customers/lstm_v53/checkpoints/best.pt \
  --synthetic-spine outputs/hm-10k-customers/lstm_v53/spines/test_spine.csv \
  --output outputs/hm-10k-customers/lstm_v53/samples/real_spine_test_attribute_diagnostic.csv \
  --num-rows all \
  --device cuda \
  --seed 42
```

Primary full-spine attribute generation:

```bash
python src/scripts/sample_lstm_joint_full_review_text.py \
  --config configs/attribute_generation/lstm_hm_10k_customers.yaml \
  --checkpoint outputs/hm-10k-customers/lstm_v53/checkpoints/best.pt \
  --synthetic-spine data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --output outputs/hm-10k-customers/lstm_v53/samples/full_synthetic_interactions.csv \
  --num-rows all \
  --device cuda \
  --seed 42
```

Paper-grade evaluation:

```bash
python src/scripts/evaluate_single_event_table_paper_metrics.py \
  --config configs/evaluation/single_event_table_paper_metrics_hm_10k_customers.yaml \
  --real-table data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --synthetic-table outputs/hm-10k-customers/lstm_v53/samples/full_synthetic_interactions.csv \
  --output-dir outputs/hm-10k-customers/lstm_v53/evaluation/paper_grade \
  --seed 42
```

Simple training-only baselines for later comparison:

```bash
python src/scripts/run_nontext_attr_baselines.py \
  --real-reviews data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --synthetic-spine data/processed/interaction_benchmarks/hm_10k_customers/interactions.csv \
  --output-dir outputs/hm-10k-customers/baselines \
  --customer-id-col customer_id \
  --product-id-col article_id \
  --timestamp-col event_time \
  --cat-cols sales_channel_id \
  --num-cols price \
  --seed 42
```

Tests:

```bash
pytest -q \
  tests/test_interaction_benchmarks.py \
  tests/test_pretokenized_dataset_loading.py \
  tests/test_movielens_lstm_compatibility.py
```

## Remaining Notes

- RelBench `rel-hm` files were not present in this local checkout during preparation.
- Exact raw counts and Rel-H&M price/channel domains should be read from the generated `subset_manifest.json` after running preprocessing on the VM.
- The static paper-metrics config intentionally does not hard-code a sales-channel domain; the training vocabulary is derived from the training split.
