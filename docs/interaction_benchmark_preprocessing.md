# Interaction Benchmark Preprocessing

These benchmarks synthesize one designated temporal interaction table while retaining source and destination entity tables as fixed support tables.

They do not claim to generate arbitrary complete multi-table databases. The method scope is:

```text
source entity -> temporal interaction table -> destination entity
```

## Dataset Scope

| Dataset | Interaction | Source | Destination | Generated attributes |
| ------- | ----------- | ------ | ----------- | -------------------- |
| Amazon-toy | review | customer | product | rating, verified, summary, review_text |
| MovieLens-100K-induced | rating | user | movie | rating |
| Yelp-100K-induced | review | user | business | stars, useful, funny, cool, review_text |
| RetailRocket-100K-induced | event | visitor | item | event_type |
| H&M-100K-induced | transaction | customer | article | price, sales_channel_id |

Amazon-toy is left unchanged. The new benchmark code lives under `src/data_preprocessing/interaction_datasets/`.

## Acquisition

MovieLens 25M can be downloaded directly from GroupLens:

```bash
python src/scripts/download_interaction_datasets.py \
  --dataset movielens \
  --raw-root data/raw
```

Yelp requires manual license acceptance or Kaggle credentials. A local archive is supported:

```bash
python src/scripts/download_interaction_datasets.py \
  --dataset yelp \
  --archive /path/to/yelp_dataset.tar \
  --raw-root data/raw
```

RetailRocket and H&M use Kaggle datasets/competitions when credentials and terms acceptance are available:

```bash
python src/scripts/download_interaction_datasets.py \
  --datasets retailrocket hm \
  --raw-root data/raw
```

The downloader records acquisition status and hashes in `download_metadata.json`. Archive extraction rejects absolute paths and path traversal members.

## Canonical Schemas

Adapters keep physical source column names local. Shared code works through canonical roles:

- source foreign key
- destination foreign key
- timestamp
- generated categorical attributes
- generated numerical attributes
- generated text attributes

The schema metadata supports:

```text
categorical
ordinal_categorical
boolean
continuous_numerical
count_numerical
text
datetime
foreign_key
```

## Source-Entity-Induced Selection

The central subset rule is:

```text
Select source entities, preserve all interactions for those source entities,
and include every referenced destination entity.
```

The generic selector:

1. streams the full interaction source and counts complete source histories,
2. deterministically orders source entities using the seed,
3. greedily selects complete histories near the target count,
4. performs local add/remove refinement,
5. materializes every interaction for selected source IDs,
6. filters source and destination support tables.

Exact 100,000 rows may be impossible without splitting a source history. In that case, the closest complete-history subset is retained and the deviation is recorded in `subset_manifest.json`.

## Output Layout

Processed subsets are written to:

```text
data/processed/interaction_benchmarks/<dataset>_100k/
```

Each subset contains:

```text
interactions.csv
schema.yaml
subset_manifest.json
statistics.json
statistics.md
validation_report.json
README.md
```

plus dataset-specific support tables such as `users.csv`, `movies.csv`, `businesses.csv`, `items.csv`, `customers.csv`, and `articles.csv`.

## Chronological Splits

The subset builder sorts by timestamp and event ID, then writes split labels into `interactions.csv`:

- earliest 70%: `train`
- next 15%: `validation`
- latest 15%: `test`

No random row split is used.

## Numerical Modeling Choices

The LSTM pipeline now supports schema-driven numerical fields.

Continuous numerical fields use a Gaussian head over standardized training-split values.

Count numerical fields use a stochastic `log1p` Gaussian fallback by default:

```text
raw count -> log1p -> standardize -> Gaussian NLL
```

Sampling inverts the transformation, clips to training-derived bounds, and rounds count fields to nonnegative integers. This is intentionally minimal and robust; a zero-inflated negative binomial can be added later as a stronger count model.

## Validation

Validation checks include:

- nonempty outputs,
- unique event IDs,
- parseable timestamps,
- perfect source and destination FK coverage,
- complete selected source histories,
- train/validation/test split coverage,
- generated attribute presence,
- dataset-specific domain checks.

## Commands

Download:

```bash
python src/scripts/download_interaction_datasets.py \
  --datasets movielens retailrocket hm \
  --raw-root data/raw
```

Build subsets:

```bash
python src/scripts/build_interaction_subsets.py \
  --datasets movielens yelp retailrocket hm \
  --raw-root data/raw \
  --processed-root data/processed/interaction_benchmarks \
  --target-interactions 100000 \
  --allowed-relative-error 0.01 \
  --seed 42
```

Validate:

```bash
python src/scripts/validate_interaction_subsets.py \
  --processed-root data/processed/interaction_benchmarks
```

Summarize:

```bash
python src/scripts/summarize_interaction_benchmarks.py \
  --processed-root data/processed/interaction_benchmarks
```

LSTM smoke test:

```bash
python src/scripts/validate_lstm_dataset_configs.py \
  --datasets movielens_100k yelp_100k retailrocket_100k hm_100k \
  --run-forward-pass \
  --run-loss-pass \
  --run-sampling-smoke-test
```

## Known Limitations

This implementation does not train full models. It provides download handlers, subset construction, validation, schema/config definitions, numerical LSTM heads, and fixture-based smoke tests. Yelp, RetailRocket, and H&M may require manual license acceptance or Kaggle credentials before raw data can be processed.
