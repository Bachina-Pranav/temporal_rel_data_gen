# M2 Global-Support Transfer: Amazon-toy and MovieLens-toy

This is a controlled seed-42 transfer experiment for the LSTM v5.3 attribute generator. It derives each run from the existing dataset-specific config and changes only the numeric-valued `rating` target from its categorical output head to the training-derived M2 support head.

The following components remain dataset-specific and unchanged:

- row encoder and stochastic latent;
- categorical heads other than `rating`;
- graph and strict-past temporal conditioning;
- summary and review-text decoders;
- optimizer, training schedule, and early stopping;
- paper-grade and legacy evaluation definitions.

For Amazon, the sampled M2 support ID replaces the old categorical rating ID in the same text-context slot. This preserves the decoder context width and rating conditioning without adding a new conditioning signal. Amazon also keeps the established v5.3 length-preserving text-sampling policy. MovieLens has no text decoder and therefore uses the ordinary optimized fast sampler. No new decoder, prior, or calibration module is introduced.

Support values and support counts are fit from `shared/spines/train_real.csv` only. No rating values are hard-coded into the M2 config.

## Prepared inputs

The runner expects existing files and does not download or rebuild either dataset:

```text
Amazon-toy:
  data/original/rel-amazon-toy/review.csv
  outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv

MovieLens-toy (prepared subset name: movielens_100k):
  data/processed/interaction_benchmarks/movielens_100k/interactions.csv
```

MovieLens uses its explicit 70/15/15 split. Amazon has no explicit `split` column, so the runner uses the same chronological 90/5/5 fallback already used by the LSTM pretokenizer.

## One-command run

From the repository root:

```bash
python -u src/scripts/run_lstm_m2_transfer_experiments.py \
  --stage all \
  --device cuda \
  --sample-batch-size 8192 \
  --minimum-free-disk-gb 5 \
  --skip-existing
```

The command performs, in order:

1. dataset/config/output inventory;
2. derived M2 config creation;
3. split materialization;
4. train-only pretokenization and numerical-support fitting;
5. past-only neighbor-cache preparation;
6. smoke run;
7. full seed-42 training;
8. fixed full-spine sampling;
9. paper-grade and existing attribute diagnostics;
10. baseline-versus-M2 comparison.

Use `--rebuild-precomputed` only if the runner reports that an M2 cache is stale. The M2 cache paths are separate from all previous LSTM caches.

## Staged run

To inspect paths and row counts before training:

```bash
python src/scripts/run_lstm_m2_transfer_experiments.py --stage inventory
```

Then run and summarize separately:

```bash
python -u src/scripts/run_lstm_m2_transfer_experiments.py \
  --stage run \
  --device cuda \
  --sample-batch-size 8192 \
  --minimum-free-disk-gb 5 \
  --skip-existing

python src/scripts/run_lstm_m2_transfer_experiments.py --stage summarize
```

## Outputs

```text
outputs/amazon-toy/lstm_m2_global_support/runs/seed_42/
outputs/movielens-toy/lstm_m2_global_support/runs/seed_42/
outputs/lstm-m2-transfer-amazon-movielens/seed_42/
```

The comparison directory contains:

```text
dataset_inventory.json
model_comparison.csv
numerical_attribute_comparison.csv
dataset_specific_metric_comparison.csv
comparison.json
report.md
```

Display the compact results with:

```bash
cat outputs/lstm-m2-transfer-amazon-movielens/seed_42/report.md
python -m json.tool \
  outputs/lstm-m2-transfer-amazon-movielens/seed_42/comparison.json
```

The detailed run artifacts include the checkpoint, synthetic interaction table, sampling runtime metadata, paper-grade metrics, legacy Amazon diagnostics, and numerical/categorical attribute diagnostics.
