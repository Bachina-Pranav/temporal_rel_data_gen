# LSTM Numerical-Head Experiment Protocol

This protocol compares the existing Rel-HM `lstm_v53` numerical path (M0)
with destination-conditioned and support-aware alternatives (M1-M4). Support,
type inference, calibration maps, and priors are fitted from the training split
only. The test split is used only for final evaluation.

## Outputs

- Architecture report: `docs/lstm_v53_numerical_architecture.md`
- Q0-Q4 diagnostics:
  `outputs/hm-10k-customers/lstm_numerical_heads/calibration_q0_q4/`
- Generated variant configs:
  `outputs/hm-10k-customers/lstm_numerical_heads/variant_configs/`
- M1-M4 runs:
  `outputs/hm-10k-customers/lstm_numerical_heads/<variant>/`
- Consolidated decision:
  `outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/`

Each run retains the existing checkpoint, sampling, paper-metrics, and
attribute-diagnostics layout. Numerical type inference is saved in both the
resolved configuration and `metadata/numerical_type_inference.json`.

## 1. Write And Inspect Configs

```bash
cd ~/temporal_rel_data_gen
conda activate reldiff

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage write-configs

for file in outputs/hm-10k-customers/lstm_numerical_heads/variant_configs/*.yaml; do
  echo "===== $file ====="
  grep -A40 '^numerical_heads:' "$file"
done
```

## 2. Run Q0-Q4 Without Retraining

The known aligned real-numerical oracle C2ST is supplied only for reporting the
fraction of oracle improvement recovered. It is not used to fit a mapping.

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage calibration \
  --real-numerical-oracle-c2st 0.1927

cat outputs/hm-10k-customers/lstm_numerical_heads/calibration_q0_q4/calibration_interpretation.json
```

## 3. Refresh M0 Diagnostics Without Retraining

This preserves the existing M0 checkpoints and old metric files while adding
the same support, group-C2ST, and paired-context diagnostics used for M1-M4.

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage baseline-eval \
  --device cuda
```

## 4. Smoke-Test M1 And M2

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage smoke \
  --variants M1_destination_continuous M2_global_support \
  --smoke-rows 256 \
  --device cuda
```

Verify that sampling validation passes, losses are finite, support outputs are
valid numerical values, and gradient norms are nonzero.

## 5. Smoke-Test M3 And M4

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage smoke \
  --variants M3_destination_support M4_destination_support_prior \
  --smoke-rows 256 \
  --device cuda
```

## 6. Run One Full Seed

This trains only seed 42 for M1-M4.

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage one-seed \
  --variants \
    M1_destination_continuous \
    M2_global_support \
    M3_destination_support \
    M4_destination_support_prior \
  --device cuda \
  --skip-existing

cat outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/final_decision.json
```

Inspect each seed-42 paper report, attribute diagnostics, numerical type
report, training log, and `evaluation/numerical_context_usage.json`. Eliminate
clearly inferior variants before the next step.

## 7. Promote Only Selected Variants To Three Seeds

Replace the example variant list with the strongest one or two variants from
the seed-42 comparison. The `--promote-variants` argument is mandatory.

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage full \
  --promote-variants \
    M3_destination_support \
    M4_destination_support_prior \
  --device cuda \
  --skip-existing
```

## 8. Rebuild The Final Report

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage summarize

cat outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/report.md
cat outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/final_decision.json
```

The reporter refuses to recommend replacing M0 unless a candidate and M0 have
all three seeds and the required validity, fidelity, consistency, trend,
categorical-regression, and runtime checks pass.
