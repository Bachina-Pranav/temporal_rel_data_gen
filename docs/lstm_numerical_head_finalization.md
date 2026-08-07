# LSTM Numerical-Head Finalization

This workflow performs the final numerical-head iteration without completing
the dominated M1 and M3 seed grids. Run it from the repository root in the
`reldiff` environment.

## 1. Comparability and M0

```bash
cd ~/temporal_rel_data_gen
conda activate reldiff

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage comparability

python -m json.tool \
  outputs/hm-10k-customers/lstm_numerical_heads/comparability_report.json

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage baseline-full \
  --device cuda \
  --sample-batch-size 8192
```

`baseline-full` uses `--skip-existing` internally. It retains the completed
comparable M0 seed and runs only missing seeds.

## 2. Three-Seed M2

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage full \
  --promote-variants M2_global_support \
  --skip-existing \
  --device cuda \
  --sample-batch-size 8192

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage summarize
```

The mean, sample standard deviation, runtime, memory, and parameter results are
written to:

```text
outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/
```

## 3. M2 Calibration and M2C

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage m2-support-analysis \
  --skip-existing \
  --device cuda \
  --sample-batch-size 8192

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage m2-posthoc \
  --device cuda \
  --sample-batch-size 8192
```

M2C estimates its target marginal from training data, estimates the M2
distribution from generated validation rows, selects lambda using validation,
and evaluates test data once. Only the selected corrected checkpoint is kept.

## 4. Prior-Residual Screen

First run cheap smoke tests:

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage smoke \
  --variants \
    M2P_R0_global_prior \
    M2P_R1_weak_residual \
    M2P_R2_moderate_residual \
    M2P_R3_full_residual \
  --smoke-rows 256 \
  --device cuda
```

Then run the four controlled variants on seed 42:

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage one-seed \
  --variants \
    M2P_R0_global_prior \
    M2P_R1_weak_residual \
    M2P_R2_moderate_residual \
    M2P_R3_full_residual \
  --skip-existing \
  --device cuda \
  --sample-batch-size 8192

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage summarize
```

Inspect:

```bash
column -s, -t \
  < outputs/hm-10k-customers/lstm_numerical_heads/results_comparison/per_model_per_seed_metrics.csv \
  | less -S
```

Promote only the best one or two variants. Replace the example variant below
with the validation-selected winner:

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage full \
  --promote-variants M2P_R1_weak_residual \
  --skip-existing \
  --device cuda \
  --sample-batch-size 8192

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage summarize
```

Do not run additional M1 or M3 seeds. Implement the optional low-capacity
residual only if this screen shows that a nonzero residual is consistently
useful.

## 5. Router and Regression Audit

```bash
python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage router-regression

python -m json.tool \
  outputs/lstm_numerical_router_regressions/regression_audit.json
```

MovieLens and Amazon-toy currently declare `rating` as categorical. The
numerical router therefore leaves those models exactly unchanged rather than
silently reclassifying a numeric-coded categorical target. Rel-HM `price` is
profiled from the training split and should route to `support_prior`.

## 6. Tests and Freeze Report

```bash
pytest -q

python src/scripts/run_lstm_numerical_head_experiments.py \
  --stage freeze-report \
  --tests-passed
```

The final artifacts are:

```text
outputs/hm-10k-customers/lstm_numerical_heads/architecture_freeze/model_freeze_decision.json
outputs/hm-10k-customers/lstm_numerical_heads/architecture_freeze/model_freeze_report.md
outputs/hm-10k-customers/lstm_numerical_heads/architecture_freeze/cross_dataset_architecture_table.csv
```

The freeze report uses `passed`, `failed`, and `not_evaluable` states. Missing
runs or metrics never become false failures.
