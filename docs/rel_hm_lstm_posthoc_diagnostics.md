# Rel-HM LSTM v5.3 post-hoc diagnostics

This diagnostic pass uses the completed seeds 17, 42 and 73. It does not train
or modify a model.

## Run

Run it in the background so an SSH disconnect does not stop it:

```bash
cd ~/temporal_rel_data_gen
mkdir -p outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1/logs

nohup bash run_rel_hm_lstm_posthoc_diagnostics.sh \
  > outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1/logs/nohup.log \
  2>&1 &

echo $! > outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1/logs/pid
```

The default diagnostic C2ST cap is 10,000 rows per class. To use every held-out
row:

```bash
MAX_C2ST_ROWS=0 bash run_rel_hm_lstm_posthoc_diagnostics.sh
```

Completed phase outputs are reused. Add `--force` to the Python command only
when an intentional full recomputation is needed.

## Monitor

```bash
tail -f outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1/logs/nohup.log
```

```bash
ps -fp "$(cat outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1/logs/pid)"
```

## Results

```bash
OUT=outputs/hm-10k-customers/lstm_v53_three_seed/diagnostics/posthoc_v1

cat "$OUT/diagnosis_report.md"
python -m json.tool "$OUT/diagnosis.json"
column -s, -t < "$OUT/03_c2st_feature_ablation_aggregate.csv" | less -S
column -s, -t < "$OUT/05_price_projection_ablation_aggregate.csv" | less -S
column -s, -t < "$OUT/06_oracle_attribute_ablation_aggregate.csv" | less -S
```

The full output set is:

- `01_current_experiment_audit.json`
- `02_c2st_sanity.json`
- `03_c2st_feature_ablation.csv`
- `04_numerical_support.json`
- `05_price_projection_ablation.csv`
- `06_oracle_attribute_ablation.csv`
- `07_c2st_feature_importance.json`
- `08_conditional_support.json`
- `diagnosis.json`
- `diagnosis_report.md`

The diagnosis report explicitly answers whether price support, entity
conditioning, or sales channel explains the observed C2ST; whether C2ST passed
its controls; and which single architecture change is justified next.
