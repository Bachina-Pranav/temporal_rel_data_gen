# V4 Diffusion Sampler Diagnosis

This document diagnoses the v4 TabDLM-inspired masked-diffusion attribute sampler before any architectural or training changes. The goal is to separate three possible causes of the weak v4 result:

1. the trained checkpoint,
2. accelerated 25-step inference,
3. text-length/EOS calibration.

No retraining, checkpoint editing, LSTM changes, event-spine changes, evaluator changes, or target-text leakage are part of this diagnosis.

## Compared Runs

### Current LSTM v5.3 Reference

Run directory:

```text
outputs/amazon-toy/conditional_tabdlm_exp5_3_lstm_length_preserving_privacy_sampler/runs/v51_length_preserving_exact_block
```

Known metrics:

| Metric | Value |
|---|---:|
| constraint_violation_rate | 0.000000 |
| shape_error | 0.050096 |
| single_table_c2st_error | 0.634083 |
| text_embedding_c2st_error | 0.548058 |
| trend_error | 0.040950 |
| rating shape error | 0.088874 |
| verified shape error | 0.074238 |
| summary shape error | 0.026637 |
| review_text shape error | 0.060731 |
| real review mean length | 98.560 tokens |
| synthetic review mean length | 94.392 tokens |

### Current V4 Diffusion Run

Run directory:

```text
outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512_full
```

Known metrics:

| Metric | Value |
|---|---:|
| constraint_violation_rate | 0.000803 |
| shape_error | 0.112853 |
| single_table_c2st_error | 0.964441 |
| text_embedding_c2st_error | 0.738696 |
| trend_error | 0.036384 |
| rating shape error | 0.098415 |
| verified shape error | 0.036378 |
| summary shape error | 0.187628 |
| review_text shape error | 0.241844 |
| real review mean length | 98.560 tokens |
| synthetic review mean length | 80.387 tokens |

## Exact V4 Implementation Inspection

Training entry point:

```text
src/scripts/train_v2_full_review_text_tabdlm.py
```

Sampling entry point:

```text
src/scripts/sample_v2_full_review_text_tabdlm.py
```

Reusable sampler:

```text
src/attribute_generation/conditional_tabdlm/sample.py
```

Configuration:

```text
configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml
```

Checkpoint used by the optimized 25-step run:

```text
outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt
```

Command corresponding to `optimized_steps25_topk512_full`:

```bash
python src/scripts/sample_v2_full_review_text_tabdlm.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --checkpoint outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt \
  --synthetic-spine outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512_full/synthetic_review_attrs.csv \
  --num-rows all \
  --sample-batch-size 128 \
  --sampling-steps 25 \
  --text-top-k 512 \
  --inference-dtype bfloat16 \
  --profile \
  --device cuda
```

Training diffusion horizon:

```text
diffusion.timesteps: 50
```

Default sampling horizon in config:

```text
diffusion.sampling_steps: 50
```

The optimized run overrides this to 25. Because the trained horizon is 50, `--sampling-steps full`, `--sampling-steps 50`, `--sampling-steps 100`, and `--sampling-steps 250` all resolve to the same full 50-step reverse schedule unless a checkpoint trained with a longer horizon is used.

Current 25-step inference timestep sequence under uniform spacing:

```text
[50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
```

Full 50-step sequence:

```text
[50, 49, 48, ..., 3, 2, 1]
```

Timestep-spacing policy:

```text
uniform
```

Masking/noise schedule:

```text
mask_schedule: linear
min_mask_prob: 0.05
max_mask_prob: 0.95
```

Token commitment schedule:

- At each selected timestep, the model predicts all still-masked categorical and text positions.
- A reveal probability is computed from the current and next selected timesteps.
- Sampled tokens selected by the reveal mask are committed.
- At the final selected timestep, all remaining positions are committed.

Already committed tokens can be revised:

```text
false
```

The sampler is a commit-and-never-revise masked denoiser. Reducing the number of reverse steps therefore changes not just runtime but also how aggressively tokens are finalized.

Final model pass:

- After the denoising loop, the sampler runs one final forward pass at `diffusion_t = 0`.
- That pass is used by length/EOS enforcement to fill special-token positions when enforcing text lengths.

EOS handling:

- The tokenizer inserts BOS/EOS/PAD during training examples.
- During sampling, text tensors begin with BOS and all remaining positions masked.
- The denoising loop can sample EOS anywhere unless length enforcement later adjusts it.
- Current normal mode uses generated length buckets when configured.
- With `force_eos_after_sampled_length: soft`, EOS is moved to keep decoded text inside the generated length-bucket range where possible.

PAD handling:

- `force_pad_after_eos: true` pads all positions after the selected EOS.
- PAD is also part of the learned text vocabulary.

Field-boundary handling:

- Each schema text field is represented by a separate tensor keyed by field name.
- Summary tokens cannot spill into review text because summary and review_text are sampled in separate tensors.

Summary maximum length:

```text
text.max_length.summary: 32
```

Review-text maximum length:

```text
review_text.max_tokens: auto
review_text.max_feasible_tokens: 512
review_text.min_coverage_rate: 0.99
```

The resolved checkpoint/config determines the actual `review_text` token cap from the training CSV.

Top-k and temperature settings for the optimized run:

```text
text_top_k: 512
temperature: 0.9
top_p: config default
```

Sampling batch size for the optimized run:

```text
sample_batch_size: 128
```

Random seed behavior:

- `sample_from_config` calls the project `set_seed(seed)`.
- The default/configured seed is 42 unless overridden.
- The Python RNG used for length-bucket sampling is seeded from the same seed.

Shared schedule:

- `rating`, `verified`, auxiliary length buckets, `summary`, and `review_text` share the same denoising schedule.
- Categorical targets and text tokens differ in their logits constraints:
  - categorical columns use valid-value masking;
  - length-bucket categorical columns may use calibration ratios;
  - text columns use top-p/top-k token sampling over the text vocabulary.

## New Diagnostic Controls

The sampler now records the actual selected timestep sequence to:

```text
<run_dir>/metadata/timestep_schedule.json
```

It also records:

- requested and resolved sampling steps,
- model forward passes per batch,
- total model forward passes,
- seconds per denoising step,
- length mode,
- GPU model and peak memory when CUDA is available.

Supported length modes:

```text
normal
empirical_length
oracle_length
```

`oracle_length` is labeled:

```text
NOT A VALID GENERATIVE BASELINE
```

It uses only row-level real text lengths, never real text tokens, ratings, verified values, or target attributes.

## Step Ablation Table

Run:

```bash
python src/scripts/run_v4_diffusion_step_ablation.py
```

to populate this table.

<!-- DIFFUSION_STEP_ABLATION_TABLE_START -->

| Steps | Length Mode | Runtime | Rows/s | Shape ↓ | Review KS ↓ | Summary KS ↓ | Text C2ST Error ↓ | Table C2ST Error ↓ | Trend ↓ |
| ----: | ----------- | ------: | -----: | ------: | ----------: | -----------: | ----------------: | -----------------: | ------: |

<!-- DIFFUSION_STEP_ABLATION_TABLE_END -->

## Interpretation Checklist

After the ablation is run, answer:

1. Does quality improve substantially from 25 to 50 steps?
2. Does requesting 100 or 250 steps differ from 50/full? For this checkpoint, it should not, because the training horizon is 50.
3. Does full-step sampling approach the LSTM v5.3 result?
4. Which metrics are sensitive to step count?
5. Which metrics remain poor regardless of step count?
6. Is the verified-related advantage preserved?
7. Does 25-step sampling cause shorter reviews or premature EOS?
8. What is the runtime-quality Pareto frontier?

## Preliminary Diagnosis Before Ablation

The current measured v4 run is much worse than LSTM on summary length, review-text length, text embedding C2ST, and table C2ST, while being slightly better on verified shape and trend error. This suggests that the problem is concentrated in text generation and length/EOS behavior, not in the fixed event spine. The step-count and length-mode ablations are required before deciding whether the primary limitation is accelerated sampling, length calibration, or checkpoint/training quality.
