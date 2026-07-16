# Conditional TABDLM Diffusion Optimization Report

This report covers the diffusion-based attribute generator that can generate complete review rows, including `review_text`.

Target experiment:

```text
conditional_tabdlm_exp4_v2_full_review_text
```

Primary config:

```text
configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml
```

---

## Current Architecture

The model is `ConditionalTABDLM`, a masked denoising transformer conditioned on:

- Foreign-key columns such as `customer_id` and `product_id`
- Datetime columns such as `review_time`
- Optional structure-only temporal graph context

It generates:

- Categorical attributes: `rating`, `verified`
- Auxiliary categorical attributes: `summary_length_bucket`, `review_text_length_bucket`
- Text attributes: `summary`, `review_text`

Text is generated directly as token IDs from a simple schema-driven tokenizer. It is not a latent-text diffusion model.

For exp4, `review_text` can resolve to a long fixed token length, up to the configured feasible cap. In the current Amazon-toy config this cap is `512` tokens.

---

## Sampling Flow

At sampling time, the model starts with masked target fields:

```text
rating = [MASK]
verified = [MASK]
summary_length_bucket = [MASK]
review_text_length_bucket = [MASK]
summary tokens = [BOS] [MASK] ... [MASK]
review_text tokens = [BOS] [MASK] ... [MASK]
```

For each denoising step:

1. Encode row conditions.
2. Optionally compute graph context.
3. Run the transformer over the full target sequence.
4. Sample categorical and text logits.
5. Reveal a subset of still-masked positions.

With 50 denoising steps and 512 review-text positions, sampling 50,000 rows requires many large transformer forward passes plus many token-sampling operations.

---

## Main Bottlenecks

The main expected sampling bottlenecks are:

- Transformer denoising over long review-text sequences.
- Full-vocabulary nucleus sampling over a vocabulary that can be around 30k tokens.
- Repeating the denoising loop for all configured diffusion timesteps.
- CPU postprocessing and detokenization after sampling.
- Per-batch condition encoding and graph-context construction.

The main expected training bottlenecks are:

- Dataset-side tokenization and auxiliary length-bucket encoding.
- Transformer forward/backward passes over long text sequences.
- Validation, length calibration, and checkpoint writes every epoch.
- DataLoader throughput when `num_workers=0`.

---

## Implemented Safe Optimizations

The baseline behavior remains available.

The following changes are configurable:

- `--sample-batch-size`
- `--sampling-steps`
- `--timestep-spacing`
- `--inference-dtype float32|float16|bfloat16`
- `--compile-model`
- `--text-top-k`
- `--profile`
- `--profile-output`

The sampler now records profiling fields for:

- Model loading
- Spine loading
- Graph history construction
- Condition encoding
- Initial masked tensor construction
- Denoising loop
- Denoising step forward time
- Final forward pass
- Length enforcement
- Categorical decoding
- Text decoding
- Postprocessing
- CSV writing
- Peak CUDA memory

---

## Reduced-Step Masked Sampling

This model is a masked categorical/text diffusion model, not a Gaussian DDPM.

Because of that, DDIM is not directly applicable without changing the model parameterization. The implemented acceleration is instead a reduced-step masked denoising schedule.

For the full schedule:

```text
50, 49, 48, ..., 1
```

the reveal probability is unchanged:

```text
p(reveal at step s | still masked) = 1 / s
```

For a reduced schedule:

```text
50, 38, 25, 13, 1
```

the reveal probability is:

```text
p(reveal) = 1 - next_step / current_step
```

This preserves the old rule when using all steps, while allowing faster ablations such as 25, 10, or 5 steps.

---

## Optional Top-K Text Sampling

The old text sampler performs full-vocabulary top-p filtering.

For large vocabularies and long review text, this can be expensive because it sorts many logits repeatedly.

The new `text_top_k` option limits sampling to the top-k tokens before applying top-p filtering inside that subset.

This changes sampling semantics, so it is opt-in and should be evaluated as a speed-quality ablation.

Recommended first values:

```text
text_top_k = null
text_top_k = 512
text_top_k = 256
text_top_k = 128
```

---

## Training Optimizations

The training loop now exposes:

- `pin_memory`
- `persistent_workers`
- `prefetch_factor`
- `allow_tf32`
- `fused_adamw`
- `compile_model`
- `length_calibration_interval`
- `checkpoint_interval`
- `save_last_every_epoch`

Defaults preserve the old behavior where practical. For speed experiments, start with:

```yaml
training:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  allow_tf32: true
  fused_adamw: true
  length_calibration_interval: 2
  checkpoint_interval: 2
```

Use `compile_model: true` only as an ablation because compile overhead and compatibility vary by GPU/PyTorch version.

---

## Baseline Sampling Command

This reproduces the old full-step behavior:

```bash
python src/scripts/sample_v2_full_review_text_tabdlm.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --checkpoint outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt \
  --synthetic-spine outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/baseline_50k/synthetic_review_attrs.csv \
  --num-rows 50000 \
  --sample-batch-size 64 \
  --sampling-steps 50 \
  --inference-dtype float32 \
  --profile \
  --profile-output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/baseline_50k/runtime_diffusion_sampling.json \
  --device cuda
```

---

## Balanced Optimized Sampling Command

Start here for the first serious speed-quality run:

```bash
python src/scripts/sample_v2_full_review_text_tabdlm.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --checkpoint outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt \
  --synthetic-spine outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512/synthetic_review_attrs.csv \
  --num-rows 50000 \
  --sample-batch-size 128 \
  --sampling-steps 25 \
  --timestep-spacing uniform \
  --text-top-k 512 \
  --inference-dtype bfloat16 \
  --profile \
  --profile-output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512/runtime_diffusion_sampling.json \
  --device cuda
```

If bfloat16 is unsupported, the sampler falls back safely to float32.

---

## Fast Exploratory Sampling Command

Use this only for quick iteration:

```bash
python src/scripts/sample_v2_full_review_text_tabdlm.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --checkpoint outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt \
  --synthetic-spine outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --output outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/fast_steps10_topk256/synthetic_review_attrs.csv \
  --num-rows 50000 \
  --sample-batch-size 128 \
  --sampling-steps 10 \
  --text-top-k 256 \
  --inference-dtype bfloat16 \
  --profile \
  --device cuda
```

This is expected to be faster, but it needs quality evaluation before being used as a main result.

---

## Benchmark Command

Run a small grid first:

```bash
python src/scripts/benchmark_diffusion_sampling.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --checkpoint outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt \
  --synthetic-spine outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --output-dir outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/diffusion_sampling_benchmark \
  --num-rows 1000 \
  --sampling-steps 50 25 10 \
  --sample-batch-sizes 64 128 \
  --inference-dtypes float32 bfloat16 \
  --text-top-k none 512 256 \
  --device cuda
```

Then scale the best candidates to 50,000 rows.

---

## Quality Evaluation Command

Evaluate any generated sample with the existing evaluator:

```bash
python src/scripts/evaluate_v2_full_review_text_tabdlm.py \
  --config configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml \
  --synthetic-reviews outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512/synthetic_review_attrs.csv \
  --output-dir outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/optimized_steps25_topk512/evaluation
```

Compare speed and quality together. Do not choose the fastest setting unless quality remains acceptable.

---

## Recommended Ablations

Maximum quality:

```text
sampling_steps=50
text_top_k=null
inference_dtype=float32
```

Balanced:

```text
sampling_steps=25
text_top_k=512
inference_dtype=bfloat16
sample_batch_size=128
```

Fast exploration:

```text
sampling_steps=10
text_top_k=256
inference_dtype=bfloat16
sample_batch_size=128
```

---

## Remaining Bottlenecks

The largest remaining algorithmic bottleneck is direct diffusion over long token sequences.

Even with fewer steps, the transformer still processes the full fixed-length `review_text` window. A stronger architectural experiment would be latent-text diffusion or a hybrid model that diffuses attributes and length/style controls while using a faster decoder for text.

That would be a separate model change and should not be mixed into these engineering/sampling ablations.
