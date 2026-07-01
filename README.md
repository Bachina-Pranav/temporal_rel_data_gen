# RelDiff: Relational Data Generative Modeling with Graph-Based Diffusion Models

<p align="center">
  <a href="https://github.com/ValterH/RelDiff/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <!-- <a href="">
    <img alt="Openreview" src="">
  </a> -->
  <!-- <a href="">
    <img alt="Paper URL" src="https://img.shields.io/badge/-B31B1B.svg">
  </a> -->
</p>


This repository provides the official implementation of the paper "RelDiff: Relational Data Generative Modeling with Graph-Based Diffusion Models".

## Latest Update

- [2025.]：Our code is at the final stage of cleaning up. Please check back soon for its release!

## Introduction

<div align="center">
  <img src="images/reldiff_flowchart.svg" alt="RelDiff Pipeline" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: A high-level overview of RelDiff</a></em></p>
</div>
RelDiff is a novel generative framework for synthesizing relational databases with arbitrarily complex schemas, achieving high fidelity and utility. Its key innovations include:

1) A principled framework for **generating foreign key structures** in relational databases, incorporating hard constraints for **referential integrity** via Bayesian stochastic block models.
2) A **joint diffusion model** for synthesizing mixed-type attributes, utilizing GNNs to capture global **inter-table dependencies**.
3) Explicitly modeling **dimension tables** as a distinct data type and defining our **diffusion model in data space**.

The schema of RelDiff is presented in the figure above. <!--  For more details, please refer to [our paper](). -->

## ContinuousTimeTemporalSBMGenerator

`ContinuousTimeTemporalSBMGenerator` is a structural-only temporal event
generator for Amazon-style review tables. It generates only the event spine:

```text
customer_id, product_id, review_time
```

It reuses RelDiff's type-constrained, degree-corrected SBM idea to infer
customer/product blocks on the aggregate customer-product graph, then extends
the static structure model to continuous-time review events. Conceptually, it
samples from a temporal degree-corrected SBM:

```text
lambda_ui(tau) = omega_blockpair(tau) * theta_user(tau) * phi_product(tau)
```

Timestamps are sampled from learned continuous KDE-style intensities, not from
uniform assignments inside coarse temporal bins. Attribute generation is
intentionally not implemented yet.

Example:

```bash
python src/scripts/generate_amazon_temporal_sbm.py \
  --customers data/amazon-toy/customer.csv \
  --products data/amazon-toy/product.csv \
  --reviews data/amazon-toy/review.csv \
  --output outputs/amazon-toy/continuous_time_temporal_sbm/synthetic_review.csv \
  --seed 42 \
  --debug-dir outputs/amazon-toy/continuous_time_temporal_sbm/debug
```

To evaluate the generated event spine:

```bash
python src/scripts/evaluate_temporal_sbm_event_spine.py \
  --real-reviews data/amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/continuous_time_temporal_sbm/synthetic_review.csv
```

## ContinuousTime2KSBMPlusGenerator / ct_2k_sbm_plus

`ContinuousTime2KSBMPlusGenerator` is an improved continuous-time temporal SBM
structural generator. It keeps the older `continuous_time_temporal_sbm` method
available, reuses RelDiff's SBM block inference, and generates only:

```text
customer_id, product_id, review_time
```

The method preserves block-pair event cardinalities and uses microcanonical
customer/product endpoint stubs inside each block pair, which is closer in
spirit to RelDiff's 2K+SBM structure generator. Timestamps are still sampled
from continuous learned KDE intensities, then passed through a timestamp
granularity model so date-only datasets do not receive fake hour/min/sec values.
Attribute generation is intentionally not implemented yet.

Example:

```bash
python src/scripts/generate_amazon_ct_2k_sbm_plus.py \
  --customers data/amazon-toy/customer.csv \
  --products data/amazon-toy/product.csv \
  --reviews data/amazon-toy/review.csv \
  --output outputs/amazon-toy/ct_2k_sbm_plus/synthetic_review.csv \
  --seed 42 \
  --stub-pairing time_sorted \
  --debug-dir outputs/amazon-toy/ct_2k_sbm_plus/debug
```

Evaluate:

```bash
python src/scripts/evaluate_ct_2k_sbm_plus.py \
  --real-reviews data/amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/ct_2k_sbm_plus/synthetic_review.csv
```

## ContinuousTime2KSBMTemporalStubsGenerator / ct_2k_sbm_temporal_stubs

`ContinuousTime2KSBMTemporalStubsGenerator` is the third structural event-spine
generator. It keeps `continuous_time_temporal_sbm` and `ct_2k_sbm_plus` intact,
reuses RelDiff's SBM block inference, and generates temporal review events by
preserving three stub multisets inside each SBM block pair:

```text
customer stubs
product stubs
timestamp stubs
```

The method preserves customer/product degree counts and block-pair timestamp
distributions while using local temporal-window shuffling to reduce direct edge
memorization. Timestamp granularity is preserved, so date-only datasets do not
receive fake hour/min/sec values. Attribute generation is intentionally not
implemented yet.

Example for the rel-amazon toy data:

```bash
python src/scripts/generate_amazon_ct_2k_sbm_temporal_stubs.py \
  --customers data/original/rel-amazon-toy/customer.csv \
  --products data/original/rel-amazon-toy/product.csv \
  --reviews data/original/rel-amazon-toy/review.csv \
  --output outputs/amazon-toy/ct_2k_sbm_temporal_stubs/synthetic_review.csv \
  --seed 42 \
  --stub-pairing temporal_window_shuffle \
  --timestamp-stub-mode reuse_block_pair_timestamps \
  --avoid-real-edge-prob 0.95 \
  --debug-dir outputs/amazon-toy/ct_2k_sbm_temporal_stubs/debug
```

Evaluate:

```bash
python src/scripts/evaluate_ct_2k_sbm_temporal_stubs.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/ct_2k_sbm_temporal_stubs/synthetic_review.csv
```

## ContinuousTime2KSBMTemporalKDEStubsGenerator / ct_2k_sbm_temporal_kde_stubs

`ContinuousTime2KSBMTemporalKDEStubsGenerator` is the generated-timestamp
variant of the temporal stub generator. It preserves customer/product degree
stubs and SBM block-pair counts exactly, but samples timestamps from a learned
block-pair temporal intensity model instead of reusing exact timestamp stubs.

Use `ct_2k_sbm_temporal_stubs` as the microcanonical upper-bound baseline when
you want exact timestamp-multiset preservation. Use
`ct_2k_sbm_temporal_kde_stubs` when you want the relaxed, generative temporal
stub method.

Example:

```bash
python src/scripts/generate_ct_2k_sbm_temporal_kde_stubs.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --output-dir outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --sbm-block-level bottom \
  --timestamp-model auto \
  --pairing-mode temporal_window_shuffle \
  --avoid-real-edge-prob 0.95 \
  --seed 42
```

Evaluate all four structural methods together:

```bash
python src/scripts/evaluate_all_structure_methods.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --outputs-root outputs/amazon-toy \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --output-json outputs/amazon-toy/all_structure_metrics.json \
  --output-csv outputs/amazon-toy/all_structure_metrics.csv
```

Block-pair diagnostics require customer/product block assignment metadata. When
`--debug-dir` is available, the evaluator looks for `customer_blocks.csv` and
`product_blocks.csv` there. If these files are missing, block-pair KS and
block-pair count metrics are skipped with a warning instead of falling back to a
fake single block pair.

Standalone block diagnostics:

```bash
python src/scripts/diagnose_temporal_sbm_blocks.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/ct_2k_sbm_temporal_stubs/synthetic_review.csv \
  --debug-dir outputs/amazon-toy/ct_2k_sbm_temporal_stubs/debug \
  --output outputs/amazon-toy/ct_2k_sbm_temporal_stubs/debug/block_diagnostics.json
```

Inspect the graph-tool nested SBM hierarchy and extraction levels:

```bash
python src/scripts/inspect_sbm_hierarchy.py \
  --reviews data/original/rel-amazon-toy/review.csv \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --output-dir outputs/amazon-toy/sbm_hierarchy_inspection \
  --seed 42
```

Audit all structural method outputs under one root:

```bash
python src/scripts/audit_all_structure_methods.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --outputs-root outputs/amazon-toy \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --output outputs/amazon-toy/structure_method_audit.json
```

## TemporalLatentTextAttributeDiffusion

`TemporalLatentTextAttributeDiffusion` is an attribute generator conditioned on
an existing temporal review spine. It does not regenerate:

```text
customer_id, product_id, review_time
```

Instead, it generates Amazon-style review attributes:

```text
rating, verified, summary, review_text
```

The v1 model is inspired by RelDiff's graph-conditioned mixed-type diffusion. It
uses temporally filtered customer/product history, masked categorical diffusion
for `rating` and `verified`, latent Gaussian diffusion for `summary` and
`review_text`, and nearest-neighbor latent retrieval for text decoding. The
default text encoder tries local transformer embeddings when available and falls
back to deterministic hashing embeddings for offline runs.

Train on the real rel-amazon toy review table:

```bash
python src/scripts/train_temporal_attr_diffusion.py \
  --reviews data/original/rel-amazon-toy/review.csv \
  --cat-cols rating verified \
  --text-cols summary review_text \
  --output-dir outputs/amazon-toy/temporal_attr_diffusion \
  --seed 42
```

Sample attributes onto a generated temporal spine:

```bash
python src/scripts/sample_temporal_attr_diffusion.py \
  --synthetic-spine outputs/amazon-toy/ct_2k_sbm_temporal_stubs/synthetic_review.csv \
  --checkpoint outputs/amazon-toy/temporal_attr_diffusion/checkpoints/best.pt \
  --output outputs/amazon-toy/final_synthetic_review.csv \
  --seed 42
```

Evaluate the full synthetic review table:

```bash
python src/scripts/evaluate_temporal_attr_generation.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/final_synthetic_review.csv
```
