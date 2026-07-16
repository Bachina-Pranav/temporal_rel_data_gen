# Event Spine Generator Results

Amazon Toy and Full Rel-Amazon Validation

Pranav / RELDIFF temporal relational generation

---

## What We Need To Show

The event spine generator is responsible for producing:

```text
customer_id, product_id, review_time
```

Before generating ratings, verified flags, summaries, or review text, we need evidence that the structural backbone is working.

The validation question is:

```text
Does the synthetic event table preserve who interacts, what they interact with,
and when those interactions happen?
```

---

## Success Criteria

The event spine is working if it preserves the hard structural constraints and stays scalable.

Core checks:

- Row count matches the real event table
- Customer degree sequence is preserved
- Product degree sequence is preserved
- Daily event counts are preserved
- Block-pair-time counts are preserved
- Runtime scales to full Rel-Amazon

Secondary checks:

- Entity lifecycle correlations are high
- Duplicate customer-product rates are realistic
- Exact event overlap is low enough to avoid copying
- C2ST is close to chance when sampled

---

## Current Best Method

The current event-spine method is:

```text
Time-Biased Block Stub Matching
```

Short name in the result scripts:

```text
time_biased_local_kernel_main
```

Presentation name:

```text
TBSM-local-kernel-dynamic
```

It combines exact constraint preservation with probabilistic temporal and pairwise matching.

---

## Why This Method Is Different

Most earlier methods sampled events directly.

This method instead builds the synthetic event table in constrained pieces:

1. Create exact block-pair-time slots
2. Create exact customer and product degree stubs
3. Assign stubs to time slots using time-biased activity models
4. Pair customer and product stubs inside each cell
5. Verify exact constraints after generation

This makes the generator much easier to validate.

---

## Amazon Toy Dataset

The Amazon toy dataset is the first end-to-end validation setting.

It is small enough to iterate quickly, but still relational and temporal:

```text
customer_id, product_id, review_time
```

The point of this experiment is not just quality. It verifies that the complete structural pipeline runs without pathological slowdown or constraint failure.

---

## Amazon Toy Run Evidence

Observed run:

```text
outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/
```

Key terminal evidence:

| Quantity | Value |
|---|---:|
| Synthetic events | 79,663 |
| Block-pair-time cells | 14,356 |
| Fit time | 7.46 s |
| Pairing time | 9.93 s |
| Total time | 17.88 s |
| Throughput | 7,641.2 events/s |
| Verification | all exact constraints passed |

This is the strongest simple evidence that the event spine generator works end to end on Amazon toy.

---

## Amazon Toy Debug Artifacts

The toy run produced the expected output and debug files:

```text
outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/
  synthetic_review.csv
  metadata.json
  debug/
    block_pair_time_counts.csv
    customer_blocks.csv
    product_blocks.csv
    customer_stub_assignment_summary.json
    product_stub_assignment_summary.json
    dynamic_pairing_summary.json
    lowrank_time_gated_affinity_summary.json
```

These files make the run auditable instead of being a black-box sample.

---

## Amazon Toy Interpretation

The toy result shows:

- The generator creates the correct number of events
- Customer and product stubs can be assigned exactly
- Block-pair-time slots can be filled exactly
- Pairing completes quickly
- The final verifier passes

Advisor-facing phrasing:

```text
On Amazon toy, the event-spine generator runs end to end in under 20 seconds
and passes all exact structural constraints.
```

---

## Full Rel-Amazon Preflight

Full Rel-Amazon is much larger:

| Quantity | Value |
|---|---:|
| Reviews | 12,644,508 |
| Customers in reviews | 1,584,084 |
| Products in reviews | 416,125 |
| Days | 2,923 |
| Months | 97 |
| Date range | 2008-01-01 to 2016-01-01 |

This is the actual scale target for the event-spine generator.

---

## Full Rel-Amazon Structure

The full preflight found the needed block assignments:

| Quantity | Value |
|---|---:|
| Customer blocks | 5 |
| Product blocks | 5 |
| Block-pair-time cells | 72,458 |
| Average cell size | 174.5 |
| P95 cell size | 755.1 |
| P99 cell size | 3,678.0 |
| Max cell size | 14,091 |

Important interpretation:

```text
The 12.6M-row problem decomposes into 72k smaller pairing cells.
```

---

## Full Rel-Amazon Feasibility

The preflight warning was:

```text
Large cells detected. Exact pairing will fallback to projection for those cells.
```

This is expected.

It means:

- Most cells are small enough for direct constrained pairing
- A few large cells use the scalable projection fallback
- The generator avoids building huge dense affinity matrices
- The full run should be feasible on the available machine

This is a scalability check, not a failure.

---

## Full Rel-Amazon Integrity Checks

The full data audit showed the review spine is internally valid:

| Check | Result |
|---|---:|
| Real review rows | 12,644,508 |
| Synthetic spine rows | 12,644,508 |
| Spine row count matches real | true |
| Customer FK valid rate | 1.0 |
| Product FK valid rate | 1.0 |
| Timestamp parse error rate | 0.0 |

This means the generated spine is structurally compatible with the real customer and product tables.

---

## Main Metrics To Report

After running evaluation, the results table should report:

| Metric | Meaning | Direction |
|---|---|---|
| `customer_degree_exact_match` | exact customer degree preservation | true |
| `product_degree_exact_match` | exact product degree preservation | true |
| `daily_count_l1` | daily count error | lower is better |
| `block_pair_time_exact_match` | exact block-time preservation | true |
| `monthly_count_corr` | monthly volume alignment | higher is better |
| `duplicate_rate_ratio` | repeated customer-product realism | close to 1 |
| `exact_event_overlap_rate` | direct copied event tuples | lower is safer |
| `event_tuple_c2st_auc` | real/synthetic distinguishability | close to 0.5 |
| `total_seconds` | full runtime | lower is better |
| `events_per_second` | throughput | higher is better |

---

## Lifecycle Metrics To Report

These show whether products and customers appear at realistic times.

| Metric | Meaning | Direction |
|---|---|---|
| `product_first_time_corr` | product start-time alignment | higher is better |
| `product_last_time_corr` | product end-time alignment | higher is better |
| `product_peak_time_corr` | product peak-time alignment | higher is better |
| `customer_first_time_corr` | customer start-time alignment | higher is better |
| `customer_last_time_corr` | customer end-time alignment | higher is better |
| `customer_peak_time_corr` | customer peak-time alignment | higher is better |
| `product_active_span_ks` | product active-window distribution | lower is better |
| `customer_active_span_ks` | customer active-window distribution | lower is better |

---

## Suggested Results Slide Table

Fill this directly from:

```text
outputs/rel-amazon/event_spine_paper_table.md
```

| Method | Degree KS C/P | Block-time L1 | Product lifecycle corr | Customer lifecycle corr | Duplicate ratio | Exact overlap | C2ST AUC | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StaticDegree | FILL | FILL | FILL | FILL | FILL | FILL | FILL | FILL |
| CT-2K-SBM | FILL | FILL | FILL | FILL | FILL | FILL | FILL | FILL |
| TBSM-local-kernel-random | FILL | FILL | FILL | FILL | FILL | FILL | FILL | FILL |
| **TBSM-local-kernel-dynamic** | FILL | FILL | FILL | FILL | FILL | FILL | FILL | FILL |

The bold row is the method to emphasize.

---

## How To Read The Table

The strongest result is not one single metric.

The event spine is convincing when the main method simultaneously has:

- Exact degree preservation
- Exact or near-exact block-pair-time preservation
- Strong lifecycle correlations
- Reasonable duplicate rate
- Low exact event overlap
- C2ST close to 0.5
- Practical runtime on full Rel-Amazon

This combination says the method is preserving structure without simply copying events.

---

## Commands: Evaluate Amazon Toy

Use this if the toy run directory is:

```text
outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing
```

Command:

```bash
python src/scripts/evaluate_time_biased_block_stub_matching.py \
  --real-reviews data/original/rel-amazon-toy/review.csv \
  --synthetic-reviews outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/synthetic_review.csv \
  --structure-debug-dir outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/debug \
  --metadata outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/metadata.json \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --time-granularity day \
  --time-gate-granularity month \
  --rank 32 \
  --seed 42 \
  --output outputs/amazon-toy/time_biased_block_stub_matching_exact_pairing/metrics.json
```

---

## Commands: Evaluate Full Rel-Amazon

Start with the faster evaluation:

```bash
python src/scripts/evaluate_time_biased_block_stub_matching.py \
  --real-reviews data/original/rel-amazon/review.csv \
  --synthetic-reviews outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --structure-debug-dir outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/debug \
  --metadata outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/metadata.json \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --time-granularity day \
  --time-gate-granularity month \
  --rank 32 \
  --seed 42 \
  --skip-dynamic-affinity \
  --output outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/metrics.json
```

Then run dynamic affinity and C2ST separately if needed.

---

## Commands: Full Rel-Amazon C2ST Sample

C2ST is heavier, so run it as a second pass:

```bash
python src/scripts/evaluate_time_biased_block_stub_matching.py \
  --real-reviews data/original/rel-amazon/review.csv \
  --synthetic-reviews outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/synthetic_review.csv \
  --structure-debug-dir outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/debug \
  --metadata outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/metadata.json \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --time-granularity day \
  --time-gate-granularity month \
  --rank 32 \
  --seed 42 \
  --compute-c2st \
  --c2st-sample-size 200000 \
  --output outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/eval_metrics_c2st.json
```

---

## Commands: Build Comparison Table

After all generator outputs exist:

```bash
python src/scripts/compare_rel_amazon_event_spine_generators.py \
  --real-reviews data/original/rel-amazon/review.csv \
  --structure-debug-dir outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/debug \
  --reuse-existing-metrics \
  --skip-dynamic-affinity \
  --output-json outputs/rel-amazon/event_spine_generator_comparison.json \
  --output-csv outputs/rel-amazon/event_spine_generator_comparison.csv
```

Then create the paper-style table:

```bash
python src/scripts/make_event_spine_paper_table.py \
  --input-json outputs/rel-amazon/event_spine_generator_comparison.json \
  --output-csv outputs/rel-amazon/event_spine_paper_table.csv \
  --output-md outputs/rel-amazon/event_spine_paper_table.md
```

---

## Commands: Build Written Summary

Generate a cautious text summary:

```bash
python src/scripts/summarize_event_spine_results.py \
  --input-json outputs/rel-amazon/event_spine_generator_comparison.json \
  --output outputs/rel-amazon/event_spine_result_summary.md
```

Use this as the speaker note for the results slide.

---

## Advisor Talk Track

The event spine is now separated from the attribute generator.

On Amazon toy, the generator completed in 17.88 seconds, produced 79,663 events, and passed all exact structural constraints.

On full Rel-Amazon, the preflight confirms the problem is large but decomposes into 72,458 block-pair-time cells. Most cells are small, and the largest cells are handled by the projection fallback.

The next slide should show the final metric table from the full run.

---

## Bottom Line

The evidence supports the claim that the event-spine generator is working:

- It runs end to end on Amazon toy
- It verifies exact constraints on Amazon toy
- It has the correct debug artifacts for auditability
- It is designed to scale to full Rel-Amazon
- Full Rel-Amazon preflight confirms the structure files and cell decomposition are feasible

Once the full metrics table is inserted, this becomes the advisor-facing validation slide deck.
