# Temporal Event Spine Generator v2

Time-Biased Block Stub Matching for Rel-Amazon

Pranav / RELDIFF temporal relational generation

---

## Goal

Generate the event spine of a temporal relational database:

```text
(customer_id, product_id, review_time)
```

This is the structural backbone. Attribute models later fill in:

```text
rating, verified, summary, review_text, ...
```

The v2 event-spine generator focuses only on who interacts with what and when.

---

## Why Event Spine First?

Temporal relational data has two coupled structures:

1. Static relational structure
   - Which customers/products exist
   - Degree distributions
   - Customer-product interaction patterns

2. Temporal structure
   - Daily/monthly volume
   - Entity lifecycles
   - Product/customer seasonality

If the spine is wrong, attribute generation is conditioned on the wrong graph.

---

## Problem With Naive Sampling

Naive independent sampling might preserve marginal distributions, but breaks joint structure.

Example failure modes:

- A product appears before its real lifecycle
- A heavy customer loses their degree
- Daily counts drift
- Block-pair seasonality disappears
- Customer-product pairs become random

We need a method that is both structurally constrained and scalable.

---

## v2 Method In One Sentence

Create exact block-pair-time slots, assign exact customer/product degree stubs into those slots using time-biased desired times, then pair customer and product stubs inside each cell using a low-rank time-gated affinity model.

Core idea:

```text
exact constraints first, probabilistic ordering second
```

---

## Inputs And Output

Inputs:

- Real review table
- Columns: `customer_id`, `product_id`, `review_time`
- Optional SBM block assignments:
  - `customer_blocks.csv`
  - `product_blocks.csv`

Output:

```text
synthetic_review.csv
customer_id, product_id, review_time
```

No ratings or text are generated here.

---

## Stage 1: Normalize Time

The generator first canonicalizes timestamps into discrete buckets.

Current defaults:

- Event bucket: day
- Time gate bucket: month

So each event gets:

```text
day_code      = exact event day
month_gate    = coarser temporal context for affinity
```

Daily counts are preserved exactly.

---

## Stage 2: Load Blocks

Customers and products are assigned to structural blocks.

These usually come from a prior SBM-style structure extraction step.

If block files are unavailable, the generator falls back to one customer block and one product block.

Blocks let us preserve meso-scale structure:

```text
customer block x product block x day
```

---

## Stage 3: Count Block-Pair-Time Cells

From real data, compute:

```text
N[b_c, b_p, t] =
  number of events from customer block b_c
  to product block b_p
  on day t
```

Then create exactly that many slots for each cell.

This guarantees:

```text
synthetic block-pair-time counts = real block-pair-time counts
```

---

## Slot View

Each real event becomes an abstract slot:

```text
slot = (customer_block, product_block, day, month_gate)
```

At this point, slots do not yet have actual customer/product IDs.

The slot table has exactly the same number of rows as the real review table.

---

## Stage 4: Customer And Product Stubs

For every customer, create one stub per real event degree.

Example:

```text
customer C17 has degree 5
=> create 5 customer stubs for C17
```

Same for products.

This guarantees exact degree preservation if every stub is assigned exactly once.

---

## Stage 5: Temporal Activity Model

Each entity gets a temporal activity profile.

For a customer or product, the model uses:

- Empirical event days for that entity
- Block-level temporal behavior
- Global temporal distribution

The default v2 mode uses local kernel sampling around observed event days.

---

## Local Kernel Desired Times

For each entity stub:

1. Pick one of the entity's observed event days
2. Add local temporal noise
3. Use this as the stub's desired day

Bandwidths are estimated from real inter-event gaps.

Important:

```text
Bandwidth selection uses only real data,
not synthetic evaluation metrics.
```

---

## Stage 6: Assign Stubs To Slots

Within each block:

1. Sort slots by actual day
2. Sort entity stubs by desired day
3. Match them in sorted order

This is the main trick.

It preserves exact degrees and block membership while keeping each entity near its natural lifecycle.

---

## Why Sorted Matching Works

Suppose a customer has desired times:

```text
Jan, Feb, Mar
```

And the block has available slots:

```text
Jan 02, Feb 10, Mar 03
```

Sorted matching assigns the customer stubs to nearby available dates without rejection loops.

It is fast because it is mostly sorting, not per-event search.

---

## Stage 7: Pair Customers With Products

After stub assignment, each slot has:

```text
customer_id, product_id_candidate, day
```

But products can still be reordered within the same exact cell:

```text
(customer_block, product_block, day)
```

Reordering products inside a cell preserves all exact slot constraints.

---

## Dynamic Affinity Model

The pairing score is:

```text
F(u, i, t) = (z_u * g_t)^T z_i
```

Where:

- `z_u` = low-rank customer embedding
- `z_i` = low-rank product embedding
- `g_t` = time-gate vector, usually monthly
- `*` = elementwise multiplication

This captures time-varying customer-product compatibility.

---

## How Embeddings Are Learned

The model builds a sparse customer-product interaction matrix from real data.

Then it applies truncated SVD to get low-rank embeddings.

This avoids a dense tensor:

```text
customers x products x time
```

which would be too large for Rel-Amazon.

---

## Pairing Inside A Cell

For small cells:

- Build a local score matrix
- Greedily match high-scoring customer-product assignments
- Apply penalties for:
  - duplicate synthetic pairs
  - real pair overlap
  - exact real event overlap

For large cells:

- Use projection sorting fallback
- Optionally apply small local overlap repair

---

## Exact Guarantees

The generator verifies:

- Same number of events as real
- Exact customer degree sequence
- Exact product degree sequence
- Exact daily counts
- Exact customer block assignment for every slot
- Exact product block assignment for every slot
- Exact block-pair-time counts

These are hard constraints, not just evaluation targets.

---

## What Is Stochastic?

Even with exact constraints, the generator remains stochastic through:

- Local kernel desired times
- Jitter in sorted matching
- Product reordering choices
- Tie-breaking in greedy pairing

So it is not simply copying the real event table.

---

## Scalability Design

The method avoids slow per-event candidate scoring.

Main scalable choices:

- Precompute slots from counts
- Use integer arrays, not Python object loops where possible
- Assign stubs by sorting
- Use low-rank affinity instead of dense pair-time tensor
- Use projection fallback for large cells

This is why full Rel-Amazon becomes feasible.

---

## Debug Outputs

The generator writes:

```text
debug/customer_blocks.csv
debug/product_blocks.csv
debug/block_pair_time_counts.csv
debug/customer_time_activity_summary.json
debug/product_time_activity_summary.json
debug/lowrank_time_gated_affinity_summary.json
debug/customer_stub_assignment_summary.json
debug/product_stub_assignment_summary.json
debug/dynamic_pairing_summary.json
metadata.json
```

These make the run auditable.

---

## Metrics We Care About

Structural metrics:

- Customer degree KS
- Product degree KS
- Edge overlap rate
- Duplicate customer-product rate

Temporal metrics:

- Global timestamp KS
- Daily count L1
- Inter-event time KS
- Top-product trajectory correlation

Constraint metrics:

- Degree exactness
- Block-pair-time exactness

---

## How It Fits The Full Pipeline

Pipeline:

```text
1. Event spine generator
   -> synthetic_review.csv with customer/product/time

2. Attribute generator
   -> rating, verified, summary, review_text

3. Paper-grade evaluator
   -> structural, temporal, attribute, privacy metrics
```

The event spine is the conditioning graph for attribute generation.

---

## Main Limitations

The current v2 spine generator:

- Preserves selected constraints exactly, which can be strong
- Depends on useful customer/product block assignments
- Does not generate new customers/products
- Does not model attributes directly
- Uses day-level event buckets
- Uses approximate pairing fallback for very large cells

These are acceptable for the current single-event-table Rel-Amazon setup.

---

## Advisor Takeaway

The v2 event-spine generator is not a black-box neural model.

It is a constrained probabilistic generator:

```text
Exact relational-temporal constraints
+ entity lifecycle sampling
+ low-rank time-aware pairing
```

This gives us a scalable, auditable structural backbone for temporal relational synthetic data.

---

## Backup: Pseudocode

```python
fit(real):
    bucket timestamps by day and month
    load customer/product blocks
    count block-pair-day cells
    fit customer/product temporal activity models
    estimate local temporal bandwidths
    fit low-rank time-gated affinity

sample():
    create exact block-pair-day slots
    create customer stubs from real customer degrees
    create product stubs from real product degrees
    assign customer stubs to slots by desired time sorting
    assign product stubs to slots by desired time sorting
    reorder products within each cell by dynamic affinity
    verify exact constraints
    return synthetic spine
```

---

## Backup: Core Equation

Dynamic pairing score:

```text
F(u, i, t) = (z_u * g_t)^T z_i
```

Penalized score for small-cell matching:

```text
score = affinity
      - lambda_dup   * duplicate_pair_penalty
      - lambda_pair  * real_pair_overlap_penalty
      - lambda_event * exact_event_overlap_penalty
```

This keeps affinity high while discouraging memorization.

---

## Backup: Run Command

```bash
python src/scripts/run_time_biased_block_stub_matching_generator.py \
  --real-reviews data/original/rel-amazon/review.csv \
  --structure-debug-dir outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/debug \
  --customer-id-col customer_id \
  --product-id-col product_id \
  --timestamp-col review_time \
  --output-dir outputs/rel-amazon/time_biased_block_stub_matching_kernel_main \
  --time-granularity day \
  --time-gate-granularity month \
  --rank 32 \
  --desired-time-sampling-mode local_kernel \
  --pairing-mode dynamic_exact_penalized \
  --max-exact-affinity-cell-size 128 \
  --large-cell-pairing projection_sort \
  --seed 42
```

