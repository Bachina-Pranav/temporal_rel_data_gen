# V4.1 Graph-Conditioned Hierarchical Multimodal Diffusion

This document describes the current flat v4 implementation and the v4.1 hierarchical design.

## Current V4 Computation Graph

Entry points:

- Training: `src/scripts/train_v2_full_review_text_tabdlm.py`
- Sampling: `src/scripts/sample_v2_full_review_text_tabdlm.py`
- Model: `src/attribute_generation/conditional_tabdlm/model.py`
- Sampler: `src/attribute_generation/conditional_tabdlm/sample.py`
- Config: `configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml`

### Event-Spine Encoding

The event spine provides conditioning columns:

- foreign keys from `schema.foreign_key_columns`,
- datetimes from `schema.datetime_columns`.

Foreign keys are hashed into learned embeddings. Datetimes are encoded with a Fourier timestamp encoder. The condition tokens are projected into a shared condition space.

### Graph Construction

The graph context is built by `TemporalHistoryIndex`. For each target event, it retrieves customer/product ego histories strictly before the target timestamp unless the config explicitly allows same-timestamp events. The Amazon-toy v4 config sets:

```yaml
allow_same_timestamp_events: false
exclude_target_event_from_neighbors: true
graph_uses_future_events: false
graph_uses_target_attributes: false
```

The structure-only graph encoder uses customer/product hashes and event timestamps from past histories. It does not include real rating, verified, summary, or review text.

### Current Graph Fusion

Original v4 graph fusion is weak/global:

1. FK tokens, datetime tokens, and the graph token are projected into `condition_dim`.
2. These condition tokens are averaged.
3. The average is projected to hidden size.
4. The resulting condition vector is added as a shared bias to every target token.

This means graph context can influence predictions, but it is not represented as an explicit token in the denoising Transformer sequence.

V4.1 adds an optional explicit fusion method:

```yaml
model:
  graph_fusion:
    method: gated_residual
```

The implementation applies:

```text
h' = h + sigmoid(W_g g) * W_v g
```

to target token states, where `g` is the row-level graph embedding. Existing v4 configs continue to use the old condition-mean behavior unless this option is enabled.

### Flat V4 Target Sequence

The flat v4 model uses one Transformer over:

```text
rating
verified
summary_length_bucket
review_text_length_bucket
summary tokens
review_text tokens
```

All target fields are denoised under one shared schedule. This is truly joint in the architectural sense because fields can attend to one another inside the Transformer. However, it is only implicitly structured-to-text conditioned during sampling.

### Why Structured Attributes May Weakly Affect Text

The flat sampler starts with every structured attribute and every text token masked. During each reverse step:

1. the model predicts all still-masked fields,
2. some fields are committed,
3. committed fields are never revised.

This creates two failure modes:

- A text token can be committed before `rating`, `verified`, or a length bucket has been revealed.
- Once a text token is committed, later structured information cannot revise it.

So even though the Transformer has joint attention, sampling does not guarantee the text stage observes generated structured values before text tokens are finalized.

## V4.1 Factorization

V4.1 explicitly implements:

```text
p(z_i | c_i) * p(x_i | z_i, c_i)
```

where:

- `c_i`: event-spine and graph context,
- `z_i`: structured categorical attributes and text-length variables,
- `x_i`: text fields.

The generation plan is schema-driven:

```yaml
generation:
  factorization: structured_then_text
  stages:
    - name: structured
      fields:
        - rating
        - verified
        - summary_length_bucket
        - review_text_length_bucket
      condition_on:
        - event_context
        - graph_context

    - name: text
      fields:
        - summary
        - review_text
      condition_on:
        - structured
        - event_context
        - graph_context
```

Reusable code validates that every generated field belongs to exactly one stage and that text fields have a length mechanism.

## Stage 1: Structured Diffusion

Stage 1 denoises schema-declared categorical variables:

- regular structured targets,
- auxiliary text-length buckets.

Text tensors are inactive in this stage. Structured outputs use existing valid-domain masking through the current categorical vocab and constraint utilities.

## Stage 2: Conditional Text Diffusion

Stage 2 freezes the generated structured values as visible, non-maskable conditioning values. Text fields are then denoised jointly.

The text stage receives:

- generated structured categorical values,
- generated length buckets,
- event context,
- graph context,
- diffusion timestep,
- field/position embeddings.

Generated structured values are used during valid inference. Oracle structured conditioning is available only as a diagnostic and is marked `NOT A VALID GENERATIVE BASELINE`.

## Length-Conditioned Text Generation

V4.1 converts generated length buckets into exact content lengths before text diffusion begins. The text tensor is initialized as:

```text
BOS, MASK...MASK, EOS, PAD...PAD
```

Only content positions are denoised. EOS and PAD are fixed from the start, so the sampler does not generate a full max-length sequence and then repair it row by row.

## Training Conditioning Mixture

The hierarchical trainer supports:

```yaml
training:
  text_conditioning:
    mode: mixed
    clean_probability: 0.5
    corrupted_probability: 0.25
    generated_probability: 0.25
```

Modes:

- `clean`: condition text on ground-truth structured values.
- `corrupted`: condition text on schema-valid corrupted structured values.
- `generated`: condition text on detached stage-1 generated values.

This reduces training-inference mismatch without backpropagating through discrete samples.
Validation defaults to generated structured conditioning, so the selected checkpoint is judged under the same kind of conditioning used by valid sampling. Clean validation can still be enabled explicitly for diagnostics.

## Loss Structure

The staged trainer records structured and text losses separately. Existing per-field loss weights and text token weights remain configurable. Text loss is normalized per field by the current denoising loss implementation, and EOS/PAD/content components continue to be logged for text fields.

## Graph Diagnostic Requirement

`src/scripts/diagnose_v4_graph_conditioning.py` compares logits under:

- correct graph context,
- zero graph context,
- shuffled graph context,
- identity/time-only graph context,
- graph fusion disabled.

It also checks graph gradient flow and graph-context variation across rows. These diagnostics prove signal reaches logits, but final claims about graph usefulness require generated-table ablations.

The same script also includes a schema-driven structured-to-text counterfactual: it changes one valid structured categorical value while holding event and graph context fixed, then measures text-logit changes. This tests whether visible structured tokens can affect text predictions without baking in rating-specific or Amazon-specific rules.

## Known Limitations Of This Implementation Pass

The local implementation provides the v4.1 staged training/sampling machinery and diagnostics. The actual GPU experiments must still be run on the VM to decide whether v4.1 beats flat v4 or LSTM v5.3.
