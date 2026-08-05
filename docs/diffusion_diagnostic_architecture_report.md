# Hierarchical Diffusion Diagnostic Architecture Report

## Scope

This report traces the code paths used by:

- `src/scripts/train_hierarchical_v41_tabdlm.py`
- `src/scripts/sample_hierarchical_v41_tabdlm.py`
- `src/scripts/evaluate_single_event_table_paper_metrics.py`

It describes the implemented v4.1 hierarchical diffusion model, not an intended
or earlier design.

## Implementation Plan

1. Preserve the current checkpoint and entry-point interfaces.
2. Add schema-driven oracle field overrides and verified event alignment.
3. Add O1--O5 progressive conditioning and graph-history ablations.
4. Make text decoding policy configurable and structurally exclude control
   tokens from content positions.
5. Make the optimized loss and logged loss agree, and add optional modality
   gradient diagnostics.
6. Add benchmark provenance, fingerprints, unique run directories, and a
   consolidated matrix runner.
7. Add structured-only, graph-coverage, text-quality, and C2ST integrity
   diagnostics.
8. Validate with synthetic fixtures before any full Amazon-toy run.

## Exact Current Architecture

### Data and schema path

`load_config` in `schema.py` builds a `ConditionalTABDLMSchema` from column
roles. The active Amazon-toy v4.1 configuration declares:

- fixed conditions: two foreign keys and one timestamp;
- structured targets: two categorical fields and two categorical text-length
  buckets;
- text targets: summary and long review text.

`ConditionalTABDLMDataset` hashes each foreign key, converts timestamps to Unix
seconds, vocabulary-encodes categorical targets, and tokenizes every text field
with one shared deterministic tokenizer. `collate_and_mask` samples one
continuous diffusion time per row and independently masks categorical and text
positions using the configured schedule.

For batch size \(B\), number of foreign keys \(F\), timestamps \(D\),
structured fields \(C\), text length \(L_f\), hidden width \(H\), graph width
\(G\), and vocabulary sizes \(V_c,V_t\), the active tensors are:

| Tensor | Shape |
| --- | --- |
| `foreign_key_ids` | `[B, F]` |
| `datetime_values` | `[B, D]` |
| `categorical_input_ids` | `[B, C]` |
| `categorical_labels` | `[B, C]` |
| `text_input_ids[f]` | `[B, L_f]` |
| `text_labels[f]` | `[B, L_f]` |
| `text_attention[f]` | `[B, L_f]` |
| `diffusion_t` | `[B]` |
| `graph_context` | `[B, G]` |
| transformer sequence | `[B, C + sum_f L_f, H]` |
| categorical logits | `[B, V_c]` per field |
| text logits | `[B, L_f, V_t]` per field |

The Amazon-toy configuration uses \(H=384\), \(G=256\), six transformer
layers, six heads, and one shared text vocabulary.

### Condition injection

`ConditionalTABDLM.encode_conditions` embeds and projects each foreign key and
timestamp. The graph vector is projected as one additional condition token.
All condition tokens are mean-pooled and projected to the hidden width.

The pooled condition and an MLP embedding of the normalized diffusion time are
added as one shared bias to every target position. With
`graph_fusion.method=gated_residual`, the graph vector is also projected to a
value and gate and added to every target position before the transformer.

Conditioning is therefore injected twice at the transformer input, but not
through per-layer cross-attention, feature-wise modulation, or layer-specific
gates. Structured fields condition text because their current embeddings are
members of the same self-attention sequence.

### Timestep and corruption process

Training samples `diffusion_t ~ Uniform(0,1)` per row. The mask probability is
linear or cosine between the configured minimum and maximum. Masked
categorical values use each field vocabulary's extra mask ID. Masked text
positions use the shared `[MASK]` ID.

The reverse process uses a decreasing discrete schedule. At each step the
model predicts all unresolved positions, samples values, and permanently
reveals a schedule-dependent subset. This is absorbing masked diffusion:
revealed values are not re-masked or revised.

### Hierarchical training path

Every training batch performs two model forwards:

1. The structured forward disables text and optimizes only categorical and
   length-bucket targets.
2. The text forward freezes a selected structured condition and optimizes only
   text targets.

The structured condition is selected once per batch:

- clean real values;
- randomly replaced categorical values;
- a one-forward sampled prediction from the current corrupted structured
  input.

The default probabilities are 0.50, 0.25, and 0.25. Validation defaults to the
one-forward generated condition. This is not the same distribution as the
full multi-step structured reverse process used during sampling.

### Hierarchical sampling path

The sampler first runs the structured reverse schedule. It converts predicted
length buckets into one randomly selected exact length in each bucket. It then
constructs fixed text layouts:

- position zero is `[BOS]`;
- requested content positions are `[MASK]`;
- the next position is `[EOS]`;
- all later positions are `[PAD]`.

Only the content positions are iteratively revealed. Summary and review are
denoised jointly and may attend to one another. Content logits are now
structurally restricted to ordinary vocabulary tokens, while BOS, EOS, PAD,
MASK, UNK, and the internal empty token remain impossible at content
positions. Greedy, top-k, nucleus, temperature, and the current constrained
policy share this restriction. `SimpleTextTokenizer.decode` stops at EOS or
PAD and removes control tokens as a second line of defense.

### Graph context

`TemporalHistoryIndex` builds strict-past histories for the first and second
foreign-key roles. The structure-only encoder embeds target IDs, historical
opposite-side IDs, timestamps, history-source type, log history counts, and
history-coverage bits. It mean-pools customer and product histories and fuses
them with the target event.

Training builds graph histories from real training rows. Validation uses
training plus validation rows. Standard sampling rebuilds the graph from the
synthetic event spine. Thus causality is enforced, but graph coverage and
history composition need not match between training and sampling.

### Loss normalization

Each categorical field uses summed cross-entropy divided by its number of
masked rows. Each text field uses token-weighted summed cross-entropy divided
by the sum of token weights. Field means are multiplied by configured weights
and summed for optimization. This prevents long review text from dominating
solely because it has more tokens.

The canonical v4.1 retraining configuration excludes padding from attention
and denoising labels. Legacy checkpoints remain loadable, and the old
pad-token weights remain in configuration for provenance, but they contribute
zero when padding masking is enabled.

The current epoch logger also accumulates text pad/EOS/content diagnostic
subcomponents, but those diagnostics are no longer added to the objective.
The reported `total_loss` is the actual row-weighted optimized objective.
Raw, weighted, and loss-group values are reported separately, and optional
modality gradient audits report structured and text gradient norms.

### Checkpoints

`save_hierarchical_checkpoint` delegates to the shared `save_checkpoint` and
then adds optimizer state, generation plan, conditioning mixture, and loss
metadata. `load_model_checkpoint` reconstructs the schema, vocabularies,
tokenizer, denoiser, and optional graph encoder from the checkpoint. Existing
checkpoints remain loadable; newly written checkpoints include
`checkpoint_format_version=2` plus hierarchical diagnostics, loss-group, and
conditioning-mixture metadata.

## Bugs and Design Risks Found During Audit

| Severity | Issue | Status |
| --- | --- | --- |
| Critical | Text content logits could sample control tokens, causing truncation and empty text. | Fixed by constrained content support in `hierarchical_sample.py`. |
| Critical | Shared logit sanitization converted category masks from negative infinity to finite values, leaving invalid/missing categories with tiny nonzero sampling support. | Fixed by sampling only a reduced valid-ID vocabulary and mapping sampled IDs back. |
| High | PyTorch nested-tensor padding optimization can return a shorter active text prefix than the fixed input width, causing a sampler reshape failure. | Fixed by prefix-aligned sampling with an assertion that all active positions are covered. |
| High | Oracle conditions used row order rather than event alignment. | Fixed with condition-key plus duplicate-occurrence alignment and hard failure on missing events. |
| High | Oracle lengths supplied only a bucket rather than the exact real length. | Fixed with field-selective exact-length overrides. |
| High | Logged and early-stopping loss double-counted token diagnostics. | Fixed; logged total now equals the optimized objective. |
| High | “Generated-condition” training uses one prediction, not the full reverse sampler. | Open design mismatch; measured by clean/corrupted/mixed training runs. |
| Medium | Padded positions were treated inconsistently between training and sampling. | Fixed behind `training.mask_padding_in_attention`. |
| Medium | Condition corruption had no probabilities, masking, or bounded length perturbation. | Fixed with schema-valid configuration controls. |
| Medium | Sampling graph coverage can differ from real training coverage. | Audited through O1--O5, graph-source metadata, query-only coverage, and graph ablations. |
| Medium | Graph context is injected only before the transformer and may be ignored. | Open architecture risk; condition shuffling/zeroing now measures it. |
| Medium | Text-embedding cache keys omitted data/model identity. | Fixed with content and model fingerprints. |
| Medium | C2ST preprocessing was fit before cross-validation. | Fixed: only stateless transforms occur globally; fitted scaling remains inside each fold. |
| Low | Condition mixture is selected per batch. | Retained and logged as realized row proportions. |
| Low | Checkpoints lacked a format version. | Fixed for new checkpoints; old checkpoints remain compatible. |

## Diagnostic Interpretation

The existing aggregate results do not yet isolate the cause of the diffusion
gap. Structured marginals and temporal trends can remain good while full-row
and text C2ST are poor. The critical content-token sampling bug can directly
damage text, but the graph distribution shift, one-step/full-reverse condition
mismatch, and weak input-only conditioning are also plausible contributors.

The O1--O5 benchmark is therefore required before choosing a replacement
architecture:

- O1 measures the text denoiser and decoder with fully real conditions.
- O2 isolates generated structured values.
- O3 additionally isolates generated lengths.
- O4 introduces the standard sampling graph source.
- O5 removes graph history.

No architectural recommendation should be treated as final until those runs,
the structured-only comparison, condition shuffling, and C2ST sanity controls
have been executed with multiple seeds.
