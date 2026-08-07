# LSTM v5.3 numerical path

This report traces the existing numerical path before the support-aware changes.
Tensor shapes use the Rel-HM configuration in
`configs/attribute_generation/lstm_hm_10k_customers.yaml`, with batch size `B`.

## Inputs and causal context

The event spine supplies two foreign keys and one timestamp:

- source foreign key: `customer_id`
- destination foreign key: `article_id`
- timestamp: `event_time`

`RelationalEventDataset` hashes each foreign key and encodes the timestamp as a
floating-point datetime value. The model receives:

- `foreign_key_ids`: `[B, 2]`
- `datetime_values`: `[B, 1]`
- structure-only graph context: `[B, 64]`

Each foreign key has its own `Embedding(262144, 64)` in
`lstm_joint.py`. The direct source and destination representations are therefore
each `[B, 64]`. `DateTimeEncoder(32)` produces `[B, 1, 32]`.

The graph encoder in `graph_encoder.py` is past-only and structure-only. Its
target event combines source identity, destination identity, event type, and
timestamp. Its two history branches aggregate past source-side and
destination-side event identities and timestamps. It does not receive `price`,
`sales_channel_id`, or any other target attribute.

For Rel-HM the graph encoder dimensions are:

- history event input: `4 * 64 + 32 = 288`
- target event input: `3 * 64 + 32 = 224`
- source history, destination history, and target encodings: `[B, 64]` each
- count/coverage features: `[B, 4]`
- graph fusion input: `3 * 64 + 4 = 196`
- graph output: `[B, 64]`

No static customer or article table attributes are loaded by this model.
Historical target attributes are deliberately excluded from graph context.

## Shared fusion and row latent

`JointLSTMRelationalAttributeGenerator.encode_condition()` concatenates:

```text
source embedding       [B, 64]
destination embedding  [B, 64]
timestamp encoding     [B, 32]
graph context          [B, 64]
--------------------------------
condition input        [B, 224]
```

One shared MLP immediately compresses this to `[B, 128]`. Consequently,
destination information reaches the numerical head, but it has no dedicated
path after this shared compression.

`row_latent()` concatenates `[B, 128]` condition features with independent
Gaussian noise `[B, 32]`. The row encoder returns `[B, 128]`. The same row
latent feeds the categorical and numerical heads.

## Existing numerical distribution and loss

For every numerical target, the current head is `Linear(128, 2)`. For `price`
it returns `[B, 2]`:

- column 0: standardized Gaussian mean
- column 1: standardized Gaussian log standard deviation

Training uses independent Gaussian negative log likelihood in
`lstm_joint_loss()`. Log standard deviation is clamped to `[-7, 5]`.
Generation samples

```text
mean + Normal(0, 1) * exp(log_std) * numerical_temperature
```

The head is therefore stochastic in two ways:

1. row-latent noise changes the predicted Gaussian parameters;
2. Gaussian decoding samples again from those parameters.

The training-only numerical transform in `numerical.py` standardizes `price`
using the training mean and standard deviation. Sampling applies the inverse
transform and clips to the observed training range. Count-valued fields use
log1p standardization and rounding, but Rel-HM `price` uses ordinary
standardization.

The Gaussian decoder is continuous and does not know the 3,124-value training
support. This explains why nearly every generated price can be unique even
when it lies very close to a valid training value.

## Conditioning and gradient conclusions

All context is injected once before the shared row encoder. There is no direct
destination-to-price residual, modulation, attention, or conditional prior.
Destination identity and destination structural history can influence price
only through the shared `[B, 128]` condition and row latent.

The numerical loss has a differentiable path to:

- the numerical head;
- the shared row encoder and condition MLP;
- both direct foreign-key embedding tables;
- timestamp encoder;
- graph encoder, including source and destination history branches.

This establishes gradient reachability, not useful gradient magnitude. The
paired-context and gradient diagnostics must measure whether the trained model
actually uses these paths.

## Evidence still requiring a checkpoint

The following cannot be concluded from static code and must be measured on the
trained checkpoint:

- row-to-row variation in predicted Gaussian variance;
- sensitivity to destination, source, and timestamp shuffling at fixed latent;
- gradient norm reaching each context path;
- whether predicted means collapse toward the global price distribution.

## Design implication

The smallest compatible intervention is:

1. retain the shared event-spine and stochastic row path;
2. expose source, destination, temporal, and graph components explicitly;
3. add source-plus-temporal projection and gated destination residual fusion
   only for numerical targets;
4. select a continuous or support-aware output head from training-only
   numerical-type metadata;
5. optionally add smoothed training-only support priors.

The existing Gaussian path remains the default so old v5.3 checkpoints retain
their exact parameter layout and behavior.
