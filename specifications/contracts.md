# SHINRA v2 implementation contracts

## Shape and parameter contract

The canonical dimensions and all modality components are in `ShinraConfig`.
`parameter_ledger` calculates counts using integer arithmetic without importing
PyTorch. Every frontend/core/head parameter is included. Tests compare the algebra
to actual `named_parameters()` on a reduced instance of the identical graph.
Default totals: core 23,413,976,064; multimodal 1,763,842,816;
combined **25,177,818,880**. No trainable recurrent initial state; tied text head.

RMSNorm has one affine scale, no bias. Attention QK norm shares a head-dimension
scale across heads. Memory output norm has separate affine scales for every head.
No third-party model configuration is instantiated.

## Recurrence

For normalized q/k, per-channel decay D and scalar beta in [0,1]:

    prior = D * state
    state = prior + beta * outer(k, v - transpose(prior) @ k)
    output = transpose(state) @ q

State and decay arithmetic use FP32. Zero initial state is runtime data. Causal
Q/K/V depthwise convolutions retain exactly kernel-1 history elements. Each segment
returns a differentiable state; training never detaches recurrence between segments.
The FLA adapter uses scale=1 and precomputed log-decay/normalized QK to match this
equation. Backend absence raises; there is no silent semantic substitution.

## Causality and boundaries

The backbone accepts dense, unpadded, single-episode sequences per batch row.
Padding masks and arbitrary packed document boundaries are not ignored: they must
be represented as independent calls/episodes. Do not concatenate unrelated documents
and assume a text EOS clears recurrent memory. Use a fresh cache/episode.

Text labels align with input positions; the model shifts by one for causal CE.
Latent positions have label -100. Start/end control IDs are supplied by the hashed
tokenizer contract, not invented automatically by the model.

Multimodal positions consume the same 327680 total budget as text. Physical
coordinates are converted to fixed Fourier features; callers must preserve units,
crop scale and acquisition/availability time. `LatentSegment.coordinates` aligns
one coordinate vector to each latent. Text is ordered by stream position.

Video compression looks at up to four frames. All tokens emitted by such a group
carry the group's last timestamp as their availability time. A real-time caller
must not use these tokens before that time. Partial groups flush only on final=True.

## Streaming audio

Causal convolution retains phase and sample history across arbitrary chunk cuts.
Encoder local KV is bounded to window-1 prior frames. Resampling groups two encoded
frames; one pending frame is carried across nonfinal calls. Query slot cycling is
derived from absolute frame number, so API chunking does not change outputs.

At 24kHz, stem stride 240 and two stride-2 stages yield 25 frames/s, then 12.5
latents/s. Boundary tokens are added by stream assembly at caller-defined events.
Zero-length audio calls are rejected; terminate the last nonempty call with final=True.

## Streaming video

Image encoder/resampler is shared. Default video uses 64 spatial tokens/frame,
8 fast tokens/frame and 64 temporally compressed tokens/group of four. Frame rate
is a data contract, not inferred from an index. Increasing frame rate increases
backbone token use. The cache retains at most three pending encoded frames.

## World prediction

World slots pool observation latents. Action values are normalized externally and
paired with masks, action type, continuous nonnegative delta_t and horizon ID.
Conditioned slots pass through the shared backbone before future-state prediction.
Counterfactual branches start from the same cache snapshot and different actions.
No real actuator is controlled by the model API.

Future target latents are detached in prediction loss. The target perception
representation also needs reconstruction/variance/consistency objectives to prevent
collapse. Mean/log-variance heads are diagonal distributions, not a guarantee of
calibrated multimodal futures or physical correctness.

## Output decoders

Image and audio use conditional rectified-flow objectives and midpoint integration.
They are randomly initialized SHINRA modules, not external VAE/CLIP/Whisper models.
The visual decoder sees spatial positions. Audio output has four waveform patches
per latent. Full quality, temporal fidelity and synthesis stability require training.
Long output media must be decoded in bounded windows; the reference frontend
attention guards against an unbounded score allocation.

## Loss and cache memory

Vocabulary logits are computed per chunk. Nonreentrant checkpointing recomputes CE
chunks in backward instead of retaining every vocabulary tensor. Training forward
returns hidden states and scalar loss by default, not full logits. Selected log-probs
for RL use exact chunked logsumexp. Higher-order CE gradient compatibility is not
a released guarantee.

`ShinraCache` stores global KV, recurrent matrix/conv state, tensor-valued frontend
side state and world slots. New forward calls return a new cache and never mutate
old tensor prefixes. `fork` shares read-only tensors; callers must not modify them
in place. Reset clears all streams and world state. Global KV is contiguous and
grows to the explicit context limit; it is not paged or quantized.

## Checkpoints

Native checkpoints use safetensors shards, exact parameter-key/shape validation,
SHA256 file hashes and atomic publication to a new directory. Partial writes remain
under `.incomplete-*` and are never treated as completed checkpoints. An individual
parameter larger than shard_bytes is stored alone rather than split inconsistently.
CPU optimizer state and RNG use locally produced tensor state files loaded with
weights_only=True. Never load arbitrary remote optimizer pickles.

FSDP/TE runtime state uses PyTorch distributed checkpointing. Model-only native
save refuses TE-converted modules because quantization metadata must not silently
disappear. Experiment drivers must save config, curriculum/data cursor and RNG
alongside distributed checkpoints before accepting a resumable run.
