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

Architecture is separate from `ShinraRuntimeConfig` and `ShinraTrainingConfig`.
See `cfg_review.md` and `memory_ledger.json` for every memory tensor's exact shape.
The single implemented rule is channel decay + coupled scalar beta, not GDN2.
`memory_train_segment_size` is the requested execution chunk. On the reference
memory backend and on SDPA it is clamped by `reference_backend_max_tokens`. Neither
number is the 327680 context. Global attention stays causal over the carried KV
prefix; query chunks are not windows. Fused backends may hold a KV prefix longer
than the reference cap. CPU/SDPA refuses that prefix instead of building it.
Default control IDs belong to [130816,131071] inside the total vocabulary.

## Recurrence

For normalized q/k, per-channel decay D and scalar beta in [0,1]:

    prior = D * state
    state = prior + beta * outer(k, v - transpose(prior) @ k)
    output = transpose(state) @ q

State and decay arithmetic use FP32. Zero initial state is runtime data. Causal
Q/K/V depthwise convolutions retain exactly kernel-1 history elements. Projections,
convolution, gates and the scan all run inside the execution chunk. The carried
state is the FP32 matrix plus that convolution tail. Full sequence execution and
chunk-then-carry execution are the same recurrence. Training never detaches that
state; truncated BPTT is not implemented. `memory_backend=auto` selects this
reference recurrence only. `fla.ops.kda.chunk_kda` is a different operator and is
rejected. A future accelerated kernel has to match this equation before dispatch
can select it.

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

## Positional systems

These are separate on purpose. Each theta and dimension is a `ShinraConfig` field,
so changing it changes the architecture fingerprint without changing the parameter
count. `rotary_dim=64` and `memory_gate_rank=256` stay the baseline and are marked
experimental hypotheses, not measured quality results.

- Text/global: partial RoPE, `rotary_dim=64`, `rope_theta=1e6`, unit = token index.
  Memory layers do not rotate. Their order is the recurrent state.
- Image: x/y spatial RoPE, `spatial_rope_theta=10000`, half the head for each axis.
  Visual blocks are global every `visual_global_cadence` layers and otherwise use
  `visual_window` tiles.
- Audio: full-head RoPE on encoder frame index, `audio_rope_theta=1e6`.
- Video/temporal: Fourier features of the availability timestamp plus full-head
  RoPE on that same timestamp, `temporal_rope_theta=1e6`. Frame x/y stay in the
  image encoder. There is no text-to-video planner.

Video compression looks at up to four frames. All tokens emitted by such a group
carry the group's last timestamp as their availability time. A real-time caller
must not use these tokens before that time. Partial groups flush only on final=True.

## Streaming audio

Causal convolution retains phase and sample history across arbitrary chunk cuts.
Encoder local KV is bounded to window-1 prior frames. Resampling groups
`audio_frames_per_latent` encoder frames (2 in the baseline). A partial group is
carried across nonfinal calls. Query slot cycling uses the absolute frame number.

The stem is `audio_stem_kernel`/`audio_stem_stride` followed by
`audio_stem_stages` convolutions of `audio_conv_kernel` and `audio_conv_stride`.
Baseline: kernel 480, stride 240, then two kernel-4 stride-2 stages. At 24kHz that
is 25 encoder frames/s and 12.5 latents/s. Timestamps use that hop. Boundary tokens
are added by stream assembly at caller-defined events.
Zero-length audio calls are rejected; terminate the last nonempty call with final=True.

## Streaming video

Image encoder/resampler is shared. `video_spatial_tokens` (64) is the per-frame
resampler count. `video_fast_tokens` (8) is a second pass of that same resampler
over those per-frame latents, not an independent encoder. `video_frame_group` (4)
is the temporal group. Slow tokens are the temporal module output resampled to
`video_spatial_tokens`. These three fields are in the architecture fingerprint.
Frame rate is a data contract. The cache retains at most `frame_group-1` pending
encoded frames. No text-to-video planner exists.

## World prediction

A transition reads current observations, the previous slot tensor when the episode
already has one, the action, and a nonnegative base `delta_t`. The slot resampler
sees previous slots concatenated with the new observations. Episode reset drops
that tensor; the next transition has no world history. Training keeps the next
state differentiable. The inference cache stores a detached copy and does not
write into the source cache.

`world_horizon_steps` are multipliers of the caller `delta_t`. The baseline is
`1,2,4,8,16,32,64,128`. One backbone transition feeds every horizon head. This is
not an 8-step simulator. The learned horizon table is a per-offset bias.

The four auxiliary channels, in order, are reward, terminal, value and cost.
They are named logits on the updated slot, not an unexplained width.

Counterfactual calls reuse one source cache and different actions. Each call
returns its own branch. The source slots and prefix length stay unchanged.
No real actuator is controlled by the model API.

Future target latents are detached in prediction loss. The target perception
representation also needs reconstruction/variance/consistency objectives to prevent
collapse. Mean/log-variance heads are diagonal distributions, not a guarantee of
calibrated multimodal futures or physical correctness.

## Output decoders

Image and audio use conditional rectified-flow objectives and midpoint integration.
They are randomly initialized SHINRA modules, not external VAE/CLIP/Whisper models.
The visual decoder sees spatial positions. Audio output has
`audio_decoder_expand` waveform patches per latent (4 in the baseline). Full
quality, temporal fidelity and synthesis stability require training.
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
