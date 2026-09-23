# NULLXES SHINRA-25B-MULTIMODAL

Native dense SHINRA, initialized from scratch. One model family; no external model
configs, pretrained perception encoders or SHINRA-4B checkpoint migration.

**Architecture:** 25,177,818,880 trainable parameters including image/audio/video/world
components. Hidden 7168, 40 blocks, SwiGLU 19456, 32 memory blocks + 8 global GQA
blocks, 56 Q / 8 KV heads of width 128. Native design budget: **327680 combined
text/latent positions**, including generated continuation.

## Status and evidence

The repository contains executable native model, perception/decoder, world-head,
cache, loss, serialization and opt-in training code. It does **not** contain trained
weights or a trained tokenizer. Architectural support is not evidence of learned
320K capability. No full-size allocation, training, dataset/checkpoint download,
GPU workload or distributed job was performed while implementing this repository.

CPU numerical tests use a reduced instance of the same configuration contract.
They cover recurrence gradients, causal prefix independence, cached prefill,
serialization, chunked-loss gradients, multimodal gradients, streaming equivalence,
parameter accounting and episode guards. They are not evidence of convergence or
hardware throughput. See [verification](specifications/verification.md).

GPU adapters for FlashAttention, FLA, Transformer Engine, FSDP2 and DeepSpeed need
qualification in the intended compute image. Missing backends fail explicitly;
long inputs never silently fall back to a quadratic CPU/reference implementation.

## Environment ownership

**Install/use the appropriate CUDA + PyTorch environment for the target hardware before installing SHINRA dependencies.**

SHINRA does not install PyTorch. `torch` is absent from package dependencies and all
requirements. The compute image also owns CUDA-specific kernels. Install approved
non-runtime dependencies explicitly, then install this repository with:

```console
python -m pip install --no-deps --no-build-isolation -e .
```

The image must already provide setuptools, safetensors, sentencepiece and protobuf. Do not
run an unrestricted optional-package installation that resolves or replaces
PyTorch/CUDA transitively. See [environment groups](environments/README.md).

## Safe, allocation-free architecture audit

```console
python -m shinra.audit
```

This command imports no PyTorch, initializes no model and reports the exact ledger.
The architectural specification is [configs/shinra25b.json](configs/shinra25b.json).
Execution dispatch is in [configs/runtime.json](configs/runtime.json); checkpointing,
segment sizes, precision and losses are in [configs/training.json](configs/training.json).
No setting launches a workload. See [exact memory/CFG review](specifications/cfg_review.md).

```python
from shinra import ShinraConfig, ShinraRuntimeConfig, ShinraTrainingConfig
from shinra.audit import parameter_ledger

config = ShinraConfig.load("configs/shinra25b.json")
runtime = ShinraRuntimeConfig.load("configs/runtime.json")
training = ShinraTrainingConfig.load("configs/training.json")
assert parameter_ledger(config)["total"] == 25_177_818_880
```

Constructing a model above 100M parameters requires `allow_large_init=True`.
Do not pass it without a separately approved memory/runtime plan. Merely loading
a config or importing SHINRA performs no full-size allocation.
When separately authorized to construct a model, pass `runtime=runtime, training=training`
to its constructor; execution settings do not belong to the model fingerprint.

The memory identity is `shinra_channel_delta_v1`, a channel-decay delta rule with
shared scalar erase/write beta. It is not FLA's GatedDeltaNet, not KDA, and not
GDN2. `memory_backend=auto` runs the reference recurrence. One mixer has exactly
**149,971,008 parameters**. Execution chunks carry the recurrent matrix and the
convolution tail; only a sequence boundary resets memory. `8192` is that chunk
request, and `4096` is only the reference-backend cap. Neither is the 327680 context.

The architecture contract is frozen for bootstrap preparation. That is not a claim
of trained context, retrieval, world-model behavior, or GPU kernel parity. See
[architecture freeze v2](specifications/architecture_freeze_v2.md). `rotary_dim=64`
and `memory_gate_rank=256` remain experimental hypotheses inside that freeze.

## Interfaces

- `ShinraForCausalLM.forward`: IDs or native embeddings, causal shifted text labels,
  bounded optional logits and explicit cache. No implicit padded/packed episodes.
- `multimodal.image`: dynamic RGB resolution within the explicit patch budget.
- `multimodal.audio`: streaming mono waveform -> timestamped continuous latents.
- `multimodal.video`: frame/time stream -> temporal latents. Fast tokens are a
  second resample of the per-frame latents, not a separate encoder.
- `assemble_stream`: text + typed continuous latents + boundary IDs + coordinates.
- `predict_world`: observations + previous slots + action + Δt -> next slots,
  all configured horizon heads, and named reward/terminal/value/cost logits.
- `flow_matching_loss` / `sample_flow`: native visual/audio output decoders.
- `video_flow_loss` / `sample_video`: temporal conditioning + shared visual decoder;
  requires per-frame latent conditions, not a text-to-video latent planner.
- `prefill` / `generate`: chunked prefill and bounded-context text generation.
- `ShinraCache`: branch/reset/serialize recurrence, KV and modality/world side state.
- `save_checkpoint` / `load_checkpoint`: strict sharded safetensors with hashes.
- `training`: CPU master/gradient offload, grouped activation checkpointing,
  data-parallel token normalization, FSDP2/ZeRO adapters, capability replay/gates.

See [contracts](specifications/contracts.md) before constructing datasets or serving
streams. The tokenizer wrapper accepts only local, hashed SentencePiece Unigram
artifacts with byte fallback and exactly the configured vocabulary.
`131072` includes the 256 reserved controls `[130816,131071]`. Audio's 128 learned
queries form a cyclic bank; output duration scales at 12.5 latents/s, not 128/clip.

## Precision and scaling

BF16 is the reference production precision; recurrence, decay and normalization
accumulation use FP32. FP8/MXFP8 conversion is explicit and limited to selected
backbone GEMMs. It does not quantize the recurrent state or text CE.

The current distributed implementation is FSDP2 data parallelism or DeepSpeed
ZeRO, not TP/SP/CP. On two A100s, use the documented full CPU-offload profile and
measure it; the earlier theoretical TP2 throughput estimate is **not** a measured
claim about this runtime. No incompatible TP/CP flags are exposed as working features.
Global KV storage is contiguous, not a paged serving allocator. These distinctions
are deliberate and are recorded in the qualification checklist.

## Verification

In an already provisioned environment:

```console
python -m pytest -q
python -m ruff check src tests
```

Tests never construct the 25B model. The large-allocation guard is tested before
any parameter module is created. No test runs an optimizer update on a model.

## Training policy

No automatic training command is provided. Training functions must be invoked
explicitly by the approved experiment driver. Bootstrap from random initialization
is distinct from capability cycles. A capability cycle uses two focused datasets,
replay and regression gates, capped at 850 updates. Dataset selection, tokenizer
training, corpus acquisition and GPU experiments require separate authorization.

Copyright NULLXES. No redistribution license is granted by this repository.
