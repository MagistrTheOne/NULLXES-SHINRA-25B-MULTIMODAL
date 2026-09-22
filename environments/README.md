# Compute-owned dependencies

Install/use the appropriate CUDA + PyTorch environment for the target hardware before installing SHINRA dependencies.

No requirements group installs torch, torchvision, torchaudio, CUDA or GPU kernels.
Do not let optional dependency resolution modify those packages. Environment images
are created outside the project installer and recorded by their immutable digest.

| Group | Components supplied explicitly |
|---|---|
| CORE | safetensors, sentencepiece, protobuf; environment-provided PyTorch |
| TRAINING | Native PyTorch optimizer or CPU master AdamW; optional DeepSpeed for ZeRO |
| HOPPER OPTIONAL | FlashAttention-4, FLA, Transformer Engine |
| BLACKWELL OPTIONAL | FlashAttention-4/CuTeDSL, FLA, TE MXFP8 |
| AMPERE | FlashAttention-2, FLA; BF16, no FP8 assumption |
| DEV | pytest, ruff |
| HF COMPATIBILITY | transformers supplied by the environment, optional explicit registration |

Production configuration selects FA4/FLA. Ampere changes only backend dispatch to
`flash2`; it does not change weights or architecture fingerprint. Kernels must be
tested against the exact installed revision. Do not install optional runtimes from
this document automatically.

## Proposed host profiles (not measured)

- **2 × A100 80GB:** FSDP2 with CPUOffloadPolicy or ZeRO-3 CPU offload, BF16,
  activation offload, grouped checkpointing; 768GB–1TB host RAM planning envelope.
  CPUOffloadPolicy offloads parameters, gradients and optimizer, not just moments.
- **H200 141GB:** CPUAdamW master/gradient offload, BF16, activation offload.
  Qualify FP8 separately; retain BF16 reference checkpoints and regression results.
- **B300 288GB:** CPUAdamW master/moment offload, grouped activation checkpointing,
  BF16 first; MXFP8 only after forward/backward numerical and throughput comparisons.

The 25.178B default has 50.356GB BF16 weights and 402.845GB conventional 16-byte
Adam training states before activations. Standard FP8 GEMMs do not remove FP32
master/moment storage. No profile is declared to fit without measurement.

## Unsupported combinations

- CPUAdamW on sharded DTensor/FSDP parameters: rejected; let FSDP/ZeRO own states.
- Generic local runtime loop around a DeepSpeed engine: use engine.backward/step
  and its own accumulation semantics instead.
- TP/SP/CP, FP8 KV cache, quantized optimizer states and paged serving caches are
  not implemented by a configuration flag.
- Persistent inference caches with autograd enabled: rejected.
- Reference SDPA/recurrence beyond configured small-input limit: rejected.

## References used for adapters

- https://github.com/Dao-AILab/flash-attention
- https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py
- https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html
- https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html

These are operation/runtime dependencies, not imported model architectures.
