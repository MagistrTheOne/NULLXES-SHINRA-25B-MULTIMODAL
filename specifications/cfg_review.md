# CFG review: mathematical identity, exact count, unresolved quality hypotheses

Baseline checked against local/remote main `0ca4668a6d0c77afa8a819090fc6ce53c228ce49`.
No training, GPU workload, full-model initialization or weight/dataset download.

## One memory mixer: native SHINRA, not fla.layers.GatedDeltaNet

`memory_type = shinra_channel_delta_v1`; `memory_rule = coupled_delta`.
The family architecture version remains 2. These are independent version namespaces.
Backend selection (`auto`, `reference`, `fla`) does not define mathematics or shapes.

Let H=7168, J=32, K=V=128, M=J*K=4096, R=256. There is **no value expansion**,
no grouped extra value heads, no imported FLA layer and no 6H² mixer assumption.
Actual parameter names/shapes are exported by `memory_parameter_shapes` and checked
against every parameter of an instantiated reduced same-contract mixer.

PyTorch storage order is [out_features, in_features]:

| Trainable tensor | Shape | Parameters |
|---|---|---:|
| q.weight | [4096,7168] | 29,360,128 |
| k.weight | [4096,7168] | 29,360,128 |
| v.weight | [4096,7168] | 29,360,128 |
| o.weight | [7168,4096] | 29,360,128 |
| output_gate.weight | [4096,7168] | 29,360,128 |
| gate_down.weight | [256,7168] | 1,835,008 |
| decay_up.weight | [4096,256] | 1,048,576 |
| beta.weight | [32,7168] | 229,376 |
| beta.bias | [32] | 32 |
| dt_bias | [4096] | 4,096 |
| A_log | [32] | 32 |
| conv_weight: Q/K/V depthwise, bias-free | [3,4096,4] | 49,152 |
| norm.weight: per-head RMS affine | [32,128] | 4,096 |
| **Total mixer** | | **149,971,008** |

Separate erase/write matrices: **none**. Both edits share beta, one scalar per head
and token. No trainable initial state. Q/K L2 normalization has zero parameters.
Each of the three short convolutions contributes 16,384 parameters.

Full **memory block**, including FFN and two pre-norms:

    149,971,008 + 3*(7168*19456) + 2*7168 = 568,367,168

Full **global block**, including FFN, pre-norms and QK norm:

    2*7168² + 2*7168*(8*128) + 3*7168*19456 + 2*7168 + 2*128
    = 535,836,928

## Exact mathematical operation

For each token x and head:

    q,k,v = SiLU(causal_depthwise_conv(Wq*x, Wk*x, Wv*x))
    q,k = L2_normalize(q), L2_normalize(k)
    u = SiLU(W_gate_down*x)
    log_alpha = -exp(A_log) * softplus(W_decay_up*u + dt_bias)
    beta = sigmoid(W_beta*x + beta_bias)
    prior = diag(exp(log_alpha)) * previous_state
    state = prior + beta * outer(k, v - transpose(prior)*k)
    read = transpose(state)*q
    y = Wo * (flatten(per_head_RMSNorm(read)) * SiLU(W_output_gate*x))

State is [batch,32,128,128], FP32. Reference and optimized paths have the same
declared operation: FLA receives normalized Q/K, log-decay, post-sigmoid beta and
scale=1. Only `fla.ops.kda.chunk_kda` is imported. GPU parity is still unqualified;
we do not replace that test with a mocked backend.

For unit k and beta in [0,1], the homogeneous transition `(I-beta*k*kT)*D`
has spectral norm <=1 when all diagonal D entries are in (0,1]. This does not
guarantee bounded forced state, stable long gradients or sufficient memory capacity.

## Exact whole model

The complete component-by-component JSON is `parameter_ledger.json`.

| Component | Exact parameters |
|---|---:|
| Token embeddings / tied output weight | 939,524,096 |
| 32 memory mixers (including their gates/conv/output norm) | 4,799,072,256 |
| 8 global Q projections | 411,041,792 |
| 8 global K projections | 58,720,256 |
| 8 global V projections | 58,720,256 |
| 8 global O projections | 411,041,792 |
| 40 FFN gate projections | 5,578,424,320 |
| 40 FFN up projections | 5,578,424,320 |
| 40 FFN down projections | 5,578,424,320 |
| 80 pre-RMSNorm vectors | 573,440 |
| 8 global QK norms | 2,048 |
| Final norm | 7,168 |
| Additional LM head weights | 0 |
| **Core** | **23,413,976,064** |
| Vision encoder + patch input + final norm | 839,931,136 |
| Audio encoder + stem + final norm | 411,586,560 |
| Visual resampler input projection | 1,310,720 |
| Visual resampler + queries + final norm | 68,171,264 |
| Audio resampler + query bank + final norm | 67,253,760 |
| Temporal module + time input + final norm | 67,184,128 |
| Six modality projections + six interface norms | 44,046,336 |
| Type/coordinate metadata | 1,146,880 |
| World slots + extractor | 33,594,624 |
| World future mean/log-variance | 12,582,912 |
| World action input/type/output | 327,680 |
| World continuous time / horizons | 73,728 |
| World event / reward / terminal heads | 266,240 |
| Visual flow decoder | 136,923,136 |
| Audio flow decoder | 79,443,712 |
| Additional video-specific weights (shared modules) | 0 |
| **Multimodal** | **1,763,842,816** |
| **TOTAL** | **25,177,818,880** |
| **Difference from 25,000,000,000** | **+177,818,880 (+0.71127552%)** |

## GDN2 assessment: not rejected, not silently substituted

The current implementation is channel-decay, coupled-delta (KDA-like), **not ordinary
scalar-decay GDN** and not GDN2. There is no evidence that it is inherently better
than GDN2 for SHINRA-25B. Keeping it identifies the actually implemented/tested
baseline; it is not a final quality approval.

A concrete same-width GDN2-style proposal replaces scalar beta by two channel gates
from the existing rank-256 trunk:

    erase = sigmoid(W_erase*u + bias_erase)    # 4096 channels
    write = sigmoid(W_write*u + bias_write)   # 4096 channels
    e = erase * k
    state = (I-k*transpose(e))*prior + outer(k,write*v)

Remove beta.weight/bias: 229,408 per mixer. Add two [4096,256] matrices and two
[4096] biases: 2,105,344. Net **+1,875,936 per mixer**, **+60,029,952** over 32.
Proposed mixer 151,846,944; proposed whole model **25,237,848,832**. This factorized
gate parameterization is a SHINRA proposal, not a claimed copy of NVIDIA's layer.

Independent channel erase makes `(I-k*(erase*k)T)` generally non-normal. For example,
k=(1,1)/sqrt(2), erase=(1,0), D=I produces [[0.5,0],[-0.5,1]]: eigenvalues .5 and 1,
but spectral norm >1. Eigenvalue bounds alone do not prove contractivity. This
requires state/gradient stability checks and comparative retrieval/world-state
training, not a boolean pretending to implement delta_v2. Those experiments and
the replacement are not authorized by a CFG review alone.

Sources checked: [FLA layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py),
[FLA KDA operation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py),
[NVIDIA GDN2](https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention).

## Four choices requiring rationale

### Partial RoPE 64 / head 128

With theta=1e6, periods are `2*pi*theta^(2*i/rotary_dim)`; all alternatives rotate
the fastest pair with period 2*pi. Rotating more dimensions increases frequency
sampling density, not maximum native context by itself.

| Rotary dim | Rotary pairs | Slowest period (tokens) | Pairs with period >=327680 | Unrotated channels |
|---|---:|---:|---:|---:|
| 64 | 32 | 4,080,185 | 6 | 64 |
| 96 | 48 | 4,711,724 | 10 | 32 |
| 128 | 64 | 5,063,256 | 13 | 0 |

64 reserves half the Q/K space for position-invariant content matching. 128 provides
denser positional frequencies; 96 is intermediate. **No measurement establishes
64 as superior.** It remains the proposed ablation baseline, not an approved quality
fact. All three have the same parameter count. Use retrieval/order/cross-modal/world
acceptance at native 327680 before selecting a release value.

No extrapolation or YaRN is being applied: 327680 is the native training target.
Frequency coverage alone is not proof of usable context. Memory Q/K do not use
RoPE; order comes from recurrence. Image x/y are encoded in the visual frontend;
video/audio timestamps in temporal features; typed physical coordinates enter the
backbone through its metadata projection. Global RoPE encodes sequence order, not
a mixture of token indices, pixels and seconds in one axis.

### Gate rank 256

This is the nonlinear bottleneck for **channel decay only**, not Q/K/V, output gate
or FFN. `7168*256 + 256*4096 = 2,883,584` weights per mixer, versus 29,360,128 for a
direct decay matrix. Its local input/output Jacobian has rank at most 256. That is
a real expressivity tradeoff; independent beta and output gates are not bottlenecked.
Rank 128 reduces whole-model parameters by 46,137,344; rank 512 adds 92,274,688.
All are Tensor-Core-friendly. 256 is a cost/capacity hypothesis requiring ablation.

### Training segment 8192

Renamed to `ShinraTrainingConfig.memory_train_segment_size`. Each segment receives
the previous final matrix and returns a differentiable next matrix. There is no
reset or detach at a segment boundary. `memory_state_reset=sequence_boundary` is
the architectural rule. Numerical tests compare values, final state and every
gradient across segment sizes 1,3,5,7 versus an unsplit sequence.

One FP32 state per memory layer =2MiB; all 32 =64MiB. At length 327680, retaining
40 segment boundary sets costs 2.5GiB, versus 160GiB if every 128-token chunk boundary
were retained. These are boundary-state counts, not a claim about total peak memory.
8192 is a tunable recomputation/memory policy, never a context-length limit.

### Hardware dispatch

`attention_backend=auto` is now only in runtime.json. Pure dispatch policy: Ampere
FA2/cuDNN; Hopper FA3/cuDNN/eligible FA4/FA2; Blackwell FA4/cuDNN/FA2. Availability,
dtype, head shape and causal alignment are checked. No backend is installed by code.
cuDNN is allowed only for masks representable without a dense SxS allocation.
Unsupported fused paths raise; there is no unbounded math fallback. CPU/reference
uses bounded SDPA. Kernel eligibility/performance still requires hardware tests.

[TE backend selection](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html)
and [FA4 prerelease](https://pypi.org/project/flash-attn-4/4.0.0b19/) justify keeping
kernel versions out of the architecture contract. Auto policy is deterministic
eligibility-based dispatch, not a claim to select the fastest kernel without profiling.

## Remaining clarified semantics

- `audio_query_bank_size=128`: learned cyclic query bank. **One query is used per
  two 25Hz encoder frames**, so 10s ->125 latents, 60s ->750, 10min ->7500. It is
  neither a clip-wide cap nor 128 outputs per arbitrary caller chunk. The model
  preserves absolute frame phase across API chunks.
- `vocab_size=131072` TOTAL. 256 controls are inside it: **[130816,131071]**.
  Tokenizer and stream assembly reject control IDs outside this range.
- World latent dimension is exactly shared `latent_dim=1024`; explicit property
  `world_latent_dim` exposes it. Continuous time/counterfactual/reset semantics are
  in the model contract.
- `vision_generation=true`, `audio_generation=true`, video generation is explicitly
  `latent_conditioned_shared_visual_flow`. The implemented video decoder consumes
  per-frame latents, temporally conditions them and invokes shared visual flow.
  **There is no implemented text-to-video latent planner or text-only media generation
  pipeline.** These flags describe output heads, not acquired text-to-media capability.
- Architecture, training and runtime are separate files. Checkpoints save all three;
  only the architectural contract determines model/cache fingerprint. Old mixed
  configuration files must be explicitly split; they are not silently reinterpreted.
