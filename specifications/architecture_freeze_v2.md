# NULLXES SHINRA-25B-MULTIMODAL

Architecture Baseline V2

Status:
FROZEN FOR BOOTSTRAP PREPARATION

Parameters:
25,177,818,880 trainable

Deviation from nominal 25B:
+177,818,880 (+0.7113%)

Architecture:
Dense proprietary SHINRA

Initialization:
From scratch

Context target:
327,680

This is the native design length. It is not a measured retrieval length.

Backbone:
23,413,976,064

Multimodal/world/decoders:
1,763,842,816

Memory topology:
32 coupled channel-wise delta memory layers
8 global GQA layers
MMMMG × 8

Memory rule:
shinra_channel_delta_v1 / coupled_delta / shared scalar beta

Execution:
Chunks carry the FP32 recurrent matrix and the convolution tail.
Projections, convolution, gates and the scan stay inside the chunk.
Full BPTT. Truncated BPTT is not implemented.
8192 is the requested chunk. 4096 is only the reference-backend cap.

Global attention:
Causal over the carried KV prefix. Query chunks are not local windows.
SDPA refuses a prefix above the reference cap. Fused kernels may exceed it.

Memory backend:
auto selects the reference recurrence.
fla.ops.kda.chunk_kda is not this recurrence and is not dispatched.

Positional contract:
Text/global partial RoPE, rotary_dim=64, theta=1e6, token index.
Memory has no RoPE.
Image spatial RoPE, theta=10000, plus visual_window=16 and global cadence 4.
Audio full-head frame-index RoPE, theta=1e6.
Temporal Fourier features plus full-head RoPE on the availability timestamp, theta=1e6.

World:
Previous slots enter the next transition.
Reset clears that state.
Horizons are delta_t times (1, 2, 4, 8, 16, 32, 64, 128).
Auxiliary channels are reward, terminal, value, cost.
Counterfactual actions branch without writing the source state.

Video:
spatial tokens 64, fast tokens 8, frame group 4.
Fast tokens are a second resample of per-frame latents.
No text-to-video planner.

Audio:
Stem kernel 480 / stride 240, then two kernel-4 stride-2 stages.
2 encoder frames per latent. 128-query bank is cyclic, not a duration cap.

Fingerprint:
72829df8abf33d673c7518c618a05383f15dfaa73415cc12c09cd6d27734c1ba

Contract versions:
architecture 2, cache 2, memory 2, world 2

Known experimental hypotheses:
rotary_dim=64
memory_gate_rank=256

Known unverified properties:
GPU backend performance
A fast kernel that matches the reference delta
327,680 useful-context acquisition
training convergence
world-model capability acquisition
multimodal quality
generation quality

Known missing capability:
finished text-to-video latent planner
trained weights
trained tokenizer
