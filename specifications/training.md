# Bootstrap and capability execution

Random-init bootstrap and focused capability acquisition are separate experiment
types. No runner starts automatically and no corpus is fetched by this repository.

Bootstrap begins with curated language/explanations, code with tests, verified math,
multilingual material and paired/procedural modalities. Native positional frequencies
do not change later. Long examples are introduced during bootstrap, not as a post-hoc
extension. Proportions and budgets are experiment inputs; they are not evidence that
100B or any other number of tokens will be sufficient.

CapabilityCycle validates two distinct focused dataset identities, replay fraction,
evaluation cadence and the 850-update ceiling. ReplaySampler persists RNG state and
draw count. Dataset revisions/hashes must be held fixed and recorded by the caller.
Regression gates compare all required metrics, rejecting missing metrics.

`optimizer_update` normalizes causal text CE over valid targets across microbatches.
The data_parallel mode additionally accounts for DDP/FSDP gradient averaging.
Each rank must execute the same number of microbatches. The caller owns device
placement and must avoid retaining all raw videos/graphs in GPU memory.

`CPUAdamW` keeps FP32 CPU master parameters and moments. Optional post-accumulation
hooks move gradients to CPU and clear GPU .grad storage. It is not a sharded optimizer.
FSDP/ZeRO have separate state ownership; combining them with CPUAdamW is rejected.
No 8-bit optimizer or FP8 master-weight mode is advertised.

For custom multimodal objectives, supply a callable returning (loss_sum, valid_count)
and the batch's valid_count. Weight modality losses explicitly per valid target;
do not normalize image/video reconstruction by a text-token count accidentally.
Future-state targets must not appear as conditioning observations before the predicted
transition. Use flow_matching_loss, future_state_loss and latent_regularization as
building blocks and record the actual mixture weights in the experiment manifest.

The default hardware path is BF16. FP8/MXFP8 are optional TE GEMM conversions, require
matching hardware and numerical qualification, and do not remove FP32 recurrence or
optimizer master states. Save TE extra state with distributed checkpointing.
