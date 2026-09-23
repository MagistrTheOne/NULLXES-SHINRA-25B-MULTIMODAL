# Verification and release gates

## Executed in this implementation session

- Final result: **51 CPU tests passed** (pytest), Ruff lint passed, Ruff format
  check passed, and compileall passed. No CUDA workload was used. The duration of
  that CPU run is test time, not model throughput.
- Allocation-free audit independently returned total 25,177,818,880, core
  23,413,976,064, multimodal 1,763,842,816 and context 327680. `torch` was absent
  from `sys.modules` after this audit.
- Static configuration/parameter calculations (no full-size model allocation).
- CPU tests using reduced same-family tensors/modules, including autograd checks.
- CPU-only local tiny checkpoint serialization/corruption checks.
- Source lint/format checks.
- Streaming cache storage tests verify actual backing-storage size, not just view
  dimensions; short history slices cannot retain an entire prefill allocation.
- HF adapter/native output equality, greedy cached/manual generation equality,
  offloaded/model gradient equivalence and counterfactual cache isolation.
- CFG and execution-contract verification: memory matrix/convolution/gradient
  equivalence across chunks and across separate forwards, reference-only memory
  dispatch, positional fingerprint, production-profile rejection of topology
  drift, world-state carry, multi-horizon outputs, counterfactual branch
  isolation, and audio/video geometry. Last full run: 51 passed on 2026-09-24.
  That duration is test time, not model throughput.

No optimizer update on a model, training run, GPU query/workload, distributed job,
dataset download, pretrained checkpoint download or full-size model initialization
was performed. No trained capabilities are claimed.

## Required before approving hardware use

1. Fix image digest and exact PyTorch/CUDA/driver/kernel versions.
2. Run small GPU numerical equivalence and backward checks against CPU equations.
3. Check FLA gradient flow through initial/final state across all segment cuts.
4. Check FA2/FA4 cached causal alignment, GQA and partial RoPE.
5. Verify FSDP2/ZeRO resumption, tied embeddings and unequal-token loss normalization.
6. Measure host RAM, PCIe traffic, actual peak allocation and update time.
7. Qualify TE recomputation and quantization state checkpoint/resume separately.
8. Escalate sequence lengths gradually; exercise 327680 with real cross-context tasks.
9. Bootstrap acceptance: held-out language/code/math/multilingual/world metrics.
10. Long-context release gate: retrieval, conflicting updates, temporal ordering,
    action-conditioned rollout, multi-hop cross-modal reasoning at full length.

## Remaining engineering/performance boundaries

The implementation supplies FSDP2/ZeRO data-parallel/offload paths. It does not
implement TP/SP/CP for the hybrid recurrence. GPU adapter code is not yet qualified
on A100/H200/B300. Global KV is contiguous, not paged. Frontend attention is a
bounded native implementation. These are explicit limits, not flags claiming
features that have not been implemented.

The proposed report's theoretical throughput numbers must not be reused as measured
performance of this code. Meeting the native usable-context requirement additionally
requires training and evaluation; a configuration integer or correctness test cannot
certify acquisition.
