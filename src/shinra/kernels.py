"""Backend adapters import operations only; never third-party model layers/configs."""

import torch
import torch.nn.functional as F


def attention(q, k, v, *, backend="sdpa", causal=True, offset=0, window=None, reference_limit=4096):
    # q/k/v are B,S,H,D. Cached decoding uses bottom-right aligned causality.
    if backend in ("flash2", "flash4"):
        if q.device.type != "cuda":
            raise RuntimeError("FlashAttention requires an environment-provided CUDA runtime")
        if backend == "flash2":
            from flash_attn import flash_attn_func
        else:
            from flash_attn.cute.interface import flash_attn_func
        kwargs = {"causal": causal}
        if window is not None:
            kwargs["window_size"] = (window - 1, 0 if causal else window - 1)
        result = flash_attn_func(q, k, v, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    if max(q.shape[1], k.shape[1]) > reference_limit:
        raise RuntimeError("Long attention requires an explicit qualified FlashAttention backend")
    mask = None
    simple_causal = causal and offset == 0 and q.shape[1] == k.shape[1] and window is None
    if not simple_causal and (causal or window is not None):
        qi = torch.arange(q.shape[1], device=q.device)[:, None] + offset
        ki = torch.arange(k.shape[1], device=q.device)[None, :]
        mask = ki <= qi if causal else torch.ones_like(qi - ki, dtype=torch.bool)
        if window is not None:
            mask = mask & ((qi - ki) < window)
            if not causal:
                mask = mask & ((ki - qi) < window)
    groups = q.shape[2] // k.shape[2]
    # Reference path expands GQA explicitly. Production flash kernels do not.
    k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        is_causal=simple_causal,
        dropout_p=0.0,
    )
    return y.transpose(1, 2)


def delta_reference(q, k, v, log_decay, beta, state):
    """Differentiable exact recurrence; tensor inputs support FP64 gradcheck."""
    outputs = []
    for index in range(q.shape[1]):
        state = log_decay[:, index].exp().unsqueeze(-1) * state
        key = k[:, index]
        error = v[:, index] - torch.einsum("bhk,bhkv->bhv", key, state)
        state = state + beta[:, index, :, None, None] * key.unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, index], state))
    return torch.stack(outputs, dim=1), state


def delta_chunk(q, k, v, log_decay, beta, state, backend):
    if backend == "reference":
        return delta_reference(q, k, v, log_decay, beta, state)
    if q.device.type != "cuda":
        raise RuntimeError("FLA backend requires CUDA; no silent fallback")
    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=log_decay,
        beta=beta,
        initial_state=state,
        scale=1.0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        disable_recompute=False,
    )
