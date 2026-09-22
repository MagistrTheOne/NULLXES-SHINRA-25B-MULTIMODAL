import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .normalization import RMSNorm
from .cache import MemoryState
from .kernels import delta_chunk
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig


class ShinraMemoryMixer(nn.Module):
    def __init__(self, c, runtime=None, training=None):
        super().__init__()
        self.config = c
        self.runtime = runtime or ShinraRuntimeConfig()
        self.training_config = training or ShinraTrainingConfig()
        h, m = c.hidden_size, c.memory_heads * c.memory_key_dim
        self.q = nn.Linear(h, m, bias=False)
        self.k = nn.Linear(h, m, bias=False)
        self.v = nn.Linear(h, m, bias=False)
        self.o = nn.Linear(m, h, bias=False)
        self.output_gate = nn.Linear(h, m, bias=False)
        self.gate_down = nn.Linear(h, c.memory_gate_rank, bias=False)
        self.decay_up = nn.Linear(c.memory_gate_rank, m, bias=False)
        self.beta = nn.Linear(h, c.memory_heads, bias=True)
        self.dt_bias = nn.Parameter(torch.empty(m))
        self.A_log = nn.Parameter(torch.zeros(c.memory_heads))
        self.conv_weight = nn.Parameter(torch.empty(3, m, c.memory_conv_kernel))
        self.norm = RMSNorm(c.memory_value_dim, c.norm_eps)
        # Separate per-head affine norm, unlike a shared-head QK norm.
        self.norm.weight = nn.Parameter(torch.ones(c.memory_heads, c.memory_value_dim))
        nn.init.normal_(self.conv_weight, std=c.memory_conv_kernel**-0.5)
        self.reset_gates()

    def reset_gates(self):
        c = self.config
        with torch.no_grad():
            tau = torch.logspace(
                math.log10(32),
                math.log10(2 * c.max_context_length),
                c.memory_heads,
                device=self.dt_bias.device,
            )
            self.dt_bias.copy_(torch.log(torch.expm1(1 / tau)).repeat_interleave(c.memory_key_dim))
            self.A_log.zero_()
            self.beta.bias.fill_(-2.197224577)

    def forward(self, x, state=None):
        c = self.config
        b, s, _ = x.shape
        runtime, policy = self.runtime, self.training_config
        backend = runtime.memory_backend
        if backend == "auto":
            backend = "fla" if x.device.type == "cuda" else "reference"
        m, j, d = c.memory_heads * c.memory_key_dim, c.memory_heads, c.memory_key_dim
        if backend == "reference" and s > runtime.reference_backend_max_tokens:
            raise RuntimeError("Reference recurrence is bounded; select qualified FLA for long training")
        projected = torch.stack((self.q(x), self.k(x), self.v(x)), dim=1).transpose(-1, -2)
        history = projected.new_zeros(b, 3, m, c.memory_conv_kernel - 1) if state is None else state.conv
        joined = torch.cat((history, projected), dim=-1)
        convolved = F.conv1d(
            joined.reshape(b, 3 * m, -1), self.conv_weight.reshape(3 * m, 1, -1), groups=3 * m
        )
        convolved = F.silu(convolved.reshape(b, 3, m, s).transpose(-1, -2))
        q, k, v = [t.reshape(b, s, j, d) for t in convolved.unbind(1)]
        q = F.normalize(q.float(), dim=-1, eps=c.norm_eps)
        k = F.normalize(k.float(), dim=-1, eps=c.norm_eps)
        gate = self.decay_up(F.silu(self.gate_down(x))).float().reshape(b, s, j, d)
        decay = -self.A_log.float().exp()[None, None, :, None] * F.softplus(
            gate + self.dt_bias.float().view(j, d)
        )
        beta = self.beta(x).float().sigmoid()
        matrix = (
            torch.zeros(b, j, d, d, device=x.device, dtype=torch.float32) if state is None else state.matrix
        )
        outputs = []
        # Segment boundaries checkpoint/recompute; they NEVER reset/detach state.
        for start in range(0, s, policy.memory_train_segment_size):
            end = min(start + policy.memory_train_segment_size, s)
            args = (
                q[:, start:end],
                k[:, start:end],
                v[:, start:end].float(),
                decay[:, start:end],
                beta[:, start:end],
                matrix,
            )

            def run(q_, k_, v_, g_, b_, s_):
                if backend == "fla":
                    # Preserve state/gates FP32; kernel GEMMs use the activation dtype.
                    return delta_chunk(q_.to(x.dtype), k_.to(x.dtype), v_.to(x.dtype), g_, b_, s_, "fla")
                return delta_chunk(q_, k_, v_, g_, b_, s_, "reference")

            if self.training and policy.gradient_checkpointing and torch.is_grad_enabled():
                out, matrix = checkpoint(run, *args, use_reentrant=False)
            else:
                out, matrix = run(*args)
            outputs.append(out)
        output = self.norm(torch.cat(outputs, dim=1).to(x.dtype)).flatten(-2)
        y = self.o(output * F.silu(self.output_gate(x)))
        # A narrow view would retain the entire prefill projection storage in the cache.
        return y, MemoryState(matrix.float(), joined[..., -(c.memory_conv_kernel - 1) :].clone())
