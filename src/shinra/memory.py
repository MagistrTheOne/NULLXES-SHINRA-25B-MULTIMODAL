import math
import torch
from torch import nn
import torch.nn.functional as F
from .normalization import RMSNorm
from .cache import MemoryState
from .kernels import delta_chunk
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig, execution_segment_size


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

    def _backend(self):
        name = self.runtime.memory_backend
        if name in ("auto", "reference"):
            return "reference"
        raise RuntimeError(
            "memory_backend does not select fla.ops.kda.chunk_kda. "
            "That kernel is not shinra_channel_delta_v1; auto runs the reference recurrence"
        )

    def _chunk(self, xs, history, matrix):
        """One causal chunk: projections, convolution, gates and the delta scan.

        Full BPTT: history and matrix are returned as live tensors. Nothing is detached.
        """
        c = self.config
        b, s, _ = xs.shape
        m, j, d = c.memory_heads * c.memory_key_dim, c.memory_heads, c.memory_key_dim
        projected = torch.stack((self.q(xs), self.k(xs), self.v(xs)), dim=1).transpose(-1, -2)
        joined = torch.cat((history, projected), dim=-1)
        convolved = F.conv1d(
            joined.reshape(b, 3 * m, -1), self.conv_weight.reshape(3 * m, 1, -1), groups=3 * m
        )
        convolved = F.silu(convolved.reshape(b, 3, m, s).transpose(-1, -2))
        q, k, v = [t.reshape(b, s, j, d) for t in convolved.unbind(1)]
        q = F.normalize(q.float(), dim=-1, eps=c.norm_eps)
        k = F.normalize(k.float(), dim=-1, eps=c.norm_eps)
        gate = self.decay_up(F.silu(self.gate_down(xs))).float().reshape(b, s, j, d)
        decay = -self.A_log.float().exp()[None, None, :, None] * F.softplus(
            gate + self.dt_bias.float().view(j, d)
        )
        beta = self.beta(xs).float().sigmoid()
        out, matrix = delta_chunk(q, k, v.float(), decay, beta, matrix, self._backend())
        y = self.o(self.norm(out.to(xs.dtype)).flatten(-2) * F.silu(self.output_gate(xs)))
        # Own the kernel tail. A view would pin the whole chunk projection in the cache.
        return y, joined[..., -(c.memory_conv_kernel - 1) :].clone(), matrix

    def forward(self, x, state=None):
        c = self.config
        b, s, _ = x.shape
        m, j, d = c.memory_heads * c.memory_key_dim, c.memory_heads, c.memory_key_dim
        self._backend()
        width = execution_segment_size(self.training_config, self.runtime, device_type=x.device.type)
        history = x.new_zeros(b, 3, m, c.memory_conv_kernel - 1) if state is None else state.conv
        matrix = (
            torch.zeros(b, j, d, d, device=x.device, dtype=torch.float32) if state is None else state.matrix
        )
        outputs = []
        # Chunks carry live state. Full BPTT stays intact; truncated BPTT is not this path.
        for start in range(0, s, width):
            y, history, matrix = self._chunk(x[:, start : start + width], history, matrix)
            outputs.append(y)
        return torch.cat(outputs, dim=1), MemoryState(matrix.float().clone(), history)
