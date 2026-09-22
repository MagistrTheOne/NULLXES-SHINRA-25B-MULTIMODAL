from torch import nn
import torch
import torch.nn.functional as F
from .checkpointing import activation_checkpoint
from .normalization import RMSNorm
from .attention import ShinraGlobalAttention
from .memory import ShinraMemoryMixer


class SwiGLU(nn.Module):
    def __init__(self, hidden, intermediate, chunk=2048):
        super().__init__()
        self.gate, self.up = (
            nn.Linear(hidden, intermediate, bias=False),
            nn.Linear(hidden, intermediate, bias=False),
        )
        self.down = nn.Linear(intermediate, hidden, bias=False)
        self.chunk = chunk

    def forward(self, x):
        def run(part):
            return self.down(F.silu(self.gate(part)) * self.up(part))

        # Checkpoint each FFN chunk so hidden intermediates are not all retained.
        parts = []
        for part in x.split(self.chunk, dim=1):
            parts.append(
                activation_checkpoint(run, part, use_te=getattr(self, "_shinra_te", False))
                if self.training and torch.is_grad_enabled()
                else run(part)
            )
        return torch.cat(parts, dim=1)


class ShinraBlock(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.norm1, self.norm2 = RMSNorm(c.hidden_size, c.norm_eps), RMSNorm(c.hidden_size, c.norm_eps)
        self.mixer = ShinraMemoryMixer(c) if kind == "memory" else ShinraGlobalAttention(c)
        self.mlp = SwiGLU(c.hidden_size, c.intermediate_size, c.mlp_chunk_size)

    def forward(self, x, state=None, offset=0):
        result, state = (
            self.mixer(self.norm1(x), state)
            if self.kind == "memory"
            else self.mixer(self.norm1(x), state, offset)
        )
        x = x + result
        return x + self.mlp(self.norm2(x)), state
