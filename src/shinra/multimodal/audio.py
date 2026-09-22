import torch
from torch import nn
import torch.nn.functional as F
from .layers import FrontendBlock, Resampler
from ..normalization import RMSNorm


def causal_conv_stream(x, conv, stride, state, prefix):
    kernel = conv.weight.shape[-1]
    history = state.get(prefix + ".history", x.new_zeros(x.shape[0], x.shape[1], kernel - 1))
    consumed = int(state.get(prefix + ".consumed", torch.tensor(0)))
    joined = torch.cat((history, x), -1)
    dense = F.conv1d(joined, conv.weight, bias=None)
    out = dense[..., (-consumed) % stride :: stride]
    state[prefix + ".history"] = joined[..., -(kernel - 1) :].clone()
    state[prefix + ".consumed"] = torch.tensor(consumed + x.shape[-1])
    return F.silu(out)


class ShinraAudioFrontend(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.stem = nn.ModuleList(
            [
                nn.Conv1d(1, c.audio_dim, c.audio_stem_kernel, bias=False),
                nn.Conv1d(c.audio_dim, c.audio_dim, 4, bias=False),
                nn.Conv1d(c.audio_dim, c.audio_dim, 4, bias=False),
            ]
        )
        self.layers = nn.ModuleList(
            [
                FrontendBlock(c.audio_dim, c.audio_ffn, c.audio_heads, c.norm_eps)
                for _ in range(c.audio_layers)
            ]
        )
        self.norm = RMSNorm(c.audio_dim, c.norm_eps)
        self.resampler = Resampler(
            c.latent_dim, c.latent_ffn, c.latent_heads, c.resampler_layers, c.audio_query_count, c.norm_eps
        )

    def forward(self, waveform, state=None, final=False):
        """waveform B,N; returns latents, exact timestamps, a new serializable stream state.

        Nonfinal calls retain one unmatched encoder frame. final flushes it once.
        No model caches are mutated in place. Chunk boundaries do not alter output.
        """
        c = self.config
        if waveform.ndim != 2 or waveform.shape[1] < 1:
            raise ValueError("Expected nonempty mono B,N waveform")
        state = {} if state is None else state.copy()
        if int(state.get("closed", torch.tensor(0))):
            raise ValueError("Audio episode already finalized; reset its state")
        x = waveform.unsqueeze(1)
        for index, (conv, stride) in enumerate(zip(self.stem, (c.audio_stem_stride, 2, 2))):
            if x.shape[-1] == 0:
                break
            x = causal_conv_stream(x, conv, stride, state, f"stem{index}")
        offset = int(state.get("frames", torch.tensor(0)))
        if x.shape[-1] == 0:
            encoded = waveform.new_empty(waveform.shape[0], 0, c.latent_dim)
        else:
            x = x.transpose(1, 2)
            positions = torch.arange(offset, offset + x.shape[1], device=x.device)
            for index, layer in enumerate(self.layers):
                key = f"layer{index}"
                cache = (state[key + ".k"], state[key + ".v"]) if key + ".k" in state else None
                x, kv = layer(x, positions=positions, causal=True, window=c.audio_window, cache=cache)
                state[key + ".k"], state[key + ".v"] = kv
            encoded = self.norm(x)
        new_count = encoded.shape[1]
        old_pending = state.get("pending")
        if old_pending is not None:
            encoded = torch.cat((old_pending, encoded), dim=1)
        pending_start = offset - (0 if old_pending is None else old_pending.shape[1])
        used = encoded.shape[1] if final else (encoded.shape[1] // 2) * 2
        outputs, times = [], []
        for start in range(0, used, 2):
            group = encoded[:, start : min(start + 2, used)]
            query = ((pending_start + start) // 2) % c.audio_query_count
            outputs.append(self.resampler(group, 1, query))
            times.append(
                (pending_start + start + group.shape[1] - 1) * c.audio_stem_stride * 4 / c.sample_rate
            )
        state["pending"] = encoded[:, used:].clone()
        state["frames"] = torch.tensor(offset + new_count)
        state["closed"] = torch.tensor(int(final))
        latents = torch.cat(outputs, 1) if outputs else waveform.new_empty(waveform.shape[0], 0, c.latent_dim)
        return latents, torch.tensor(times, device=waveform.device, dtype=torch.float32), state
