import torch
from torch import nn
from .image import ShinraImageFrontend
from .audio import ShinraAudioFrontend
from .temporal import ShinraTemporalModule
from .decoders import VisualFlowDecoder, AudioFlowDecoder, flow_matching_loss, sample_flow
from ..normalization import RMSNorm


class ModalityInterface(nn.Module):
    def __init__(self, latent, hidden, eps):
        super().__init__()
        self.input_norm, self.output_norm = RMSNorm(latent, eps), RMSNorm(latent, eps)
        self.input = nn.Linear(latent, hidden, bias=False)
        self.output = nn.Linear(hidden, latent, bias=False)

    def encode(self, x):
        return self.input(self.input_norm(x))

    def decode(self, x):
        return self.output_norm(self.output(x))


class ShinraMultimodal(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.image = ShinraImageFrontend(c)
        self.audio = ShinraAudioFrontend(c)
        self.temporal = ShinraTemporalModule(c)
        self.interfaces = nn.ModuleDict(
            {
                name: ModalityInterface(c.latent_dim, c.hidden_size, c.norm_eps)
                for name in ("visual", "audio", "state")
            }
        )
        self.types = nn.Embedding(c.modality_types, c.hidden_size)
        self.metadata = nn.Linear(c.metadata_features, c.hidden_size, bias=False)
        self.visual_decoder = VisualFlowDecoder(c)
        self.audio_decoder = AudioFlowDecoder(c)

    def embed(self, latents, modality, type_ids, metadata):
        if metadata.shape[:-1] != latents.shape[:-1] or metadata.shape[-1] != self.config.metadata_features:
            raise ValueError("Metadata must align with each latent")
        return (
            self.interfaces[modality].encode(latents)
            + self.types(type_ids)
            + self.metadata(metadata.to(latents.dtype))
        )

    def video(self, frames, times, state=None, final=False, spatial_tokens=64, fast_tokens=8):
        """Frame groups of four. B,F,3,H,W -> timestamped latents and immutable side state.

        Frames within a group become available together at its final timestamp. This
        avoids lookahead leakage when temporal compression sees all four frames.
        """
        if frames.ndim != 5 or times.ndim != 1 or len(times) != frames.shape[1]:
            raise ValueError("Expected video B,F,3,H,W and one timestamp per frame")
        if len(times) and (times[1:] < times[:-1]).any():
            raise ValueError("Video timestamps must be monotonic")
        state = {} if state is None else state.copy()
        if int(state.get("closed", torch.tensor(0))):
            raise ValueError("Video stream finalized; reset episode")
        if "last_time" in state and len(times) and times[0] <= state["last_time"].to(times.device):
            raise ValueError("Video timestamps must advance across chunks")
        encoded = [self.image(frame, spatial_tokens) for frame in frames.unbind(1)]
        latent = (
            torch.stack(encoded, 1)
            if encoded
            else frames.new_empty(frames.shape[0], 0, spatial_tokens, self.config.latent_dim)
        )
        if "pending" in state:
            latent = torch.cat((state["pending"], latent), 1)
            times = torch.cat((state["pending_times"].to(times.device), times))
        available = latent.shape[1] if final else latent.shape[1] // 4 * 4
        outputs, output_times = [], []
        for start in range(0, available, 4):
            group = latent[:, start : min(start + 4, available)]
            stamp = times[start : start + group.shape[1]]
            timed = self.temporal(group.flatten(1, 2), stamp.repeat_interleave(spatial_tokens))
            slow = self.image.resampler(timed, spatial_tokens)
            fast = [self.image.resampler(frame, fast_tokens) for frame in group.unbind(1)]
            emitted = torch.cat((*fast, slow), 1)
            outputs.append(emitted)
            output_times.append(stamp[-1].expand(emitted.shape[1]))
        state["pending"], state["pending_times"] = latent[:, available:].clone(), times[available:].clone()
        if len(times):
            state["last_time"] = times[-1]
        state["closed"] = torch.tensor(int(final))
        result = (
            torch.cat(outputs, 1) if outputs else frames.new_empty(frames.shape[0], 0, self.config.latent_dim)
        )
        stamps = torch.cat(output_times) if output_times else times.new_empty(0)
        return result, stamps, state

    def _video_conditions(self, conditions, times):
        if conditions.ndim != 4 or len(times) != conditions.shape[1]:
            raise ValueError("Expected B,F,N,D video conditions and F timestamps")
        b, f, n, d = conditions.shape
        return self.temporal(conditions.flatten(1, 2), times.repeat_interleave(n)).reshape(b, f, n, d)

    def video_flow_loss(self, frames, conditions, times, generator=None):
        if frames.ndim != 5 or frames.shape[:2] != conditions.shape[:2] or frames.shape[1] == 0:
            raise ValueError("Video frames must align with per-frame latent conditions")
        conditions = self._video_conditions(conditions, times)
        losses = [
            flow_matching_loss(self.visual_decoder, frame, condition, generator)
            for frame, condition in zip(frames.unbind(1), conditions.unbind(1))
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample_video(self, noise, conditions, times, steps=32):
        if noise.ndim != 5 or noise.shape[:2] != conditions.shape[:2] or noise.shape[1] == 0:
            raise ValueError("Video noise must align with per-frame latent conditions")
        conditions = self._video_conditions(conditions, times)
        return torch.stack(
            [
                sample_flow(self.visual_decoder, frame, condition, steps)
                for frame, condition in zip(noise.unbind(1), conditions.unbind(1))
            ],
            1,
        )
