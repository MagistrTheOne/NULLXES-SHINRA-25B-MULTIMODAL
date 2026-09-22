"""Typed multimodal assembly. Latents consume context exactly like text positions."""

from dataclasses import dataclass
import torch
from ..positions import fourier_features


@dataclass(frozen=True)
class TextSegment:
    input_ids: torch.Tensor
    supervise: bool = True


@dataclass(frozen=True)
class LatentSegment:
    latents: torch.Tensor
    modality: str
    type_id: int
    start_token_id: int
    end_token_id: int
    # Coordinates have physical units declared by the caller; timestamps are availability times.
    coordinates: torch.Tensor
    supervise_boundaries: bool = True


@dataclass
class AssembledStream:
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    spans: list[dict]


def assemble_stream(model, segments):
    if not segments:
        raise ValueError("A stream needs at least one segment")
    c = model.config
    embeddings, labels, spans = [], [], []
    offset, batch = 0, None
    for segment in segments:
        if isinstance(segment, TextSegment):
            ids = segment.input_ids
            x = model.embed_tokens(ids)
            targets = ids if segment.supervise else torch.full_like(ids, -100)
            kind = "text"
        elif isinstance(segment, LatentSegment):
            latent = segment.latents
            if latent.shape[1] == 0:
                continue
            if segment.coordinates.shape[:2] != latent.shape[:2]:
                raise ValueError("Every latent requires coordinates")
            for token in (segment.start_token_id, segment.end_token_id):
                if not c.control_token_range[0] <= token <= c.control_token_range[1]:
                    raise ValueError(
                        "Boundary token must be in the reserved control range inside the vocabulary"
                    )
            types = torch.full(latent.shape[:2], segment.type_id, device=latent.device, dtype=torch.long)
            metadata = fourier_features(segment.coordinates, c.metadata_features)
            content = model.multimodal.embed(latent, segment.modality, types, metadata)
            start = torch.full(
                (latent.shape[0], 1), segment.start_token_id, device=latent.device, dtype=torch.long
            )
            end = torch.full_like(start, segment.end_token_id)
            x = torch.cat((model.embed_tokens(start), content, model.embed_tokens(end)), 1)
            targets = torch.full(x.shape[:2], -100, device=x.device, dtype=torch.long)
            if segment.supervise_boundaries:
                targets[:, 0], targets[:, -1] = segment.start_token_id, segment.end_token_id
            kind = segment.modality
        else:
            raise TypeError("Unknown SHINRA stream segment")
        if batch is not None and batch != x.shape[0]:
            raise ValueError("Segments must share a batch size")
        batch = x.shape[0]
        spans.append({"modality": kind, "start": offset, "end": offset + x.shape[1]})
        offset += x.shape[1]
        if offset > c.max_context_length:
            raise ValueError("Combined multimodal stream exceeds native context budget")
        embeddings.append(x)
        labels.append(targets)
    if not embeddings:
        raise ValueError("No available stream elements")
    return AssembledStream(torch.cat(embeddings, 1), torch.cat(labels, 1), spans)
