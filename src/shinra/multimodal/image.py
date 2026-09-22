import torch
from torch import nn
from .layers import FrontendBlock, Resampler
from ..normalization import RMSNorm


def patchify(images, patch):
    b, ch, h, w = images.shape
    if ch != 3 or h % patch or w % patch:
        raise ValueError("Expected RGB dimensions divisible by patch size")
    return (
        images.reshape(b, 3, h // patch, patch, w // patch, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(b, -1, 3 * patch * patch)
    )


def unpatchify(tokens, height, width, patch):
    return (
        tokens.reshape(tokens.shape[0], height // patch, width // patch, patch, patch, 3)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(-1, 3, height, width)
    )


class ShinraImageFrontend(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.patch = nn.Linear(3 * c.patch_size**2, c.visual_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                FrontendBlock(c.visual_dim, c.visual_ffn, c.visual_heads, c.norm_eps)
                for _ in range(c.visual_layers)
            ]
        )
        self.norm = RMSNorm(c.visual_dim, c.norm_eps)
        self.resample_in = nn.Linear(c.visual_dim, c.latent_dim, bias=False)
        self.resampler = Resampler(
            c.latent_dim, c.latent_ffn, c.latent_heads, c.resampler_layers, c.max_visual_latents, c.norm_eps
        )

    def forward(self, images, latent_count=None):
        c = self.config
        x = self.patch(patchify(images, c.patch_size))
        if x.shape[1] > c.max_visual_patches:
            raise ValueError("Image exceeds patch budget; use explicit coordinate-preserving crops")
        rows, cols = images.shape[-2] // c.patch_size, images.shape[-1] // c.patch_size
        yy, xx = torch.meshgrid(
            torch.arange(rows, device=x.device), torch.arange(cols, device=x.device), indexing="ij"
        )
        xy = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(x.shape[0], -1, -1)
        for index, layer in enumerate(self.layers):
            # Frame-global every fourth block; other blocks use 16x16 spatial windows.
            if index % 4 == 3 or (rows <= 16 and cols <= 16):
                x, _ = layer(x, positions=xy)
            else:
                grid = x.reshape(x.shape[0], rows, cols, -1)
                coords = xy.reshape(x.shape[0], rows, cols, 2)
                bands = []
                for row in range(0, rows, 16):
                    cells = []
                    for col in range(0, cols, 16):
                        tile = grid[:, row : row + 16, col : col + 16]
                        result, _ = layer(
                            tile.flatten(1, 2),
                            positions=coords[:, row : row + 16, col : col + 16].flatten(1, 2),
                        )
                        cells.append(result.reshape_as(tile))
                    bands.append(torch.cat(cells, dim=2))
                x = torch.cat(bands, dim=1).flatten(1, 2)
        count = latent_count if latent_count is not None else max(1, x.shape[1] // 4)
        return self.resampler(self.resample_in(self.norm(x)), count)
