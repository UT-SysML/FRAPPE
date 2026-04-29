"""Filter visualization for RAE encoder models."""

import torch
import numpy as np
import einops
import PIL.Image
from sympy import divisors
from torchvision.transforms.v2.functional import to_pil_image


def stack_grid(n):
    h = min(d for d in divisors(n) if d >= np.ceil(n**0.5))
    return h, n // h


def make_filter_grid(filters, biases, n_channels, layout=None):
    """Build an RGBA grid of filter visualizations with transparent padding."""
    gh, gw = layout if layout else stack_grid(n_channels)
    weights_grid = einops.rearrange(filters, '(H W) c h w -> H W c h w', H=gh, W=gw)
    biases_grid = einops.rearrange(biases, '(H W) -> H W 1 1 1', H=gh, W=gw)
    filters_grid = weights_grid + biases_grid / torch.prod(torch.tensor(filters.shape[-3:]))
    filters_norm = filters_grid / (8 * filters_grid.std()) + 0.5
    pad = 1
    c, h, w = filters.shape[1], filters.shape[2], filters.shape[3]
    ph, pw = h + pad, w + pad
    canvas = torch.zeros(4, gh * ph + pad, gw * pw + pad)
    for r in range(gh):
        for col in range(gw):
            patch = filters_norm[r, col].clamp(0, 1)
            y0, x0 = pad + r * ph, pad + col * pw
            canvas[:3, y0:y0+h, x0:x0+w] = patch
            canvas[3, y0:y0+h, x0:x0+w] = 1.0
    return canvas


def extract_filter_grids(merged, n_channels):
    """Extract filter grids from a merged encoder, auto-detecting architecture.

    Returns a list of (label, RGBA_grid) tuples.
    """
    if isinstance(merged.encoder, torch.nn.ModuleList):
        # PPE: one grid per prefilter index
        all_prefilters = torch.stack([enc.prefilter.weight.data for enc in merged.encoder])
        all_biases = torch.stack([enc.prefilter.bias.data for enc in merged.encoder])
        H = all_prefilters.shape[1]
        grids = []
        for h in range(H):
            grid = make_filter_grid(all_prefilters[:, h], all_biases[:, h], n_channels)
            grids.append((f"Prefilter {h}", grid))
        return grids
    else:
        # LF8/ASN8: single Conv2d
        filters = merged.encoder[0].weight.data
        biases = merged.encoder[0].bias.data
        grid = make_filter_grid(filters, biases, n_channels)
        return [("Filters", grid)]


def display_filter_grids(merged, n_channels, scale=10):
    """Extract and display filter grids from a merged encoder."""
    grids = extract_filter_grids(merged, n_channels)
    images = []
    for label, grid in grids:
        w_px, h_px = grid.shape[2] * scale, grid.shape[1] * scale
        img = to_pil_image(grid).resize((w_px, h_px), resample=PIL.Image.Resampling.NEAREST)
        print(label)
        display(img)
        images.append((label, img))
    return images
