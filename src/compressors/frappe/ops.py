"""Latent-shape helpers for FRAPPE: per-scale grouping, resolution adaption,
and nested multi-scale grid packing/unpacking."""

import torch
import torch.nn.functional as F
from einops import rearrange


def get_scale_groups(ps_list, n_ch):
    groups = []
    i = 0
    while i < n_ch:
        ps = ps_list[i]
        start = i
        while i < n_ch and ps_list[i] == ps:
            i += 1
        groups.append((ps, start, i))
    return groups


def adapt_to_decoder(z, encoder_ps, decoder_ps):
    """Adapt encoder latent resolution to decoder resolution."""
    if encoder_ps == decoder_ps:
        return z
    elif encoder_ps < decoder_ps:
        f = decoder_ps // encoder_ps
        return rearrange(z, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=f, p2=f)
    else:
        f = encoder_ps // decoder_ps
        return F.interpolate(z, scale_factor=f, mode='nearest')


def decoder_channels_per_encoder(encoder_ps, decoder_ps):
    """How many decoder input channels one encoder channel contributes."""
    if encoder_ps <= decoder_ps:
        f = decoder_ps // encoder_ps
        return f * f
    else:
        return 1


def nest_latent_grid(latents):
    """Pack multi-scale latents into a nested 2x2 grid.

    The coarsest scale (4 channels) forms a 2x2 grid. Each subsequent
    scale (3 channels) places the accumulated grid in the top-left quadrant
    and fills top-right, bottom-left, bottom-right with the 3 new channels.

    This produces a single 2D image with a recursive wavelet-like layout
    where coarser information is in the top-left and finer detail radiates
    toward the bottom-right.

    Args:
        latents: list of (C, H, W) tensors, one per scale group.
                 latents[0]: (4, H0, W0) — coarsest, 4 channels
                 latents[i>0]: (3, Hi, Wi) — finer scales, 3 channels each
                 Hi = 2*H(i-1), Wi = 2*W(i-1)

    Returns:
        (H_final, W_final) 2D tensor
    """
    z = latents[0]
    assert z.shape[0] == 4, f"Coarsest scale must have 4 channels, got {z.shape[0]}"
    grid = torch.cat([
        torch.cat([z[0], z[1]], dim=1),
        torch.cat([z[2], z[3]], dim=1),
    ], dim=0)

    for z_s in latents[1:]:
        assert z_s.shape[0] == 3, f"Subsequent scales must have 3 channels, got {z_s.shape[0]}"
        grid = torch.cat([
            torch.cat([grid, z_s[0]], dim=1),
            torch.cat([z_s[1], z_s[2]], dim=1),
        ], dim=0)

    return grid


def unnest_latent_grid(grid, n_scales):
    """Unpack a nested 2x2 grid into per-scale latent tensors.

    Args:
        grid: (H, W) 2D tensor — nested latent grid
        n_scales: number of scale groups (coarsest + number of finer scales)

    Returns:
        list of (C, H, W) tensors: [(4, H0, W0), (3, H1, W1), ...]
    """
    latents = []
    for _ in range(n_scales - 1, 0, -1):
        H, W = grid.shape
        hH, hW = H // 2, W // 2
        latents.insert(0, torch.stack([grid[:hH, hW:], grid[hH:, :hW], grid[hH:, hW:]]))
        grid = grid[:hH, :hW]

    H, W = grid.shape
    hH, hW = H // 2, W // 2
    latents.insert(0, torch.stack([grid[:hH, :hW], grid[:hH, hW:], grid[hH:, :hW], grid[hH:, hW:]]))
    return latents
