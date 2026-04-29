"""Default latent arrangement and entropy coding for FRAPPE.

Four module-level functions form the contract used by
:mod:`compressors.frappe.evaluate_rate_distortion`::

    arrange_latents(latents) -> ArrangedLatents
    unarrange_latents(arranged, scale_groups) -> list[Tensor]
    encode_latents(arranged) -> bytes
    decode_latents(blob) -> ArrangedLatents

Any user-supplied Python file that exposes these four names with matching
signatures can be substituted via the harness's ``--latent-module`` /
``--entropy-module`` flags. ``ArrangedLatents`` is whatever opaque type the
arrangement produces; the entropy-coding side does not have to be agnostic
across arrangements.

The default implementation reproduces FRAPPE.ipynb's per-scale JPEG-LS
behavior: each scale's int8 latent ``(1, C, H, W)`` is reshaped to
``(C * H, W)`` and saved as a single grayscale JPEG-LS image. The
resulting blob concatenates per-scale JPEG-LS streams, each preceded by a
4-byte big-endian length, so :func:`decode_latents` is fully
self-describing — the caller does not need to supply per-scale shape
metadata to read it back.
"""

from __future__ import annotations

import io
import struct

import PIL.Image
import pillow_jpls  # noqa: F401 -- registers the JPEG-LS plugin with PIL
import torch
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image


# Type alias. The default arrangement is a list of 2D int8 tensors, one per
# scale, each shaped ``(C * H, W)``.
ArrangedLatents = list


def arrange_latents(latents):
    """Per-scale 2D reshape.

    Each input latent has shape ``(1, C, H, W)`` (int8, on any device).
    It is reshaped to ``(C * H, W)`` and moved to CPU — the same layout
    used in FRAPPE.ipynb's ``compute_bpp`` so JPEG-LS sees a single
    grayscale image per scale.
    """
    arranged = []
    for z in latents:
        if z.dim() != 4 or z.shape[0] != 1:
            raise ValueError(f"Expected (1, C, H, W) latent, got shape {tuple(z.shape)}")
        C, H, W = z.shape[1], z.shape[2], z.shape[3]
        arranged.append(z[0].reshape(C * H, W).contiguous().cpu())
    return arranged


def unarrange_latents(arranged, scale_groups):
    """Inverse of :func:`arrange_latents`.

    ``scale_groups`` is the ``(ps_s, start, end)`` list from
    :class:`compressors.frappe.model.MergedAutoencoder`. We pull
    ``C = end - start`` per scale and derive ``H`` from the arranged
    tensor's row count.
    """
    if len(arranged) != len(scale_groups):
        raise ValueError(
            f"Got {len(arranged)} arranged scales for {len(scale_groups)} scale groups"
        )
    latents = []
    for z_2d, (_, start, end) in zip(arranged, scale_groups):
        C = end - start
        rows, W = z_2d.shape
        if rows % C != 0:
            raise ValueError(f"Cannot split {rows} rows evenly into {C} channels")
        H = rows // C
        latents.append(z_2d.reshape(1, C, H, W))
    return latents


def encode_latents(arranged):
    """Encode each per-scale int8 2D tensor as a length-prefixed JPEG-LS stream.

    Blob layout::

        [4-byte BE length][JPEG-LS bytes]   (one chunk per scale)

    The reader walks until the buffer is exhausted, so the number of
    scales is implicit.
    """
    chunks = []
    for z_2d in arranged:
        # JPEG-LS handles 8-bit unsigned grayscale; shift int8 [-127, 127]
        # to uint8 [0, 254] before saving and back on decode.
        u8 = (z_2d.cpu().to(torch.long) + 127).to(torch.uint8)
        buf = io.BytesIO()
        to_pil_image(u8).save(buf, format='JPEG-LS')
        data = buf.getvalue()
        chunks.append(struct.pack('>I', len(data)))
        chunks.append(data)
    return b''.join(chunks)


def decode_latents(blob):
    """Inverse of :func:`encode_latents`.

    Reads length-prefixed JPEG-LS streams from ``blob`` and returns the
    per-scale 2D int8 tensors. Each JPEG-LS stream carries its own
    ``(rows, cols)`` header, and the chunk boundaries are recovered from
    the length prefixes.
    """
    arranged = []
    offset = 0
    n = len(blob)
    while offset < n:
        if offset + 4 > n:
            raise ValueError("Truncated blob: missing 4-byte length prefix")
        (length,) = struct.unpack_from('>I', blob, offset)
        offset += 4
        if offset + length > n:
            raise ValueError(
                f"Truncated blob: declared chunk of {length} bytes overruns buffer"
            )
        img = PIL.Image.open(io.BytesIO(blob[offset:offset + length]))
        img.load()
        offset += length
        u8 = pil_to_tensor(img)[0]  # (H, W) uint8
        arranged.append((u8.to(torch.long) - 127).to(torch.int8))
    return arranged
