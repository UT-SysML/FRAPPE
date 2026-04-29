"""Evaluation helpers for FRAPPE checkpoints."""

import io
import numpy as np
import torch
import PIL.Image
import pillow_jpls  # noqa: F401 — registers JPEG-LS codec with PIL
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

from .quantize import srgb_to_linear


def validate(merged, device, val_dataset, config):
    merged.eval()
    max_ps = max(config.ps)
    psnrs, bpps = [], []
    for sample in val_dataset:
        img = sample['image']
        w, h = img.size
        h_rs = max_ps * (h // max_ps)
        w_rs = max_ps * (w // max_ps)
        img = img.resize((w_rs, h_rs), PIL.Image.Resampling.BICUBIC)
        x = pil_to_tensor(img.convert("RGB")).to(torch.float).to(device).unsqueeze(0) / 127.5 - 1.0
        x_in = srgb_to_linear(x) if config.linear_input else x
        with torch.inference_mode():
            latents = merged.encode(x_in)
            latents = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
            xhat = merged.decode(latents).clamp(-1, 1)
        x_01 = x / 2 + 0.5
        xhat_01 = xhat / 2 + 0.5
        psnr = -10 * torch.nn.functional.mse_loss(x_01, xhat_01).log10().item()
        psnrs.append(psnr)
        n_pixels = x.shape[2] * x.shape[3]
        total_bytes = 0
        for z in latents:
            z_2d = z[0].reshape(z.shape[1] * z.shape[2], z.shape[3])
            buff = io.BytesIO()
            to_pil_image((z_2d.long() + 127).to(torch.uint8)).save(buff, format='JPEG-LS')
            total_bytes += len(buff.getbuffer())
        bpp = total_bytes * 8 / n_pixels
        bpps.append(bpp)
    mean_bpp = np.mean(bpps)
    mean_cr = 24.0 / mean_bpp
    return np.mean(psnrs), mean_cr
