"""Quantizer factory and color-space helpers for FRAPPE.

`Softsign` / `UniformTrainingNoise` / `ChannelAffine` / `SoftsignCompanding`
are vendored from gigatorch.ops — refresh by re-copying from
https://github.com/danjacobellis/gigatorch (src/gigatorch/ops.py).
"""

import torch
from livecodec.codec import QuantizeLF8


def srgb_to_linear(x):
    x_01 = x / 2 + 0.5
    return torch.where(x_01 <= 0.04045, x_01 / 12.92, ((x_01 + 0.055) / 1.055) ** 2.4) * 2.0 - 1.0


class Softsign(torch.nn.Module):
    """Learnable per-channel companding: r * x / (σ + |x|). The output is guaranteed to lie in [-(2^(bits-1)-1), 2^(bits-1)-1], i.e. the representable range of a signed `bits`-bit integer (default [-127, 127] for 8-bit). Requires 3 MACs per element: one abs, one add, one div (the multiply by r is a fixed scalar and can be fused)."""
    def __init__(self, dim, num_channels, bits=8):
        super().__init__()
        self.shape = [1, -1] + [1]*dim
        self.r = 2**(bits - 1) - 1
        self._σ = torch.nn.Parameter(torch.full((num_channels,), float(self.r - 1)))

    def forward(self, x):
        σ = (self._σ.abs() + 1e-6).view(self.shape)
        return self.r * x / (σ + x.abs())


class UniformTrainingNoise(torch.nn.Module):
    """Adds uniform noise in [-delta, delta] during training only, where delta = k + 0.5. With k=0 (delta=0.5). With k >= 1, it simulates the wider reconstruction tolerance of near-lossless codecs (e.g. JPEG-LS allows reconstructed values to differ by +/-k from the original)."""
    def __init__(self, k=0):
        super().__init__()
        self.delta = k + 0.5
    def forward(self, x):
        if self.training:
            x = x + torch.rand_like(x)*(2*self.delta) - self.delta
        return x


class ChannelAffine(torch.nn.Module):
    """Per-channel affine transform: x * γ (+ β if bias=True). Learnable scale γ is initialized to 1 and optional bias β to 0."""
    def __init__(self, dim, num_channels, bias=False):
        super().__init__()
        self.shape = [1, -1] + [1] * dim
        self.γ = torch.nn.Parameter(torch.ones(num_channels))
        self.β = torch.nn.Parameter(torch.zeros(num_channels)) if bias else None

    def forward(self, x):
        x = x * self.γ.view(self.shape)
        if self.β is not None:
            x = x + self.β.view(self.shape)
        return x


def SoftsignCompanding(dim, num_channels, qat=False, bits=8, k=0, affine=True, bias=False):
    """(Quantization-aware) learnable softsign companding. Somewhat similar to to DyTanh (arXiv:2503.10622) but uses softsign for lower MAC cost and optionally uses the limited range companded output as an opportunity to quantize activations."""
    return torch.nn.Sequential(
        Softsign(dim, num_channels, bits),
        UniformTrainingNoise(k) if qat else torch.nn.Identity(),
        ChannelAffine(dim, num_channels, bias=bias) if affine else torch.nn.Identity()
    )


def make_quantizer(quantize_type, n_channels):
    if quantize_type == 'LF8':
        return QuantizeLF8(n_channels)
    elif quantize_type == 'SC8':
        return SoftsignCompanding(dim=2, num_channels=n_channels, qat=True)
    else:
        raise ValueError(f"Unknown quantize type: {quantize_type}")
