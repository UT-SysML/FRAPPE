"""FRAPPE progressive multi-scale autoencoder with unified decoder.

Finer-scale latents are rearranged (einops patchify) to the coarsest resolution,
then concatenated and fed through a single decoder. The decoder architecture is
fixed across scale transitions — only the input channel count grows.

`ConvND` / `LayerNormND` / `Residual` / `LayerScale` / `SAPE` / `AttentionND`
/ `ConvBlockND` / `SAPETransformerBlockND` are vendored from gigatorch.ops —
refresh by re-copying from https://github.com/danjacobellis/gigatorch
(src/gigatorch/ops.py).
"""

import torch
from .ops import adapt_to_decoder, decoder_channels_per_encoder, get_scale_groups
from .quantize import make_quantizer


def ConvND(dim, *args, **kwargs):
    conv_cls = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d][dim-1]
    return conv_cls(*args, **kwargs)


class LayerNormND(torch.nn.Module):
    """Per-position channel normalization for channels-first tensors of any spatial dimensionality.
    Permutes to channels-last, applies F.layer_norm, permutes back. Matches timm's LayerNorm2d."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_channels = num_channels

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.movedim(1, -1), (self.num_channels,), self.weight, self.bias, self.eps
        ).movedim(-1, 1)


class _ChannelsLast(torch.nn.Module):
    def forward(self, x):
        return x.movedim(1, -1)


class _ChannelsFirst(torch.nn.Module):
    def forward(self, x):
        return x.movedim(-1, 1)


class Residual(torch.nn.Module):
    """During training, the residual branch is dropped with probability `drop_prob` and scaled by 1/(1-drop_prob) when kept, preserving the expected value of the output (same scaling convention as timm's DropPath). An optional `scale` module (e.g. `LayerScale`) is applied to the branch output before the skip connection, recovering the CaiT/ConvNeXt formulation `x + γ * branch(x)`."""
    def __init__(self, main, drop_prob=0.0, scale=None):
        super().__init__()
        self.main = main
        self.scale = scale
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob and self.training:
            if torch.rand(()).item() < self.drop_prob:
                return x
            branch = self.main(x)
            if self.scale is not None:
                branch = self.scale(branch)
            return x + branch / (1.0 - self.drop_prob)
        branch = self.main(x)
        if self.scale is not None:
            branch = self.scale(branch)
        return x + branch


class LayerScale(torch.nn.Module):
    """Per-channel learnable scale γ, initialized to a small value (default 1e-6). Applied to the output of a residual branch as in CaiT (arXiv:2103.17239) and ConvNeXt (arXiv:2201.03545)."""
    def __init__(self, dim, num_channels, init=1e-6):
        super().__init__()
        self.shape = [1, -1] + [1] * dim
        self.γ = torch.nn.Parameter(torch.full((num_channels,), float(init)))

    def forward(self, x):
        return x * self.γ.view(self.shape)


class SAPE(torch.nn.Module):
    """Stacked Absolute Position Embedding. Concatenates `dim` new channels encoding centered integer coordinates (in patch units) for each spatial dimension. No learnable parameters. See NaViT (arXiv:2307.06304) for an ablation on a similar position encoding strategy."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        spatial = x.shape[2:]
        coords = []
        for i, size in enumerate(spatial):
            pos = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2
            shape = [1, 1] + [1] * self.dim
            shape[2 + i] = size
            coords.append(pos.view(shape).expand(1, 1, *spatial))
        pos = torch.cat(coords, dim=1).expand(x.shape[0], -1, *spatial)
        return torch.cat([x, pos], dim=1)


class AttentionND(torch.nn.Module):
    """N-dimensional multi-head self-attention. Flattens spatial dims into the sequence and uses torch.nn.functional.scaled_dot_product_attention. QKV and output projections are 1×1 convolutions. The internal attention dimension is num_heads × head_dim, which may differ from num_channels."""
    def __init__(self, dim, num_channels, num_heads, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_dim = num_heads * head_dim
        self.qkv = ConvND(dim, num_channels, 3 * attn_dim, 1, bias=False)
        self.proj = ConvND(dim, attn_dim, num_channels, 1)

    def forward(self, x):
        B, C, *spatial = x.shape
        qkv = self.qkv(x).flatten(2)                               # (B, 3*attn_dim, N)
        qkv = qkv.unflatten(1, (3, self.num_heads, self.head_dim))  # (B, 3, H, Dh, N)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)             # 3× (B, H, N, Dh)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # (B, H, N, Dh)
        return self.proj(out.transpose(2, 3).reshape(B, self.num_heads * self.head_dim, *spatial))


def ConvBlockND(dim, num_channels, kernel_size=5, mlp_ratio=4, drop_prob=0.0,
                norm_layer=None, act_layer=None, conv_mlp=True, layerscale=False, layerscale_init=1e-6):
    """ConvNeXt-style block: depthwise conv → norm → pointwise expand → act → pointwise shrink. Wrapped in Residual
    with drop path. When conv_mlp=False, the pointwise layers use nn.Linear (channels-last) instead of 1×1 conv.
    When layerscale=True, a per-channel LayerScale (γ init `layerscale_init`) is appended to the residual branch."""
    mid = num_channels * mlp_ratio
    norm = LayerNormND(num_channels) if norm_layer is None else norm_layer()
    act = torch.nn.GELU(approximate='tanh') if act_layer is None else act_layer()
    layers = [ConvND(dim, num_channels, num_channels, kernel_size, padding=kernel_size//2, groups=num_channels), norm]
    if conv_mlp:
        layers += [ConvND(dim, num_channels, mid, 1), act, ConvND(dim, mid, num_channels, 1)]
    else:
        layers += [_ChannelsLast(), torch.nn.Linear(num_channels, mid), act,
                   torch.nn.Linear(mid, num_channels), _ChannelsFirst()]
    scale = LayerScale(dim, num_channels, init=layerscale_init) if layerscale else None
    return Residual(torch.nn.Sequential(*layers), drop_prob, scale=scale)


class SAPETransformerBlockND(torch.nn.Module):
    """Pre-norm transformer block with per-layer SAPE position injection. Takes (B, num_channels, *spatial),
    internally operates at num_channels + dim channels (with dim position channels via SAPE), outputs
    (B, num_channels, *spatial). Interleaves cleanly with ConvBlockND at the same num_channels."""
    def __init__(self, dim, num_channels, num_heads, head_dim=32, mlp_ratio=4, drop_prob=0.0,
                 norm_layer=None, act_layer=None):
        super().__init__()
        inner = num_channels + dim
        self.num_channels = num_channels
        self.sape = SAPE(dim)
        self.norm1 = LayerNormND(inner) if norm_layer is None else norm_layer()
        self.norm2 = LayerNormND(inner) if norm_layer is None else norm_layer()
        self.attn = AttentionND(dim, inner, num_heads, head_dim)
        act = torch.nn.GELU(approximate='tanh') if act_layer is None else act_layer()
        mid = inner * mlp_ratio
        self.mlp = torch.nn.Sequential(
            ConvND(dim, inner, mid, 1),
            ConvND(dim, mid, mid, 3, padding=1, groups=mid),
            act,
            ConvND(dim, mid, inner, 1),
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        x_pos = self.sape(x)                              # (B, C+dim, *spatial)
        if self.drop_prob and self.training:
            if torch.rand(()).item() >= self.drop_prob:
                x_pos = x_pos + self.attn(self.norm1(x_pos)) / (1.0 - self.drop_prob)
            if torch.rand(()).item() >= self.drop_prob:
                x_pos = x_pos + self.mlp(self.norm2(x_pos)) / (1.0 - self.drop_prob)
        else:
            x_pos = x_pos + self.attn(self.norm1(x_pos))
            x_pos = x_pos + self.mlp(self.norm2(x_pos))
        return x_pos[:, :self.num_channels]                # strip position


def make_decoder_blocks(arch, dim, kernel_size=5, mlp_ratio=4, layerscale=False, layerscale_init=1e-6):
    mlp_ratio = int(mlp_ratio)
    blocks = []
    for c in arch.upper():
        if c == 'C':
            blocks.append(ConvBlockND(2, dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                                      layerscale=layerscale, layerscale_init=layerscale_init))
        elif c == 'T':
            blocks.append(SAPETransformerBlockND(2, dim, num_heads=8, mlp_ratio=mlp_ratio))
        else:
            raise ValueError(f"Unknown block type '{c}' in decoder_arch")
    return blocks


class AutoencoderSingleChannel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_ps = config.ps
        self.decoder_ps = config.decoder_ps
        dec_ch = decoder_channels_per_encoder(self.encoder_ps, self.decoder_ps)
        dps = self.decoder_ps

        self.analysis_transform = torch.nn.Sequential(
            torch.nn.Conv2d(config.input_channels, 1, kernel_size=config.ps, stride=config.ps),
            make_quantizer(config.encoder_arch, 1)
        )
        self.synthesis_transform = torch.nn.Sequential(
            torch.nn.Conv2d(dec_ch, config.decoder_dim,
                            kernel_size=config.decoder_kernel_size, stride=1,
                            padding='same', padding_mode='reflect'),
            *make_decoder_blocks(config.decoder_arch, config.decoder_dim,
                                 kernel_size=config.decoder_kernel_size,
                                 mlp_ratio=config.decoder_mlp_ratio,
                                 layerscale=config.decoder_layerscale,
                                 layerscale_init=config.decoder_layerscale_init),
            torch.nn.Conv2d(config.decoder_dim,
                            config.input_channels * dps * dps,
                            kernel_size=1, stride=1),
            torch.nn.ConvTranspose2d(config.input_channels * dps * dps,
                                     config.input_channels,
                                     kernel_size=dps, stride=dps),
            torch.nn.Hardtanh(),
        )

    def forward(self, x):
        z = self.analysis_transform(x)
        self._last_z = z
        z_a = adapt_to_decoder(z, self.encoder_ps, self.decoder_ps)
        return self.synthesis_transform(z_a.to(torch.float))


class MergedAutoencoder(torch.nn.Module):
    def __init__(self, config, n_ch):
        super().__init__()
        self.scale_groups = get_scale_groups(config.ps, n_ch)
        self.decoder_ps = config.decoder_ps

        self.encoders = torch.nn.ModuleList()
        self.adapt_factors = []
        total_decoder_ch = 0

        for ps_s, start, end in self.scale_groups:
            n_ch_group = end - start
            self.encoders.append(torch.nn.Sequential(
                torch.nn.Conv2d(config.input_channels, n_ch_group,
                                kernel_size=ps_s, stride=ps_s),
                make_quantizer(config.encoder_arch, n_ch_group),
            ))
            self.adapt_factors.append((ps_s, self.decoder_ps))
            total_decoder_ch += n_ch_group * decoder_channels_per_encoder(ps_s, self.decoder_ps)

        self.n_ch = total_decoder_ch

        dim = config.decoder_dim
        ks = config.decoder_kernel_size
        arch = config.decoder_arch
        mlp_ratio = config.decoder_mlp_ratio
        C = config.input_channels
        dps = self.decoder_ps

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(total_decoder_ch, dim, kernel_size=ks, stride=1,
                            padding='same', padding_mode='reflect'),
            *make_decoder_blocks(arch, dim, kernel_size=ks, mlp_ratio=mlp_ratio,
                                 layerscale=config.decoder_layerscale,
                                 layerscale_init=config.decoder_layerscale_init),
            torch.nn.Conv2d(dim, C * dps * dps, kernel_size=1, stride=1),
            torch.nn.ConvTranspose2d(C * dps * dps, C, kernel_size=dps, stride=dps),
            torch.nn.Hardtanh(),
        )

    def encode(self, x):
        return [enc(x) for enc in self.encoders]

    def decode(self, latents):
        adapted = []
        for z, (enc_ps, dec_ps) in zip(latents, self.adapt_factors):
            adapted.append(adapt_to_decoder(z.to(torch.float), enc_ps, dec_ps))
        return self.decoder(torch.cat(adapted, dim=1))

    def forward(self, x):
        return self.decode(self.encode(x))


def load_progressive_model(weights, config, n_ch, device):
    """Build a MergedAutoencoder using the first n_ch channels and load weights
    from a flat tensor mapping with the safetensors namespace
    ``merged.{n_ch}.encoders.{s}.{...}`` and ``merged.{n_ch}.decoder.{...}``
    (the layout shipped on the Hugging Face model hub)."""
    model = MergedAutoencoder(config, n_ch).to(device)
    e_pref = f"merged.{n_ch}.encoders."
    d_pref = f"merged.{n_ch}.decoder."
    encoder_sds = {}
    decoder_sd = {}
    for k, v in weights.items():
        if k.startswith(e_pref):
            s_str, _, inner = k[len(e_pref):].partition(".")
            encoder_sds.setdefault(int(s_str), {})[inner] = v
        elif k.startswith(d_pref):
            decoder_sd[k[len(d_pref):]] = v
    for s in sorted(encoder_sds):
        model.encoders[s].load_state_dict(encoder_sds[s])
    model.decoder.load_state_dict(decoder_sd)
    return model.eval()


def load_from_hub(repo_id="danjacobellis/FRAPPE", subdir="FRAPPE"):
    """Download ``config.json`` and ``FRAPPE_pytorch_model.safetensors`` from
    a Hugging Face model repo and return ``(config, weights, n_trained)``,
    ready to feed into :func:`load_progressive_model`. ``subdir`` is the
    folder inside the repo holding the two files."""
    import json
    from types import SimpleNamespace
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/config.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/FRAPPE_pytorch_model.safetensors")
    with open(config_path) as f:
        cfg_dict = json.load(f)
    n_trained = cfg_dict.pop("n_trained")
    config = SimpleNamespace(**cfg_dict)
    weights = load_file(weights_path)
    return config, weights, n_trained
