"""Rate-distortion evaluation harness for FRAPPE on Kodak.

Sweeps every truncated channel count ``n_ch`` in ``[1, n_trained]`` and
writes a single self-describing JSON containing the full
(``n_channel_counts × n_images × n_metrics``) matrix plus per-channel-count
means.

CLI: ``python -m compressors.frappe.evaluate_rate_distortion`` with the
flags listed in :func:`run`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from . import entropy_coding as default_entropy_coding
from .model import load_from_hub, load_progressive_model
from .quantize import srgb_to_linear


METRICS = ("bpp", "PSNR_dB", "SSIM", "LPIPS_dB", "DISTS_dB")
NULL_FALLBACKS = {
    "bpp": 24.0,
    "PSNR_dB": 0.0,
    "SSIM": 0.0,
    "LPIPS_dB": 0.0,
    "DISTS_dB": 0.0,
}
DEFAULT_REPO_ID = "danjacobellis/FRAPPE"
DEFAULT_SUBDIR = "FRAPPE"
WEIGHTS_FILENAME = "FRAPPE_pytorch_model.safetensors"
CONFIG_FILENAME = "config.json"

DTYPE_NAMES = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


# ----------------------------------------------------------------------
# Pluggable module loading
# ----------------------------------------------------------------------

def _file_provenance(path):
    """Return ``{module_path, source_sha256, source}`` for an absolute path."""
    abs_path = str(Path(path).resolve())
    source = Path(abs_path).read_text()
    return {
        "module_path": abs_path,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source": source,
    }


def _load_module_from_path(path):
    """Import a Python file as an anonymous module without polluting sys.modules."""
    abs_path = Path(path).resolve()
    name = f"_compressors_codec_{abs_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_codec_module(path_arg, required_fns):
    """Return ``(module, provenance_dict)`` for a path arg or the bundled default."""
    if path_arg is None:
        path = Path(default_entropy_coding.__file__)
        mod = default_entropy_coding
    else:
        path = Path(path_arg).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Module path does not exist: {path}")
        mod = _load_module_from_path(path)
    for fn_name in required_fns:
        if not callable(getattr(mod, fn_name, None)):
            raise AttributeError(f"Module {path} does not expose callable {fn_name!r}")
    return mod, _file_provenance(path)


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------

def _sanitize(val):
    """Convert NaN / +/- inf to ``None`` (for JSON storage); else return ``float(val)``."""
    if val is None:
        return None
    f = float(val)
    if not math.isfinite(f):
        return None
    return f


def _mean_with_fallback(per_image, metric):
    fallback = NULL_FALLBACKS[metric]
    vals = [m[metric] if m[metric] is not None else fallback for m in per_image]
    return float(np.mean(vals))


def _check_image_shape(h, w, max_ps):
    if h % max_ps != 0 or w % max_ps != 0:
        raise ValueError(
            f"Image {h}x{w} not divisible by max encoder patch size {max_ps}; "
            "this harness fails loudly rather than silently resampling."
        )


# ----------------------------------------------------------------------
# Per-image evaluation
# ----------------------------------------------------------------------

def _evaluate_one(
    *,
    model,
    img_pil,
    config,
    device,
    torch_dtype,
    arrange_fn,
    unarrange_fn,
    encode_fn,
    decode_fn,
    lpips_loss,
    dists_loss,
    ssim_loss,
):
    """Run one (image, model) RD evaluation. Returns a metrics dict."""
    rgb = img_pil.convert("RGB")
    x = pil_to_tensor(rgb).to(torch_dtype).to(device).unsqueeze(0) / 127.5 - 1.0
    n_pixels = x.shape[2] * x.shape[3]
    x_in = srgb_to_linear(x) if getattr(config, "linear_input", False) else x

    with torch.inference_mode():
        latents = model.encode(x_in)
        latents_q = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]

    arranged = arrange_fn(latents_q)
    blob = encode_fn(arranged)
    arranged_back = decode_fn(blob)
    latents_back = unarrange_fn(arranged_back, model.scale_groups)
    latents_back = [z.to(device) for z in latents_back]

    with torch.inference_mode():
        xhat = model.decode(latents_back).clamp(-1, 1)

    x_01 = (x.float() / 2 + 0.5)
    xhat_01 = (xhat.float() / 2 + 0.5)
    mse = torch.nn.functional.mse_loss(x_01, xhat_01).item()
    psnr_db = -10 * np.log10(mse) if mse > 0 else float("inf")
    ssim_val = 1 - ssim_loss(x_01, xhat_01).item()
    lpips_val = lpips_loss(x_01, xhat_01).item()
    lpips_db = -10 * np.log10(lpips_val) if lpips_val > 0 else float("inf")
    dists_val = dists_loss(x_01, xhat_01).item()
    dists_db = -10 * np.log10(dists_val) if dists_val > 0 else float("inf")
    bpp = len(blob) * 8 / n_pixels

    return {
        "bpp": _sanitize(bpp),
        "PSNR_dB": _sanitize(psnr_db),
        "SSIM": _sanitize(ssim_val),
        "LPIPS_dB": _sanitize(lpips_db),
        "DISTS_dB": _sanitize(dists_db),
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def run(
    *,
    output=None,
    device="cuda:0",
    dtype="float",
    repo_id=DEFAULT_REPO_ID,
    subdir=DEFAULT_SUBDIR,
    latent_module=None,
    entropy_module=None,
    verbose=True,
):
    """Run a single RD evaluation of FRAPPE on Kodak. Returns the output ``Path``.

    See module docstring and ``--help`` for a description of each argument.
    Always sweeps every ``n_ch`` in ``[1, n_trained]``; there is no
    ``--n-ch`` flag.
    """
    run_id = int(time.time())

    # Resolve output path
    if output is None:
        output_path = Path.cwd() / f"rate_distortion_{run_id}.json"
    else:
        output_path = Path(output)
        if output_path.is_dir() or str(output).endswith("/"):
            output_path = output_path / f"rate_distortion_{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve dtype
    if isinstance(dtype, str):
        if dtype not in DTYPE_NAMES:
            raise ValueError(
                f"Unknown dtype {dtype!r}; expected one of {sorted(DTYPE_NAMES)}"
            )
        torch_dtype = DTYPE_NAMES[dtype]
    else:
        torch_dtype = dtype
    dtype_label = f"torch.{str(torch_dtype).rsplit('.', 1)[-1]}"

    # Resolve pluggable modules. Both halves of the contract may live in the
    # same file (the default case) or in different files. Each half is
    # recorded with its own absolute path + sha256.
    arr_mod, arr_prov = _resolve_codec_module(
        latent_module, ["arrange_latents", "unarrange_latents"]
    )
    ent_mod, ent_prov = _resolve_codec_module(
        entropy_module, ["encode_latents", "decode_latents"]
    )
    arrange_fn = arr_mod.arrange_latents
    unarrange_fn = arr_mod.unarrange_latents
    encode_fn = ent_mod.encode_latents
    decode_fn = ent_mod.decode_latents

    # Load model assets
    if verbose:
        print(f"Loading FRAPPE assets from {repo_id}/{subdir} ...", flush=True)
    config, weights, n_trained = load_from_hub(repo_id=repo_id, subdir=subdir)
    max_ps_at_n_trained = max(config.ps[:n_trained])

    # Load Kodak (deferred import: heavy and only needed in run())
    import datasets
    if verbose:
        print("Loading danjacobellis/kodak ...", flush=True)
    dataset = datasets.load_dataset("danjacobellis/kodak", split="validation")
    n_images = len(dataset)
    image_pixels = []
    images_pil = []
    for sample in dataset:
        img = sample["image"].convert("RGB")
        w, h = img.size
        _check_image_shape(h, w, max_ps_at_n_trained)
        image_pixels.append(h * w)
        images_pil.append(img)

    # Metric models
    import piq
    lpips_loss = piq.LPIPS().to(device).eval()
    dists_loss = piq.DISTS().to(device).eval()
    ssim_loss = piq.SSIMLoss().to(device)

    # Sweep. Re-encode per n_ch: progressive checkpoints carry distinct
    # encoder weights for each channel count, so we cannot reuse a prefix
    # of the n_trained latents for smaller n_ch without risking mismatch.
    results = {}
    for n_ch in range(1, n_trained + 1):
        if verbose:
            print(f"  n_ch={n_ch:2d} ...", end="", flush=True)
        model = load_progressive_model(weights, config, n_ch, device).eval()
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
        per_image = []
        t0 = time.time()
        for img in images_pil:
            metrics = _evaluate_one(
                model=model,
                img_pil=img,
                config=config,
                device=device,
                torch_dtype=torch_dtype,
                arrange_fn=arrange_fn,
                unarrange_fn=unarrange_fn,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                lpips_loss=lpips_loss,
                dists_loss=dists_loss,
                ssim_loss=ssim_loss,
            )
            per_image.append(metrics)
        mean = {m: _mean_with_fallback(per_image, m) for m in METRICS}
        results[str(n_ch)] = {"per_image": per_image, "mean": mean}
        if verbose:
            print(
                f" bpp={mean['bpp']:.4f}  PSNR={mean['PSNR_dB']:.2f}dB  "
                f"SSIM={mean['SSIM']:.4f}  ({time.time()-t0:.1f}s)",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build the latent_arrangement / entropy_coding config blocks. When both
    # halves point at the same source, emit the source string only once
    # (under entropy_coding) to keep the JSON compact; the latent_arrangement
    # block keeps the path + sha256 so the consumer can verify.
    arr_block = {
        "module_path": arr_prov["module_path"],
        "source_sha256": arr_prov["source_sha256"],
        "functions": ["arrange_latents", "unarrange_latents"],
    }
    ent_block = {
        "module_path": ent_prov["module_path"],
        "source_sha256": ent_prov["source_sha256"],
        "functions": ["encode_latents", "decode_latents"],
        "source": ent_prov["source"],
    }
    if arr_prov["source_sha256"] != ent_prov["source_sha256"]:
        arr_block["source"] = arr_prov["source"]

    out_data = {
        "id": run_id,
        "codec": "frappe",
        "task": "rate_distortion",
        "dataset": "danjacobellis/kodak",
        "n_images": n_images,
        "channel_counts": list(range(1, n_trained + 1)),
        "metrics": list(METRICS),
        "image_pixels": image_pixels,
        "config": {
            "weights": {
                "repo_id": repo_id,
                "subdir": subdir,
                "weights_filename": WEIGHTS_FILENAME,
                "config_filename": CONFIG_FILENAME,
                "n_trained": int(n_trained),
            },
            "dtype": dtype_label,
            "device": device,
            "linear_input": bool(getattr(config, "linear_input", False)),
            "max_ps_at_n_trained": int(max_ps_at_n_trained),
            "latent_arrangement": arr_block,
            "entropy_coding": ent_block,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(out_data, indent=2))
    if verbose:
        print(f"Wrote {output_path}")
    return output_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser():
    p = argparse.ArgumentParser(
        prog="python -m compressors.frappe.evaluate_rate_distortion",
        description=(
            "FRAPPE rate-distortion sweep on Kodak. Always evaluates every "
            "channel count n_ch in [1, n_trained]; writes one JSON per run."
        ),
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output path or directory. Default: cwd/rate_distortion_<id>.json.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", default="float", choices=sorted(DTYPE_NAMES),
        help="Inference dtype (default: float; fp16/bf16 not exercised in v1).",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--subdir", default=DEFAULT_SUBDIR)
    p.add_argument(
        "--latent-module", default=None,
        help=(
            "Path to a Python file exposing arrange_latents/unarrange_latents. "
            "Default: bundled compressors.frappe.entropy_coding."
        ),
    )
    p.add_argument(
        "--entropy-module", default=None,
        help=(
            "Path to a Python file exposing encode_latents/decode_latents. "
            "Default: bundled compressors.frappe.entropy_coding."
        ),
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    run(
        output=args.output,
        device=args.device,
        dtype=args.dtype,
        repo_id=args.repo_id,
        subdir=args.subdir,
        latent_module=args.latent_module,
        entropy_module=args.entropy_module,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
