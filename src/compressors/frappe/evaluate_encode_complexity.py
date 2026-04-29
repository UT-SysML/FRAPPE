"""Encode-complexity evaluation harness for FRAPPE on Kodak.

Sweeps every truncated channel count ``n_ch ∈ [1, n_trained]`` on
512×512 center crops of Kodak, with timing decomposed into three stages
(mirrors ``liveaction/eval/intel.ipynb``):

- ``analysis`` — encoder forward pass + int8 quantization (accelerator).
- ``transfer`` — accelerator → host copy of the quantized latents.
- ``store`` — latent arrangement + entropy coding (CPU; uses the same
  pluggable four-function contract as v1's rate–distortion harness).

Preprocessing (PIL decode, center-crop, host → device copy) is excluded
from the timer; the loop body starts with the [-1, 1] float input
tensor already resident on the accelerator.

Writes one self-describing ``encode_<id>.json`` per run with per-n_ch
throughput (in megapixels/second) and per-stage timings (median + mean
in seconds).

CLI: ``python -m compressors.frappe.evaluate_encode_complexity`` with
the flags listed in :func:`run`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import socket
import sys
import time
import uuid
from pathlib import Path

import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from . import entropy_coding as default_entropy_coding
from .model import load_from_hub, load_progressive_model
from .quantize import srgb_to_linear


DEFAULT_REPO_ID = "danjacobellis/FRAPPE"
DEFAULT_SUBDIR = "FRAPPE"
WEIGHTS_FILENAME = "FRAPPE_pytorch_model.safetensors"
CONFIG_FILENAME = "config.json"
INPUT_RESOLUTION = (512, 512)
DEFAULT_N_WARMUP = 1
DEFAULT_N_MEASUREMENT = 5
STAGES = ("analysis", "transfer", "store")

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
# Pluggable module loading (same pattern as v1)
# ----------------------------------------------------------------------

def _file_provenance(path):
    abs_path = str(Path(path).resolve())
    source = Path(abs_path).read_text()
    return {
        "module_path": abs_path,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source": source,
    }


def _load_module_from_path(path):
    abs_path = Path(path).resolve()
    name = f"_compressors_codec_{abs_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_codec_module(path_arg, required_fns):
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
# Device + testbed metadata
# ----------------------------------------------------------------------

def _device_kind(device):
    """Return the wallclock-compatible device kind ('cuda', 'xpu', or 'cpu')."""
    s = str(device).lower()
    for kind in ("cuda", "xpu"):
        if s.startswith(kind):
            return kind
    return "cpu"


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _testbed_info(device, n_warmup, n_measurement):
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "n_warmup": int(n_warmup),
        "n_measurement": int(n_measurement),
    }
    cpu = _cpu_model()
    if cpu is not None:
        info["cpu_model"] = cpu
    kind = _device_kind(device)
    if kind == "cuda" and torch.cuda.is_available():
        idx = torch.device(device).index
        if idx is None:
            idx = torch.cuda.current_device()
        info["accelerator_name"] = torch.cuda.get_device_name(idx)
        info["cuda_version"] = torch.version.cuda
    elif kind == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = torch.device(device).index
        if idx is None:
            idx = torch.xpu.current_device()
        info["accelerator_name"] = torch.xpu.get_device_name(idx)
    return info


# ----------------------------------------------------------------------
# Input staging
# ----------------------------------------------------------------------

def _center_crop_to_device(img_pil, crop_h, crop_w, torch_dtype, device):
    """PIL → 512×512 center crop → [-1, 1] float tensor on device."""
    rgb = img_pil.convert("RGB")
    iw, ih = rgb.size
    if ih < crop_h or iw < crop_w:
        raise ValueError(
            f"Image {iw}x{ih} smaller than required crop {crop_w}x{crop_h}"
        )
    left = (iw - crop_w) // 2
    top = (ih - crop_h) // 2
    cropped = rgb.crop((left, top, left + crop_w, top + crop_h))
    t = pil_to_tensor(cropped).to(torch_dtype).unsqueeze(0) / 127.5 - 1.0
    return t.to(device).contiguous()


# ----------------------------------------------------------------------
# Per-n_ch measurement
# ----------------------------------------------------------------------

def _stats(values):
    """Return median + mean of a list of floats."""
    if not values:
        return {"median": 0.0, "mean": 0.0}
    vs = sorted(values)
    n = len(vs)
    median = vs[n // 2] if n % 2 == 1 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])
    return {"median": float(median), "mean": float(sum(vs) / n)}


def _measure_one_n_ch(
    *,
    model,
    inputs_on_device,
    config,
    arrange_fn,
    encode_fn,
    wallclock,
    n_warmup,
    n_measurement,
):
    """Run warmup + measurement loops.

    Returns ``(timings_per_stage, blob_sizes_per_image)``. Blob sizes are
    captured during the *first* timed pass via ``len(blob)`` *outside* the
    ``wallclock("store")`` context, so the bpp measurement does not perturb
    throughput timings. The encoded output is deterministic across passes,
    so a single pass is sufficient.
    """

    def _run_pass(timed, collect_sizes):
        sizes = []
        for x in inputs_on_device:
            x_in = srgb_to_linear(x) if getattr(config, "linear_input", False) else x
            if timed:
                with wallclock("analysis"):
                    with torch.inference_mode():
                        latents = model.encode(x_in)
                        latents_q = [
                            z.round().clamp(-127, 127).to(torch.int8) for z in latents
                        ]
                with wallclock("transfer"):
                    latents_cpu = [z.cpu() for z in latents_q]
                with wallclock("store"):
                    arranged = arrange_fn(latents_cpu)
                    blob = encode_fn(arranged)
                if collect_sizes:
                    sizes.append(len(blob))
            else:
                with torch.inference_mode():
                    latents = model.encode(x_in)
                    latents_q = [
                        z.round().clamp(-127, 127).to(torch.int8) for z in latents
                    ]
                latents_cpu = [z.cpu() for z in latents_q]
                arranged = arrange_fn(latents_cpu)
                blob = encode_fn(arranged)
                if collect_sizes:
                    sizes.append(len(blob))
        return sizes

    for _ in range(n_warmup):
        _run_pass(timed=False, collect_sizes=False)
    wallclock.reset()
    blob_sizes = []
    for i in range(n_measurement):
        sizes = _run_pass(timed=True, collect_sizes=(i == 0))
        if i == 0:
            blob_sizes = sizes

    return (
        {stage: list(wallclock.timings[stage]) for stage in STAGES},
        blob_sizes,
    )


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
    n_warmup=DEFAULT_N_WARMUP,
    n_measurement=DEFAULT_N_MEASUREMENT,
    verbose=True,
):
    """Run a single encode-complexity sweep of FRAPPE on Kodak. Returns the output ``Path``.

    Always sweeps every ``n_ch ∈ [1, n_trained]``; there is no ``--n-ch`` flag.
    """
    run_id = int(time.time())

    if output is None:
        output_path = Path.cwd() / f"encode_{run_id}.json"
    else:
        output_path = Path(output)
        if output_path.is_dir() or str(output).endswith("/"):
            output_path = output_path / f"encode_{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dtype, str):
        if dtype not in DTYPE_NAMES:
            raise ValueError(
                f"Unknown dtype {dtype!r}; expected one of {sorted(DTYPE_NAMES)}"
            )
        torch_dtype = DTYPE_NAMES[dtype]
    else:
        torch_dtype = dtype
    dtype_label = f"torch.{str(torch_dtype).rsplit('.', 1)[-1]}"

    arr_mod, arr_prov = _resolve_codec_module(
        latent_module, ["arrange_latents", "unarrange_latents"]
    )
    ent_mod, ent_prov = _resolve_codec_module(
        entropy_module, ["encode_latents", "decode_latents"]
    )
    arrange_fn = arr_mod.arrange_latents
    encode_fn = ent_mod.encode_latents

    if verbose:
        print(f"Loading FRAPPE assets from {repo_id}/{subdir} ...", flush=True)
    config, weights, n_trained = load_from_hub(repo_id=repo_id, subdir=subdir)
    max_ps_at_n_trained = max(config.ps[:n_trained])
    crop_h, crop_w = INPUT_RESOLUTION
    if crop_h % max_ps_at_n_trained or crop_w % max_ps_at_n_trained:
        raise ValueError(
            f"Crop {crop_w}x{crop_h} not divisible by max patch size "
            f"{max_ps_at_n_trained}; choose a multiple."
        )

    import datasets
    if verbose:
        print("Loading danjacobellis/kodak ...", flush=True)
    dataset = datasets.load_dataset("danjacobellis/kodak", split="validation")
    n_images = len(dataset)
    if verbose:
        print(
            f"Pre-staging {n_images} center crops {crop_w}x{crop_h} "
            f"on {device} (excluded from timing) ...",
            flush=True,
        )
    inputs_on_device = [
        _center_crop_to_device(s["image"], crop_h, crop_w, torch_dtype, device)
        for s in dataset
    ]

    # Configure the wallclock singleton for this device. The throughput
    # package exposes `wallclock` as a module-level instance whose `device`
    # attribute drives torch.{cuda,xpu}.synchronize() calls inside the
    # context manager.
    from throughput.image import wallclock
    wallclock.device = _device_kind(device)

    n_pixels_per_image = crop_h * crop_w
    results = {}
    for n_ch in range(1, n_trained + 1):
        if verbose:
            print(f"  n_ch={n_ch:2d} ...", end="", flush=True)
        model = load_progressive_model(weights, config, n_ch, device).eval()
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)

        t0 = time.time()
        timings, blob_sizes = _measure_one_n_ch(
            model=model,
            inputs_on_device=inputs_on_device,
            config=config,
            arrange_fn=arrange_fn,
            encode_fn=encode_fn,
            wallclock=wallclock,
            n_warmup=n_warmup,
            n_measurement=n_measurement,
        )

        per_stage = {stage: _stats(timings[stage]) for stage in STAGES}
        # Per-image total time = sum of three stages, image by image.
        per_image_totals = [
            timings["analysis"][i] + timings["transfer"][i] + timings["store"][i]
            for i in range(len(timings["analysis"]))
        ]
        total_stats = _stats(per_image_totals)
        median_total = total_stats["median"]
        mean_total = total_stats["mean"]
        throughput_block = {
            "median_MPx_per_s": (
                n_pixels_per_image * 1e-6 / median_total
                if median_total > 0
                else float("inf")
            ),
            "mean_MPx_per_s": (
                n_pixels_per_image * 1e-6 / mean_total
                if mean_total > 0
                else float("inf")
            ),
            "total_per_image": total_stats,
        }

        bpp_per_image = [s * 8 / n_pixels_per_image for s in blob_sizes]
        bpp_block = {
            "per_image": bpp_per_image,
            "mean": float(sum(bpp_per_image) / len(bpp_per_image)) if bpp_per_image else 0.0,
        }

        results[str(n_ch)] = {
            "throughput": throughput_block,
            "stages": per_stage,
            "bpp": bpp_block,
        }

        if verbose:
            ms = lambda s: per_stage[s]["median"] * 1000
            print(
                f" analysis={ms('analysis'):6.2f}ms"
                f"  transfer={ms('transfer'):6.2f}ms"
                f"  store={ms('store'):6.2f}ms"
                f"  bpp={bpp_block['mean']:5.3f}"
                f"  → {throughput_block['median_MPx_per_s']:.2f} MPx/s"
                f"  ({time.time()-t0:.1f}s)",
                flush=True,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    # latent_arrangement / entropy_coding provenance blocks (mirror v1).
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
        "task": "encode_complexity",
        "dataset": "danjacobellis/kodak",
        "input_resolution": [crop_h, crop_w],
        "n_images": n_images,
        "channel_counts": list(range(1, n_trained + 1)),
        "stages": list(STAGES),
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
        "testbed": _testbed_info(device, n_warmup, n_measurement),
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
        prog="python -m compressors.frappe.evaluate_encode_complexity",
        description=(
            "FRAPPE encode-complexity sweep on Kodak (512x512 center crops). "
            "Always evaluates every n_ch in [1, n_trained]; writes one JSON per run."
        ),
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output path or directory. Default: cwd/encode_<id>.json.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", default="float", choices=sorted(DTYPE_NAMES),
        help="Inference dtype (default: float).",
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
    p.add_argument(
        "--n-warmup", type=int, default=DEFAULT_N_WARMUP,
        help=(
            "Number of untimed warmup epochs over the dataset per n_ch. "
            f"Default: {DEFAULT_N_WARMUP}."
        ),
    )
    p.add_argument(
        "--n-measurement", type=int, default=DEFAULT_N_MEASUREMENT,
        help=(
            "Number of timed measurement epochs over the dataset per n_ch. "
            f"Default: {DEFAULT_N_MEASUREMENT} (matches liveaction/eval/intel.ipynb)."
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
        n_warmup=args.n_warmup,
        n_measurement=args.n_measurement,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
