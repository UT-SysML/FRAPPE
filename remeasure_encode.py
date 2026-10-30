"""Remeasure encoder throughput for the eight FRAPPE paper codecs at each library's default threading.

Each codec runs in its own process, one at a time, through the evaluator in the installed
``compressors`` wheel, with the settings read from the pinned JSON. Pillow encoders receive no
``max_threads``/``num_threads`` argument, and no thread-count environment variable is set.

The mbt2018 and MCUCoder evaluators call ``torch.set_num_threads(1)`` and write the literal
``torch_num_threads: 1`` to their config. This script replaces that call with a recorder during the
run, then replaces the literal with the observed value. Threading is added to ``testbed`` as observed.

Usage: python remeasure_encode.py [codec ...]
"""

import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HYBRID_V2 = Path.home() / "danjacobellis/compressors/experiments/encoder_optimization/hybrid_v2.py"

# codec -> (evaluator module, pinned JSON, extra CLI arguments)
CODECS = {
    "jpeg": ("compressors.pillow.evaluate_encode_complexity", "encode_1777321267.json", ["--format", "JPEG"]),
    "avif": ("compressors.pillow.evaluate_encode_complexity", "encode_1777321337.json", ["--format", "AVIF"]),
    "jxl": ("compressors.jxl.evaluate_encode_complexity", "encode_1777603371.json", []),
    "mbt2018": ("compressors.mbt2018.evaluate_encode_complexity", "encode_1777585754.json", []),
    "walloc": ("compressors.walloc.evaluate_encode_complexity", "encode_1777571761.json", []),
    "liveaction": ("compressors.liveaction.evaluate_encode_complexity", "encode_1777571794.json", []),
    "mcucoder": ("compressors.mcucoder.evaluate_encode_complexity", "encode_1777680434.json", []),
    "frappe": (
        "compressors.frappe.evaluate_encode_complexity",
        "encode_1777342722.json",
        ["--device", "cpu", "--latent-module", str(HYBRID_V2), "--entropy-module", str(HYBRID_V2)],
    ),
}
QUALITY_FLAG = {"jpeg", "avif", "jxl", "walloc", "liveaction"}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
VERSIONS = ("torch", "pillow", "pillow-jxl-plugin", "imagecodecs", "zstandard", "pillow-jpls", "walloc", "livecodec", "compressors", "throughput")


def observed_threading(requested):
    info = {
        "cpu_count": os.cpu_count(),
        "sched_affinity_count": len(os.sched_getaffinity(0)),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
    }
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()
        info["torch_set_num_threads_calls_suppressed"] = requested
    if "PIL.AvifImagePlugin" in sys.modules:
        info["pillow_avif_default_max_threads"] = sys.modules["PIL.AvifImagePlugin"]._get_default_max_threads()
    if "pillow_jxl" in sys.modules:
        info["pillow_jxl_default_num_threads"] = -1  # JpegXLImagePlugin._save: info.get("num_threads", -1)
    return info


def child(codec):
    module_name, pinned_name, extra = CODECS[codec]
    pinned = json.loads((RESULTS / codec / pinned_name).read_text())
    argv = ["--output", f"{RESULTS / codec}/", *extra]
    if codec in QUALITY_FLAG:
        argv += ["--quality", *map(str, pinned["quality_values"])]

    requested = []
    if codec in ("mbt2018", "mcucoder"):
        import torch
        torch.set_num_threads = requested.append

    module = importlib.import_module(module_name)
    start = time.time()
    loadavg_start = os.getloadavg()
    cpu_start = os.times()
    status = module.main(argv)
    cpu_end = os.times()
    wall = time.time() - start
    threads_at_exit = next(l for l in Path("/proc/self/status").read_text().splitlines() if l.startswith("Threads:")).split()[1]

    out = max((RESULTS / codec).glob("encode_*.json"), key=lambda p: p.stat().st_mtime)
    data = json.loads(out.read_text())
    if not data["id"] >= int(start):
        raise RuntimeError(f"{out} predates this run")
    if "torch_num_threads" in data["config"]:
        data["config"]["torch_num_threads"] = sys.modules["torch"].get_num_threads()
    data["testbed"].update(observed_threading(requested))
    data["testbed"].update({
        "hostname": socket.gethostname(),
        "loadavg_at_start": list(loadavg_start),
        "process_wall_seconds": wall,
        "process_cpu_seconds": (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system),
        "process_threads_at_exit": int(threads_at_exit),
        "library_versions": {p: _version(p) for p in VERSIONS},
        "pinned_run": pinned_name,
    })
    out.write_text(json.dumps(data, indent=2))
    print(f"RESULT {codec} {out}", flush=True)
    return status


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    if sys.argv[1:2] == ["--child"]:
        return child(sys.argv[2])
    for codec in sys.argv[1:] or CODECS:
        print(f"=== {codec} {time.strftime('%H:%M:%S')} load={os.getloadavg()[0]:.2f}", flush=True)
        subprocess.run([sys.executable, __file__, "--child", codec], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
