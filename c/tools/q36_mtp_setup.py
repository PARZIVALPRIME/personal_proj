#!/usr/bin/env python3
"""One-time setup of Qwen3.6's MTP draft head next to an existing colibri container.

The converted container from Hugging Face has no `model-mtp.safetensors`. This adds it
without touching the original container:

  1. a virtual environment with CPU-only PyTorch, safetensors and huggingface_hub
     (only needed for the conversion, not to run the model);
  2. from Qwen/Qwen3.6-35B-A3B: config.json, the shard index and ONLY the shards that
     hold `mtp.*` tensors (two shards, about 6 GB);
  3. `convert_qwen36.py --mtp-only`: a new directory with the head quantized like the
     base (int4, group size 64) and every base file linked, never copied or modified;
  4. a check that the head file exists, then the downloaded shards are deleted.

Running it again after success does nothing. Use with Q36_MTP=1 and SNAP set to the
new directory (the launcher picks it automatically when Q36_MTP=1 and SNAP is unset).
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "c" / "tools" / "convert_qwen36.py"
REPO = "Qwen/Qwen3.6-35B-A3B"
HEAD = "model-mtp.safetensors"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
MIN_FREE = 8 * 1024 ** 3


def run(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def venv_python(venv):
    return Path(venv) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def can_import(command):
    return subprocess.run([str(part) for part in command], capture_output=True).returncode == 0


def ensure_venv(venv, run_cmd=run, probe=can_import):
    python = venv_python(venv)
    if not python.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv)])
    if not probe([python, "-c", "import torch, safetensors, numpy, huggingface_hub"]):
        run_cmd([python, "-m", "pip", "install", "--upgrade", "pip"])
        run_cmd([python, "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX])
        run_cmd([python, "-m", "pip", "install", "safetensors", "numpy", "huggingface_hub"])
    return python


def download(python, repo, names, dest, run_cmd=run):
    """hf_hub_download per file: resumable, and only the files named."""
    script = ("import sys\nfrom huggingface_hub import hf_hub_download\n"
              "for name in sys.argv[3:]:\n"
              "    print(hf_hub_download(sys.argv[1], name, local_dir=sys.argv[2]), flush=True)\n")
    run_cmd([python, "-c", script, repo, str(dest), *names])


def mtp_shards(index_path):
    weight_map = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
    shards = sorted({shard for name, shard in weight_map.items() if name.startswith("mtp.")})
    if not shards:
        raise RuntimeError(f"{index_path} lists no mtp.* tensors")
    return shards


def check_base(base):
    meta = base / "qwen36_meta.json"
    if not meta.is_file():
        raise RuntimeError(f"{base} is not a converted Qwen3.6 container (no qwen36_meta.json)")
    if (base / HEAD).exists():
        raise RuntimeError(f"{base} already has {HEAD}: use it directly with Q36_MTP=1")


def main(argv=None, run_cmd=run, probe=can_import):
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path(os.environ.get("SNAP") or home / "colibri-models" / "qwen36-35b"))
    parser.add_argument("--out", type=Path, help="default: <base>-mtp")
    parser.add_argument("--source", type=Path, default=home / "colibri-models" / "qwen36-mtp-source",
                        help="where the official shards are downloaded (deleted after success)")
    parser.add_argument("--venv", type=Path, default=home / ".venv-colibri")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--keep-source", action="store_true", help="keep the downloaded shards")
    args = parser.parse_args(argv)
    base = args.base.expanduser().resolve()
    out = (args.out or Path(str(base) + "-mtp")).expanduser()
    source = args.source.expanduser()
    if (out / HEAD).is_file():
        print(f"MTP head already set up: {out / HEAD}")
        print(f"Use it with:  Q36_MTP=1 SNAP={out}")
        return 0
    try:
        check_base(base)
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"{out} exists and is not empty, but has no {HEAD}: "
                               "remove it (it holds only links and metadata) and run again")
        source.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(source).free
        if free < MIN_FREE:
            raise RuntimeError(f"need about 8 GB free at {source}, found {free / 1024 ** 3:.1f} GB")
        python = ensure_venv(args.venv, run_cmd, probe)
        download(python, args.repo, ["config.json", "model.safetensors.index.json"], source, run_cmd)
        shards = mtp_shards(source / "model.safetensors.index.json")
        print(f"MTP tensors are in {', '.join(shards)}", flush=True)
        download(python, args.repo, shards, source, run_cmd)
        run_cmd([python, CONVERTER, "--model", source, "--mtp-only",
                 "--base-container", base, "--out", out])
        if not (out / HEAD).is_file() or (out / HEAD).stat().st_size == 0:
            raise RuntimeError(f"conversion finished but {out / HEAD} is missing or empty")
    except (RuntimeError, OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.keep_source:
        for name in shards:
            (source / name).unlink(missing_ok=True)
        print(f"Deleted the downloaded shards from {source}")
    print(f"MTP head ready: {out / HEAD} ({(out / HEAD).stat().st_size / 1024 ** 2:.0f} MB)")
    print(f"Use it with:  Q36_MTP=1 SNAP={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
