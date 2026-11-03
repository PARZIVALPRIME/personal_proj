# Qwen3.6-35B-A3B on a small machine (colibri, turbo build)

Runs Qwen3.6-35B-A3B (int4, 256 experts, top-8) on a 16 GB RAM laptop with a 4 GB GPU by
streaming experts from NVMe. Measured on Ryzen 7 5800HS + RTX 3050 Laptop 4 GB:

| | Start | Now |
|---|---:|---:|
| Decode | 1.83 tok/s | 14.8-16.0 tok/s |
| TTFT, 32-token prompt | several s | ~1.2 s |
| TTFT, 5,338-token prompt | 81.6 s | 19.1 s |

What was optimized: parallel expert loaders, segmented-LRU cache with heat warm start, Vulkan
dense chain + lm_head, packed `experts.q36x` store with unbuffered reads, lookahead, expert-major
and layer-major prompt processing, GPU expert tier for prompts, batch loads and read-ahead.
Full history: [HANDOFF.md](HANDOFF.md) and [benchmark-results/OPTIMIZATION_REPORT.md](benchmark-results/OPTIMIZATION_REPORT.md).

## Run on CachyOS (Arch, XFCE4)

Needs about 20 GB free disk for the model plus 18 GB for `experts.q36x`, and an NVMe drive.

### 1. Packages

```bash
sudo pacman -Syu --needed base-devel git python python-pip openmp \
    vulkan-icd-loader vulkan-headers shaderc vulkan-tools
# GPU driver (pick yours)
sudo pacman -S --needed nvidia-utils            # NVIDIA (CachyOS: use its nvidia driver package)
# sudo pacman -S --needed vulkan-radeon         # AMD
# sudo pacman -S --needed vulkan-intel          # Intel
vulkaninfo --summary                            # your GPU must be listed
```

### 2. Get the code

```bash
git clone https://github.com/PARZIVALPRIME/personal_proj.git
cd personal_proj
```

### 3. Build the engine (CPU + Vulkan)

```bash
make -C c qwen36 VK=1 -j"$(nproc)"
```

CPU only: `make -C c qwen36 -j"$(nproc)"`. If `glslc` fails on a shader the committed `.spv` is kept.

### 4. Download the model

```bash
pip install --user -U huggingface_hub   # on Arch use a venv if pip refuses: python -m venv ~/.venv && . ~/.venv/bin/activate
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir ~/Models/qwen36_i4_gs64
```

### 5. One-time expert pack

```bash
chmod +x benchmark-results/turbo-qwen36.sh
benchmark-results/turbo-qwen36.sh xpack      # writes experts.q36x (~18 GB) next to the model
```

### 6. Run

```bash
benchmark-results/turbo-qwen36.sh            # OpenAI-compatible server
benchmark-results/turbo-qwen36.sh chat       # interactive chat
benchmark-results/turbo-qwen36.sh bench      # speed on prompt.txt
benchmark-results/turbo-qwen36.sh long       # ~2k-token prompt, shows time to first token
```

Another model folder: `SNAP=/path/to/model benchmark-results/turbo-qwen36.sh`.
Prefer a guided setup? `./start-here.sh` picks a model and builds for you.

### Defaults in the launcher

`Q36_IO_THREADS=8`, `Q36_EVICT=slru`, `Q36_HEAT=1`, `Q36_RAM_FRAC=0.92`, `Q36_LOOKAHEAD=1`,
`Q36_XPACK=1`, `Q36_MAXT=16384`, `COLI_LOOKUP=1`, Vulkan trunk, GPU expert tier for prompts only
(`Q36_VK_TIER_MIN_ROWS=64`), `COLI_VK_TIER_RESERVE_GB=0.5`.

### If something regresses

`GPU=0` (CPU only), `COLI_VK_TIER=0` (CPU experts), `Q36_VK_LAYERWISE=0`, `Q36_RA=0`, `Q36_PF_EXPERT=0`.
Close browsers first: the expert cache is sized from free RAM.

## Run on Windows

```bat
call c\build-turbo.bat vk
call benchmark-results\turbo-qwen36.bat xpack
call benchmark-results\turbo-qwen36.bat
```

Paths inside the `.bat` files point at the original machine; set `SNAP`, `SRC`, `COLI_RUNTIME`
and `LLVM_MINGW_BIN` before calling.

## Push changes

```bash
git add -A
git commit -m "Add README and Linux launcher"
git push origin main
```

Benchmark logs are gitignored; add them with `git add -f benchmark-results/sweep-N/*.log`.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
