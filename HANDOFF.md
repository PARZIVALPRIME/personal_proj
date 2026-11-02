# Qwen3.6-35B-A3B turbo work: status and next steps

Handoff for the next session. Code is edited on a dev box with no compiler; it is built and
run on the target PC, and results come back through git (`benchmark-results/sweep-N/`).

## Target machine

- Windows 11, Ryzen 7 5800HS (8C/16T, AVX2), RTX 3050 Laptop 4 GB, 16 GB RAM, 512 GB NVMe
- Model: `qwen36-35b-a3b-colibri-i4-gs64` at `D:\colibri-models\qwen36-35b`
  (40 layers: 30 DeltaNet + 10 attention, 256 experts top-8, int4 gs64, 1.77 MB per expert)
- Source `D:\colibri-main\colibri-main`, runtime `C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0`
- Toolchain: LLVM-MinGW (clang); no nvcc or MSVC. Vulkan shaders are compiled by `c\build-turbo.bat vk`
  (glslc, or glslang downloaded once)

## How to run

```bat
git pull
call c\build-turbo.bat vk
set N_NEW=128
call benchmark-results\turbo-qwen36.bat sweep      &rem or: (no arg) serve, chat, bench, long, selftest, xpack
git add -A & git add -f benchmark-results\sweep-N\*.log & git commit -m "sweep-N" & git push
```

One-time step: `turbo-qwen36.bat xpack` writes `experts.q36x` (about 18 GB) beside the model.

## Results (measured on the target, normal quality)

| | Start | Now |
|---|---:|---:|
| Decode | 1.83 tok/s | 14.8-16.0 tok/s (15.29 in sweep-8) |
| TTFT, 32-token prompt | several s | ~1.2 s |
| TTFT, 5,338-token prompt | 81.6 s (sweep-1) | **19.1 s** (sweep-8, read-ahead on) |
| Exits in the last sweeps | some failures | all exit 0 |

Layer-by-layer prompt output was checked identical (hashed text) to chunk-by-chunk output,
on both the CPU and the GPU expert paths (sweep-6).

## What was built (by round)

1. Parallel expert loaders (`Q36_IO_THREADS=8`), segmented LRU cache (`Q36_EVICT=slru`),
   heat warm start (`Q36_HEAT`), real-size cache cap, Windows tuning.
2. Vulkan dense chain + lm_head on the GPU (`COLI_VK_TIER_RESERVE_GB=0.5`).
3. `experts.q36x` packed store with unbuffered split reads (`Q36_XPACK`), next-layer lookahead
   (`Q36_LOOKAHEAD`).
4. Missing Vulkan expert shaders compiled at build time (`build-turbo.bat :shaders`).
5. Expert-major prompt MoE: each expert read once per pass (`Q36_PF_EXPERT`, `Q36_PF_ROWS`).
6. Fixed a RAM leak: `vk_slot_src` attached a 1.5 MB unpacked copy to every cache slot.
7. GPU expert tier for prompts only; decode stays on the CPU (`Q36_VK_TIER_MIN_ROWS=64`,
   `COLI_VK_TIER_EXCLUSIVE=0`). On this card the tier is slower than the CPU for decode.
8. Layer-major long prompts on the Vulkan chain (`q36c_forward_lm`, `Q36_VK_LAYERWISE`,
   `Q36_VK_LAYERWISE_ROWS`): each layer's experts run once over the whole prompt.
9. Tier batch loads (`vk_load_batch`, `Q36_VK_LOAD_BATCH`), RAM-resident experts streamed first.
10. Round 13: read-ahead of each layer's misses while the GPU runs its dense part (`Q36_RA`,
    `Q36_RA_N`), 8192-row segments by default, smaller prefill buffers. Sweep-8: 19.1 s with it,
    21.3 s without (`Q36_RA=0`), 20.2 s with 256 buffers: 128 kept.

Launcher defaults (`benchmark-results\turbo-qwen36.bat`): GPU=vk, tier for prompts only,
`Q36_RAM_FRAC=0.92`, `Q36_MAXT=16384`, `COLI_LOOKUP=1`, quality mode (LOSSY=0).
Details per round: `benchmark-results/OPTIMIZATION_REPORT.md`.

## Where the time goes now

- Decode (~68 ms/token): ~38 ms GPU dense chain, ~20 ms CPU expert compute, ~12 ms expert fetch,
  92-94% cache hits.
- Long prompt (19.1 s, sweep-8): ~7 s GPU dense (attention/DeltaNet), ~11.9 s experts (about 4 s
  GPU compute, the rest loading ~6k experts and uploading them). CPU-only experts: 40.8 s.

Realistic targets on this hardware: decode 18-22 tok/s, 5.3k-token TTFT 14-17 s.
Hard floors: decode ~25-30 tok/s, TTFT ~11-12 s.

## Next session

1. Done: sweep-8 kept `Q36_RA` (128 buffers). Start with item 2.
2. Persist prompt state to disk for coding agents: the pin snapshots (`pin_save`,
   `q36_pin_state_save`: DeltaNet state) plus the KV rows, keyed by token prefix, so a repeated
   system prompt/tool list loads in ~1-2 s instead of a full prefill, also after a restart.
3. Run `turbo-qwen36.bat selftest` (OpenAI API, three chat turns) and connect a coding client.
4. Optional decode work: GPU dense chain overhead (38 ms against a ~8 ms bandwidth floor),
   CPU expert kernel (xf16_body256 register pressure on AVX2).
5. gpt-oss-120b: no engine in the repo (only CUDA MXFP4 kernels). Needs a new engine; on this
   hardware expect ~2-4 tok/s decode and 30-60 s TTFT on long prompts (60 GB of experts vs 16 GB RAM).
   The layer-major prefill, batch loading and read-ahead designs carry over.

## Cautions

- No build or test happens on the dev box: every change is unverified until the next sweep.
- `q36_io_worker` keeps `g_io_mx` held across its loop: do not add an unlock.
- Sweep logs are gitignored: push them with `git add -f`.
- If a default regresses: `set COLI_VK_TIER=0` (CPU experts), `Q36_VK_LAYERWISE=0` (chunk by chunk),
  `Q36_RA=0`, `Q36_PF_EXPERT=0`.
