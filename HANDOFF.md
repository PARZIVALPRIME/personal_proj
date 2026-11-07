# Qwen3.6-35B-A3B turbo work: status and next steps

## CURRENT STATE (read this first): moved from Windows to CachyOS Linux

### November 7: round 4 analyzed; round 5 implementation awaiting laptop measurements

Pulled `8709f68`. Final medians: short decode 20.01 tok/s (TTFT 1.454 s);
5,338-token prompt decode 16.82 tok/s (TTFT 16.743 s). No round-4 lever qualified.
MTP really ran on the GPU but reduced short decode to 11.5-14.1 tok/s.
The previous 22-25 tok/s forecast was not achieved and is not the current expectation.
Coding-agent follow-ups reused ~99% of history and started in 1.1-1.3 s;
actual coding generation was 12-13 tok/s with a smaller effective cache.
Thinking runs diverged on resent reasoning whitespace.

Implemented for round 5:
- Reference environment reset, with six threads/0.80 RAM/MTP off retained.
- Cache publication condition-variable waiting instead of millisecond polling.
- Exact direct-run timing and token fingerprints; per-layer decode cache counters;
  decode-only Vulkan frames/waits/transfers and opt-in timestamp scope.
- Identical private heat seeds and alternating profile runs.
- Opt-in spin0/spin200/spin2000/spin2000-io4 profiles; no automatic default change.
- Complete-workload, memory, timing and token-parity gates before winner saving;
  effective settings replace stale config rather than merging it.
- Persistent serving comparisons with telemetry, strict stream/usage validation,
  partial-result retention, token-limit failure and optional all-follow-up reuse gate.
- Opt-in Qwen36 whitespace-preserving history roundtrip, exercised by agent --think.

No laptop throughput improvement has been established for this implementation.
Development validation: 85 Q36-tool tests passed; 6 history tests passed with one
optional dependency skip; 13 focused gateway/splitter/keepalive tests passed.
Python and all 15 runbook Bash blocks passed syntax checks. The modified C engine,
Vulkan backend and cache regression compile-check with Linux/Vulkan/OpenMP/AVX2
headers. This Windows checkout cannot execute the native Linux/Vulkan inference
or the C concurrency regression: rebuild and run those on CachyOS in step 3.
GPU kernel rewrites, unequal layer cache allocations and new I/O backends remain
evidence-gated rather than speculative defaults. Current steps are in
[docs/CACHYOS-PERFORMANCE.md](docs/CACHYOS-PERFORMANCE.md).
The sections below are historical, including superseded forecasts and winner rules.

### November 7 (later): round 4 = optimize decode toward 30 tok/s, then coding tests

Decode-only speed (new `decode_tps` metric: tokens after the first / time after TTFT),
recomputed from the November 7 logs: **20.6 tok/s short, 16.8 after the 5.3k prompt**
(6 threads, 0.80 cache). 30 tok/s needs 1.45x; the forward is a serial GPU-dense ->
CPU-experts chain x 40 layers, so the only lossless lever with real headroom left is
more tokens per forward.

- **MTP draft head:** the engine supports Qwen3.6's MTP head (`Q36_MTP=1`, verified
  drafts, identical output) but the HF container `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64`
  has no `model-mtp.safetensors`. `turbo-qwen36.sh mtp-setup` (`c/tools/q36_mtp_setup.py`)
  downloads only the 2 official shards holding `mtp.*` (model-00025/26-of-00026, ~6 GB),
  runs `convert_qwen36.py --mtp-only` into `<model>-mtp` (head + links, base untouched),
  then deletes the download. Config/meta geometry was checked to match. Elsewhere one
  draft gave +7% (CPU) to +12-14% (GPU chain); here unmeasured, the head's VRAM/RAM fit
  decides whether it runs on the device or the CPU.
- **Free levers in the sweep:** `la2`, `la2x8` (lookahead depth/width), `tierdec`
  (GPU tier for decode rows; was slower on Windows), `gemv2` (other int8 GEMV on NVIDIA),
  `lookup7`, `mtp1`/`mtp2`/`mtpauto`.
- **Measured winners persist:** `turbo-qwen36.sh best <summary.json> --save` applies fixed
  rules (>=3% decode gain, no failures, >=512 MiB free, long prompt <=5% worse) and merges
  the winner into `~/.config/colibri/q36.env`, which the launcher applies unless the shell
  sets the variable (`Q36_CONF=none` ignores it). MTP profiles/settings switch SNAP to
  `<model>-mtp` automatically.

Expected: ~22-25 tok/s decode. If short of 30, next work is the ~33 ms/token outside the
expert timers (GPU chain, hand-offs): the round also captures `COLI_VK_CHAIN_PROF=2`.
Steps 1-14 in [docs/CACHYOS-PERFORMANCE.md](docs/CACHYOS-PERFORMANCE.md).

### November 7: memory comparison done, new defaults, round 3 is next

Pulled `35bd4da` (`benchmark-results/linux-memory-20261107-000107`, 256 tokens out,
medians of 3). Six threads + `Q36_RAM_FRAC=0.80`: short 18.52 tok/s, long TTFT
16.457 s, long decode step 62.6 ms, >=1.41 GiB free. Six threads at 0.92: short
19.81 tok/s but long TTFT 18.14 s, 129 MiB free and up to 226,580 pages swapped.
**The launcher now defaults to `OMP_NUM_THREADS=6` and `Q36_RAM_FRAC=0.80`**
(`OMP_NUM_THREADS=auto`, `Q36_RAM_FRAC=0.92` restore the old behaviour; the runner's
`previous` profile does the same).

Remaining decode step (long, 62.6 ms): expert fetch 10.8, expert compute 18.3,
outside the expert timers 33.1 ms. Next round (all commands in
[docs/CACHYOS-PERFORMANCE.md](docs/CACHYOS-PERFORMANCE.md)):
1. `turbo-qwen36.sh agent`: persistent server, 4 streamed coding turns over the 5.3k-token
   context, per-turn TTFT and the engine's `[PREFIX]` reuse lines (`COLI_PREFIX_LOG=1`).
   The server already reuses the state when the history is resent; thinking is off by
   default through the API. Check that turns 2-4 actually reuse.
2. Defaults vs previous sweep at 256 tokens.
3. Vulkan chain profile with the engine's own `COLI_VK_CHAIN_PROF=2` (control runs without
   it). The runner now parses `[VK] qwen36 chain` waits/forwards and keeps the profile lines.
Then pick ONE implementation from the profile (dense shapes/shaders, submissions/fusion,
or the CPU expert kernel); profile prefill separately for the 16.5 s long TTFT.

### November 5: controlled results received, memory pressure was next

Pulled `64f0e5e` and reviewed `benchmark-results/linux-session-20261105-200755`.
Six compute threads improved the sweep's short median from 15.74 to 17.15 tok/s
(step 65.5 -> 59.7 ms); passive waiting is not a clear winner. Long TTFT is about
16.2 s, but sustained long runs have major memory pressure: minimum available
83,240 KiB, swap-out up to 832,430 system pages, decode step median 102.5 ms.
System-wide swap cannot identify ownership, but it invalidates a claim of
comfortable desktop headroom. Keep swap enabled; compare reduced expert caching.

The `long/` directory accidentally contains short `bench` runs. Actual long
workloads exist under `sweep/`, `sustained-long/` and `rebuilt-long/`.

The runner now has `ram80` / `threads6-ram80`, memory-warning counts and optional
`--stop-on-pressure`, plus explicit prompt path/output-limit metadata. That
comparison has since been run (see November 7 above).

The target machine is now the same laptop booted into CachyOS (XFCE4). Same hardware (Ryzen 7 5800HS,
RTX 3050 Laptop 4 GB, 16 GB RAM, NVMe). Model is at `~/colibri-models/qwen36-35b`, so always set `SNAP`.
Engine is built with `make -C c qwen36 VK=1 -j"$(nproc)"`. Launcher: `benchmark-results/turbo-qwen36.sh`
(Linux twin of `turbo-qwen36.bat`, same defaults). Vulkan finds the RTX 3050, all 40 layers on the GPU.

### Latest verified target results: xpack is FIXED

Commit `45d8fae` contains successful post-xpack CachyOS runs:

| Metric | Windows sweep-8 | CachyOS post-xpack |
|---|---:|---:|
| Short reported speed | 15.29 tok/s | 16.42-16.74 tok/s |
| Short TTFT | 1.218 s | 1.256-1.291 s |
| 5,338-token TTFT | 19.121 s | 17.363-18.522 s |
| Short expert fetch | 11.99 ms/token | 8.22-8.80 ms/token |

`experts.q36x` is active. The old missing-pack diagnosis below is historical.
Long-run Speed includes prefill, not decode-only throughput. Short/long answers
were truncated by the 128/48-token limits; use completed coding tasks for quality.

### November 4 (done): controlled offline-prepared experiments

Historical: these tooling steps were completed on November 5. For the current steps,
read [docs/CACHYOS-PERFORMANCE.md](docs/CACHYOS-PERFORMANCE.md). The launcher
exports SNAP, propagates failures, isolates quality mode and delegates experiments
to `c/tools/q36_bench.py` (sweeps) and `c/tools/q36_agent_bench.py` (coding agent).

```bash
cd ~/personal_proj
git pull --ff-only
export SNAP="$HOME/colibri-models/qwen36-35b"
python3 -m unittest discover -s c/tests -p test_q36_bench.py -v
bash benchmark-results/turbo-qwen36.sh doctor
bash benchmark-results/turbo-qwen36.sh sweep --profiles baseline passive io4 --repeats 3
```

The runbook explains engine backup, optional forced rebuild, longer output tests,
result review/sharing and rollback. Preserve the current binary before rebuilding.
No new laptop speedup has been measured. No kernels, synchronization or prefix
state formats have been changed blindly. Default cache/placement stays unchanged.

Next implementation decisions need target evidence: CPU/GPU timing breakdown,
CPU bandwidth/thread contention, decode-only misses, then prefix-state reuse audit.
Q36 already has `kv_prefix_reuse`, `pin_save` and DeltaNet pin-state helpers;
do not assume prefix reuse is absent or restart persistence is complete.

### Historical pre-xpack Linux results against Windows sweep-8

| | Windows | Linux, no xpack |
|---|---:|---:|
| Decode | 14.8-16 tok/s | 9.77 (warm), 4.46 (cold) |
| Expert fetch per token | ~12 ms | 28 ms warm, 79 ms cold |
| TTFT 5,338 tokens | 19.1 s | 22.9 s warm, 30.2 s cold |
| Warm-start read | 2.09 GB/s | 0.7-1.05 GB/s |

In these old runs, every log said `experts.q36x does not exist`. The packed store (Q36_XPACK, one unbuffered
1.77 MB read per miss) was never built on Linux, so experts come from the original safetensors.
Secondary: peak RSS 12-13 GB of 16 GB (desktop apps shrink the expert cache), and the usage/heat
history (`.coli_usage`, `.q36_heat`) is still warming up.

### Historical recovery commands (already completed; not the next experiment)

```bash
cd ~/personal_proj && git pull
export SNAP=$HOME/colibri-models/qwen36-35b
df -h "$SNAP"; free -h; lsblk -d -o NAME,ROTA,SIZE,MODEL      # need ~18 GB free; model must be on the NVMe
benchmark-results/turbo-qwen36.sh xpack                      # ONE TIME, writes $SNAP/experts.q36x (~18 GB)
ls -lh "$SNAP/experts.q36x"                                   # must be ~18 GB
# close browser/heavy apps, then each twice (second run is warm):
benchmark-results/turbo-qwen36.sh bench
benchmark-results/turbo-qwen36.sh bench
benchmark-results/turbo-qwen36.sh long
benchmark-results/turbo-qwen36.sh long
git add -f benchmark-results/bench-turbo-*.log
git commit -m "linux bench after xpack" && git push origin main
```

Success check: the log must no longer say `experts.q36x does not exist`; target decode 14+ tok/s,
fetch ~12 ms/token, 5.3k-token TTFT about 19-20 s. If still slow after xpack, try `OMP_NUM_THREADS=8`,
`GPU=0` (CPU only), `powerprofilesctl set performance`, and report `vulkaninfo --summary`, `lscpu`, `free -h`.
Plan after that: item 2 under "Next session" below (persist prompt state to disk).

---

# Windows-era status (previous machine setup)

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
