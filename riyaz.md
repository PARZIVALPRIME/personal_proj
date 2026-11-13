# riyaz: project status (Qwen3.6-35B-A3B on a 4 GB laptop)

Summary of everything done so far, what is measured and confirmed, what was achieved, and what was not.
Sources: `HANDOFF.md`, `docs/CACHYOS-PERFORMANCE.md`, `benchmark-results/`, git history (latest commit `94dea6e`, Round 5).

## Goal

Run Qwen3.6-35B-A3B (40 layers: 30 DeltaNet + 10 attention, 256 experts, top-8, int4 gs64, ~18 GB of experts) fast and usably on a
laptop with limited hardware, using the colibri engine (`c/`) with a Vulkan GPU backend.
Original aim for decode was about 30 tok/s.

## Hardware

Ryzen 7 5800HS (8C/16T, AVX2), RTX 3050 Laptop 4 GB, 16 GB RAM, NVMe.
First run on Windows 11, then moved to CachyOS Linux (XFCE) on the same laptop. Model at `~/colibri-models/qwen36-35b`.

## What we did (timeline)

### Windows era (sweeps 1-8, 13 optimisation rounds)
1. Parallel expert loaders, segmented LRU cache, heat warm start.
2. Vulkan dense chain + lm_head on the GPU.
3. Packed expert store `experts.q36x` (unbuffered split reads), next-layer lookahead.
4. Vulkan expert shaders compiled at build time.
5. Expert-major prompt MoE (each expert read once per pass).
6. Fixed a RAM leak (1.5 MB unpacked copy per cache slot).
7. GPU expert tier for prompts only; decode stays on the CPU.
8. Layer-major long-prompt processing on the Vulkan chain.
9. Tier batch loads, RAM-resident experts streamed first.
10. Read-ahead of each layer's misses during its dense part.

### Linux (CachyOS) era
- Ported launcher (`benchmark-results/turbo-qwen36.sh`), README and runbook.
- Found `experts.q36x` was missing on Linux (decode 4.5-9.8 tok/s); built it (xpack) and it was fixed.
- Controlled experiment tooling: `c/tools/q36_bench.py` (sweeps), `c/tools/q36_agent_bench.py` (coding agent), memory and pressure tracking.
- Round 3: memory comparison, 6 threads vs auto, RAM fraction 0.80 vs 0.92. Adopted 6 threads / 0.80 as defaults.
- Round 4: decode-only metric, MTP draft head setup (`mtp-setup`), coding-agent benchmark, lever sweep.
- Round 5: reference-environment reset, cache-publication condition variable (no polling), exact timing and token fingerprints, per-layer
  decode cache counters, decode-only Vulkan counters, spin/IO scheduling profiles (spin0/spin200/spin2000/io4), stricter winner gates,
  persistent serving comparisons, whitespace-preserving thinking history. Results pushed in `benchmark-results/linux-memory-20261109-230841`.

## Results (measured)

| Stage | Short decode | Short TTFT | 5,338-token TTFT |
|---|---:|---:|---:|
| Start (Windows) | 1.83 tok/s | several s | 81.6 s |
| Windows sweep-8 | 15.29 tok/s | 1.22 s | 19.1 s |
| Linux post-xpack | 16.4-16.7 tok/s | 1.26-1.29 s | 17.4-18.5 s |
| Linux, 6 threads + 0.80 cache (round 4) | ~20.0 tok/s | 1.45 s | 16.7 s |
| Round 5 scheduling profiles, short (medians of 5) | 20.2-20.8 tok/s | ~1.3 s | n/a |
| Round 5 scheduling profiles, 5.3k prompt (decode) | 16.1-17.1 tok/s | n/a | ~17.4 s |

Round 5 detail: expert-cache hit rate ~95.7% (short) and ~91.6% (long); free memory stayed above the safety floor with the 0.80 cache.
Profiles `baseline`, `io4`, `spin0`, `spin200`, `spin2000`, `spin2000-io4` all landed within noise of each other (differences under 3%).
Confirmation runs show occasional slow outliers (minimum decode 4-7 tok/s) while medians stay ~18.4 short / 14-16 long.

## Established / confirmed

- `experts.q36x` packed store is essential on Linux and is active.
- Six threads + `Q36_RAM_FRAC=0.80` is the best measured, safe default (0.92 is slightly faster on short runs but causes heavy swap and slower long prompts).
- Layer-by-layer prompt output is identical to chunk-by-chunk output (hashed) on CPU and GPU expert paths.
- All 40 layers run on the Vulkan chain on the RTX 3050.
- Coding-agent follow-ups reuse ~99% of history and start in 1.1-1.3 s.
- Decode step is a serial GPU-dense to CPU-experts chain x 40 layers; ~33 ms/token is outside the expert timers.
- Round 5 scheduling/spin/IO profiles give no gain that qualifies under the winner rules (>=3% decode gain, no failures, memory and parity gates).
- Development validation: 85 Q36-tool tests, 6 history tests, 13 gateway/splitter/keepalive tests passed; all 15 runbook Bash blocks syntax-checked.

## Achieved

- ~11x decode speedup from the initial 1.83 tok/s to ~20 tok/s on a 4 GB GPU / 16 GB RAM laptop.
- 5.3k-token prompt TTFT cut from 81.6 s to ~16.7-17.4 s.
- Reproducible, gated, failure-aware benchmark and serving harness with results committed to git.
- Memory-safe defaults and a working coding-agent loop with prefix reuse.

## Not achieved / disproved

- The 30 tok/s decode target (need ~1.45x more); the 22-25 tok/s forecast was not reached.
- MTP draft head ran on the GPU but reduced short decode to 11.5-14.1 tok/s; not enabled.
- `tierdec` (GPU tier for decode) was slower; passive-wait/spin profiles gave no clear win.
- Coding generation in agent mode is only 12-13 tok/s with a smaller effective cache.
- Thinking runs diverged on resent reasoning whitespace (an opt-in roundtrip fix exists, not yet proven default-safe).
- Long-prompt TTFT is still ~17 s; sustained runs occasionally show slow outliers and memory pressure.
- Quality on completed coding tasks is not yet evaluated (short/long answers were truncated by token limits).

## Next steps

1. Pick ONE implementation from the decode profile: dense GPU shapes/shaders, submissions/fusion, or the CPU expert kernel.
2. Profile prefill separately to attack the ~17 s long-prompt TTFT.
3. Investigate outlier runs and memory pressure; consider unequal per-layer cache allocation (only if evidence supports it).
4. Run complete coding tasks to evaluate output quality and agent throughput.
5. Re-run the native Linux/Vulkan C regression and rebuild (`make -C c qwen36 VK=1 -j"$(nproc)"`) after any engine change.

## How to reproduce

```bash
cd ~/personal_proj && git pull --ff-only
export SNAP="$HOME/colibri-models/qwen36-35b"
bash benchmark-results/turbo-qwen36.sh doctor
bash benchmark-results/turbo-qwen36.sh sweep --profiles baseline io4 --repeats 3
```

Full runbook: `docs/CACHYOS-PERFORMANCE.md`. Detailed history: `HANDOFF.md`.
