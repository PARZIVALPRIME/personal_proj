# Qwen3.6-35B-A3B engineering roadmap (no more blind sweeps)

Hardware: Ryzen 7 5800HS (8C/16T Zen3, AVX2), RTX 3050 Laptop 4 GB, 16 GB DDR4, NVMe, CachyOS.
Goal 1: a usable local coding agent on this model. Goal 2 (later): gpt-oss-120b without the agent.
State: ~20 tok/s short decode, ~16 after a 5.3k prompt, 17 s first-prompt TTFT, agent generation 12-13 tok/s.
Sweeps of threads/RAM/spin/IO/bind/MTP are exhausted: all within noise. Run-to-run variance from free RAM at launch is larger than any
remaining setting, so every result below must be compared in one session with a fixed cache size.

## Where the time goes (from our own logs)

Decode step ~49-62 ms = expert fetch ~9-11 + expert compute ~18 + **~33 ms in the GPU chain and hand-offs** (40 serial GPU-dense ->
CPU-experts round trips, ~0.8 ms each). Cache hit 95.7% short, 91.6% long (77% when memory is tight). Memory bandwidth is not the limit
(~11 of ~50 GB/s). Prefill is 17 s for 5.3k tokens. The agent re-sends a large, mostly constant prefix every turn.

## Prior art we checked, and what we take from it

| Source | Idea | Use for us |
|---|---|---|
| PowerInfer (arXiv 2312.12456, SOSP'24) | Hot neurons on GPU, cold on CPU, power-law locality | Pin the hottest experts in VRAM, executed inside the chain (track D) |
| Fiddler (arXiv 2402.07033, ICLR'25) | Run experts where the data is; move activations, not weights | Already our model; justifies CPU experts + batch-size-aware placement |
| LLM in a flash (arXiv 2312.11514) | Windowing and row-column bundling for flash reads | Co-activation-ordered pack layout, bigger contiguous reads (track E) |
| KTransformers (kvcache-ai/ktransformers, kt-kernel AVX2 backend) | AMX/AVX CPU expert kernels + CPU-GPU expert scheduling, NUMA-aware | Compare our int4 AVX2 kernel; copy scheduling ideas |
| ik_llama.cpp (ikawrakow) | Fused MoE ops, repacked quants, `-ot` expert offload, fast CPU GEMV | Benchmark our expert kernel against its iqk kernels on the same shapes |
| MoE-Infinity, ProMoE, Pre-gated MoE, AdapMoE, ExpertFlow | Predict next-layer experts from the current hidden state, prefetch | Track C (we only prefetch top-k+extra from the router today) |
| HOBBIT, EdgeMoE, SwapMoE | Mixed-precision experts: cold ones at lower bits | Track G, quality-gated |
| EAGLE-3, Medusa, prompt-lookup / n-gram decoding | Several tokens per forward | Track F; our MTP head lost, but lookup is cheap and code is repetitive |
| llama.cpp prompt/slot cache, vLLM prefix cache, SGLang RadixAttention | Reuse KV of shared prefixes, across sessions | Track A, persisted to disk |

## Tracks, in priority order

### A. Persistent prefix state for the agent (biggest agent win, zero decode risk)
Agents resend a ~5k-token system prompt + tool schema + repo context every session. Today that costs ~17 s of TTFT per fresh server.
- Serialize the engine state (attention KV for the 10 attention layers + DeltaNet recurrent state for the 30 linear layers) after a prefix
  to `~/.cache/colibri/prefix/<model-hash>/<token-hash>.bin`, load with one sequential read. Hooks exist: `pin_save`, `kv_prefix_reuse`
  in `c/qwen36.c` (~8344, ~9481) and `c/kv_prefix.h`.
- Multi-slot LRU of several prefixes (system prompt, repo map, previous session) keyed by token-id hash, with exact-prefix check.
- Chunk-hashed radix lookup so a changed tail reuses the unchanged head.
- Success: first request with a known prefix has TTFT < 2 s instead of ~17 s; identical output tokens (fingerprint test).
- DeltaNet makes the state small and constant-size (no per-token KV for 30 layers), so the file is a few tens of MB.

### B. Instrument the 33 ms before changing anything
One timeline run (`COLI_VK_CHAIN_PROF=2` plus timestamps per layer: dense submit, GPU done, router readback, expert dispatch, expert done,
upload). Output: a per-layer table of GPU busy, idle-waiting-on-CPU, CPU idle-waiting-on-GPU, PCIe time. This decides C vs D. Deliverable
is a committed trace summary, not a sweep.

### C. Speculative next-layer expert execution (attack the serial hop)
Use the layer-L hidden state to predict layer L+1's experts (cheap router on the residual stream, as in Pre-gated MoE / ProMoE), start
their CPU compute or read early, then run the true router and keep the result only if the sets match (lossless; wasted work otherwise).
Even a 70% hit rate overlaps most CPU expert time with the next layer's GPU dense work. Measure prediction accuracy offline from logged
routing first (cheap, no engine change) and proceed only if it exceeds ~75%.

### D. GPU-resident hot experts inside the decode chain
VRAM: 4 GB minus 1.4 GB dense minus reserve leaves ~2 GB, i.e. ~1,100 of 10,240 experts (11%). Routing is skewed (heat file already
tracks it), so pinning the top experts per layer could serve a large share of activations with no hop. Earlier `tierdec` lost because
each expert call went through a separate tier path; the change is to execute pinned experts as part of the chain submission. Gate on
the B trace and on measured heat coverage (what fraction of decode activations fall in the pinned set).

### E. Cheaper misses and a stable cache
- Fixed cache size (`Q36_CACHE_SLOTS`) instead of sizing from free RAM at launch; mlock the expert cache; hugepages for it.
- Reorder `experts.q36x` so experts that co-activate sit adjacent (one larger read, LLM-in-a-flash style); `io_uring` with O_DIRECT
  and registered buffers for miss batches.
- Avoid zram competing with the cache: the benchmark runs showed ~3.9M pages swapped out. Consider a small zram or a swapfile on NVMe.

### F. More tokens per forward (works with the agent workload)
Coding output copies existing code (file edits, diffs, echoed identifiers). Prompt-lookup/n-gram drafts verified in one forward amortize
all 40 hops and expert loads over several tokens. Measure `tokens_per_forward` and accepted-draft rate on real agent turns
(`q36_agent_bench.py`), raise draft length on edit-style turns, and add suffix-automaton lookup over the whole conversation instead of
only the prompt. Our MTP head slowed decode because the draft head's cost and expert-union cost outweighed gains; lookup has no model cost.

### G. Lossy knobs, only behind a quality eval
Lower-bit cold experts, dynamic top-k (skip low-gate experts), expert pruning. The launcher currently forbids lossy modes; enable only
if a code-quality eval (track H) shows no regression.

### H. Coding-quality and agent harness (needed before any lossy change)
A small fixed set of completed coding tasks (edit a file, fix a failing test, write a function) scored pass/fail with the agent in `agent/`,
plus tokens-to-finish and wall time. Aider-polyglot / SWE-bench-Lite subsets are the reference for comparing with other local setups.

### I. Later: gpt-oss-120b
~117B params, ~5.1B active, MXFP4 (~60 GB). At 16 GB RAM the cache holds a small fraction, so it is NVMe-bound: roughly 5.1B active at
4 bits is ~2.5 GB/token of which most must come from disk on a miss, so expect low single-digit tok/s unless tracks C, E and F work. Reuse
the Q36 expert streamer and pack format.

## Order of work and exit criteria

1. **A** (prefix persistence) and **B** (trace) in parallel; both are low risk.
2. **F** measurement on real agent turns (cheap, may give the biggest decode gain for coding).
3. **C** predictor accuracy offline, then **D** heat-coverage offline; build whichever the numbers justify.
4. **E** stability work; **H** harness; **G** only if H exists.

Each item ships with: a unit test where the code is testable, an output-token-fingerprint parity check (lossless items), and one
single-session A/B (fixed cache size, alternating order) before being made default. A change becomes default only with >=3% decode or
>=25% TTFT gain, no new memory warnings, and identical output.
