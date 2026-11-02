# Qwen3.6-35B-A3B on a 4 GB laptop: engineering notes

Target: Ryzen 7 5800HS (8C/16T, Zen 3, AVX2), RTX 3050 Laptop 4 GB, 16 GB RAM,
Windows 11, NVMe. Model: `qwen36-35b-a3b` Colibri int4 group-64.

**What these notes are and aren't.** No line here was measured on the target. The
changes were compiled and unit-tested on a different Windows host without a GPU,
and the C tests ran with zig cc, not MSVC/nvcc. Each claim below is tagged:

- **[code]**: read in the source.
- **[test]**: covered by a test that passes in this repository.
- **[estimate]**: comes from `c/tools/q36_target_plan.py`'s model with assumed hardware numbers.
- **[hypothesis]**: needs a run on the target.

## 1. Where the time goes (architecture map)

```
coli serve|chat --model <dir>
  c/openai_server.py      gateway: resource plan -> cap_for_arch() -> spawns the engine
    argv: qwen36 <cap> <bits>   env: SNAP=<model dir>, plan env via setdefault (user env wins)
  c/qwen36.c main()
    load dense trunk (attention, DeltaNet, shared expert, router, lm_head)
    cap == 0 -> qwen36_cap_for_ram(): expert-cache slots per layer from free RAM
    CUDA tier placement (COLI_CUDA): trunk matrices, then experts, into the allowance
    trunk probe: GPU vs CPU GEMV timing -> keep or withdraw trunk / experts
    warm start: fill VRAM and RAM slots (heat order with HEAT_FILE)
    CUDA chain (COLI_CUDA_CHAIN): whole layer stack on the card, host runs routed experts
  decode step, per layer (40: 30 DeltaNet + 10 gated attention)
    router top-8 of 256 -> each expert: VRAM tier | RAM LRU slot | pread from SSD
```

Per token the 35B-A3B touches 40 x 8 routed experts of about 1.6 MB each at
int4/gs64, about 0.5 GB in all **[estimate]**. Every expert that misses the RAM
cache is an SSD read, so decode speed depends mostly on the cache hit rate
**[estimate]**.

## 2. Root causes found and fixes made

| # | Finding | Evidence | Fix | Status |
|---|---|---|---|---|
| 1 | On 16 GB the gateway's RAM plan came out at 0 slots. `cap_for_arch` floored that to top-k = 8 and passed 8 as an **explicit** cap, which turned off the engine's own auto-sizing. The cache was then pinned at 8 of 256 experts per layer, so nearly every token streamed its experts from the SSD. | **[code]** `c/openai_server.py` `cap_for_arch` | Under the floor, qwen36 now gets cap 0 (auto). qwen38 is unchanged. | **[test]** `tests.test_openai_server -k cap` |
| 2 | The engine's auto-size used peak RSS, did not hold back RAM for the KV cache (which is allocated after the sizing), and could go below top-k. | **[code]** `qwen36_cap_for_ram` | Uses the current working set (`rss_now_gb`), reserves the KV cache at `Q36_MAXT` (`Q36_KV_RESERVE`), and floors at top-k. | **[test]** `test_qwen36_cap_precedence.c` |
| 3 | The trunk probe timed GEMVs cold. Each call carried a host↔device round trip that the CUDA chain pays only about once per layer. A laptop GPU at idle clocks with a full sync per GEMV can lose to the CPU, withdraw the trunk, and so disable the chain. | **[code]** `trunk_probe_gpu_wins` | Time-based warm-up (`COLI_TRUNK_PROBE_WARM_MS`). The probe measures the fixed round-trip cost and, when the chain will run, spreads it over `COLI_TRUNK_PROBE_PER_SYNC` GEMVs. The experts keep the synchronous verdict (#2211). | **[test]** `test_qwen36_trunk_probe.c`. Whether the verdict flips on a 3050 is a **[hypothesis]**. |
| 4 | On this 16 GB host, the automatic chat context selected 32k positions, reserving more KV space and reducing the expert cache. A warm 32-token request took 12.63 s at 32k versus 10.06 s at an explicit 8k context. | **[measurement]** 8 physical CPU threads, same model/prompt/sampling, two warm requests per setting | Automatic Qwen3.6 context selection now keeps the engine's 8192-token default on hosts with at most 16 GB when the operator did not set `--ctx`, `Q36_MAXT`, or `--ram`. | **[test]** `test_context_compaction.py`; longer contexts remain opt-in. |

Commits (local only): `58207297` (fixes 1-3, docs), `1794b88f` (planner).

## 3. The planner

```powershell
python c\tools\q36_target_plan.py                       # text report
python c\tools\q36_target_plan.py --json                # machine-readable
python c\tools\q36_target_plan.py --ps1 run-q36.ps1 --model-dir D:\colibri-models\qwen36-35b
python c\tools\q36_target_plan.py --config <model>\config.json --ram-gb 16 --vram-gb 4
```

The defaults below are **[estimate]** and every hardware number is an assumption:

- **Expert cache:** about 109 slots/layer.
- **KV reserve:** 0.62 GiB at 32k positions in f16. That is 20 KiB/token, since only the 10 attention layers keep KV.
- **CUDA trunk:** 1.81 GiB, which fits a 2.70 GiB allowance.
- **Decode:** about 5 tok/s with the trunk on the GPU vs about 4 on the CPU.

The generated `.ps1` only sets variables you have not already set, then runs `coli`. Override the assumptions with flags (`--help`) once you have measured the target.

## 4. Knobs that matter on this machine

| Variable | Suggested | Why |
|---|---|---|
| cap (engine argv[1]) | leave 0 | Auto-sizing from measured RAM (fix 1). |
| `Q36_MAXT` | 8192-32768 | 8192 is the automatic small-host default; set this explicitly for a longer context. Every 1k positions holds back 20 MiB of the expert cache. |
| `COLI_CUDA`, `COLI_CUDA_CHAIN` | `1` | Trunk and chain on the card if it fits. |
| `CUDA_EXPERT_GB` | unset first | Numeric values replace the 1 GiB headroom and disable expert withdrawal. Use only if the log says the chain declined for lack of room. |
| `HEAT_FILE` | `<model>\heat.bin` | Warm start in heat order. The second run starts placed and the promotion brake engages. |
| `DIRECT`, `PIPE` | A/B each | Unbuffered reads and I/O overlap. Gains depend on the drive (see ENVIRONMENT.md). |
| `PILOT` | A/B, chain off only | Cross-layer prefetch. The CUDA chain declines under PILOT, so it is an either/or. |
| `CACHE_ROUTE` | off by default | Lossy (changes routing). A/B with `ROUTE_AGREE` before any use. |
| `OMP_NUM_THREADS` | 8 | Physical cores. SMT siblings rarely help GEMV. |
| `COLI_TRUNK_PROBE_WARM_MS` / `_PER_SYNC` | 250 / 4 | New. `PER_SYNC=1` restores the old verdict. |
| `Q36_KV_RESERVE` | on | New. `0` gives the KV room back to the cache. |

## 5. What other projects do (research summary)

Sources were read by a research pass on the projects' READMEs and design docs.
Figures are **as reported by each project on its own hardware**. Nothing here
was re-run, and items marked unverified could not be confirmed from source.

| Project | Technique relevant here | Windows | Qwen3.6 hybrid | License | Takeaway for Colibri |
|---|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `-ot` regex tensor placement, `--cpu-moe`, `--n-cpu-moe`, `--moe-cache-mib` (GPU cache for CPU experts) | native | `QWEN35MOE` arch in source (correctness unverified) | MIT | Baseline to compare against on the target: `-ot ".ffn_.*_exps.=CPU"` with the trunk on the 3050. |
| [HotPin](https://github.com/LozzKappa/hotpin-llm) | Pin a popularity-ranked hot set in RAM, plus readahead | Win/Linux | no | AGPL-3.0 | Pinning helps only when the model exceeds RAM (it does here). Ideas only, no code reuse (AGPL). |
| [expert-stream](https://github.com/Heman10x-NGU/expert-stream) | Batched `NO_BUFFERING\|OVERLAPPED` expert reads instead of mmap faults | Windows-only | no | MIT | Queue depth was the win on a close sibling (4800H, 16 GB, 1660 Ti). Colibri already has overlapped/unbuffered `pread` (`compat.h`, `DIRECT`, `PIPE`). Check reads in flight per layer on the target. |
| [NeutronStar](https://github.com/giannisanni/neutronstar) | LFU host cache, persistent warm-start index, router look-ahead prefetch | Linux (io_uring) | no | MIT | Reports a concentrated hot set (about 4% of experts serve about 30% of lookups). Supports LFU/heat-weighted eviction over plain LRU (**[hypothesis]** for Qwen3.6). |
| [Pulsar](https://github.com/giannisanni/pulsar) | Rust+CUDA, runs Qwen3.6-35B-A3B; KV on 10/40 layers, split-K long-context attention | Linux-only | **yes** | MIT | The only project here with Qwen3.6 numbers (16 GB cards). Its K-quant vs IQ-quant result agrees with staying on int4 gs64. |
| [moe-ssd-streaming-windows](https://github.com/tonbistudio/moe-ssd-streaming-windows) | Stock llama.cpp plus Windows page cache; Win32 I/O microbenchmarks | native | no | MIT (README) | Its microbenchmarks and a PCIe x2-lane misconfiguration finding make a cheap drive sanity check on the target. |
| [AirLLM](https://github.com/lyogavin/airllm) | Layer-by-layer HF checkpoint streaming | cross-platform | unconfirmed | Apache-2.0 | Not expert-selective as far as could be verified. Low value here. |
| [ds4](https://github.com/antirez/ds4) | Native C SSD-streaming engine for a fixed model list | no | no | MIT | Ancestor of NeutronStar/Pulsar. Little direct value for a 4 GB Windows laptop. |

## 6. Prioritized backlog

Each item is a **[hypothesis]** until it is A/B'd on the target with the protocol in §7.

1. **Validate fixes 1-3 on the target.** Run `coli doctor` and `coli plan`, then check the log for `cache auto-sized: N slots/layer`, where N should be far above 8, and for whether the trunk probe keeps the trunk.
2. **Measure the drive.** Use `c/iobench.c`, or the tonbistudio microbenchmarks, at queue depths 1/4/16 with and without `DIRECT`. This tells whether `PIPE`/`DIRECT` can pay at all.
3. **`HEAT_FILE` warm start.** One training conversation, then compare decode tok/s and hit rate with and without the file. Cheap, and already implemented.
4. **CUDA chain on 4 GB.** Check the KV mirror on the device at the chosen `Q36_MAXT`. If the chain declines, try a smaller `Q36_MAXT` before raising `CUDA_EXPERT_GB`.
5. **Heat-weighted (LFU-like) eviction for the RAM cache.** Prompted by NeutronStar's popularity skew. Needs a route trace from Qwen3.6 first to confirm the skew.
6. **PILOT vs chain.** On a card where the chain fits, compare chain-on against chain-off with `PILOT=1`.
7. **`CACHE_ROUTE`.** Lossy, so judge it on quality as well as speed: `ROUTE_AGREE` overlap/KL plus a task eval, not just tok/s.
8. **Prefill chunking on 4 GB.** Measure time-to-first-token for 2k/8k prompts. The chain's prompt chunks compete with the expert tier for VRAM.

## 7. How to validate on the target

```powershell
coli doctor                                   # hardware and model detection
python c\tools\q36_target_plan.py --ps1 run-q36.ps1
.\run-q36.ps1                                 # check the startup log lines below
$env:COLI_TIMERS=1; $env:N_NEW=200; $env:COLI_TEMP=0   # fixed, greedy, timed runs
```

Startup lines to read:

- `[qwen36] cache auto-sized: <N> slots/layer of 256 experts ...`, which includes the KV reserve.
- The trunk probe verdict line and whether the CUDA chain started or declined (it names the first matrix on the CPU).
- `[qtier] HEAT_FILE loaded/saved`.

A/B protocol:

- Same prompt, `COLI_TEMP=0`, 3 runs each, report the median.
- Change one variable at a time.
- Record the drive, power plan (AC, "best performance"), and free RAM at start.
- Keep raw logs under `experiments/` like the existing runs.

`agent/` (coliagent) can drive this: `python -m coliagent improve --gate ... --metric-argv ...` accepts a change only if every gate passes and the median metric beats the baseline by `--min-gain`.

## 8. Limitations

- No measurement on the RTX 3050 / 5800HS. Bandwidths, OS RAM use, VRAM taken by the display, and the probe verdict on that GPU are assumptions.
- The fake-CUDA chain test is timing-sensitive. It failed once out of six runs under concurrent compile load, and that test does not cover the changed code.
- Build used zig cc (`x86_64-windows-gnu`). An MSVC + nvcc CUDA build was not done here.
