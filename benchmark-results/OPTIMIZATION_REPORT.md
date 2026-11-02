# Colibri Qwen3.6 Optimization Report

Date: 2026-11-02

## Original configuration

- OS: Windows 11
- CPU: AMD Ryzen 7 5800HS, 8 physical cores / 16 threads
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, sm_86
- RAM: 16 GB
- Model: `D:\colibri-models\qwen36-35b`
- Model size: 23,031,433,063 bytes
- Runtime: `C:\Users\riyaz\AppData\Local\colibri\runtime\colibri-v2.1.0`
- Executable: `qwen36.exe` v2.1.0
- Source checkout: `D:\colibri-main\colibri-main`
- Git state: this path is not a Git working tree, so no commit or patch state was available.

The downloaded model was not modified. The installed executable and CUDA DLLs were
also left untouched. No source files required modification because the installed
runtime already contains the CUDA build.

## Source and runtime inspection

Inspected:

- `c\qwen36.c`
- `c\expert_ffn.h`
- `c\gsgemv.h`
- `c\qwen36_tier.h`
- `c\backend_cuda.cu`
- `c\cu_tier.c`
- `c\build_cuda.bat`
- `c\omp_tune.h`
- `c\expert_store.h`
- `c\spec_draft.h`
- `c\pin_pool.h`
- `c\Makefile`
- `docs\qwen36.md`
- `docs\qwen36-cuda-tier.md`
- `docs\ENVIRONMENT.md`

The source supports planar INT4 expert execution, INT8 dense matrices, FP16 KV
cache, prompt lookup (`COLI_LOOKUP`, enabled by default), physical-core OpenMP
sizing, and an optional CUDA VRAM tier. The installed LLVM-MinGW toolchain
provided a native CPU rebuild; CUDA still requires `nvcc` and MSVC, which are
not installed on this machine.

## Baseline

Command environment:

```text
SNAP=D:\colibri-models\qwen36-35b
TOK=D:\colibri-models\qwen36-35b\tokenizer.json
N_NEW=32
COLI_CUDA=0
COLI_VULKAN=0
OMP_NUM_THREADS=8
Q36_MAXT=2048
```

The benchmark prompt is in `prompt.txt`. Each direct-engine run completed with
exit code 0 and generated 32 tokens.

| Configuration | Trial speeds (tok/s) | Median tok/s | Median TTFT | Median peak RSS |
|---|---:|---:|---:|---:|
| CPU, 8 threads | 1.25, 1.73, 1.35 | **1.35** | 11.342 s | 9.61 GB |
| CUDA, 1 GB allowance | 1.47, 1.37, 1.34 | **1.37** | 13.152 s | 9.20 GB |
| Native CPU rebuild, 8 threads | 1.87, 1.83, 1.81 | **1.83** | 9.083 s | 9.72 GB |

The native CPU rebuild is approximately **35.6% faster** than the original
CPU runtime and approximately **33.6% faster** than the CUDA runtime on this
workload. Its median time to first token is also lower. Results remain somewhat
noisy because the workload is short and expert reads are storage/cache
sensitive.

The complete Windows launcher I/O defaults (`DIRECT=1`, `PIPE=1`,
`PILOT_REAL=1`) were also tested on the native binary. Three completed trials
measured 1.26, 1.52, and 1.53 tok/s (median **1.52 tok/s**), slower than the
native buffered-I/O median of **1.83 tok/s**. The native launcher therefore
explicitly sets `DIRECT=0`, `PIPE=0`, and `PILOT_REAL=0`; this is a measured
machine-specific choice, not a claim that direct I/O is generally slower.

## GPU utilization follow-up

The research folder was inspected, including its `moe-streaming/pulsar`
implementation and GPU-kernel projects. Pulsar uses a different runtime and
checkpoint representation, so copying its kernels directly would not preserve
Colibri's Qwen3.6 numerical and storage contracts. The useful compatible idea
was already present in Colibri's own CUDA tier: reserve VRAM for routed experts,
keep CPU fallback, and make placement explicit.

Two GPU configurations were tested:

| Configuration | GPU telemetry | Speed | Result |
|---|---|---:|---|
| Experts only, 0.5 GB, cap 64 | 283 resident experts; 19.2% decode VRAM hit rate | 1.30 tok/s | Stable, slower |
| LM head + experts, 1.5 GB, cap 64 | 0.47 GB LM head + 1.03 GB experts; 29.0% decode VRAM hit rate | **1.35 tok/s** | Stable GPU-enabled option |

The GPU kernels reported zero transfer/kernel milliseconds in the short
summary because the tier's asynchronous work is overlapped; the authoritative
proof of activation is the resident-expert count, GPU hit count, CUDA mode, and
LM-head placement in the logs. Both tests completed with exit code 0 and valid
output. The hybrid configuration is intentionally separate from the fastest
native CPU launcher: it uses the GPU as requested, but the native CPU build
still wins this workload at 1.83 tok/s.

Use `known-good-qwen36-gpu-hybrid.bat` when GPU utilization is preferred over
maximum measured throughput. It leaves DeltaNet and attention projections on
the CPU, places the LM head on CUDA, and keeps a bounded 1.03 GB routed-expert
budget. The explicit placement avoids the earlier failure mode where automatic
placement filled the VRAM budget with the dense trunk and left no experts, or
where an experts-only 1.5 GB configuration exhausted VRAM during startup.

## SSD overlap follow-up

The research implementation's most transferable idea is overlapping future
expert reads with current-layer compute. Qwen36 now has an opt-in bounded
single-producer/multiple-consumer pilot queue controlled by `PILOT_WORKERS`.
Workers reserve cache slots under the existing lock before reading, so
concurrent reads cannot publish the wrong expert or evict a slot in use. The
default remains one worker and `PILOT=0`; routing and numerical output are
unchanged.

On this laptop, two workers were measured at **1.54 tok/s cold** and **1.94
tok/s warm**, versus **2.17 tok/s** for the same warm native CPU path without
pilot. The extra prediction matmul and CPU contention outweigh the SSD overlap
for this workload, so the feature is not enabled by the recommended launcher.
It is available for future tests with `PILOT=1 PILOT_WORKERS=2` and should only
be kept when `COLI_TIMERS=1` shows an end-to-end gain.

Prompt lookup remains the best quality-preserving local acceleration. In
controlled runs on the same prompt, lookup-off measured **1.82 tok/s**, while
the default gated lookup measured **2.40 tok/s** and five-draft lookup reached
**2.47 tok/s**. Ungated lookup fell to **1.89 tok/s**, confirming that forcing
low-confidence drafts wastes full verification work. The native launcher
already keeps lookup enabled and uses the five-draft default.

The installed model directory contains no `mtp.*` or next-token draft weights,
so `Q36_MTP` cannot provide the larger speculative-decoding gain described in
the model documentation. Adding that head is the remaining software path
likely to approach a further 30--50% on repetitive/code output; it requires
the matching MTP checkpoint weights and cannot be synthesized from the base
35B weights without changing response quality.

## Practical performance limit

The 35B checkpoint is approximately 23 GB on disk, while the RTX 3050 has 4 GB
VRAM and the machine has 16 GB RAM. Only a fraction of the routed experts can
be resident on the GPU, so most decode steps still wait for expert data from
storage. Moving a small expert subset or the LM head to CUDA adds PCIe/WDDM
transfer and synchronization costs; the measured GPU configurations were
slower than the native CPU build. The research Pulsar numbers are from a
different engine and substantially larger GPUs, so they are not transferable
to this laptop.

Cloud-like response speed requires a model whose active weights fit mostly in
VRAM (a much smaller checkpoint), a larger GPU, or remote inference. Further
35B tuning on this machine can improve stability and cache behavior but cannot
remove the storage and memory-bandwidth ceiling.

## Selected configuration and exact environment

The selected configuration is the native CPU rebuild with `ARCH=native`.
It was built from the source checkout using the available LLVM-MinGW compiler:

```text
make -C c qwen36 ARCH=native LDFLAGS="-lm -fopenmp -pthread -lpsapi -LC:\Users\riyaz\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\x86_64-w64-mingw32\lib"
```

The explicit OpenMP library path was required because the package provides
`libomp.dll.a` rather than a static `libomp.a`. The runtime DLL is added to
`PATH` by `known-good-qwen36-native.bat`.

The selected binary reports the planar INT4 expert kernel and interleaved INT8
CPU trunk at startup. All three 32-token trials completed with exit code 0 and
valid output.

The previous CUDA configuration remains available in
`known-good-qwen36-cuda.bat`. Its runtime telemetry confirmed:
Runtime telemetry confirmed:

- CUDA device opened successfully as sm_86.
- Dense trunk placed on CUDA.
- CUDA VRAM used approximately 1.01 GB.
- Zero routed experts fit after the dense trunk placement.
- Routed experts remained on the CPU.
- CUDA runs completed normally with valid output.

The 1.5 GB test had the same placement and used approximately 1.51 GB VRAM,
without improving the measured speed. The 1 GB setting leaves more safety
margin on a 4 GB card.

## Recommended launch

Use `known-good-qwen36-native.bat`. Its effective environment is:

```text
COLI_ENGINE=D:\colibri-main\colibri-main\c\qwen36.exe
COLI_CUDA=0
COLI_VULKAN=0
OMP_NUM_THREADS=8
Q36_MAXT=2048
```

## Failed or unhelpful optimizations

- Local CUDA rebuild: not attempted because `nvcc` and MSVC `cl` are absent.
- The first native build attempt failed at OpenMP linking; adding the existing
  LLVM-MinGW import-library path and `-pthread` produced the validated binary.
- CUDA experts-only placement with
  `COLI_PLACE=experts=0,lmhead=cpu,dnproj=cpu,dnout=cpu,attnproj=cpu` and
  `CUDA_EXPERT_GB=1.5`: startup failed with `CUDA tensor allocation: out of
  memory`. This configuration is not recommended.
- Increasing the allowance from 1 GB to 1.5 GB did not leave room for routed
  experts under automatic dense-trunk placement and did not improve speed.
- Prompt lookup produced no accepted drafts on this short Fibonacci workload;
  it was left at the engine default rather than forcing a behavior change.
- Vulkan was not tested because the installed runtime and objective target the
  NVIDIA CUDA path, and no Vulkan build change was needed.

## Stability and limitations

- The selected CUDA path is stable for the tested 32-token prompt and reports
  valid output with exit code 0.
- The measured gain is small; CPU may win on other prompts or system load.
- The benchmark did not include long-context or multi-request serving.
- GPU utilization was not continuously sampled during the direct runs; CUDA
  activation and VRAM placement were verified by engine telemetry.
- The model remains unchanged. The native binary depends on the LLVM-MinGW
`libomp.dll` supplied by the existing compiler package; the launcher adds its
directory to `PATH`.

## Rollback

Run the same command with `COLI_CUDA=0` and `COLI_VULKAN=0`, or remove the
CUDA-related environment variables. The original installed executable and model
remain in place; no replacement executable was installed.

Benchmark logs and the environment record are stored beside this report:

- `baseline-cpu-trial1.log` through `baseline-cpu-trial3.log`
- `cuda-1gb-trial1.log` through `cuda-1gb-trial3.log`
- `cuda-1.5gb-trial1.log`
- `cuda-experts-only-trial1.log`
- `cpu-pilot-real-trial1.log`
- `native-cpu-smoke.log`
- `native-cpu-trial1.log` through `native-cpu-trial3.log`
- `qwen36-source-before-native.exe`
- `baseline-environment.json`

## Follow-up bottleneck test

The existing value-preserving prefetch path was tested with:

```text
PILOT=1
PILOT_REAL=1
PILOT_K=6
PILOT_WORKERS=2
PILOT_EVICT_GUARD=1
```

It completed successfully, but was slower on the same workload: **1.16 tok/s**
with **240.03 ms/token** fetch time and **123.83 ms/token** expert compute,
versus the baseline's 1.35 tok/s median. It was therefore not enabled in the
recommended configuration. The result indicates that this short workload's
prefetch predictions and extra I/O contention do not hide SSD latency reliably.

## Turbo launcher validation on this PC

The turbo scripts were built and tested on the Ryzen 7 5800HS machine with the
same model and `N_NEW=128`. The new parallel `Q36_IO_THREADS=8` path crashed
with a Windows access violation during expert loading. The stable launcher now
honors an explicit `Q36_IO_THREADS` value and defaults it to `0` for this
safetensors build. The worker completion path also now releases its queue mutex
after publishing a loaded slot. The existing model and rollback executable were
preserved.

Three stable native runs completed successfully:

| Run | TTFT | Generation speed | Fetch | Compute | Cache hit |
|---|---:|---:|---:|---:|---:|
| Cold | 10.604 s | 2.19 tok/s | 223.08 ms/token | 36.22 ms/token | 76.9% |
| Warm 1 | 4.173 s | 3.48 tok/s | 124.58 ms/token | 26.84 ms/token | 84.3% |
| Warm 2 | 3.594 s | 3.71 tok/s | 112.75 ms/token | 26.68 ms/token | 86.6% |

With `LOSSY=1` (`Q36_SKIP_MISS=0.06`, four experts always retained), the
three runs measured 4.10, 5.04, and 5.22 tok/s. They skipped 1.1%, 0.7%, and
0.5% of routed experts respectively. This is a speed/quality tradeoff and is
not enabled by default.

The requested Vulkan build could not be run because `VULKAN_SDK` is not set on
this PC. No Vulkan executable or benchmark result is claimed. Install the
Vulkan SDK and reopen the terminal before running `c\build-turbo.bat vk`.

The raw turbo logs are `bench-turbo-1.log` through `bench-turbo-3.log` for the
stable native runs and `bench-turbo-4.log` through `bench-turbo-6.log` for the
lossy runs.

The subsequent warm native run (`bench-turbo-7.log`) completed normally at
**4.42 tok/s**, with **2.912 s TTFT**, **87.8% cache hit rate**, **91.09
ms/token expert fetch**, and **23.15 ms/token expert compute**. It used the
same stable CPU configuration with gated lookup and the persisted heat cache.
This is a warm-run result, not a guaranteed cold-start speed.

## Development handoff

The current pushed implementation is ready for development on another
computer. The model weights are not stored in Git. On the second machine,
install the required compiler/runtime dependencies, update the paths in
`c\build-turbo.bat` and `benchmark-results\turbo-qwen36.bat`, then run:

```cmd
call c\build-turbo.bat
set N_NEW=128
call benchmark-results\turbo-qwen36.bat bench
```

Use the same prompt and output length for comparisons. Keep
`Q36_IO_THREADS=0` unless the parallel loader is fixed and validated for that
machine; it currently crashes on this Windows safetensors configuration.
Test `LOSSY=1` separately because it trades response fidelity for speed.
The next high-value work is to fix and benchmark parallel expert loading,
install and test Vulkan, and add an output-quality comparison for lossy mode.
Do not enable CUDA or Vulkan by default without an end-to-end speedup over the
native CPU path.

## Parallel loader crash: root cause and next round

The access violation with `Q36_IO_THREADS=8` was in `moe_xf_run`, not in the
loaders: a prefill run of 64 pairs or fewer took the overlapped
residents-then-misses path, but its output buffer had only been allocated when
the whole batch was that small, so a long prompt summed from a NULL pointer.
The buffer is now always sized for 64 pairs and the path checks it. The extra
`pthread_mutex_unlock` added to `q36_io_worker` was reverted: the worker keeps
the queue mutex across the loop because the wait at its top needs it held.
`Q36_IO_THREADS=8` is the launcher default again.

Also changed for the next round:

- Vulkan builds no longer need the SDK: `build-turbo.bat vk` clones the
  Khronos headers when `VULKAN_SDK` is unset (shaders are committed).
- A Vulkan run used to rebuild the 2 GB f32 embedding table in RAM; it now
  reads the rows in place like the CPU run, so the trunk the GPU takes off the
  host (about 1.9 GB) actually goes to the expert cache.
- `COLI_DENSE_BITS=4` no longer rebuilds the f32 embedding table either, which
  makes `LOSSY=1` (lm_head in int4) a RAM saving instead of a 2 GB loss.
- `turbo-qwen36.bat sweep` runs every configuration once, warm, and writes
  `sweep-<n>\summary.txt`.

Run on the target:

```cmd
call c\build-turbo.bat vk
set N_NEW=128
call benchmark-results\turbo-qwen36.bat sweep
```

## Vulkan benchmark follow-up

The controlled 128-token Vulkan run completed successfully at **6.09 tok/s**
with **2.046 s TTFT**, **91.4% expert-cache hits**, and **62.45 ms/token
expert fetch**. The dense trunk was on the RTX 3050, but the routed experts
remained on the CPU because the optional expert SPIR-V files were not present.

Vulkan with `LOSSY=1` completed at **9.76 tok/s** with **94.1% cache hits**,
**37.28 ms/token expert fetch**, and 251 skipped routed experts out of 42,880
(0.6%). The lossy Vulkan result is the fastest measured configuration so far,
but it is not numerically equivalent to the normal path: it skips low-weight
cache-miss experts and uses the int4 LM head. Keep it opt-in until a response
quality check is completed.

The committed raw logs are:

- `bench-turbo-12.log`: Vulkan, normal quality
- `bench-turbo-13.log`: Vulkan with `LOSSY=1`

Fastest measured launch:

```cmd
call c\build-turbo.bat vk
set GPU=vk
set LOSSY=1
set N_NEW=128
call benchmark-results\turbo-qwen36.bat bench
```

The next performance opportunity is compiling and loading
`qmatmul_gate_up.spv`, `rmsnorm.spv`, and `attention_absorb.spv`; without them,
the Vulkan expert tier reports `no expert batch on this device` and routed
experts stay on the host. A shader compiler such as `glslc` is not currently
available on this PC.

## Round 3: where the 164 ms of a Vulkan token went, and what changed

From `bench-turbo-12.log` (6.09 tok/s): expert fetch 62.5 ms, expert compute
32.1 ms, lm_head 22.4 ms on the CPU, about 51 ms in the chain and outside the
phases. The heat warm-up read experts at 1.25 GB/s with eight loaders.

- **lm_head on the GPU.** The chain left the 486 MiB head on the CPU because
  1 GiB of VRAM was reserved for an expert tier that cannot start here (no
  expert shaders). The launcher sets `COLI_VK_TIER_RESERVE_GB=0.5`, so the head
  fits beside the 40 layers (the lossy run, whose int4 head fit, showed the
  head at 0.1 ms).
- **`Q36_XPACK`: experts.q36x.** Every expert stored as its RAM slot holds it
  (xf16 groups with their scales), 4 KiB aligned. A miss is one unbuffered read
  straight into the slot, split into 4 reads in flight, instead of two buffered
  reads plus a CPU repack. One-time build of about 18 GB beside the model.
- **`Q36_LOOKAHEAD`: next-layer prefetch.** Layer i+1's router is applied to
  layer i's MoE input, and the predicted experts not in RAM are read while
  layer i computes and the GPU runs layer i+1. Demand reads go to the front of
  the loader queue; prefetches never evict a protected expert and are not
  counted as hits or misses. Routers stay on the host for this (20 MB).
- `GPU=vk` is the launcher default now.

Run on the target:

```cmd
git pull
call c\build-turbo.bat vk
call benchmark-results\turbo-qwen36.bat xpack
set N_NEW=128
call benchmark-results\turbo-qwen36.bat sweep
```

## Round 4: packed-store benchmark

The packed expert store was created successfully at
`D:\colibri-models\qwen36-35b\experts.q36x` (about 18.1 GB). The controlled
128-token runs below used Vulkan, eight loader threads, `Q36_LOOKAHEAD=1`, and
the packed store:

| Configuration | Speed | Expert fetch | Expert compute | Cache hits | Result |
|---|---:|---:|---:|---:|---|
| Packed Vulkan, normal quality | **12.49 tok/s** | 19.20 ms/token | 25.75 ms/token | 90.8% | Best quality-preserving result |
| Packed Vulkan, `LOSSY=1` | **12.89 tok/s** | 17.63 ms/token | 27.38 ms/token | 91.6% | Fastest; skipped 338/43,200 misses (0.8%) |

Compared with the earlier normal Vulkan result of 6.09 tok/s, packed Vulkan
is approximately **2.05x faster**. Compared with the earlier Vulkan
`LOSSY=1` result of 9.76 tok/s, packed `LOSSY=1` is approximately **32%
faster**. These are direct-engine results for the same 128-token benchmark;
short interactive responses can vary with cache warmth.

The dense chain placed all 40 layers on the RTX 3050. Routed experts still ran
on the host because `qmatmul_gate_up.spv`, `rmsnorm.spv`, and
`attention_absorb.spv` are unavailable, so the next substantial optimization
is compiling the Vulkan expert shaders.

Recommended normal-quality run:

```cmd
cd /d D:\colibri-main\colibri-main
set GPU=vk
set LOSSY=0
set N_NEW=128
call benchmark-results\turbo-qwen36.bat bench
```

Optional maximum-speed tradeoff:

```cmd
set LOSSY=1
call benchmark-results\turbo-qwen36.bat bench
```

`LOSSY=1` is not numerically equivalent: it skips a small fraction of
low-weight cache-miss experts and uses an int4 LM head. Keep it opt-in until
the generated code quality is acceptable.

## Round 5: quality-mode reliability check

Two later normal-quality runs were completed with the packed Vulkan
configuration:

| Log | Mode | Speed | Expert fetch | Expert compute | Cache hits | Exit |
|---|---|---:|---:|---:|---:|---:|
| `bench-turbo-16.log` | `LOSSY=0` | **15.03 tok/s** | 11.02 ms/token | 25.15 ms/token | 95.2% | 0 |
| `bench-turbo-20.log` | `LOSSY=0` | **11.84 tok/s** | 20.19 ms/token | 26.70 ms/token | 89.7% | 0 |

Across the three quality-preserving packed runs (12.49, 15.03, and 11.84
tok/s), the median is **12.49 tok/s** and the observed range is
**11.84-15.03 tok/s**. The variation is caused by expert-cache residency and
SSD read throughput, not a runtime failure. Use the median as the reliable
planning figure; do not treat 15 tok/s as a guaranteed floor.

The later `bench-turbo-17.log` result reached 16.97 tok/s with `LOSSY=1`;
that run is retained as a comparison only and is not part of the recommended
quality configuration.

The launcher now explicitly clears `Q36_SKIP_MISS`, `Q36_SKIP_KEEP`,
`COLI_DENSE_BITS`, and `COLI_DENSE_INT4` whenever `LOSSY` is unset, `0`, or
another non-lossy value. This prevents a parent Command Prompt with stale
lossy variables from silently changing a requested quality run.

Every tested quality run exited successfully. The only repeated warnings are
the expected missing Vulkan expert shaders; they keep routed experts on the
CPU but do not invalidate the dense Vulkan path.

The short `bench-turbo-21.log` smoke test also exited successfully after the
launcher reset change, with stale lossy variables intentionally injected into
the parent shell. No `Q36_SKIP_MISS` line appeared, confirming that
`LOSSY=0` selected the quality path.

## Round 6: Vulkan expert shaders, long prompts, API self-test

The "missing Vulkan expert shaders" warning came from eight `.spv` files that the
repo does not ship because they are gitignored build products (`qmatmul_gate_up`,
`qmatmul_bd`, `qmatmul_gate_up_bd`, `expert_act`, `expert_act_v4`, `rmsnorm`,
`attention_absorb`, `chain_dnconv_rows`). Without them the Vulkan expert tier
turned itself off, so every routed expert ran on the CPU, including the 61-token
prefill at 2.2 s. `build-turbo.bat vk` now compiles them. It uses glslc if
installed, and otherwise downloads Khronos' glslang once into
`c\third_party\glslang`. A shader that fails to compile is skipped and the
engine runs as before.

Changes:
- Launcher `GPU=vk`: expert tier on, tier host pool off (`COLI_VK_TIER_HOST=0`),
  so the RAM cache keeps its size.
- Cross-layer lookahead now also runs on the Vulkan expert path. It skips
  experts the device already holds.
- `turbo-qwen36.bat long` runs a coding-assistant style turn (about 2k tokens of
  C source plus an instruction). The number to watch is time to first token.
- `turbo-qwen36.bat selftest` starts the OpenAI-compatible server, sends three
  chat turns to `/v1/chat/completions`, prints tokens/s per turn, then stops the
  server. Exit 0 means every turn was answered.
- `sweep` adds the tier, no-tier, host-pool, and long-prompt configurations.

The completed `sweep-1` results show that the short quality benchmark remains
fast and reliable with the current host-routed expert path:

| Configuration | Speed | TTFT | Cache hits | Exit |
|---|---:|---:|---:|---:|
| `vk-tier` | 15.12 tok/s | 0.975 s | 93.0% | 0 |
| `vk-notier` | 15.32 tok/s | 0.949 s | 93.4% | 0 |
| `vk-tier-la2` | 15.30 tok/s | 0.941 s | 93.7% | 0 |
| `vk-tier-hostpool` | 15.56 tok/s | 0.876 s | 93.8% | 0 |
| `vk-tier-ram92` | **16.09 tok/s** | 0.822 s | 94.7% | 0 |
| `vk-tier-omp12` | 14.45 tok/s | 0.884 s | 94.1% | 0 |

The recommended quality configuration for this machine is therefore
`Q36_RAM_FRAC=0.92`, Vulkan, packed experts, and normal precision. It measured
16.09 tok/s in this sweep, but the prior quality median remains 12.49 tok/s;
SSD and cache warmth make the lower median the safer planning number.

The long coding-style prompt contained 5,338 input tokens. It measured
0.57-0.62 tok/s for 48 generated tokens and 73.7-81.6 seconds TTFT. This is
expected prompt-prefill cost, not a decode crash; short prompts must not be
used to predict long-context responsiveness.

All sweep configurations exited with code 0. The logs still report missing
expert shaders, so this run did **not** validate Vulkan routed-expert
execution. It validated the existing dense Vulkan chain and host expert path.
The next build should confirm whether `build-turbo.bat vk` actually creates
the `.spv` files; the runtime must log an expert batch on the device before
claiming a shader speedup.

Reproduce the tracked validation:

```bat
git pull
call c\build-turbo.bat vk
set N_NEW=128
call benchmark-results\turbo-qwen36.bat sweep
call benchmark-results\turbo-qwen36.bat selftest
```

If decode speed drops with the tier on, `set COLI_VK_TIER=0` restores the
round 5 configuration.

## Round 7: long-prompt prefill (to run)

The 80 s TTFT for 5,338 tokens came from SSD reads, not compute. The long run had
63,531 cache misses, about 112 GB read for one prompt. The prompt's routed experts
ran in runs of rows that fit the free cache (about 140 experts a layer), while a
1280-row chunk routes to nearly all 256. Each run evicted experts the next run read
again: about 1,600 misses a layer against about 110 needed.

`Q36_PF_EXPERT` (default on) now processes each chunk expert by expert. Every
distinct expert is read at most once a chunk, in groups of half the free cache,
and the next group's misses load while the current group computes. Each
(token, expert) output and the rank-order sum are the same bits as before.
`Q36_PF_EXPERT=0` restores the old path for an A/B. The Vulkan tier's CPU share
of a prompt block uses the same routine; before, it went one row at a time.
Expected: SSD traffic for the 5.3k prompt drops from about 112 GB to about 8-20 GB.

Other changes:
- `Q36_RAM_FRAC=0.92` is the launcher default (sweep-1's best, 16.09 tok/s).
- The glslang download forces TLS 1.2. Windows PowerShell 5.1 can default to
  an older protocol that GitHub refuses.
- The launcher warns when the expert shaders are missing.
- The sweep adds `long-vk-pf0` (old path), `long-vk-rows2560` (bigger chain
  chunks, fewer re-reads), and `vk-tier-ram88`. It drops la2 and omp12, which
  showed nothing.

Save the build output too; it shows whether the shaders compiled:

```bat
git pull
call c\build-turbo.bat vk > benchmark-results\build-vk.log 2>&1
dir c\shaders\qmatmul_gate_up.spv
set N_NEW=128
call benchmark-results\turbo-qwen36.bat sweep
call benchmark-results\turbo-qwen36.bat selftest
```

## Round 8: sweep-2 regression and fix

The shaders compiled, so the Vulkan expert tier turned on for the first time,
and everything with it got slower:

| Run | Speed | TTFT | Exit |
|---|---:|---:|---:|
| `vk-tier` (32-token prompt) | 1.70 tok/s | 15.2 s | 0 |
| `vk-notier` | 13.56 tok/s | 1.0 s | 0 |
| `long-vk-tier` (5,338 tokens) | 0.08 tok/s | 537 s | 0 |
| `long-vk-notier` (round 7 prefill) | 1.12 tok/s | **39.2 s** (was 81.6) | 0 |
| `vk-tier-ram88`, `long-vk-rows2560` | - | - | 1 |

Cause: with the xf16 expert layout, every expert the CPU computed while the tier
was on received a second, unpacked 1.5 MB copy that stayed on the cache slot for
the rest of the run. Over time that grows to about 10 GB on a 16 GB machine. The
result was paging (the 44.8 s CPU share for 128 tokens) and probably the two
exit-1 runs. The prefill fix itself worked: misses fell from 63.5k to 17.9k and
TTFT halved.

Fix: the unpacked copy is made in one buffer a thread, only for an expert the tier
will take (`vkt_wants`), and is never attached to the cache slot.

Launcher defaults now match the best measured configuration: `COLI_VK_TIER=0`
(dense chain and lm_head on the GPU, routed experts on CPU/RAM/SSD), and the
engine's default RAM fraction (0.92 was not reliably better: 16.09 then 13.56).
The sweep tries the tier again (`COLI_VK_TIER=1`, with and without prompt
streaming). A failed run now keeps its last 12 log lines in the summary.
