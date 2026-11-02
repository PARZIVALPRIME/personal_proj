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
