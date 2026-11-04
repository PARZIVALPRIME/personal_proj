# CachyOS: next performance experiments on the 5800HS / RTX 3050 laptop

## Status and boundaries

The working target is CachyOS/XFCE, Ryzen 7 5800HS (8 cores/16 threads), RTX 3050
Laptop 4 GB, 16 GB RAM and NVMe. The model is Qwen3.6-35B-A3B int4 gs64 at
`~/colibri-models/qwen36-35b`.

The last reported post-xpack runs (commit `45d8fae`) measured:

| Metric | CachyOS |
|---|---:|
| Short benchmark reported speed | 16.42-16.74 tok/s |
| Short prompt TTFT | 1.256-1.291 s |
| 5,338-token prompt TTFT | 17.363-18.522 s |
| Short expert compute / fetch | about 21 / 8 ms per timed token |
| Short outside-phase time | about 33 ms per timed token |
| Long-run peak RSS | 14.17 GB |

`experts.q36x` is now active. Do **not** rebuild it routinely or diagnose it as
missing without checking. The dense chain uses Vulkan; decode experts use the CPU;
larger prompt batches use the GPU expert tier. Keep that placement.

These updates were developed off-target on Windows. The launcher and runner have
regression tests, including actual subprocess failure/timeout tests and Bash
launcher tests. **There is no new measured laptop speedup yet, and no claim of
2x performance.** The engine kernels, synchronization, cache capacity and model
semantics have not been changed blindly.

## What changed

- The launcher exports the correct default `SNAP` to direct engine invocations.
- Failures are not swallowed by a `grep | ... || true` pipeline.
- Invalid GPU modes are errors. CPU-only mode explicitly disables the GPU tier.
- Quality mode clears inherited expert-skipping and alternate dense-precision
  settings; a nonzero `LOSSY` is rejected.
- Physical-core sizing uses the engine's existing topology-aware helper unless
  `OMP_NUM_THREADS` is explicitly supplied. It no longer guesses `nproc/2`.
- The Python standard-library runner records complete output, engine/prompt
  hashes, selected environment, exit status, wall time and parsed engine timers.
- It samples Linux system memory, swap counters, memory pressure, CPU MHz and
  process I/O/status once a second while the engine runs. Optional `nvidia-smi`
  sampling records GPU clocks, utilization, temperature and power.
- Every run has a timeout. Sweeps stop on a failed engine or missing required
  TTFT/Speed metrics, and preserve partial results.
- Summaries report median/min/max/count, exclude warm-ups and failures, and do
  not automatically install a winning profile.
- Results live in ignored `benchmark-results/linux-*` directories. They are
  never staged automatically.

## Exact commands when the laptop is available

Use Bash, AC power and a consistent desktop workload. Close heavy applications,
but do not disable swap, delete heat history, or change global memory settings.

### Complete start-to-finish sequence

Follow A through F in order. Run blocks in a dedicated Bash terminal. Blocks B
through D use `set -e` so an error stops the block rather than continuing with
invalid results. Keep that terminal open: it holds the `SESSION` variable.
The detailed sections below explain the same operations and optional experiments.

#### A. Receive the changes

On the development computer, the earlier push received HTTP 403. Until that is
resolved, pulling on the laptop will **not** obtain the new tooling.

On the development computer, from its repository, retry:

```powershell
git status -sb
git push origin main
```

Only if that push succeeds, use this on the laptop:

```bash
cd ~/personal_proj
git status --short
git pull --ff-only
test -f c/tools/q36_bench.py
test -f docs/CACHYOS-PERFORMANCE.md
```

If Git reports local conflicts, stop and preserve the local edits; do not reset
or force the pull. If those files are absent, do not proceed with the old launcher.

**Alternative when development-computer pushing remains blocked:** export all
local task commits using **Git Bash** on the development computer. Avoid
PowerShell 5.1 redirection, which can change patch encoding:

```bash
git format-patch --stdout origin/main..HEAD > "$HOME/Downloads/cachyos-performance-series.patch"
```

Transfer that file to the laptop's Downloads directory. From a clean laptop
checkout, apply it **instead of receiving those same commits through a pull**:

```bash
cd ~/personal_proj
git status --short
# Continue only if there are no local edits:
git am "$HOME/Downloads/cachyos-performance-series.patch"
```

If application fails, stop and inspect `git status`; do not rerun it repeatedly.
`git am --abort` cancels this patch application. Do not apply a series that has
already arrived through Git. A later push from the laptop can publish the applied
commits along with the results.

#### B. Validate and preserve the known-good engine

```bash
(
set -e
cd ~/personal_proj
export SNAP="$HOME/colibri-models/qwen36-35b"
test -d "$SNAP"
test -s "$SNAP/experts.q36x"
test -x c/qwen36
python3 -m unittest discover -s c/tests -p test_q36_bench.py -v
bash -n benchmark-results/turbo-qwen36.sh
mkdir -p benchmark-results/linux-known-good
if [ ! -e benchmark-results/linux-known-good/qwen36 ]; then
    cp -p c/qwen36 benchmark-results/linux-known-good/qwen36
    sha256sum c/qwen36 > benchmark-results/linux-known-good/engine.sha256
fi
)
```

If the engine is missing, there is nothing to preserve: build using the fresh-build
section below, then rerun B. If the packed store is missing, verify `SNAP` first;
only then run `SNAP="$HOME/colibri-models/qwen36-35b" bash
benchmark-results/turbo-qwen36.sh xpack` as one command and rerun B.

#### C. Run diagnostics, baseline, sweep and longer outputs

Create a new session directory outside your model directory:

```bash
cd ~/personal_proj
export SNAP="$HOME/colibri-models/qwen36-35b"
SESSION="benchmark-results/linux-session-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
printf 'Session directory: %s\n' "$SESSION"
```

Then run this block. It uses the current engine; do not rebuild before collecting
this baseline:

```bash
(
set -e
bash benchmark-results/turbo-qwen36.sh doctor --output "$SESSION/doctor"
bash benchmark-results/turbo-qwen36.sh bench --repeats 3 --output "$SESSION/short"
bash benchmark-results/turbo-qwen36.sh long --repeats 3 --output "$SESSION/long"
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline threads6 io4 passive passive-io4 --repeats 3 \
  --output "$SESSION/sweep"
bash benchmark-results/turbo-qwen36.sh bench --tokens 512 --repeats 3 \
  --output "$SESSION/sustained-short"
bash benchmark-results/turbo-qwen36.sh long --tokens 256 --repeats 3 \
  --output "$SESSION/sustained-long"
)
```

If this stops, inspect the last printed result directory and its logs. Do not
describe partial results as a completed sweep. Do not rerun into existing output
directories; create a new `SESSION` or choose a new output directory.

#### D. Optional fresh build and comparison

Only after C completes and the backup exists:

```bash
(
set -e
make -B -C c qwen36 VK=1 ARCH=native -j4 > "$SESSION/build.log" 2>&1
bash benchmark-results/turbo-qwen36.sh bench --repeats 3 \
  --output "$SESSION/rebuilt-short"
bash benchmark-results/turbo-qwen36.sh long --repeats 3 \
  --output "$SESSION/rebuilt-long"
)
```

If the build fails, read `"$SESSION/build.log"` and use the preserved engine via
`COLI_ENGINE` as described under rollback. Do not stage rebuilt objects/shaders.

#### E. Review what will be shared

```bash
printf 'Reviewing: %s\n' "$SESSION"
find "$SESSION" -name summary.csv -print
cat "$SESSION/sweep/summary.csv"
find "$SESSION" -name '*.log' -print
git status --short
```

Open the full engine logs, diagnostics and recorded environment before sharing.
Remove private experiment data from the sharing selection if needed. Do not assume
logs are safe merely because they contain benchmark output. Check failures, swap
warnings, temperature/clocks, truncated answers, and all workload regressions.
Do not install a profile solely because one reported Speed value is higher.

#### F. Commit only the reviewed results and push

First inspect existing staged files:

```bash
git diff --cached --name-only
```

If this lists unrelated staged changes, stop and separate them before continuing.
Then stage just this reviewed session:

```bash
git add -f -- "$SESSION"
git diff --cached --stat
git diff --cached --name-only
```

Only after checking that list:

```bash
git commit -m "CachyOS controlled performance experiment results"
git push origin main
```

If push fails with HTTP 403, your results remain committed locally; repository
write access/authentication must be repaired. Do not share credentials in logs or
chat. Retry `git push origin main` after access is restored.

If push is rejected because the remote advanced, do not force-push. Check the
worktree first. Build experiments may have changed tracked generated files.
Preserve those local changes before any rebase. Once the worktree is clean:

```bash
git fetch origin
git rebase origin/main
git push origin main
```

Stop on conflicts rather than choosing an automatic destructive resolution.
After a successful push, tell the reviewing assistant the session directory and
commit ID from `git log -1 --oneline`.

### 1. Pull, test, preserve the current engine

```bash
cd ~/personal_proj
git pull --ff-only
export SNAP="$HOME/colibri-models/qwen36-35b"
test -d "$SNAP" || { echo "Wrong model path"; exit 1; }
test -s "$SNAP/experts.q36x" || { echo "Packed expert store missing"; exit 1; }

python3 -m unittest discover -s c/tests -p test_q36_bench.py -v
bash -n benchmark-results/turbo-qwen36.sh

mkdir -p benchmark-results/linux-known-good
if [ -x c/qwen36 ] && [ ! -e benchmark-results/linux-known-good/qwen36 ]; then
    cp -p c/qwen36 benchmark-results/linux-known-good/qwen36
    sha256sum c/qwen36 > benchmark-results/linux-known-good/engine.sha256
fi
```

Stop if a test fails. The `exit` commands above are for a dedicated Bash terminal.
No new dependency installation is needed for the runner/tests.

### 2. Capture diagnostics and the current binary baseline FIRST

```bash
bash benchmark-results/turbo-qwen36.sh doctor
bash benchmark-results/turbo-qwen36.sh bench --repeats 3
bash benchmark-results/turbo-qwen36.sh long --repeats 3
```

Each command prints its result directory. `doctor` does not require the engine or
model to exist. Missing optional tools appear as explicit errors in `system.json`;
they do not prevent benchmarking. Do not install/replace GPU drivers just because
an optional probe is unavailable.

`bench` and `long` now default to one unmeasured warm-up plus **three measured
runs**, not one invocation. Default output limits remain 128 and 48 respectively,
unless `N_NEW` or `--tokens` overrides them. Each engine process has a 1,800-second
timeout by default; `--timeout` overrides it.

The packed-store check verifies presence/nonzero size only. It is not a format or
integrity validator; inspect the engine's `Q36_XPACK` log line.

### 3. Run the bounded tuning sweep

```bash
bash benchmark-results/turbo-qwen36.sh sweep --repeats 3
```

This runs five configurations on both reference workloads, with one warm-up per
configuration/workload and three measured repetitions: **40 engine invocations**.
Allow time for initialization and warm loading as well as inference.

| Profile | Difference from your invoking environment |
|---|---|
| `baseline` | None |
| `threads6` | `OMP_NUM_THREADS=6` |
| `io4` | `Q36_IO_THREADS=4` |
| `passive` | passive OMP waits, `GOMP_SPINCOUNT=0`, `KMP_BLOCKTIME=0` |
| `passive-io4` | passive settings plus four loader threads |
| `bind` (optional) | `OMP_PLACES=cores`, `OMP_PROC_BIND=close` |
| `ram88` (optional) | `Q36_RAM_FRAC=0.88`, testing more system headroom |

Other caller-supplied settings stay fixed. Baseline reflects those settings;
the runner does not silently clear your chosen OMP policy. Start comparisons
from the same shell and inspect recorded environments.

For a shorter first comparison:

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline passive io4 --workloads bench --repeats 3
```

Optional second pass, only after the first succeeds:

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline bind ram88 --repeats 3
```

The engine's existing Q36 hot-team helper checks CUDA/Metal when deciding whether
to inject active waits, but does not check Vulkan in that call. Explicit passive
wait variables are respected by its non-overwriting environment setup. This makes
passive waiting worth **testing**, not an established faster default. Do not change
the engine's global wait policy without target measurements.

### 4. Compare a fresh build only after preserving the current baseline

If investigating stale objects or compiler flags, use a forced rebuild rather than
assuming tracked objects are suitable for Linux:

```bash
make -B -C c qwen36 VK=1 ARCH=native -j4
bash benchmark-results/turbo-qwen36.sh bench --repeats 3
bash benchmark-results/turbo-qwen36.sh long --repeats 3
```

Stop if the build fails. `-B` may rebuild tracked shaders/objects as well; do not
stage those changes. The moderate job count avoids an unnecessarily large build
memory spike. This command uses existing build prerequisites; it installs nothing.

### 5. Check sustained and useful-answer performance

```bash
bash benchmark-results/turbo-qwen36.sh bench --tokens 512 --repeats 3
bash benchmark-results/turbo-qwen36.sh long --tokens 256 --repeats 3
```

These raise limits, but cannot guarantee the original prompts produce useful
completed answers. To test your own unambiguous coding prompt:

```bash
PROMPT="$HOME/coding-benchmark.txt" \
  bash benchmark-results/turbo-qwen36.sh bench --tokens 2048 --repeats 3
```

Create that prompt first; choose a task with executable checks and inspect the
full generated answer. Custom prompts and generated output may be private.

The GPU monitor is optional. Compare a repeat with `--no-gpu-monitor` to check
monitoring overhead. Host memory/process sampling remains enabled.

### 6. Save and push ONLY the selected experiment directory

First review the logs. They contain generated text, selected runtime variables,
local paths, hardware details, and possibly private prompt material.

For a named comparison you intend to share, choose a **new** directory:

```bash
OUT="benchmark-results/linux-share-$(date +%Y%m%d-%H%M%S)"
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline passive io4 --repeats 3 --output "$OUT"
# Only continue after the runner succeeds and you have reviewed "$OUT".
git add -f -- "$OUT"
git diff --cached --stat
git commit -m "CachyOS controlled baseline and tuning results"
git push origin main
```

Do not use `git add -A`: previous commits included generated binaries/objects and
web build output. Leave those unrelated artifacts alone.

## Reading the results

- `system.json`: initial diagnostics and exact probe errors/status.
- `*.log`: unfiltered engine output.
- `*.json`: per-run command, hashes, environment, timings, status and memory counters.
- `*.system.jsonl`: time series of Linux memory/pressure/process counters.
- `*.gpu.csv` / `*.gpu.stderr`: GPU sampling, if available.
- `runs.json`: all completed run records, including warm-ups/failures.
- `summary.csv` / `summary.json`: measured successful-run statistics.

`reported_speed_tps` is the engine's original Speed metric. It includes prompt
processing in these runs; especially for the long workload, **do not call it
decode-only throughput**. `step_ms` is the engine's own timed step metric; do not
convert it to accepted-token throughput without checking forward accounting.
`wall_s` includes engine startup/warm loading and monitoring lifecycle, but excludes
the initial diagnostic probes and binary hashing.

Swap counters are **system-wide page counts**, not engine-only bytes. Memory
availability is sampled, not an exact minimum between samples. GPU diagnostics
may include other running GPU processes. Sampling is not a per-layer timeline.

History evolves across runs. Profiles are interleaved with alternating order to
reduce drift, not eliminate it. The runner does not wipe or restore `.q36_heat`,
`.coli_usage`, the OS page cache or model state. This is a warmed-history comparison,
not a scientifically controlled cold-cache benchmark.

Keep a profile only after its improvement survives repeated end-to-end testing,
does not cause swap pressure, and does not materially worsen long prompts or the
desktop. No profile is automatically applied.

For example, **only if measured better**, use passive waiting for a session:

```bash
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0 \
  bash benchmark-results/turbo-qwen36.sh chat
```

## Rollback and next engineering decisions

No tuning profile modifies global settings or writes a persistent configuration.
Start a fresh shell and use the launcher normally to return to defaults.

To compare/use the preserved engine:

```bash
COLI_ENGINE="$PWD/benchmark-results/linux-known-good/qwen36" \
  bash benchmark-results/turbo-qwen36.sh bench --repeats 3
```

Further changes require laptop evidence:

1. Split the approximately 33 ms outside bucket into GPU execution, submission,
   CPU waits, transfers, routing and sampling. Use Vulkan timestamps without
   adding a wait per dispatch; verify instrumentation overhead.
2. Compare expert computation to measured RAM bandwidth and profile spills,
   unpacking and OpenMP barriers before changing the kernel.
3. Audit decode-only late/duplicate/evicted prefetches before changing admission.
4. Audit existing prefix support before adding snapshots. Q36 already calls
   `kv_prefix_reuse`, `pin_save` and DeltaNet pin-state helpers. Their existence
   does not establish complete cross-request or restart-persistent reuse.
5. Any new prefix reuse must preserve attention KV, DeltaNet recurrent/convolution
   state and exact token boundaries. Chat-template changes can invalidate matches.

Do not blindly add spin waits, replace synchronization, quantize activations,
skip experts, port backends, or allocate a draft model. These can reduce quality,
break correctness, or consume the RAM needed for the working expert cache.
