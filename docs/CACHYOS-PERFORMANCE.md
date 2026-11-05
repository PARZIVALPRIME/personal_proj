# CachyOS XFCE4: run the tests and push the results

## Complete start-to-finish sequence

Use your Ryzen 7 5800HS / RTX 3050 4 GB / 16 GB laptop.
Plug in AC power and close heavy applications.

Open **Terminal Emulator** in XFCE4, type `bash`, and press Enter.
Run each numbered block in order in that same terminal. Keep it open.
**If any command fails, stop and save the error before continuing.**
Do not delete model files or expert history, disable swap, or change GPU drivers.

## Where we are after the November 5 results

Results commit: `64f0e5e`, directory
`benchmark-results/linux-session-20261105-200755`.

| Measured sweep profile | Short Speed median | Short step median | Long TTFT median |
|---|---:|---:|---:|
| Baseline | 15.74 tok/s | 65.5 ms | 16.207 s |
| Six compute threads | 17.15 tok/s | 59.7 ms | 16.265 s |
| Four I/O threads | 16.52 tok/s | 62.2 ms | 16.164 s |
| Passive waits | 16.20 tok/s | 64.4 ms | 16.519 s |

Six threads are the best measured short-run candidate, not proof of a universal
best configuration. Do not combine six compute/four loader threads and claim
their separate gains add up. Passive waiting is not the next default.

**The immediate issue is memory pressure.** The 256-token long runs reached only
83,240 KiB system MemAvailable in one run and up to 832,430 system swap-out pages
in another. These counters are system-wide; they establish pressure during the
workload, not ownership of every swapped page. Long decode step medians rose to
102.5 ms. The current 0.92 RAM fraction leaves inadequate headroom in these runs.
Do not disable swap; test a smaller expert cache instead.

The folder named `long` actually contains `bench` runs with 128 output tokens.
Use `sweep/long-*`, `sustained-long/long-*` and `rebuilt-long/long-*` for long-prompt
analysis. The runner now records explicit prompt path, requested output length,
and prompt-token count when emitted by the engine.

The sustained short benchmark reached 19.91 reported tok/s, but uses 512 output
tokens instead of 128. It is not a directly comparable improvement over the short
reference: prompt amortization and cache access differ.

### Next commands for this already-tested laptop

Use this focused sequence instead of repeating the full old sweep below. Run
each command in the same Bash terminal and stop if it fails:

```bash
cd ~/personal_proj
git pull --ff-only
export SNAP="$HOME/colibri-models/qwen36-35b"
unset RAM_GB PROMPT N_NEW OMP_NUM_THREADS OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
unset OMP_PLACES OMP_PROC_BIND
python3 -m unittest discover -s c/tests -p test_q36_bench.py -v
SESSION="benchmark-results/linux-memory-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
```

Then compare the known six-thread candidate with a lower RAM fraction, using
the same 256-output-token limit for each workload/profile:

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles threads6 ram80 threads6-ram80 --tokens 256 --repeats 3 \
  --output "$SESSION/comparison"
cat "$SESSION/comparison/summary.csv"
```

This makes 24 engine invocations including warm-ups. `ram80` changes only
`Q36_RAM_FRAC=0.80`; `threads6-ram80` additionally selects six compute threads.
Neither changes quantization, routing, context size, global memory settings or
permanent defaults. The reduced cache may increase fetches: that is an explicit
trade-off to measure against swap avoidance and desktop usability.

Inspect `memory_warning_runs` and the swap/minimum-memory columns, not just speed.
The runner warns on system swap activity and available memory below 512 MiB.
Warnings do not exclude runs or stop the default comparison. For a deliberately
strict run, add `--stop-on-pressure`; it saves results and exits 2 after the first
affected run, including warm-up. Background-process swap can also trigger it.

If even the reduced-cache runs swap heavily, stop adding optimizations. Close
unnecessary applications and retest a smaller RAM fraction explicitly, rather
than growing buffers or disabling swap. `RAM_GB` must remain unset because the
engine gives it precedence over the fraction.

For temporary chat with additional desktop headroom, this is a conservative
**unmeasured combination**, not a proven faster configuration:

```bash
OMP_NUM_THREADS=6 Q36_RAM_FRAC=0.80 \
  bash benchmark-results/turbo-qwen36.sh chat
```

After reviewing logs for private data, share only the new comparison:

```bash
git diff --cached --name-only
# Continue only if there are no unrelated staged files.
git add -f -- "$SESSION"
git diff --cached --stat
git commit -m "Compare Q36 memory headroom and six-thread configuration"
git push origin main
```

Do not rebuild again just for this comparison: the uploaded rebuild succeeded,
but did not remove memory pressure. Once the lower-cache trade-off is measured,
the next engineering task remains GPU/CPU timing instrumentation, followed by
an audit of existing prefix/state reuse. Neither is a substitute for adequate RAM.

## Original full workflow

### 1. Pull the code

```bash
cd ~/personal_proj
git status --short
git pull --ff-only
```

If Git reports conflicts or local changes preventing the pull, stop; do not force
it or discard your work.

### 2. Set the model path and check the required files

```bash
export SNAP="$HOME/colibri-models/qwen36-35b"
test -d "$SNAP" && echo "Model directory OK"
test -s "$SNAP/experts.q36x" && echo "Packed expert store OK"
test -x c/qwen36 && echo "Current engine OK"
test -f c/tools/q36_bench.py && echo "New benchmark runner OK"
```

All four checks must print `OK`. If the runner is missing, the new changes have
not reached your checkout yet. Stop rather than running the old instructions.
The packed expert store already worked in your last tests; do not rebuild it.

### 3. Run the regression tests

```bash
python3 -m unittest discover -s c/tests -p test_q36_bench.py -v
bash -n benchmark-results/turbo-qwen36.sh
```

The Python tests must end with `OK`. The Bash check should print no errors.

### 4. Back up the current working engine

```bash
mkdir -p benchmark-results/linux-known-good
if [ ! -e benchmark-results/linux-known-good/qwen36 ]; then
    cp -p c/qwen36 benchmark-results/linux-known-good/qwen36
    sha256sum c/qwen36 > benchmark-results/linux-known-good/engine.sha256
fi
```

### 5. Create this test session's results folder

```bash
SESSION="benchmark-results/linux-session-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
printf 'Keep this folder name: %s\n' "$SESSION"
```

Keep using the same terminal so `$SESSION` remains available.

### 6. Record system information

```bash
bash benchmark-results/turbo-qwen36.sh doctor --output "$SESSION/doctor"
```

Missing optional diagnostic tools are recorded in the output. Do not install or
replace drivers just because a diagnostic command is unavailable.

### 7. Measure the current engine before rebuilding anything

```bash
bash benchmark-results/turbo-qwen36.sh bench --repeats 3 --output "$SESSION/short"
bash benchmark-results/turbo-qwen36.sh long --repeats 3 --output "$SESSION/long"
```

Each command performs one warm-up and three measured runs.

### 8. Compare thread and waiting settings

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline threads6 io4 passive passive-io4 \
  --repeats 3 --output "$SESSION/sweep"
```

Wait for completion. This performs 40 engine runs across the short and long
prompts, including warm-ups. It changes no permanent settings.

### 9. Test longer generated answers

```bash
bash benchmark-results/turbo-qwen36.sh bench --tokens 512 --repeats 3 \
  --output "$SESSION/sustained-short"
bash benchmark-results/turbo-qwen36.sh long --tokens 256 --repeats 3 \
  --output "$SESSION/sustained-long"
```

Inspect the logs to see whether answers actually finish. A higher token limit
alone does not prove answer correctness.

### 10. Rebuild and compare with the current engine

Do this only after steps 4 and 7 succeed.

```bash
make -B -C c qwen36 VK=1 ARCH=native -j4 > "$SESSION/build.log" 2>&1
```

If the build fails, stop and read:

```bash
cat "$SESSION/build.log"
```

Only if the build succeeds, run:

```bash
bash benchmark-results/turbo-qwen36.sh bench --repeats 3 \
  --output "$SESSION/rebuilt-short"
bash benchmark-results/turbo-qwen36.sh long --repeats 3 \
  --output "$SESSION/rebuilt-long"
```

### 11. Read the results before sharing

```bash
printf 'Results folder: %s\n' "$SESSION"
find "$SESSION" -name summary.csv -print
cat "$SESSION/sweep/summary.csv"
find "$SESSION" -name '*.log' -print
```

Review the full logs and diagnostics for private content before staging them.
Check failures, swap warnings and incomplete answers.

- `ttft_s`: time to first token.
- `step_ms`: engine step timing; lower is better for equivalent work.
- `reported_speed_tps`: engine Speed metric; it includes prompt processing here,
  so the long run's value is **not decode-only throughput**.
- Summary values exclude warm-ups and failed runs.
- History evolves during tests; these are warmed-history comparisons.

Previous results were 16.42-16.74 tok/s short speed and 17.36-18.52 seconds long
TTFT. No new speedup is guaranteed. Do not permanently apply a profile yet.

### 12. Stage only this session's reviewed results

```bash
git diff --cached --name-only
```

If this lists unrelated staged files, stop and separate them first.
Otherwise:

```bash
git add -f -- "$SESSION"
git diff --cached --stat
git diff --cached --name-only
```

Only this session's results should be listed. **Do not use `git add -A`.**
Rebuilding may change tracked binaries, objects and shaders; do not stage those.

### 13. Commit and push

```bash
git commit -m "CachyOS XFCE controlled performance results"
git push origin main
```

If GitHub reports HTTP 403, stop: the commit is saved locally, but write access
needs fixing. If it reports that the remote advanced, stop and preserve local
build changes before attempting to reconcile branches. Never force-push.

### 14. Report the commit and session folder

```bash
git log -1 --oneline
printf 'Results folder: %s\n' "$SESSION"
```

Send those two lines to the reviewing assistant after the push succeeds.

## If the new build is worse

Use the preserved engine without overwriting anything:

```bash
COLI_ENGINE="$PWD/benchmark-results/linux-known-good/qwen36" \
  bash benchmark-results/turbo-qwen36.sh chat
```

For normal chat using the current engine:

```bash
bash benchmark-results/turbo-qwen36.sh chat
```

Do not rerun tests into an existing output folder. For a fresh attempt, repeat
step 5 to create a new session.
