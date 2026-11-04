# CachyOS XFCE4: run the tests and push the results

## Complete start-to-finish sequence

Use your Ryzen 7 5800HS / RTX 3050 4 GB / 16 GB laptop.
Plug in AC power and close heavy applications.

Open **Terminal Emulator** in XFCE4, type `bash`, and press Enter.
Run each numbered block in order in that same terminal. Keep it open.
**If any command fails, stop and save the error before continuing.**
Do not delete model files or expert history, disable swap, or change GPU drivers.

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
