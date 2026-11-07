# CachyOS XFCE4: Qwen3.6 optimization, from pull to results upload

Machine: Ryzen 7 5800HS, RTX 3050 Laptop 4 GB, 16 GB RAM, NVMe.
Model: `~/colibri-models/qwen36-35b`, int4 gs64, packed `experts.q36x`.

## Current evidence and what changed

Round 4 upload `8709f68` measured about **20 tok/s short decode**, **16.8 tok/s
after a 5,338-token prompt**, and **12-13 tok/s in the coding conversation**.
Short TTFT was 1.45 s; long TTFT 16.74 s. Same-conversation coding follow-ups
started in 1.1-1.3 s with prefix reuse. These are different workloads, not
interchangeable speed claims.

Keep the reference: six compute threads, eight loaders, RAM fraction 0.80,
Vulkan dense layers, CPU decode experts, GPU prompt experts, lookup on, MTP off.
0.92 caused severe long-run memory pressure. Lookahead/GEMV/lookup sweeps did not
find a meaningful winner. MTP ran on the GPU but was substantially slower.
Do not repeat MTP setup, rebuild the packed store, or delete your model.

This implementation adds:

- Event-driven cache publication waiting instead of millisecond polling when every
  slot is in flight. It preserves the eviction rules and checks spurious wakeups.
- Exact monotonic generation timing and a generated-token fingerprint for direct
  runs, plus per-layer decode cache counters and decode-only Vulkan wait/transfer
  counters. Legacy logs still parse, but cannot automatically establish a winner.
- Decode-only Vulkan timestamp profiling, with prefill excluded.
- Private, identical heat-history copies for controlled runs. Your normal heat
  history is not overwritten by these experiments.
- Opt-in shorter GPU fence-polling profiles. The existing discrete-GPU default
  spins up to 50 ms; this laptop shares a power/cooling budget between CPU and GPU.
  Less polling could help or hurt. We will measure it, not assume it wins.
- Repeated, alternating persistent-server conversations with GPU/system telemetry.
  Broken streams fail explicitly; completed earlier turns survive later failures;
  token-limit answers are marked incomplete and return a nonzero status.
  Summaries/settings are replaced atomically, and captured API-key environment
  values are redacted. Still inspect prompts/logs yourself before uploading.
- Opt-in exact thinking-whitespace preservation. The agent requests it when
  `--think` is used. Other clients can send
  `"preserve_thinking_whitespace": true` along with resent `reasoning_content`.
  Default official-template formatting remains unchanged.
  Keepalive choices carry `x_colibri_keepalive: true`, so the agent retains real
  generated periods instead of mistaking punctuation for a ping.
- Strict winner selection: complete short and long runs, exact timing, matching
  generated-token fingerprints, known RAM headroom, at least 5% primary gain,
  no failures, and at most 5% TTFT/long-decode regression. Saving replaces stale
  settings with the effective measured settings; it does not merge old experiments.

**No new laptop speedup is established yet. 30 tok/s remains a target.**
These changes do not reduce expert count, precision or model size. Shader fusion,
unequal per-layer cache allocation, a new I/O API and renewed MTP work are gated on
the new measurements; they are not enabled speculatively.

## Run these numbered steps in order

Open XFCE4 Terminal Emulator and run `bash`. Use the same terminal for all steps.
Connect AC power, close heavy applications, and stop any previous model server
with Ctrl+C in its terminal. Do not disable swap or change drivers/governors.
If a command fails, stop and retain its output. Do not force Git or delete files.

### 1. Pull the implementation

```bash
cd ~/personal_proj
git status --short
git pull --ff-only
```

If the status lists your own uncommitted work, resolve that before pulling.
Do not discard it. The development checkout publishes code first; this command
pulls that code onto your CachyOS laptop.

### 2. Check the existing model and prepare the result directory

```bash
M="$HOME/colibri-models/qwen36-35b"
test -d "$M" || { echo "STOP: model directory missing"; exit 1; }
test -s "$M/experts.q36x" || { echo "STOP: packed experts missing"; exit 1; }
test -s "$M/.q36_heat" || { echo "STOP: existing heat history missing"; exit 1; }
SESSION="benchmark-results/linux-round5-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
printf 'Results folder: %s\n' "$SESSION"
git rev-parse HEAD > "$SESSION/revision.txt"
```

The previous successful runs should have created the heat history. If it is
missing, run one ordinary reference benchmark first, then return to this step:

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh bench --tokens 256
```

### 3. Rebuild and run regression tests

```bash
make -C c qwen36 VK=1 -j2
make -C c tests/test_qwen36_cache_index VK=1 -j2
timeout 30s c/tests/test_qwen36_cache_index
python3 -m unittest discover -s c/tests -p 'test_q36_*.py'
python3 -m unittest discover -s c/tests -p 'test_chat_history_reuse.py'
bash -n benchmark-results/turbo-qwen36.sh
```

The C test must end with `qwen36 cache index: ok`; Python tests must report `OK`.
A skip is not hardware validation. Two build jobs limit peak memory. Installing
packages is unnecessary if your existing build environment still works.
Do not reuse an old binary: new exact metrics require the rebuilt engine.

### 4. Capture a clean reference system snapshot

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh doctor \
  --output "$SESSION/doctor"
```

`Q36_REFERENCE=1` clears inherited Q36/COLI/OpenMP experiments and ignores saved
settings. It keeps the model path (`SNAP`, if customized), restores the tested
launcher defaults, and applies only to this command. Normal daily launches still
respect saved settings and explicit environment overrides.

### 5. Measure one focused sweep

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline spin0 spin200 spin2000 io4 spin2000-io4 \
  --workloads bench long --tokens 256 --repeats 5 \
  --heat-seed "$M/.q36_heat" --output "$SESSION/scheduling"
```

This is 72 engine processes including warm-ups. Profiles alternate order across
repeats; each starts from an identical private heat copy. Original history stays
unchanged. `spin0` blocks immediately, `spin200` polls up to 200 microseconds,
and `spin2000` up to 2 ms. The combination tests one specific interaction, not
an uncontrolled collection of switches.

Swap warnings are system-wide. A few swapped pages are not proof of engine
failure; very low available RAM or persistent pressure still invalidate a
comfortable daily-use claim. Do not raise the RAM fraction to compensate.

### 6. Inspect the candidate, without saving yet

```bash
bash benchmark-results/turbo-qwen36.sh best "$SESSION/scheduling/summary.json"
```

No qualifying candidate means keep the reference. That is a valid result.
Older summaries or incomplete workloads cannot be automatically promoted.
The tool requires at least 1 GiB free RAM, full timing/memory coverage, stable
baseline output and matching candidate token fingerprints.

If it identifies a winner, repeat only `baseline` and that printed profile in a
new output directory, with the same options and five repeats. Run this confirmation
in a separate session under comparable AC/desktop conditions. Substitute the
printed profile for `io4` below; **do not assume io4 won**:

```bash
# Example only: use the actual qualifying profile instead of io4.
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline io4 --workloads bench long --tokens 256 --repeats 5 \
  --heat-seed "$M/.q36_heat" --output "$SESSION/confirmation"
bash benchmark-results/turbo-qwen36.sh best "$SESSION/confirmation/summary.json"
```

Only after both comparisons qualify, and after inspecting pressure and slow runs,
save the confirmation winner:

```bash
bash benchmark-results/turbo-qwen36.sh best \
  "$SESSION/confirmation/summary.json" --save
```

This replaces `~/.config/colibri/q36.env` with measured settings. It does not save
private heat filenames, profiling flags or secrets. Explicit shell overrides
still win. `Q36_REFERENCE=1` deliberately ignores this saved winner.

### 7. Measure actual persistent serving

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh agent \
  --profiles baseline spin2000 io4 --repeats 3 \
  --heat-seed "$M/.q36_heat" --max-tokens 512 --require-prefix-reuse \
  --output "$SESSION/serving"
```

Nine independent servers run sequentially, each with four turns in one
conversation. Each result includes startup, first output, first answer, decode
rate, token usage, finish reason, cache capacity and prefix events.
`system.jsonl` samples gateway/engine processes and memory; `server.gpu.csv`
captures GPU telemetry. The original uploaded coding run had a smaller cache
than direct benchmarks: compare actual capacity, not just RAM fraction.

Missing reuse or truncated answers return nonzero. Keep the logs; do not hide
the error. These tasks measure interaction and consistency, not validated coding
quality. Never apply a generated patch merely because the model proposes it.

### 8. Capture decode-only profiling and its control

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 3 --heat-seed "$M/.q36_heat" \
  --output "$SESSION/profile-control"
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 3 --heat-seed "$M/.q36_heat" --profile-after 2 \
  --output "$SESSION/profile-decode"
```

The profile must log `chain profile scope: decode only (after prefill)`.
Unsupported timestamps fail explicitly. Profiling changes execution overhead:
use the unprofiled runs to judge throughput.

Inspect `[q36 decode vk]` for decode-only waits, frames and transfers, and
`[q36 decode cache]` for per-layer misses/load time. Load times sum work across
loader threads; fence time includes useful device work. Neither is automatically
an independent additive latency component.

### 9. Optional thinking-prefix check

```bash
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh agent \
  --think --max-tokens 2048 --require-prefix-reuse \
  --output "$SESSION/thinking"
```

This is a diagnostic, not the fast default. The agent resends reasoning and opts
into exact whitespace preservation. Truncated thinking/answers are reported as
incomplete, not successful coding responses. Review `[PREFIX]` lines even if the
request finishes. If this fails, retain logs and continue using thinking off.

### 10. Upload only the reviewed results

```bash
git status --short
git add -f "$SESSION"
git diff --cached --stat
git diff --cached --check
git commit -m "Round 5: controlled scheduling, serving and decode profiling results"
git push origin main
```

Before committing, inspect the files for private prompts, paths you do not want
published, API keys or other sensitive information. Do not use `git add .`:
local shader rebuilds and unrelated work should not be bundled with benchmark
results. Do not force push. If the remote advanced, commit your reviewed results,
then `git pull --rebase` and push normally; stop if conflicts occur.

Send the new commit hash. The next development step is to pull these logs and
select the highest-cost verified bottleneck, not assume the experiments won.

## Daily use and rollback

For the known-good reference:

```bash
cd ~/personal_proj
Q36_REFERENCE=1 bash benchmark-results/turbo-qwen36.sh serve
```

For an independently confirmed saved winner, use a fresh terminal:

```bash
cd ~/personal_proj
bash benchmark-results/turbo-qwen36.sh serve
```

Leave the terminal open. Client URL: `http://127.0.0.1:8000/v1`.
Keep thinking off, retain the same conversation, and leave room for output within
the context limit. First large-context requests still require full prefill.
Ctrl+C stops the server. The reference launch is the rollback; no model deletion,
cache enlargement, swap disabling or destructive Git command is needed.

## What happens after these measurements

1. Match serving/direct workload and memory conditions; explain the observed gap.
2. Attribute decode cost to GPU arithmetic, host submission, transfers and expert
   misses. Investigate the slower tail as well as median throughput.
3. Change one measured bottleneck, with output/state regressions and a feature gate
   for any behavior shift. Per-layer cache counters can justify a trace/replay study
   before changing allocation; current totals alone do not justify it.
4. Validate near 8K and 12K prompts with output room and safe memory before claiming
   long-context improvements.
5. Compile/test reviewed generated code only after inference is stable.

About 50% more short throughput and 78% more long throughput are needed for
30 tok/s. A smaller model or different hardware is a separate quality/capacity
decision, not an optimization silently applied to this checkpoint.
