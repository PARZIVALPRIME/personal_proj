# CachyOS XFCE4: fast offline coding with Qwen3.6-35B-A3B

Laptop: Ryzen 7 5800HS (8 cores / 16 threads), RTX 3050 Laptop 4 GB, 16 GB RAM, NVMe.
Model: `~/colibri-models/qwen36-35b` (int4 gs64, 256 experts, top-8, packed `experts.q36x`).

## Where we are

| Stage | Short prompt | 5,338-token prompt TTFT | Notes |
|---|---:|---:|---|
| Start (Windows) | 1.83 tok/s | 81.6 s | |
| Windows sweep-8 | 15.29 tok/s | 19.1 s | |
| CachyOS, packed experts | 16.4-16.7 tok/s | 17.4-18.5 s | |
| **CachyOS, 6 threads + 0.80 cache (now default)** | **18.5 tok/s** | **16.5 s** | ~1.4 GiB RAM stays free |

The last row comes from `benchmark-results/linux-memory-20261107-000107` (256 output
tokens, medians of three runs). Six threads at the old 0.92 cache fraction reached
19.8 tok/s on the short prompt, but on long prompts left only 129 MiB free, swapped
heavily and took 18.1 s to the first token. **0.80 trades ~6% short-prompt speed for
a stable machine and faster long prompts.** These are now the launcher defaults.

What already runs: dense layers + lm_head on the GPU (Vulkan), routed experts on the
CPU for decode and on the GPU for prompts, expert cache in RAM with heat history,
experts streamed from the packed NVMe file with 8 loader threads, read-ahead,
lookahead and exact prompt-lookup speculation.

## Daily use: fastest coding setup

**Keep one server running.** Loading the model and warming the expert cache takes
roughly 10-20 seconds; a persistent server pays that once. It also keeps the
conversation's state: when a coding client resends the same history, the engine
processes only the new part instead of the whole prompt (prefix reuse).

```bash
cd ~/personal_proj
bash benchmark-results/turbo-qwen36.sh serve
```

Leave that terminal open. Point your coding client at the OpenAI-compatible URL
`http://127.0.0.1:8000/v1` (any API key unless you set `--api-key`). Stop with Ctrl+C.

- Thinking is **off by default** through the API: answers start immediately. A client
  can enable it per request (`enable_thinking: true`); expect much longer answers.
- Keep one conversation going rather than starting new chats: one KV slot holds one
  conversation, so switching conversations costs a full prefill again.
- For an interactive terminal chat instead: `bash benchmark-results/turbo-qwen36.sh chat`.
- Close browsers and heavy apps when you can; the cache is sized from free RAM at start.

## The plan, in order

1. **Prove the coding path is fast (this round).** Measure a 4-turn coding conversation
   on a persistent server: first-token time per turn, generation rate, and whether turns
   2-4 reuse the prompt. With reuse, a later turn should start in seconds, not ~16 s.
2. **Profile the remaining decode time (this round).** About 33 ms of each ~62 ms decode
   step is outside the expert timers. The engine already has Vulkan timestamp profiling
   (`COLI_VK_CHAIN_PROF`); we run it, with unprofiled control runs for honest numbers.
3. **Implement from the evidence (next round).** Depending on step 2:
   - the dense GPU matrices dominate: tune the leading matrix shapes/shaders;
   - many small GPU operations or waits dominate: fewer submissions, command reuse,
     targeted fusion;
   - CPU expert compute dominates: profile the AVX2 kernel against RAM bandwidth.
4. **Long-prompt prefill.** 16.5 s for 5,338 tokens: profile prefill separately (expert
   streaming vs GPU compute) before changing anything.
5. **Prefix reuse gaps.** If step 1 shows no reuse, fix that first: it is the largest
   win for coding agents.

Not doing: disabling swap, larger caches or buffers, expert skipping, lower precision,
a backend port, or a draft model. Each either failed on this laptop already, would
cost quality, or needs RAM we do not have.

## Step by step: this round, from pull to push

Open **Terminal Emulator** in XFCE4, type `bash`, press Enter. Run the blocks in order
in that same terminal. Plug in AC power, close heavy applications.
**If a command fails, stop and keep its output.**

### 1. Pull

```bash
cd ~/personal_proj
git status --short
git pull --ff-only
```

If Git reports conflicts, stop; do not force anything.

### 2. Set the environment

```bash
export SNAP="$HOME/colibri-models/qwen36-35b"
unset RAM_GB PROMPT N_NEW OMP_NUM_THREADS Q36_RAM_FRAC OMP_WAIT_POLICY GOMP_SPINCOUNT
unset KMP_BLOCKTIME OMP_PLACES OMP_PROC_BIND COLI_NO_OMP_TUNE COLI_OMP_TUNED COLI_ENGINE
test -d "$SNAP" && echo "Model directory OK"
test -s "$SNAP/experts.q36x" && echo "Packed store OK"
test -x c/qwen36 && echo "Engine OK"
```

All three must print `OK`. The launcher now supplies 6 threads and the 0.80 cache.

### 3. Run the tests

```bash
python3 -m unittest discover -s c/tests -p 'test_q36_*.py' -v
bash -n benchmark-results/turbo-qwen36.sh
```

The tests must end with `OK`.

### 4. Create this round's results folder

```bash
SESSION="benchmark-results/linux-round3-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
printf 'Results folder: %s\n' "$SESSION"
```

### 5. Coding-agent test (thinking off)

Make sure no other server is using port 8011, then:

```bash
bash benchmark-results/turbo-qwen36.sh agent --output "$SESSION/agent"
```

It starts its own server, sends 4 coding turns with the 5,338-token code file as
repository context, prints one line per turn, then stops the server. Look for:
- turn 1: first token around 16-17 s (full prompt);
- turns 2-4: much lower first-token time;
- no `WARNING: no prefix reuse` line.

### 6. Coding-agent test (thinking on)

```bash
bash benchmark-results/turbo-qwen36.sh agent --think --max-tokens 1024 \
  --output "$SESSION/agent-think"
```

This shows the cost of thinking and whether reuse still works when the reasoning is
sent back with the history.

### 7. Confirm the new defaults against the old ones

```bash
bash benchmark-results/turbo-qwen36.sh sweep --profiles baseline previous \
  --tokens 256 --repeats 3 --output "$SESSION/defaults"
cat "$SESSION/defaults/summary.csv"
```

`baseline` = new defaults (6 threads, 0.80). `previous` = engine thread count, 0.92.
16 engine runs including warm-ups.

### 8. Profile the GPU chain

Control runs first (no profiling, for honest speed), then profiled runs:

```bash
COLI_VK_CHAIN_PROF=0 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 3 --output "$SESSION/short-control"
COLI_VK_CHAIN_PROF=2 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 3 --output "$SESSION/short-profile"
COLI_VK_CHAIN_PROF=0 bash benchmark-results/turbo-qwen36.sh long \
  --tokens 256 --repeats 3 --output "$SESSION/long-control"
COLI_VK_CHAIN_PROF=2 bash benchmark-results/turbo-qwen36.sh long \
  --tokens 256 --repeats 3 --output "$SESSION/long-profile"
```

Profiling adds timestamps and may slow the runs: speed claims come from the control
runs, the cost breakdown from the profiled ones (`[VK] chain profile` lines).

### 9. Look at the results

```bash
for f in "$SESSION"/*/summary.csv; do echo "== $f"; cat "$f"; done
for f in "$SESSION"/agent*/agent.json; do echo "== $f"; python3 -c '
import json,sys
r=json.load(open(sys.argv[1]))
print("error:", r.get("error"), "| reuse:", r.get("prefix_reuse_engaged"), "| events:", r.get("prefix_events"))
for t in r.get("turns", []):
    print(t["turn"], "ttft", t["ttft_s"], "total", t["total_s"], "tokens", t["completion_tokens"], "rate", t["generation_tps"], t["finish_reason"])
' "$f"; done
```

Check: failed runs, `memory_warning_runs`, agent errors, prefix reuse, finish reasons
(`length` means the answer hit the token limit). The logs contain generated text and
local paths; review before sharing.

### 10. Commit and push only this round's results

```bash
git diff --cached --name-only
```

If that lists anything, stop and sort it out first. Otherwise:

```bash
git add -f -- "$SESSION"
git diff --cached --stat
git commit -m "Round 3: coding-agent, defaults and Vulkan profile results"
git push origin main
git log -1 --oneline
```

Do not use `git add -A`. If the push fails, the commit stays local: do not force-push.
Send the commit line and the results folder name for the next analysis.

## If something is worse

Everything is per-command; nothing is changed permanently. Return to the previous
configuration for one command:

```bash
OMP_NUM_THREADS=auto Q36_RAM_FRAC=0.92 bash benchmark-results/turbo-qwen36.sh serve
```

Use the engine binary saved before the November 5 rebuild:

```bash
COLI_ENGINE="$PWD/benchmark-results/linux-known-good/qwen36" \
  bash benchmark-results/turbo-qwen36.sh serve
```

Engine features can be switched off one at a time: `COLI_VK_TIER=0` (CPU experts for
prompts), `Q36_VK_LAYERWISE=0`, `Q36_RA=0`, `Q36_PF_EXPERT=0`, `COLI_KV_PREFIX=0`
(prefix reuse, only for comparison).

## Reference: what the tools record

`bench` / `long` / `sweep` (`c/tools/q36_bench.py`): one warm-up plus measured runs per
configuration, full logs, per-run JSON (command, hashes, environment, timings, memory and
swap counters, `[VK] chain profile` lines), GPU samples, and `summary.csv` with medians.
Profiles: `baseline`, `previous`, `auto-threads`, `threads6`, `threads7`, `io4`, `io6`,
`passive`, `passive-io4`, `bind`, `ram75`, `ram80`, `ram88`, `threads6-ram80`.

`agent` (`c/tools/q36_agent_bench.py`): persistent server, streamed multi-turn
conversation, per-turn first-token time, first answer text, total time, tokens,
generation rate, finish reason, and the engine's `[PREFIX]` reuse decisions.

- `reported_speed_tps` includes prompt processing; on long prompts it is not decode speed.
- `step_ms` is the engine's decode step; with prompt lookup a step can yield >1 token.
- Swap counters are system-wide. Memory is sampled once a second.
- History files evolve across runs; profiles are interleaved to limit drift.
