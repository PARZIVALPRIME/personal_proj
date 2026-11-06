# CachyOS XFCE4: fast offline coding with Qwen3.6-35B-A3B

Laptop: Ryzen 7 5800HS (8 cores / 16 threads), RTX 3050 Laptop 4 GB, 16 GB RAM, NVMe.
Model: `~/colibri-models/qwen36-35b` (int4 gs64, 256 experts, top-8, packed `experts.q36x`).

## Where we are

| Stage | Generation speed | 5,338-token prompt TTFT | Notes |
|---|---:|---:|---|
| Start (Windows) | 1.83 tok/s | 81.6 s | |
| Windows sweep-8 | 15.3 tok/s | 19.1 s | |
| CachyOS, packed experts | 16.4-16.7 tok/s | 17.4-18.5 s | |
| **CachyOS, 6 threads + 0.80 cache (defaults)** | **18.5 tok/s; decode 20.6 (short) / 16.8 (after the long prompt)** | **16.5 s** | ~1.4 GiB RAM stays free |

Source: `benchmark-results/linux-memory-20261107-000107` (256 output tokens, medians of 3).
"Generation speed" is the engine's `Speed:` line, which includes the prompt;
**decode** excludes the time to the first token: (tokens - 1) / (total - TTFT). The
benchmark tools now report both; decode is the number to compare against 30 tok/s.

What already runs: dense layers + lm_head on the GPU (Vulkan), routed experts on the
CPU for decode and on the GPU for prompts, an expert cache in RAM with heat history,
experts streamed from the packed NVMe file by 8 loader threads with read-ahead and
lookahead, and exact prompt-lookup speculation.

## The 30 tok/s target, honestly

Decode is about 20.6 tok/s (about 48 ms per token). 30 tok/s means 33 ms per token: a
1.45x speed-up. Each forward is serial: GPU dense layer, then the 8 routed experts on the
CPU, then the next layer, 40 times. Nothing in this round removes that chain. What can help:

| Lever | What it does | Expected | Quality |
|---|---|---:|---|
| **MTP draft head** (`Q36_MTP=1`) | Qwen's own next-token head guesses 1 token ahead; one forward checks it. Accepted guesses give 2 tokens per forward. | +10-25% (other machines: +7% CPU-only, +12-14% with a GPU) | Identical output: every token is verified by the full model |
| Lookahead 2 layers (`la2`, `la2x8`) | Reads expert misses earlier, hiding SSD time (~10 ms of fetch per token today) | 0-5% | Identical |
| GPU experts for decode (`tierdec`) | Lets the 550 experts already in VRAM run decode rows | Probably slower (it was on Windows); re-checked because it is free to test | Identical |
| Other int8 GEMV shader (`gemv2`) | Alternative GPU kernel for the dense layers | 0-5% | Identical |
| Longer lookup drafts (`lookup7`) | More prompt-lookup tokens per verify, helps code edits | 0-5% | Identical |

**Realistic result of this round: about 22-25 tok/s decode.** 30 tok/s is a stretch:
if the measurements fall short, the next round works on the ~33 ms per token spent
outside the expert timers (GPU chain, CPU/GPU hand-offs) from the profile in step 12.
Not doing: expert skipping, lower precision, smaller models, disabling swap or larger
caches. Those trade quality or stability for speed.

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

- The launcher applies the settings measured best on this laptop automatically: they are
  saved in `~/.config/colibri/q36.env` by step 6 and step 8 below.
- Thinking is **off by default** through the API: answers start immediately. A client
  can enable it per request (`enable_thinking: true`); expect much longer answers.
- Keep one conversation going rather than starting new chats: one KV slot holds one
  conversation, so switching conversations costs a full prefill again.
- For an interactive terminal chat instead: `bash benchmark-results/turbo-qwen36.sh chat`.
- Close browsers and heavy apps when you can; the cache is sized from free RAM at start.

## Step by step: this round, from pull to push

Order: **optimize first** (steps 5-9), **then the coding tests** (steps 10-11), then an
optional GPU profile (step 12) and the push (steps 13-14).

Open **Terminal Emulator** in XFCE4, type `bash`, press Enter. Run the blocks in order in
that same terminal. Plug in AC power and close heavy applications.
**If a command fails, stop and keep its output.** Allow about 2-3 hours in total, most of
it unattended benchmark runs.

### 1. Pull

```bash
cd ~/personal_proj
git status --short
git pull --ff-only
```

If Git reports conflicts, stop; do not force anything.

### 2. Set the environment

```bash
unset SNAP RAM_GB PROMPT N_NEW OMP_NUM_THREADS Q36_RAM_FRAC OMP_WAIT_POLICY GOMP_SPINCOUNT
unset KMP_BLOCKTIME OMP_PLACES OMP_PROC_BIND COLI_NO_OMP_TUNE COLI_OMP_TUNED COLI_ENGINE
unset Q36_MTP Q36_MTP_DRAFTS Q36_LOOKAHEAD Q36_LOOKAHEAD_EXTRA Q36_CONF
M="$HOME/colibri-models/qwen36-35b"
test -d "$M" && echo "Model directory OK"
test -s "$M/experts.q36x" && echo "Packed store OK"
test -x c/qwen36 && echo "Engine OK"
```

All three must print `OK`. Leave `SNAP` unset: the launcher finds the model itself and
switches to the MTP copy when an MTP setting asks for it.

### 3. Run the tests

```bash
python3 -m unittest discover -s c/tests -p 'test_q36_*.py' -v
bash -n benchmark-results/turbo-qwen36.sh
```

The tests must end with `OK`.

### 4. Create this round's results folder

```bash
SESSION="benchmark-results/linux-round4-$(date +%Y%m%d-%H%M%S)"
mkdir "$SESSION"
printf 'Results folder: %s\n' "$SESSION"
```

### 5. Optimize, part A: decode settings that need no setup

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline la2 la2x8 tierdec gemv2 lookup7 \
  --tokens 256 --repeats 3 --output "$SESSION/levers"
```

48 engine runs (6 settings x short and long prompt x one warm-up + 3 measured), about
40-60 minutes. Each finished configuration prints one line with its decode speed.

### 6. Keep part A's winner

```bash
bash benchmark-results/turbo-qwen36.sh best "$SESSION/levers/summary.json" --save
```

It prints why each setting won or lost. A setting is kept only if it is at least 3%
faster on decode than `baseline`, has no failed runs, leaves at least 512 MiB of RAM
free, and is not more than 5% worse on the long prompt (decode and first token). If
nothing qualifies it says `keep the current settings` and saves nothing. A saved winner
goes into `~/.config/colibri/q36.env`; from then on `baseline` includes it.

### 7. Optimize, part B: set up the MTP draft head (one time)

The container you downloaded has no MTP head. This adds it from Qwen's official release
without touching your model folder:

```bash
bash benchmark-results/turbo-qwen36.sh mtp-setup
```

What it does: creates `~/.venv-colibri` with CPU-only PyTorch (about 1 GB, needed only
for this conversion), downloads the two official shards that hold the head (about 6 GB)
into `~/colibri-models/qwen36-mtp-source`, writes `~/colibri-models/qwen36-35b-mtp`
(the head, about 0.5 GB, plus links to every file of your model), then deletes the 6 GB
download. It needs internet once and about 8 GB of free disk. Make sure no model server
is running: the conversion uses several GB of RAM. Running it again does nothing.

It must end with `MTP head ready`. If the download is interrupted, run it again: it resumes.
If installing PyTorch fails (no wheel yet for CachyOS's Python version), use the distribution's
packages instead, then rerun with that environment:

```bash
sudo pacman -S --needed python-pytorch python-safetensors python-numpy python-huggingface-hub
python3 -m venv --system-site-packages ~/.venv-colibri-sys
bash benchmark-results/turbo-qwen36.sh mtp-setup --venv ~/.venv-colibri-sys
```

### 8. Optimize, part B: measure the MTP head and keep the winner

```bash
bash benchmark-results/turbo-qwen36.sh sweep \
  --profiles baseline mtp1 mtp2 mtpauto \
  --tokens 256 --repeats 3 --output "$SESSION/mtp"
bash benchmark-results/turbo-qwen36.sh best "$SESSION/mtp/summary.json" --save
cat ~/.config/colibri/q36.env 2>/dev/null || echo "No saved settings: defaults are best"
```

32 engine runs, about 30-40 minutes. `mtp1` = one guessed token per forward (best on
other machines), `mtp2` = two, `mtpauto` = the engine decides per step. The logs show
`[qwen36 mtp] run: N tokens/forward | acceptance X%`. If the head cannot fit in GPU
memory it runs on the CPU and may be slower; `best` then rejects it automatically.

### 9. Confirm the final settings

```bash
bash benchmark-results/turbo-qwen36.sh bench --tokens 256 --repeats 3 \
  --output "$SESSION/final-short"
bash benchmark-results/turbo-qwen36.sh long --tokens 256 --repeats 3 \
  --output "$SESSION/final-long"
cp ~/.config/colibri/q36.env "$SESSION/" 2>/dev/null || true
```

These run with everything saved in steps 6 and 8. The last line of each prints the
median decode speed: this is the round's result.

### 10. Coding-agent test (thinking off)

Make sure no other server is using port 8011, then:

```bash
bash benchmark-results/turbo-qwen36.sh agent --output "$SESSION/agent"
```

It starts a server with the saved settings, sends 4 coding turns with the 5,338-token
code file as repository context, prints one line per turn, then stops the server. Look for:
- turn 1: first token around 16-17 s (the full prompt);
- turns 2-4: much lower first-token time (only the new part of the prompt);
- no `WARNING: no prefix reuse` line.

### 11. Coding-agent test (thinking on)

```bash
bash benchmark-results/turbo-qwen36.sh agent --think --max-tokens 1024 \
  --output "$SESSION/agent-think"
```

### 12. Profile the GPU chain (for the next round)

```bash
COLI_VK_CHAIN_PROF=0 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 2 --output "$SESSION/short-control"
COLI_VK_CHAIN_PROF=2 bash benchmark-results/turbo-qwen36.sh bench \
  --tokens 256 --repeats 2 --output "$SESSION/short-profile"
```

Profiling adds timestamps and slows the runs: speed numbers come from the control runs,
the cost breakdown from the profiled ones (`[VK] chain profile` lines in the logs).

### 13. Look at the results

```bash
for f in "$SESSION"/*/summary.json; do echo "== $f"; python3 -c '
import json,sys
for r in json.load(open(sys.argv[1])):
    print("%-5s %-14s runs=%s fail=%s decode=%s tok/s ttft=%s s tok/fwd=%s mem_warn=%s" % (
        r["workload"], r["profile"], r["valid_runs"], r["failed_runs"], r.get("decode_tps_median"),
        r.get("ttft_s_median"), r.get("tokens_per_forward_median"), r.get("memory_warning_runs")))
' "$f"; done
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

### 14. Commit and push only this round's results

```bash
git diff --cached --name-only
```

If that lists anything, stop and sort it out first. Otherwise:

```bash
git add -f -- "$SESSION"
git diff --cached --stat
git commit -m "Round 4: decode levers, MTP head, coding-agent and profile results"
git push origin main
git log -1 --oneline
```

Do not use `git add -A`. If the push fails, the commit stays local: do not force-push.
Send the commit line and the results folder name for the next analysis.

## If something is worse

Nothing changes your model folder, and every saved setting can be undone:

```bash
Q36_CONF=none bash benchmark-results/turbo-qwen36.sh serve     # ignore saved settings once
rm ~/.config/colibri/q36.env                                   # forget them for good
OMP_NUM_THREADS=auto Q36_RAM_FRAC=0.92 bash benchmark-results/turbo-qwen36.sh serve   # pre-November 7
```

A setting in the shell always wins over the saved file, e.g. `Q36_MTP=0 bash ... serve`.
The MTP copy `~/colibri-models/qwen36-35b-mtp` holds only the head and links; deleting
it frees about 0.5 GB and does not touch the original model.

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
swap counters, `[VK] chain profile` lines), GPU samples, and `summary.csv` with medians of
`decode_tps`, `ttft_s`, `step_ms`, `tokens_per_forward`, `mtp_acceptance_pct` and more.
Profiles: `baseline`, `previous`, `auto-threads`, `threads6`, `threads7`, `io4`, `io6`,
`passive`, `passive-io4`, `bind`, `ram75`, `ram80`, `ram88`, `threads6-ram80`, `la2`,
`la2x8`, `tierdec`, `gemv2`, `lookup7`, `mtp1`, `mtp2`, `mtpauto`.

`best` (`q36_bench.py best <summary.json> [--save [FILE]]`): applies the fixed rules of
step 6 and prints `export` lines for the winner; `--save` merges it into the settings file.

`agent` (`c/tools/q36_agent_bench.py`): persistent server, streamed multi-turn
conversation, per-turn first-token time, first answer text, total time, tokens,
generation rate, finish reason, and the engine's `[PREFIX]` reuse decisions.

`mtp-setup` (`c/tools/q36_mtp_setup.py`): the one-time MTP head conversion of step 7.

- `reported_speed_tps` includes prompt processing; `decode_tps` does not.
- `step_ms` is the engine's decode step; with speculation a step can yield >1 token.
- Swap counters are system-wide. Memory is sampled once a second.
- History files evolve across runs; profiles are interleaved to limit drift.
