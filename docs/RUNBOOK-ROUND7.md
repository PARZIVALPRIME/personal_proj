# Round 7 runbook: pull, run, push

Nothing here was run on the laptop by the author; the Windows dev box cannot compile or run the engine. Report failures as they are.

## What changed
- `c/openai_server.py`: `warm_prefix`. With `COLI_PREFIX_WARM=1` (qwen36 only), the first chat whose system prompt is 3000+ characters is preceded by a one-token pinned request of that prefix. Later sessions with the same prefix restore the pinned state and prefill only their own tail. Failure is logged and ignored. Tests: `WarmPrefixTest`.
- `benchmark-results/turbo-qwen36.sh`: `COLI_PREFIX_WARM=1` and `COLI_PIN_SLOTS=6` by default; `Q36_CACHE_GB=N` fixes the expert cache (sets `RAM_GB`) so runs are comparable.

## Steps (laptop)
1. `cd ~/personal_proj && git pull`
2. Optional OS tuning: `sudo benchmark-results/cachy-tune.sh apply` (undo with `revert`). Record `status` output.
3. Free memory before a run: `swapon --show`, `free -m`; reboot or `sudo swapoff -a && sudo swapon -a` if zram is heavily used.
4. Fixed-cache A/B, one session, same order:
   - `Q36_CACHE_GB=9 COLI_PREFIX_WARM=0 benchmark-results/turbo-qwen36.sh agent`
   - `Q36_CACHE_GB=9 COLI_PREFIX_WARM=1 benchmark-results/turbo-qwen36.sh agent`
   Lower `Q36_CACHE_GB` if memory warnings appear.
5. Look at the first turn of each new session: TTFT should fall from about 17 s to the tail only. The server log should show `prefix warm-up: system prefix of N characters pinned`. If it shows a second full prefill, or "skipped", send the log.
6. Push results: `git add benchmark-results && git commit -m "Round 7 results" && git push`.

## Adopt or drop
Adopt if new-session TTFT drops 25% or more, output is identical, and there are no new memory warnings. Otherwise set `COLI_PREFIX_WARM=0` and tell me.
