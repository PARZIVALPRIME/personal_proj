#!/usr/bin/env bash
# Linux (CachyOS/Arch) twin of turbo-qwen36.bat: Qwen3.6-35B-A3B i4-gs64, Vulkan trunk + CPU experts.
#
#   ./turbo-qwen36.sh              OpenAI-compatible server (coli serve)
#   ./turbo-qwen36.sh chat         interactive chat
#   ./turbo-qwen36.sh xpack        ONE TIME: write experts.q36x beside the model (~18 GB)
#   ./turbo-qwen36.sh bench        warm-up + three measured runs on prompt.txt
#   ./turbo-qwen36.sh long         5,338-token reference prompt
#   ./turbo-qwen36.sh doctor       read-only system diagnostics
#   ./turbo-qwen36.sh sweep        repeated baseline/thread/wait-policy experiments
#   ./turbo-qwen36.sh agent        multi-turn coding-agent test against a persistent server
#
# Defaults are the configuration measured best on the 5800HS / RTX 3050 / 16 GB laptop
# (benchmark-results/linux-memory-20261107-000107): 6 compute threads, 80% of free RAM
# for the expert cache. Override before calling: SNAP (model dir), GPU=0 (CPU only),
# COLI_VK_TIER=0 (CPU experts for prompts), OMP_NUM_THREADS=auto (engine's own count),
# Q36_RAM_FRAC=0.92 (the previous, larger cache).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/.." && pwd)

export SNAP=${SNAP:-$HOME/colibri-models/qwen36-35b}
GPU=${GPU:-vk}
export COLI_ENGINE=${COLI_ENGINE:-$src/c/qwen36}
export COLI_VK_SHADERS="$src/c/shaders"

case "$GPU" in vk|0) ;; *) echo "ERROR: GPU must be vk or 0" >&2; exit 2 ;; esac
if [ "${LOSSY:-0}" != 0 ]; then
    echo "ERROR: this launcher runs quality mode only; unset LOSSY." >&2
    exit 2
fi
unset Q36_SKIP_MISS Q36_SKIP_KEEP Q36_SKIP_RENORM COLI_DENSE_BITS COLI_DENSE_INT4
# Six compute threads beat the engine's physical-core count in both laptop sweeps
# (short step 65.5 -> 59.7 ms; with the smaller cache 64.5 -> 62.6 ms on the long run),
# leaving two cores for the expert loaders and the desktop. auto = the engine decides.
if [ "${OMP_NUM_THREADS:-}" = auto ]; then
    unset OMP_NUM_THREADS
else
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
fi
export Q36_IO_THREADS=${Q36_IO_THREADS:-8}
export Q36_EVICT=slru Q36_SLRU_PROT=0.8 Q36_HEAT=${Q36_HEAT:-1}
# 0.92 left 81-129 MiB free and swapped hundreds of thousands of pages on long prompts;
# 0.80 kept about 1.4 GiB free with a lower long TTFT. RAM_GB, if set, overrides this.
export Q36_RAM_FRAC=${Q36_RAM_FRAC:-0.80}
export Q36_LOOKAHEAD=${Q36_LOOKAHEAD:-1}
export Q36_XPACK=${Q36_XPACK:-1}
export Q36_MAXT=${Q36_MAXT:-16384}
export COLI_LOOKUP=${COLI_LOOKUP:-1} COLI_CUDA=0 PILOT=0

if [ "$GPU" = vk ]; then
    export COLI_VULKAN=1 COLI_VK_TIER=${COLI_VK_TIER:-1} COLI_VK_TIER_RESERVE_GB=${COLI_VK_TIER_RESERVE_GB:-0.5}
    export Q36_VK_TIER_MIN_ROWS=${Q36_VK_TIER_MIN_ROWS:-64} COLI_VK_TIER_EXCLUSIVE=${COLI_VK_TIER_EXCLUSIVE:-0}
    export COLI_VK_TIER_HOST=${COLI_VK_TIER_HOST:-0}
else
    export COLI_VULKAN=0 COLI_VK_TIER=0
fi

case "${1:-serve}" in
    bench|long|doctor|sweep)
        command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 1; }
        exec python3 "$src/c/tools/q36_bench.py" "$@"
        ;;
    agent)
        command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 1; }
        shift
        exec python3 "$src/c/tools/q36_agent_bench.py" "$@"
        ;;
esac
[ -d "$SNAP" ] || { echo "ERROR: model directory not found: $SNAP. Export SNAP first." >&2; exit 1; }
[ -x "$COLI_ENGINE" ] || { echo "ERROR: $COLI_ENGINE not executable. Run: make -C \"$src/c\" qwen36 VK=1" >&2; exit 1; }

case "${1:-serve}" in
    chat)  shift; exec "$src/c/coli" chat --model "$SNAP" --cap 0 "$@" ;;
    xpack) Q36_XPACK_BUILD=1 N_NEW=16 exec "$COLI_ENGINE" 0 4 "$here/prompt.txt" ;;
    serve) [ "${1:-}" = serve ] && shift; exec "$src/c/coli" serve --model "$SNAP" --cap 0 "$@" ;;
    *)     exec "$src/c/coli" serve --model "$SNAP" --cap 0 "$@" ;;
esac
