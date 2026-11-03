#!/usr/bin/env bash
# Linux (CachyOS/Arch) twin of turbo-qwen36.bat: Qwen3.6-35B-A3B i4-gs64, Vulkan trunk + CPU experts.
#
#   ./turbo-qwen36.sh              OpenAI-compatible server (coli serve)
#   ./turbo-qwen36.sh chat         interactive chat
#   ./turbo-qwen36.sh xpack        ONE TIME: write experts.q36x beside the model (~18 GB)
#   ./turbo-qwen36.sh bench        one direct run on prompt.txt
#   ./turbo-qwen36.sh long         same with ~2k tokens of source code in
#
# Override before calling: SNAP (model dir), GPU=0 (CPU only), COLI_VK_TIER=0 (CPU experts for prompts)
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/.." && pwd)

SNAP=${SNAP:-$HOME/Models/qwen36_i4_gs64}
GPU=${GPU:-vk}
export COLI_ENGINE="$src/c/qwen36"
export COLI_VK_SHADERS="$src/c/shaders"
[ -x "$COLI_ENGINE" ] || { echo "ERROR: $COLI_ENGINE not found. Run: make -C $src/c qwen36 VK=1" >&2; exit 1; }

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / 2 ))}   # physical cores; SMT siblings serve the loaders
export Q36_IO_THREADS=${Q36_IO_THREADS:-8}
export Q36_EVICT=slru Q36_SLRU_PROT=0.8 Q36_HEAT=1
export Q36_RAM_FRAC=${Q36_RAM_FRAC:-0.92}
export Q36_LOOKAHEAD=${Q36_LOOKAHEAD:-1}
export Q36_XPACK=${Q36_XPACK:-1}
export Q36_MAXT=${Q36_MAXT:-16384}
export COLI_LOOKUP=1 COLI_CUDA=0

if [ "$GPU" = vk ]; then
    export COLI_VULKAN=1 COLI_VK_TIER=${COLI_VK_TIER:-1} COLI_VK_TIER_RESERVE_GB=${COLI_VK_TIER_RESERVE_GB:-0.5}
    export Q36_VK_TIER_MIN_ROWS=${Q36_VK_TIER_MIN_ROWS:-64} COLI_VK_TIER_EXCLUSIVE=${COLI_VK_TIER_EXCLUSIVE:-0}
    export COLI_VK_TIER_HOST=${COLI_VK_TIER_HOST:-0}
else
    export COLI_VULKAN=0
fi

mklong() {
    [ -f "$here/prompt-long.txt" ] && return
    { echo "Review the following C code from an inference engine. List the three most important bugs or risks, one line each."
      cat "$src/c/kv16.h" "$src/c/pin_pool.h"; } > "$here/prompt-long.txt"
}

bench() {
    export N_NEW=${N_NEW:-128} COLI_TIMERS=1 Q36_HEAT_WAIT=1
    "$COLI_ENGINE" 0 4 "${PROMPT:-$here/prompt.txt}" 2>&1 | tee "$here/bench-turbo-$(date +%s).log" |
        grep -E "cache auto-sized|loader threads|Speed:|TTFT|prompt tokens|hit rate|\[timers\]|\[VK\]" || true
}

case "${1:-serve}" in
    chat)  shift; exec "$src/c/coli" chat --model "$SNAP" --cap 0 "$@" ;;
    xpack) Q36_XPACK_BUILD=1 N_NEW=16 exec "$COLI_ENGINE" 0 4 "$here/prompt.txt" ;;
    bench) bench ;;
    long)  mklong; N_NEW=${N_NEW:-48} PROMPT="$here/prompt-long.txt" bench ;;
    serve) [ "${1:-}" = serve ] && shift; exec "$src/c/coli" serve --model "$SNAP" --cap 0 "$@" ;;
    *)     exec "$src/c/coli" serve --model "$SNAP" --cap 0 "$@" ;;
esac
