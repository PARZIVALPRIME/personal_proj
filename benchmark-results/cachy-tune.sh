#!/usr/bin/env bash
# Reversible CachyOS/Linux system tuning for the Qwen3.6 laptop run (Ryzen 7 5800HS, RTX 3050 4 GB, 16 GB RAM).
#
#   sudo bash benchmark-results/cachy-tune.sh status    show current values, change nothing
#   sudo bash benchmark-results/cachy-tune.sh apply     apply the tuning, remembering the old values
#   sudo bash benchmark-results/cachy-tune.sh revert    restore the remembered values
#
# Why: decode is latency-bound, not bandwidth-bound (~0.57 GB of experts per token at 20 tok/s is
# ~11 GB/s of ~50 GB/s). The 40-layer GPU<->CPU chain waits on the device and on idle cores, so
# anything that makes cores/GPU wake slowly or lets pages of the expert cache be swapped costs speed.
#
# Optional switches (environment): GPU_LOCK=1 pin NVIDIA graphics clocks to the maximum (more heat),
# THP=always transparent huge pages for the expert cache, SWAPPINESS=20, DMA_LATENCY=0 (microseconds).
# Everything is runtime-only: a reboot restores the system defaults.
set -uo pipefail
state=/run/colibri-tune.state
pidfile=/run/colibri-tune.dma.pid

[ "$(id -u)" = 0 ] || { echo "ERROR: run with sudo" >&2; exit 1; }

get() { [ -r "$1" ] && cat "$1" 2>/dev/null || echo "n/a"; }
thp_cur() { sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null; }
save() { echo "$1=$2" >> "$state"; }
setv() { # path value: write if the file exists, remember the old value once
    [ -w "$1" ] || return 0
    grep -q "^F:$1=" "$state" 2>/dev/null || save "F:$1" "$(cat "$1")"
    echo "$2" > "$1" 2>/dev/null || echo "  could not set $1" >&2
}

status() {
    echo "governor:            $(get /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "amd_pstate status:   $(get /sys/devices/system/cpu/amd_pstate/status)"
    echo "EPP:                 $(get /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
    echo "boost:               $(get /sys/devices/system/cpu/cpufreq/boost)"
    echo "THP:                 $(thp_cur)"
    echo "swappiness:          $(sysctl -n vm.swappiness)"
    echo "page-cluster:        $(sysctl -n vm.page_cluster)"
    echo "vfs_cache_pressure:  $(sysctl -n vm.vfs_cache_pressure)"
    echo "compaction_proact.:  $(sysctl -n vm.compaction_proactiveness 2>/dev/null || echo n/a)"
    echo "zram/swap:"; swapon --show 2>/dev/null | sed 's/^/  /'
    echo "dma latency holder:  $( [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null && echo "running (pid $(cat "$pidfile"))" || echo off)"
    for d in /sys/block/nvme*n1; do [ -d "$d" ] && echo "$(basename "$d") scheduler: $(get "$d/queue/scheduler") read_ahead_kb: $(get "$d/queue/read_ahead_kb")"; done
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,pstate,clocks.gr,clocks.max.gr,power.draw,persistence_mode --format=csv,noheader 2>/dev/null
    echo "scx scheduler:       $(get /sys/kernel/sched_ext/root/ops)"
}

apply() {
    rm -f "$state"; : > "$state"
    for g in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
        setv "$g/scaling_governor" performance
        setv "$g/energy_performance_preference" performance
    done
    setv /sys/devices/system/cpu/cpufreq/boost 1
    for k in vm.swappiness vm.vfs_cache_pressure vm.compaction_proactiveness vm.watermark_scale_factor; do
        save "S:$k" "$(sysctl -n "$k" 2>/dev/null)"
    done
    sysctl -qw vm.swappiness="${SWAPPINESS:-20}" vm.vfs_cache_pressure=50 \
        vm.compaction_proactiveness=0 vm.watermark_scale_factor=125 2>/dev/null
    [ -n "${THP:-}" ] && setv /sys/kernel/mm/transparent_hugepage/enabled "$THP"
    for d in /sys/block/nvme*n1; do [ -d "$d" ] && setv "$d/queue/scheduler" none; done
    # Hold PM QoS so cores never drop into deep C-states while waiting on the GPU or the loaders.
    if ! { [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; }; then
        python3 - "${DMA_LATENCY:-0}" "$pidfile" <<'PY' &
import os, struct, sys, time
fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
os.write(fd, struct.pack("i", int(sys.argv[1])))
open(sys.argv[2], "w").write(str(os.getpid()))
while True:
    time.sleep(3600)
PY
        sleep 1
    fi
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi -pm 1 >/dev/null 2>&1
        if [ "${GPU_LOCK:-0}" = 1 ]; then
            max=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -1)
            [ -n "$max" ] && nvidia-smi -lgc "$max,$max" >/dev/null 2>&1 && save GPULOCK 1
        fi
    fi
    echo "applied. Old values remembered in $state (revert: sudo bash $0 revert)"
    status
}

revert() {
    [ -f "$state" ] || { echo "nothing to revert (no $state); a reboot also restores defaults"; return 0; }
    while IFS= read -r line; do
        case "$line" in
            F:*) key=${line%%=*}; echo "${line#*=}" > "${key#F:}" 2>/dev/null ;;
            S:*) key=${line%%=*}; sysctl -qw "${key#S:}=${line#*=}" 2>/dev/null ;;
            GPULOCK=1) nvidia-smi -rgc >/dev/null 2>&1 ;;
        esac
    done < "$state"
    [ -f "$pidfile" ] && kill "$(cat "$pidfile")" 2>/dev/null; rm -f "$pidfile" "$state"
    echo "reverted."
}

case "${1:-status}" in
    status) status ;;
    apply) apply ;;
    revert) revert ;;
    *) echo "usage: $0 status|apply|revert" >&2; exit 2 ;;
esac
