#!/usr/bin/env python3
"""Reproducible Q36 experiments; no packages, root, or global tuning required."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark-results"
PROFILES = {
    "baseline": {},
    "threads6": {"OMP_NUM_THREADS": "6"},
    "io4": {"Q36_IO_THREADS": "4"},
    "passive": {
        "OMP_WAIT_POLICY": "passive", "GOMP_SPINCOUNT": "0", "KMP_BLOCKTIME": "0",
    },
    "passive-io4": {
        "OMP_WAIT_POLICY": "passive", "GOMP_SPINCOUNT": "0", "KMP_BLOCKTIME": "0",
        "Q36_IO_THREADS": "4",
    },
    "bind": {"OMP_PLACES": "cores", "OMP_PROC_BIND": "close"},
    "ram88": {"Q36_RAM_FRAC": "0.88"},
}
METRICS = {
    "ttft_s": r"TTFT:\s*([\d.]+)\s*s",
    "reported_speed_tps": r"Speed:\s*([\d.]+)\s*tok/s",
    "step_ms": r"step\(\) total:\s*([\d.]+)\s*ms/token",
    "fetch_ms": r"expert kernel: fetch\s*([\d.]+)",
    "compute_ms": r"expert kernel: fetch\s*[\d.]+\s*\|\s*compute\s*([\d.]+)",
    "cache_hit_pct": r"Expert cache hit rate:\s*([\d.]+)%",
    "peak_rss_gb": r"PEAK RSS:\s*([\d.]+)\s*GB",
    "tokens_per_forward": r"\[qwen36 lookup\] run:\s*([\d.]+)\s*tokens/forward",
}


def parse_metrics(text):
    return {
        key: float(matches[-1])
        for key, pattern in METRICS.items()
        if (matches := re.findall(pattern, text))
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_environment(env):
    prefixes = ("Q36_", "QWEN", "COLI_", "OMP_", "GOMP_", "KMP_", "VK_")
    return {
        k: v for k, v in sorted(env.items())
        if k.startswith(prefixes) or k in ("SNAP", "N_NEW", "RAM_GB", "PILOT")
    }


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", timeout=30)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def read_optional(path):
    try:
        return Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return None


def sample_system(pid=None):
    sample = {"monotonic_s": time.monotonic()}
    for name in ("meminfo", "vmstat", "pressure/memory"):
        sample[name] = read_optional(Path("/proc") / name)
    cpuinfo = read_optional("/proc/cpuinfo")
    if cpuinfo is not None:
        sample["cpu_mhz"] = re.findall(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
    if pid is not None:
        for name in ("status", "io"):
            try:
                sample[f"process_{name}"] = read_optional(Path("/proc") / str(pid) / name)
            except ProcessLookupError:
                sample[f"process_{name}"] = None
    return sample


def counter(text, key):
    match = re.search(rf"^{re.escape(key)}\s*:?\s+(\d+)", text or "", re.MULTILINE)
    return int(match[1]) if match else None


@contextmanager
def gpu_monitor(output, label, enabled):
    info = {}
    command = shutil.which("nvidia-smi") if enabled else None
    if not command:
        info["note"] = "nvidia-smi unavailable" if enabled else "disabled by request"
        yield info
        return
    fields = ("timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,"
              "memory.used,clocks.current.sm,clocks.current.memory,power.draw,power.limit")
    with (output / f"{label}.gpu.csv").open("w") as data, \
            (output / f"{label}.gpu.stderr").open("w") as errors:
        try:
            proc = subprocess.Popen(
                [command, f"--query-gpu={fields}", "--format=csv", "--loop=1"],
                stdout=data, stderr=errors)
        except OSError as exc:
            info["error"] = str(exc)
            print(f"WARNING: GPU monitor could not start: {exc}", file=sys.stderr)
            yield info
            return
        try:
            yield info
        finally:
            code = proc.poll()
            if code is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                info["stopped_by_runner"] = True
            else:
                info["exit_code"] = code
                if code != 0:
                    print(f"WARNING: GPU monitor exited {code}; inspect {label}.gpu.stderr",
                          file=sys.stderr)


def diagnostics(env):
    snap = Path(env.get("SNAP", str(Path.home() / "colibri-models" / "qwen36-35b")))
    commands = [
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        ["git", "-C", str(ROOT), "status", "--short"],
        ["uname", "-a"], ["lscpu"], ["free", "-h"],
        ["lsblk", "-o", "NAME,TYPE,ROTA,SIZE,FSTYPE,MOUNTPOINTS,MODEL"],
        ["findmnt", "-T", str(snap)], ["df", "-h", str(snap)],
        ["nvidia-smi"], ["vulkaninfo", "--summary"],
        ["powerprofilesctl", "get"], ["sensors"],
        ["cc", "--version"], ["glslc", "--version"],
    ]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "environment": selected_environment(env),
        "system": sample_system(),
        "build_config": read_optional(ROOT / "c" / ".build-config"),
        "shader_sha256": {
            path.name: sha256(path) for path in sorted((ROOT / "c" / "shaders").glob("*.spv"))
        },
        "probes": [probe(command) for command in commands],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_run(env, prompt):
    if not env.get("SNAP") or not Path(env["SNAP"]).is_dir():
        raise ValueError("SNAP must name an existing model directory")
    engine = Path(env.get("COLI_ENGINE", str(ROOT / "c" / "qwen36")))
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise ValueError(f"engine is missing or not executable: {engine}")
    if not prompt.is_file():
        raise ValueError(f"prompt does not exist: {prompt}")
    if env.get("Q36_XPACK", "1") != "0":
        packed = Path(env["SNAP"]) / "experts.q36x"
        if not packed.is_file() or packed.stat().st_size == 0:
            raise ValueError("experts.q36x is missing/empty; run the launcher xpack first, "
                             "or explicitly set Q36_XPACK=0 for an unpacked comparison")
    return engine


def run_once(env, prompt, output, label, timeout, monitor_gpu=True):
    engine = validate_run(env, prompt)
    log_path = output / f"{label}.log"
    command = [str(engine), "0", "4", str(prompt)]
    record = {
        "label": label, "command": command, "environment": selected_environment(env),
        "engine_sha256": sha256(engine), "prompt_sha256": sha256(prompt),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    start = time.monotonic()
    timed_out = False
    first = sample_system()
    available = []
    with log_path.open("w", encoding="utf-8") as log, \
            (output / f"{label}.system.jsonl").open("w", encoding="utf-8") as telemetry, \
            gpu_monitor(output, label, monitor_gpu) as gpu:
        with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT) as proc:
            try:
                while True:
                    sample = sample_system(proc.pid)
                    mem = counter(sample["meminfo"], "MemAvailable")
                    if mem is not None:
                        available.append(mem)
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                    try:
                        code = proc.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - start >= timeout:
                            timed_out = True
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            code = 124
                            break
            except (KeyboardInterrupt, OSError):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
    text = log_path.read_text(encoding="utf-8", errors="replace")
    last = sample_system()
    metrics = parse_metrics(text)
    complete = "ttft_s" in metrics and "reported_speed_tps" in metrics
    record.update(exit_code=code, timed_out=timed_out,
                  wall_s=time.monotonic() - start, metrics=metrics,
                  status="ok" if code == 0 and complete else "failed")
    record["gpu_monitor"] = gpu
    record["system_memory"] = {
        "min_available_kib": min(available) if available else None,
    }
    for key in ("pswpin", "pswpout", "pgmajfault"):
        before, after = counter(first["vmstat"], key), counter(last["vmstat"], key)
        record["system_memory"][key + "_delta"] = (
            after - before if before is not None and after is not None else None)
    swap_out = record["system_memory"]["pswpout_delta"]
    if swap_out:
        print(f"WARNING: {label}: system swapped out {swap_out} pages; "
              "this is system-wide, not proof the engine caused it.", file=sys.stderr)
    if code == 0 and not complete:
        record["error"] = "engine exited 0 without required TTFT and Speed metrics"
    write_json(output / f"{label}.json", record)
    print(f"{label}: {record['status']}; {json.dumps(metrics)}", flush=True)
    return record


def summarize(records):
    grouped = {}
    for record in records:
        key = (record["profile"], record["workload"])
        grouped.setdefault(key, []).append(record)
    result = []
    for (profile, workload), runs in grouped.items():
        valid = [r for r in runs if r["status"] == "ok" and not r["warmup"]]
        measured = [r for r in runs if not r["warmup"]]
        row = {
            "profile": profile, "workload": workload,
            "valid_runs": len(valid), "failed_runs": sum(r["status"] != "ok" for r in measured),
        }
        for key in METRICS:
            values = [r["metrics"][key] for r in valid if key in r["metrics"]]
            if values:
                row[f"{key}_count"] = len(values)
                row[f"{key}_median"] = statistics.median(values)
                row[f"{key}_min"] = min(values)
                row[f"{key}_max"] = max(values)
        for key in ("min_available_kib", "pswpin_delta", "pswpout_delta"):
            values = [r.get("system_memory", {}).get(key) for r in valid]
            values = [value for value in values if value is not None]
            if values:
                row[key + "_min"] = min(values)
                row[key + "_max"] = max(values)
        result.append(row)
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("doctor", "bench", "long", "sweep"))
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=["baseline", "threads6", "io4", "passive", "passive-io4"])
    parser.add_argument("--workloads", nargs="+", choices=("bench", "long"),
                        default=["bench", "long"])
    parser.add_argument("--tokens", type=positive_int,
                        help="same generation limit for every selected workload")
    parser.add_argument("--timeout", type=positive_int, default=1800,
                        help="seconds per engine process, including initialization")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-gpu-monitor", action="store_true",
                        help="disable nvidia-smi sampling for overhead comparisons")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    env = dict(os.environ)
    env["COLI_TIMERS"] = "1"
    env["Q36_HEAT_WAIT"] = "1"
    if "N_NEW" in env:
        try:
            positive_int(env["N_NEW"])
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("N_NEW must be a positive integer")
    output = args.output or RESULTS / ("linux-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "system.json", diagnostics(env))
    print(f"Results: {output}", flush=True)
    if args.mode == "doctor":
        print("Read-only probes saved; unavailable commands have explicit errors in system.json.")
        return 0
    workloads = args.workloads if args.mode == "sweep" else [args.mode]
    profiles = list(dict.fromkeys(args.profiles)) if args.mode == "sweep" else ["baseline"]
    prompts = {w: Path(env["PROMPT"]) if env.get("PROMPT") else
               RESULTS / ("prompt-long.txt" if w == "long" else "prompt.txt") for w in workloads}
    for prompt in prompts.values():
        validate_run(env, prompt)
    records = []
    # Warm each profile/workload once, then interleave measured configurations.
    rounds = ([0] if not args.no_warmup else []) + list(range(1, args.repeats + 1))
    for repeat in rounds:
        order = profiles if repeat % 2 else list(reversed(profiles))
        for workload in workloads:
            for profile in order:
                run_env = {**env, **PROFILES[profile]}
                run_env["N_NEW"] = str(args.tokens or positive_int(
                    env.get("N_NEW", "48" if workload == "long" else "128")))
                label = f"{workload}-{profile}-r{repeat}"
                record = run_once(run_env, prompts[workload], output, label, args.timeout,
                                  monitor_gpu=not args.no_gpu_monitor)
                record.update(profile=profile, workload=workload, warmup=repeat == 0)
                records.append(record)
                write_json(output / f"{label}.json", record)
                write_json(output / "runs.json", records)
                rows = summarize(records)
                write_json(output / "summary.json", rows)
                fields = sorted({key for row in rows for key in row})
                with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                if record["status"] != "ok":
                    print(f"ERROR: stopping; inspect {output / (label + '.log')}", file=sys.stderr)
                    return record["exit_code"] if record["exit_code"] > 0 else 1
    print("Summary: " + str(output / "summary.csv"))
    for row in summarize(records):
        print(f"{row['workload']} / {row['profile']}: {row['valid_runs']} measured runs; "
              f"median TTFT={row.get('ttft_s_median', 'unavailable')} s; "
              f"step={row.get('step_ms_median', 'unavailable')} ms; "
              f"reported Speed={row.get('reported_speed_tps_median', 'unavailable')} tok/s")
    print("Reported Speed includes prefill; step_ms is not accepted-token throughput. "
          "History evolves; no caches were reset. No profile is automatically applied.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
