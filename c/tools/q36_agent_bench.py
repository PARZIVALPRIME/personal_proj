#!/usr/bin/env python3
"""Measure the offline coding-agent path: one persistent server, several streamed turns.

The engine keeps the conversation state between requests. A turn that resends the same
history should prefill only its new tail; this tool shows whether that actually happens
(the engine's [PREFIX] log lines) and what each turn costs: time to first token, time to
the first answer text, generation rate and total time.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from q36_bench import (PROFILES, controlled_heat, diagnostics, gpu_monitor,
                       profile_env, sample_system, selected_environment, sha256,
                       system_monitor, write_json)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark-results"
LAUNCHER = RESULTS / "turbo-qwen36.sh"
TASKS = (
    "In three short bullet points, say what the two headers above are responsible for.",
    "Write a C function `int kv_rows_needed(int len, int cap)` that returns how many rows a "
    "cache of capacity `cap` must keep for a sequence of `len` tokens: return 0 if either "
    "argument is non-positive, otherwise return their minimum. Code only.",
    "Review the pin pool code above for a concrete bug. Check the existing guards before "
    "proposing a fix; if no bug is established, say so instead of inventing one. "
    "Code if needed and one sentence only.",
    "Add a short doc comment to the function you wrote two answers ago. Code only.",
)
PREFIX_RE = re.compile(r"\[PREFIX\] (reusing (\d+) of (\d+) prompt tokens|no reuse[^\n]*)")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def http_json(url, timeout, api_key=None):
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8") or "{}")


def wait_ready(base, proc, timeout, api_key=None, poll=2.0):
    """Wait for /health ok; fail at once if the server process ends."""
    start = time.monotonic()
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during start")
        try:
            status, body = http_json(base + "/health", 5, api_key)
            if status == 200 and body.get("status") == "ok":
                return time.monotonic() - start
        except (OSError, ValueError):
            pass
        if time.monotonic() - start > timeout:
            raise RuntimeError(f"no healthy server within {timeout} s")
        time.sleep(poll)


def model_id(base, api_key=None):
    _, body = http_json(base + "/v1/models", 30, api_key)
    data = body.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError("/v1/models listed no model")
    return data[0]["id"]


def stream_turn(base, body, timeout, api_key=None):
    """One streamed chat completion; returns text, reasoning, usage and timings."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(base + "/v1/chat/completions", data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    sent = time.monotonic()
    first_output = first_content = None
    content, reasoning = [], []
    usage, finish, done = {}, None, False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if time.monotonic() - sent > timeout:
                raise RuntimeError(f"completion stream exceeded {timeout} s")
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            try:
                event = json.loads(payload)
            except ValueError as exc:
                raise RuntimeError("malformed JSON in completion stream") from exc
            if not isinstance(event, dict):
                raise RuntimeError("completion stream event is not an object")
            if isinstance(event.get("error"), dict):
                raise RuntimeError(event["error"].get("message", "server error"))
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not isinstance(choices, list):
                raise RuntimeError("completion stream choices is not an array")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise RuntimeError("completion stream choice is not an object")
                if choice.get("x_colibri_keepalive") is True:
                    continue
                reason = choice.get("finish_reason")
                if reason is not None and not isinstance(reason, str):
                    raise RuntimeError("completion stream finish reason is not text")
                finish = reason or finish
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise RuntimeError("completion stream delta is not an object")
                think = delta.get("reasoning_content")
                text = delta.get("content")
                if (think is not None and not isinstance(think, str)
                        or text is not None and not isinstance(text, str)):
                    raise RuntimeError("completion stream content/reasoning is not text")
                now = time.monotonic() - sent
                if think:
                    reasoning.append(think)
                    first_output = now if first_output is None else first_output
                if text:
                    content.append(text)
                    first_output = now if first_output is None else first_output
                    first_content = now if first_content is None else first_content
    total = time.monotonic() - sent
    completion = usage.get("completion_tokens")
    if not done or finish is None:
        raise RuntimeError("incomplete completion stream: missing finish reason or [DONE]")
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 1:
        raise RuntimeError("completion stream has no positive completion_tokens usage")
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < 1:
        raise RuntimeError("completion stream has no positive prompt_tokens usage")
    if first_output is None:
        raise RuntimeError("completion stream contains no generated output")
    rate = None
    if isinstance(completion, int) and completion > 1 and first_output is not None:
        rate = (completion - 1) / max(total - first_output, 1e-6)
    return {
        "content": "".join(content), "reasoning": "".join(reasoning),
        "ttft_s": first_output, "first_answer_s": first_content, "total_s": total,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
        "generation_tps": rate, "finish_reason": finish,
        "truncated": finish == "length",
    }


def prefix_events(log_text):
    events = []
    for match in PREFIX_RE.finditer(log_text):
        if match.group(2):
            events.append({"reused": int(match.group(2)), "prompt": int(match.group(3))})
        else:
            events.append({"reused": 0, "detail": match.group(1)})
    return events


def start_server(command, env, log_path):
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    log = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs), log
    except OSError:
        log.close()
        raise


def stop_server(proc, log):
    try:
        if proc.poll() is None:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    except ProcessLookupError:
        pass
    finally:
        log.close()


def build_messages(context_text):
    system = ("You are an offline coding assistant. Answer briefly and precisely.\n\n"
              "Repository context:\n" + context_text)
    return [{"role": "system", "content": system}]


def run_conversation(base, args, messages, model, turns=None):
    turns = [] if turns is None else turns
    for index, task in enumerate(TASKS[:args.turns], 1):
        messages.append({"role": "user", "content": task})
        body = {
            "model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": args.max_tokens, "enable_thinking": args.think,
        }
        if args.think:
            body["preserve_thinking_whitespace"] = True
        result = stream_turn(base, body, args.request_timeout, args.api_key)
        assistant = {"role": "assistant", "content": result["content"]}
        if args.think:
            # Sending the reasoning back keeps the history identical to the engine's state.
            assistant["reasoning_content"] = result["reasoning"]
        messages.append(assistant)
        result["turn"] = index
        turns.append(result)
        if result["truncated"]:
            print(f"WARNING: turn {index} reached max_tokens; answer is incomplete", file=sys.stderr)
        print(f"turn {index}: prompt={result['prompt_tokens']} ttft={_fmt(result['ttft_s'])} s "
              f"answer={_fmt(result['first_answer_s'])} s total={_fmt(result['total_s'])} s "
              f"tokens={result['completion_tokens']} rate={_fmt(result['generation_tps'])} tok/s "
              f"finish={result['finish_reason']}", flush=True)
    return turns


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def run_session(args, output, env=None):
    env = dict(os.environ) if env is None else dict(env)
    output.mkdir(parents=True, exist_ok=False)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "think": args.think,
              "max_tokens": args.max_tokens, "context": str(args.context.resolve()),
              "context_sha256": sha256(args.context), "environment": selected_environment(env),
              "starting_system": sample_system(), "turns": []}
    proc = log = None
    log_path = output / "server.log"
    code = 0
    try:
        with ExitStack() as monitors:
            if args.url:
                base = args.url.rstrip("/")
                record["server"] = "existing"
                wait_ready(base, None, args.start_timeout, args.api_key)
            else:
                base = f"http://127.0.0.1:{args.port}"
                env["COLI_PREFIX_LOG"] = "1"
                env["COLI_TIMERS"] = "1"
                command = args.server_command or ["bash", str(LAUNCHER), "serve", "--port", str(args.port)]
                record["server_command"] = command
                monitors.enter_context(gpu_monitor(output, "server", not args.no_gpu_monitor))
                proc, log = start_server(command, env, log_path)
                monitors.enter_context(system_monitor(output, proc.pid))
                record["startup_s"] = wait_ready(base, proc, args.start_timeout, args.api_key)
                print(f"server ready in {record['startup_s']:.1f} s", flush=True)
            model = model_id(base, args.api_key)
            record["model"] = model
            messages = build_messages(args.context.read_text(encoding="utf-8", errors="replace"))
            run_conversation(base, args, messages, model, record["turns"])
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        record["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        record["error"] = "interrupted by user; earlier completed turns retained"
        print(f"ERROR: {record['error']}", file=sys.stderr)
        code = 130
    finally:
        if proc is not None:
            stop_server(proc, log)
    record["ending_system"] = sample_system()
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        record["prefix_events"] = prefix_events(text)
        capacities = re.findall(r"expert cache \d+ -> (\d+) slots a layer", text)
        if capacities:
            record["cache_slots_per_layer"] = int(capacities[-1])
    turns = record["turns"]
    later = record.get("prefix_events", [])[1:]
    if proc is not None and len(turns) > 1:
        record["prefix_reuse_engaged"] = any(e["reused"] > 0 for e in later)
        record["prefix_reuse_all_followups"] = (
            len(later) == len(turns) - 1 and all(e["reused"] > 0 for e in later))
        if not record["prefix_reuse_all_followups"]:
            print("WARNING: no prefix reuse on one or more follow-ups; inspect server.log", file=sys.stderr)
            if args.require_prefix_reuse:
                record["error"] = record.get("error", "prefix reuse missing on a follow-up")
                code = 2 if code == 0 else code
    if any(t.get("truncated") for t in turns):
        record["error"] = record.get("error", "one or more answers reached the token limit")
        code = 2 if code == 0 else code
    record["exit_code"] = code
    write_json(output / "agent.json", record)
    print(f"Results: {output / 'agent.json'}")
    return code, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="use an already running server instead of starting one")
    parser.add_argument("--port", type=positive_int, default=8011)
    parser.add_argument("--turns", type=positive_int, default=4, choices=range(1, len(TASKS) + 1))
    parser.add_argument("--max-tokens", type=positive_int, default=512)
    parser.add_argument("--think", action="store_true", help="enable model thinking (slower answers)")
    parser.add_argument("--context", type=Path, default=RESULTS / "prompt-long.txt",
                        help="text put in the system prompt as repository context")
    parser.add_argument("--start-timeout", type=positive_int, default=900)
    parser.add_argument("--request-timeout", type=positive_int, default=1800)
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--require-prefix-reuse", action="store_true",
                        help="fail if any completed follow-up does not reuse its prefix")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        help="alternate isolated persistent conversations for these profiles")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--heat-seed", type=Path,
                        help="start each server with a private copy of this heat history")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)  # tests: replaces the launcher command
    args = parser.parse_args(argv)
    if not args.context.is_file():
        parser.error(f"context file not found: {args.context}")
    if args.heat_seed is not None and not args.heat_seed.is_file():
        parser.error("--heat-seed must name an existing heat history")
    if args.url and (args.profiles or args.heat_seed):
        parser.error("--profiles/--heat-seed require an owned server, not --url")
    if args.url and args.require_prefix_reuse:
        parser.error("--require-prefix-reuse requires an owned server's prefix logs")
    output = args.output or RESULTS / ("linux-agent-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not args.profiles and args.heat_seed is None:
        return run_session(args, output)[0]
    output.mkdir(parents=True, exist_ok=False)
    seed = args.heat_seed
    if seed is not None:
        shutil.copyfile(seed, output / "heat-seed.bin")
        seed = output / "heat-seed.bin"
    write_json(output / "system.json", diagnostics(os.environ))
    profiles = list(dict.fromkeys(args.profiles or ["baseline"]))
    env = dict(os.environ)
    records = []
    code = 0
    for repeat in range(1, args.repeats + 1):
        for profile in profiles if repeat % 2 else list(reversed(profiles)):
            label = f"r{repeat}-{profile}"
            with controlled_heat(profile_env(env, profile), output, label, seed) as run_env:
                status, record = run_session(args, output / label, run_env)
            record.update(profile=profile, repeat=repeat)
            records.append(record)
            write_json(output / "runs.json", records)
            if status:
                return status
    return code


if __name__ == "__main__":
    sys.exit(main())
