#!/usr/bin/env python3
"""Measure the offline coding-agent path: one persistent server, several streamed turns.

The engine keeps the conversation state between requests. A turn that resends the same
history should prefill only its new tail; this tool shows whether that actually happens
(the engine's [PREFIX] log lines) and what each turn costs: time to first token, time to
the first answer text, generation rate and total time.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark-results"
LAUNCHER = RESULTS / "turbo-qwen36.sh"
TASKS = (
    "In three short bullet points, say what the two headers above are responsible for.",
    "Write a C function `int kv_rows_needed(int len, int cap)` that returns how many rows a "
    "cache of capacity `cap` must keep for a sequence of `len` tokens. Code only.",
    "Point out the single riskiest line in the pin pool code above and show a corrected "
    "version of it. Code and one sentence only.",
    "Add a short doc comment to the function you wrote two answers ago. Code only.",
)
PREFIX_RE = re.compile(r"\[PREFIX\] (reusing (\d+) of (\d+) prompt tokens|no reuse[^\n]*)")
PING = {"", "."}


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
    usage, finish = {}, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if isinstance(event.get("error"), dict):
                raise RuntimeError(event["error"].get("message", "server error"))
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                think = delta.get("reasoning_content")
                text = delta.get("content")
                now = time.monotonic() - sent
                if think and think not in PING:
                    reasoning.append(think)
                    first_output = now if first_output is None else first_output
                if text:
                    content.append(text)
                    first_output = now if first_output is None else first_output
                    first_content = now if first_content is None else first_content
    total = time.monotonic() - sent
    completion = usage.get("completion_tokens")
    rate = None
    if isinstance(completion, int) and completion > 1 and first_output is not None:
        rate = (completion - 1) / max(total - first_output, 1e-6)
    return {
        "content": "".join(content), "reasoning": "".join(reasoning),
        "ttft_s": first_output, "first_answer_s": first_content, "total_s": total,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
        "generation_tps": rate, "finish_reason": finish,
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


def run_conversation(base, args, messages, model):
    turns = []
    for index, task in enumerate(TASKS[:args.turns], 1):
        messages.append({"role": "user", "content": task})
        body = {
            "model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": args.max_tokens, "enable_thinking": args.think,
        }
        result = stream_turn(base, body, args.request_timeout, args.api_key)
        assistant = {"role": "assistant", "content": result["content"]}
        if args.think:
            # Sending the reasoning back keeps the history identical to the engine's state.
            assistant["reasoning_content"] = result["reasoning"]
        messages.append(assistant)
        result["turn"] = index
        turns.append(result)
        print(f"turn {index}: prompt={result['prompt_tokens']} ttft={_fmt(result['ttft_s'])} s "
              f"answer={_fmt(result['first_answer_s'])} s total={_fmt(result['total_s'])} s "
              f"tokens={result['completion_tokens']} rate={_fmt(result['generation_tps'])} tok/s "
              f"finish={result['finish_reason']}", flush=True)
    return turns


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


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
    parser.add_argument("--server-command", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)  # tests: replaces the launcher command
    args = parser.parse_args(argv)
    if not args.context.is_file():
        parser.error(f"context file not found: {args.context}")
    output = args.output or RESULTS / ("linux-agent-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "think": args.think,
              "max_tokens": args.max_tokens, "context": str(args.context.resolve())}
    proc = log = None
    log_path = output / "server.log"
    code = 0
    try:
        if args.url:
            base = args.url.rstrip("/")
            record["server"] = "existing"
            wait_ready(base, None, args.start_timeout, args.api_key)
        else:
            base = f"http://127.0.0.1:{args.port}"
            env = dict(os.environ)
            env["COLI_PREFIX_LOG"] = "1"
            command = args.server_command or ["bash", str(LAUNCHER), "serve", "--port", str(args.port)]
            record["server_command"] = command
            proc, log = start_server(command, env, log_path)
            record["startup_s"] = wait_ready(base, proc, args.start_timeout, args.api_key)
            print(f"server ready in {record['startup_s']:.1f} s", flush=True)
        model = model_id(base, args.api_key)
        record["model"] = model
        messages = build_messages(args.context.read_text(encoding="utf-8", errors="replace"))
        record["turns"] = run_conversation(base, args, messages, model)
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        record["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    finally:
        if proc is not None:
            stop_server(proc, log)
    if log_path.is_file():
        record["prefix_events"] = prefix_events(log_path.read_text(encoding="utf-8", errors="replace"))
    turns = record.get("turns") or []
    later = [e for e in record.get("prefix_events", [])[1:]]
    if proc is not None and len(turns) > 1:
        record["prefix_reuse_engaged"] = any(e["reused"] > 0 for e in later)
        if not record["prefix_reuse_engaged"]:
            print("WARNING: no prefix reuse logged after the first turn: each turn paid a full "
                  "prefill. See server.log ([PREFIX] lines).", file=sys.stderr)
    (output / "agent.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output / 'agent.json'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
