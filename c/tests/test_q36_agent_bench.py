import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c" / "tools"))
SPEC = importlib.util.spec_from_file_location("q36_agent_bench", ROOT / "c" / "tools" / "q36_agent_bench.py")
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)

FAKE_SERVER = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1]); mode = sys.argv[2]; record = sys.argv[3]
if mode == "exit":
    sys.exit(5)
started = time.monotonic()
count = {"n": 0}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            ready = time.monotonic() - started > 0.3
            self._json(200 if ready else 503, {"status": "ok" if ready else "restarting"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "qwen36-test"}]})
        else:
            self._json(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(record, "a") as stream:
            stream.write(json.dumps(body) + "\n")
        count["n"] += 1
        prompt = 100 * count["n"]
        if count["n"] == 1 or mode == "noreuse":
            print(f"[PREFIX] no reuse: held=0 cap=16384 prompt={prompt}", file=sys.stderr, flush=True)
        else:
            print(f"[PREFIX] reusing {prompt - 40} of {prompt} prompt tokens (60%)",
                  file=sys.stderr, flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        def send(event):
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        send({"choices": [{"delta": {"reasoning_content": "."}, "x_colibri_keepalive": True}]})
        time.sleep(0.05)
        if body.get("enable_thinking"):
            send({"choices": [{"delta": {"reasoning_content": "plan"}}]})
        for piece in ("int ", "f(void);"):
            send({"choices": [{"delta": {"content": piece}}]})
        send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        send({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": 5}})
        self.wfile.write(b"data: [DONE]\n\n")

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class AgentBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.script = self.base / "fake_server.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.context = self.base / "context.txt"
        self.context.write_text("int kv(void);", encoding="utf-8")
        self.requests = self.base / "requests.jsonl"

    def run_bench(self, mode, *extra):
        port = free_port()
        out = self.base / f"out-{mode}"
        argv = ["--port", str(port), "--context", str(self.context), "--output", str(out),
                "--start-timeout", "30", "--request-timeout", "30", *extra,
                "--server-command", sys.executable, str(self.script), str(port), mode,
                str(self.requests)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                mock.patch.object(agent, "gpu_monitor", return_value=contextlib.nullcontext()):
            code = agent.main(argv)
        record = json.loads((out / "agent.json").read_text())
        return code, record, stderr.getvalue()

    def test_reuse_is_detected_and_turn_metrics_recorded(self):
        code, record, errors = self.run_bench("reuse", "--turns", "3")
        self.assertEqual(code, 0, record.get("error"))
        self.assertEqual(len(record["turns"]), 3)
        self.assertTrue(record["prefix_reuse_engaged"])
        self.assertEqual(record["prefix_events"][1], {"reused": 160, "prompt": 200})
        first = record["turns"][0]
        self.assertEqual(first["content"], "int f(void);")
        self.assertEqual(first["reasoning"], "")   # the keepalive ping is not reasoning
        self.assertEqual(first["completion_tokens"], 5)
        self.assertEqual(first["finish_reason"], "stop")
        self.assertIsNotNone(first["ttft_s"])
        self.assertIsNotNone(first["generation_tps"])
        self.assertNotIn("WARNING", errors)

    def test_history_is_resent_and_thinking_is_off_by_default(self):
        self.run_bench("reuse", "--turns", "2")
        bodies = [json.loads(line) for line in self.requests.read_text().splitlines()]
        self.assertEqual(len(bodies), 2)
        self.assertFalse(bodies[0]["enable_thinking"])
        self.assertTrue(bodies[0]["stream"])
        self.assertEqual(bodies[0]["temperature"], 0)
        roles = [m["role"] for m in bodies[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(bodies[1]["messages"][2]["content"], "int f(void);")
        self.assertNotIn("reasoning_content", bodies[1]["messages"][2])
        self.assertIn("int kv(void);", bodies[0]["messages"][0]["content"])

    def test_thinking_sends_reasoning_back(self):
        code, record, _ = self.run_bench("reuse", "--turns", "2", "--think")
        self.assertEqual(code, 0, record.get("error"))
        bodies = [json.loads(line) for line in self.requests.read_text().splitlines()]
        self.assertTrue(bodies[0]["enable_thinking"])
        self.assertEqual(bodies[1]["messages"][2]["reasoning_content"], "plan")

    def test_missing_reuse_is_reported(self):
        code, record, errors = self.run_bench("noreuse", "--turns", "2")
        self.assertEqual(code, 0)
        self.assertFalse(record["prefix_reuse_engaged"])
        self.assertIn("no prefix reuse", errors)

    def test_required_reuse_fails_missing_followups(self):
        code, record, _ = self.run_bench("noreuse", "--turns", "2", "--require-prefix-reuse")
        self.assertEqual(code, 2)
        self.assertFalse(record["prefix_reuse_all_followups"])

    def test_server_exit_during_start_fails(self):
        code, record, errors = self.run_bench("exit", "--turns", "1")
        self.assertEqual(code, 1)
        self.assertIn("exited with code 5", record["error"])
        self.assertIn("ERROR", errors)

    def test_existing_server_url_mode(self):
        port = free_port()
        proc = subprocess.Popen([sys.executable, str(self.script), str(port), "reuse",
                                 str(self.requests)], stderr=subprocess.DEVNULL)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.terminate)
        out = self.base / "url"
        with contextlib.redirect_stdout(io.StringIO()):
            code = agent.main(["--url", f"http://127.0.0.1:{port}", "--turns", "1",
                               "--context", str(self.context), "--output", str(out),
                               "--start-timeout", "30"])
        record = json.loads((out / "agent.json").read_text())
        self.assertEqual(code, 0, record.get("error"))
        self.assertEqual(record["server"], "existing")
        self.assertEqual(record["model"], "qwen36-test")

    def test_prefix_event_parser(self):
        events = agent.prefix_events(
            "[PREFIX] no reuse: held=0 cap=8 prompt=10\n"
            "[PREFIX] reusing 900 of 1000 prompt tokens (90%)\n")
        self.assertEqual(events, [{"reused": 0, "detail": "no reuse: held=0 cap=8 prompt=10"},
                                  {"reused": 900, "prompt": 1000}])

    def test_stream_rejects_malformed_truncated_or_missing_usage(self):
        for lines in (
            [b"data: nope\n"],
            [b'data: {"choices":"wrong"}\n'],
            [b'data: {"choices":[{"delta":{"content":7}}]}\n'],
            [b'data: {"choices":[{"delta":{"content":"x"}}]}\n'],
            [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n',
             b"data: [DONE]\n"],
        ):
            with self.subTest(lines=lines), mock.patch.object(
                    agent.urllib.request, "urlopen", return_value=contextlib.nullcontext(lines)):
                with self.assertRaises(RuntimeError):
                    agent.stream_turn("http://localhost", {}, 1)

    def test_length_finish_is_explicitly_marked(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}\n',
            b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(agent.urllib.request, "urlopen",
                               return_value=contextlib.nullcontext(lines)):
            self.assertTrue(agent.stream_turn("http://localhost", {}, 1)["truncated"])

    def test_generated_period_is_not_a_keepalive(self):
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"."},"x_colibri_keepalive":true}]}\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"."}}]}\n',
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: {"usage":{"prompt_tokens":5,"completion_tokens":3}}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(agent.urllib.request, "urlopen",
                               return_value=contextlib.nullcontext(lines)):
            self.assertEqual(agent.stream_turn("http://localhost", {}, 1)["reasoning"], ".")

    def test_prior_turns_survive_later_failure(self):
        turns = []
        args = mock.Mock(turns=2, max_tokens=8, request_timeout=1, api_key=None, think=False)
        completed = {"content": "ok", "reasoning": "", "truncated": False,
                     "prompt_tokens": 1, "ttft_s": 1, "first_answer_s": 1,
                     "total_s": 2, "completion_tokens": 2, "generation_tps": 1,
                     "finish_reason": "stop"}
        with mock.patch.object(agent, "stream_turn", side_effect=[
                completed, RuntimeError("broken stream")]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "broken stream"):
                agent.run_conversation("http://localhost", args, [], "test", turns)
        self.assertEqual(len(turns), 1)

    def test_profile_runs_alternate_without_mutating_environment(self):
        records = []
        seed = self.base / "seed"
        seed.write_bytes(b"initial")

        def session(args, output, env=None):
            records.append((output.name, dict(env)))
            self.assertEqual(Path(env["Q36_HEAT"]).read_bytes(), b"initial")
            Path(env["Q36_HEAT"]).write_bytes(b"changed")
            return 0, {"turns": []}

        with mock.patch.object(agent, "run_session", side_effect=session), \
                mock.patch.object(agent, "diagnostics", return_value={}):
            before = dict(os.environ)
            code = agent.main(["--context", str(self.context), "--output", str(self.base / "profiles"),
                               "--profiles", "baseline", "io4", "--repeats", "2",
                               "--heat-seed", str(seed)])
        self.assertEqual(code, 0)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual([r[0] for r in records],
                         ["r1-baseline", "r1-io4", "r2-io4", "r2-baseline"])
        self.assertEqual(seed.read_bytes(), b"initial")


if __name__ == "__main__":
    unittest.main()
