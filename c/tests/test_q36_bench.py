import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("q36_bench", ROOT / "c" / "tools" / "q36_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
BASH = shutil.which("bash")
if not BASH:
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
    BASH = str(candidate) if candidate.is_file() else None

OUTPUT = """TTFT: 1.256 s (time to first token)
[timers] step() total: 62.5 ms/token (outside the phases: 33.1)
[timers] expert kernel: fetch 8.22 | compute 20.72 ms/token
[qwen36 lookup] run: 1.10 tokens/forward
PEAK RSS: 12.50 GB
Expert cache hit rate: 96.4% (hit=41920 miss=1588)
Speed: 16.74 tok/s (7.6s for 128 tokens)
"""


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / "model with spaces"
        self.model.mkdir()
        (self.model / "experts.q36x").write_bytes(b"fixture")
        self.prompt = self.base / "prompt.txt"
        self.prompt.write_text("test prompt", encoding="utf-8")
        engine = self.base / "engine"
        engine.write_bytes(b"fixture engine identity")
        engine.chmod(0o755)
        self.env = {**os.environ, "SNAP": str(self.model), "COLI_ENGINE": str(engine)}

    def test_metrics_and_missing_values(self):
        values = bench.parse_metrics(OUTPUT)
        self.assertEqual(values["reported_speed_tps"], 16.74)
        self.assertEqual(values["step_ms"], 62.5)
        self.assertEqual(values["tokens_per_forward"], 1.10)
        self.assertEqual(bench.parse_metrics("engine failed"), {})

    def test_metrics_match_actual_post_xpack_log(self):
        log = ROOT / "benchmark-results" / "bench-turbo-1793731906.log"
        values = bench.parse_metrics(log.read_text(encoding="utf-8"))
        self.assertEqual(values["reported_speed_tps"], 16.74)
        self.assertEqual(values["ttft_s"], 1.256)
        self.assertEqual(values["compute_ms"], 20.72)

    def test_summary_excludes_warmup_and_failures(self):
        records = [
            dict(profile="baseline", workload="long", warmup=warmup, status=status,
                 metrics={"ttft_s": value})
            for warmup, status, value in [
                (True, "ok", 100), (False, "ok", 17), (False, "ok", 19),
                (False, "failed", 1),
            ]
        ]
        row = bench.summarize(records)[0]
        self.assertEqual(row["ttft_s_median"], 18)
        self.assertEqual(row["ttft_s_count"], 2)
        self.assertEqual(row["failed_runs"], 1)

    def test_validation_requires_pack_but_allows_explicit_unpacked(self):
        (self.model / "experts.q36x").unlink()
        with self.assertRaisesRegex(ValueError, "missing/empty"):
            bench.validate_run(self.env, self.prompt)
        bench.validate_run({**self.env, "Q36_XPACK": "0"}, self.prompt)
        with self.assertRaisesRegex(ValueError, "SNAP"):
            bench.validate_run({**self.env, "SNAP": ""}, self.prompt)

    def run_engine_fixture(self, script, timeout=5):
        real_popen = subprocess.Popen

        def fake_engine(command, **kwargs):
            return real_popen([sys.executable, "-c", script], **kwargs)

        with mock.patch.object(bench.subprocess, "Popen", side_effect=fake_engine), \
                contextlib.redirect_stdout(io.StringIO()):
            return bench.run_once(self.env, self.prompt, self.base, "run", timeout,
                                  monitor_gpu=False)

    def test_success_preserves_log_and_telemetry(self):
        record = self.run_engine_fixture(f"print({OUTPUT!r})")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["metrics"]["fetch_ms"], 8.22)
        self.assertIn("Speed:", (self.base / "run.log").read_text())
        self.assertTrue((self.base / "run.system.jsonl").read_text())
        self.assertEqual(len(record["engine_sha256"]), 64)

    def test_nonzero_engine_exit_is_not_hidden(self):
        record = self.run_engine_fixture("import sys; print('failed'); sys.exit(7)")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["exit_code"], 7)

    def test_zero_exit_without_metrics_is_failure(self):
        record = self.run_engine_fixture("print('no model')")
        self.assertEqual(record["status"], "failed")
        self.assertIn("required TTFT", record["error"])

    def test_timeout_stops_engine(self):
        record = self.run_engine_fixture("import time; time.sleep(60)", timeout=1)
        self.assertEqual(record["exit_code"], 124)
        self.assertTrue(record["timed_out"])

    def test_sweep_interleaves_and_preserves_overrides(self):
        calls = []

        def run(env, prompt, output, label, timeout, **kwargs):
            calls.append((label, env))
            return {"status": "ok", "metrics": {"step_ms": 60}, "exit_code": 0}

        out = self.base / "sweep"
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(bench, "diagnostics", return_value={}), \
                mock.patch.object(bench, "run_once", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            code = bench.main(["sweep", "--profiles", "baseline", "io4",
                               "--workloads", "bench", "--repeats", "3",
                               "--output", str(out), "--tokens", "256"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 8)
        self.assertEqual([x[0] for x in calls[2:6]], [
            "bench-baseline-r1", "bench-io4-r1", "bench-io4-r2", "bench-baseline-r2",
        ])
        self.assertTrue(all(env["N_NEW"] == "256" for _, env in calls))
        rows = json.loads((out / "summary.json").read_text())
        self.assertTrue(all(row["valid_runs"] == 3 for row in rows))
        self.assertTrue((out / "summary.csv").is_file())

    def test_main_propagates_engine_failure_and_saves_summary(self):
        out = self.base / "failed"
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(bench, "diagnostics", return_value={}), \
                mock.patch.object(bench, "run_once", return_value={
                    "status": "failed", "metrics": {}, "exit_code": 7,
                }), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = bench.main(["bench", "--output", str(out), "--no-warmup"])
        self.assertEqual(code, 7)
        self.assertTrue((out / "runs.json").is_file())

    def test_probe_errors_are_explicit(self):
        with mock.patch.object(bench.subprocess, "run", side_effect=FileNotFoundError("missing")):
            self.assertIn("missing", bench.probe(["not-installed"])["error"])

    def test_doctor_cli_works_without_model_or_engine(self):
        out = self.base / "doctor"
        result = subprocess.run(
            [sys.executable, str(ROOT / "c" / "tools" / "q36_bench.py"),
             "doctor", "--output", str(out)],
            env={**os.environ, "SNAP": str(self.base / "missing"),
                 "COLI_ENGINE": str(self.base / "missing-engine")},
            text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((out / "system.json").read_text())
        self.assertIn("probes", data)
        self.assertIn("shader_sha256", data)

    def test_gpu_monitor_is_stopped(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        with mock.patch.object(bench.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(bench.subprocess, "Popen", return_value=proc):
            with bench.gpu_monitor(self.base, "gpu-test", True) as info:
                pass
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        self.assertTrue(info["stopped_by_runner"])

    def test_gpu_monitor_failure_is_explicit(self):
        proc = mock.Mock()
        proc.poll.return_value = 9
        with mock.patch.object(bench.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(bench.subprocess, "Popen", return_value=proc), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with bench.gpu_monitor(self.base, "gpu-test", True) as info:
                pass
        self.assertEqual(info["exit_code"], 9)
        self.assertIn("exited 9", errors.getvalue())

    def test_environment_capture_does_not_dump_unrelated_secrets(self):
        values = bench.selected_environment({"Q36_IO_THREADS": "8", "SECRET_KEY": "secret"})
        self.assertEqual(values, {"Q36_IO_THREADS": "8"})

    def test_linux_counters(self):
        self.assertEqual(bench.counter("MemAvailable:   1234 kB\n", "MemAvailable"), 1234)
        self.assertEqual(bench.counter("pswpout 42\n", "pswpout"), 42)
        self.assertIsNone(bench.counter(None, "pswpout"))

    def test_memory_pressure_is_not_hidden(self):
        self.assertEqual(bench.memory_warnings({
            "min_available_kib": 1024 * 1024, "pswpin_delta": 0, "pswpout_delta": 0,
        }), [])
        warnings = bench.memory_warnings({
            "min_available_kib": 83000, "pswpin_delta": 10, "pswpout_delta": 100,
        })
        self.assertEqual(len(warnings), 2)
        self.assertIn("512 MiB", warnings[0])

    def test_ram_override_comparison_is_rejected(self):
        with mock.patch.dict(os.environ, {**self.env, "RAM_GB": "14"}), \
                mock.patch.object(bench, "diagnostics", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "unset RAM_GB"):
                bench.main(["sweep", "--profiles", "baseline", "ram80",
                            "--output", str(self.base / "override")])

    def test_stop_on_pressure_saves_run_and_stops_before_next_profile(self):
        out = self.base / "pressure"
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(bench, "diagnostics", return_value={}), \
                mock.patch.object(bench, "run_once", return_value={
                    "status": "ok", "metrics": {}, "exit_code": 0,
                    "warnings": ["memory pressure"],
                }) as run, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = bench.main(["sweep", "--profiles", "baseline", "threads6",
                               "--stop-on-pressure", "--output", str(out)])
        self.assertEqual(code, 2)
        self.assertEqual(run.call_count, 1)
        self.assertTrue((out / "runs.json").is_file())


@unittest.skipUnless(BASH, "Bash is required for launcher integration tests")
class LauncherTests(unittest.TestCase):
    def test_quality_mode_exports_model_and_propagates_engine_status(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model = base / "model with spaces"
            model.mkdir()
            engine = base / "fake-engine.sh"
            engine.write_text(
                '#!/usr/bin/env bash\n'
                'printf "SNAP=%s\\nSKIP=%s\\nBITS=%s\\nVULKAN=%s\\nTIER=%s\\n" '
                '"$SNAP" "${Q36_SKIP_MISS-unset}" "${COLI_DENSE_BITS-unset}" '
                '"$COLI_VULKAN" "$COLI_VK_TIER"\nexit 7\n', encoding="utf-8")
            engine.chmod(0o755)
            env = {**os.environ, "SNAP": model.as_posix(), "COLI_ENGINE": engine.as_posix(),
                   "Q36_SKIP_MISS": "0.5", "COLI_DENSE_BITS": "4", "GPU": "0", "LOSSY": "0"}
            result = subprocess.run(
                [BASH, (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix(), "xpack"],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn(f"SNAP={model.as_posix()}", result.stdout)
            self.assertIn("SKIP=unset", result.stdout)
            self.assertIn("BITS=unset", result.stdout)
            self.assertIn("VULKAN=0", result.stdout)
            self.assertIn("TIER=0", result.stdout)

    def test_invalid_gpu_is_rejected(self):
        result = subprocess.run(
            [BASH, (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix(), "bench"],
            env={**os.environ, "GPU": "mistyped"}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU must", result.stderr)

    def test_lossy_mode_is_rejected(self):
        result = subprocess.run(
            [BASH, (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix(), "bench"],
            env={**os.environ, "GPU": "vk", "LOSSY": "1"}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("quality mode only", result.stderr)

    def test_default_model_path_is_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model = base / "colibri-models" / "qwen36-35b"
            model.mkdir(parents=True)
            engine = base / "engine.sh"
            engine.write_text('#!/usr/bin/env bash\nprintf "%s" "$SNAP"\n', encoding="utf-8")
            engine.chmod(0o755)
            env = {**os.environ, "HOME": base.as_posix(), "GPU": "vk", "LOSSY": "0",
                   "COLI_ENGINE": engine.as_posix()}
            env.pop("SNAP", None)
            result = subprocess.run(
                [BASH, (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix(), "xpack"],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = result.stdout
            if os.name == "nt":
                actual = subprocess.run(
                    [BASH, "-c", 'cygpath -m "$1"', "test", actual],
                    text=True, capture_output=True, check=True).stdout.strip()
            self.assertEqual(Path(actual), model)


if __name__ == "__main__":
    unittest.main()
