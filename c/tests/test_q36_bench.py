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

    def test_previous_profile_unsets_thread_override(self):
        env = bench.profile_env({"OMP_NUM_THREADS": "6", "Q36_RAM_FRAC": "0.80", "X": "1"},
                                "previous")
        self.assertNotIn("OMP_NUM_THREADS", env)
        self.assertEqual(env["Q36_RAM_FRAC"], "0.92")
        self.assertEqual(env["X"], "1")
        self.assertEqual(bench.profile_env({"A": "1"}, "baseline"), {"A": "1"})

    def test_vulkan_chain_metrics_and_profile_lines(self):
        log = ("[VK] qwen36 chain: 251 forwards, 20873 frames (189102 ops), 11966.7 ms waiting "
               "for the device, 17280.2 ms of routed experts on the host, 0 layer steps\n"
               "[VK] chain profile: 812.5 ms of device time | GEMV int8 400.0 ms (10)\n")
        values = bench.parse_metrics(log)
        self.assertEqual(values["vk_forwards"], 251)
        self.assertEqual(values["vk_wait_ms"], 11966.7)
        self.assertEqual(values["vk_host_experts_ms"], 17280.2)
        self.assertEqual(values["vk_profile_device_ms"], 812.5)

    def test_profile_lines_are_kept_in_run_record(self):
        record = self.run_engine_fixture(
            f"print({OUTPUT!r}); print('[VK] chain profile: 5.0 ms of device time | norm 5.0 ms (2)')")
        self.assertEqual(record["vk_profile"],
                         ["[VK] chain profile: 5.0 ms of device time | norm 5.0 ms (2)"])
        self.assertEqual(record["metrics"]["vk_profile_device_ms"], 5.0)

    def test_decode_speed_excludes_prompt_time(self):
        log = ROOT / "benchmark-results" / "bench-turbo-1793731906.log"
        values = bench.parse_metrics(log.read_text(encoding="utf-8"))
        # Speed: 16.74 tok/s (7.6s for 128 tokens), TTFT 1.256 s -> 127 / 6.344 s
        self.assertAlmostEqual(values["decode_tps"], 20.02, places=2)
        self.assertNotIn("decode_tps", bench.parse_metrics("Speed: 5.0 tok/s (2.0s for 10 tokens)"))
        self.assertNotIn("decode_tps", bench.parse_metrics(
            "TTFT: 3.000 s\nSpeed: 1.0 tok/s (2.0s for 2 tokens)"))

    def test_mtp_metrics(self):
        log = ("[qwen36 mtp] run: 1.62 tokens/forward (158 forwards per 255 tokens) | "
               "acceptance 81.3% (97/119 drafts) | gate on\n"
               "[qwen36 lookup] run: 1.62 tokens/forward (158 forwards per 255 tokens) | x\n")
        values = bench.parse_metrics(log)
        self.assertEqual(values["mtp_acceptance_pct"], 81.3)
        self.assertEqual(values["tokens_per_forward"], 1.62)

    def test_mtp_needs_the_head_file(self):
        env = {**self.env, "Q36_MTP": "1"}
        with self.assertRaisesRegex(ValueError, "model-mtp.safetensors"):
            bench.validate_run(env, self.prompt)
        (self.model / "model-mtp.safetensors").write_bytes(b"head")
        bench.validate_run(env, self.prompt)

    def test_mtp_profiles_run_from_the_mtp_directory(self):
        alt = self.base / (self.model.name + "-mtp")
        alt.mkdir()
        self.assertEqual(bench.profile_env(self.env, "mtp1")["SNAP"], str(self.model))
        (alt / "model-mtp.safetensors").write_bytes(b"head")
        self.assertEqual(bench.profile_env(self.env, "mtp1")["SNAP"], str(alt))
        self.assertEqual(bench.profile_env(self.env, "baseline")["SNAP"], str(self.model))
        (self.model / "model-mtp.safetensors").write_bytes(b"head")
        self.assertEqual(bench.profile_env(self.env, "mtp1")["SNAP"], str(self.model))

    def test_sweep_checks_every_profile_before_running(self):
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(bench, "diagnostics", return_value={}), \
                mock.patch.object(bench, "run_once") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "mtp-setup"):
                bench.main(["sweep", "--profiles", "baseline", "mtp1", "--workloads", "bench",
                            "--output", str(self.base / "precheck")])
        run.assert_not_called()

    def test_new_profiles_are_known_and_reproducible(self):
        for name in ("la2", "la2x8", "tierdec", "gemv2", "lookup7", "mtp1", "mtp2", "mtpauto"):
            self.assertIn(name, bench.PROFILES)
        self.assertEqual(bench.profile_settings("mtp2"), {"Q36_MTP": "1", "Q36_MTP_DRAFTS": "2"})
        self.assertEqual(bench.profile_settings("previous"),
                         {"OMP_NUM_THREADS": "auto", "Q36_RAM_FRAC": "0.92"})


def _row(profile, workload="bench", decode=20.0, ttft=1.3, runs=3, failed=0, avail=1_400_000):
    return {"profile": profile, "workload": workload, "valid_runs": runs, "failed_runs": failed,
            "decode_tps_median": decode, "ttft_s_median": ttft, "min_available_kib_min": avail}


class PickBestTests(unittest.TestCase):
    def test_fastest_qualifying_profile_wins(self):
        rows = [_row("baseline"), _row("la2", decode=21.0), _row("mtp1", decode=24.0)]
        best, reasons = bench.pick_best(rows)
        self.assertEqual(best, "mtp1")
        self.assertIn("qualifies", reasons["la2"])

    def test_small_gain_keeps_baseline(self):
        best, reasons = bench.pick_best([_row("baseline"), _row("gemv2", decode=20.4)])
        self.assertEqual(best, "baseline")
        self.assertIn("+2.0%", reasons["gemv2"])

    def test_failures_memory_and_long_regressions_are_rejected(self):
        rows = [_row("baseline"), _row("baseline", "long", decode=17.0, ttft=16.5),
                _row("tierdec", decode=25.0), _row("tierdec", "long", decode=15.0, ttft=16.5),
                _row("la2", decode=25.0), _row("la2", "long", decode=17.0, ttft=18.0),
                _row("io6", decode=26.0, failed=1),
                _row("threads7", decode=26.0, avail=100_000),
                _row("mtp1", decode=23.0), _row("mtp1", "long", decode=17.5, ttft=16.6)]
        best, reasons = bench.pick_best(rows)
        self.assertEqual(best, "mtp1")
        self.assertIn("long decode_tps", reasons["tierdec"])
        self.assertIn("long TTFT", reasons["la2"])
        self.assertIn("rejected", reasons["io6"])
        self.assertIn("rejected", reasons["threads7"])

    def test_missing_baseline_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            bench.pick_best([_row("la2")])

    def test_best_command_prints_exports_and_merges_saved_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            summary = base / "summary.json"
            summary.write_text(json.dumps([_row("baseline"), _row("mtp2", decode=26.0)]))
            conf = base / "conf" / "q36.env"
            conf.parent.mkdir()
            conf.write_text("Q36_LOOKAHEAD=2\nQ36_MTP_DRAFTS=1\n", encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = bench.main(["best", str(summary), "--save", str(conf)])
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().split(), ["export", "Q36_MTP=1", "export", "Q36_MTP_DRAFTS=2"])
            saved = bench.read_conf(conf)
            self.assertEqual(saved, {"Q36_LOOKAHEAD": "2", "Q36_MTP": "1", "Q36_MTP_DRAFTS": "2"})
            self.assertIn("Result: mtp2", err.getvalue())

    def test_best_command_without_winner_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp) / "summary.json"
            summary.write_text(json.dumps([_row("baseline"), _row("la2", decode=20.1)]))
            conf = Path(temp) / "q36.env"
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(bench.main(["best", str(summary), "--save", str(conf)]), 0)
            self.assertEqual(out.getvalue(), "")
            self.assertFalse(conf.exists())
            self.assertIn("keep the current settings", err.getvalue())


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

    def _engine_env(self, conf_text=None, **overrides):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model = base / "model"
            model.mkdir()
            engine = base / "engine.sh"
            engine.write_text(
                '#!/usr/bin/env bash\n'
                'printf "THREADS=%s\\nRAM=%s\\nIO=%s\\nLA=%s\\n" "${OMP_NUM_THREADS-unset}" '
                '"$Q36_RAM_FRAC" "$Q36_IO_THREADS" "$Q36_LOOKAHEAD"\n', encoding="utf-8")
            engine.chmod(0o755)
            conf = base / "q36.env"
            if conf_text is not None:
                conf.write_text(conf_text, encoding="utf-8")
            env = {**os.environ, "SNAP": model.as_posix(), "COLI_ENGINE": engine.as_posix(),
                   "GPU": "vk", "LOSSY": "0", "Q36_CONF": conf.as_posix()}
            for key in ("OMP_NUM_THREADS", "Q36_RAM_FRAC", "Q36_IO_THREADS", "Q36_LOOKAHEAD", "Q36_MTP"):
                env.pop(key, None)
            env.update(overrides)
            result = subprocess.run(
                [BASH, (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix(), "xpack"],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

    def test_saved_settings_apply_but_shell_values_win(self):
        conf = "# comment\nQ36_LOOKAHEAD=2\nOMP_NUM_THREADS=7\nbad key=1\n\nQ36_IO_THREADS=6"
        out = self._engine_env(conf)
        self.assertIn("LA=2", out)
        self.assertIn("THREADS=7", out)
        self.assertIn("IO=6", out)   # last line without a newline still counts
        out = self._engine_env(conf, Q36_LOOKAHEAD="1")
        self.assertIn("LA=1", out)
        out = self._engine_env(conf, Q36_CONF="none")
        self.assertIn("LA=1", out)
        self.assertIn("THREADS=6", out)

    def test_mtp_without_head_is_refused_and_switches_to_mtp_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "colibri-models" / "qwen36-35b").mkdir(parents=True)
            (base / "colibri-models" / "qwen36-35b-mtp").mkdir(parents=True)
            engine = base / "engine.sh"
            engine.write_text('#!/usr/bin/env bash\nprintf "%s" "$SNAP"\n', encoding="utf-8")
            engine.chmod(0o755)
            env = {**os.environ, "HOME": base.as_posix(), "COLI_ENGINE": engine.as_posix(),
                   "GPU": "vk", "LOSSY": "0", "Q36_MTP": "1", "Q36_CONF": "none"}
            env.pop("SNAP", None)
            script = (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix()
            result = subprocess.run([BASH, script, "xpack"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("model-mtp.safetensors", result.stderr)
            self.assertIn("qwen36-35b-mtp", result.stderr)
            self.assertIn("mtp-setup", result.stderr)
            (base / "colibri-models" / "qwen36-35b-mtp" / "model-mtp.safetensors").write_bytes(b"h")
            # SNAP unset, and SNAP pointing at the base model: both switch to the head's directory
            for extra in ({}, {"SNAP": (base / "colibri-models" / "qwen36-35b").as_posix() + "/"}):
                result = subprocess.run([BASH, script, "xpack"], env={**env, **extra},
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith("colibri-models/qwen36-35b-mtp"), result.stdout)

    def test_mtp_setup_and_best_are_dispatched(self):
        script = (ROOT / "benchmark-results" / "turbo-qwen36.sh").as_posix()
        env = {**os.environ, "Q36_CONF": "none"}
        result = subprocess.run([BASH, script, "mtp-setup", "--help"], env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--keep-source", result.stdout)
        result = subprocess.run([BASH, script, "best", "--help"], env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("summary", result.stdout)

    def test_measured_defaults_are_six_threads_and_smaller_cache(self):
        out = self._engine_env()
        self.assertIn("THREADS=6", out)
        self.assertIn("RAM=0.80", out)
        self.assertIn("IO=8", out)

    def test_explicit_overrides_and_auto_threads(self):
        out = self._engine_env(OMP_NUM_THREADS="auto", Q36_RAM_FRAC="0.92")
        self.assertIn("THREADS=unset", out)
        self.assertIn("RAM=0.92", out)
        self.assertIn("THREADS=7", self._engine_env(OMP_NUM_THREADS="7"))


if __name__ == "__main__":
    unittest.main()
