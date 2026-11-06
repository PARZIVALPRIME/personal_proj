import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("q36_mtp_setup", ROOT / "c" / "tools" / "q36_mtp_setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)

INDEX = {"weight_map": {
    "model.language_model.layers.0.mlp.gate.weight": "model-00001-of-00026.safetensors",
    "mtp.fc.weight": "model-00026-of-00026.safetensors",
    "mtp.layers.0.mlp.experts.down_proj": "model-00026-of-00026.safetensors",
    "mtp.norm.weight": "model-00025-of-00026.safetensors",
}}


class FakeRunner:
    """Records commands; 'downloads' write files; the converter writes the head."""

    def __init__(self, write_head=True):
        self.commands = []
        self.write_head = write_head

    def __call__(self, command, **kwargs):
        command = [str(part) for part in command]
        self.commands.append(command)
        if command[1:2] == ["-c"] and "hf_hub_download" in command[2]:
            dest = Path(command[4])
            for name in command[5:]:
                content = json.dumps(INDEX) if name.endswith("index.json") else "x"
                (dest / name).write_text(content, encoding="utf-8")
        if any(part.endswith("convert_qwen36.py") for part in command):
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            if self.write_head:
                (out / "model-mtp.safetensors").write_bytes(b"head" * 100)


class MtpSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "models" / "qwen36-35b"
        self.base.mkdir(parents=True)
        (self.base / "qwen36_meta.json").write_text("{}", encoding="utf-8")
        self.source = self.root / "source"
        self.venv = self.root / "venv"

    def run_setup(self, runner, probe=lambda command: True, *extra):
        argv = ["--base", str(self.base), "--source", str(self.source), "--venv", str(self.venv), *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = setup.main(argv, run_cmd=runner, probe=probe)
        return code, out.getvalue(), err.getvalue()

    def test_full_flow_downloads_only_mtp_shards_and_cleans_up(self):
        runner = FakeRunner()
        code, out, err = self.run_setup(runner)
        self.assertEqual(code, 0, err)
        downloads = [c[5:] for c in runner.commands if c[1:2] == ["-c"]]
        self.assertEqual(downloads[0], ["config.json", "model.safetensors.index.json"])
        self.assertEqual(downloads[1], ["model-00025-of-00026.safetensors",
                                        "model-00026-of-00026.safetensors"])
        convert = next(c for c in runner.commands if c[1].endswith("convert_qwen36.py"))
        self.assertIn("--mtp-only", convert)
        self.assertEqual(Path(convert[convert.index("--base-container") + 1]), self.base.resolve())
        out_dir = Path(str(self.base.resolve()) + "-mtp")
        self.assertEqual(Path(convert[convert.index("--out") + 1]), out_dir)
        self.assertTrue((out_dir / "model-mtp.safetensors").is_file())
        self.assertFalse((self.source / "model-00026-of-00026.safetensors").exists())
        self.assertIn("Q36_MTP=1", out)

    def test_second_run_does_nothing(self):
        self.assertEqual(self.run_setup(FakeRunner())[0], 0)
        runner = FakeRunner()
        code, out, _ = self.run_setup(runner)
        self.assertEqual(code, 0)
        self.assertEqual(runner.commands, [])
        self.assertIn("already set up", out)

    def test_missing_dependencies_are_installed_cpu_only(self):
        runner = FakeRunner()
        code, _, err = self.run_setup(runner, lambda command: False)
        self.assertEqual(code, 0, err)
        installs = [c for c in runner.commands if "install" in c]
        self.assertTrue(any("torch" in c and setup.TORCH_INDEX in c for c in installs))
        self.assertTrue(any("safetensors" in c and "huggingface_hub" in c for c in installs))

    def test_keep_source_keeps_shards(self):
        code, _, err = self.run_setup(FakeRunner(), lambda command: True, "--keep-source")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.source / "model-00026-of-00026.safetensors").exists())

    def test_refusals(self):
        out_dir = Path(str(self.base.resolve()) + "-mtp")
        out_dir.mkdir()
        (out_dir / "leftover").write_text("x")
        code, _, err = self.run_setup(FakeRunner())
        self.assertEqual(code, 1)
        self.assertIn("not empty", err)
        (out_dir / "leftover").unlink()
        (self.base / "qwen36_meta.json").unlink()
        code, _, err = self.run_setup(FakeRunner())
        self.assertEqual(code, 1)
        self.assertIn("not a converted", err)

    def test_converter_without_output_is_an_error(self):
        code, _, err = self.run_setup(FakeRunner(write_head=False))
        self.assertEqual(code, 1)
        self.assertIn("missing or empty", err)

    def test_low_disk_is_refused_before_downloading(self):
        runner = FakeRunner()
        with mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=1024 ** 3)):
            code, _, err = self.run_setup(runner)
        self.assertEqual(code, 1)
        self.assertIn("8 GB", err)
        self.assertEqual(runner.commands, [])

    def test_index_without_mtp_tensors(self):
        index = self.root / "index.json"
        index.write_text(json.dumps({"weight_map": {"model.x": "a.safetensors"}}))
        with self.assertRaisesRegex(RuntimeError, "no mtp"):
            setup.mtp_shards(index)


if __name__ == "__main__":
    unittest.main()
