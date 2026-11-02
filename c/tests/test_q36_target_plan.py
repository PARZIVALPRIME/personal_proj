"""tools/q36_target_plan.py: the small-machine estimator mirrors the engine's rules."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools import q36_target_plan as tp  # noqa: E402


class GeometryTest(unittest.TestCase):
    def test_default_is_the_35b_a3b(self):
        g = tp.Geometry()
        self.assertEqual((g.layers, g.attn_layers, g.dn_layers), (40, 10, 30))
        # the probe's dn_qkv is [8192 x 2048] on the 35B (qwen36.c trunk_probe_gpu_wins)
        self.assertEqual(2 * g.dn_k_heads * g.dn_k_dim + g.dn_v_heads * g.dn_v_dim, 8192)

    def test_config_overrides_and_layer_types(self):
        cfg = {"text_config": {"hidden_size": 1024, "num_hidden_layers": 8, "num_experts": 64,
                               "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            g = tp.Geometry.from_config(p)
        self.assertEqual((g.hidden, g.layers, g.experts, g.attn_layers), (1024, 8, 64, 2))


class CapTest(unittest.TestCase):
    def test_kv_reserve_lowers_the_cap(self):
        g = tp.Geometry()
        self.assertLess(tp.cap_for_ram(g, 8.0, 0.65), tp.cap_for_ram(g, 8.0, 0.0))

    def test_floor_is_topk_and_ceiling_is_experts(self):
        g = tp.Geometry()
        self.assertEqual(tp.cap_for_ram(g, 0.0, 0.0), g.topk)
        self.assertEqual(tp.cap_for_ram(g, 1000.0, 0.0), g.experts)

    def test_matches_engine_formula(self):
        # qwen36_cap_for_ram: (avail * 0.88 - reserve) / slot_gb / layers
        g = tp.Geometry()
        slot = (2048 * 512 * 1.5 + (1024 + 2048) * 4) / 1e9
        self.assertEqual(tp.cap_for_ram(g, 6.0, 0.65), int((6.0 * 0.88 - 0.65) / slot / 40))

    def test_kv_per_token(self):
        g = tp.Geometry()
        self.assertEqual(tp.kv_bytes_per_token(g, 2), 2 * 2 * 256 * 2 * 10)
        self.assertEqual(tp.kv_bytes_per_token(g, 4), 2 * tp.kv_bytes_per_token(g, 2))


class PlanTest(unittest.TestCase):
    def test_target_trunk_fits_a_4gb_card(self):
        p = tp.build(tp.Geometry(), tp.Hardware())
        self.assertTrue(p.trunk_fits)
        self.assertNotIn("CUDA_EXPERT_GB", p.env)
        self.assertLess(p.ms_token_gpu_trunk, p.ms_token_cpu_trunk)
        self.assertEqual(p.env["OMP_NUM_THREADS"], "8")

    def test_tight_card_raises_the_allowance_only_when_it_can(self):
        p = tp.build(tp.Geometry(), tp.Hardware(vram_gb=3.0, vram_display_gb=0.2))
        self.assertFalse(p.trunk_fits)
        self.assertIn("CUDA_EXPERT_GB", p.env)
        p = tp.build(tp.Geometry(), tp.Hardware(vram_gb=2.0))
        self.assertNotIn("CUDA_EXPERT_GB", p.env)

    def test_hit_rate_drives_the_ssd_time(self):
        g, hw = tp.Geometry(), tp.Hardware()
        cold = tp.build(g, hw, hit_rate=0.0)
        warm = tp.build(g, hw, hit_rate=1.0)
        self.assertEqual(warm.ms_experts_ssd, 0.0)
        self.assertGreater(cold.ms_experts_ssd, 0.0)

    def test_ps1_keeps_user_env(self):
        p = tp.build(tp.Geometry(), tp.Hardware())
        s = tp.render_ps1(p, r"D:\m", "coli")
        self.assertIn("if (-not $env:COLI_CUDA) { $env:COLI_CUDA = '1' }", s)
        self.assertIn("& $Coli $Mode --model $Model", s)

    def test_cli_json(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(tp.main(["--json", "--ctx", "8192"]), 0)
        self.assertEqual(json.loads(buf.getvalue())["ctx"], 8192)


if __name__ == "__main__":
    unittest.main()
