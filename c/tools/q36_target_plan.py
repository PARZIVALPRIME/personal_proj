#!/usr/bin/env python3
"""Memory, placement and decode-time estimate for qwen36 on a small machine.

Targets the configuration the engine work in docs/q36-rtx3050-engineering.md is
about: Qwen3.6-35B-A3B int4 (group 64) on a Ryzen 7 5800HS (8C/16T, AVX2,
DDR4), an RTX 3050 Laptop with 4 GB and 16 GB of RAM, Windows 11. Every number
below is an *estimate* from the model's geometry and the hardware figures
given on the command line. None of them is a measurement, and the defaults for
bandwidths are assumptions you should replace with your own (``coli tune`` and
the engine's ``[place] probe`` / ``cache auto-sized`` lines measure them).

What it answers:
  * how many expert slots per layer the engine's auto cap (cap 0) will pick,
    mirroring qwen36_cap_for_ram (88% of available RAM, KV reserve, top-k floor);
  * whether the dense trunk fits the card under the tier's default allowance
    (free VRAM - 1 GiB), which the CUDA chain needs whole;
  * a bandwidth-bound decode estimate per token, split into trunk, cached
    experts and experts streamed from the SSD;
  * the environment to launch with, optionally as a PowerShell script.

Pure functions, no engine and no model needed: ``python tools/q36_target_plan.py``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

GIB = float(1 << 30)


@dataclass
class Geometry:
    """Qwen3.6-35B-A3B text model. Override with --config (an HF config.json)."""
    hidden: int = 2048
    layers: int = 40
    full_attention_interval: int = 4
    experts: int = 256
    topk: int = 8
    moe_inter: int = 512
    shared_inter: int = 512
    q_heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    attn_output_gate: bool = True
    dn_k_heads: int = 16
    dn_v_heads: int = 32
    dn_k_dim: int = 128
    dn_v_dim: int = 128
    vocab: int = 248320
    group_size: int = 64

    @property
    def attn_layers(self) -> int:
        return self.layers // self.full_attention_interval if self.full_attention_interval > 0 else 0

    @property
    def dn_layers(self) -> int:
        return self.layers - self.attn_layers

    @classmethod
    def from_config(cls, path: Path) -> "Geometry":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cfg.get("text_config", cfg)
        g = cls()
        pick = {
            "hidden": "hidden_size", "layers": "num_hidden_layers",
            "full_attention_interval": "full_attention_interval",
            "experts": "num_experts", "topk": "num_experts_per_tok",
            "moe_inter": "moe_intermediate_size",
            "shared_inter": "shared_expert_intermediate_size",
            "q_heads": "num_attention_heads", "kv_heads": "num_key_value_heads",
            "head_dim": "head_dim", "attn_output_gate": "attn_output_gate",
            "dn_k_heads": "linear_num_key_heads", "dn_v_heads": "linear_num_value_heads",
            "dn_k_dim": "linear_key_head_dim", "dn_v_dim": "linear_value_head_dim",
            "vocab": "vocab_size",
        }
        for attr, key in pick.items():
            if key in cfg and cfg[key] is not None:
                setattr(g, attr, type(getattr(g, attr))(cfg[key]))
        types = cfg.get("layer_types")
        if isinstance(types, list) and types:
            n_full = sum(1 for t in types if t == "full_attention")
            if n_full:
                g.full_attention_interval = max(1, round(len(types) / n_full))
        return g


@dataclass
class Hardware:
    """The target laptop. Bandwidths are assumptions, not measurements."""
    ram_gb: float = 16.0
    os_reserved_gb: float = 4.5        # Windows 11 + a browser/IDE after boot
    vram_gb: float = 4.0
    vram_display_gb: float = 0.3       # Optimus laptop: the iGPU drives the display
    physical_cores: int = 8
    ram_bw_gbs: float = 27.0           # effective DDR4-3200 dual channel for a GEMV
    gpu_bw_gbs: float = 130.0          # effective of 192 GB/s peak on a 3050 Laptop
    ssd_gbs: float = 2.0               # effective for ~1.6 MB random reads (NVMe Gen3)
    dense_resident_gb: float = 2.9     # engine's own report on the 35B ("dense resident")


@dataclass
class Plan:
    geometry: Dict
    hardware: Dict
    ctx: int
    kv_bytes_per_token: float
    kv_reserve_gb: float
    expert_gb: float
    slot_gb: float
    all_experts_gb: float
    available_gb: float
    budget_gb: float
    cap: int
    cache_fraction: float
    trunk_device_gb: float
    vram_allowance_gb: float
    trunk_fits: bool
    expert_vram_gb: float
    est_hit_rate: float
    ms_trunk_gpu: float
    ms_trunk_cpu: float
    ms_experts_ram: float
    ms_experts_ssd: float
    ms_token_gpu_trunk: float
    ms_token_cpu_trunk: float
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def kv_bytes_per_token(g: Geometry, kv_elem_bytes: int = 2) -> float:
    """K and V of every full-attention layer for one position (DeltaNet holds a fixed state)."""
    return 2.0 * g.kv_heads * g.head_dim * kv_elem_bytes * g.attn_layers


def expert_bytes(g: Geometry) -> float:
    """One routed expert in int4 with a float32 scale per group of group_size inputs."""
    n = 3.0 * g.hidden * g.moe_inter
    return n / 2.0 + n / g.group_size * 4.0


def engine_slot_gb(g: Geometry) -> float:
    """qwen36_cap_for_ram's per-slot estimate (int4 / xf_mode), in GB (1e9)."""
    return (g.hidden * g.moe_inter * 1.5 + (2 * g.moe_inter + g.hidden) * 4) / 1e9


def trunk_device_bytes(g: Geometry) -> float:
    """int8 matrices the CUDA chain needs on the card: projections, shared expert,
    router and lm_head (the embedding stays on the host). Scales are per row."""
    h = g.hidden
    q_out = g.q_heads * g.head_dim * (2 if g.attn_output_gate else 1)
    attn = (q_out + 2 * g.kv_heads * g.head_dim) * h + h * g.q_heads * g.head_dim
    qkv = 2 * g.dn_k_heads * g.dn_k_dim + g.dn_v_heads * g.dn_v_dim
    z = g.dn_v_heads * g.dn_v_dim
    dn = (qkv + z + 2 * g.dn_v_heads) * h + h * z
    per_layer = 3 * h * g.shared_inter + g.experts * h + h
    total = g.attn_layers * attn + g.dn_layers * dn + g.layers * per_layer + g.vocab * h
    return total * 1.003   # per-row float scales, about 4 bytes per 2048


def cap_for_ram(g: Geometry, avail_gb: float, reserve_gb: float) -> int:
    """Mirror of qwen36_cap_for_ram with the auto budget (resident + 0.88 * avail)."""
    room = avail_gb * 0.88 - max(reserve_gb, 0.0)
    slot = engine_slot_gb(g)
    derived = int(room / slot / max(g.layers, 1)) if room > 0 and slot > 0 else 0
    return min(max(derived, 1, g.topk), g.experts) if g.experts > 0 else max(derived, 1)


def build(g: Geometry, hw: Hardware, ctx: int = 32768, kv_elem_bytes: int = 2,
          hit_rate: Optional[float] = None) -> Plan:
    kv_tok = kv_bytes_per_token(g, kv_elem_bytes)
    kv_res = kv_tok * ctx / GIB
    e_bytes = expert_bytes(g)
    avail = max(hw.ram_gb - hw.os_reserved_gb - hw.dense_resident_gb, 0.0)
    cap = cap_for_ram(g, avail, kv_res)
    frac = cap / g.experts if g.experts else 1.0
    # Uniform routing makes an LRU of `cap` of `experts` hit cap/experts of the time.
    # Real routing is skewed, so this is a floor, not a forecast.
    hit = frac if hit_rate is None else min(max(hit_rate, 0.0), 1.0)
    trunk = trunk_device_bytes(g)
    vram_free = max(hw.vram_gb - hw.vram_display_gb, 0.0)
    allowance = max(vram_free - 1.0, 0.0)
    trunk_gb = trunk / GIB
    fits = trunk_gb <= allowance
    per_tok_experts = g.topk * g.layers * e_bytes
    ms_tg = trunk / (hw.gpu_bw_gbs * 1e9) * 1e3
    ms_tc = trunk / (hw.ram_bw_gbs * 1e9) * 1e3
    ms_er = per_tok_experts * hit / (hw.ram_bw_gbs * 1e9) * 1e3
    ms_es = per_tok_experts * (1.0 - hit) / (hw.ssd_gbs * 1e9) * 1e3
    # missed experts are computed from RAM too, once read
    ms_miss_compute = per_tok_experts * (1.0 - hit) / (hw.ram_bw_gbs * 1e9) * 1e3
    plan = Plan(
        geometry=asdict(g), hardware=asdict(hw), ctx=ctx,
        kv_bytes_per_token=kv_tok, kv_reserve_gb=kv_res,
        expert_gb=e_bytes / 1e9, slot_gb=engine_slot_gb(g),
        all_experts_gb=e_bytes * g.experts * g.layers / GIB,
        available_gb=avail, budget_gb=avail * 0.88, cap=cap, cache_fraction=frac,
        trunk_device_gb=trunk_gb, vram_allowance_gb=allowance, trunk_fits=fits,
        expert_vram_gb=max(allowance - trunk_gb, 0.0) if fits else allowance,
        est_hit_rate=hit, ms_trunk_gpu=ms_tg, ms_trunk_cpu=ms_tc,
        ms_experts_ram=ms_er + ms_miss_compute, ms_experts_ssd=ms_es,
        ms_token_gpu_trunk=ms_tg + ms_er + ms_miss_compute + ms_es,
        ms_token_cpu_trunk=ms_tc + ms_er + ms_miss_compute + ms_es,
    )
    plan.env = recommend_env(plan, hw)
    plan.notes = notes_for(plan, g, hw)
    return plan


def recommend_env(p: Plan, hw: Hardware) -> Dict[str, str]:
    env = {
        "COLI_CUDA": "1",
        "COLI_CUDA_CHAIN": "1",
        "Q36_MAXT": str(p.ctx),
        "Q36_KV": "f16",
        "OMP_NUM_THREADS": str(hw.physical_cores),
        "COLI_TRUNK_PROBE_WARM_MS": "250",
    }
    if not p.trunk_fits:
        # The tier keeps 1 GiB of headroom; a numeric CUDA_EXPERT_GB replaces it.
        need = p.trunk_device_gb + 0.25
        vram_free = max(hw.vram_gb - hw.vram_display_gb, 0.0)
        if need <= vram_free - 0.35:
            env["CUDA_EXPERT_GB"] = f"{need:.2f}"
    return env


def notes_for(p: Plan, g: Geometry, hw: Hardware) -> List[str]:
    n = []
    if p.all_experts_gb > hw.ram_gb - hw.os_reserved_gb - hw.dense_resident_gb:
        n.append(f"all routed experts take {p.all_experts_gb:.1f} GiB: they cannot all be in RAM, "
                 f"the cache holds {p.cap}/{g.experts} per layer and the rest streams from the SSD")
    if p.cap <= g.topk:
        n.append("the cache is at its top-k floor: expect most experts to come from the SSD; "
                 "close other programs or lower Q36_MAXT")
    if p.trunk_fits:
        n.append(f"the trunk ({p.trunk_device_gb:.2f} GiB) fits the tier's allowance "
                 f"({p.vram_allowance_gb:.2f} GiB): the CUDA chain can run")
    else:
        n.append(f"the trunk ({p.trunk_device_gb:.2f} GiB) exceeds the default allowance "
                 f"({p.vram_allowance_gb:.2f} GiB = free - 1 GiB): the chain stays off unless "
                 "CUDA_EXPERT_GB raises it (see env)")
    n.append(f"KV at {p.ctx} positions: {p.kv_reserve_gb:.2f} GiB host (float16), "
             "reserved out of the expert cache by the engine")
    if p.trunk_fits and p.expert_vram_gb > 0:
        n.append(f"about {int(p.expert_vram_gb * GIB / (p.expert_gb * 1e9))} experts fit the card beside the "
                 "trunk (not counted in the hit rate)")
    n.append("hit rate assumes uniform routing (a floor); a HEAT_FILE and warm cache do better")
    return n


def render_text(p: Plan) -> str:
    rows = [
        ("context", f"{p.ctx} positions"),
        ("KV cache", f"{p.kv_bytes_per_token / 1024:.1f} KiB/token, {p.kv_reserve_gb:.2f} GiB reserved"),
        ("expert", f"{p.expert_gb * 1e3:.2f} MB int4 gs64; all experts {p.all_experts_gb:.1f} GiB"),
        ("RAM available after load", f"{p.available_gb:.1f} GiB (budget 88%: {p.budget_gb:.1f})"),
        ("auto cap", f"{p.cap} slots/layer ({p.cache_fraction:.0%} of experts)"),
        ("trunk on device", f"{p.trunk_device_gb:.2f} GiB; allowance {p.vram_allowance_gb:.2f} GiB; "
                            f"fits: {'yes' if p.trunk_fits else 'no'}"),
        ("hit rate (assumed)", f"{p.est_hit_rate:.0%}"),
        ("trunk per token", f"GPU {p.ms_trunk_gpu:.1f} ms, CPU {p.ms_trunk_cpu:.1f} ms"),
        ("experts per token", f"RAM {p.ms_experts_ram:.1f} ms + SSD {p.ms_experts_ssd:.1f} ms"),
        ("token (trunk on GPU)", f"{p.ms_token_gpu_trunk:.0f} ms -> {1e3 / p.ms_token_gpu_trunk:.1f} tok/s"),
        ("token (trunk on CPU)", f"{p.ms_token_cpu_trunk:.0f} ms -> {1e3 / p.ms_token_cpu_trunk:.1f} tok/s"),
    ]
    w = max(len(k) for k, _ in rows)
    out = ["qwen36 target plan (estimates, not measurements)"]
    out += [f"  {k.ljust(w)}  {v}" for k, v in rows]
    out.append("  notes:")
    out += [f"    - {s}" for s in p.notes]
    out.append("  env:")
    out += [f"    {k}={v}" for k, v in p.env.items()]
    return "\n".join(out)


def render_ps1(p: Plan, model_dir: str, launcher: str) -> str:
    lines = [
        "# Generated by c/tools/q36_target_plan.py: launch qwen36 with the planned environment.",
        "# Estimates, not measurements: read the engine's '[qwen36] cache auto-sized' and",
        "# '[place] probe' lines on the first run, and adjust.",
        "param(",
        f"    [string]$Model = '{model_dir}',",
        f"    [string]$Coli = '{launcher}',",
        "    [ValidateSet('serve','chat','plan','doctor')][string]$Mode = 'serve'",
        ")",
        "$ErrorActionPreference = 'Stop'",
    ]
    for k, v in p.env.items():
        lines.append(f"if (-not $env:{k}) {{ $env:{k} = '{v}' }}")
    lines += [
        "Get-ChildItem env: | Where-Object { $_.Name -match '^(COLI_|Q36_|CUDA_EXPERT_GB|OMP_NUM_THREADS)' } |",
        "    Sort-Object Name | ForEach-Object { Write-Host (\"{0}={1}\" -f $_.Name, $_.Value) }",
        "& $Coli $Mode --model $Model",
        "exit $LASTEXITCODE",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, help="HF config.json of the model (default: Qwen3.6-35B-A3B)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kv", choices=["f16", "f32"], default="f16")
    ap.add_argument("--hit-rate", type=float, help="expert cache hit rate to assume (0..1)")
    hw = Hardware()
    for name, val in asdict(hw).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(val), default=val)
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    ap.add_argument("--ps1", type=Path, help="write a PowerShell launch script here")
    ap.add_argument("--model-dir", default=r"D:\colibri-models\qwen36-35b")
    ap.add_argument("--coli", default="coli", help="the coli launcher to call from the script")
    a = ap.parse_args(argv)
    g = Geometry.from_config(a.config) if a.config else Geometry()
    hw = Hardware(**{k: getattr(a, k) for k in asdict(hw)})
    p = build(g, hw, ctx=a.ctx, kv_elem_bytes=2 if a.kv == "f16" else 4, hit_rate=a.hit_rate)
    if a.kv == "f32":
        p.env["Q36_KV"] = "f32"
    print(json.dumps(asdict(p), indent=2) if a.json else render_text(p))
    if a.ps1:
        with open(a.ps1, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(render_ps1(p, a.model_dir, a.coli))
        print(f"wrote {a.ps1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
