"""Explicit project state and evidence-backed claims.

A claim moves up a ladder; every rung above "proposed" needs evidence of the
matching kind, so "it is faster" cannot be recorded without a measurement and
"the tests pass" cannot be recorded without a passing run:

  proposed       an idea or hypothesis; no evidence needed
  applied        a change exists        evidence: a non-empty diff
  built          it compiles            evidence: a successful build command
  tested         tests pass             evidence: successful test commands
  perf-verified  a metric improved      evidence: baseline and candidate samples
                                        on the same machine, plus the command
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

from .tools import CommandResult

LEVELS = ("proposed", "applied", "built", "tested", "perf-verified")


class EvidenceError(Exception):
    pass


@dataclass
class Claim:
    id: str
    statement: str
    level: str = "proposed"
    evidence: List[Dict] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


def _cmd_evidence(results: List[CommandResult], kind: str) -> Dict:
    if not results:
        raise EvidenceError(f"{kind} needs at least one command result")
    bad = [r.summary() for r in results if not r.ok]
    if bad:
        raise EvidenceError(f"{kind} evidence includes failures: {bad}")
    return {"kind": kind, "commands": [r.summary() for r in results], "at": time.time()}


@dataclass
class ProjectState:
    path: str
    claims: Dict[str, Claim] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "ProjectState":
        p = Path(path)
        st = cls(str(p))
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            st.claims = {k: Claim(**v) for k, v in data.get("claims", {}).items()}
            st.notes = list(data.get("notes", []))
        return st

    def save(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps({"claims": {k: asdict(v) for k, v in self.claims.items()},
                                   "notes": self.notes}, indent=2), encoding="utf-8")
        tmp.replace(p)

    def propose(self, cid: str, statement: str) -> Claim:
        c = self.claims.get(cid) or Claim(cid, statement)
        self.claims[cid] = c
        return c

    def _raise(self, cid: str, level: str, ev: Dict) -> Claim:
        c = self.claims[cid]
        if LEVELS.index(level) > LEVELS.index(c.level):
            c.level = level
        c.evidence.append(ev)
        c.updated = time.time()
        return c

    def applied(self, cid: str, diff: str) -> Claim:
        if not diff.strip():
            raise EvidenceError("applied needs a non-empty diff")
        return self._raise(cid, "applied", {"kind": "diff", "lines": diff.count("\n"), "at": time.time()})

    def built(self, cid: str, results: List[CommandResult]) -> Claim:
        return self._raise(cid, "built", _cmd_evidence(results, "build"))

    def tested(self, cid: str, results: List[CommandResult]) -> Claim:
        return self._raise(cid, "tested", _cmd_evidence(results, "test"))

    def perf_verified(self, cid: str, metric: str, baseline: List[float], candidate: List[float],
                      lower_is_better: bool, min_gain: float, results: List[CommandResult],
                      machine: str = "") -> Claim:
        if len(baseline) < 1 or len(candidate) < 1:
            raise EvidenceError("perf-verified needs baseline and candidate samples")
        _cmd_evidence(results, "benchmark")
        b, c = median(baseline), median(candidate)
        gain = gain_of(b, c, lower_is_better)
        if gain < min_gain:
            raise EvidenceError(f"{metric}: gain {gain:.3%} is below the required {min_gain:.3%}")
        return self._raise(cid, "perf-verified", {"kind": "metric", "metric": metric, "baseline": baseline,
                                                  "candidate": candidate, "gain": gain, "machine": machine,
                                                  "commands": [r.summary() for r in results], "at": time.time()})

    def report(self) -> str:
        lines = []
        for c in sorted(self.claims.values(), key=lambda c: (-LEVELS.index(c.level), c.id)):
            lines.append(f"[{c.level:>13}] {c.id}: {c.statement}")
        return "\n".join(lines) or "(no claims)"


def gain_of(baseline: float, candidate: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline if lower_is_better else (candidate - baseline) / baseline
