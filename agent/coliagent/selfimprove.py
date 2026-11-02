"""A controlled self-improvement cycle.

One cycle tests one hypothesis with one patch:

  1. refuse the patch if it touches a protected path (the policy, this module,
     .git, CI) - the agent cannot loosen its own rules or its own gate;
  2. create an isolated git worktree on a fresh branch agent/improve-<id>;
  3. measure the baseline metric there (if the hypothesis has a metric);
  4. apply the patch, run every gate command, measure the metric again;
  5. accept only if every gate passes and the metric moved by at least
     ``min_gain`` in the right direction (median of ``repeats`` samples);
  6. on acceptance commit to the branch (locally) and keep the branch for a
     person to review; on rejection delete the branch. The worktree is always
     removed, and every outcome is appended to a JSONL ledger.

It never merges, never touches the main checkout, and never pushes.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

from . import patch as patchmod
from .permissions import PermissionDenied, Policy
from .repo import RepoManager
from .state import gain_of
from .tools import CommandResult, ToolRunner


@dataclass
class Hypothesis:
    statement: str
    metric: Optional[str] = None            # e.g. "decode tok/s"
    metric_argv: Optional[List[str]] = None  # command that prints the metric
    metric_regex: str = r"([-+]?\d+(?:\.\d+)?)"
    metric_cwd: str = "."
    lower_is_better: bool = False
    min_gain: float = 0.02
    repeats: int = 3


@dataclass
class Gate:
    argv: List[str]
    cwd: str = "."
    timeout: Optional[float] = None


@dataclass
class Outcome:
    id: str
    accepted: bool
    reason: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    baseline: List[float] = field(default_factory=list)
    candidate: List[float] = field(default_factory=list)
    gain: Optional[float] = None
    gates: List[str] = field(default_factory=list)
    diff: str = ""


class ImprovementCycle:
    def __init__(self, repo: RepoManager, policy: Policy, gates: List[Gate], ledger: Optional[str] = None,
                 workdir: Optional[str] = None):
        self.repo = repo
        self.policy = policy
        self.gates = gates
        self.workdir = workdir or tempfile.gettempdir()
        if ledger is None:
            common = repo.git("rev-parse", "--git-common-dir").stdout.strip()
            common_path = Path(common) if Path(common).is_absolute() else Path(repo.root) / common
            ledger = str(common_path / "coliagent" / "ledger.jsonl")
        self.ledger = ledger

    def _metric(self, runner: ToolRunner, root: Path, h: Hypothesis) -> tuple:
        samples: List[float] = []
        results: List[CommandResult] = []
        rx = re.compile(h.metric_regex)
        for _ in range(max(1, h.repeats)):
            res = runner.run(h.metric_argv, cwd=str(root / h.metric_cwd))
            results.append(res)
            if not res.ok:
                return samples, results, f"metric command failed: {res.summary()}"
            m = rx.search(res.stdout + "\n" + res.stderr)
            if not m:
                return samples, results, f"metric regex {h.metric_regex!r} found nothing"
            samples.append(float(m.group(1)))
        return samples, results, ""

    def _record(self, out: Outcome, h: Hypothesis) -> None:
        p = Path(self.ledger)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = dataclasses.asdict(out)
        rec["hypothesis"] = dataclasses.asdict(h)
        rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["diff_lines"] = out.diff.count("\n")
        rec.pop("diff")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def run(self, h: Hypothesis, candidate: patchmod.Patch, cid: Optional[str] = None) -> Outcome:
        cid = cid or uuid.uuid4().hex[:8]
        branch = f"agent/improve-{cid}"
        if self.policy.touches_protected(candidate.paths):
            out = Outcome(cid, False, "patch touches a protected path")
            self._record(out, h)
            return out
        if h.metric_argv is None and h.metric:
            raise ValueError("a metric needs metric_argv")
        wt = Path(self.workdir) / f"coliagent-{cid}"
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        out = Outcome(cid, False, "", branch=branch)
        created = False
        try:
            self.repo.worktree_add(str(wt), branch)
            created = True
            wt_policy = dataclasses.replace(self.policy, root=str(wt.resolve()))
            # no .pyc: a patch that keeps a file's size within the same mtime
            # second would otherwise be masked by the baseline run's bytecode
            runner = ToolRunner(wt_policy, base_env={"PYTHONDONTWRITEBYTECODE": "1"})
            if h.metric_argv:
                out.baseline, _, err = self._metric(runner, wt, h)
                if err:
                    out.reason = "baseline " + err
                    return out
            try:
                out.diff = patchmod.apply(candidate, str(wt), wt_policy)
            except (patchmod.PatchError, PermissionDenied) as e:
                out.reason = f"patch rejected: {e}"
                return out
            for g in self.gates:
                res = runner.run(g.argv, cwd=str(wt / g.cwd), timeout=g.timeout)
                out.gates.append(res.summary())
                if not res.ok:
                    out.reason = f"gate failed: {res.summary()}\n{(res.stdout + res.stderr)[-2000:]}"
                    return out
            if h.metric_argv:
                out.candidate, _, err = self._metric(runner, wt, h)
                if err:
                    out.reason = "candidate " + err
                    return out
                out.gain = gain_of(median(out.baseline), median(out.candidate), h.lower_is_better)
                if out.gain < h.min_gain:
                    out.reason = f"{h.metric}: gain {out.gain:.2%} below the required {h.min_gain:.2%}"
                    return out
            wt_repo = RepoManager(runner, str(wt))
            msg = f"agent: {h.statement}\n\n{candidate.description}".rstrip()
            if out.gain is not None:
                msg += f"\n\n{h.metric}: median {median(out.baseline):g} -> {median(out.candidate):g} ({out.gain:+.2%})"
            out.commit = wt_repo.commit(msg, paths=candidate.paths)
            out.accepted = True
            out.reason = "all gates passed" + ("" if out.gain is None else f"; gain {out.gain:+.2%}")
            return out
        finally:
            if created:
                self.repo.worktree_remove(str(wt), None if out.accepted else branch)
            if not out.accepted:
                out.branch = None
            self._record(out, h)

    def history(self) -> List[Dict]:
        p = Path(self.ledger)
        if not p.is_file():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
