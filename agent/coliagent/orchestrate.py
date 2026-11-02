"""Build and test orchestration, with test selection from changed files.

Project knowledge lives in a small JSON config rather than in code:

    {
      "build": [{"cwd": "c", "argv": ["make", "qwen36"]}],
      "tests": [
        {"pattern": "c/tests/test_*.py", "cwd": "c",
         "steps": [["python", "-m", "unittest", "tests.{stem}"]]},
        {"pattern": "c/tests/test_*.c", "cwd": "c",
         "steps": [["make", "tests/{stem}"], ["./tests/{stem}"]]}
      ],
      "full": [{"cwd": ".", "argv": ["python", "-m", "unittest", "discover"]}],
      "always": ["c/tests/test_openai_server.py"]
    }

A test is selected when it changed, when it imports/includes a changed file
(transitively), or when its name mentions a changed file's stem.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional

from .index import RepoIndex
from .tools import CommandResult, ToolRunner

DEFAULT_CONFIG: Dict = {
    "build": [],
    "tests": [{"pattern": "tests/test_*.py", "cwd": ".", "steps": [["python", "-m", "unittest", "tests.{stem}"]]},
              {"pattern": "**/tests/test_*.py", "cwd": "{parent2}",
               "steps": [["python", "-m", "unittest", "tests.{stem}"]]}],
    "full": [{"cwd": ".", "argv": ["python", "-m", "unittest", "discover"]}],
    "always": [],
}


def load_config(root: str, path: Optional[str] = None) -> Dict:
    p = Path(path) if path else Path(root) / ".coliagent.json"
    if p.is_file():
        cfg = json.loads(p.read_text(encoding="utf-8"))
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        return merged
    return dict(DEFAULT_CONFIG)


def _match(path: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])


def _fill(s: str, test: str) -> str:
    p = PurePosixPath(test)
    parent2 = p.parent.parent.as_posix() if len(p.parts) > 2 else "."
    return s.replace("{stem}", p.stem).replace("{path}", test).replace("{parent2}", parent2).replace("{parent}", p.parent.as_posix())


@dataclass
class TestRun:
    test: str
    results: List[CommandResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)


class Orchestrator:
    def __init__(self, runner: ToolRunner, root: str, config: Optional[Dict] = None):
        self.runner = runner
        self.root = Path(root).resolve()
        self.config = config if config is not None else load_config(root)

    def _spec(self, test: str) -> Optional[Dict]:
        for spec in self.config.get("tests", []):
            if _match(test, spec["pattern"]):
                return spec
        return None

    def is_test(self, path: str) -> bool:
        return self._spec(path) is not None

    def select_tests(self, changed: Iterable[str], index: Optional[RepoIndex] = None) -> List[str]:
        changed = [c.replace("\\", "/") for c in changed]
        idx = index or RepoIndex(str(self.root))
        tests = [f for f in idx.files if self.is_test(f)]
        chosen = {c for c in changed if c in tests}
        dependents = idx.dependents(changed)
        chosen.update(t for t in tests if t in dependents)
        stems = {PurePosixPath(c).stem.lower() for c in changed if not self.is_test(c)}
        for t in tests:
            name = PurePosixPath(t).stem.lower()
            if any(len(s) >= 3 and re.search(rf"(^|_){re.escape(s)}(_|$)", name) for s in stems):
                chosen.add(t)
        chosen.update(t for t in self.config.get("always", []) if t in tests)
        return sorted(chosen)

    def run_test(self, test: str) -> TestRun:
        spec = self._spec(test)
        run = TestRun(test)
        if spec is None:
            return run
        cwd = self.root / _fill(spec.get("cwd", "."), test)
        for step in spec["steps"]:
            res = self.runner.run([_fill(a, test) for a in step], cwd=str(cwd), timeout=spec.get("timeout"))
            run.results.append(res)
            if not res.ok:
                break
        return run

    def run_tests(self, tests: Iterable[str]) -> List[TestRun]:
        return [self.run_test(t) for t in tests]

    def _run_list(self, key: str) -> List[CommandResult]:
        out = []
        for item in self.config.get(key, []):
            res = self.runner.run(item["argv"], cwd=str(self.root / item.get("cwd", ".")), timeout=item.get("timeout"))
            out.append(res)
            if not res.ok:
                break
        return out

    def build(self) -> List[CommandResult]:
        return self._run_list("build")

    def full(self) -> List[CommandResult]:
        return self._run_list("full")
