"""Task planning with explicit dependencies.

A language model may *propose* a plan (JSON); this module checks its structure
(unique ids, known dependencies, no cycles) before anything acts on it.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

STATUSES = ("pending", "in_progress", "done", "blocked", "failed")


class PlanError(Exception):
    pass


@dataclass
class Step:
    id: str
    title: str
    detail: str = ""
    deps: List[str] = field(default_factory=list)
    status: str = "pending"
    files: List[str] = field(default_factory=list)
    verify: List[List[str]] = field(default_factory=list)  # argv lists that prove the step


@dataclass
class Plan:
    goal: str
    steps: List[Step] = field(default_factory=list)

    def by_id(self) -> Dict[str, Step]:
        return {s.id: s for s in self.steps}

    def validate(self) -> None:
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate step ids")
        known = set(ids)
        for s in self.steps:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", s.id):
                raise PlanError(f"bad step id {s.id!r} (kebab-case)")
            if s.status not in STATUSES:
                raise PlanError(f"{s.id}: unknown status {s.status}")
            for d in s.deps:
                if d not in known:
                    raise PlanError(f"{s.id}: unknown dependency {d}")
            for v in s.verify:
                if not isinstance(v, list) or not v or not all(isinstance(a, str) for a in v):
                    raise PlanError(f"{s.id}: verify entries are argv lists")
        self.order()

    def order(self) -> List[str]:
        """Topological order; raises on a cycle."""
        steps = self.by_id()
        state: Dict[str, int] = {}
        out: List[str] = []

        def visit(i: str, path: List[str]) -> None:
            if state.get(i) == 2:
                return
            if state.get(i) == 1:
                raise PlanError("dependency cycle: " + " -> ".join(path + [i]))
            state[i] = 1
            for d in steps[i].deps:
                visit(d, path + [i])
            state[i] = 2
            out.append(i)

        for s in self.steps:
            visit(s.id, [])
        return out

    def ready(self) -> List[Step]:
        steps = self.by_id()
        return [s for s in self.steps if s.status == "pending"
                and all(steps[d].status == "done" for d in s.deps)]

    def mark(self, step_id: str, status: str) -> None:
        if status not in STATUSES:
            raise PlanError(f"unknown status {status}")
        self.by_id()[step_id].status = status

    def to_json(self) -> str:
        return json.dumps({"goal": self.goal, "steps": [asdict(s) for s in self.steps]}, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "Plan":
        if not isinstance(d, dict) or "steps" not in d:
            raise PlanError("a plan is an object with a 'steps' list")
        steps = []
        for raw in d["steps"]:
            if not isinstance(raw, dict) or "id" not in raw or "title" not in raw:
                raise PlanError("each step needs an id and a title")
            steps.append(Step(id=str(raw["id"]), title=str(raw["title"]), detail=str(raw.get("detail", "")),
                              deps=[str(x) for x in raw.get("deps", [])], status=str(raw.get("status", "pending")),
                              files=[str(x) for x in raw.get("files", [])],
                              verify=[list(map(str, v)) if isinstance(v, list) else v for v in raw.get("verify", [])]))
        p = cls(str(d.get("goal", "")), steps)
        p.validate()
        return p


PLAN_PROMPT = """You plan software changes. Reply with JSON only:
{"goal": str, "steps": [{"id": "kebab-id", "title": str, "detail": str,
 "deps": [ids], "files": [paths], "verify": [[argv...]]}]}
Each step must be small, testable, and list the commands that prove it."""


def extract_json(text: str) -> Dict:
    """The first JSON object in a model reply (tolerates ```json fences and chatter)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:text.rfind("}") + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise PlanError("no JSON object in the reply")


def propose(llm, goal: str, context: str = "") -> Plan:
    reply = llm.complete([{"role": "system", "content": PLAN_PROMPT},
                          {"role": "user", "content": f"Goal: {goal}\n\nContext:\n{context}"}])
    d = extract_json(reply)
    d.setdefault("goal", goal)
    return Plan.from_dict(d)
