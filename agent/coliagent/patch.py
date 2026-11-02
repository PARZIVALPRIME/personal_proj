"""Exact, all-or-nothing multi-file edits.

An edit names a file, an exact old string that must occur exactly once, and its
replacement (an empty old string creates a new file). Every edit of a patch is
checked against the policy and the current contents before anything is written;
if a write fails part way, the files already written are restored.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .permissions import Policy


class PatchError(Exception):
    pass


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str


@dataclass
class Patch:
    edits: List[Edit] = field(default_factory=list)
    description: str = ""

    @property
    def paths(self) -> List[str]:
        seen: List[str] = []
        for e in self.edits:
            if e.path not in seen:
                seen.append(e.path)
        return seen

    @classmethod
    def from_dict(cls, d: Dict) -> "Patch":
        edits = [Edit(str(e["path"]), str(e.get("old", "")), str(e["new"])) for e in d.get("edits", [])]
        return cls(edits, str(d.get("description", "")))

    def to_dict(self) -> Dict:
        return {"description": self.description,
                "edits": [{"path": e.path, "old": e.old, "new": e.new} for e in self.edits]}


def _read(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def plan(patch: Patch, root: str, policy: Optional[Policy] = None) -> Dict[str, tuple]:
    """Validate and compute {path: (before, after)} without writing anything."""
    if not patch.edits:
        raise PatchError("empty patch")
    base = Path(root)
    result: Dict[str, tuple] = {}
    for e in patch.edits:
        if policy is not None:
            policy.check_write(base / e.path)
        before, cur = result.get(e.path, (None, None))
        if e.path not in result:
            before = _read(base / e.path)
            cur = before
        if e.old == "":
            if cur is not None:
                raise PatchError(f"{e.path}: create edit but the file already exists")
            cur = e.new
        else:
            if cur is None:
                raise PatchError(f"{e.path}: file does not exist")
            # match on LF so a CRLF file still matches an LF old string
            crlf = "\r\n" in cur
            body = cur.replace("\r\n", "\n") if crlf else cur
            old, new = e.old.replace("\r\n", "\n"), e.new.replace("\r\n", "\n")
            n = body.count(old)
            if n != 1:
                raise PatchError(f"{e.path}: old text matches {n} times (needs exactly 1)")
            body = body.replace(old, new, 1)
            cur = body.replace("\n", "\r\n") if crlf else body
        result[e.path] = (before, cur)
    return result


def apply(patch: Patch, root: str, policy: Optional[Policy] = None) -> str:
    """Apply atomically; return the unified diff of what changed."""
    changes = plan(patch, root, policy)
    base = Path(root)
    written: List[str] = []
    try:
        for path, (_, after) in changes.items():
            _write(base / path, after)
            written.append(path)
    except Exception:
        for path in written:
            before = changes[path][0]
            if before is None:
                (base / path).unlink(missing_ok=True)
            else:
                _write(base / path, before)
        raise
    return unified_diff(changes)


def revert(changes: Dict[str, tuple], root: str) -> None:
    base = Path(root)
    for path, (before, _) in changes.items():
        if before is None:
            (base / path).unlink(missing_ok=True)
        else:
            _write(base / path, before)


def unified_diff(changes: Dict[str, tuple]) -> str:
    out: List[str] = []
    for path, (before, after) in changes.items():
        a = (before or "").splitlines(keepends=True)
        b = (after or "").splitlines(keepends=True)
        out.extend(difflib.unified_diff(a, b, "/dev/null" if before is None else f"a/{path}", f"b/{path}"))
    return "".join(line if line.endswith("\n") else line + "\n" for line in out)
