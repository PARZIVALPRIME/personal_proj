"""Compiler and test log parsing into structured diagnostics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Diagnostic:
    tool: str
    severity: str  # error | warning | note | fail
    path: Optional[str]
    line: Optional[int]
    col: Optional[int]
    message: str


_GCC = re.compile(r"^(?P<path>[^\s:][^:]*?|[A-Za-z]:[^:]+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
                  r"(?P<sev>fatal error|error|warning|note):\s*(?P<msg>.+)$")
_MSVC = re.compile(r"^(?P<path>[^()]+)\((?P<line>\d+)(?:,(?P<col>\d+))?\)\s*:\s*"
                   r"(?P<sev>fatal error|error|warning)\s+(?P<code>[A-Z]+\d+)\s*:\s*(?P<msg>.+)$")
_LD = re.compile(r"(?:undefined reference to|unresolved external symbol)\s*[`'\"]?(?P<sym>[^`'\"\s]+)")
_PY_FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)')
_PY_EXC = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|AssertionError)(?::\s*(?P<msg>.*))?$")
_UNITTEST = re.compile(r"^(?P<kind>FAIL|ERROR):\s+(?P<test>\S+)\s+\((?P<cls>[^)]+)\)")
_PYTEST = re.compile(r"^FAILED\s+(?P<id>\S+?)(?:\s+-\s+(?P<msg>.*))?$")
_CTEST = re.compile(r"^\s*FAIL[:\s]+(?P<msg>.+)$")


def parse(log: str) -> List[Diagnostic]:
    out: List[Diagnostic] = []
    last_frame = None
    for raw in log.splitlines():
        line = raw.rstrip()
        m = _MSVC.match(line)
        if m:
            sev = "error" if "error" in m["sev"] else "warning"
            out.append(Diagnostic("msvc", sev, m["path"].strip(), int(m["line"]),
                                  int(m["col"]) if m["col"] else None, f"{m['code']}: {m['msg']}"))
            continue
        m = _GCC.match(line)
        if m:
            sev = "error" if "error" in m["sev"] else m["sev"]
            out.append(Diagnostic("cc", sev, m["path"], int(m["line"]),
                                  int(m["col"]) if m["col"] else None, m["msg"]))
            continue
        m = _LD.search(line)
        if m:
            out.append(Diagnostic("ld", "error", None, None, None, f"unresolved symbol {m['sym']}"))
            continue
        m = _PY_FRAME.match(line)
        if m:
            last_frame = (m["path"], int(m["line"]))
            continue
        m = _UNITTEST.match(line)
        if m:
            out.append(Diagnostic("unittest", "fail", None, None, None, f"{m['kind']} {m['cls']}.{m['test']}"))
            continue
        m = _PYTEST.match(line)
        if m:
            out.append(Diagnostic("pytest", "fail", m["id"].split("::")[0], None, None,
                                  f"{m['id']}" + (f": {m['msg']}" if m["msg"] else "")))
            continue
        m = _PY_EXC.match(line)
        if m and last_frame:
            out.append(Diagnostic("python", "error", last_frame[0], last_frame[1], None,
                                  m["exc"] + (f": {m['msg']}" if m["msg"] else "")))
            last_frame = None
            continue
        m = _CTEST.match(line)
        if m and not line.lstrip().startswith("FAILED"):
            out.append(Diagnostic("test", "fail", None, None, None, m["msg"].strip()))
    return out


def errors(log: str) -> List[Diagnostic]:
    return [d for d in parse(log) if d.severity in ("error", "fail")]


def summarize(diags: List[Diagnostic], limit: int = 20) -> str:
    lines = []
    for d in diags[:limit]:
        loc = f"{d.path}:{d.line}" if d.path and d.line else (d.path or "-")
        lines.append(f"[{d.tool} {d.severity}] {loc} {d.message}")
    if len(diags) > limit:
        lines.append(f"... {len(diags) - limit} more")
    return "\n".join(lines)
