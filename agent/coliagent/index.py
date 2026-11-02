"""Repository index: inventory, symbols, text search, include/import graph.

Regex based on purpose: it is fast, dependency free, and good enough to point
the agent (or a person) at the right file. It is not a compiler front end.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

SKIP_DIRS = {".git", "__pycache__", "node_modules", "build", "dist", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
TEXT_EXT = {".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".js", ".ts", ".tsx", ".go", ".rs", ".md",
            ".txt", ".toml", ".json", ".yml", ".yaml", ".ps1", ".sh", ".cmake", ""}
MAX_BYTES = 4 * 1024 * 1024

_SYM = {
    ".py": [(re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M), "function"),
            (re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M), "class")],
    ".c": [(re.compile(r"^(?!\s)(?:static\s+|inline\s+|extern\s+)*[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{", re.M), "function"),
           (re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)", re.M), "macro"),
           (re.compile(r"^\s*typedef\s+struct\s*\w*\s*\{[^}]*\}\s*([A-Za-z_]\w*)\s*;", re.M | re.S), "type")],
    ".js": [(re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M), "function"),
            (re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.M), "class")],
}
_SYM[".h"] = _SYM[".c"] + [(re.compile(r"^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*;", re.M), "prototype")]
_SYM[".cu"] = _SYM[".cc"] = _SYM[".cpp"] = _SYM[".c"]
_SYM[".ts"] = _SYM[".tsx"] = _SYM[".js"]
_C_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "do"}

_PY_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+(\([^)]*\)|[^\n#]+)|import\s+([\w.]+))", re.M)
_C_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    path: str
    line: int


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    text: str


class RepoIndex:
    def __init__(self, root: str, files: Optional[Iterable[str]] = None):
        self.root = Path(root).resolve()
        self.files: List[str] = sorted(files) if files is not None else self._walk()
        self._text: Dict[str, str] = {}

    def _walk(self) -> List[str]:
        out = []
        for d, dirs, names in os.walk(self.root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for n in names:
                out.append((Path(d) / n).relative_to(self.root).as_posix())
        return sorted(out)

    def text(self, rel: str) -> str:
        if rel not in self._text:
            p = self.root / rel
            if Path(rel).suffix.lower() not in TEXT_EXT or not p.is_file() or p.stat().st_size > MAX_BYTES:
                self._text[rel] = ""
            else:
                try:
                    self._text[rel] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    self._text[rel] = ""
        return self._text[rel]

    def symbols(self, rel: Optional[str] = None) -> List[Symbol]:
        out: List[Symbol] = []
        for f in [rel] if rel else self.files:
            pats = _SYM.get(Path(f).suffix.lower())
            if not pats:
                continue
            src = self.text(f)
            for rx, kind in pats:
                for m in rx.finditer(src):
                    name = m.group(1)
                    if name in _C_KEYWORDS:
                        continue
                    out.append(Symbol(name, kind, f, src.count("\n", 0, m.start(1)) + 1))
        return sorted(out, key=lambda s: (s.path, s.line))

    def find_symbol(self, name: str) -> List[Symbol]:
        return [s for s in self.symbols() if s.name == name]

    def search(self, pattern: str, regex: bool = False, glob: Optional[str] = None, limit: int = 200) -> List[Hit]:
        rx = re.compile(pattern if regex else re.escape(pattern))
        hits: List[Hit] = []
        for f in self.files:
            if glob and not Path(f).match(glob):
                continue
            for i, line in enumerate(self.text(f).splitlines(), 1):
                if rx.search(line):
                    hits.append(Hit(f, i, line.strip()[:240]))
                    if len(hits) >= limit:
                        return hits
        return hits

    def dependency_graph(self) -> Dict[str, Set[str]]:
        """file -> files in the repository it imports or includes."""
        by_module: Dict[str, str] = {}
        by_name: Dict[str, List[str]] = defaultdict(list)
        for f in self.files:
            p = Path(f)
            if p.suffix == ".py":
                parts = list(p.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                for i in range(len(parts)):
                    by_module.setdefault(".".join(parts[i:]), f)
            by_name[p.name].append(f)
        graph: Dict[str, Set[str]] = {}
        for f in self.files:
            suf = Path(f).suffix.lower()
            deps: Set[str] = set()
            src = self.text(f)
            if suf == ".py":
                for m in _PY_IMPORT.finditer(src):
                    if m.group(1):
                        names = [part.split()[0] for part in m.group(2).strip("() \t").replace("\n", " ").split(",")
                                 if part.split()]
                        mods = [f"{m.group(1)}.{n}" for n in names] + [m.group(1)]
                    else:
                        mods = [m.group(3)]
                    for k, mod in enumerate(mods):
                        # "from pkg import sub" names a submodule when one exists
                        exact = by_module.get(mod)
                        if exact:
                            deps.add(exact)
                            continue
                        if k != len(mods) - 1:
                            continue
                        while mod:
                            if mod in by_module:
                                deps.add(by_module[mod])
                                break
                            mod = mod.rpartition(".")[0]
            elif suf in (".c", ".h", ".cc", ".cpp", ".cu", ".hpp"):
                here = Path(f).parent
                for m in _C_INCLUDE.finditer(src):
                    inc = m.group(1)
                    cand = os.path.normpath(here / inc).replace("\\", "/")
                    if cand in self.files:
                        deps.add(cand)
                    elif len(by_name.get(Path(inc).name, [])) == 1:
                        deps.add(by_name[Path(inc).name][0])
            deps.discard(f)
            graph[f] = deps
        return graph

    def dependents(self, changed: Iterable[str]) -> Set[str]:
        """Every file that transitively imports or includes one of ``changed``."""
        rev: Dict[str, Set[str]] = defaultdict(set)
        for f, deps in self.dependency_graph().items():
            for d in deps:
                rev[d].add(f)
        seen: Set[str] = set()
        stack = list(changed)
        while stack:
            cur = stack.pop()
            for nxt in rev.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen
