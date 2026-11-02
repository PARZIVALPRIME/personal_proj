"""What the agent may run and write.

The policy is a frozen value handed to every component that acts. Nothing in
the agent constructs a wider policy from a narrower one, and the self-improvement
cycle refuses any change that touches the files listed in ``protected``
(this module among them), so the agent cannot widen its own permissions.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Tuple


class PermissionDenied(Exception):
    pass


DEFAULT_COMMANDS: Tuple[str, ...] = (
    "git", "python", "python3", "py", "make", "cmake", "ninja", "gcc", "clang", "cc",
    "zig", "rg", "pytest", "node", "npm", "go", "cargo",
)
# git subcommands that publish or rewrite shared history
DENIED_GIT: Tuple[str, ...] = ("push", "send-email", "request-pull", "remote")
DEFAULT_PROTECTED: Tuple[str, ...] = (
    "agent/coliagent/permissions.py", "agent/coliagent/selfimprove.py", ".git/*", ".github/*",
)
SECRET_PATTERNS: Tuple[str, ...] = ("*.pem", "*.key", "*id_rsa*", "*.pfx", ".env", "*.env", "*credentials*")


@dataclass(frozen=True)
class Policy:
    root: str
    commands: Tuple[str, ...] = DEFAULT_COMMANDS
    write_roots: Tuple[str, ...] = (".",)
    protected: Tuple[str, ...] = DEFAULT_PROTECTED
    allow_network: bool = False
    max_seconds: float = 1800.0
    max_output_bytes: int = 200_000
    allow_repo_binaries: bool = True
    extra_denied_git: Tuple[str, ...] = field(default_factory=tuple)

    def _rel(self, path: str | os.PathLike) -> str:
        root = Path(self.root).resolve()
        p = Path(path)
        p = (root / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            raise PermissionDenied(f"{path} is outside the repository {root}")

    def check_write(self, path: str | os.PathLike) -> str:
        rel = self._rel(path)
        name = Path(rel).name
        if any(fnmatch.fnmatch(rel, pat) for pat in self.protected):
            raise PermissionDenied(f"{rel} is protected")
        if any(fnmatch.fnmatch(name, pat) for pat in SECRET_PATTERNS):
            raise PermissionDenied(f"{rel} looks like a secret")
        roots = [_norm_root(r) for r in self.write_roots]
        if not any(r == "" or rel == r or rel.startswith(r + "/") for r in roots):
            raise PermissionDenied(f"{rel} is outside the write roots {self.write_roots}")
        return rel

    def check_read(self, path: str | os.PathLike) -> str:
        rel = self._rel(path)
        if any(fnmatch.fnmatch(Path(rel).name, pat) for pat in SECRET_PATTERNS):
            raise PermissionDenied(f"{rel} looks like a secret")
        return rel

    def check_command(self, argv: Sequence[str]) -> None:
        if not argv or not isinstance(argv, (list, tuple)):
            raise PermissionDenied("a command is a non-empty argv list (no shell strings)")
        exe = Path(argv[0]).name.lower()
        for suffix in (".exe", ".cmd", ".bat"):
            if exe.endswith(suffix):
                exe = exe[: -len(suffix)]
        if self.allow_repo_binaries and ("/" in argv[0] or "\\" in argv[0]):
            # a binary the build produced inside the repository (a test runner);
            # its code is under review anyway, exactly like a Python test file
            self._rel(argv[0])
            return
        if exe not in self.commands:
            raise PermissionDenied(f"{argv[0]} is not an allowed command")
        if exe == "git":
            sub = _git_subcommand(argv[1:])
            if sub in DENIED_GIT + tuple(self.extra_denied_git):
                raise PermissionDenied(f"git {sub} is not allowed: the agent never publishes")
            if sub in ("fetch", "pull", "clone", "ls-remote") and not self.allow_network:
                raise PermissionDenied(f"git {sub} needs the network, which the policy denies")
        if exe in ("npm", "pip", "cargo", "go") and not self.allow_network:
            if any(a in ("install", "add", "get", "fetch", "update") for a in argv[1:3]):
                raise PermissionDenied(f"{exe} {argv[1]} needs the network, which the policy denies")
        if exe in ("python", "python3", "py") and not self.allow_network:
            if len(argv) > 3 and argv[1:3] == ["-m", "pip"] and argv[3] in ("install", "download"):
                raise PermissionDenied("pip install needs the network, which the policy denies")

    def touches_protected(self, paths: Iterable[str]) -> bool:
        for p in paths:
            try:
                rel = self._rel(p)
            except PermissionDenied:
                return True
            if any(fnmatch.fnmatch(rel, pat) for pat in self.protected):
                return True
        return False


def _norm_root(r: str) -> str:
    r = r.replace("\\", "/").strip("/")
    while r.startswith("./"):
        r = r[2:]
    return "" if r in ("", ".") else r


def _git_subcommand(args: Sequence[str]) -> str:
    """The first argument that is not an option or an option's value (-c k=v, -C dir)."""
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("-c", "-C", "--git-dir", "--work-tree", "--namespace"):
            skip = True
            continue
        if a.startswith("-"):
            continue
        return a
    return ""
