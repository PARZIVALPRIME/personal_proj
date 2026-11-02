"""Git integration through the ToolRunner. There is deliberately no push."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .tools import CommandResult, ToolRunner

TRAILER = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"


class GitError(Exception):
    def __init__(self, result: CommandResult):
        super().__init__(f"{result.summary()}\n{result.stderr.strip()}")
        self.result = result


@dataclass
class StatusEntry:
    code: str
    path: str


class RepoManager:
    def __init__(self, runner: ToolRunner, root: Optional[str] = None):
        self.runner = runner
        self.root = str(Path(root or runner.policy.root).resolve())

    def git(self, *args: str, check: bool = True, cwd: Optional[str] = None) -> CommandResult:
        res = self.runner.run(["git", "-c", "core.quotepath=off", "--no-pager", *args], cwd=cwd or self.root)
        if check and not res.ok:
            raise GitError(res)
        return res

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").stdout.strip()

    def branch(self) -> str:
        return self.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def status(self) -> List[StatusEntry]:
        out = self.git("status", "--porcelain=v1", "--untracked-files=all").stdout
        entries = []
        for line in out.splitlines():
            if len(line) > 3:
                path = line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                entries.append(StatusEntry(line[:2], path.strip('"')))
        return entries

    def is_clean(self) -> bool:
        return not self.status()

    def changed_files(self, base: Optional[str] = None) -> List[str]:
        if base:
            out = self.git("diff", "--name-only", f"{base}...HEAD").stdout
            return [l for l in out.splitlines() if l]
        return [e.path for e in self.status()]

    def diff(self, base: Optional[str] = None, paths: Optional[List[str]] = None) -> str:
        args = ["diff"] + ([base] if base else []) + (["--", *paths] if paths else [])
        return self.git(*args).stdout

    def files(self, untracked: bool = False) -> List[str]:
        out = [l for l in self.git("ls-files").stdout.splitlines() if l]
        if untracked:
            out += [e.path for e in self.status() if e.code == "??"]
        return sorted(set(out))

    def add(self, paths: List[str]) -> None:
        for p in paths:
            self.runner.policy.check_write(Path(self.root) / p)
        self.git("add", "--", *paths)

    def commit(self, message: str, paths: Optional[List[str]] = None, trailer: bool = True) -> str:
        if paths:
            self.add(paths)
        if trailer and TRAILER not in message:
            message = message.rstrip() + "\n\n" + TRAILER + "\n"
        self.git("commit", "-q", "-m", message)
        return self.head()

    def worktree_add(self, path: str, branch: str, base: str = "HEAD") -> str:
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or ".." in branch:
            raise ValueError(f"unsafe branch name {branch!r}")
        self.git("worktree", "add", "-q", "-b", branch, path, base)
        return str(Path(path).resolve())

    def worktree_remove(self, path: str, branch: Optional[str] = None) -> None:
        self.git("worktree", "remove", "--force", path, check=False)
        self.git("worktree", "prune", check=False)
        if branch:
            self.git("branch", "-D", branch, check=False)
