"""argv execution under the policy. No shell strings; every run is recorded."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .permissions import Policy


@dataclass
class CommandResult:
    argv: List[str]
    cwd: str
    returncode: Optional[int]
    seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self) -> str:
        status = "timeout" if self.timed_out else f"exit {self.returncode}"
        return f"{' '.join(self.argv)} -> {status} in {self.seconds:.1f}s"

    def to_dict(self) -> Dict:
        return asdict(self)


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} bytes cut ...]\n" + text[-limit:]


class ToolRunner:
    def __init__(self, policy: Policy, log: Optional[List[CommandResult]] = None,
                 base_env: Optional[Dict[str, str]] = None):
        self.policy = policy
        self.log: List[CommandResult] = log if log is not None else []
        self.base_env = dict(base_env or {})

    def run(self, argv: Sequence[str], cwd: Optional[str] = None, timeout: Optional[float] = None,
            env: Optional[Dict[str, str]] = None, input_text: Optional[str] = None) -> CommandResult:
        argv = list(argv)
        self.policy.check_command(argv)
        workdir = str(Path(cwd or self.policy.root).resolve())
        self.policy.check_read(workdir)
        if ("/" in argv[0] or "\\" in argv[0]) and not os.path.isabs(argv[0]):
            # resolve against the run's cwd; Windows would otherwise use ours
            argv[0] = str(Path(workdir, argv[0]).resolve())
            self.policy.check_command(argv)
        full_env = dict(os.environ)
        full_env.update(self.base_env)
        if env:
            full_env.update(env)
        limit = min(timeout or self.policy.max_seconds, self.policy.max_seconds)
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, cwd=workdir, env=full_env, input=input_text, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=limit, shell=False)
            res = CommandResult(argv, workdir, p.returncode, time.monotonic() - t0,
                                _tail(p.stdout or "", self.policy.max_output_bytes),
                                _tail(p.stderr or "", self.policy.max_output_bytes))
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            res = CommandResult(argv, workdir, None, time.monotonic() - t0,
                                _tail(out, self.policy.max_output_bytes),
                                _tail(err, self.policy.max_output_bytes), timed_out=True)
        except FileNotFoundError as e:
            res = CommandResult(argv, workdir, 127, time.monotonic() - t0, "", str(e))
        self.log.append(res)
        return res
