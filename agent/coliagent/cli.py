"""Command line: python -m coliagent <command> (run from agent/ or with agent/ on PYTHONPATH)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import diagnose, patch as patchmod, planner
from .index import RepoIndex
from .orchestrate import Orchestrator, load_config
from .permissions import Policy
from .repo import RepoManager
from .selfimprove import Gate, Hypothesis, ImprovementCycle
from .state import ProjectState
from .tools import ToolRunner


def _ctx(args):
    root = str(Path(args.root).resolve())
    policy = Policy(root=root, allow_network=args.allow_network)
    runner = ToolRunner(policy)
    return root, policy, runner


def _git_root(start: str) -> str:
    p = Path(start).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return str(cand)
    return str(p)


def cmd_index(args) -> int:
    root, _, runner = _ctx(args)
    try:
        files = RepoManager(runner, root).files()
    except Exception:
        files = None
    idx = RepoIndex(root, files)
    if args.symbol:
        for s in idx.find_symbol(args.symbol):
            print(f"{s.path}:{s.line} {s.kind} {s.name}")
    elif args.search:
        for h in idx.search(args.search, regex=args.regex, glob=args.glob):
            print(f"{h.path}:{h.line}: {h.text}")
    elif args.dependents:
        for f in sorted(idx.dependents(args.dependents)):
            print(f)
    else:
        by_ext = {}
        for f in idx.files:
            by_ext[Path(f).suffix or "(none)"] = by_ext.get(Path(f).suffix or "(none)", 0) + 1
        print(f"{len(idx.files)} files")
        for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {ext:10} {n}")
    return 0


def cmd_tests(args) -> int:
    root, _, runner = _ctx(args)
    repo = RepoManager(runner, root)
    orch = Orchestrator(runner, root, load_config(root, args.config))
    changed = args.files or repo.changed_files(args.base)
    tests = orch.select_tests(changed, RepoIndex(root, repo.files(untracked=True)))
    if not args.run:
        print("\n".join(tests) or "(no tests selected)")
        return 0
    state = ProjectState.load(args.state) if args.state else None
    failed = 0
    all_results = []
    for run in orch.run_tests(tests):
        all_results.extend(run.results)
        print(("PASS " if run.ok else "FAIL ") + run.test)
        if not run.ok:
            failed += 1
            last = run.results[-1] if run.results else None
            if last is not None:
                diags = diagnose.errors(last.stdout + "\n" + last.stderr)
                print(diagnose.summarize(diags) or (last.stdout + last.stderr)[-1500:])
    if state and args.claim and tests and not failed:
        state.propose(args.claim, args.claim)
        state.tested(args.claim, all_results)
        state.save()
    return 1 if failed else 0


def cmd_diagnose(args) -> int:
    text = Path(args.log).read_text(encoding="utf-8", errors="replace") if args.log != "-" else sys.stdin.read()
    print(diagnose.summarize(diagnose.parse(text), limit=args.limit) or "(nothing recognised)")
    return 0


def cmd_plan(args) -> int:
    if args.file:
        plan = planner.Plan.from_dict(json.loads(Path(args.file).read_text(encoding="utf-8")))
    else:
        from .llm import OpenAICompatible
        plan = planner.propose(OpenAICompatible(base_url=args.base_url, model=args.model), args.goal)
    print(plan.to_json())
    print("ready:", ", ".join(s.id for s in plan.ready()) or "(none)", file=sys.stderr)
    return 0


def cmd_patch(args) -> int:
    root, policy, _ = _ctx(args)
    p = patchmod.Patch.from_dict(json.loads(Path(args.file).read_text(encoding="utf-8")))
    if args.apply:
        print(patchmod.apply(p, root, policy), end="")
    else:
        print(patchmod.unified_diff(patchmod.plan(p, root, policy)), end="")
    return 0


def cmd_improve(args) -> int:
    root, policy, runner = _ctx(args)
    repo = RepoManager(runner, root)
    p = patchmod.Patch.from_dict(json.loads(Path(args.patch).read_text(encoding="utf-8")))
    gates = [Gate(json.loads(g)) for g in args.gate]
    if not gates:
        print("refusing to run without at least one --gate (a JSON argv list)", file=sys.stderr)
        return 2
    h = Hypothesis(args.hypothesis, metric=args.metric, metric_argv=json.loads(args.metric_argv) if args.metric_argv else None,
                   metric_regex=args.metric_regex, metric_cwd=args.metric_cwd, lower_is_better=args.lower_is_better,
                   min_gain=args.min_gain, repeats=args.repeats)
    out = ImprovementCycle(repo, policy, gates).run(h, p, args.id)
    print(json.dumps({k: v for k, v in out.__dict__.items() if k != "diff"}, indent=2))
    return 0 if out.accepted else 1


def cmd_ledger(args) -> int:
    root, policy, runner = _ctx(args)
    for rec in ImprovementCycle(RepoManager(runner, root), policy, []).history():
        mark = "ACCEPT" if rec["accepted"] else "reject"
        print(f"{rec['at']} {mark} {rec['id']} {rec['hypothesis']['statement']} :: {rec['reason'].splitlines()[0]}")
    return 0


def cmd_state(args) -> int:
    print(ProjectState.load(args.state).report())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="coliagent", description=__doc__)
    ap.add_argument("--root", default=_git_root("."), help="repository root (default: enclosing git repo)")
    ap.add_argument("--allow-network", action="store_true", help="allow fetch/install commands")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="inventory, symbol lookup, search, dependents")
    s.add_argument("--symbol")
    s.add_argument("--search")
    s.add_argument("--regex", action="store_true")
    s.add_argument("--glob")
    s.add_argument("--dependents", nargs="+")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("tests", help="select (and optionally run) tests for the changed files")
    s.add_argument("files", nargs="*")
    s.add_argument("--base", help="compare against this revision instead of the working tree")
    s.add_argument("--config")
    s.add_argument("--run", action="store_true")
    s.add_argument("--state", help="project state JSON to record a tested claim into")
    s.add_argument("--claim")
    s.set_defaults(fn=cmd_tests)

    s = sub.add_parser("diagnose", help="parse a build/test log ('-' for stdin)")
    s.add_argument("log")
    s.add_argument("--limit", type=int, default=40)
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("plan", help="validate a plan file, or ask a model for one")
    s.add_argument("goal", nargs="?", default="")
    s.add_argument("--file")
    s.add_argument("--base-url")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("patch", help="preview (default) or --apply a JSON patch")
    s.add_argument("file")
    s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_patch)

    s = sub.add_parser("improve", help="run one gated improvement cycle in an isolated worktree")
    s.add_argument("--patch", required=True)
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--gate", action="append", default=[], help='JSON argv, e.g. \'["python","-m","unittest"]\'')
    s.add_argument("--metric")
    s.add_argument("--metric-argv")
    s.add_argument("--metric-regex", default=r"([-+]?\d+(?:\.\d+)?)")
    s.add_argument("--metric-cwd", default=".")
    s.add_argument("--lower-is-better", action="store_true")
    s.add_argument("--min-gain", type=float, default=0.02)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--id")
    s.set_defaults(fn=cmd_improve)

    s = sub.add_parser("ledger", help="show the improvement ledger")
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("state", help="show evidence-backed claims")
    s.add_argument("state")
    s.set_defaults(fn=cmd_state)

    args = ap.parse_args(argv)
    return args.fn(args)
