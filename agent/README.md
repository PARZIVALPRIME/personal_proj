# coliagent

A small, inspectable coding-agent framework for this repository. It uses only the
standard library (Python 3.9+). Every action goes through a frozen permission
policy, and every result is recorded.

```powershell
cd agent
python -m coliagent index --symbol qwen36_cap_for_ram      # where is it defined
python -m coliagent index --dependents c/qwen36.c          # what includes it
python -m coliagent tests --config examples/colibri.json   # tests for the working-tree changes
python -m coliagent tests --config examples/colibri.json --run --state ..\.git\coliagent\state.json --claim cap-fix
python -m coliagent diagnose build.log                     # gcc/clang/MSVC/ld/Python/unittest/pytest
python -m coliagent plan --file plan.json                  # validate a plan, list ready steps
python -m coliagent plan "speed up decode"                 # ask a local model (coli serve) for a plan
python -m coliagent patch change.json                      # preview a JSON patch; --apply to write
python -m coliagent improve --patch change.json --hypothesis "..." `
    --gate '["python","-m","unittest","tests.test_q36_target_plan"]' `
    --metric "tok/s" --metric-argv '["python","bench.py"]' --metric-regex "tok/s: ([\d.]+)"
python -m coliagent ledger
```

## Modules

| Module | Role |
|---|---|
| `permissions` | `Policy`: allowed commands, no shell strings, no `git push`/`remote`, network off by default. Writes stay inside the repo and the write roots, never to protected paths or secret-looking files. |
| `tools` | `ToolRunner`: argv-only execution with timeout and output cap. Every `CommandResult` is logged. |
| `repo` | Git status, diff, files, commit (with the Co-authored-by trailer) and worktrees. There is no push. |
| `index` | File inventory, regex symbols (Python, C/C++, JS/TS), text search, and the Python import / C `#include` graph with transitive dependents. |
| `patch` | Exact-match edits. Each edit's old text must occur once. All-or-nothing across files, CRLF-safe, with a unified diff. |
| `planner` | Steps with dependencies, topological order, ready set. Model proposals are validated before use. |
| `llm` | OpenAI-compatible client (default `http://127.0.0.1:8000/v1`, i.e. a local `coli serve`; `COLIAGENT_BASE_URL`/`COLIAGENT_MODEL`/`COLIAGENT_API_KEY`), plus a scripted backend for tests. |
| `diagnose` | Turns build and test logs into structured diagnostics. |
| `orchestrate` | Build/test runs from a JSON config, and test selection from changed files (changed tests, transitive dependents, name match). |
| `state` | Claims move up proposed → applied → built → tested → perf-verified. Each rung needs the matching evidence (a diff, passing runs, baseline and candidate samples). |
| `selfimprove` | One hypothesis, one patch, one isolated worktree (see below). |

## The self-improvement cycle

1. Refuse patches that touch protected paths: `permissions.py`, `selfimprove.py`, `.git/`, `.github/`. The agent cannot loosen its own rules or its own gate.
2. Create a fresh worktree on branch `agent/improve-<id>` (in the temp dir), leaving the main checkout untouched.
3. Measure the baseline metric there (median of `--repeats`).
4. Apply the patch, then run every gate. If any gate fails, reject.
5. Measure again. Reject unless the median moved by at least `--min-gain` in the right direction.
6. If accepted, commit to the branch and keep the branch for review. If rejected, delete the branch. The worktree is always removed, and each outcome is appended to `.git/coliagent/ledger.jsonl`.

The cycle never merges and never pushes. Promoting a branch is a human decision.

Python bytecode is disabled inside the worktree. Without that, a patch that keeps a
file's size and lands in the same mtime second would run the baseline's stale `.pyc`.

## Tests

```powershell
cd agent; python -m unittest discover -s tests
```

The tests build throwaway git repositories in the temp dir and cover:

- policy denials;
- atomic patches;
- index and dependents;
- plan validation;
- log parsing;
- test selection;
- evidence rules;
- an accepted cycle, plus rejections for a regression, a gate failure and a protected path.
