"""coliagent: a modular, inspectable coding-agent framework (stdlib only).

Modules:
  permissions  what the agent may run and write (frozen; never self-modified)
  tools        argv execution under the policy, every result recorded
  repo         git status/diff/worktree/commit; no push, by design
  index        file inventory, symbols, text search, include/import graph
  patch        exact, all-or-nothing multi-file edits and unified diffs
  planner      steps with dependencies; an LLM may propose, the structure checks
  llm          backends: any OpenAI-compatible server (a local Colibri), or scripted
  diagnose     compiler/test log parsing into structured diagnostics
  orchestrate  build/test runs, test selection from changed files
  state        explicit project state and evidence-backed claims
  selfimprove  controlled improvement cycle in an isolated worktree
"""

__version__ = "0.1.0"
