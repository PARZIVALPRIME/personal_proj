import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coliagent import diagnose, patch as patchmod, planner  # noqa: E402
from coliagent.index import RepoIndex  # noqa: E402
from coliagent.llm import ScriptedBackend  # noqa: E402
from coliagent.orchestrate import Orchestrator  # noqa: E402
from coliagent.permissions import PermissionDenied, Policy  # noqa: E402
from coliagent.repo import TRAILER, RepoManager  # noqa: E402
from coliagent.selfimprove import Gate, Hypothesis, ImprovementCycle  # noqa: E402
from coliagent.state import EvidenceError, ProjectState  # noqa: E402
from coliagent.tools import ToolRunner  # noqa: E402

PY = Path(sys.executable).name


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


class TempRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="coliagent-test-")).resolve()
        self.root = self.tmp / "repo"
        self.root.mkdir()
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "config", "user.email", "t@example.com")
        git(self.root, "config", "user.name", "t")
        git(self.root, "config", "core.autocrlf", "false")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("")
        (self.root / "pkg" / "speed.py").write_text("FACTOR = 2\n\n\ndef work(n):\n    return n * FACTOR\n")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "test_speed.py").write_text(
            "import unittest\nfrom pkg.speed import work\n\n\nclass T(unittest.TestCase):\n"
            "    def test_work(self):\n        self.assertEqual(work(3) % 3, 0)\n")
        (self.root / "tests" / "test_other.py").write_text(
            "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
        (self.root / "bench.py").write_text("from pkg.speed import FACTOR\nprint(f'tok/s: {FACTOR * 1.5}')\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "init")
        self.policy = Policy(root=str(self.root), commands=("git", PY.lower().removesuffix(".exe"), "python"))
        self.runner = ToolRunner(self.policy)
        self.repo = RepoManager(self.runner)

    def tearDown(self):
        subprocess.run(["git", "worktree", "prune"], cwd=self.root, capture_output=True)
        shutil.rmtree(self.tmp, ignore_errors=True)


class PermissionTests(TempRepo):
    def test_push_and_remote_are_denied(self):
        for argv in (["git", "push"], ["git", "-c", "x=y", "push", "origin"], ["git", "remote", "add", "o", "u"]):
            with self.assertRaises(PermissionDenied):
                self.policy.check_command(argv)

    def test_network_commands_need_permission(self):
        with self.assertRaises(PermissionDenied):
            self.policy.check_command(["git", "fetch"])
        Policy(root=str(self.root), allow_network=True).check_command(["git", "fetch"])

    def test_shell_strings_and_unknown_commands_denied(self):
        with self.assertRaises(PermissionDenied):
            self.policy.check_command("git status")
        with self.assertRaises(PermissionDenied):
            self.policy.check_command(["powershell", "-c", "rm -r /"])

    def test_writes_outside_root_protected_or_secret_denied(self):
        for p in (self.tmp / "outside.txt", "agent/coliagent/permissions.py", ".git/config", "deploy.pem", ".env"):
            with self.assertRaises(PermissionDenied, msg=str(p)):
                self.policy.check_write(p)
        self.assertEqual(self.policy.check_write("pkg/speed.py"), "pkg/speed.py")

    def test_write_roots_restrict(self):
        pol = Policy(root=str(self.root), write_roots=("./pkg",))
        pol.check_write("pkg/speed.py")
        with self.assertRaises(PermissionDenied):
            pol.check_write("tests/test_speed.py")

    def test_runner_records_results(self):
        res = self.runner.run(["git", "status", "--porcelain"])
        self.assertTrue(res.ok)
        self.assertIs(self.runner.log[-1], res)


class RepoTests(TempRepo):
    def test_status_commit_adds_trailer(self):
        self.assertTrue(self.repo.is_clean())
        (self.root / "new.txt").write_text("x\n")
        self.assertEqual([e.path for e in self.repo.status()], ["new.txt"])
        self.repo.commit("add new", paths=["new.txt"])
        msg = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=self.root, capture_output=True, text=True).stdout
        self.assertIn(TRAILER, msg)
        self.assertTrue(self.repo.is_clean())

    def test_unsafe_branch_rejected(self):
        with self.assertRaises(ValueError):
            self.repo.worktree_add(str(self.tmp / "wt"), "a..b")


class IndexTests(TempRepo):
    def test_symbols_search_and_dependents(self):
        (self.root / "c").mkdir()
        (self.root / "c" / "eng.c").write_text('#include "eng.h"\nstatic int fast_path(int x) {\n  return x;\n}\n')
        (self.root / "c" / "eng.h").write_text("#define ENG_MAX 4\nint fast_path(int x);\n")
        (self.root / "c" / "test_eng.c").write_text('#include "eng.c"\nint main(void) { return 0; }\n')
        idx = RepoIndex(str(self.root))
        self.assertEqual([(s.path, s.line) for s in idx.find_symbol("work")], [("pkg/speed.py", 4)])
        self.assertIn(("c/eng.c", 2, "function"), [(s.path, s.line, s.kind) for s in idx.find_symbol("fast_path")])
        self.assertEqual(idx.find_symbol("ENG_MAX")[0].kind, "macro")
        self.assertEqual([h.path for h in idx.search("FACTOR = ", glob="*.py")], ["pkg/speed.py"])
        self.assertEqual(idx.dependents(["c/eng.h"]), {"c/eng.c", "c/test_eng.c"})
        self.assertIn("tests/test_speed.py", idx.dependents(["pkg/speed.py"]))
        (self.root / "tests" / "test_sub.py").write_text("from pkg import (\n    speed as s,\n)\nimport os\n")
        self.assertIn("tests/test_sub.py", RepoIndex(str(self.root)).dependents(["pkg/speed.py"]))


class PatchTests(TempRepo):
    def test_multi_file_patch_is_all_or_nothing(self):
        good = patchmod.Edit("pkg/speed.py", "FACTOR = 2", "FACTOR = 4")
        bad = patchmod.Edit("bench.py", "not there", "x")
        before = (self.root / "pkg" / "speed.py").read_text()
        with self.assertRaises(patchmod.PatchError):
            patchmod.apply(patchmod.Patch([good, bad]), str(self.root), self.policy)
        self.assertEqual((self.root / "pkg" / "speed.py").read_text(), before)

    def test_ambiguous_match_and_protected_write_refused(self):
        (self.root / "dup.txt").write_text("a\na\n")
        with self.assertRaises(patchmod.PatchError):
            patchmod.apply(patchmod.Patch([patchmod.Edit("dup.txt", "a", "b")]), str(self.root), self.policy)
        with self.assertRaises(PermissionDenied):
            patchmod.apply(patchmod.Patch([patchmod.Edit(".git/hooks/x", "", "y")]), str(self.root), self.policy)

    def test_apply_create_and_crlf(self):
        (self.root / "w.txt").write_bytes(b"one\r\ntwo\r\n")
        diff = patchmod.apply(patchmod.Patch([patchmod.Edit("w.txt", "one\ntwo", "one\n2"),
                                              patchmod.Edit("made.txt", "", "hi\n")]), str(self.root), self.policy)
        self.assertEqual((self.root / "w.txt").read_bytes(), b"one\r\n2\r\n")
        self.assertEqual((self.root / "made.txt").read_text(), "hi\n")
        self.assertIn("+++ b/made.txt", diff)
        self.assertIn("-two", diff)


class PlannerTests(unittest.TestCase):
    def test_ready_order_and_cycles(self):
        p = planner.Plan.from_dict({"goal": "g", "steps": [
            {"id": "a", "title": "A"}, {"id": "b", "title": "B", "deps": ["a"]}, {"id": "c", "title": "C"}]})
        self.assertEqual([s.id for s in p.ready()], ["a", "c"])
        p.mark("a", "done")
        self.assertEqual([s.id for s in p.ready()], ["b", "c"])
        with self.assertRaises(planner.PlanError):
            planner.Plan.from_dict({"steps": [{"id": "a", "title": "A", "deps": ["b"]},
                                              {"id": "b", "title": "B", "deps": ["a"]}]})
        with self.assertRaises(planner.PlanError):
            planner.Plan.from_dict({"steps": [{"id": "a", "title": "A", "deps": ["zzz"]}]})

    def test_llm_proposal_is_validated(self):
        llm = ScriptedBackend(['Sure!\n```json\n{"steps": [{"id": "x", "title": "X", "verify": [["python", "-V"]]}]}\n```',
                               '{"steps": [{"id": "Bad Id", "title": "X"}]}'])
        self.assertEqual(planner.propose(llm, "goal").steps[0].verify, [["python", "-V"]])
        with self.assertRaises(planner.PlanError):
            planner.propose(llm, "goal")


class DiagnoseTests(unittest.TestCase):
    def test_parsers(self):
        log = "\n".join([
            "qwen36.c:120:5: error: 'x' undeclared (first use in this function)",
            "qwen36.c:9:1: warning: unused variable 'y'",
            r"C:\src\q.c(44,7): error C2065: 'z': undeclared identifier",
            "ld: q.o: undefined reference to `cuda_init'",
            'Traceback (most recent call last):',
            '  File "c/tools/plan.py", line 12, in <module>',
            "ValueError: bad cap",
            "FAIL: test_cap (tests.test_plan.CapTests)",
            "FAILED tests/test_a.py::test_b - assert 1 == 2",
        ])
        d = diagnose.parse(log)
        self.assertEqual([(x.tool, x.severity) for x in d], [
            ("cc", "error"), ("cc", "warning"), ("msvc", "error"), ("ld", "error"),
            ("python", "error"), ("unittest", "fail"), ("pytest", "fail")])
        self.assertEqual((d[0].path, d[0].line, d[0].col), ("qwen36.c", 120, 5))
        self.assertEqual((d[2].path, d[2].line), (r"C:\src\q.c", 44))
        self.assertEqual((d[4].path, d[4].line), ("c/tools/plan.py", 12))
        self.assertEqual(len(diagnose.errors(log)), 6)


class OrchestrateAndStateTests(TempRepo):
    def test_selects_and_runs_dependent_tests(self):
        orch = Orchestrator(self.runner, str(self.root))
        self.assertEqual(orch.select_tests(["pkg/speed.py"]), ["tests/test_speed.py"])
        self.assertEqual(orch.select_tests(["tests/test_other.py"]), ["tests/test_other.py"])
        run = orch.run_test("tests/test_speed.py")
        self.assertTrue(run.ok, run.results and run.results[-1].stderr)

    def test_claims_need_matching_evidence(self):
        st = ProjectState.load(str(self.tmp / "state.json"))
        st.propose("faster", "decode is faster")
        with self.assertRaises(EvidenceError):
            st.applied("faster", "")
        failing = self.runner.run([PY, "-c", "raise SystemExit(1)"])
        with self.assertRaises(EvidenceError):
            st.tested("faster", [failing])
        ok = self.runner.run([PY, "-c", "print(1)"])
        st.tested("faster", [ok])
        with self.assertRaises(EvidenceError):
            st.perf_verified("faster", "tok/s", [3.0], [3.01], False, 0.02, [ok])
        st.perf_verified("faster", "tok/s", [3.0, 3.1], [3.6, 3.5], False, 0.02, [ok], machine="test")
        st.save()
        again = ProjectState.load(st.path)
        self.assertEqual(again.claims["faster"].level, "perf-verified")


class SelfImproveTests(TempRepo):
    def cycle(self):
        gates = [Gate([PY, "-m", "unittest", "-q", "tests.test_speed"])]
        return ImprovementCycle(self.repo, self.policy, gates, ledger=str(self.tmp / "ledger.jsonl"),
                                workdir=str(self.tmp))

    def hyp(self, **kw):
        return Hypothesis("a bigger factor raises tok/s", metric="tok/s", metric_argv=[PY, "bench.py"],
                          metric_regex=r"tok/s: ([\d.]+)", repeats=2, **kw)

    def branches(self):
        return subprocess.run(["git", "branch", "--format=%(refname:short)"], cwd=self.root,
                              capture_output=True, text=True).stdout.split()

    def test_accepts_improvement_on_a_branch_without_touching_main(self):
        head = self.repo.head()
        out = self.cycle().run(self.hyp(), patchmod.Patch([patchmod.Edit("pkg/speed.py", "FACTOR = 2", "FACTOR = 3")]), "win")
        self.assertTrue(out.accepted, out.reason)
        self.assertAlmostEqual(out.gain, 0.5)
        self.assertEqual(self.repo.head(), head)
        self.assertTrue(self.repo.is_clean())
        self.assertIn("agent/improve-win", self.branches())
        self.assertFalse((self.tmp / "coliagent-win").exists())
        shown = subprocess.run(["git", "show", "agent/improve-win:pkg/speed.py"], cwd=self.root,
                               capture_output=True, text=True).stdout
        self.assertIn("FACTOR = 3", shown)

    def test_rejects_regression_gate_failure_and_protected_paths(self):
        c = self.cycle()
        slower = c.run(self.hyp(), patchmod.Patch([patchmod.Edit("pkg/speed.py", "FACTOR = 2", "FACTOR = 1")]), "slow")
        self.assertFalse(slower.accepted)
        self.assertIn("below the required", slower.reason)
        broken = c.run(self.hyp(), patchmod.Patch([patchmod.Edit("pkg/speed.py", "return n * FACTOR", "return n * FACTOR + 1")]), "brk")
        self.assertFalse(broken.accepted)
        self.assertIn("gate failed", broken.reason)
        sneaky = c.run(self.hyp(), patchmod.Patch([patchmod.Edit("agent/coliagent/selfimprove.py", "", "x")]), "sneak")
        self.assertFalse(sneaky.accepted)
        self.assertIn("protected", sneaky.reason)
        self.assertEqual([b for b in self.branches() if b.startswith("agent/")], [])
        self.assertEqual([r["id"] for r in c.history()], ["slow", "brk", "sneak"])
        self.assertTrue(self.repo.is_clean())


if __name__ == "__main__":
    unittest.main()
