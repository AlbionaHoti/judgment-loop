"""Tests for the M0 harness. No model calls: Ollama is stubbed.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jl import config, data, stats  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_dataset = _load("build_dataset", "data/build_dataset.py")
blind_ab = _load("blind_ab", "eval/blind_ab.py")
import_sheet = _load("import_sheet", "data/import_sheet.py")
render_modelfile = _load("render_modelfile", "notebook/render_modelfile.py")


def decision(did, split="train", **kw):
    d = {"decision_id": did, "situation": f"situation {did}", "call": f"call {did}",
         "why": "because", "split": split, "status": "active"}
    d.update(kw)
    return d


def example(did, scenario="a scenario", **kw):
    e = {"decision_id": did, "scenario": scenario, "response": "my call",
         "kind": "apply", "source": "authored"}
    e.update(kw)
    return e


def test_row(tid, did, suite, question="what now?"):
    return {"test_id": tid, "decision_id": did, "suite": suite, "question": question}


class SingleSourceOfTruth(unittest.TestCase):
    def test_modelfile_matches_config(self):
        with open(os.path.join(ROOT, "notebook", "Modelfile")) as f:
            self.assertEqual(f.read(), render_modelfile.render(config.load()),
                             "run: python3 notebook/render_modelfile.py")

    def test_no_script_hardcodes_the_system_prompt(self):
        prompt = config.load()["system_prompt"]
        for rel in ("data/build_dataset.py", "eval/blind_ab.py",
                    "notebook/train_judgment_model.ipynb"):
            with open(os.path.join(ROOT, rel)) as f:
                self.assertNotIn(prompt[:40], f.read(), rel)

    def test_shipped_examples_are_valid(self):
        decisions = data.load_decisions(data.DECISIONS_EXAMPLE)
        examples = data.load_examples(data.EXAMPLES_EXAMPLE)
        self.assertEqual(data.validate_decisions(decisions), [])
        usable, errors, _ = data.trainable_examples(decisions, examples)
        self.assertEqual(errors, [])
        tests = [t for t in data.load_tests(data.TESTS)
                 if t["test_id"].startswith("T-EX")]
        if tests:
            t_errors, _ = data.validate_tests(decisions, usable, tests)
            self.assertEqual(t_errors, [])
            self.assertEqual(data.near_duplicates(usable, tests), [])


class Holdout(unittest.TestCase):
    def test_training_on_heldout_decision_is_an_error(self):
        ds = [decision("D-1"), decision("D-2", split="test")]
        usable, errors, _ = data.trainable_examples(ds, [example("D-1"), example("D-2")])
        self.assertEqual(len(usable), 1)
        self.assertTrue(any("held-out" in e for e in errors))

    def test_retired_decision_is_skipped_not_fatal(self):
        ds = [decision("D-1", status="retired", superseded_by="D-2"), decision("D-2")]
        self.assertEqual(data.validate_decisions(ds), [])
        usable, errors, warnings = data.trainable_examples(ds, [example("D-1"), example("D-2")])
        self.assertEqual([e["decision_id"] for e in usable], ["D-2"])
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

    def test_suite_must_agree_with_split(self):
        ds = [decision("D-1"), decision("D-2", split="test")]
        ex = [example("D-1")]
        errors, _ = data.validate_tests(ds, ex, [
            test_row("T-1", "D-1", "seen"),
            test_row("T-2", "D-2", "unseen"),
            test_row("T-3", "D-1", "exception"),
        ])
        self.assertEqual(errors, [])
        errors, _ = data.validate_tests(ds, ex, [
            test_row("T-1", "D-1", "unseen"),
            test_row("T-2", "D-2", "seen"),
        ])
        self.assertEqual(len(errors), 2)

    def test_near_duplicate_is_caught_and_paraphrase_is_not(self):
        ex = [example("D-1", "We could ship tonight if we skip the second review pass.")]
        copy = [test_row("T-1", "D-1", "seen",
                         "We could ship tonight if we skip the second review pass?")]
        fresh = [test_row("T-2", "D-1", "seen",
                          "Legal has not signed off and the launch email is queued for 9am.")]
        self.assertEqual(len(data.near_duplicates(ex, copy)), 1)
        self.assertEqual(data.near_duplicates(ex, fresh), [])

    def test_stuffed_prompt_excludes_heldout_and_retired(self):
        ds = [decision("D-1"), decision("D-2", split="test"),
              decision("D-3", status="retired")]
        prompt = data.stuffed_system_prompt("SYS", ds)
        self.assertIn("call D-1", prompt)
        self.assertNotIn("call D-2", prompt)
        self.assertNotIn("call D-3", prompt)


class Stats(unittest.TestCase):
    def test_wilson_brackets_the_point_estimate(self):
        lo, hi = stats.wilson(7, 10)
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.7)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))

    def test_sign_test_known_values(self):
        self.assertAlmostEqual(stats.sign_test(5, 5), 1.0)
        self.assertAlmostEqual(stats.sign_test(10, 0), 2 / 1024)
        self.assertEqual(stats.sign_test(0, 0), 1.0)

    def test_small_samples_are_not_called_wins(self):
        self.assertTrue(stats.verdict(4, 0, 30).startswith("TOO FEW TESTS"))
        self.assertEqual(stats.verdict(25, 5, 30), "WIN")
        self.assertEqual(stats.verdict(5, 25, 30), "LOSS")
        self.assertTrue(stats.verdict(17, 13, 30).startswith("INCONCLUSIVE"))

    def test_cluster_bootstrap(self):
        mean, lo, hi = stats.cluster_bootstrap_mean(
            [1, 1, 0, 2, 1, 1], ["a", "a", "b", "b", "c", "c"])
        self.assertAlmostEqual(mean, 1.0)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)
        self.assertEqual(stats.cluster_bootstrap_mean([1, 2], ["a", "a"])[1], None)


class BuildDataset(unittest.TestCase):
    def test_build_writes_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "train.jsonl")
            man = os.path.join(tmp, "manifest.json")
            logs = []
            m = build_dataset.build(
                out_path=out, manifest_path=man,
                decisions_path=data.DECISIONS_EXAMPLE,
                examples_path=data.EXAMPLES_EXAMPLE,
                tests_path=os.path.join(tmp, "none.jsonl"),
                mined_path=os.path.join(tmp, "none.json"), log=logs.append)
            rows = data.read_jsonl(out)
            self.assertEqual(len(rows), m["examples"])
            self.assertEqual(rows[0]["messages"][0]["content"],
                             config.load()["system_prompt"])
            self.assertEqual(m["dataset_sha256"], config.file_sha256(out))
            self.assertEqual(m["heldout_decisions"], 1)
            self.assertNotIn("D-05", m["per_decision"])

    def test_build_refuses_a_leaky_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dpath = os.path.join(tmp, "d.jsonl")
            epath = os.path.join(tmp, "e.jsonl")
            data.write_jsonl(dpath, [decision("D-1"), decision("D-2", split="test")])
            data.write_jsonl(epath, [example("D-1"), example("D-2")])
            out = os.path.join(tmp, "train.jsonl")
            with self.assertRaises(SystemExit):
                build_dataset.build(
                    out_path=out, manifest_path=os.path.join(tmp, "m.json"),
                    decisions_path=dpath, examples_path=epath,
                    tests_path=os.path.join(tmp, "none.jsonl"),
                    mined_path=os.path.join(tmp, "none.json"), log=lambda *_: None)
            self.assertFalse(os.path.exists(out))


class ImportSheet(unittest.TestCase):
    def test_sheet_rows_become_decisions_and_tests(self):
        rows = [
            {"decisionid": "EXAMPLE", "situationcontext": "x", "mycall": "y"},
            {"decisionid": "D-1", "domain": "Ship", "situationcontext": "s1",
             "mycall": "c1", "why": "w", "split": "Train", "keep": "Yes"},
            {"decisionid": "D-2", "situationcontext": "s2", "mycall": "c2",
             "split": "Test", "keep": "Yes"},
            {"decisionid": "D-3", "situationcontext": "s3", "mycall": "c3",
             "split": "Train", "keep": "No"},
        ]
        ds = import_sheet.decisions_from_rows(rows)
        self.assertEqual([d["decision_id"] for d in ds], ["D-1", "D-2"])
        self.assertEqual(ds[1]["split"], "test")
        self.assertEqual(data.validate_decisions(ds), [])
        ts = import_sheet.tests_from_rows([
            {"testid": "T-EXAMPLE", "sourcedecisionid": "EXAMPLE"},
            {"testid": "T-1", "sourcedecisionid": "D-1", "newunseensituation": "q1"},
            {"testid": "T-2", "sourcedecisionid": "D-2", "newunseensituation": "q2"},
            {"testid": "T-3", "sourcedecisionid": "D-1", "newunseensituation": "q3",
             "suite": "Exception"},
        ], ds)
        self.assertEqual([t["suite"] for t in ts], ["seen", "unseen", "exception"])


class BlindRun(unittest.TestCase):
    """End to end with a stubbed model: run -> judge -> reveal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: getattr(blind_ab, k) for k in
                      ("PAIRS_CSV", "KEY_JSON", "RESULTS_JSON", "ask")}
        # Hermetic: never read the user's own decisions, examples or tests.
        self.saved_data = {k: getattr(data, k) for k in
                           ("DECISIONS", "EXAMPLES", "TESTS")}
        data.DECISIONS = data.DECISIONS_EXAMPLE
        data.EXAMPLES = data.EXAMPLES_EXAMPLE
        data.TESTS = os.path.join(self.tmp.name, "tests.jsonl")
        data.write_jsonl(data.TESTS, [
            test_row("T-1", "D-01", "seen", "A memo hides its one real risk in a footnote."),
            test_row("T-2", "D-05", "unseen", "Asked for a typo fix; the layout is also ugly."),
            test_row("T-3", "D-03", "exception", "Everything is down and a rollback fixes it."),
            test_row("T-4", "D-02", "seen", "The deck quotes a figure nobody can source."),
        ])
        blind_ab.PAIRS_CSV = os.path.join(self.tmp.name, "pairs.csv")
        blind_ab.KEY_JSON = os.path.join(self.tmp.name, "key.json")
        blind_ab.RESULTS_JSON = os.path.join(self.tmp.name, "results.json")
        self.calls = []

        def fake_ask(cfg, model, system, question):
            stuffed = "standing decisions" in system
            self.calls.append((model, stuffed))
            who = ("SPECIALIST" if model == cfg["specialist"]
                   else "STUFFED" if stuffed else "BASE")
            return f"{who} answer to: {question[:20]}"
        blind_ab.ask = fake_ask

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(blind_ab, k, v)
        for k, v in self.saved_data.items():
            setattr(data, k, v)
        self.tmp.cleanup()

    def _judge(self, rank_for):
        with open(blind_ab.PAIRS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for L in "abc":
                who = r[f"answer_{L}"].split()[0]
                r[f"rank_{L}"] = rank_for[who]
                r[f"fidelity_{L}"] = {1: 2, 2: 1, 3: 0}[rank_for[who]]
                r[f"unsupported_{L}"] = 0
        with open(blind_ab.PAIRS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return rows

    def test_run_judge_reveal(self):
        logs = []
        blind_ab.run(log=logs.append)
        n_tests = len(data.load_tests(data.TESTS))
        self.assertEqual(len(self.calls), n_tests * 3)
        self.assertTrue(any(s for _, s in self.calls), "stuffed arm was never asked")

        with open(blind_ab.PAIRS_CSV, newline="") as f:
            header = next(csv.reader(f))
        self.assertIn("answer_c", header)
        self.assertNotIn("suite", header)
        with open(blind_ab.PAIRS_CSV) as f:
            body = f.read()
        for label in ("My model", "Base model", "my-judgment", "qwen"):
            self.assertNotIn(label, body, "the judging sheet leaks an arm identity")

        self._judge({"SPECIALIST": 1, "STUFFED": 2, "BASE": 3})
        res = blind_ab.reveal(log=logs.append)
        for opp in ("base", "stuffed"):
            o = res["vs"][opp]["overall"]
            self.assertEqual((o["wins"], o["losses"], o["ties"]), (n_tests, 0, 0))
            self.assertGreater(o["fidelity_diff_mean"], 0)
        # a handful of straight wins is still not a result
        self.assertTrue(res["vs"]["base"]["overall"]["verdict"].startswith("TOO FEW TESTS"))
        self.assertTrue(os.path.exists(blind_ab.RESULTS_JSON))

    def test_ties_and_losses_are_counted_as_such(self):
        blind_ab.run(log=lambda *_: None)
        self._judge({"SPECIALIST": 2, "STUFFED": 1, "BASE": 2})
        res = blind_ab.reveal(log=lambda *_: None)
        n = res["vs"]["base"]["overall"]["tests"]
        self.assertEqual(res["vs"]["base"]["overall"]["ties"], n)
        self.assertEqual(res["vs"]["stuffed"]["overall"]["losses"], n)

    def test_bad_rank_stops_reveal_instead_of_counting_a_loss(self):
        blind_ab.run(log=lambda *_: None)
        rows = self._judge({"SPECIALIST": 1, "STUFFED": 2, "BASE": 3})
        rows[0]["rank_a"] = "B"
        with open(blind_ab.PAIRS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(SystemExit):
            blind_ab.reveal(log=lambda *_: None)

    def test_unfinished_judging_stops_reveal(self):
        blind_ab.run(log=lambda *_: None)
        with self.assertRaises(SystemExit):
            blind_ab.reveal(log=lambda *_: None)

    def test_run_will_not_clobber_judging(self):
        blind_ab.run(log=lambda *_: None)
        self._judge({"SPECIALIST": 1, "STUFFED": 2, "BASE": 3})
        with self.assertRaises(SystemExit):
            blind_ab.run(log=lambda *_: None)
        blind_ab.run(force=True, log=lambda *_: None)

    def test_shuffle_actually_varies(self):
        seen = set()
        for _ in range(12):
            blind_ab.run(force=True, log=lambda *_: None)
            with open(blind_ab.KEY_JSON) as f:
                key = json.load(f)
            for t in key["tests"].values():
                seen.add(t["letters"]["A"])
        self.assertEqual(seen, {"specialist", "base", "stuffed"})


if __name__ == "__main__":
    unittest.main()
