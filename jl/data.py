"""Decisions, training examples and hidden tests: load, validate, check leakage.

Three files, all JSONL, joined by decision_id:

  data/decisions.jsonl   one row per repeated decision
      decision_id, domain, situation, call, why, evidence, confidence,
      split ("train" | "test"), status ("active" | "retired"),
      superseded_by (optional decision_id), exceptions (optional text)

  data/examples.jsonl    one row per training example
      decision_id, scenario, response,
      kind ("apply" | "exception"), source ("authored" | "mined"),
      created_at (optional ISO date)

  eval/hidden-tests.jsonl  one row per blind test
      test_id, decision_id, suite, question, expected,
      must_mention, must_avoid

Suites:
  seen       a NEW scenario for a decision the model trained on.
             Measures application: did it learn the decision?
  unseen     a scenario for a held-out decision (split = "test").
             Measures transfer: does it generalise your taste?
  exception  a scenario where a trained decision should NOT fire.
             Measures over-triggering: did it learn a rule or a reflex?
"""

import difflib
import json
import os
import re

from . import config

DATA = os.path.join(config.ROOT, "data")
EVAL = os.path.join(config.ROOT, "eval")

DECISIONS = os.path.join(DATA, "decisions.jsonl")
EXAMPLES = os.path.join(DATA, "examples.jsonl")
DECISIONS_EXAMPLE = os.path.join(DATA, "decisions.example.jsonl")
EXAMPLES_EXAMPLE = os.path.join(DATA, "examples.example.jsonl")
TESTS = os.path.join(EVAL, "hidden-tests.jsonl")

SPLITS = ("train", "test")
STATUSES = ("active", "retired")
SUITES = ("seen", "unseen", "exception")
KINDS = ("apply", "exception")

# Near-duplicate thresholds for train-scenario vs test-question.
JACCARD_LIMIT = 0.6
RATIO_LIMIT = 0.8


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: not valid JSON ({e})")
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve(real, example):
    """Return (path, is_example). Falls back to the shipped example file."""
    if os.path.exists(real):
        return real, False
    return example, True


def load_decisions(path=None):
    if path is None:
        path, _ = resolve(DECISIONS, DECISIONS_EXAMPLE)
    rows = read_jsonl(path)
    for r in rows:
        r.setdefault("split", "train")
        r.setdefault("status", "active")
        r["split"] = str(r["split"]).strip().lower()
        r["status"] = str(r["status"]).strip().lower()
    return rows


def load_examples(path=None):
    if path is None:
        path, _ = resolve(EXAMPLES, EXAMPLES_EXAMPLE)
    rows = read_jsonl(path)
    for r in rows:
        r.setdefault("kind", "apply")
        r.setdefault("source", "authored")
    return rows


def load_tests(path=TESTS):
    rows = read_jsonl(path)
    for r in rows:
        # v0 tests had no suite: they were written from held-out decisions.
        r.setdefault("suite", "unseen")
        r.setdefault("decision_id", "")
        r["suite"] = str(r["suite"]).strip().lower()
    return rows


def _dupes(ids):
    seen, out = set(), []
    for i in ids:
        if i in seen and i not in out:
            out.append(i)
        seen.add(i)
    return out


def validate_decisions(decisions):
    errors = []
    ids = [d.get("decision_id", "") for d in decisions]
    if any(not i for i in ids):
        errors.append("a decision is missing decision_id")
    for d in _dupes(ids):
        errors.append(f"duplicate decision_id: {d}")
    known = set(ids)
    for d in decisions:
        did = d.get("decision_id", "?")
        if d["split"] not in SPLITS:
            errors.append(f"{did}: split must be train or test, got {d['split']!r}")
        if d["status"] not in STATUSES:
            errors.append(f"{did}: status must be active or retired, got {d['status']!r}")
        sup = d.get("superseded_by")
        if sup and sup not in known:
            errors.append(f"{did}: superseded_by points at unknown decision {sup}")
        if sup and d["status"] != "retired":
            errors.append(f"{did}: has superseded_by but status is not retired")
        for field in ("situation", "call"):
            if not str(d.get(field, "")).strip():
                errors.append(f"{did}: missing {field}")
    return errors


def trainable_examples(decisions, examples):
    """Split examples into (usable, errors, warnings).

    An example for a held-out decision is an error, not a warning: it is
    the exact leak the holdout exists to prevent.
    """
    by_id = {d["decision_id"]: d for d in decisions}
    usable, errors, warnings = [], [], []
    for n, ex in enumerate(examples, 1):
        did = ex.get("decision_id", "")
        where = ex.get("example_id") or f"example #{n}"
        if not did:
            errors.append(f"{where}: missing decision_id")
            continue
        d = by_id.get(did)
        if d is None:
            errors.append(f"{where}: unknown decision_id {did}")
            continue
        if ex["kind"] not in KINDS:
            errors.append(f"{where}: kind must be apply or exception, got {ex['kind']!r}")
            continue
        if not str(ex.get("scenario", "")).strip() or not str(ex.get("response", "")).strip():
            errors.append(f"{where}: scenario and response are both required")
            continue
        if d["split"] == "test":
            errors.append(
                f"{where}: trains on held-out decision {did} - "
                "holdout decisions must never appear in training data")
            continue
        if d["status"] == "retired":
            warnings.append(f"{where}: decision {did} is retired - example skipped")
            continue
        usable.append(ex)
    return usable, errors, warnings


def validate_tests(decisions, examples, tests):
    """Return (errors, warnings) for the hidden test set."""
    by_id = {d["decision_id"]: d for d in decisions}
    trained = {e["decision_id"] for e in examples}
    errors, warnings = [], []
    for t in _dupes([t.get("test_id", "") for t in tests]):
        errors.append(f"duplicate test_id: {t}")
    for t in tests:
        tid = t.get("test_id") or "?"
        if not t.get("test_id"):
            errors.append("a test is missing test_id")
        if not str(t.get("question", "")).strip():
            errors.append(f"{tid}: missing question")
        if t["suite"] not in SUITES:
            errors.append(f"{tid}: suite must be one of {', '.join(SUITES)}, got {t['suite']!r}")
            continue
        did = t["decision_id"]
        if not did:
            warnings.append(f"{tid}: no decision_id - cannot check it against the split")
            continue
        d = by_id.get(did)
        if d is None:
            errors.append(f"{tid}: unknown decision_id {did}")
            continue
        if t["suite"] == "unseen" and d["split"] != "test":
            errors.append(
                f"{tid}: suite is unseen but decision {did} is split=train. "
                "Mark the test seen, or hold the decision out.")
        if t["suite"] in ("seen", "exception"):
            if d["split"] != "train":
                errors.append(
                    f"{tid}: suite is {t['suite']} but decision {did} is held out. "
                    "Mark the test unseen.")
            elif did not in trained:
                warnings.append(
                    f"{tid}: decision {did} has no training examples yet, "
                    f"so this is not really a {t['suite']} test")
    return errors, warnings


_WORD = re.compile(r"[a-z0-9']+")


def _norm(text):
    return " ".join(_WORD.findall(str(text).lower()))


def similarity(a, b):
    """(jaccard over word sets, difflib ratio) of two normalised texts."""
    na, nb = _norm(a), _norm(b)
    sa, sb = set(na.split()), set(nb.split())
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return jac, ratio


def near_duplicates(examples, tests):
    """Test questions that are near-copies of a training scenario.

    This is a lexical check. It catches copy-paste and light rewording; it
    does not catch a careful paraphrase. You still own the Leakage check
    column in the sheet.
    """
    hits = []
    for t in tests:
        for ex in examples:
            jac, ratio = similarity(t.get("question", ""), ex.get("scenario", ""))
            if jac >= JACCARD_LIMIT or ratio >= RATIO_LIMIT:
                hits.append({
                    "test_id": t.get("test_id", "?"),
                    "decision_id": ex.get("decision_id", "?"),
                    "jaccard": round(jac, 2),
                    "ratio": round(ratio, 2),
                    "scenario": ex.get("scenario", ""),
                })
    return hits


def stuffed_system_prompt(system_prompt, decisions):
    """System prompt for the prompt-stuffed baseline.

    The base model gets the same information the fine-tune got: every
    active TRAIN decision, written out. Held-out decisions are excluded for
    the same reason they are excluded from training.
    """
    lines = []
    for d in decisions:
        if d["split"] != "train" or d["status"] != "active":
            continue
        line = f"- When: {d['situation'].strip()} My call: {d['call'].strip()}"
        if str(d.get("why", "")).strip():
            line += f" Why: {d['why'].strip()}"
        if str(d.get("exceptions", "")).strip():
            line += f" Exception: {d['exceptions'].strip()}"
        lines.append(line)
    if not lines:
        return system_prompt
    return (system_prompt
            + "\n\nMy standing decisions. Apply them where they fit, and "
              "reason from them where they do not:\n"
            + "\n".join(lines))
