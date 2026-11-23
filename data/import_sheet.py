"""Turn the sheet tabs into the JSONL files the scripts read.

In the sheet: File -> Download -> Comma-separated values (.csv), once on
the `Decision Dataset` tab and once on the `Hidden Test Set` tab. Then:

  python3 data/import_sheet.py decisions.csv            # -> data/decisions.jsonl
  python3 data/import_sheet.py hidden-tests.csv         # -> eval/hidden-tests.jsonl

The script recognises the tab from its header row. It refuses to overwrite
an existing file unless you pass --force.

Rows with `Keep?` = No, and the EXAMPLE rows, are skipped.

Suites for tests: add a `Suite` column to the Hidden Test Set tab to set
seen / unseen / exception by hand. Without it, a test is `unseen` when its
source decision is held out and `seen` when the decision is in training.
Import decisions first so that lookup works.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jl import data  # noqa: E402


def _key(header):
    return "".join(ch for ch in header.lower() if ch.isalnum())


def _get(row, *names):
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def decisions_from_rows(rows):
    out = []
    for r in rows:
        did = _get(r, "decisionid")
        if not did or did.upper() == "EXAMPLE":
            continue
        if _get(r, "keep").lower() in ("no", "n", "false", "0"):
            continue
        d = {
            "decision_id": did,
            "domain": _get(r, "domain"),
            "situation": _get(r, "situationcontext", "situation"),
            "call": _get(r, "mycall", "call"),
            "why": _get(r, "why"),
            "evidence": _get(r, "evidencedatedreferences", "evidence"),
            "confidence": _get(r, "confidence"),
            "split": (_get(r, "split") or "train").lower(),
            "status": (_get(r, "status") or "active").lower(),
        }
        for opt in ("exceptions", "supersededby"):
            v = _get(r, opt)
            if v:
                d["superseded_by" if opt == "supersededby" else opt] = v
        out.append(d)
    return out


def tests_from_rows(rows, decisions):
    split = {d["decision_id"]: d["split"] for d in decisions}
    out = []
    for r in rows:
        tid = _get(r, "testid")
        if not tid or tid.upper() == "T-EXAMPLE":
            continue
        did = _get(r, "sourcedecisionid", "decisionid")
        suite = _get(r, "suite").lower()
        if not suite:
            suite = "seen" if split.get(did) == "train" else "unseen"
        out.append({
            "test_id": tid,
            "decision_id": did,
            "suite": suite,
            "question": _get(r, "newunseensituation", "question"),
            "expected": _get(r, "expectedprinciplecall", "expected"),
            "must_mention": _get(r, "mustmention"),
            "must_avoid": _get(r, "mustavoid"),
        })
    return out


def main(argv):
    force = "--force" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        sys.exit("usage: import_sheet.py <exported-tab.csv> [--force]")
    with open(paths[0], newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [_key(h) for h in next(reader)]
        rows = [dict(zip(header, line)) for line in reader]

    if "decisionid" in header and "testid" not in header:
        out_rows, dest = decisions_from_rows(rows), data.DECISIONS
        errors = data.validate_decisions(out_rows)
    elif "testid" in header:
        if not os.path.exists(data.DECISIONS):
            sys.exit("import the Decision Dataset tab first - tests are checked against it")
        decisions = data.load_decisions(data.DECISIONS)
        out_rows, dest = tests_from_rows(rows, decisions), data.TESTS
        examples = data.load_examples(data.EXAMPLES) if os.path.exists(data.EXAMPLES) else []
        usable, _, _ = data.trainable_examples(decisions, examples)
        errors, warnings = data.validate_tests(decisions, usable, out_rows)
        for w in warnings:
            print(f"warning: {w}")
    else:
        sys.exit("could not recognise the tab: expected a `Decision ID` or `Test ID` column")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(f"{len(errors)} error(s) - nothing written")
    if not out_rows:
        sys.exit("no usable rows found (EXAMPLE rows and Keep? = No are skipped)")
    if os.path.exists(dest) and not force:
        # The repo ships hidden-tests.jsonl with T-EX-* placeholders; those
        # are safe to replace. Anything else is your work.
        existing = data.read_jsonl(dest)
        placeholder = dest == data.TESTS and all(
            str(r.get("test_id", "")).startswith("T-EX") for r in existing)
        if not placeholder:
            sys.exit(f"{dest} already exists - pass --force to overwrite it")
    data.write_jsonl(dest, out_rows)
    print(f"wrote {len(out_rows)} rows -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1:])
