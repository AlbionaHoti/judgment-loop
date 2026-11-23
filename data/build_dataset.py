"""Build your judgment-model training set.

Inputs (JSONL, see jl/data.py for the fields):
  data/decisions.jsonl   your repeated decisions, each marked train or test
  data/examples.jsonl    3-5 novel scenarios per TRAIN decision, each
                         answered with your call, in your voice

First run: copy the two *.example.jsonl files to those names and replace
the rows with your own. Until you do, this script builds from the examples
and says so.

What it enforces:
  - No example may use a held-out (split = test) decision. That is an
    error, not a warning.
  - Retired decisions are skipped.
  - If eval/hidden-tests.jsonl exists, test questions that are near-copies
    of a training scenario are reported. Pass --strict to fail on them.

Strip anything you would not upload to a cloud VM before training there:
names, keys, personal context.

Outputs:
  data/decisions-train.jsonl    chat-messages format, one example per line
  data/dataset-manifest.json    hash + counts, so a result can be traced
                                back to the exact data that produced it

Run: python3 data/build_dataset.py [--strict]
"""

import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jl import config, data  # noqa: E402

OUT = os.path.join(data.DATA, "decisions-train.jsonl")
MANIFEST = os.path.join(data.DATA, "dataset-manifest.json")
MINED = os.path.join(data.DATA, "mined-corrections.curated.json")


def load_mined(path=MINED):
    """Optional second source: corrections mined from your own transcripts,
    curated into {"prompt", "response", "decision_id"?} objects."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [{
        "decision_id": item.get("decision_id", ""),
        "scenario": item["prompt"],
        "response": item["response"],
        "kind": item.get("kind", "apply"),
        "source": "mined",
    } for item in items]


def build(strict=False, out_path=OUT, manifest_path=MANIFEST,
          decisions_path=None, examples_path=None, tests_path=data.TESTS,
          mined_path=MINED, log=print):
    cfg = config.load()

    if decisions_path is None:
        decisions_path, d_example = data.resolve(data.DECISIONS, data.DECISIONS_EXAMPLE)
    else:
        d_example = False
    if examples_path is None:
        examples_path, e_example = data.resolve(data.EXAMPLES, data.EXAMPLES_EXAMPLE)
    else:
        e_example = False
    if d_example or e_example:
        log("NOTE: building from the shipped EXAMPLE data. Copy "
            "data/decisions.example.jsonl -> data/decisions.jsonl and "
            "data/examples.example.jsonl -> data/examples.jsonl, then "
            "replace the rows with your own.")

    decisions = data.load_decisions(decisions_path)
    examples = data.load_examples(examples_path)

    errors = data.validate_decisions(decisions)
    usable, ex_errors, warnings = data.trainable_examples(decisions, examples)
    errors += ex_errors

    # Mined corrections without a decision_id are allowed (they are real
    # corrections, not expansions of a sheet row) but cannot be leak-checked
    # against the split, so say how many there are.
    mined = load_mined(mined_path)
    tagged = [m for m in mined if m["decision_id"]]
    untagged = [m for m in mined if not m["decision_id"]]
    m_usable, m_errors, m_warnings = data.trainable_examples(decisions, tagged)
    errors += m_errors
    warnings += m_warnings
    if untagged:
        warnings.append(
            f"{len(untagged)} mined correction(s) have no decision_id - "
            "they cannot be checked against the holdout")
    usable = usable + m_usable + untagged

    dupes = []
    if os.path.exists(tests_path):
        tests = data.load_tests(tests_path)
        t_errors, t_warnings = data.validate_tests(decisions, usable, tests)
        errors += t_errors
        warnings += t_warnings
        dupes = data.near_duplicates(usable, tests)
        for h in dupes:
            msg = (f"possible leakage: test {h['test_id']} is close to a "
                   f"training scenario for {h['decision_id']} "
                   f"(word overlap {h['jaccard']}, similarity {h['ratio']}): "
                   f"{h['scenario'][:80]}")
            (errors if strict else warnings).append(msg)

    for w in warnings:
        log(f"warning: {w}")
    if errors:
        for e in errors:
            log(f"ERROR: {e}")
        raise SystemExit(f"{len(errors)} error(s) - nothing written")
    if not usable:
        raise SystemExit("no usable training examples - nothing written")

    rows = [{"messages": [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": ex["scenario"]},
        {"role": "assistant", "content": ex["response"]},
    ]} for ex in usable]
    random.Random(cfg["eval"]["seed"]).shuffle(rows)
    data.write_jsonl(out_path, rows)

    per_decision = {}
    for ex in usable:
        key = ex["decision_id"] or "(untagged mined)"
        per_decision[key] = per_decision.get(key, 0) + 1
    train_ids = {d["decision_id"] for d in decisions
                 if d["split"] == "train" and d["status"] == "active"}
    thin = sorted(d for d in train_ids if per_decision.get(d, 0) < 3)
    manifest = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": config.file_sha256(out_path),
        "config_hash": config.config_hash(cfg),
        "examples": len(rows),
        "by_source": {s: sum(1 for e in usable if e["source"] == s)
                      for s in sorted({e["source"] for e in usable})},
        "by_kind": {k: sum(1 for e in usable if e["kind"] == k)
                    for k in sorted({e["kind"] for e in usable})},
        "per_decision": dict(sorted(per_decision.items())),
        "train_decisions": len(train_ids),
        "heldout_decisions": sum(1 for d in decisions
                                 if d["split"] == "test" and d["status"] == "active"),
        "retired_decisions": sum(1 for d in decisions if d["status"] == "retired"),
        "built_from_example_data": bool(d_example or e_example),
        "near_duplicate_warnings": len(dupes),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    log(f"wrote {len(rows)} examples across {len(per_decision)} decisions -> {out_path}")
    if thin:
        log(f"thin coverage (<3 examples): {', '.join(thin)}")
    if not manifest["by_kind"].get("exception"):
        log("no exception examples yet - a model that never sees an exception "
            "learns a reflex, not a rule")
    log(f"manifest -> {manifest_path}  (dataset {manifest['dataset_sha256'][:12]})")
    return manifest


if __name__ == "__main__":
    build(strict="--strict" in sys.argv[1:])
