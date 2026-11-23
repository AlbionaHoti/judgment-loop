"""Blind test: your specialist vs the baselines, judged without labels.

Arms (set in judgment.config.json -> eval.opponents):
  specialist  your fine-tuned model
  base        the generic base model, same system prompt
  stuffed     the generic base model with your TRAIN decisions written
              into its system prompt. This is the baseline that matters:
              if fine-tuning cannot beat pasting the decisions into a
              prompt, the fine-tune is not earning its keep.

Flow:
  1. Write tests to eval/hidden-tests.jsonl (fields and the three suites
     are described in jl/data.py).
  2. python3 eval/blind_ab.py check
       - validates tests against your decisions and training examples,
         reports near-duplicates. No model calls.
  3. python3 eval/blind_ab.py run
       - asks every arm every question at temperature 0, shuffles which
         answer is A/B/C per test, writes blind_pairs.csv for judging and
         answers-key.json (the hidden mapping - do not open it yet).
  4. Judge in blind_pairs.csv. For each answer fill:
       rank_*         1 = closest to YOUR call. Equal numbers = a tie.
       fidelity_*     0-2, how faithful to your call (optional)
       unsupported_*  0/1, did it invent a claim (optional)
  5. python3 eval/blind_ab.py reveal
       - unmasks the arms, prints win/loss/tie per baseline and per suite
         with a 95% interval, and writes eval/results.json.

Requires: Ollama running locally (ollama serve). Standard library only.
"""

import csv
import datetime
import json
import os
import random
import secrets
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jl import config, data, stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV = os.path.join(HERE, "blind_pairs.csv")
KEY_JSON = os.path.join(HERE, "answers-key.json")
RESULTS_JSON = os.path.join(HERE, "results.json")
DATASET_MANIFEST = os.path.join(data.DATA, "dataset-manifest.json")

LETTERS = "ABCDEF"
LABELS = {
    "specialist": "My model",
    "base": "Base model",
    "stuffed": "Base + decisions",
}
VALID_OPPONENTS = ("base", "stuffed")


def ask(cfg, model, system, question):
    ev = cfg["eval"]
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {
            "temperature": ev["temperature"],
            "seed": ev["seed"],
            "num_predict": ev["max_tokens"],
        },
    }).encode()
    req = urllib.request.Request(
        f"{cfg['ollama_url']}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        sys.exit(f"Ollama returned {e.code} for model {model!r}: {detail}\n"
                 f"Is it pulled/created?  ollama list")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Ollama at {cfg['ollama_url']} ({e.reason}). "
                 "Start it with: ollama serve")


def load_inputs(log=print):
    """Load and validate tests against decisions/examples.

    Returns (tests, decisions, errors, warnings, dupes, used_example_data).
    """
    tests = data.load_tests(data.TESTS)
    d_path, d_example = data.resolve(data.DECISIONS, data.DECISIONS_EXAMPLE)
    e_path, e_example = data.resolve(data.EXAMPLES, data.EXAMPLES_EXAMPLE)
    decisions = data.load_decisions(d_path)
    examples = data.load_examples(e_path)
    errors = data.validate_decisions(decisions)
    usable, ex_errors, warnings = data.trainable_examples(decisions, examples)
    errors += ex_errors
    t_errors, t_warnings = data.validate_tests(decisions, usable, tests)
    errors += t_errors
    warnings += t_warnings
    dupes = data.near_duplicates(usable, tests)
    return tests, decisions, errors, warnings, dupes, (d_example or e_example)


def report(errors, warnings, dupes, used_example, log=print):
    if used_example:
        log("NOTE: checking against the shipped EXAMPLE decisions/examples. "
            "Create data/decisions.jsonl and data/examples.jsonl for a real run.")
    for w in warnings:
        log(f"warning: {w}")
    for h in dupes:
        log(f"warning: possible leakage - test {h['test_id']} is close to a "
            f"training scenario for {h['decision_id']} "
            f"(word overlap {h['jaccard']}, similarity {h['ratio']})")
    for e in errors:
        log(f"ERROR: {e}")


def check(log=print):
    tests, _, errors, warnings, dupes, used_example = load_inputs(log)
    report(errors, warnings, dupes, used_example, log)
    by_suite = {}
    for t in tests:
        by_suite[t["suite"]] = by_suite.get(t["suite"], 0) + 1
    log(f"{len(tests)} tests: " + ", ".join(
        f"{by_suite.get(s, 0)} {s}" for s in data.SUITES))
    cfg = config.load()
    need = cfg["eval"]["min_tests_for_claim"]
    if len(tests) < need:
        log(f"note: {len(tests)} tests is below the {need} needed before a "
            "win rate means much. The run still works; the verdict will read "
            "TOO FEW TESTS.")
    if errors:
        raise SystemExit(f"{len(errors)} error(s)")
    return tests


def _has_judging(path):
    if not os.path.exists(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if any(str(v or "").strip() for k, v in r.items() if k.startswith("rank_")):
                return True
    return False


def run(force=False, opponents=None, log=print):
    cfg = config.load()
    tests, decisions, errors, warnings, dupes, used_example = load_inputs(log)
    report(errors, warnings, dupes, used_example, log)
    if errors:
        raise SystemExit(f"{len(errors)} error(s) - fix the tests before running")
    if _has_judging(PAIRS_CSV) and not force:
        raise SystemExit(
            f"{PAIRS_CSV} already has judging in it. Run `reveal`, or pass "
            "--force to throw that judging away and start over.")

    opponents = list(opponents or cfg["eval"]["opponents"])
    for o in opponents:
        if o not in VALID_OPPONENTS:
            raise SystemExit(f"unknown opponent {o!r}: use base and/or stuffed")
    if not opponents:
        raise SystemExit("need at least one opponent")
    sys_prompt = cfg["system_prompt"]
    arms = {"specialist": (cfg["specialist"], sys_prompt)}
    if "base" in opponents:
        arms["base"] = (cfg["base_model"], sys_prompt)
    if "stuffed" in opponents:
        stuffed = data.stuffed_system_prompt(sys_prompt, decisions)
        if stuffed == sys_prompt:
            raise SystemExit("stuffed baseline needs at least one active train decision")
        arms["stuffed"] = (cfg["base_model"], stuffed)
    names = list(arms)
    letters = LETTERS[:len(names)]

    salt = secrets.token_hex(8)
    key = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "salt": salt,
        "arms": {n: {"model": arms[n][0], "label": LABELS[n]} for n in names},
        "config_hash": config.config_hash(cfg),
        "eval": cfg["eval"],
        "used_example_data": used_example,
        "tests": {},
    }
    if os.path.exists(DATASET_MANIFEST):
        with open(DATASET_MANIFEST) as f:
            key["dataset_sha256"] = json.load(f).get("dataset_sha256")

    rows = []
    for t in tests:
        tid, q = t["test_id"], t["question"]
        answers = {}
        for n in names:
            model, system = arms[n]
            log(f"[{tid}] asking {LABELS[n]} ({model}) ...")
            answers[n] = ask(cfg, model, system, q)
        order = names[:]
        random.Random(f"{salt}:{tid}").shuffle(order)
        key["tests"][tid] = {
            "letters": dict(zip(letters, order)),
            "suite": t["suite"],
            "decision_id": t["decision_id"],
        }
        row = {"test_id": tid, "question": q,
               "expected": t.get("expected", ""),
               "must_mention": t.get("must_mention", ""),
               "must_avoid": t.get("must_avoid", "")}
        for L, n in zip(letters, order):
            row[f"answer_{L.lower()}"] = answers[n]
        for field in ("rank", "fidelity", "unsupported"):
            for L in letters:
                row[f"{field}_{L.lower()}"] = ""
        rows.append(row)

    with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(KEY_JSON, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    log(f"\n{len(rows)} tests x {len(names)} answers -> {PAIRS_CSV}")
    log("Judge blind: fill rank_* (1 = closest to your call, equal = tie), "
        "plus fidelity_* and unsupported_* if you use them.")
    log("Do NOT open answers-key.json yet.")


def _int_field(row, name, allowed, tid, required):
    raw = str(row.get(name, "") or "").strip()
    if not raw:
        if required:
            raise SystemExit(f"{tid}: {name} is empty - finish judging first")
        return None
    try:
        v = int(float(raw))
    except ValueError:
        raise SystemExit(f"{tid}: {name} must be a number, got {raw!r}")
    if v not in allowed:
        raise SystemExit(f"{tid}: {name} must be one of {sorted(allowed)}, got {raw!r}")
    return v


def score(rows, key):
    """Turn judged rows + the key into per-test, per-arm scores."""
    names = list(key["arms"])
    k = len(names)
    scored = []
    for r in rows:
        tid = r["test_id"]
        meta = key["tests"].get(tid)
        if meta is None:
            raise SystemExit(f"{tid}: not in answers-key.json - was the CSV edited?")
        per_arm = {}
        for L, n in meta["letters"].items():
            s = L.lower()
            per_arm[n] = {
                "letter": L,
                "answer": r.get(f"answer_{s}", ""),
                "rank": _int_field(r, f"rank_{s}", set(range(1, k + 1)), tid, True),
                "fidelity": _int_field(r, f"fidelity_{s}", {0, 1, 2}, tid, False),
                "unsupported": _int_field(r, f"unsupported_{s}", {0, 1}, tid, False),
            }
        scored.append({"test_id": tid, "question": r["question"],
                       "suite": meta["suite"],
                       "decision_id": meta["decision_id"] or tid,
                       "arms": per_arm})
    return scored


def summarise(scored, opponent, min_tests):
    """Specialist vs one opponent: overall and per suite."""
    def block(items):
        wins = losses = ties = 0
        diffs, clusters = [], []
        for s in items:
            me, them = s["arms"]["specialist"], s["arms"][opponent]
            if me["rank"] < them["rank"]:
                wins += 1
            elif me["rank"] > them["rank"]:
                losses += 1
            else:
                ties += 1
            if me["fidelity"] is not None and them["fidelity"] is not None:
                diffs.append(me["fidelity"] - them["fidelity"])
                clusters.append(s["decision_id"])
        decided = wins + losses
        lo, hi = stats.wilson(wins, decided)
        out = {
            "tests": len(items), "wins": wins, "losses": losses, "ties": ties,
            "win_rate_decided": round(wins / decided, 3) if decided else None,
            "win_rate_ci95": [round(lo, 3), round(hi, 3)] if decided else None,
            "sign_test_p": round(stats.sign_test(wins, losses), 4),
            "verdict": stats.verdict(wins, losses, min_tests),
        }
        if diffs:
            mean, blo, bhi = stats.cluster_bootstrap_mean(diffs, clusters)
            out["fidelity_diff_mean"] = round(mean, 3)
            out["fidelity_diff_ci95"] = (
                [round(blo, 3), round(bhi, 3)] if blo is not None else None)
            out["fidelity_scored"] = len(diffs)
        return out

    result = {"overall": block(scored), "by_suite": {}}
    for suite in data.SUITES:
        items = [s for s in scored if s["suite"] == suite]
        if items:
            result["by_suite"][suite] = block(items)
    return result


def _fmt(b):
    line = f"{b['wins']}W / {b['losses']}L / {b['ties']}T of {b['tests']}"
    if b["win_rate_decided"] is not None:
        lo, hi = b["win_rate_ci95"]
        line += (f"   win rate {b['win_rate_decided']:.0%} of decided "
                 f"(95% CI {lo:.0%}-{hi:.0%}, p={b['sign_test_p']})")
    if "fidelity_diff_mean" in b:
        line += f"   fidelity {b['fidelity_diff_mean']:+.2f}"
        if b.get("fidelity_diff_ci95"):
            flo, fhi = b["fidelity_diff_ci95"]
            line += f" (95% CI {flo:+.2f} to {fhi:+.2f})"
    return line


def reveal(log=print):
    cfg = config.load()
    with open(KEY_JSON, encoding="utf-8") as f:
        key = json.load(f)
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("blind_pairs.csv is empty")
    scored = score(rows, key)
    min_tests = cfg["eval"]["min_tests_for_claim"]
    opponents = [n for n in key["arms"] if n != "specialist"]

    results = {
        "revealed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_created_at": key.get("created_at"),
        "arms": key["arms"],
        "config_hash": key.get("config_hash"),
        "dataset_sha256": key.get("dataset_sha256"),
        "eval": key.get("eval"),
        "used_example_data": key.get("used_example_data", False),
        "vs": {},
        "unsupported_rate": {},
    }
    for n in key["arms"]:
        flags = [s["arms"][n]["unsupported"] for s in scored
                 if s["arms"][n]["unsupported"] is not None]
        if flags:
            results["unsupported_rate"][n] = round(sum(flags) / len(flags), 3)

    for opp in opponents:
        log(f"\n=== {LABELS['specialist']} vs {LABELS[opp]} "
            f"({key['arms'][opp]['model']}) ===")
        log("Sheet rows (Blind Comparison tab, columns in order):")
        log("Test ID | Question | Answer A | Answer B | A model | B model | "
            "Winner | Fidelity A | Fidelity B | Unsupported A | Unsupported B")
        for s in scored:
            me, them = s["arms"]["specialist"], s["arms"][opp]
            winner = ("A" if me["rank"] < them["rank"]
                      else "B" if me["rank"] > them["rank"] else "Tie")

            def cell(v):
                return "" if v is None else str(v)
            log(" | ".join([
                s["test_id"], s["question"][:60].replace("|", "/"),
                me["answer"][:40].replace("|", "/").replace("\n", " ") + "...",
                them["answer"][:40].replace("|", "/").replace("\n", " ") + "...",
                LABELS["specialist"], LABELS[opp], winner,
                cell(me["fidelity"]), cell(them["fidelity"]),
                cell(me["unsupported"]), cell(them["unsupported"]),
            ]))
        summary = summarise(scored, opp, min_tests)
        results["vs"][opp] = summary
        log(f"\noverall    {_fmt(summary['overall'])}")
        for suite, b in summary["by_suite"].items():
            log(f"{suite:<10} {_fmt(b)}")
        log(f"verdict    {summary['overall']['verdict']}")

    if results["unsupported_rate"]:
        log("\nunsupported-claim rate: " + ", ".join(
            f"{LABELS[n]} {v:.0%}" for n, v in results["unsupported_rate"].items()))
    if results["used_example_data"]:
        log("\nNOTE: this run used the shipped EXAMPLE data. It is a smoke "
            "test, not a result.")
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    log(f"\nresults -> {RESULTS_JSON}")
    log("TOO FEW TESTS, INCONCLUSIVE and LOSS are all valid results. Publish it either way.")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    flags = args[1:]
    if cmd == "check":
        check()
    elif cmd == "run":
        opp = None
        for a in flags:
            if a.startswith("--opponents="):
                opp = [o.strip() for o in a.split("=", 1)[1].split(",") if o.strip()]
        run(force="--force" in flags, opponents=opp)
    elif cmd == "reveal":
        reveal()
    else:
        sys.exit("usage: blind_ab.py check | run [--opponents=base,stuffed] [--force] | reveal")
