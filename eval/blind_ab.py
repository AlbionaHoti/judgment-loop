"""Blind A/B: your specialist vs the generic base model, judged without labels.

Flow:
  1. Write unseen scenarios (from your held-out decisions) to hidden-tests.jsonl:
       {"test_id": "T-01", "question": "...", "must_mention": "...", "must_avoid": "..."}
  2. python3 blind_ab.py run
       - asks both models every question (same system prompt, only the
         weights differ), shuffles which answer is A/B per test,
       - writes blind_pairs.csv for judging and answers-key.json (the
         hidden mapping - do not open it before judging).
  3. Judge: fill winner (A/B/Tie), fidelity_a, fidelity_b (0-2, how faithful
     to YOUR call), unsupported_a, unsupported_b (0/1, invented claims)
     in blind_pairs.csv. Score against must_mention / must_avoid.
  4. python3 blind_ab.py reveal
       - unmasks models, prints rows in the sheet's Blind Comparison column
         order, plus the summary.

Requires: Ollama running locally (ollama serve). Standard library only.
"""

import csv
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "hidden-tests.jsonl")
PAIRS_CSV = os.path.join(HERE, "blind_pairs.csv")
KEY_JSON = os.path.join(HERE, "answers-key.json")

SPECIALIST = os.environ.get("JL_SPECIALIST", "my-judgment")
BASE = os.environ.get("JL_BASE", "qwen2.5:3b-instruct")
OLLAMA = os.environ.get("JL_OLLAMA", "http://localhost:11434")

SYS = (
    "You are my judgment model - a private specialist trained on my "
    "repeated decisions. Given a situation, answer with my call: lead with "
    "the decision, give the reason, name the caveat or exception if one "
    "exists. Direct, specific, no filler."
)


def ask(model, question):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"temperature": 0.4},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip()


def run():
    tests = [json.loads(l) for l in open(TESTS) if l.strip()]
    key, rows = {}, []
    for t in tests:
        tid, q = t["test_id"], t["question"]
        print(f"[{tid}] asking {SPECIALIST} ...", flush=True)
        mine = ask(SPECIALIST, q)
        print(f"[{tid}] asking {BASE} ...", flush=True)
        base = ask(BASE, q)
        # deterministic shuffle per test id
        mine_is_a = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 2 == 0
        a, b = (mine, base) if mine_is_a else (base, mine)
        key[tid] = {"A": SPECIALIST if mine_is_a else BASE,
                    "B": BASE if mine_is_a else SPECIALIST}
        rows.append({
            "test_id": tid, "question": q,
            "answer_a": a, "answer_b": b,
            "must_mention": t.get("must_mention", ""),
            "must_avoid": t.get("must_avoid", ""),
            "winner": "", "fidelity_a": "", "fidelity_b": "",
            "unsupported_a": "", "unsupported_b": "",
        })
    with open(PAIRS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(KEY_JSON, "w") as f:
        json.dump(key, f, indent=2)
    print(f"\n{len(rows)} pairs -> {PAIRS_CSV}")
    print("Judge blind: fill winner/fidelity/unsupported. Do NOT open answers-key.json yet.")


def reveal():
    key = json.load(open(KEY_JSON))
    rows = list(csv.DictReader(open(PAIRS_CSV)))
    unjudged = [r["test_id"] for r in rows if not r["winner"].strip()]
    if unjudged:
        sys.exit(f"unjudged tests: {', '.join(unjudged)} - fill winner first")
    wins = losses = ties = 0
    label = {SPECIALIST: "My model", BASE: "Base model"}
    print("\nPaste into the Blind Comparison tab (columns in order):")
    print("Test ID | Question | Answer A | Answer B | A model | B model | Winner | "
          "Fidelity A | Fidelity B | Unsupported A | Unsupported B")
    for r in rows:
        k = key[r["test_id"]]
        w = r["winner"].strip().upper()
        picked = k.get(w) if w in ("A", "B") else None
        if w == "TIE":
            ties += 1
        elif picked == SPECIALIST:
            wins += 1
        else:
            losses += 1
        print(" | ".join([
            r["test_id"], r["question"][:60].replace("|", "/"),
            r["answer_a"][:40].replace("|", "/") + "...",
            r["answer_b"][:40].replace("|", "/") + "...",
            label[k["A"]], label[k["B"]], w.title(),
            r["fidelity_a"], r["fidelity_b"],
            r["unsupported_a"], r["unsupported_b"],
        ]))
    n = len(rows)
    print(f"\nMy model: {wins}W / {losses}L / {ties}T of {n}"
          f"  ({wins/n:.0%} win rate)")
    print("Full answers are in blind_pairs.csv; model identities in answers-key.json.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
    elif cmd == "reveal":
        reveal()
    else:
        sys.exit("usage: blind_ab.py run|reveal")
