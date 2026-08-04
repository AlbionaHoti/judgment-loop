"""Build your judgment-model training set.

Replace the EXAMPLE pairs below with your own: for each TRAIN decision in
your sheet, write 3-5 novel scenarios (situations the evidence never
mentioned) and answer each with your call, in your voice.

Rules:
  - Decisions you marked `Test` in the sheet must NOT appear here,
    paraphrased or otherwise. They are the holdout.
  - Strip anything you would not upload to a cloud VM: names, keys,
    personal context. Training runs on Colab; the model comes back to you,
    the dataset goes up.

Output: decisions-train.jsonl (chat-messages format, one example per line).
Run: python3 build_dataset.py
"""

import json
import os
import random

SYS = (
    "You are my judgment model - a private specialist trained on my "
    "repeated decisions. Given a situation, answer with my call: lead with "
    "the decision, give the reason, name the caveat or exception if one "
    "exists. Direct, specific, no filler."
)

# (situation, your_call) — REPLACE these examples with your own pairs.
PAIRS = [
    ("The draft buries the one caveat that would change the decision under three paragraphs of polite framing.",
     "State the caveat in the first sentence and cut the padding. The caveat changes the decision; tone must not obscure it. Politeness can survive - but after the point, not instead of it."),
    ("The report is ready but one number in it comes from memory, not from the source.",
     "Hold the report until the number is verified against the live source. A remembered number presented as fact is an invented number. If it can't be verified today, ship the report with the number marked unverified."),
    ("We could ship tonight if we skip the second review pass.",
     "Skip nothing; ship tomorrow. The second pass exists because the first pass misses things - that's not a schedule problem, it's the reason the step exists. If tonight truly matters, cut scope, not checks."),
    ("The prototype impressed everyone. Should we announce the feature as available?",
     "No. Announce what users can touch today, nothing more. A prototype is a promise with no delivery date; announcing it as available converts excitement into a credibility debt that comes due immediately."),
]


def main():
    random.seed(11)
    out_path = os.path.join(os.path.dirname(__file__), "decisions-train.jsonl")
    rows = []
    for user, assistant in PAIRS:
        rows.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})

    # Optional second source: corrections mined from your own transcripts,
    # curated into {"prompt": ..., "response": ...} objects.
    curated = os.path.join(os.path.dirname(__file__), "mined-corrections.curated.json")
    n_mined = 0
    if os.path.exists(curated):
        with open(curated) as f:
            for item in json.load(f):
                rows.append({"messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]})
                n_mined += 1

    random.shuffle(rows)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples ({len(PAIRS)} decision pairs, {n_mined} mined) -> {out_path}")


if __name__ == "__main__":
    main()
