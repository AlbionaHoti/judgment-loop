# Judgment Loop

Your corrections were a curriculum. You just never enrolled your own model.

## The prompt, the notebook, the blind test

If you came from the video, these are the three things:

1. **The prompt** — [`prompt/extraction-prompt.md`](prompt/extraction-prompt.md), or use the
   [Google Sheet template](https://docs.google.com/spreadsheets/d/1CdrKHDUEvt6gM4lgEKufdj68Vd8eP787UlcurEy3HZs/edit?gid=2073357882#gid=2073357882)
   (File → Make a copy) — it has the prompt plus the dataset, hidden-test,
   and scoring tabs ready to fill.
2. **The notebook** — [`notebook/train_judgment_model.ipynb`](notebook/train_judgment_model.ipynb).
   Open in Google Colab, free T4 GPU, upload your dataset, run top to
   bottom, download your model. No ML background needed.
3. **The blind test** — [`eval/blind_ab.py`](eval/blind_ab.py). Your model
   vs the generic base model, same questions, judged without labels. The
   sheet computes the win rate. A loss is a valid result — publish either way.

Everything runs free: the extraction in your existing AI assistant, the
training on Colab's free tier, the model on your own laptop via Ollama.

Judgment Loop is a small framework for turning the decisions you keep
repeating at your AI assistant — "no, shorter", "staging first", "don't push
without asking" — into a private specialist model you own, then testing it
honestly against the generic base model.

The goal is not a clone of you. It is a falsifiable test: does a model you
control preserve your repeated decisions on unseen situations better than a
generic model? A loss is a valid result. Publish it either way.

## The loop

1. **Extract** — run `prompt/extraction-prompt.md` in your AI assistant. It
   searches whatever history the assistant can actually access and returns up
   to 20 repeated decisions with evidence.
2. **Dataset** — fill the `Decision Dataset` tab in
   `sheet/Judgment-Loop-Test-Sheet.xlsx`. Mark most rows `Train`, hold a few
   out as `Test`. The Test rows never touch training.
3. **Expand** — write 3–5 novel scenarios per Train decision in
   `data/build_dataset.py` and run it to produce `decisions-train.jsonl`.
4. **Train** — open `notebook/train_judgment_model.ipynb` in Google Colab
   (free T4 GPU), upload the JSONL, run top to bottom, download the GGUF.
5. **Deploy** — put the GGUF next to `notebook/Modelfile`, then
   `ollama create my-judgment -f Modelfile`. Inference is now fully local.
   Turn off Wi-Fi and ask it something. It still answers.
6. **Test blind** — write unseen scenarios from your held-out decisions in
   `eval/hidden-tests.jsonl`, then:

   ```
   ollama pull qwen2.5:3b-instruct   # the comparison base model (one-time)
   python3 eval/blind_ab.py run      # queries both models, shuffles A/B
   # judge answers blind in blind_pairs.csv (fill winner + scores)
   python3 eval/blind_ab.py reveal   # unmasks, prints sheet-ready rows
   ```

   Paste the revealed rows into the `Blind Comparison` tab. The `Results`
   tab computes win rate and fidelity on its own.

## Rules that keep the result honest

- The held-out decisions never appear in training data, paraphrased or
  otherwise. The `Leakage check` column in the sheet exists for this.
- Both models get the same system prompt. The only difference is the weights.
- Judge blind. `blind_ab.py` hides which answer came from which model until
  after you have scored them.
- Do not invent numbers. The Results tab is the claim; screenshot it as-is.
- Curate before you upload. Training on Colab sends your dataset to a cloud
  VM — strip names, keys, and anything private first. The model you download
  afterward is yours and runs offline.

## What you need

- Any AI assistant that can search your history (for the extraction prompt)
- A Google account (Colab, free tier is enough for a 3B model)
- [Ollama](https://ollama.com) on your machine, with the base model pulled
  for the blind test: `ollama pull qwen2.5:3b-instruct`
- Python 3 (standard library only)

## Roadmap through the repo

Six steps. Each step uses one file and leaves you holding one artifact you
didn't have before.

| Step | You do | File | You leave with |
|---|---|---|---|
| 1 · Extract | Run the prompt in your AI assistant | `prompt/extraction-prompt.md` | up to 20 repeated decisions, with dates and evidence |
| 2 · Dataset | Fill the Decision Dataset tab, mark Train/Test | `sheet/Judgment-Loop-Test-Sheet.xlsx` | your judgment corpus + a clean holdout |
| 3 · Expand | Write 3–5 novel scenarios per Train decision | `data/build_dataset.py` | `decisions-train.jsonl` — your training set |
| 4 · Train | Run the notebook on a Colab GPU (free T4) | `notebook/train_judgment_model.ipynb` | `my-judgment-q4_k_m.gguf` — weights you own |
| 5 · Deploy | `ollama create my-judgment -f Modelfile` | `notebook/Modelfile` | a specialist that answers with Wi-Fi off |
| 6 · Test blind | Author unseen tests, run, judge, reveal | `eval/blind_ab.py` + `eval/hidden-tests.jsonl` | a win rate and fidelity score you can screenshot |

The order matters: the holdout in step 2 is what makes the number in step 6
honest. If you skip the split, you're grading the model on its homework.

## Where this goes (planned, not promised)

- **Correction mining** — a script that digs real correction moments ("no,
  shorter", "don't push") out of your own session transcripts and curates
  them into training pairs. The highest-signal data source; currently a
  manual step.
- **Fidelity rubric** — a structured scoring guide for step 6 so two people
  judging the same answers land on similar scores.
- **Weekly loop** — re-extract, append fresh corrections, retrain. Judgment
  compounds; the dataset should too.

Issues and PRs welcome — especially your blind-test results, wins or losses.
