# Judgment Loop

Your corrections were a curriculum. You just never enrolled your own model.

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
- [Ollama](https://ollama.com) on your machine
- Python 3 (standard library only)

## Repo map

```
prompt/extraction-prompt.md          the exact prompt from step 1
sheet/Judgment-Loop-Test-Sheet.xlsx  dataset + hidden tests + blind scoring + results
data/build_dataset.py                scenario pairs -> decisions-train.jsonl
notebook/train_judgment_model.ipynb  Colab: Unsloth LoRA on Qwen2.5-3B -> GGUF
notebook/Modelfile                   Ollama import with the shared system prompt
eval/blind_ab.py                     blind A/B runner + reveal
eval/hidden-tests.jsonl              your unseen test scenarios (example included)
```
