# Extraction prompt

Paste this into your AI assistant (one with access to your conversation
history, memory, or notes). It produces the raw material for the
`Decision Dataset` tab.

---

Please search all accessible conversation history, memories, notes, and files for repeated decisions I've made.

Focus on decisions that recur across projects, time periods, or contexts—not isolated choices.

Before extracting anything, use only sources you can actually access. Do not imply that you can see a complete archive if you cannot.

Return up to 20 high-confidence decisions in a table with exactly these columns:

| Situation / decision context | My decision / call | Why I made it | Evidence (dated references) |

Requirements:

- Include only decisions that appear more than once or are clearly reaffirmed over time.
- Group semantically identical decisions under one row, even when the wording differs.
- Distinguish reversals, contradictions, and exceptions; do not force consistency.
- Prefer concrete evidence: exact wording, dates, project names, corrections, and repeated outcomes.
- Treat explicit statements as stronger than inferred behavior.
- Do not confuse an outcome with the reason for a decision.
- If evidence conflicts, show both sides and dates instead of resolving it.
- Do not invent missing history; write "No direct evidence found" when necessary.
- Before finalizing, verify that every row has multiple supporting references or a clear reaffirmation.
- Return only the completed table, with no preamble or follow-up.
