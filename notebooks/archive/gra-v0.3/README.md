# Gauge-Restored Agents (GRA) v0.3
**Contract:** Invariant → Restored → True (or Abstain)

This repo ships a tiny, enforceable safety contract for LLM answers:
1) Keep meaning invariant under symbol-preserving transformations,
2) Restore outputs to a domain-safe format when form wobbles,
3) Verify claims (or ABSTAIN if truth is uncertain).

## Domains (v0.3)
- **Math:** structured gate (equation + right triangle + hypotenuse + valid (a,b,c)) + canonical restorer.
- **Policy/CNT:** deterministic 2×2 (2 benefits, 2 risks) with citations + **hybrid truth** (semantic OR stem/alias match).
- **MCQ:** exact-label invariance with **majority consensus** and principled **ABSTAIN** when consensus < 0.60.

## Quickstart
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python demo/demo.py --out runs/demo_20251016-101054
```
Artifacts: CSV table, policy auto-cite HTML views, and MCQ label histogram/majority confidence.

## Contract Details
- **Transform set (𝒯):** paraphrase, reorder, formatting, whitespace, numbering, light hedges.
- **Primary gate:** domain-specific (Math=structure; Policy/CNT=2×2 coverage; MCQ=label invariance).
- **Restoration (R):** Math→canonical; Policy/CNT→deterministic 2×2 w/ citations; MCQ→majority or abstain.
- **Truth:** math numeric/field check; policy hybrid (semantic ≥ τ OR stem/alias match) on cited snippets; MCQ vs key when available.

## Limits & Roadmap
- Citations use a curated snippet bank (extend with retrieval/NLI to broaden coverage).
- Add new domains by defining Gate → Restoration → Truth & wiring into `gra/runner.py`.

MIT License.
\n\n> CNT-Gloss (2025-10-26T21:18:00Z): Clarify hypothesis; add test recipe & falsifier.\n- Hypothesis: …\n- Measurement: …\n- Expected shift: …\n- Falsifier: …\n

> CNT-Gloss (2025-10-26T21:51:27Z): Clarify hypothesis; add test recipe & falsifier.
- Hypothesis: …
- Measurement: …
- Expected shift: …
- Falsifier: …

```cnt-test
name: test_readme
type: csv_threshold
path: <fill_me_csv_path>
column: <metric_column>
op: '>'
value: 0.0
agg: mean
skip: true
```