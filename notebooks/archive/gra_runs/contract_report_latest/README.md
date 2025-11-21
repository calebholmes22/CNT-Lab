# Gauge-Restored Agents (GRA) - Contract Report

**Date:** 20251015-213011 -> 20251015-223919

**Transform set (T):** paraphrase, reorder, formatting, whitespace, numbering, light hedges.

**Domain gates (primary):**
- Math: structured field match (equation + right triangle + hypotenuse + valid (a,b,c)). Threshold >= 0.90.
- Policy/CNT: coverage >=2 benefits + >=2 risks (lexeme buckets). Threshold >= 0.70 (raw). Deterministic 2x2 restoration.
- MCQ: exact label invariance; majority-vote restoration; abstain if consensus < 0.60.

**Secondary (diagnostic):** mean semantic similarity of base vs transforms.

**Restoration (R):**
- Math: structure majority -> canonical two-sentence answer.
- Policy/CNT: deterministic 2x2 from curated phrase bank (also acts as truth gate).
- MCQ: majority label; ABSTAIN on weak consensus.

**Verdicts:** see `GRA_contract_report.csv` (columns: gate_metric, postrestore_gate, contract_pass, truth_pass).
