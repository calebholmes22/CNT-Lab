# GRA v0.3 Upgrades — 20251016-100253

This bundle includes:
- **MCQ confidence plots** (`mcq_plots/`): label histogram and majority-confidence bar; ABSTAIN if < 0.60.
- **Policy auto-cite UI** (`policy_autocite/*.html`): 2×2 bullets with per-bullet citations and the best-matching source snippet.
- **Public mini-bench** (`mini_bench_results.csv`): 10 math + 10 policy prompts with invariance/contract/truth signals.

**How to skim:**
- Open `policy_autocite/policy_01.html` and `policy_autocite/cnt_01.html`.
- Peek at `mcq_plots/mcq_01_label_hist.png` and `mcq_plots/mcq_01_majority_conf.png`.
- Scan `mini_bench_results.csv` for PASS/ABSTAIN pattern across domains.

**Contract:** Invariant → Restored → True (or Abstain).
