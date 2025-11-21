CNT Θ* Early-Warning — Publish Pack v0.1
Built: 2025-10-18T04:15:16.858939Z

Contents:
- ewb_prereg.yaml / .yml  → frozen claim + parameters + datasets + holdout plan
- ewb_results.md / .html   → headline results + invariance table
- runs/*.csv               → latest run summaries
- data/examples/*.csv      → top-5 cooling segments (raw), for quick reproduction
- checksums.sha256         → SHA-256 for all files above

Fixed parameters:
- Θ* = 0.55, persistence k=3
- Hazard target ≈ 25% (bounds 15–35%)
- Score = returns-domain Θ² (variance + AC1 + spectral redness → [0,1])
- Gauge tests: affine (a=2.5,b=7.0), decimate(f=2)

How to reproduce (quick):
1) Load any CSV from data/examples/ (column 'value'); run the v2.3 evaluator cell.
2) Confirm AUC > 0.5 and Lead@Hit > 0 on at least several segments.

Holdout plan:
- Evaluate all *new* cooling segments after the prereg date with the frozen Θ* and k.
- Success if ≥60% have Lead@Hit > 0 and median AUC ≥ 0.60, with gauge invariance preserved.
