# CALM v1.1 — EEG Cognitive Alphabet (K=3) and Zoom Law

**Project:** Cognitive Nexus Theory (CNT)  
**Module:** CALM — Cognitive Alphabet for Latent Microstates  
**Version:** v1.1  
**Author:** Caleb "Telos" Holmes  
**Generated:** 2025-11-17 06:47:14Z UTC

## Overview

CALM v1.1 proposes that resting-state human EEG can be represented by
a compact, subject-aware alphabet of **K=3 microstates**. These states:

- Are reproducible across recordings.
- Are subject-specific yet comparable across people.
- Support a coarse→fine "zoom" hierarchy where refining one key state
  (S1) yields a robust cross-subject gain in predictive/compression
  metrics.

This bundle contains the main findings PDF, figures, and analysis
artifacts needed to reproduce the v1.1 results.

## Contents

- `CALM_Findings_v1_1.pdf` — main report (methods, results, figures).
- `CALM_K3_Mechanism_Figure.png` — visualization of the three-state
  alphabet and its mechanism diagram.
- `CALM_Hierarchical_Grammar.pdf` — hierarchical coarse→fine map and
  zoom grammar.
- `analysis/zoom_S1_summary.json` — summary of the S1 zoom statistic
  across subjects.
- `analysis/zoom_S1_subjects.csv` — per-subject zoom metrics.
- `analysis/calm_eval_min.json` — minimal evaluation scoreboard.
- `analysis/calm_eval_min.py` — tiny evaluator for applying CALM to
  new EEG datasets.
- `CALM_10k_KIT/` (if present) — optional public dashboard + utilities.

## Reproducing / Using CALM

1. Inspect `CALM_Findings_v1_1.pdf` for the conceptual overview and
   detailed methods.
2. Use `calm_eval_min.py` with your own EEG (EDF/BDF converted to the
   expected format) to estimate CALM states and compute zoom metrics.
3. Compare your results to `zoom_S1_summary.json` and
   `calm_eval_min.json` to test the CALM v1.1 claims on new cohorts.

## Licensing

- Data and figures: CC BY 4.0 (attribution required).
- Code (evaluator scripts, helpers): MIT License (see LICENSE_CODE.txt).