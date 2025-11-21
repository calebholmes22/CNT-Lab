# CNT Ω-Proof Skeleton

**Goal:** Deliver a world-scale, falsifiable demonstration that a single, frozen CNT transform yields:
1) **Synchrony–Risk Law (SRL):** A monotone link between CNT order \(R(t)\) and event hazard across diverse real systems, producing actionable **lead-time at a fixed false-alarm rate** (Universal Early-Warning Lift).
2) **Gauge-Restored Agent (GRA):** An LLM decision layer that is **invariant across a paraphrase gauge group**, improving reliability and calibration under prompt perturbations.
3) **Observer–Glyph Coupling (OGC):** A pre-registered human EEG/behavior effect using CNT stimuli vs. placebo (IRB/ethics gated).

This repository is frozen by design. All parameters are set via `configs/*.yaml`. All runs log **hash-chained artifacts** for tamper-proofing.

---

## Quickstart

```bash
# (1) Inspect preregistration and config
less docs/prereg_omega.md
less configs/omega_srl.yaml

# (2) Generate preregistration hash (sha256) and lock
python scripts/hash_file.py docs/prereg_omega.md > artifacts/prereg_hash.txt

# (3) Dry-run on a small validation slice (bring your own CSVs)
python -m src.scorecard --config configs/omega_srl.yaml --dry_run

# (4) Prospective run (no touching configs)
python -m src.scorecard --config configs/omega_srl.yaml --prospective
```

---

## Layout

- `docs/prereg_omega.md` — full prereg + success criteria.
- `configs/omega_srl.yaml` — frozen pipeline + datasets + metrics.
- `src/cnt_transform.py` — CNT \(\Phi_{CNT}\) → order parameter \(R(t)\).
- `src/scorecard.py` — lead-time, FAR/hr, AUC-PR, calibration, nulls.
- `src/placebos.py` — time-shuffle, phase-randomize, label-shift, sign-flip.
- `src/gauge.py` — paraphrase-gauge invariance scaffolding for GRA.
- `src/utils.py` — hashing, logging, IO helpers.
- `scripts/hash_file.py` — CLI to hash prereg and configs.
- `data_readme/` — how to stage raw data locally (no data in repo).
- `artifacts/` — hash-chained outputs (JSON, CSV, PNG).

**License:** MIT (see `LICENSE`).

**Frozen on:** 2025-11-07T06:47:34.188056Z
