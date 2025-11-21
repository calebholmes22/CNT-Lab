# CNT Ω-Proof — Preregistration (v1.0)

**Owner:** Telos (CNT)  
**Frozen:** (fill timestamp)  
**SHA-256:** (fill after hashing)

## Pillar A — Synchrony–Risk Law (SRL)
**Claim:** For each dataset \(D_k\) in a fixed set spanning ≥5 domains (e.g., finance shocks, grid alarms, climate extremes, biosignals, mobility bursts), the CNT order parameter \(R(t)\) from a single frozen transform \(\Phi_{CNT}\) yields **median lead-time ≥ L** at **false-alarm rate ≤ α** for pre-defined events \(E_k\), and outperforms strong baselines by a **pre-registered minimally interesting effect (MIE)**.

- **L, α, MIE:** (example) L=30 min (biosignal), 6 h (grid), 3 d (climate), α ≤ 0.2 FA/hr, MIE = +20% lead-time at equal α or +0.05 AUC‑PR.
- **Baselines (fixed):** EWMA/CUSUM, BOCPD, change-point ML, gradient boosting.
- **Splits:** Strict temporal embargo; validation used **once** to set \(\Theta\); test is prospective stream.
- **Nulls:** time-shuffle, phase-randomize, label-shift, sign-flip; all must FAIL to reproduce the effect.
- **Reporting:** worst-case domain performance (not average).

## Pillar B — Gauge‑Restored Agent (GRA)
**Claim:** A decision layer trained to minimize loss over a **paraphrase gauge orbit** for each prompt yields **bounded output variation** under gauge actions and **improved calibration** vs. baseline LLM prompting across ≥5 tasks (QA, extraction, routing, safety), evaluated with a fixed harness.

- **Gauge group:** semantic-preserving edits (synonym swap, shuffles, verbosity, punctuation, casing), intensity tiers 1–3.
- **Metrics:** Invariance gap Δ_inv, ECE, accuracy/F1; MIE = 30% reduction in Δ_inv and −0.02 ECE.
- **Ablations:** remove orbit loss, remove temperature hardening, remove CNT \(R(t)\)-based stabilizer.

## Pillar C — Observer–Glyph Coupling (OGC) [Ethics/IRB gate]
**Claim:** CNT stimuli produce a pre-registered EEG phase‑locking + cross‑frequency coupling signature and **behavioral gain** on n‑back vs matched placebo stimuli, with effect size d ≥ 0.5 (within‑subject), using fixed preprocessing and analysis.

- **Participants:** N ≥ 12 (power for within-subject d=0.5), prereg blocked AB/BA order.
- **Metrics:** PLV(θ), CFC(θ–γ), d′ on task; mixed‑effects model.

## Global Rules
- **No post‑hoc tuning on held‑out/prospective.**
- **All configs in `configs/*.yaml` are frozen pre‑run.**
- **All artifacts hash‑chained; any edit re‑hashes.**
