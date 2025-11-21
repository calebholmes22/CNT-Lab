# CNT Flash-Proof v5 — Preregistration (frozen)
Run ID: 20251031-002928Z_1cdb0dce
Timestamp (UTC): 20251031-002928Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv
Time col: __t__ | Label col: label | Features: n=322
Metric: agiew_spectral_entropy | Desired W=97 | Smooth=7
TRAIN window fallback (predeclared): shrink W on TRAIN until ≥ 10 finite metric samples,
never inspecting holdout. Tried (W, finite) = [(21, 7), (13, 15)]; Chosen W_eff = 13
Holdout selection (predeclared): latest split in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05] with ≥ 1 boundary-aware events
≥ 15.0s + margin 0.0s after holdout start.
Chosen split: 0.05 (events_in_holdout=0; events_meeting_margin=0)
Tail (TRAIN only): low
Θ* from TRAIN-only quantile grid (LOW [0.02, 0.05, 0.1, 0.15, 0.2]): 4.643157
Lead window: [15.0s, 90.0s]; Refractory: 30.0s
Permutations: 500

**Prediction:** On the chosen holdout, CNT will detect ≥ 65% of events within the lead window, median lead ≥ 15 s, and ≤ 1 false alarm/hour.
