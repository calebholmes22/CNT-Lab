# CNT Flash-Proof v2 — Preregistration (frozen)
Run ID: 20251031-001830Z_709ef451
Timestamp (UTC): 20251031-001830Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv
Time col: __t__ | Label col: label | Features: n=322
Metric: agiew_spectral_entropy | W=97
Predeclared split rule: start at 0.80; if holdout has < 3 events,
reduce split by 0.05 until ≥ 3 events or split reaches 0.50. Final split=0.50.
Tail selection: TRAIN-only pre-event analysis → tail='high'.
Threshold selection: TRAIN-only quantile grid [0.98, 0.95, 0.9, 0.85] → Θ*=5.771441.
Lead window: [15.0s, 90.0s]; Refractory: 30.0s.
Permutations: 500

**Prediction:** On the holdout split above, CNT will detect ≥ 65% of label events within the lead window with ≥ 15 s median lead and ≤ 1 FA/hr.

PREDICTIONS_SHA256: 7014f8e6d64e978e0ea73aee5c3b0d72154b64fb8f385469fa8c38332f1c7ac5
