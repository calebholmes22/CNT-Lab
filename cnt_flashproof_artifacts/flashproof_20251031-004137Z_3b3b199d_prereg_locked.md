# CNT Flash-Proof v8 — Preregistration (frozen)
Run ID: 20251031-004137Z_3b3b199d
Timestamp (UTC): 20251031-004137Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv

Time col: __t__ | Label col: label | Features: n=322
Metric: agiew_spectral_entropy | Desired W=97 | Smooth=7

Split scan (predeclared): SPLIT_MAX=0.99 → SPLIT_MIN=0.0 (2001 steps),
clamped to MIN_TRAIN_ROWS=16, MIN_HOLD_ROWS=64.
Scan log (first 10): [(np.float64(0.99), 424, 0), (np.float64(0.989505), 424, 0), (np.float64(0.98901), 424, 0), (np.float64(0.988515), 424, 0), (np.float64(0.98802), 424, 0), (np.float64(0.987525), 424, 0), (np.float64(0.98703), 424, 0), (np.float64(0.986535), 424, 0), (np.float64(0.98604), 424, 0), (np.float64(0.985545), 424, 0)] ... total 2001 checked.
Fallback used (event-targeted)? False

TRAIN fallback (predeclared): shrink window on TRAIN until ≥ 10 finite metrics.
Tried TRAIN (W, finite) = [(13, 7), (9, 11)]; chosen W_train = 9

HOLD fallback (predeclared): if HOLD too short for W_train, shrink to get ≥ 5 finite metrics.
Tried HOLD (W, finite) = [(9, 467)]; chosen W_hold = 9

Tail (TRAIN only): high
Θ* from TRAIN-only grid (HIGH [0.98, 0.95, 0.9, 0.85]),
with FA cap on TRAIN ≤ 0.2/hr → Θ*=4.674169 @ q=0.98 (TRAIN FA/hr ≈ 0.000)

Holdout event rule: require ≥ 1 boundary-aware events in hold,
each ≥ 0.0s + margin 0.0s after hold start.

Lead window: [0.0s, 90.0s]; Refractory: 60.0s
Permutations: 500

**Prediction:** On the chosen holdout, CNT will detect ≥ 65% of events within the lead window, with median lead ≥ 15 s, and ≤ 1 FA/hr.

PREDICTIONS_SHA256: 98930fa8cb2cda09d389d520f39107e66565152b76dda4591ae1f627e18f1135
