# CNT Flash-Proof v9 — Preregistration (frozen)
Run ID: 20251031-004602Z_eaf1aa56
Timestamp (UTC): 20251031-004602Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv

Time col: __t__ | Label col: label | Features: n=322
Metric: agiew_spectral_entropy | Desired W=97 | Smooth=7

Split policy:
  • Forced single-event mode if events_on_full ≤ 2:
    start HOLD at (latest_event_time − PREPAD_SEC=120.0s), clamped to MIN_TRAIN_ROWS=1, MIN_HOLD_ROWS=64.
    Used forced mode? False
  • Else scanned splits 0.99→0.0 (2001 steps), clamped to the same row minima.
    Scan log (first 10): [(np.float64(0.99), 424, 0), (np.float64(0.989505), 424, 0), (np.float64(0.98901), 424, 0), (np.float64(0.988515), 424, 0), (np.float64(0.98802), 424, 0), (np.float64(0.987525), 424, 0), (np.float64(0.98703), 424, 0), (np.float64(0.986535), 424, 0), (np.float64(0.98604), 424, 0), (np.float64(0.985545), 424, 0)]

TRAIN fallback (predeclared): shrink window until ≥ 10 finite metrics.
Tried TRAIN (W, finite) = []; chosen W_train = 5

HOLD fallback (predeclared): shrink window until ≥ 5 finite metrics.
Tried HOLD (W, finite) = [(5, 486)]; chosen W_hold = 5

Tail (TRAIN only): low
Θ* from TRAIN-only grid (LOW [0.01, 0.02, 0.03, 0.05, 0.1]),
with FA cap on TRAIN ≤ 0.2/hr → Θ*=-1000000000.000000 @ q=None (TRAIN FA/hr ≈ 0.000)

Holdout event rule: require ≥ 1 boundary-aware event(s) in hold,
each ≥ 15.0s + margin 0.0s after hold start.

Lead window: [15.0s, 90.0s]; Refractory: 60.0s
Permutations: 500

**Prediction:** On the chosen holdout, CNT will detect ≥ 65% of events within the lead window, with median lead ≥ 15 s, and ≤ 1 FA/hr.

PREDICTIONS_SHA256: f417cf9ab8a995ef4a8d401c1a80a621b20dd9cf62d282486d7f22c522f24d87
