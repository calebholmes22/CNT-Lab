# CNT Flash-Proof v14 — Preregistration (frozen)
Run ID: 20251031-010112Z_fac3196c
Time: 20251031-010112Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv

Label audit(top): [('label', 0)]
Chosen events: ENERGY(up@q=0.85) | total_on_full=4 | derived_used=True
Derived energy rule (if used): {"q": 0.85, "thr": 0.0032973407994512836, "n_train": 4, "med_sep": 27.0, "direction": "up", "t0": 0.0, "t1": 389.0}
Split: HOLD starts 300.0s before latest event; idx=45; HOLD=[45.0, 487.0] | events_in_hold=4

TRAIN fallback: [(31, 18)] → W_train=31
HOLD  fallback: [(31, 416)]  → W_hold=31

Tail(TRAIN): low | Θ*=4.163040 @ q=0.01 | TRAIN FA/hr≈81.818
Persistence: MIN_BREACH_DUR_SEC=20.0 | REFRACTORY=120.0s
Lead window: [15.0s, 90.0s] | N_perm=500

**Prediction:** CNT detects ≥65% of events within lead window, median lead ≥15 s, ≤1 FA/hr.

PREDICTIONS_SHA256: df1f3ba72bd173f17b74338d988bac3e7208651accdecccefc87d9ff6a6b0cdf
