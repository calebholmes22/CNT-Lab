# CNT Flash-Proof v11 — Preregistration (frozen)
Run ID: 20251031-005316Z_d1c9040e
Time: 20251031-005316Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv

Event mode: auto | chosen='file' | events_on_full=5 | audit(top): [('file', 5), ('label', 0)]
Split policy: HOLD begins 300.0s before latest event (clamped). idx=64, hold_range=[64.0, 487.0], events_in_hold=4

Train window fallback: tried [(58, 10)] → W_train=58
Hold window fallback:  tried [(58, 370)] → W_hold=58

Tail(TRAIN): high | Θ*=5.771441 @ q=0.98 | TRAIN FA/hr≈0.000
Lead window: [15.0s, 90.0s] | Refractory: 60.0s | N_perm=500

**Prediction:** CNT detects ≥65% of events within lead window, median lead ≥15 s, ≤1 FA/hr.
