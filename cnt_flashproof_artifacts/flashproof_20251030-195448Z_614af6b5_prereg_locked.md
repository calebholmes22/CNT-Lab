# CNT Flash-Proof Preregistration (frozen)
Run ID: 20251030-195448Z_614af6b5
Timestamp (UTC): 20251030-195448Z
Data: C:\Users\caleb\CNT_Lab\artifacts\tables\migrated__cnt-eeg-labeled-all__68a51fca.csv 
Time column: __t__ | Label column: label | Features: n=322
Split: 80/19 chrono | W=97
Metric: agiew_spectral_entropy | Threshold policy: {"kind": "quantile", "q": 0.98} | Breach tail: low
Frozen Θ*: 2.712630
Lead window: [15.0s, 90.0s] | Refractory: 30.0s
Permutations: 500
Prediction file (next): cnt_flashproof_artifacts\flashproof_20251030-195448Z_614af6b5_predictions.csv

**Prediction:** With W=97 and Θ* learned on the training split only, CNT will detect ≥ 65% of holdout label transitions at ≥ 15 s median lead and ≤ 1 FA/hr.

PREDICTIONS_SHA256: 8fa9d60f8dad37ff30b77801a9b895d0835ca1bd5e615ed9848226cc24cf1807
