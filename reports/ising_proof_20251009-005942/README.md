# CNT Proof: 2D Ising Universality (Binder Crossing + Collapse)

**Date (UTC):** 20251009-005942  
**Hypothesis:** Binder cumulants \(U_4(\beta,L)\) for L∈{16,24,32,48} cross near a single \(\beta^*\) ≈ \(\beta_c\); magnetization collapses with \(\nu=1\), \(\beta/\nu=1/8\).

## Results
- Estimated crossing: **β*** = 0.44093093  
- Exact \(β_c\): **0.44068679**  
- |Δ| = **2.44e-04**  
- Crossings found: **28**  
- Sizes: **[16, 24, 32, 48]**  
- β grid: [0.43, 0.4325, 0.435, 0.4375, 0.44, 0.4425, 0.445, 0.4475, 0.45, 0.4525, 0.455]

## Figures
- Binder crossing: `ising__binder__crossing__20251008-205300.png`
- Scaling collapse: `ising__scaling__collapse__20251008-205543.png`

## Verdict
PASS: crossing near beta_c; collapse rendered.

## Reproduction Notes
- Wolff cluster update; params: {'n_therm': 150, 'n_meas': 240, 'between': 1}
- Saved via `cntlab.io.*` (see manifest for hashes).  

