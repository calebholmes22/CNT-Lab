# CNT Θ* Early-Warning — Multi-channel Summary (v0.2.1)

**Generated (UTC):** 2025-10-19T04:30:08.240930Z
**Preregistered at (UTC):** 2025-10-18T04:09:49.361036+00:00

**Overall (post-prereg):** Lead@Hit>0 = 40.0% | Median AUC = 0.538 | Invariance OK = 40.0%  (n=15)

## Per-channel
- **gpu_fan_pct** → Lead@Hit>0 = n/a, Median AUC = n/a, Invariance OK = n/a (n=3)
- **gpu_power_w** → Lead@Hit>0 = 25.0%, Median AUC = 0.459, Invariance OK = 25.0% (n=5)
- **gpu_temp_c** → Lead@Hit>0 = n/a, Median AUC = n/a, Invariance OK = n/a (n=5)
- **value** → Lead@Hit>0 = 100.0%, Median AUC = 0.608, Invariance OK = 100.0% (n=1)

## Recent rows
                               segment     channel          status      AUC  Precision  Lead@Hit  Lead@Hit_affine  Lead@Hit_decim  invariance_rate                   mtime_utc
holdout_live_temp_20251018-042119Z.csv         NaN             NaN 0.608434   0.200000      18.0             18.0             NaN              1.0 2025-10-18T04:23:19.660831Z
holdout_live_temp_20251018-042119Z.csv       value              ok 0.608434   0.200000      18.0             18.0             8.0              1.0 2025-10-18T04:23:19.660831Z
 holdout_dual_gpu_20251018-043119Z.csv  gpu_temp_c degenerate_mask      NaN        NaN       NaN             18.0             NaN              NaN 2025-10-18T04:33:20.098578Z
 holdout_dual_gpu_20251018-043119Z.csv gpu_power_w              ok 0.379839   0.038462       NaN              NaN             NaN              0.0 2025-10-18T04:33:20.098578Z
 holdout_dual_gpu_20251018-043731Z.csv  gpu_temp_c degenerate_mask      NaN        NaN       NaN              NaN             NaN              NaN 2025-10-18T04:39:31.599375Z
 holdout_dual_gpu_20251018-043731Z.csv gpu_power_w              ok 0.669758   0.566667      17.0             17.0             5.0              1.0 2025-10-18T04:39:31.599375Z
 holdout_dual_gpu_20251018-044559Z.csv  gpu_temp_c degenerate_mask      NaN        NaN       NaN             17.0             5.0              NaN 2025-10-18T04:49:00.069814Z
 holdout_dual_gpu_20251018-044559Z.csv gpu_power_w              ok 0.538261   0.000000       NaN              NaN            20.0              0.0 2025-10-18T04:49:00.069814Z
 holdout_dual_gpu_20251018-044559Z.csv gpu_fan_pct degenerate_mask      NaN        NaN       NaN              NaN            20.0              NaN 2025-10-18T04:49:00.069814Z
 holdout_dual_gpu_20251018-044903Z.csv  gpu_temp_c degenerate_mask      NaN        NaN       NaN              NaN            20.0              NaN 2025-10-18T04:52:03.255264Z
 holdout_dual_gpu_20251018-044903Z.csv gpu_power_w degenerate_mask      NaN        NaN       NaN              NaN            20.0              NaN 2025-10-18T04:52:03.255264Z
 holdout_dual_gpu_20251018-044903Z.csv gpu_fan_pct degenerate_mask      NaN        NaN       NaN              NaN            20.0              NaN 2025-10-18T04:52:03.255264Z
 holdout_dual_gpu_20251018-045206Z.csv  gpu_temp_c degenerate_mask      NaN        NaN       NaN              NaN            20.0              NaN 2025-10-18T04:55:06.419861Z
 holdout_dual_gpu_20251018-045206Z.csv gpu_power_w              ok 0.308696   0.024390       NaN              NaN             NaN              0.0 2025-10-18T04:55:06.419861Z
 holdout_dual_gpu_20251018-045206Z.csv gpu_fan_pct degenerate_mask      NaN        NaN       NaN              NaN             NaN              NaN 2025-10-18T04:55:06.419861Z