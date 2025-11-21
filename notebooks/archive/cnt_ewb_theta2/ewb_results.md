# CNT Θ* Early-Warning — Results (frozen Θ* = 0.55)

**Registered (UTC):** 2025-10-18T04:09:49.361036Z
**Persistence:** k = 3
**Hazard target:** ~25% positives (bounds 15–35%)

## Top Segments (by AUC then Precision@Θ*)
                                               segment   N  W      AUC  Precision@Theta*  Lead@Hit(steps)  Lead@Any(steps)
                       cnt_cooling_log_20251015_121543  79 32 0.779503          0.353846             22.0             54.0
               cnt_cooling_log_20251015_121543_labeled  79 32 0.779503          0.353846             22.0             54.0
                   cnt_unified_cooling_20251015_133429 367 37 0.771948          0.633333             60.0            337.0
           cnt_unified_cooling_20251015_133429_labeled 367 37 0.771948          0.633333             60.0            337.0
   cnt_unified_cooling_20251015_133429_labeled_labeled 367 37 0.771948          0.633333             60.0            337.0
                   cnt_unified_cooling_20251015_132627 302 32 0.693024          0.512195             63.0            294.0
           cnt_unified_cooling_20251015_132627_labeled 302 32 0.693024          0.512195             63.0            294.0
                cnt_unified_cooling_v2_20251015_134310 319 32 0.261168          0.099237             25.0             25.0
        cnt_unified_cooling_v2_20251015_134310_labeled 319 32 0.261168          0.099237             25.0             25.0
cnt_unified_cooling_v2_20251015_134310_labeled_labeled 319 32 0.261168          0.099237             25.0             25.0

_Source CSV:_ `ewb_cooling_segments_20251017-235217.csv`
## Gauge Invariance Check (headline segments)

                                               segment  AUC_base  Precision_base  Lead@Hit_base  AUC_affine  Precision_affine  Lead@Hit_affine  AUC_decim  Precision_decim  Lead@Hit_decim                                                           sha256
                       cnt_cooling_log_20251015_121543  0.636432        0.500000            NaN    0.636432          0.500000              NaN   0.696667         0.352941            11.0 ee9da42acbd8aed7429ad596b7985d61301603eef00a1634eb0dba3498aec5b2
               cnt_cooling_log_20251015_121543_labeled  0.636432        0.500000            NaN    0.636432          0.500000              NaN   0.696667         0.352941            11.0 ee9da42acbd8aed7429ad596b7985d61301603eef00a1634eb0dba3498aec5b2
                   cnt_unified_cooling_20251015_133429  0.771948        0.633333           60.0    0.772140          0.633333             60.0   0.737873         0.435897            24.0 a1ae0a01537e27d0ed533ce37e04bd72ae09dbff5e2a584d68d26b91a485c056
           cnt_unified_cooling_20251015_133429_labeled  0.771948        0.633333           60.0    0.772140          0.633333             60.0   0.737873         0.435897            24.0 a1ae0a01537e27d0ed533ce37e04bd72ae09dbff5e2a584d68d26b91a485c056
   cnt_unified_cooling_20251015_133429_labeled_labeled  0.771948        0.633333           60.0    0.772140          0.633333             60.0   0.737873         0.435897            24.0 a1ae0a01537e27d0ed533ce37e04bd72ae09dbff5e2a584d68d26b91a485c056
                   cnt_unified_cooling_20251015_132627  0.694329        0.604167           71.0    0.709428          0.604167             71.0   0.782639         0.428571             5.0 1e1df04b94d531a850e7ae5c2c4c8283d7eac945e2574e01b6c8cba9a7916b19
           cnt_unified_cooling_20251015_132627_labeled  0.694329        0.604167           71.0    0.709428          0.604167             71.0   0.782639         0.428571             5.0 1e1df04b94d531a850e7ae5c2c4c8283d7eac945e2574e01b6c8cba9a7916b19
                cnt_unified_cooling_v2_20251015_134310  0.261168        0.099237           25.0    0.261168          0.099237             25.0   0.334371         0.077465             8.0 f6be857fe51ea968d59e66816c0884a6d932ae6951480e5db40402ed7015bb6e
        cnt_unified_cooling_v2_20251015_134310_labeled  0.261168        0.099237           25.0    0.261168          0.099237             25.0   0.334371         0.077465             8.0 f6be857fe51ea968d59e66816c0884a6d932ae6951480e5db40402ed7015bb6e
cnt_unified_cooling_v2_20251015_134310_labeled_labeled  0.261168        0.099237           25.0    0.261168          0.099237             25.0   0.334371         0.077465             8.0 f6be857fe51ea968d59e66816c0884a6d932ae6951480e5db40402ed7015bb6e


> CNT-Gloss (2025-10-26T21:29:12Z): Clarify hypothesis; add test recipe & falsifier.
- Hypothesis: …
- Measurement: …
- Expected shift: …
- Falsifier: …


```cnt-test
name: test_ewb_results
inputs: []
recipe:
  - step: load_data
    path: <fill_me>
  - step: compute_metric
    code: <python or pseudo>
  - step: assert
    expect: <delta_theta> > 0 and p < 0.05
```

> CNT-Gloss (2025-10-26T21:31:49Z): Clarify hypothesis; add test recipe & falsifier.
- Hypothesis: …
- Measurement: …
- Expected shift: …
- Falsifier: …

```cnt-test
name: test_ewb_results
type: csv_threshold
path: E:\CNT\artifacts\CNT_PLI_EC_EO_paired_bootstrap.csv
column: N_pairs
op: '>'
value: 27.0
agg: mean
gold: true
frozen_at: '2025-10-26T21:54:43Z'
hash: 40b4797ed6849bbb54f8a98c3dcd4de973cc29389cc56ebe4ce76dca8506b4aa
rows: 15
skip: true
```

# GOLD Holdout Twin (explicit)
```cnt-test
name: test_ewb_results_holdout
type: csv_threshold
path: E:\CNT\notebooks\archive\cnt_ewb_theta2\runs\ewb_holdout_results.csv
column: AUC
op: '>'
value: 27.0
agg: mean
gold: true
frozen_at: '2025-10-26T22:09:57Z'
hash: f79040bd1a73af8716a40042a3b3bfd643a7daa5043fa093a45c0c7d18189c63
rows: 15
```
