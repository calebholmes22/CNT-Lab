# CNT Techno-Anomaly — Gold Verification

- RA=209.236537, Dec=-1.289745: old_votes=4, new_votes_at_source=nan, W1−W2=0.349, W2−W3=3.768, SIMBAD=✓  ()
- RA=210.910946, Dec=-1.291592: old_votes=4, new_votes_at_source=nan, W1−W2=-0.068, W2−W3=2.526, SIMBAD=—  ()


> CNT-Gloss (2025-10-26T21:37:58Z): Clarify hypothesis; add test recipe & falsifier.
- Hypothesis: …
- Measurement: …
- Expected shift: …
- Falsifier: …

```cnt-test
name: test_gold_verification_summary
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
