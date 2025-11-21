# 3I Atlas — Paired Evidence

```cnt-test
name: pair_atlas_data_spectrumspectrum
type: csv_threshold
path: E:\CNT\notebooks\archive\cnt_3i_atlas_all8_20251024-054159Z_3de16d1a\data\spectrum_A.csv
column: wavelength_nm
op: '>'
value: 562.5
agg: mean
gold: true
frozen_at: '2025-10-26T23:11:11Z'
hash: a1f29c0562b611a464ac95b6f23783eacd26e57af623922a30053495d74bb29d
rows: 2200
```

```cnt-test
name: pair_atlas_data_spectrumspectrum_holdout
type: csv_threshold
path: E:\CNT\notebooks\archive\cnt_3i_atlas_all8_20251024-054159Z_3de16d1a\data\spectrum_B.csv
column: wavelength_nm
op: '>'
value: 562.5
agg: mean
gold: true
frozen_at: '2025-10-26T23:11:11Z'
hash: 6c34b24fee52edc525f275b593e18d3bf21b1038cba9ad5f4da96b7b25cb417b
rows: 2200
```

```cnt-test
name: pair_atlas_out_gragra
type: csv_threshold
path: E:\CNT\notebooks\archive\cnt_3i_atlas_all8_20251024-054159Z_3de16d1a\out\tables\gra_trials_A.csv
column: EW_388.3nm
op: '>'
value: -0.060529
agg: mean
gold: true
frozen_at: '2025-10-26T23:11:11Z'
hash: 94a762d3e60c50ae9f28e2def9bfd019e2aa45ca118bb62c5e053e57a9b241ca
rows: 16
```

```cnt-test
name: pair_atlas_out_gragra_holdout
type: csv_threshold
path: E:\CNT\notebooks\archive\cnt_3i_atlas_all8_20251024-054159Z_3de16d1a\out\tables\gra_trials_B.csv
column: EW_388.3nm
op: '>'
value: -0.107483
agg: mean
gold: true
frozen_at: '2025-10-26T23:11:11Z'
hash: d8c3a9a6e5665fb2db484bfc891e2ed0e3f41327a7c4e2120101db943e182983
rows: 16
```
