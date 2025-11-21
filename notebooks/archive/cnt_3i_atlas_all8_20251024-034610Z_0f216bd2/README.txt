
CNT – 3I/ATLAS All-8 Mega Pack
================================
Stamp: 20251024-034610Z

Folders:
- data/: placeholder CSVs (replace with real spectra, lightcurves, solar wind, observer geometry).
- out/tables/: machine-readable outputs.
- out/figs/: basic figures.
- out/logs/: JSON logs for Θ* alarm, observer-ring vector, nickel hypotheses, origin fingerprint.

What’s implemented (matching the “8”):
1) GRA for spectra (spectrum_A/B.csv) → trials & gra_summary.csv
2) Θ* early-warning on lightcurve.csv → lightcurve_theta.csv + alarm JSON
3) Observer-Ring triangulation (observer_ring.csv) → observer_ring_result.json
4) Plasma–coma resonance (solar_wind.csv) → xcorr/coherence tables + figs
5) Cooling Protocol fit → cooling_peaks.csv, cooling_fits.csv
6) Nickel H1 vs H2 (via GRA gate) → nickel_hypotheses.json
7) Field-age/origin fingerprint → field_origin_fingerprint.json
8) Everything exported with this README + overview.csv

Swap in real data (same column names), then re-run this same single cell:
- Spectra: wavelength_nm, flux
- Lightcurve: t_min, jet_brightness
- Solar wind: t_min, speed_kms, IMF_Bz_nT
- Observer ring: site, x_AU, y_AU, z_AU, jet_PA_deg, jet_PA_sigma_deg


> CNT-Gloss (2025-10-26T21:33:20Z): Clarify hypothesis; add test recipe & falsifier.
- Hypothesis: …
- Measurement: …
- Expected shift: …
- Falsifier: …

```cnt-test
name: test_readme
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
