
CNT × 3I/ATLAS — Upgraded All‑8 Pack
====================================
Stamp: 20251024-042112Z

Includes:
- WCS wavelength from FITS headers (CRVAL/CDELT/CRPIX) if spectrum_*.fits present.
- Multi‑feature GRA (CN/CH/C₂ + custom lines) with solar‑analog continuum normalization.
- Plasma+Phase: coherence with FDR correction, plus phase (from xcorr peak mapped to f).
- Observer‑Ring Pro: if observer_ring.csv includes comet vectors (cx_AU,cy_AU,cz_AU), true sky‑plane geometry is used; else falls back.
- Auto HTML report at out/report.html.

Data files expected in /data (any missing are synthesized):
- spectrum_A.csv / spectrum_B.csv OR spectrum_A.fits / spectrum_B.fits (must yield wavelength_nm & flux; FITS WCS supported)
- lightcurve.csv  [t_min, jet_brightness]
- solar_wind.csv  [t_min, speed_kms, IMF_Bz_nT]
- observer_ring.csv [site, x_AU, y_AU, z_AU, jet_PA_deg, jet_PA_sigma_deg] (+ optional cx_AU,cy_AU,cz_AU for Pro mode)


> CNT-Gloss (2025-10-26T21:31:49Z): Clarify hypothesis; add test recipe & falsifier.
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
