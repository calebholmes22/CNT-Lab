
CNT × 3I/ATLAS — Self-Fetching All-8 Pack (v1.3)
=================================================
Stamp: 20251024-045218Z

Online-first behavior:
- Solar wind from NOAA SWPC; Spectra via SDSS; Ephemerides via JPL Horizons.
- If any are blocked, the cell falls back automatically:
  * Spectra → synthetic lines near CN/CH/C₂
  * Observer-Ring → ANALYTIC PRO ephemeris (Earth & Mars from simple orbits; comet placed to avoid degeneracy)
  * Lightcurve → synthetic drift flip with cooling lulls
- All provenance is logged in out/logs/data_provenance.json.

Observer-Ring modes:
- "pro"            → true comet vectors from Horizons
- "pro_analytic"   → analytic ephemeris fallback (non-degenerate)
- "basic"          → only if both online & analytic paths are unavailable (should not occur in v1.3)

Results packed to out/report.html and the ZIP below.
