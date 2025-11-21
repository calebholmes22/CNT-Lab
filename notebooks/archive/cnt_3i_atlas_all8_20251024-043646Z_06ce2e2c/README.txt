
CNT × 3I/ATLAS — Self-Fetching All-8 Pack (v1.2)
=================================================
Stamp: 20251024-043646Z

Fixes in v1.2:
- Align lightcurve & solar wind onto a shared 1-min grid (or safe crop), so arrays match in length and cadence.
- Correct lag axis for unequal lengths in cross-correlation.
- Adaptive coherence window; safe fallback when series are short.

Data behavior:
- Uses local /data first. If missing, fetches: NOAA SWPC (solar_wind), JPL Horizons (lightcurve/observer geometry),
  SDSS (spectrum A/B). If an online source is unavailable, synthesizes realistic stand-ins.
- Logs provenance in /out/logs/data_provenance.json and alignment in /out/logs/align_info.json.
