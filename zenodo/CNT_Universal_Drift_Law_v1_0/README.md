# CNT Universal Drift Law v1.0 — Cross-Domain Time Series Panel

**Project:** Cognitive Nexus Theory (CNT)  
**Module:** Universal Drift Law (UDL)  
**Version:** v1.0  
**Author:** Caleb "Telos" Holmes  
**Generated:** 2025-11-17 13:13:27Z UTC

## Overview

This bundle contains the artifacts for the **CNT Universal Drift Law v1.0**
analysis. The UDL conjecture is that local drift exponents (α_short) from
diverse time series (EEG, markets, synthetic fields, sensors, etc.) cluster
into a small number of stable drift regimes that are shared across domains.

## Contents

- `artifacts_universal_drift_v1/`
    Full artifacts for the latest cnt_universal_drift_v1 run
    (20251116-160912Z). This directory typically includes:
    - JSON summaries of drift metrics and GMM fits by source.
    - PNG figures (e.g. α_short histograms, regime plots, per-domain
      panels).
    - Any additional run metadata saved by the pipeline.

- `cnt_laws/`
    - `cnt_laws_table*.csv/json`: snapshot of the CNT laws table from
      20251116-171022Z, including the entry for the Universal Drift Law.

- `README.md` — this file.
- `CITATION.md` — how to cite this dataset.
- `LICENSE_CODE.txt` — license for any code components.

## Reproducing / Extending

The core idea is:
- Drift metrics (α_short, α_long, classes) are computed per window and
  aggregated by domain.
- A global mixture model (e.g. GMM) over α_short reveals a small set of
  drift regimes (field / noise / hazard, etc.).
- The same bands appear across heterogeneous sources.

To extend or challenge the UDL:
1. Load the JSON/CSV summaries under `artifacts_universal_drift_v1/`.
2. Refit your own mixture model over α_short or related metrics.
3. Add new domains and compare their drift regime distributions to the
   reference patterns.

## Licensing

- Data and numerical results: CC BY 4.0 (attribution required).
- Code (if any scripts/notebooks are added later): MIT License
  (see LICENSE_CODE.txt).

## Acknowledgements

Analytical and documentation assistance was provided by an AI research
assistant ("Aetheron") used as a tool. All scientific claims and
interpretations remain the responsibility of the human author.