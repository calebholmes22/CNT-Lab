# Data Staging (Local)

This repo ships **no data**. Point `configs/omega_srl.yaml` to your local CSVs.
Each dataset must provide:
- a time column (UTC/naive OK, but be consistent),
- one or more value columns,
- an events CSV with start timestamps (and optional end).

Keep raw data read-only; write all outputs to `artifacts/`.
