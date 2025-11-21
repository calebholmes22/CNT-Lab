from __future__ import annotations
import os
from pathlib import Path
import tomli

def _load_settings(root: Path) -> dict:
    cfg = root / "configs" / "settings.toml"
    if cfg.exists():
        with cfg.open("rb") as f:
            return tomli.load(f)
    return {}

class _Paths:
    def __init__(self):
        # Determine root
        env_root = os.environ.get("CNT_LAB_DIR")
        if env_root:
            self.root = Path(env_root).expanduser().resolve()
        else:
            self.root = Path.home() / "CNT_Lab"

        self.settings = _load_settings(self.root)
        p = self.settings.get("paths", {})
        # Basic subdirs (with defaults)
        self.data_raw      = self.root / p.get("data_raw", "data/raw")
        self.data_interim  = self.root / p.get("data_interim", "data/interim")
        self.data_processed= self.root / p.get("data_processed", "data/processed")
        self.artifacts     = self.root / p.get("artifacts_root", "artifacts")
        self.figures       = self.root / p.get("figures", "artifacts/figures")
        self.models        = self.root / p.get("models", "artifacts/models")
        self.metrics       = self.root / p.get("metrics", "artifacts/metrics")
        self.tables        = self.root / p.get("tables", "artifacts/tables")
        self.manifests     = self.root / p.get("manifests", "artifacts/manifests")
        self.logs          = self.root / p.get("logs", "logs")

        # Ensure they exist
        for d in [self.root, self.data_raw, self.data_interim, self.data_processed,
                  self.artifacts, self.figures, self.models, self.metrics,
                  self.tables, self.manifests, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"CNT Paths(root={self.root})"

P = _Paths()
