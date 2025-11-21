from __future__ import annotations
import json, hashlib, time, os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .paths import P

MANIFEST_FILE = P.manifests / "manifest.jsonl"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def log_artifact(path: Path, kind: str, tags: list[str] | None = None, meta: dict | None = None) -> dict:
    """Register an artifact in the manifest with a content hash and metadata."""
    path = Path(path)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,  # e.g., "figure","metrics","table","model"
        "path": str(path.resolve()),
        "size": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() else None,
        "tags": tags or [],
        "meta": meta or {},
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def find_artifacts(kind: str | None = None, tags_all: list[str] | None = None, tags_any: list[str] | None = None) -> list[dict]:
    """Query by kind and tag rules. tags_all must all be present; tags_any at least one present."""
    results = []
    if not MANIFEST_FILE.exists():
        return results
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and rec.get("kind") != kind:
                continue
            rec_tags = set(rec.get("tags", []))
            if tags_all and not set(tags_all).issubset(rec_tags):
                continue
            if tags_any and rec_tags.isdisjoint(tags_any):
                continue
            results.append(rec)
    return results
