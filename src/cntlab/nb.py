from __future__ import annotations
import os, random
import numpy as np
from .paths import P
from .log import get_logger

def init(seed: int = 1337, numpy_deterministic: bool = True):
    """Initialize a notebook session: seeds, logger, paths echo."""
    if numpy_deterministic:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger = get_logger("cntlab")
    logger.info("CNTLab notebook initialized")
    logger.info(str(P))
    print("→ CNTLab ready.")
    print("   Root:", P.root)
    print("   Figures:", P.figures)
    print("   Tables:", P.tables)
    print("   Metrics:", P.metrics)
    return {"root": str(P.root), "figures": str(P.figures), "tables": str(P.tables), "metrics": str(P.metrics)}
