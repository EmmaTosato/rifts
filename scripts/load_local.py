"""Local dataset loader: read test rows exported by the DRIFTS bridge.

Drop-in replacement for ``load_ucr.load_dataset`` when the models in
``current-state/models/`` come from the DRIFTS classification pipeline instead
of the UCR archive. Each model ``<token>.joblib`` has a sibling
``<token>_test.csv`` (validation rows, features only, no header) which this
loader returns as ``X_test``.

The greedy sweep only needs ``X_test`` and ``n_test`` (the target class is the
forest's own prediction, ``def3.predict(X_test)``), so no labels are required.
"""
from __future__ import annotations

import numpy as np

from _paths import STATE_ROOT  # noqa: E402

MODELS = STATE_ROOT / "models"


def load_dataset(name: str) -> dict:
    """Return {name, X_test, n_test} read from ``models/<name>_test.csv``."""
    path = MODELS / f"{name}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found for model '{name}': {path}")
    X = np.loadtxt(path, delimiter=",", ndmin=2).astype(np.float32)
    return {"name": name, "X_test": X, "n_test": len(X)}
