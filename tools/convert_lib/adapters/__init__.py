"""Importing this package registers every adapter. Add new adapters here."""

from __future__ import annotations

from . import (
    nanotrack,  # noqa: F401  (import side effect: registers the adapter)
    siamfc,  # noqa: F401  (import side effect: registers the adapter)
    yolo,  # noqa: F401  (import side effect: registers the adapter)
)
