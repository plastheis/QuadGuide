"""Artifact path resolution (ARCHITECTURE.md §10.1, §11).

Manifests carry relative artifact paths (e.g. ``siamfc_generic.onnx``). Model
blobs are host-only and gitignored, living under a models directory rather than
in the package. Backends resolve a relative path against ``$EDGECV_MODEL_DIR``
(default ``models``); absolute paths pass through unchanged.

RKNN blobs are compiled per SoC and can't be cross-loaded, so an rknn artifact
path may carry a ``{target}`` token (e.g. ``nanotrack_quant_{target}/x.rknn``)
that the rknn backend fills via :func:`rknn_target` before resolving. onnx/host
artifacts are target-agnostic and carry no token. The mechanism mirrors how a
caller already points ``EDGECV_MODEL_DIR`` at a board's blobs.
"""

from __future__ import annotations

import os
from pathlib import Path

# SoCs whose RKNN compile target we recognise in the device tree, most-specific
# first so e.g. rk3588 wins over a substring collision.
_KNOWN_TARGETS = ("rk3588", "rk3576", "rk3568", "rk3566", "rk3562")
_DEFAULT_TARGET = "rk3588"


def resolve_artifact_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base = Path(os.environ.get("EDGECV_MODEL_DIR", "models"))
    return str(base / p)


def rknn_target() -> str:
    """The RKNN compile target (SoC) used to select per-board ``.rknn`` blobs.

    Resolution order: explicit ``$EDGECV_RKNN_TARGET`` (CI / cross-target builds
    and the knob a per-board config sets) → the SoC read from the Linux device
    tree → the ``rk3588`` default. Detection is best-effort; an unreadable or
    unrecognised device tree falls back to the default.
    """
    env = os.environ.get("EDGECV_RKNN_TARGET")
    if env:
        return env.strip()
    try:
        compat = Path("/proc/device-tree/compatible").read_bytes().decode(
            "ascii", "ignore"
        )
    except OSError:
        return _DEFAULT_TARGET
    for soc in _KNOWN_TARGETS:
        if soc in compat:
            return soc
    return _DEFAULT_TARGET


def apply_rknn_target(path: str) -> str:
    """Substitute the ``{target}`` token in an rknn artifact path, if present."""
    return path.replace("{target}", rknn_target()) if "{target}" in path else path
