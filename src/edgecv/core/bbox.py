"""Bounding-box types. BoundingBox is always normalised 0–1; PixelBox is the
explicit pixel-space helper used only at the pixel boundary. Never let a raw
pixel tuple masquerade as a BoundingBox (see ARCHITECTURE.md §5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in pixel coordinates (sub-pixel allowed)."""

    x: float  # top-left x, pixels
    y: float  # top-left y, pixels
    w: float  # width, pixels
    h: float  # height, pixels

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box, normalised to 0–1 in both dimensions."""

    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width,  normalised 0–1
    h: float  # height, normalised 0–1

    def __post_init__(self) -> None:
        if self.w < 0.0 or self.h < 0.0:
            raise ValueError(f"BoundingBox dimensions must be non-negative: {self!r}")

    def to_pixels(self, width: int, height: int) -> PixelBox:
        return PixelBox(
            x=self.x * width,
            y=self.y * height,
            w=self.w * width,
            h=self.h * height,
        )

    @classmethod
    def from_pixels(cls, box: PixelBox, width: int, height: int) -> BoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return cls(
            x=box.x / width,
            y=box.y / height,
            w=box.w / width,
            h=box.h / height,
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clamp(self) -> BoundingBox:
        """Return a copy fully contained in the unit square.

        Note: this both pins x,y into [0,1] AND shrinks w,h to fit. It is therefore
        lossy for off-frame boxes and must NOT be used on a fixed-size (no-scale)
        tracker's output — that would silently violate the "output w,h == init box"
        invariant. CF trackers report off-frame coordinates truthfully (the motion
        predictor, ARCHITECTURE.md §9, needs the real position); clamp only at a
        rendering/consumption boundary that genuinely requires a drawable box.
        """
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        w = min(self.w, 1.0 - x)
        h = min(self.h, 1.0 - y)
        return BoundingBox(x=x, y=y, w=w, h=h)
