from __future__ import annotations

from quadguide.core.messages import BoundingBox


def bbox_centroid_norm(bbox: BoundingBox) -> tuple[float, float]:
    """Image-centre-relative centroid in (-1, 1).

    cx = -1 → bbox centre at left edge; cx = +1 → right edge. Same for cy.
    """
    return (
        (bbox.x + bbox.w * 0.5 - 0.5) * 2.0,
        (bbox.y + bbox.h * 0.5 - 0.5) * 2.0,
    )
