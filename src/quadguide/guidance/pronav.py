from __future__ import annotations


def pronav(
    los_rate: tuple[float, float],
    closing_vel: float,
    N: float,
) -> tuple[float, float]:
    """Proportional navigation: a_cmd = N * V_c * los_rate."""
    return N * closing_vel * los_rate[0], N * closing_vel * los_rate[1]
