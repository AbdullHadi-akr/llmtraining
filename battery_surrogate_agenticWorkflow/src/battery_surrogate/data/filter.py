"""Small array filters used during assembly."""

from __future__ import annotations

import numpy as np


def clip_array(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return a clipped copy of the input array."""

    return np.clip(values, lower, upper)


def clip_temperature(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Clip a temperature array using the configured bounds."""

    return clip_array(values, limits[0], limits[1])


def clip_voltage(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Clip a voltage array using the configured bounds."""

    return clip_array(values, limits[0], limits[1])
