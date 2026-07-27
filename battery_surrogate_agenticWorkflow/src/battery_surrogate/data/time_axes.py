"""Helpers for validating and extracting time axes."""

from __future__ import annotations

import numpy as np


def as_float32_axis(values: np.ndarray) -> np.ndarray:
    """Return a 1D float32 copy of the input axis."""

    return np.asarray(values, dtype=np.float32).reshape(-1)


def is_strictly_increasing(values: np.ndarray) -> bool:
    """Check whether a time axis is strictly increasing."""

    axis = as_float32_axis(values)
    return bool(np.all(np.diff(axis) > 0))
