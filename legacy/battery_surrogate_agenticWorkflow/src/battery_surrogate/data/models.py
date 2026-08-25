"""Dataclasses used by the workflow package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpBundle:
    """One fully assembled OP bundle ready for cache writing or loading."""

    op_id: str
    schema_version: int
    cache_key: str
    t_fast: np.ndarray
    t_slow: np.ndarray
    bc_V: np.ndarray
    bc_OCV: np.ndarray
    bc_I: np.ndarray
    pe_P_loss: np.ndarray
    T: np.ndarray
    q_source: np.ndarray
    xyz: np.ndarray
    layer: np.ndarray
    sensor_id: np.ndarray
    fluid_props: np.ndarray
    sim_config_scalar: np.ndarray
    sim_config_scalar_names: tuple[str, ...]
    sim_config_ts: dict[str, tuple[np.ndarray, np.ndarray]]
    meta: dict[str, Any]
