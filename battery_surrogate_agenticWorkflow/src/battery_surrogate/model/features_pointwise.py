"""Feature assembly for point-wise MLP training."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..data.models import OpBundle
from ..schema.columns import CANONICAL_CHANNELS


def interp_to_fast(
    t_src: np.ndarray,
    v_src: np.ndarray,
    t_fast: np.ndarray,
    mode: str = "clamp",
) -> np.ndarray:
    """Interpolate one source signal to t_fast using edge-clamp behavior."""

    src_t = np.asarray(t_src, dtype=np.float64).reshape(-1)
    src_v = np.asarray(v_src, dtype=np.float64).reshape(-1)
    target_t = np.asarray(t_fast, dtype=np.float64).reshape(-1)
    if src_t.size == 0 or src_v.size == 0:
        raise ValueError("Cannot interpolate an empty time series")
    if src_t.size != src_v.size:
        raise ValueError("t_src and v_src must have the same length")
    if mode != "clamp":
        raise ValueError(f"Unsupported extrapolation mode: {mode}")

    if np.any(np.diff(src_t) <= 0):
        order = np.argsort(src_t)
        src_t = src_t[order]
        src_v = src_v[order]

    return np.interp(target_t, src_t, src_v, left=src_v[0], right=src_v[-1]).astype(np.float32)


def _sim_config_matrix(bundle: OpBundle, ts_extrapolation: str = "clamp") -> np.ndarray:
    """Return (n_time, 7) canonical simulation-config channels."""

    n_time = bundle.t_fast.shape[0]
    out = np.empty((n_time, len(CANONICAL_CHANNELS)), dtype=np.float32)

    scalar_lookup = {
        name: float(bundle.sim_config_scalar[index])
        for index, name in enumerate(bundle.sim_config_scalar_names)
    }

    for col_index, channel in enumerate(CANONICAL_CHANNELS):
        if channel in bundle.sim_config_ts:
            t_src, v_src = bundle.sim_config_ts[channel]
            out[:, col_index] = interp_to_fast(t_src, v_src, bundle.t_fast, mode=ts_extrapolation)
            continue

        if channel in scalar_lookup:
            out[:, col_index] = np.full(n_time, scalar_lookup[channel], dtype=np.float32)
            continue

        if channel == "soc_start" and "soc_start" in bundle.meta:
            out[:, col_index] = np.full(n_time, float(bundle.meta["soc_start"]), dtype=np.float32)
            continue

        raise ValueError(f"Missing canonical channel '{channel}' for OP {bundle.op_id}")

    return out


def assemble_pointwise_block(
    bundle: OpBundle,
    time_index: int,
    ts_extrapolation: str = "clamp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one (all-sensors) block for a single time index."""

    n_time = bundle.t_fast.shape[0]
    if time_index < 0 or time_index >= n_time:
        raise IndexError("time_index out of bounds")

    coords = np.asarray(bundle.xyz, dtype=np.float32)
    n_sensors = coords.shape[0]

    cfg_matrix = _sim_config_matrix(bundle, ts_extrapolation=ts_extrapolation)
    cfg_row = cfg_matrix[time_index]
    t_value = np.float32(bundle.t_fast[time_index])

    t_column = np.full((n_sensors, 1), t_value, dtype=np.float32)
    cfg_block = np.repeat(cfg_row.reshape(1, -1), repeats=n_sensors, axis=0)
    x_block = np.concatenate([coords, t_column, cfg_block], axis=1)

    t_targets = np.asarray(bundle.T[time_index, :], dtype=np.float32).reshape(-1)
    bc_v_target = np.float32(bundle.bc_V[time_index])
    bc_v_targets = np.full(n_sensors, bc_v_target, dtype=np.float32)
    y_block = np.column_stack([t_targets, bc_v_targets]).astype(np.float32)

    sensor_ids = np.asarray(bundle.sensor_id)
    return x_block, y_block, sensor_ids


def iter_pointwise_blocks(
    bundle: OpBundle,
    time_indices: np.ndarray,
    ts_extrapolation: str = "clamp",
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Yield point-wise feature blocks across requested time indices."""

    for time_index in time_indices:
        x_block, y_block, sensor_ids = assemble_pointwise_block(
            bundle,
            int(time_index),
            ts_extrapolation=ts_extrapolation,
        )
        yield x_block, y_block, sensor_ids, int(time_index)
