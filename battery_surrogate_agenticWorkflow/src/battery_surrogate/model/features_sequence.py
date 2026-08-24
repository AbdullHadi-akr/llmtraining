"""Feature assembly for recurrent sequence training."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.models import OpBundle
from .features_pointwise import _sim_config_matrix


def build_sequence_for_sensor(
    bundle: OpBundle,
    sensor_idx: int,
    config: dict[str, Any],
    ts_extrapolation: str = "clamp",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build per-sensor sequence features and targets.

    Parameters
    ----------
    bundle : OpBundle
        Loaded operation bundle
    sensor_idx : int
        Sensor index (0 to n_sensors-1)
    config : dict
        Config with data.subsample_time
    ts_extrapolation : str
        Time series extrapolation mode (clamp, etc.)

    Returns
    -------
    tuple
        (features (n_time, 11), targets (n_time, 2), seq_len)
        - features: [x, y, z, t, c_rate, cell_current, fluid_initial_temp, ...]
        - targets: [T, bc_V] (both on this sensor)
        - seq_len: number of time steps after subsampling
    """
    n_time = bundle.t_fast.shape[0]
    subsample_time = int(config.get("data", {}).get("subsample_time", 1))

    # Time indices (subsampled)
    time_indices = np.arange(0, n_time, subsample_time, dtype=np.int64)
    seq_len = len(time_indices)

    # Coordinates (constant across time for this sensor)
    xyz = np.asarray(bundle.xyz[sensor_idx, :], dtype=np.float32)  # (3,)

    # Time values (subsampled)
    t_vals = np.asarray(bundle.t_fast[time_indices], dtype=np.float32)  # (seq_len,)

    # Sim config matrix (all sensors, all times) → extract for subsampled times
    cfg_matrix = _sim_config_matrix(bundle, ts_extrapolation=ts_extrapolation)
    cfg_subsampled = cfg_matrix[time_indices]  # (seq_len, 7)

    # Build features: [x, y, z, t, 7×sim_config] → (seq_len, 11)
    features = np.column_stack([
        np.tile(xyz[0], seq_len),  # x (constant)
        np.tile(xyz[1], seq_len),  # y (constant)
        np.tile(xyz[2], seq_len),  # z (constant)
        t_vals,                     # t (time-varying)
        cfg_subsampled,             # 7 sim-config channels
    ]).astype(np.float32)

    # Build targets: [T, bc_V] → (seq_len, 2)
    T_subsampled = bundle.T[time_indices, sensor_idx].astype(np.float32)
    bc_V_subsampled = bundle.bc_V[time_indices].astype(np.float32)

    targets = np.column_stack([
        T_subsampled,
        bc_V_subsampled,
    ]).astype(np.float32)

    return features, targets, seq_len


def resolve_history_lengths(model_cfg: dict[str, Any]) -> tuple[int, int]:
    """
    Resolve history lengths for T and bc_V from config.

    Parameters
    ----------
    model_cfg : dict
        Model config with 'history_length' which may be:
        - int k → interpreted as k_T = k_V = k
        - dict {T: ..., bc_V: ...} → per-target lengths

    Returns
    -------
    tuple[int, int]
        (k_T, k_V) — history lengths for T and bc_V respectively
    """
    h_cfg = model_cfg.get("history_length", 8)

    if isinstance(h_cfg, dict):
        k_T = int(h_cfg.get("T", h_cfg.get("bc_V", 8)))
        k_V = int(h_cfg.get("bc_V", h_cfg.get("T", 8)))
    else:
        k = int(h_cfg)
        k_T = k_V = k

    if k_T < 1 or k_V < 1:
        raise ValueError(f"history_length must be >= 1, got k_T={k_T}, k_V={k_V}")

    return k_T, k_V


def build_history_lags_per_target(
    targets_seq: np.ndarray,
    k_T: int,
    k_V: int,
    normalizer_fn=None,
) -> np.ndarray:
    """
    Build per-target lagged history tensor (grouped layout).

    Parameters
    ----------
    targets_seq : np.ndarray
        Shape (seq_len, 2) — normalized targets [T, bc_V]
    k_T : int
        Number of steps to look back for T
    k_V : int
        Number of steps to look back for bc_V
    normalizer_fn : callable | None
        Optional normalizer (unused, kept for compatibility)

    Returns
    -------
    np.ndarray
        Shape (seq_len, k_T + k_V) — grouped history [T-block | V-block]
        - For step t < max(k_T, k_V): pad with y_0 replicated
        - Otherwise: [T_{t-k_T}, ..., T_{t-1}, V_{t-k_V}, ..., V_{t-1}]
    """
    seq_len = targets_seq.shape[0]
    history = np.zeros((seq_len, k_T + k_V), dtype=np.float32)

    # y_0 for warm-up padding
    y_0 = targets_seq[0, :].copy()  # (2,)

    for t in range(seq_len):
        k_max = max(k_T, k_V)
        if t < k_max:
            # Warm-up: pad T-block and V-block with y_0 replicated
            history[t, :k_T] = np.full(k_T, y_0[0], dtype=np.float32)
            history[t, k_T:k_T + k_V] = np.full(k_V, y_0[1], dtype=np.float32)
        else:
            # Full history: grouped [T-block | V-block]
            history[t, :k_T] = targets_seq[t - k_T:t, 0]
            history[t, k_T:k_T + k_V] = targets_seq[t - k_V:t, 1]

    return history


def build_history_lags(
    targets_seq: np.ndarray,
    history_length: int,
    normalizer_fn=None,
) -> np.ndarray:
    """
    Build lagged history tensor for sequence model input.

    Delegates to build_history_lags_per_target with k_T = k_V = history_length.

    Parameters
    ----------
    targets_seq : np.ndarray
        Shape (seq_len, 2) — normalized targets [T, bc_V]
    history_length : int
        k — number of steps to look back
    normalizer_fn : callable | None
        Optional normalizer to apply to y_0 warm-up (if None, use raw y_0)

    Returns
    -------
    np.ndarray
        Shape (seq_len, 2*k) — lagged history tensor
        - For step t < k: pad with y_0 replicated
        - For step t ≥ k: history from steps [t-k, t-1]
    """
    return build_history_lags_per_target(targets_seq, history_length, history_length, normalizer_fn)
