"""Evaluation helpers for recurrent sequence model predictions."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.loader import load_op
from .features_sequence import build_sequence_for_sensor, resolve_history_lengths
from .normalizer import PointwiseNormalizer
from .registry import build_model


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MSE, MAE, max_error, R² for a pair of arrays."""
    err = y_pred - y_true
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (float(np.sum(np.square(err))) / denom if denom > 0 else 0.0)
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2,
    }


def evaluate_sequence_model(
    model: nn.Module,
    op_ids: list[str],
    normalizer: PointwiseNormalizer,
    config: dict[str, Any],
    *,
    device: torch.device | None = None,
    max_sensors: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate recurrent model using autoregressive rollout on test OPs.

    Parameters
    ----------
    model : nn.Module
        Trained recurrent model with `rollout()` method
    op_ids : list[str]
        List of OP IDs to evaluate
    normalizer : PointwiseNormalizer
        Fitted normalizer for feature/target preprocessing
    config : dict
        Configuration with data.subsample_time, model.history_length
    device : torch.device, optional
        Compute device (default: auto-detect)
    max_sensors : int, optional
        Limit number of sensors per OP for faster evaluation (default: all)

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - mae_T, mse_T, r2_T: Temperature metrics
        - mae_bc_V, mse_bc_V, r2_bc_V: Voltage metrics
        - max_error_T, max_error_bc_V: Maximum errors
        - per_op: Per-OP breakdown {op_id: {T: {...}, bc_V: {...}}}
        - error_curves: {op_id: {sensor_id: [(t, error_T, error_bc_V), ...]}}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    subsample_time = int(data_cfg.get("subsample_time", 1))
    ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))

    k_T, k_V = resolve_history_lengths(model_cfg)
    k = max(k_T, k_V)  # use max for warm-up mask offset

    # Aggregators
    all_y_true_T: list[np.ndarray] = []
    all_y_pred_T: list[np.ndarray] = []
    all_y_true_bcV: list[np.ndarray] = []
    all_y_pred_bcV: list[np.ndarray] = []
    per_op: dict[str, dict[str, dict[str, float]]] = {}
    error_curves: dict[str, dict[int, list[tuple[float, float, float]]]] = {}

    with torch.no_grad():
        for op_id in op_ids:
            bundle = load_op(op_id)
            n_sensors = bundle.xyz.shape[0]
            if max_sensors is not None:
                n_sensors = min(n_sensors, max_sensors)

            op_y_true_T: list[np.ndarray] = []
            op_y_pred_T: list[np.ndarray] = []
            op_y_true_bcV: list[np.ndarray] = []
            op_y_pred_bcV: list[np.ndarray] = []
            error_curves[op_id] = {}

            for sensor_idx in range(n_sensors):
                # Build sequence features and targets
                features, targets, seq_len = build_sequence_for_sensor(
                    bundle, sensor_idx, config, ts_extrapolation
                )

                # Normalize features
                features_norm = normalizer.transform_X(features)
                targets_norm = normalizer.transform_Y(targets)

                # Initial condition: first normalized target
                y0 = torch.from_numpy(targets_norm[0]).float().to(device)
                features_tensor = torch.from_numpy(features_norm).float().to(device)

                # Autoregressive rollout
                preds_norm = model.rollout(features_tensor, y0).cpu().numpy()

                # Denormalize predictions
                preds = normalizer.inverse_Y(preds_norm)

                # Skip warm-up period (first k steps) for metrics
                if seq_len > k:
                    y_true_slice = targets[k:]
                    y_pred_slice = preds[k:]
                else:
                    y_true_slice = targets
                    y_pred_slice = preds

                op_y_true_T.append(y_true_slice[:, 0])
                op_y_pred_T.append(y_pred_slice[:, 0])
                op_y_true_bcV.append(y_true_slice[:, 1])
                op_y_pred_bcV.append(y_pred_slice[:, 1])

                # Error curves: per-step error for visualization
                time_vals = bundle.t_fast[::subsample_time][:seq_len]
                err_T = np.abs(preds[:, 0] - targets[:, 0])
                err_bcV = np.abs(preds[:, 1] - targets[:, 1])
                error_curves[op_id][sensor_idx] = [
                    (float(t), float(eT), float(eV))
                    for t, eT, eV in zip(time_vals, err_T, err_bcV)
                ]

            # Per-OP metrics
            op_y_true_T_arr = np.concatenate(op_y_true_T)
            op_y_pred_T_arr = np.concatenate(op_y_pred_T)
            op_y_true_bcV_arr = np.concatenate(op_y_true_bcV)
            op_y_pred_bcV_arr = np.concatenate(op_y_pred_bcV)

            per_op[op_id] = {
                "T": _metric_bundle(op_y_true_T_arr, op_y_pred_T_arr),
                "bc_V": _metric_bundle(op_y_true_bcV_arr, op_y_pred_bcV_arr),
            }

            all_y_true_T.append(op_y_true_T_arr)
            all_y_pred_T.append(op_y_pred_T_arr)
            all_y_true_bcV.append(op_y_true_bcV_arr)
            all_y_pred_bcV.append(op_y_pred_bcV_arr)

    # Global metrics
    global_y_true_T = np.concatenate(all_y_true_T)
    global_y_pred_T = np.concatenate(all_y_pred_T)
    global_y_true_bcV = np.concatenate(all_y_true_bcV)
    global_y_pred_bcV = np.concatenate(all_y_pred_bcV)

    metrics_T = _metric_bundle(global_y_true_T, global_y_pred_T)
    metrics_bcV = _metric_bundle(global_y_true_bcV, global_y_pred_bcV)

    return {
        "mae_T": metrics_T["mae"],
        "mse_T": metrics_T["mse"],
        "r2_T": metrics_T["r2"],
        "max_error_T": metrics_T["max_error"],
        "mae_bc_V": metrics_bcV["mae"],
        "mse_bc_V": metrics_bcV["mse"],
        "r2_bc_V": metrics_bcV["r2"],
        "max_error_bc_V": metrics_bcV["max_error"],
        "per_op": per_op,
        "error_curves": error_curves,
    }


def compute_lookback_seconds(
    op_ids: list[str],
    history_length: int,
    subsample_time: int = 1,
) -> dict[str, float]:
    """
    Compute actual lookback window in seconds for a given history length.

    Since time steps may have non-uniform Δt, this computes statistics
    over all sequences.

    Parameters
    ----------
    op_ids : list[str]
        OP IDs to analyze
    history_length : int
        Number of history lags (k)
    subsample_time : int
        Time subsampling factor

    Returns
    -------
    dict[str, float]
        Dictionary with keys: median, min, max, mean
    """
    lookback_vals: list[float] = []

    for op_id in op_ids:
        bundle = load_op(op_id)
        t_vals = bundle.t_fast[::subsample_time]
        seq_len = len(t_vals)

        for i in range(history_length, seq_len):
            lookback = float(t_vals[i] - t_vals[i - history_length])
            lookback_vals.append(lookback)

    if not lookback_vals:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}

    return {
        "median": float(np.median(lookback_vals)),
        "min": float(np.min(lookback_vals)),
        "max": float(np.max(lookback_vals)),
        "mean": float(np.mean(lookback_vals)),
    }


def history_length_benchmark(
    config: dict[str, Any],
    *,
    k_values: list[int] | None = None,
    epochs_per_k: int = 1,
    max_sensors: int = 10,
    device: torch.device | None = None,
    progress_cb: Any | None = None,
) -> pd.DataFrame:
    """
    Benchmark different history lengths (k) to find the accuracy-cost sweet spot.

    Parameters
    ----------
    config : dict
        Base configuration (will be modified for each k)
    k_values : list[int], optional
        History lengths to test. Default: [1, 2, 4, 8, 16, 32]
    epochs_per_k : int
        Training epochs per configuration (default: 1 for fast sweep)
    max_sensors : int
        Limit sensors for evaluation (default: 10 for speed)
    device : torch.device, optional
        Compute device (default: auto-detect)
    progress_cb : callable, optional
        Progress callback invoked after each k run as progress_cb(done, total, msg)

    Returns
    -------
    pd.DataFrame
        Benchmark results with columns:
        - k: history length
        - mae_T, mse_T, r2_T: Temperature metrics
        - mae_bc_V, mse_bc_V, r2_bc_V: Voltage metrics
        - lookback_seconds_median, lookback_seconds_min, lookback_seconds_max
        - param_count: model parameter count
        - train_time_s: training time in seconds
        - input_width: effective input width (11 + 2*k)
        - recommended: True for the recommended k (accuracy knee)
    """
    from .registry import build_datasets
    from .trainer_sequence import train_sequence_model
    from .split import resolve_split

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if k_values is None:
        # Check if config has history_sweep.values
        sweep_cfg = config.get("history_sweep", {})
        k_values = sweep_cfg.get("values", [1, 2, 4, 8, 16, 32])

    results: list[dict[str, Any]] = []
    split = resolve_split(config)
    data_cfg = config.get("data", {})
    subsample_time = int(data_cfg.get("subsample_time", 1))

    for k in k_values:
        # Modify config for this k
        cfg_k = _deep_copy_config(config)
        cfg_k["model"]["history_length"] = k
        cfg_k["train"]["epochs"] = epochs_per_k

        # Compute lookback seconds
        lookback = compute_lookback_seconds(
            split["train"], k, subsample_time
        )

        # Fit normalizer
        from .features_pointwise import iter_pointwise_blocks

        normalizer_k = PointwiseNormalizer()
        for op_id in split["train"]:
            bundle = load_op(op_id)
            time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time)
            for x_block, y_block, _, _ in iter_pointwise_blocks(
                bundle, time_indices, ts_extrapolation=data_cfg.get("ts_extrapolation", "clamp")
            ):
                normalizer_k.partial_fit(x_block, y_block)
        normalizer_k.finalize()

        # Build model
        model = build_model(cfg_k, n_sensors=363, seed=42)
        param_count = model.n_parameters

        # Build datasets
        dataloaders = build_datasets(cfg_k, normalizer_k, split, seed=42)

        # Train
        start_time = time.time()
        train_result = train_sequence_model(
            model,
            dataloaders["train"],
            dataloaders["val"],
            cfg_k,
            n_sensors=363,
            device=device,
        )
        train_time = time.time() - start_time

        # Evaluate on val OPs
        eval_result = evaluate_sequence_model(
            model,
            split["val"],
            normalizer_k,
            cfg_k,
            device=device,
            max_sensors=max_sensors,
        )

        results.append({
            "k": k,
            "mae_T": eval_result["mae_T"],
            "mse_T": eval_result["mse_T"],
            "r2_T": eval_result["r2_T"],
            "mae_bc_V": eval_result["mae_bc_V"],
            "mse_bc_V": eval_result["mse_bc_V"],
            "r2_bc_V": eval_result["r2_bc_V"],
            "lookback_seconds_median": lookback["median"],
            "lookback_seconds_min": lookback["min"],
            "lookback_seconds_max": lookback["max"],
            "param_count": param_count,
            "train_time_s": train_time,
            "input_width": 11 + 2 * k,
            "best_val_loss": train_result.best_val_loss,
        })

        if progress_cb is not None:
            progress_cb(len(results), len(k_values), f"k={k}")

    df = pd.DataFrame(results)

    # Find recommended k: smallest k within 5% of best R² for T
    if not df.empty:
        best_r2_T = df["r2_T"].max()
        threshold = best_r2_T * 0.95  # within 5%
        candidates = df[df["r2_T"] >= threshold]
        if not candidates.empty:
            recommended_k = int(candidates["k"].min())
            df["recommended"] = df["k"] == recommended_k
        else:
            df["recommended"] = False
    else:
        df["recommended"] = False

    return df


def benchmark_history_lengths(
    config: dict[str, Any],
    *,
    k_values: list[int] | None = None,
    epochs_per_k: int = 1,
    max_sensors: int = 10,
    device: torch.device | None = None,
    progress_cb: Any | None = None,
) -> pd.DataFrame:
    """Compatibility alias for history_length_benchmark()."""

    return history_length_benchmark(
        config,
        k_values=k_values,
        epochs_per_k=epochs_per_k,
        max_sensors=max_sensors,
        device=device,
        progress_cb=progress_cb,
    )


def _deep_copy_config(config: dict) -> dict:
    """Deep copy a configuration dictionary."""
    import copy
    return copy.deepcopy(config)


def collect_sequence_predictions(
    model: nn.Module,
    op_ids: list[str],
    normalizer: PointwiseNormalizer,
    config: dict[str, Any],
    *,
    device: torch.device | None = None,
    max_sensors: int | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """
    Collect true vs predicted pairs from recurrent model for scatter plotting.

    Parameters
    ----------
    model : nn.Module
        Trained recurrent model with `rollout()` method
    op_ids : list[str]
        List of OP IDs to evaluate
    normalizer : PointwiseNormalizer
        Fitted normalizer for feature/target preprocessing
    config : dict
        Configuration with data.subsample_time, model.history_length
    device : torch.device, optional
        Compute device (default: auto-detect)
    max_sensors : int, optional
        Limit number of sensors per OP for faster evaluation (default: all)
    max_points : int, optional
        Cap total points via uniform subsampling (default: None)

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - "T": {"y_true": ndarray, "y_pred": ndarray}
        - "bc_V": {"y_true": ndarray, "y_pred": ndarray}
        - "n_points": total number of points (after warm-up skip)
        - "per_sensor_example": optional time-series example for one (op, sensor) pair
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    subsample_time = int(data_cfg.get("subsample_time", 1))
    ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))

    k_T, k_V = resolve_history_lengths(model_cfg)
    k = max(k_T, k_V)

    y_true_T_all: list[np.ndarray] = []
    y_pred_T_all: list[np.ndarray] = []
    y_true_V_all: list[np.ndarray] = []
    y_pred_V_all: list[np.ndarray] = []

    per_sensor_example = None

    with torch.no_grad():
        for op_id in op_ids:
            bundle = load_op(op_id)
            n_sensors = bundle.xyz.shape[0]
            if max_sensors is not None:
                n_sensors = min(n_sensors, max_sensors)

            for sensor_idx in range(n_sensors):
                try:
                    # Build sequence for this sensor
                    features, targets, seq_len = build_sequence_for_sensor(
                        bundle, sensor_idx, config, ts_extrapolation
                    )

                    # Normalize features and targets
                    features_norm = normalizer.transform_X(features)
                    targets_norm = normalizer.transform_Y(targets)

                    # Initial condition
                    y0 = torch.from_numpy(targets_norm[0]).float().to(device)
                    features_tensor = torch.from_numpy(features_norm).float().to(device)

                    # Autoregressive rollout
                    preds_norm = model.rollout(features_tensor, y0).cpu().numpy()
                    preds = normalizer.inverse_Y(preds_norm)

                    # Skip warm-up
                    if seq_len > k:
                        y_true_slice = targets[k:]
                        y_pred_slice = preds[k:]
                    else:
                        y_true_slice = targets
                        y_pred_slice = preds

                    y_true_T_all.append(y_true_slice[:, 0])
                    y_pred_T_all.append(y_pred_slice[:, 0])
                    y_true_V_all.append(y_true_slice[:, 1])
                    y_pred_V_all.append(y_pred_slice[:, 1])

                    # Keep one time-series example
                    if per_sensor_example is None:
                        time_vals = bundle.t_fast[::subsample_time][:seq_len]
                        per_sensor_example = {
                            "op_id": op_id,
                            "sensor_idx": sensor_idx,
                            "t": time_vals,
                            "T_true": targets[:, 0],
                            "T_pred": preds[:, 0],
                            "V_true": targets[:, 1],
                            "V_pred": preds[:, 1],
                        }

                except Exception:
                    # Skip sensors that fail
                    continue

    # Pool all predictions
    y_true_T = np.concatenate(y_true_T_all) if y_true_T_all else np.array([])
    y_pred_T = np.concatenate(y_pred_T_all) if y_pred_T_all else np.array([])
    y_true_V = np.concatenate(y_true_V_all) if y_true_V_all else np.array([])
    y_pred_V = np.concatenate(y_pred_V_all) if y_pred_V_all else np.array([])

    n_points = len(y_true_T)

    # Cap points via uniform subsampling if needed
    if max_points is not None and n_points > max_points:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_points, size=max_points, replace=False)
        indices = np.sort(indices)
        y_true_T = y_true_T[indices]
        y_pred_T = y_pred_T[indices]
        y_true_V = y_true_V[indices]
        y_pred_V = y_pred_V[indices]
        n_points = max_points

    return {
        "T": {"y_true": y_true_T, "y_pred": y_pred_T},
        "bc_V": {"y_true": y_true_V, "y_pred": y_pred_V},
        "n_points": n_points,
        "per_sensor_example": per_sensor_example,
    }


def plot_error_curves(
    error_curves: dict[str, dict[int, list[tuple[float, float, float]]]],
    op_id: str,
    sensor_ids: list[int] | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """
    Plot error-vs-time curves for a given OP.

    Parameters
    ----------
    error_curves : dict
        Error curves from evaluate_sequence_model()
    op_id : str
        OP ID to plot
    sensor_ids : list[int], optional
        Sensors to plot (default: first 5)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if op_id not in error_curves:
        raise ValueError(f"OP {op_id} not found in error_curves")

    curves = error_curves[op_id]
    if sensor_ids is None:
        sensor_ids = list(curves.keys())[:5]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for sensor_id in sensor_ids:
        if sensor_id not in curves:
            continue
        data = curves[sensor_id]
        t = [d[0] for d in data]
        err_T = [d[1] for d in data]
        err_bcV = [d[2] for d in data]

        axes[0].plot(t, err_T, alpha=0.7, label=f"Sensor {sensor_id}")
        axes[1].plot(t, err_bcV, alpha=0.7, label=f"Sensor {sensor_id}")

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Absolute Error (T)")
    axes[0].set_title(f"{op_id} - Temperature Error vs Time")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Absolute Error (bc_V)")
    axes[1].set_title(f"{op_id} - Voltage Error vs Time")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
