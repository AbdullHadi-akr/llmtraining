"""Evaluation helpers for point-wise MLP predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.loader import load_op
from .features_pointwise import iter_pointwise_blocks
from .normalizer import PointwiseNormalizer


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def bc_v_spatial_variance(
    model: torch.nn.Module,
    op_ids: list[str],
    normalizer: PointwiseNormalizer,
    *,
    subsample_time: int = 1,
    ts_extrapolation: str = "clamp",
    device: torch.device | None = None,
) -> float:
    """Compute mean variance of predicted bc_V across sensors at fixed times."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    per_time_variances: list[float] = []

    with torch.no_grad():
        for op_id in op_ids:
            bundle = load_op(op_id)
            time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time, dtype=np.int64)
            for x_block, _y_block, _sensor_ids, _time_index in iter_pointwise_blocks(
                bundle,
                time_indices,
                ts_extrapolation=ts_extrapolation,
            ):
                x_norm = normalizer.transform_X(x_block)
                x_tensor = torch.from_numpy(x_norm).to(device)
                pred_norm = model(x_tensor).cpu().numpy()
                pred = normalizer.inverse_Y(pred_norm)
                per_time_variances.append(float(np.var(pred[:, 1])))

    return float(np.mean(per_time_variances)) if per_time_variances else 0.0


def evaluate_on_ops(
    model: torch.nn.Module,
    op_ids: list[str],
    normalizer: PointwiseNormalizer,
    *,
    subsample_time: int = 1,
    ts_extrapolation: str = "clamp",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate T and bc_V metrics on a list of OP ids."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    per_op: dict[str, dict[str, dict[str, float]]] = {}

    with torch.no_grad():
        for op_id in op_ids:
            bundle = load_op(op_id)
            time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time, dtype=np.int64)

            y_true_op: list[np.ndarray] = []
            y_pred_op: list[np.ndarray] = []
            for x_block, y_block, _sensor_ids, _time_index in iter_pointwise_blocks(
                bundle,
                time_indices,
                ts_extrapolation=ts_extrapolation,
            ):
                x_norm = normalizer.transform_X(x_block)
                x_tensor = torch.from_numpy(x_norm).to(device)
                pred_norm = model(x_tensor).cpu().numpy()
                pred = normalizer.inverse_Y(pred_norm)
                y_true_op.append(y_block)
                y_pred_op.append(pred)

            if y_true_op:
                y_true_op_arr = np.vstack(y_true_op)
                y_pred_op_arr = np.vstack(y_pred_op)
                per_op[op_id] = {
                    "T": _metric_bundle(y_true_op_arr[:, 0], y_pred_op_arr[:, 0]),
                    "bc_V": _metric_bundle(y_true_op_arr[:, 1], y_pred_op_arr[:, 1]),
                }
                y_true_all.append(y_true_op_arr)
                y_pred_all.append(y_pred_op_arr)

    if not y_true_all:
        return {
            "T": {"mse": 0.0, "mae": 0.0, "max_error": 0.0, "r2": 0.0},
            "bc_V": {"mse": 0.0, "mae": 0.0, "max_error": 0.0, "r2": 0.0},
            "per_op": per_op,
            "bc_V_spatial_variance": 0.0,
        }

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    return {
        "T": _metric_bundle(y_true[:, 0], y_pred[:, 0]),
        "bc_V": _metric_bundle(y_true[:, 1], y_pred[:, 1]),
        "per_op": per_op,
        "bc_V_spatial_variance": bc_v_spatial_variance(
            model,
            op_ids,
            normalizer,
            subsample_time=subsample_time,
            ts_extrapolation=ts_extrapolation,
            device=device,
        ),
    }


def collect_pointwise_predictions(
    model: torch.nn.Module,
    op_ids: list[str],
    normalizer: PointwiseNormalizer,
    *,
    subsample_time: int = 1,
    ts_extrapolation: str = "clamp",
    device: torch.device | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """
    Collect true vs predicted pairs for scatter plotting.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MLP model
    op_ids : list[str]
        List of OP IDs to evaluate
    normalizer : PointwiseNormalizer
        Fitted normalizer for feature/target preprocessing
    subsample_time : int
        Time subsampling factor (default: 1)
    ts_extrapolation : str
        Time series extrapolation mode (default: "clamp")
    device : torch.device, optional
        Compute device (default: auto-detect)
    max_points : int, optional
        Cap total points via uniform subsampling (default: None)

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - "T": {"y_true": ndarray, "y_pred": ndarray}
        - "bc_V": {"y_true": ndarray, "y_pred": ndarray}
        - "n_points": total number of points
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    y_true_T_all: list[np.ndarray] = []
    y_pred_T_all: list[np.ndarray] = []
    y_true_V_all: list[np.ndarray] = []
    y_pred_V_all: list[np.ndarray] = []

    with torch.no_grad():
        for op_id in op_ids:
            bundle = load_op(op_id)
            time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time, dtype=np.int64)

            for x_block, y_block, _sensor_ids, _time_index in iter_pointwise_blocks(
                bundle,
                time_indices,
                ts_extrapolation=ts_extrapolation,
            ):
                x_norm = normalizer.transform_X(x_block)
                x_tensor = torch.from_numpy(x_norm).to(device)
                pred_norm = model(x_tensor).cpu().numpy()
                pred = normalizer.inverse_Y(pred_norm)

                y_true_T_all.append(y_block[:, 0])
                y_pred_T_all.append(pred[:, 0])
                y_true_V_all.append(y_block[:, 1])
                y_pred_V_all.append(pred[:, 1])

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
    }
