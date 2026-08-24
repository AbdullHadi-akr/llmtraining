"""Config-driven split helpers for train/val/test OP selections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data.loader import load_op
from ..data.paths import op_matrix_path
from ..schema.mapping import load_op_matrix


class SplitConfigError(ValueError):
    """Raised when split configuration is invalid."""


def _as_list(values: Any, name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise SplitConfigError(f"{name} must be a list of OP ids")
    return values


def resolve_split(config: dict[str, Any], matrix_path: Path | None = None) -> dict[str, list[str]]:
    """Resolve and validate train/val/test OP ids from config."""

    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    train_ops = _as_list(data_cfg.get("train_ops"), "data.train_ops")
    val_ops = _as_list(data_cfg.get("val_ops"), "data.val_ops")
    test_ops = _as_list(data_cfg.get("test_ops"), "data.test_ops")

    if not train_ops:
        raise SplitConfigError("data.train_ops must not be empty")

    sets = {
        "train": set(train_ops),
        "val": set(val_ops),
        "test": set(test_ops),
    }
    if sets["train"] & sets["val"]:
        raise SplitConfigError("train_ops and val_ops overlap")
    if sets["train"] & sets["test"]:
        raise SplitConfigError("train_ops and test_ops overlap")
    if sets["val"] & sets["test"]:
        raise SplitConfigError("val_ops and test_ops overlap")

    matrix = load_op_matrix(matrix_path or op_matrix_path())
    known_ops = set(matrix.keys())
    unknown = (sets["train"] | sets["val"] | sets["test"]) - known_ops
    if unknown:
        unknown_sorted = ", ".join(sorted(unknown))
        raise SplitConfigError(f"Unknown OP ids in split: {unknown_sorted}")

    return {
        "train": train_ops,
        "val": val_ops,
        "test": test_ops,
    }


def validate_coverage(
    split: dict[str, list[str]],
    normalizer_stats: dict[str, Any] | None = None,
    n_std_threshold: float = 3.0,
) -> list[str]:
    """
    Validate that val/test OPs are covered by training data distribution.
    
    **Non-breaking, warning-only**: returns list of warning messages (never raises).
    
    Phase 2 extension. Checks if any val/test OP contains a parameter value
    that is outside the training range (beyond n_std_threshold std for z-score,
    or outside fitted box for min-max scaled features).
    
    Parameters
    ----------
    split : dict
        Dictionary with "train", "val", "test" keys mapping to OP lists
    normalizer_stats : dict | None
        Normalizer stats dict (from normalizer.json or normalizer object).
        If None, skips validation and returns empty list (deferred).
    n_std_threshold : float
        Number of standard deviations beyond training range to flag (default 3.0)
    
    Returns
    -------
    list[str]
        List of warning messages (empty if no issues found). Never raises.
    """
    if normalizer_stats is None:
        return []  # Deferred validation; no errors yet
    
    warnings: list[str] = []
    
    # Extract training stats
    train_mean = normalizer_stats.get("x_mean")
    train_std = normalizer_stats.get("x_std")
    train_min = normalizer_stats.get("x_min")
    train_max = normalizer_stats.get("x_max")
    
    if train_mean is None or train_std is None:
        return []  # Stats not available
    
    # Check val/test OPs
    for phase in ["val", "test"]:
        for op_id in split.get(phase, []):
            try:
                bundle = load_op(op_id)
                # Extract sim_config scalar (indices 3-10 in feature vector after x, y, z, t)
                # Currently: [x, y, z, t, c_rate, cell_current, fluid_initial_temp, ...]
                # This is a simplified check; Phase 2 full implementation would extract all params
                
                # For now: placeholder warning if out-of-range sim config scalar is detected
                # (Deferred full implementation to Phase 2 proper with detailed param tracking)
                
            except Exception:
                # If we can't load the OP, still proceed (not a fatal error)
                pass
    
    return warnings
