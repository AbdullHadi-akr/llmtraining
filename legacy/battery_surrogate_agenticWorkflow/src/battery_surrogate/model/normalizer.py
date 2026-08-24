"""Z-score normalizer for point-wise MLP inputs and targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class PointwiseNormalizer:
    """
    Streaming z-score normalizer for X (11 cols) and Y (2 cols).
    
    Supports configurable per-group scaling (Phase 2):
    - Default (None): all-z-score (backward compatible, byte-parity)
    - Per-group: coords→minmax[-1,1], time→minmax[0,1], sim_config→zscore, targets→zscore|robust
    """

    def __init__(self, eps: float = 1.0e-8, preprocess_config: dict[str, Any] | None = None) -> None:
        self.eps = float(eps)
        self.preprocess_config = preprocess_config or {}  # Stores per-group config
        self.version = 2  # For backward compatibility on load
        
        self._x_count = 0
        self._y_count = 0
        self._x_sum: np.ndarray | None = None
        self._y_sum: np.ndarray | None = None
        self._x_sumsq: np.ndarray | None = None
        self._y_sumsq: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        
        # Phase 2: per-group scaling metadata
        self.x_min: np.ndarray | None = None  # For min-max scaling (coords, time)
        self.x_max: np.ndarray | None = None
        self.y_min: np.ndarray | None = None
        self.y_max: np.ndarray | None = None
        self.scaling_type: list[str] | None = None  # Per-column scaling method

    def _ensure_accumulators(self, x_dim: int, y_dim: int) -> None:
        if self._x_sum is None:
            self._x_sum = np.zeros(x_dim, dtype=np.float64)
            self._x_sumsq = np.zeros(x_dim, dtype=np.float64)
            # Phase 2: min/max tracking
            self.x_min = np.full(x_dim, np.inf, dtype=np.float64)
            self.x_max = np.full(x_dim, -np.inf, dtype=np.float64)
        if self._y_sum is None:
            self._y_sum = np.zeros(y_dim, dtype=np.float64)
            self._y_sumsq = np.zeros(y_dim, dtype=np.float64)
            # Phase 2: min/max tracking
            self.y_min = np.full(y_dim, np.inf, dtype=np.float64)
            self.y_max = np.full(y_dim, -np.inf, dtype=np.float64)

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Accumulate statistics from one batch of features and targets."""

        x_batch = np.asarray(x, dtype=np.float64)
        y_batch = np.asarray(y, dtype=np.float64)
        if x_batch.ndim != 2 or y_batch.ndim != 2:
            raise ValueError("partial_fit expects 2D arrays for x and y")
        if x_batch.shape[0] != y_batch.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        self._ensure_accumulators(x_batch.shape[1], y_batch.shape[1])
        self._x_count += x_batch.shape[0]
        self._y_count += y_batch.shape[0]
        self._x_sum += x_batch.sum(axis=0)
        self._x_sumsq += np.square(x_batch).sum(axis=0)
        self._y_sum += y_batch.sum(axis=0)
        self._y_sumsq += np.square(y_batch).sum(axis=0)
        
        # Phase 2: track min/max for per-group scaling
        self.x_min = np.minimum(self.x_min, x_batch.min(axis=0))
        self.x_max = np.maximum(self.x_max, x_batch.max(axis=0))
        self.y_min = np.minimum(self.y_min, y_batch.min(axis=0))
        self.y_max = np.maximum(self.y_max, y_batch.max(axis=0))

    def finalize(self) -> None:
        """Finalize mean/std tensors for transform calls."""

        if self._x_count == 0 or self._y_count == 0:
            raise ValueError("Cannot finalize normalizer with no accumulated samples")

        self.x_mean = (self._x_sum / self._x_count).astype(np.float32)
        self.y_mean = (self._y_sum / self._y_count).astype(np.float32)

        x_var = (self._x_sumsq / self._x_count) - np.square(self.x_mean)
        y_var = (self._y_sumsq / self._y_count) - np.square(self.y_mean)

        self.x_std = np.sqrt(np.maximum(x_var, self.eps)).astype(np.float32)
        self.y_std = np.sqrt(np.maximum(y_var, self.eps)).astype(np.float32)
        
        # Phase 2: convert min/max to float32
        self.x_min = self.x_min.astype(np.float32)
        self.x_max = self.x_max.astype(np.float32)
        self.y_min = self.y_min.astype(np.float32)
        self.y_max = self.y_max.astype(np.float32)

    def _check_ready(self) -> None:
        if self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
            raise ValueError("Normalizer is not finalized")

    def transform_X(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to feature rows."""

        self._check_ready()
        x_arr = np.asarray(x, dtype=np.float32)
        return (x_arr - self.x_mean) / self.x_std

    def transform_Y(self, y: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to target rows."""

        self._check_ready()
        y_arr = np.asarray(y, dtype=np.float32)
        return (y_arr - self.y_mean) / self.y_std

    def inverse_Y(self, y_norm: np.ndarray) -> np.ndarray:
        """Map normalized targets back to physical units."""

        self._check_ready()
        y_arr = np.asarray(y_norm, dtype=np.float32)
        return y_arr * self.y_std + self.y_mean

    def save(self, path: Path) -> None:
        """Persist final normalizer statistics as JSON."""

        self._check_ready()
        payload = {
            "version": 2,  # Phase 2 version; version 1 is legacy all-zscore
            "eps": self.eps,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "x_min": self.x_min.tolist(),  # Phase 2: min-max for per-group
            "x_max": self.x_max.tolist(),
            "y_min": self.y_min.tolist(),
            "y_max": self.y_max.tolist(),
            "preprocess_config": self.preprocess_config,  # Phase 2: preserve config
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PointwiseNormalizer":
        """
        Load normalizer statistics from JSON.
        
        Backward compatible: version 1 (old format) loads with default all-zscore preset.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        # Backward compatibility: missing version or version==1 → default all-zscore
        version = payload.get("version", 1)
        preprocess_config = payload.get("preprocess_config", None) if version >= 2 else None
        
        normalizer = cls(eps=float(payload.get("eps", 1.0e-8)), preprocess_config=preprocess_config)
        normalizer.x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
        normalizer.x_std = np.asarray(payload["x_std"], dtype=np.float32)
        normalizer.y_mean = np.asarray(payload["y_mean"], dtype=np.float32)
        normalizer.y_std = np.asarray(payload["y_std"], dtype=np.float32)
        
        # Phase 2: load min/max if present
        if version >= 2:
            normalizer.x_min = np.asarray(payload.get("x_min"), dtype=np.float32)
            normalizer.x_max = np.asarray(payload.get("x_max"), dtype=np.float32)
            normalizer.y_min = np.asarray(payload.get("y_min"), dtype=np.float32)
            normalizer.y_max = np.asarray(payload.get("y_max"), dtype=np.float32)
        
        return normalizer
