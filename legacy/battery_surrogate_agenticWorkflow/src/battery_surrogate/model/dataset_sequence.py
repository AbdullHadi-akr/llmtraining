"""Sequence dataset for recurrent model training (per-sensor sequences with history lags)."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from torch.utils.data import IterableDataset

from ..data.loader import load_op
from .features_sequence import build_sequence_for_sensor, resolve_history_lengths, build_history_lags_per_target
from .normalizer import PointwiseNormalizer


class SequenceDataset(IterableDataset):
    """
    IterableDataset for recurrent models.

    Streams per-OP, per-sensor sequences with k-lag history.
    Each sample: (features (n_time, 11), history (n_time, 2*k), targets (n_time, 2))

    Mimics PointwiseDataset interface for compatibility.
    """

    def __init__(
        self,
        op_ids: list[str],
        normalizer: PointwiseNormalizer,
        config: dict[str, Any],
        shuffle_ops: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        op_ids : list[str]
            List of OP IDs to iterate over
        normalizer : PointwiseNormalizer
            Fitted normalizer for feature/target preprocessing
        config : dict
            Configuration with:
            - data.subsample_time: time subsampling factor
            - data.ts_extrapolation: interpolation mode
            - model.history_length: k for lag history (default 8)
        shuffle_ops : bool
            Whether to shuffle OP order each epoch (default False)
        seed : int
            Random seed for shuffling (default 42)
        """
        self.op_ids = op_ids
        self.normalizer = normalizer
        self.config = config
        self.shuffle_ops = shuffle_ops
        self.seed = seed

        # Extract config parameters
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        self.subsample_time = int(data_cfg.get("subsample_time", 1))
        self.ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))
        
        # Resolve per-target history lengths
        self.k_T, self.k_V = resolve_history_lengths(model_cfg)
        self.history_length = max(self.k_T, self.k_V)  # for compatibility

        # n_sensors is determined from first OP
        self.n_sensors = None
        self._infer_n_sensors()

    def _infer_n_sensors(self) -> None:
        """Infer number of sensors from first OP."""
        if not self.op_ids:
            self.n_sensors = 0
            return

        try:
            bundle = load_op(self.op_ids[0])
            self.n_sensors = bundle.xyz.shape[0]
        except Exception:
            self.n_sensors = 363  # Default fallback

    def __iter__(self) -> Iterator[tuple]:
        """
        Iterate over (OP, sensor) pairs, yielding sequences.

        Yields
        ------
        tuple
            (features, history, targets, seq_len, op_id, sensor_id)
            - features: (n_time, 11) float32
            - history: (n_time, 2*k) float32 (lagged normalized targets)
            - targets: (n_time, 2) float32 (normalized)
            - seq_len: int (number of timesteps)
            - op_id: str
            - sensor_id: int
        """
        op_order = list(self.op_ids)
        if self.shuffle_ops:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(op_order)

        for op_id in op_order:
            try:
                bundle = load_op(op_id)
            except Exception as e:
                # Skip OPs that fail to load
                print(f"Warning: Failed to load OP {op_id}: {e}")
                continue

            # Iterate over all sensors in this OP
            for sensor_idx in range(bundle.xyz.shape[0]):
                try:
                    # Build sequence for this sensor
                    features, targets, seq_len = build_sequence_for_sensor(
                        bundle,
                        sensor_idx,
                        self.config,
                        ts_extrapolation=self.ts_extrapolation,
                    )

                    # Normalize features and targets
                    features_norm = self.normalizer.transform_X(features)
                    targets_norm = self.normalizer.transform_Y(targets)

                    # Build history lags from normalized targets using per-target lengths
                    history = build_history_lags_per_target(targets_norm, self.k_T, self.k_V)

                    yield (
                        features_norm,
                        history,
                        targets_norm,
                        seq_len,
                        op_id,
                        sensor_idx,
                    )

                except Exception as e:
                    # Skip sensors that fail
                    err_msg = f"Warning: Failed to process OP {op_id} sensor {sensor_idx}: {e}"
                    print(err_msg)
                    continue
