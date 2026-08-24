"""Iterable dataset for point-wise MLP training."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..data.loader import load_op
from ..data.models import OpBundle
from .features_pointwise import iter_pointwise_blocks
from .normalizer import PointwiseNormalizer


class PointwiseDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stream point-wise samples OP-by-OP with full spatial coverage."""

    def __init__(
        self,
        op_ids: list[str],
        *,
        normalizer: PointwiseNormalizer | None = None,
        subsample_time: int = 1,
        ts_extrapolation: str = "clamp",
        shuffle_ops: bool = False,
        shuffle_time: bool = False,
        seed: int = 42,
        loader: Callable[[str], OpBundle] = load_op,
    ) -> None:
        super().__init__()
        if subsample_time < 1:
            raise ValueError("subsample_time must be >= 1")
        self.op_ids = list(op_ids)
        self.normalizer = normalizer
        self.subsample_time = int(subsample_time)
        self.ts_extrapolation = ts_extrapolation
        self.shuffle_ops = shuffle_ops
        self.shuffle_time = shuffle_time
        self.seed = int(seed)
        self.loader = loader
        self.n_sensors = self._infer_n_sensors()

    def _infer_n_sensors(self) -> int:
        if not self.op_ids:
            return 0
        bundle = self.loader(self.op_ids[0])
        return int(bundle.xyz.shape[0])

    def _iter_op_ids(self, rng: np.random.Generator) -> list[str]:
        op_ids = list(self.op_ids)
        if self.shuffle_ops:
            rng.shuffle(op_ids)
        return op_ids

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + worker_id)

        for op_id in self._iter_op_ids(rng):
            bundle = self.loader(op_id)
            time_indices = np.arange(
                0,
                bundle.t_fast.shape[0],
                self.subsample_time,
                dtype=np.int64,
            )
            if self.shuffle_time:
                rng.shuffle(time_indices)

            for x_block, y_block, _sensor_ids, _time_index in iter_pointwise_blocks(
                bundle,
                time_indices,
                ts_extrapolation=self.ts_extrapolation,
            ):
                if self.normalizer is not None:
                    x_block = self.normalizer.transform_X(x_block)
                    y_block = self.normalizer.transform_Y(y_block)

                for row_index in range(x_block.shape[0]):
                    x_tensor = torch.from_numpy(x_block[row_index].astype(np.float32, copy=False))
                    y_tensor = torch.from_numpy(y_block[row_index].astype(np.float32, copy=False))
                    yield x_tensor, y_tensor
