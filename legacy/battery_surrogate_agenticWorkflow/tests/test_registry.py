"""Tests for model and dataset registry dispatch."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from battery_surrogate.model.mlp_pointwise import PointwiseMLP
from battery_surrogate.model.normalizer import PointwiseNormalizer
from battery_surrogate.model.recurrent_pointwise import RecurrentPointwise
from battery_surrogate.model.registry import build_datasets, build_model


def test_build_model_mlp_pointwise():
    config = {
        "model": {
            "type": "mlp_pointwise",
            "n_hidden_layers": 2,
            "hidden_size": 64,
        }
    }
    model = build_model(config, n_sensors=50, seed=42)
    assert isinstance(model, PointwiseMLP)


def test_build_model_recurrent():
    config = {
        "model": {
            "type": "recurrent",
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 32,
            "history_length": 4,
        }
    }
    model = build_model(config, n_sensors=50, seed=42)
    assert isinstance(model, RecurrentPointwise)


def test_build_model_unknown_type():
    config = {"model": {"type": "unknown_model"}}
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(config, n_sensors=50, seed=42)


def _fit_normalizer() -> PointwiseNormalizer:
    normalizer = PointwiseNormalizer()
    x_dummy = np.random.randn(100, 11).astype(np.float32)
    y_dummy = np.random.randn(100, 2).astype(np.float32)
    normalizer.partial_fit(x_dummy, y_dummy)
    normalizer.finalize()
    return normalizer


class _TinyPointwiseDataset:
    def __init__(self, *args, **kwargs):
        self.n_sensors = 50

    def __iter__(self):
        for _ in range(2):
            yield torch.zeros(11, dtype=torch.float32), torch.zeros(2, dtype=torch.float32)


class _TinySequenceDataset:
    def __init__(self, *args, **kwargs):
        self.n_sensors = 50

    def __iter__(self):
        for _ in range(2):
            yield (
                torch.zeros(5, 11, dtype=torch.float32),
                torch.zeros(5, 8, dtype=torch.float32),
                torch.zeros(5, 2, dtype=torch.float32),
                5,
                "OP01",
                0,
            )


def test_build_datasets_mlp_pointwise(monkeypatch):
    monkeypatch.setattr("battery_surrogate.model.registry.PointwiseDataset", _TinyPointwiseDataset)

    config = {
        "model": {"type": "mlp_pointwise"},
        "data": {"subsample_time": 10, "ts_extrapolation": "clamp"},
        "train": {"batch_size": 32},
    }

    split = {"train": ["OP01"], "val": ["OP02"], "test": ["OP03"]}
    dataloaders = build_datasets(config, _fit_normalizer(), split, seed=42)

    assert set(dataloaders.keys()) == {"train", "val", "test"}
    assert all(isinstance(dl, DataLoader) for dl in dataloaders.values())


def test_build_datasets_recurrent(monkeypatch):
    monkeypatch.setattr("battery_surrogate.model.registry.SequenceDataset", _TinySequenceDataset)

    config = {
        "model": {
            "type": "recurrent",
            "history_length": 4,
            "rnn_type": "gru",
            "n_layers": 1,
            "hidden_size": 32,
        },
        "data": {"subsample_time": 10, "ts_extrapolation": "clamp"},
        "train": {"batch_size": 2},
    }

    split = {"train": ["OP01"], "val": ["OP02"], "test": ["OP03"]}
    dataloaders = build_datasets(config, _fit_normalizer(), split, seed=42)

    assert set(dataloaders.keys()) == {"train", "val", "test"}
    assert all(isinstance(dl, DataLoader) for dl in dataloaders.values())


def test_build_datasets_unknown_type():
    config = {
        "model": {"type": "unknown_type"},
        "data": {"subsample_time": 10},
        "train": {"batch_size": 32},
    }
    split = {"train": ["OP01"], "val": ["OP02"], "test": ["OP03"]}

    with pytest.raises(ValueError, match="Unknown model type"):
        build_datasets(config, _fit_normalizer(), split, seed=42)
