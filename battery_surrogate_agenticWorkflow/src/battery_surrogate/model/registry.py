"""Model and dataset registry with dispatch on model type."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset_pointwise import PointwiseDataset
from .dataset_sequence import SequenceDataset
from .mlp_pointwise import PointwiseMLP
from .normalizer import PointwiseNormalizer
from .recurrent_pointwise import RecurrentPointwise


def build_model(
    config: dict[str, Any],
    n_sensors: int,
    seed: int,
) -> nn.Module:
    """
    Build a model based on config["model"]["type"].

    Parameters
    ----------
    config : dict
        Model configuration with "model" key containing "type" and model-specific params
    n_sensors : int
        Number of sensors (used by some models)
    seed : int
        Random seed for deterministic initialization

    Returns
    -------
    nn.Module
        The instantiated model

    Raises
    ------
    ValueError
        If model type is unknown
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "mlp_pointwise")

    if model_type == "mlp_pointwise":
        return _build_mlp(model_cfg, seed)
    elif model_type == "recurrent":
        return _build_recurrent(config, seed, n_sensors)
    else:
        msg = f"Unknown model type: {model_type}. Expected 'mlp_pointwise' or 'recurrent'."
        raise ValueError(msg)


def _build_mlp(model_cfg: dict[str, Any], seed: int) -> PointwiseMLP:
    """Build MLP model from config."""
    torch.manual_seed(seed)
    return PointwiseMLP(
        n_features=11,
        n_hidden_layers=int(model_cfg.get("n_hidden_layers", 3)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        swish_beta_init=float(model_cfg.get("swish_beta_init", 1.0)),
        swish_beta_learnable=bool(model_cfg.get("swish_beta_learnable", True)),
    )


def _build_recurrent(
    config: dict[str, Any],
    seed: int,
    n_sensors: int,
) -> nn.Module:
    """Build recurrent sequence model from config."""
    return RecurrentPointwise(config, n_sensors=n_sensors, seed=seed)


def build_datasets(
    config: dict[str, Any],
    normalizer: PointwiseNormalizer,
    split: dict[str, list[str]],
    seed: int,
) -> dict[str, DataLoader]:
    """
    Build train/val/test dataloaders based on model type.

    Parameters
    ----------
    config : dict
        Configuration with "model" and "data" sections
    normalizer : PointwiseNormalizer
        Fitted normalizer for data preprocessing
    split : dict
        Dictionary with keys "train", "val", "test" mapping to OP lists
    seed : int
        Random seed for shuffling

    Returns
    -------
    dict[str, DataLoader]
        Dictionary with keys "train", "val", "test"
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "mlp_pointwise")

    if model_type == "mlp_pointwise":
        return _build_pointwise_datasets(config, normalizer, split, seed)
    elif model_type == "recurrent":
        return _build_sequence_datasets(config, normalizer, split, seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_pointwise_datasets(
    config: dict[str, Any],
    normalizer: PointwiseNormalizer,
    split: dict[str, list[str]],
    seed: int,
) -> dict[str, DataLoader]:
    """Build pointwise (MLP) dataloaders."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    subsample_time = int(data_cfg.get("subsample_time", 1))
    ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))
    batch_size = int(train_cfg.get("batch_size", 4096))

    train_dataset = PointwiseDataset(
        split["train"],
        normalizer=normalizer,
        subsample_time=subsample_time,
        ts_extrapolation=ts_extrapolation,
        shuffle_ops=True,
        shuffle_time=True,
        seed=seed,
    )
    val_dataset = PointwiseDataset(
        split["val"],
        normalizer=normalizer,
        subsample_time=subsample_time,
        ts_extrapolation=ts_extrapolation,
        shuffle_ops=False,
        shuffle_time=False,
        seed=seed,
    )
    test_dataset = PointwiseDataset(
        split["test"],
        normalizer=normalizer,
        subsample_time=subsample_time,
        ts_extrapolation=ts_extrapolation,
        shuffle_ops=False,
        shuffle_time=False,
        seed=seed,
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }


def _build_sequence_datasets(
    config: dict[str, Any],
    normalizer: PointwiseNormalizer,
    split: dict[str, list[str]],
    seed: int,
) -> dict[str, DataLoader]:
    """Build sequence (recurrent) dataloaders."""
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))

    train_dataset = SequenceDataset(
        split["train"],
        normalizer=normalizer,
        config=config,
        shuffle_ops=True,
        seed=seed,
    )
    val_dataset = SequenceDataset(
        split["val"],
        normalizer=normalizer,
        config=config,
        shuffle_ops=False,
        seed=seed,
    )
    test_dataset = SequenceDataset(
        split["test"],
        normalizer=normalizer,
        config=config,
        shuffle_ops=False,
        seed=seed,
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }
