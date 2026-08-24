"""Unified training entry point that dispatches on model type."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..data.loader import load_op
from ..model.features_pointwise import iter_pointwise_blocks
from ..model.normalizer import PointwiseNormalizer
from ..model.registry import build_datasets, build_model
from ..model.split import resolve_split, validate_coverage
from ..model.trainer import train_model
from ..model.trainer_sequence import train_sequence_model


# Setup logger for validate_coverage warnings
logger = logging.getLogger(__name__)


def _load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config file."""
    if isinstance(path, str):
        path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_normalizer(
    train_ops: list[str],
    *,
    subsample_time: int,
    ts_extrapolation: str,
    preprocess_config: dict[str, Any] | None = None,
) -> PointwiseNormalizer:
    """
    Fit normalizer on training OPs.

    Parameters
    ----------
    train_ops : list[str]
        List of training OP IDs
    subsample_time : int
        Time subsampling factor
    ts_extrapolation : str
        Time series extrapolation mode (clamp, repeat, etc.)
    preprocess_config : dict | None
        Preprocessing configuration for per-group scaling (Phase 2).
        If None, uses default all-z-score preset (backward compat).

    Returns
    -------
    PointwiseNormalizer
        Fitted normalizer
    """
    normalizer = PointwiseNormalizer(preprocess_config=preprocess_config)
    for op_id in train_ops:
        bundle = load_op(op_id)
        time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time, dtype=np.int64)
        for x_block, y_block, _sensor_ids, _time_index in iter_pointwise_blocks(
            bundle,
            time_indices,
            ts_extrapolation=ts_extrapolation,
        ):
            normalizer.partial_fit(x_block, y_block)
    normalizer.finalize()
    return normalizer


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Train model from configuration dictionary.

    This unified entry point dispatches on model type and handles:
    - OP splitting
    - Normalizer fitting
    - Dataset / dataloader construction
    - Model building
    - Training via the appropriate trainer
    - Checkpoint and normalizer saving

    Parameters
    ----------
    config : dict
        Configuration dictionary with sections:
        - seed (int): random seed
        - data: train_ops, val_ops, test_ops, subsample_time, ts_extrapolation
        - model: type (mlp_pointwise | recurrent), model-specific params
        - train: epochs, batch_size, lr, weight_decay, etc.
        - loss: T_weight, bc_V_weight, etc.
        - output: ckpt_dir template
        - preprocess (optional): per-group scaling config (Phase 2)

    Returns
    -------
    dict[str, Any]
        Summary with keys: ckpt_dir, best_val_loss, best_ckpt,
        normalizer, n_parameters, model_type, etc.
    """
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    # Resolve OP split
    split = resolve_split(config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "mlp_pointwise")

    subsample_time = int(data_cfg.get("subsample_time", 1))
    ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))

    # Fit normalizer on training OPs
    preprocess_config = config.get("preprocess", None)
    normalizer = _fit_normalizer(
        split["train"],
        subsample_time=subsample_time,
        ts_extrapolation=ts_extrapolation,
        preprocess_config=preprocess_config,
    )

    # Validate coverage: check if val/test OPs are within training distribution (Phase 2)
    coverage_warnings = validate_coverage(
        split,
        normalizer_stats={
            "x_mean": normalizer.x_mean.tolist() if normalizer.x_mean is not None else None,
            "x_std": normalizer.x_std.tolist() if normalizer.x_std is not None else None,
            "x_min": normalizer.x_min.tolist() if normalizer.x_min is not None else None,
            "x_max": normalizer.x_max.tolist() if normalizer.x_max is not None else None,
        },
    )
    for warning in coverage_warnings:
        logger.warning(warning)

    # Build datasets via registry
    dataloaders = build_datasets(config, normalizer, split, seed)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Determine n_sensors from first train dataset
    # (all datasets share same sensors, so just inspect train)
    train_dataset = train_loader.dataset
    n_sensors = train_dataset.n_sensors

    # Build model via registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, n_sensors=n_sensors, seed=seed)

    # Train model with type-specific trainer.
    if model_type == "mlp_pointwise":
        result = train_model(
            model,
            train_loader,
            val_loader,
            config,
            n_sensors=n_sensors,
            device=device,
        )
    elif model_type == "recurrent":
        result = train_sequence_model(
            model,
            train_loader,
            val_loader,
            config,
            n_sensors=n_sensors,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save normalizer
    normalizer_path = result.ckpt_dir / "normalizer.json"
    normalizer.save(normalizer_path)

    # Save config for reproducibility
    config_path = result.ckpt_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return {
        "best_val_loss": result.best_val_loss,
        "ckpt_dir": str(result.ckpt_dir),
        "best_ckpt": str(result.best_ckpt_path),
        "normalizer": str(normalizer_path),
        "config_path": str(config_path),
        "n_parameters": model.n_parameters if hasattr(model, "n_parameters") else 0,
        "model_type": model_type,
        "n_sensors": n_sensors,
        "betas": model.betas if hasattr(model, "betas") else None,
    }
