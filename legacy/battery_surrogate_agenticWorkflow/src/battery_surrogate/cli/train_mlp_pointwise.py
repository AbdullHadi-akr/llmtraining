"""Train the point-wise MLP baseline from YAML config."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.loader import load_op
from ..model.dataset_pointwise import PointwiseDataset
from ..model.features_pointwise import iter_pointwise_blocks
from ..model.mlp_pointwise import PointwiseMLP
from ..model.normalizer import PointwiseNormalizer
from ..model.split import resolve_split
from ..model.trainer import train_model


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _set_seed(seed: int) -> None:
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
) -> PointwiseNormalizer:
    normalizer = PointwiseNormalizer()
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


def _build_model(config: dict[str, Any]) -> PointwiseMLP:
    model_cfg = config.get("model", {})
    return PointwiseMLP(
        n_features=11,
        n_hidden_layers=int(model_cfg.get("n_hidden_layers", 3)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        swish_beta_init=float(model_cfg.get("swish_beta_init", 1.0)),
        swish_beta_learnable=bool(model_cfg.get("swish_beta_learnable", True)),
    )


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    split = resolve_split(config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    subsample_time = int(data_cfg.get("subsample_time", 1))
    ts_extrapolation = str(data_cfg.get("ts_extrapolation", "clamp"))
    batch_size = int(train_cfg.get("batch_size", 4096))

    normalizer = _fit_normalizer(
        split["train"],
        subsample_time=subsample_time,
        ts_extrapolation=ts_extrapolation,
    )

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = _build_model(config)
    result = train_model(
        model,
        train_loader,
        val_loader,
        config,
        n_sensors=train_dataset.n_sensors,
    )

    normalizer_path = result.ckpt_dir / "normalizer.json"
    normalizer.save(normalizer_path)

    return {
        "best_val_loss": result.best_val_loss,
        "ckpt_dir": str(result.ckpt_dir),
        "best_ckpt": str(result.best_ckpt_path),
        "normalizer": str(normalizer_path),
        "n_parameters": model.n_parameters,
        "betas": model.betas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/model/mlp_pointwise.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--ckpt-dir", default=None, help="Override output.ckpt_dir")
    parser.add_argument(
        "--data.subsample_time",
        dest="subsample_time_override",
        type=int,
        default=None,
        help="Override data.subsample_time",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    config.setdefault("data", {})
    config.setdefault("train", {})
    config.setdefault("output", {})

    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.ckpt_dir is not None:
        config["output"]["ckpt_dir"] = str(args.ckpt_dir)
    if args.subsample_time_override is not None:
        config["data"]["subsample_time"] = int(args.subsample_time_override)

    summary = train_from_config(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
