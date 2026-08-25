"""Smoke tests for unified train dispatcher."""

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import torch
import yaml

from battery_surrogate.cli.train import train_from_config


MINIMAL_MLP_CONFIG = {
    "seed": 42,
    "data": {
        "train_ops": ["OP01"],
        "val_ops": ["OP02"],
        "test_ops": ["OP03"],
        "subsample_time": 50,
        "ts_extrapolation": "clamp",
    },
    "model": {
        "type": "mlp_pointwise",
        "n_hidden_layers": 1,
        "hidden_size": 32,
        "swish_beta_init": 1.0,
        "swish_beta_learnable": True,
    },
    "train": {
        "epochs": 1,
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "early_stopping_patience": 10,
    },
    "loss": {
        "T_weight": 1.0,
        "bc_V_weight": 1.0,
    },
    "output": {
        "ckpt_dir": "artifacts/test_mlp/{timestamp}",
    },
}


def test_train_from_config_mlp_smoke():
    summary = train_from_config(MINIMAL_MLP_CONFIG)

    assert "best_val_loss" in summary
    assert "ckpt_dir" in summary
    assert "best_ckpt" in summary
    assert "normalizer" in summary
    assert "config_path" in summary
    assert "model_type" in summary
    assert summary["model_type"] == "mlp_pointwise"

    ckpt_path = Path(summary["ckpt_dir"])
    assert ckpt_path.exists()
    assert (ckpt_path / "best.pt").exists()
    assert (ckpt_path / "normalizer.json").exists()
    assert (ckpt_path / "config.yaml").exists()

    shutil.rmtree(ckpt_path, ignore_errors=True)


def test_train_from_config_returns_required_keys():
    summary = train_from_config(MINIMAL_MLP_CONFIG)

    required_keys = [
        "best_val_loss",
        "ckpt_dir",
        "best_ckpt",
        "normalizer",
        "config_path",
        "n_parameters",
        "model_type",
        "n_sensors",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"

    shutil.rmtree(Path(summary["ckpt_dir"]), ignore_errors=True)


def test_train_from_config_checkpoint_loadable():
    summary = train_from_config(MINIMAL_MLP_CONFIG)

    ckpt_path = Path(summary["best_ckpt"])
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(ckpt, dict)
    assert "model_state" in ckpt

    for value in ckpt["model_state"].values():
        if isinstance(value, torch.Tensor):
            assert not torch.isnan(value).any(), "Checkpoint contains NaN values"

    shutil.rmtree(Path(summary["ckpt_dir"]), ignore_errors=True)


def test_train_from_config_saved_config_matches():
    summary = train_from_config(MINIMAL_MLP_CONFIG)

    config_path = Path(summary["config_path"])
    assert config_path.exists()

    saved_config = yaml.safe_load(config_path.read_text())
    assert saved_config["seed"] == MINIMAL_MLP_CONFIG["seed"]
    assert saved_config["model"]["type"] == MINIMAL_MLP_CONFIG["model"]["type"]
    assert saved_config["train"]["epochs"] == MINIMAL_MLP_CONFIG["train"]["epochs"]

    shutil.rmtree(Path(summary["ckpt_dir"]), ignore_errors=True)


def test_train_from_config_unknown_model_type():
    bad_config = deepcopy(MINIMAL_MLP_CONFIG)
    bad_config["model"]["type"] = "unknown_model"

    with pytest.raises(ValueError, match="Unknown model type"):
        train_from_config(bad_config)


def test_train_from_config_recurrent_smoke():
    recurrent_config = deepcopy(MINIMAL_MLP_CONFIG)
    recurrent_config["model"] = {
        "type": "recurrent",
        "rnn_type": "gru",
        "n_layers": 1,
        "hidden_size": 32,
        "history_length": 4,
    }
    recurrent_config["train"]["batch_size"] = 2
    recurrent_config["output"]["ckpt_dir"] = "artifacts/test_recurrent/{timestamp}"

    summary = train_from_config(recurrent_config)
    assert summary["model_type"] == "recurrent"

    ckpt_path = Path(summary["best_ckpt"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state" in ckpt

    shutil.rmtree(Path(summary["ckpt_dir"]), ignore_errors=True)
