"""Run phased Optuna search for the point-wise MLP baseline."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from .train_mlp_pointwise import train_from_config


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/mlp_pointwise.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise SystemExit("Optuna is not installed. Install with 'pip install optuna'.") from exc

    base_config = _load_config(Path(args.config))

    def objective(trial: "optuna.Trial") -> float:
        config = copy.deepcopy(base_config)
        config.setdefault("model", {})
        config.setdefault("train", {})

        config["model"]["n_hidden_layers"] = trial.suggest_int("n_hidden_layers", 2, 5)
        config["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
        config["train"]["lr"] = trial.suggest_float("lr", 1.0e-4, 1.0e-2, log=True)
        config["train"]["epochs"] = min(int(config["train"].get("epochs", 50)), 20)

        summary = train_from_config(config)
        return float(summary["best_val_loss"])

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=int(args.n_trials))

    print("Best trial:")
    print(study.best_trial.number)
    print("Best value:")
    print(study.best_value)
    print("Best params:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
