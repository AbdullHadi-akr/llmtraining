"""Unified Optuna hyperparameter search for MLP and recurrent models."""

from __future__ import annotations

from typing import Any, Callable

import copy
import pandas as pd

from .train import train_from_config


def run_mlp_study(
    base_config: dict[str, Any],
    *,
    n_trials: int = 30,
    epochs_cap: int = 20,
    progress_cb: Callable | None = None,
):
    """
    Run Optuna hyperparameter search for MLP model.

    Parameters
    ----------
    base_config : dict
        Base configuration (will be modified for each trial)
    n_trials : int
        Number of trials, default 30
    epochs_cap : int
        Maximum epochs per trial, default 20
    progress_cb : Callable, optional
        Progress callback: progress_cb(trial_num, n_trials, best_loss)

    Returns
    -------
    optuna.Study
        Completed study object
    """
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        raise RuntimeError(
            "Optuna not installed. Install with: pip install optuna"
        )

    def objective(trial):
        config = copy.deepcopy(base_config)
        
        # Suggest hyperparameters
        n_hidden_layers = trial.suggest_int("n_hidden_layers", 2, 5)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # Update config
        config["model"]["n_hidden_layers"] = n_hidden_layers
        config["model"]["hidden_size"] = hidden_size
        config["train"]["epochs"] = min(
            config["train"].get("epochs", 50),
            epochs_cap
        )
        config["train"]["lr"] = lr
        
        # Train and return best validation loss
        try:
            result = train_from_config(config, seed=42, verbose=False)
            val_loss = result.best_val_loss
            return val_loss
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float("inf")
    
    # Create and run study
    study = optuna.create_study(direction="minimize")
    
    # Custom callback for progress
    if progress_cb:
        def trial_callback(study, trial):
            best_loss = study.best_value if study.best_trial else float("inf")
            progress_cb(len(study.trials), n_trials, f"Best: {best_loss:.6f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])
    else:
        study.optimize(objective, n_trials=n_trials)
    
    return study


def run_recurrent_study(
    base_config: dict[str, Any],
    *,
    n_trials: int = 30,
    epochs_cap: int = 10,
    progress_cb: Callable | None = None,
    per_target_history: bool = True,
):
    """
    Run Optuna hyperparameter search for recurrent model.

    Parameters
    ----------
    base_config : dict
        Base configuration (will be modified for each trial)
    n_trials : int
        Number of trials, default 30
    epochs_cap : int
        Maximum epochs per trial, default 10
    progress_cb : Callable, optional
        Progress callback
    per_target_history : bool
        Whether to optimize per-target history lengths, default True

    Returns
    -------
    optuna.Study
        Completed study object
    """
    try:
        import optuna
    except ImportError:
        raise RuntimeError(
            "Optuna not installed. Install with: pip install optuna"
        )

    def objective(trial):
        config = copy.deepcopy(base_config)
        
        # Suggest hyperparameters
        rnn_type = trial.suggest_categorical("rnn_type", ["gru", "lstm"])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        if per_target_history:
            k_T = trial.suggest_int("k_T", 1, 32)
            k_V = trial.suggest_int("k_V", 1, 32)
            config["model"]["history_length"] = {"T": k_T, "bc_V": k_V}
        else:
            k = trial.suggest_int("history_length", 1, 32)
            config["model"]["history_length"] = k
        
        # Update config
        config["model"]["rnn_type"] = rnn_type
        config["model"]["n_layers"] = n_layers
        config["model"]["hidden_size"] = hidden_size
        config["train"]["epochs"] = min(
            config["train"].get("epochs", 50),
            epochs_cap
        )
        config["train"]["lr"] = lr
        
        # Train and return best validation loss
        try:
            result = train_from_config(config, seed=42, verbose=False)
            val_loss = result.best_val_loss
            return val_loss
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float("inf")
    
    # Create and run study
    study = optuna.create_study(direction="minimize")
    
    # Custom callback for progress
    if progress_cb:
        def trial_callback(study, trial):
            best_loss = study.best_value if study.best_trial else float("inf")
            progress_cb(len(study.trials), n_trials, f"Best: {best_loss:.6f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])
    else:
        study.optimize(objective, n_trials=n_trials)
    
    return study


def study_to_dataframe(study) -> pd.DataFrame:
    """
    Convert Optuna study to DataFrame.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study

    Returns
    -------
    pd.DataFrame
        Trial results with params and value
    """
    import optuna
    
    if not isinstance(study, optuna.Study):
        raise TypeError("study must be an optuna.Study object")
    
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value if trial.value is not None else float("inf"),
            **trial.params,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values("value").reset_index(drop=True)
    
    return df
