from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from battery_surrogate.cli.train_mlp_pointwise import _fit_normalizer, train_from_config
from battery_surrogate.data.loader import load_op
from battery_surrogate.data.paths import project_root
from battery_surrogate.model.dataset_pointwise import PointwiseDataset
from battery_surrogate.model.evaluate import evaluate_on_ops
from battery_surrogate.model.features_pointwise import iter_pointwise_blocks
from battery_surrogate.model.mlp_pointwise import PointwiseMLP
from battery_surrogate.model.split import resolve_split
from battery_surrogate.model.trainer import train_model


def predict_timeseries(op_id, subsample_time, normalizer, model, device, ts_extrap):
    bundle = load_op(op_id)
    time_indices = np.arange(0, bundle.t_fast.shape[0], subsample_time, dtype=np.int64)
    times, t_true, t_pred, bcv_true, bcv_pred = [], [], [], [], []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x_block, y_block, _sids, ti in iter_pointwise_blocks(
            bundle, time_indices, ts_extrapolation=ts_extrap
        ):
            x_norm = normalizer.transform_X(x_block)
            pred = normalizer.inverse_Y(model(torch.from_numpy(x_norm).to(device)).cpu().numpy())
            times.append(float(bundle.t_fast[ti]))
            t_true.append(y_block[:, 0])
            t_pred.append(pred[:, 0])
            bcv_true.append(float(y_block[0, 1]))
            bcv_pred.append(float(np.mean(pred[:, 1])))
    return (
        np.array(times),
        np.array(t_true),
        np.array(t_pred),
        np.array(bcv_true),
        np.array(bcv_pred),
        bundle,
    )


def main() -> None:
    print("counter 1/8 setup", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = project_root()
    config_path = root / "configs" / "model" / "mlp_pointwise.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config["data"]["train_ops"] = ["OP01", "OP02"]
    config["data"]["val_ops"] = ["OP14"]
    config["data"]["test_ops"] = ["OP16"]
    config["data"]["subsample_time"] = 300
    config["model"]["n_hidden_layers"] = 3
    config["model"]["hidden_size"] = 64
    config["train"]["epochs"] = 4
    config["train"]["batch_size"] = 4096
    config["output"]["ckpt_dir"] = "artifacts/mlp_pointwise_demo/{timestamp}"

    subsample = config["data"]["subsample_time"]
    ts_extrap = config["data"]["ts_extrapolation"]
    split = resolve_split(config)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = root / "notebooks" / "plots" / f"mlp_pointwise_demo_{run_tag}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("counter 2/8 fit normalizer", flush=True)
    t0 = time.time()
    normalizer = _fit_normalizer(split["train"], subsample_time=subsample, ts_extrapolation=ts_extrap)
    print(f"normalizer_fit_seconds={time.time() - t0:.2f}", flush=True)

    print("counter 3/8 build data+model", flush=True)
    train_ds = PointwiseDataset(
        split["train"],
        normalizer=normalizer,
        subsample_time=subsample,
        ts_extrapolation=ts_extrap,
        shuffle_ops=True,
        shuffle_time=True,
        seed=42,
    )
    val_ds = PointwiseDataset(
        split["val"],
        normalizer=normalizer,
        subsample_time=subsample,
        ts_extrapolation=ts_extrap,
        shuffle_ops=False,
        shuffle_time=False,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"])

    model = PointwiseMLP(
        n_features=11,
        n_hidden_layers=config["model"]["n_hidden_layers"],
        hidden_size=config["model"]["hidden_size"],
        swish_beta_init=config["model"]["swish_beta_init"],
        swish_beta_learnable=config["model"]["swish_beta_learnable"],
    )

    print("counter 4/8 train model", flush=True)
    t0 = time.time()
    result = train_model(model, train_loader, val_loader, config, n_sensors=train_ds.n_sensors, device=device)
    normalizer.save(result.ckpt_dir / "normalizer.json")
    print(f"train_seconds={time.time() - t0:.2f}", flush=True)
    print(f"best_val_loss={result.best_val_loss:.6f}", flush=True)

    print("counter 5/8 save training plots", flush=True)
    history = result.history
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="s", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (normalized)")
    ax1.set_yscale("log")
    ax1.set_title("Training / validation loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    betas = model.betas
    ax2.bar(range(1, len(betas) + 1), betas, color="tab:purple")
    ax2.axhline(1.0, color="gray", ls="--", label="init = 1.0")
    ax2.set_xlabel("hidden layer")
    ax2.set_ylabel("learned beta")
    ax2.set_title("Learned Swish beta per layer")
    ax2.set_xticks(range(1, len(betas) + 1))
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_loss_and_betas.png", dpi=160)
    plt.close(fig)

    print("counter 6/8 evaluate and timeseries plots", flush=True)
    metrics = evaluate_on_ops(
        model,
        split["test"],
        normalizer,
        subsample_time=subsample,
        ts_extrapolation=ts_extrap,
        device=device,
    )

    test_op = split["test"][0]
    times, t_true, t_pred, bcv_true, bcv_pred, test_bundle = predict_timeseries(
        test_op, subsample, normalizer, model, device, ts_extrap
    )
    sensor_idx = t_true.shape[1] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(times, t_true[:, sensor_idx], label="true", lw=2)
    ax1.plot(times, t_pred[:, sensor_idx], label="pred", ls="--", lw=2)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("T (degC)")
    ax1.set_title(f"{test_op}: T at sensor {sensor_idx}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, bcv_true, label="true", lw=2)
    ax2.plot(times, bcv_pred, label="pred (mean over sensors)", ls="--", lw=2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("bc_V (V)")
    ax2.set_title(f"{test_op}: cell voltage")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_timeseries.png", dpi=160)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    flat_true = t_true.ravel()
    flat_pred = t_pred.ravel()
    lo = float(min(flat_true.min(), flat_pred.min()))
    hi = float(max(flat_true.max(), flat_pred.max()))
    ax1.scatter(flat_true, flat_pred, s=4, alpha=0.2)
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax1.set_xlabel("true T (degC)")
    ax1.set_ylabel("predicted T (degC)")
    ax1.set_title("Parity: T")
    ax1.grid(True, alpha=0.3)

    t_last = -1
    abs_err = np.abs(t_pred[t_last] - t_true[t_last])
    xyz = test_bundle.xyz
    sc = ax2.scatter(xyz[:, 0], xyz[:, 1], c=abs_err, cmap="viridis", s=25)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"|T error| at t={times[t_last]:.0f}s")
    fig.colorbar(sc, ax=ax2, label="|error| (degC)")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_parity_and_spatial_error.png", dpi=160)
    plt.close(fig)

    print("counter 7/8 short optuna", flush=True)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        cfg = copy.deepcopy(config)
        cfg["model"]["n_hidden_layers"] = trial.suggest_int("n_hidden_layers", 2, 4)
        cfg["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128])
        cfg["train"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        cfg["train"]["epochs"] = 2
        cfg["output"]["ckpt_dir"] = f"artifacts/mlp_pointwise_optuna/trial_{trial.number}"
        summary = train_from_config(cfg)
        print(f"optuna_trial={trial.number} best_val={summary['best_val_loss']:.6f}", flush=True)
        return summary["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    trials = [t for t in study.trials if t.value is not None]
    vals = [t.value for t in trials]
    best_so_far = np.minimum.accumulate(vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(len(vals)), vals, "o", label="trial value")
    ax1.plot(range(len(vals)), best_so_far, "-", color="tab:red", label="best so far")
    ax1.set_xlabel("trial")
    ax1.set_ylabel("best val loss")
    ax1.set_yscale("log")
    ax1.set_title("Optuna optimization history")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    sizes = [t.params["hidden_size"] for t in trials]
    lrs = [t.params["lr"] for t in trials]
    sc = ax2.scatter(lrs, vals, c=sizes, cmap="plasma", s=60)
    ax2.set_xscale("log")
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("best val loss")
    ax2.set_yscale("log")
    ax2.set_title("val loss vs lr (color = hidden size)")
    fig.colorbar(sc, ax=ax2, label="hidden size")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_optuna_history.png", dpi=160)
    plt.close(fig)

    print("counter 8/8 write summary", flush=True)
    summary = {
        "device": str(device),
        "split": split,
        "best_val_loss": float(result.best_val_loss),
        "metrics": metrics,
        "optuna_best_params": study.best_params,
        "optuna_best_val_loss": float(study.best_value),
        "plot_dir": str(plot_dir),
        "ckpt_dir": str(result.ckpt_dir),
    }
    summary_path = plot_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"plots_saved={plot_dir}", flush=True)
    print(f"summary_file={summary_path}", flush=True)


if __name__ == "__main__":
    main()
