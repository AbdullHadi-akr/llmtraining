"""Training helpers for recurrent point-wise sequence models."""

from __future__ import annotations

import json
from pathlib import Path
from time import strftime
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .trainer import TrainResult
from .features_sequence import resolve_history_lengths


def _resolve_ckpt_dir(template: str) -> Path:
    timestamp = strftime("%Y%m%d-%H%M%S")
    return Path(template.replace("{timestamp}", timestamp))


def _sequence_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    n_sensors: int,
    t_weight: float,
    bc_v_weight: float,
) -> torch.Tensor:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must have shape (batch, seq, 2)")

    pred_eff = prediction[:, k:, :]
    target_eff = target[:, k:, :]
    if pred_eff.numel() == 0:
        pred_eff = prediction
        target_eff = target

    mse = nn.functional.mse_loss
    t_loss = mse(pred_eff[..., 0], target_eff[..., 0])
    v_loss = mse(pred_eff[..., 1], target_eff[..., 1])
    return (t_weight * t_loss) + (bc_v_weight * (v_loss / float(max(n_sensors, 1))))


def _evaluate_val_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    k: int,
    n_sensors: int,
    t_weight: float,
    bc_v_weight: float,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features, history, targets, _seq_len, _op_id, _sensor_id = batch
            features = features.to(device=device, dtype=torch.float32)
            history = history.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            pred = model(features, history)
            loss = _sequence_loss(
                pred,
                targets,
                k=k,
                n_sensors=n_sensors,
                t_weight=t_weight,
                bc_v_weight=bc_v_weight,
            )
            losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    *,
    n_sensors: int,
    device: torch.device | None = None,
    verbose: bool = False,
    progress_cb = None,
) -> TrainResult:
    """Train a recurrent model with teacher forcing history tensors."""

    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    out_cfg = config.get("output", {})
    model_cfg = config.get("model", {})

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", train_cfg.get("learning_rate", 1.0e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 10))

    t_weight = float(loss_cfg.get("T_weight", 1.0))
    bc_v_weight = float(loss_cfg.get("bc_V_weight", 1.0))

    k_T, k_V = resolve_history_lengths(model_cfg)
    k = max(k_T, k_V)  # use max for warm-up mask offset

    ckpt_template = str(out_cfg.get("ckpt_dir", "artifacts/recurrent/{timestamp}"))
    ckpt_dir = _resolve_ckpt_dir(ckpt_template)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best.pt"
    epochs_without_improvement = 0

    for epoch_num in range(epochs):
        model.train()
        running: list[float] = []

        for batch in train_loader:
            features, history_seq, targets, _seq_len, _op_id, _sensor_id = batch
            features = features.to(device=device, dtype=torch.float32)
            history_seq = history_seq.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(features, history_seq)
            loss = _sequence_loss(
                pred,
                targets,
                k=k,
                n_sensors=n_sensors,
                t_weight=t_weight,
                bc_v_weight=bc_v_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            running.append(float(loss.item()))

        train_loss = float(sum(running) / max(len(running), 1))
        val_loss = _evaluate_val_loss(
            model,
            val_loader,
            device=device,
            k=k,
            n_sensors=n_sensors,
            t_weight=t_weight,
            bc_v_weight=bc_v_weight,
        )

        prev_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step(val_loss)
        new_lrs = [group["lr"] for group in optimizer.param_groups]
        lr_dropped = any(new_lr < prev_lr for new_lr, prev_lr in zip(new_lrs, prev_lrs))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if lr_dropped:
                epochs_without_improvement = 0
            if epochs_without_improvement >= patience:
                break

        # Progress reporting
        if progress_cb:
            progress_cb(
                epoch_num + 1,
                epochs,
                f"train={train_loss:.4e} val={val_loss:.4e} best={best_val_loss:.4e}"
            )
        elif verbose:
            print(
                f"Epoch {epoch_num+1}/{epochs} | "
                f"train={train_loss:.4e} val={val_loss:.4e} best={best_val_loss:.4e}"
            )

    history_path = ckpt_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return TrainResult(
        history=history,
        best_val_loss=best_val_loss,
        ckpt_dir=ckpt_dir,
        best_ckpt_path=best_ckpt_path,
    )
