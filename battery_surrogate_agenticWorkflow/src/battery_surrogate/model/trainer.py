"""Training helpers for the point-wise MLP baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import strftime
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    """Container for training artifacts and history."""

    history: dict[str, list[float]]
    best_val_loss: float
    ckpt_dir: Path
    best_ckpt_path: Path


def _batch_to_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError("Unsupported batch type for point-wise trainer")


def _resolve_ckpt_dir(template: str) -> Path:
    timestamp = strftime("%Y%m%d-%H%M%S")
    value = template.replace("{timestamp}", timestamp)
    return Path(value)


def _compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    n_sensors: int,
    t_weight: float,
    bc_v_weight: float,
) -> torch.Tensor:
    mse = nn.functional.mse_loss
    t_loss = mse(prediction[:, 0], target[:, 0])
    v_loss = mse(prediction[:, 1], target[:, 1])
    return (t_weight * t_loss) + (bc_v_weight * (v_loss / float(max(n_sensors, 1))))


def _evaluate_val_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    n_sensors: int,
    t_weight: float,
    bc_v_weight: float,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x, y = _batch_to_xy(batch)
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = _compute_loss(
                pred,
                y,
                n_sensors=n_sensors,
                t_weight=t_weight,
                bc_v_weight=bc_v_weight,
            )
            losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    *,
    n_sensors: int,
    device: torch.device | None = None,
) -> TrainResult:
    """Train the point-wise MLP and save the best checkpoint."""

    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    out_cfg = config.get("output", {})

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 10))

    t_weight = float(loss_cfg.get("T_weight", 1.0))
    bc_v_weight = float(loss_cfg.get("bc_V_weight", 1.0))

    ckpt_template = str(out_cfg.get("ckpt_dir", "artifacts/mlp_pointwise/{timestamp}"))
    ckpt_dir = _resolve_ckpt_dir(ckpt_template)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best.pt"
    epochs_without_improvement = 0

    for _epoch in range(epochs):
        model.train()
        running: list[float] = []
        for batch in train_loader:
            x, y = _batch_to_xy(batch)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = _compute_loss(
                pred,
                y,
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

    history_path = ckpt_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return TrainResult(
        history=history,
        best_val_loss=best_val_loss,
        ckpt_dir=ckpt_dir,
        best_ckpt_path=best_ckpt_path,
    )
