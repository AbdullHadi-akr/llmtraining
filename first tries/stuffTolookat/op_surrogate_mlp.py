from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _find_op_folder(dataset_root: Path, op_id: int) -> Path:
    candidates = [
        dataset_root / f"OP{op_id}" / f"OP{op_id}",
        dataset_root / f"OP{op_id}",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find folder for OP{op_id} in {dataset_root}")


def _read_csv_headers_and_data(file_path: Path) -> Tuple[list[str], np.ndarray]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {file_path}")

    headers = [h.strip() for h in rows[0]]
    if len(rows) == 1:
        return headers, np.empty((0, len(headers)), dtype=np.float32)

    data = np.array(rows[1:], dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return headers, data


def _find_column_index(headers: list[str], contains: str) -> int:
    needle = contains.lower()
    for i, name in enumerate(headers):
        if needle in name.lower():
            return i
    raise KeyError(f"Could not find column containing '{contains}' in headers: {headers}")


@dataclass
class SurrogateData:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_op_surrogate_data(
    op_id: int = 1,
    dataset_root: str | Path = r"c:\Users\M0245635\Downloads",
) -> SurrogateData:
    root = Path(dataset_root)
    op_folder = _find_op_folder(root, op_id)

    input_headers, input_data = _read_csv_headers_and_data(
        op_folder / f"OP{op_id}_Input Signale.csv"
    )
    fluid_headers, fluid_data = _read_csv_headers_and_data(
        op_folder / f"OP{op_id}_Fluidstoffwerte.csv"
    )
    fmu_headers, fmu_data = _read_csv_headers_and_data(op_folder / f"OP{op_id}_Batemo FMU1.csv")

    if input_data.shape[0] < 1 or fluid_data.shape[0] < 1 or fmu_data.shape[0] < 2:
        raise ValueError(
            "Not enough samples in one of the OP files to build a time series dataset"
        )

    t_idx = _find_column_index(fmu_headers, "Physical Time")
    q_idx = _find_column_index(fmu_headers, "bc_Q Monitor")
    soc_idx = _find_column_index(fmu_headers, "bc_SOC Monitor")

    t = fmu_data[:, t_idx: t_idx + 1]
    y = np.hstack(
        [fmu_data[:, soc_idx: soc_idx + 1], fmu_data[:, q_idx: q_idx + 1]]
    ).astype(np.float32)

    input_vec = input_data[0].astype(np.float32)
    fluid_vec = fluid_data[0].astype(np.float32)
    static_vec = np.concatenate([input_vec, fluid_vec], axis=0)
    static_matrix = np.repeat(static_vec.reshape(1, -1), t.shape[0], axis=0)

    x = np.hstack([t.astype(np.float32), static_matrix]).astype(np.float32)
    feature_names = ["t"] + input_headers + fluid_headers
    target_names = ["bc_SOC", "bc_Q"]

    return SurrogateData(
        x=x,
        y=y,
        t=t.astype(np.float32),
        feature_names=feature_names,
        target_names=target_names,
    )


def time_series_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Dict[str, np.ndarray]:
    n = x.shape[0]
    split = max(2, int(n * train_ratio))
    split = min(split, n - 1)

    return {
        "x_train": x[:split],
        "y_train": y[:split],
        "x_test": x[split:],
        "y_test": y[split:],
    }


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    x_mu = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-8
    y_mu = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    return {
        "x_train_n": (x_train - x_mu) / x_std,
        "x_test_n": (x_test - x_mu) / x_std,
        "y_train_n": (y_train - y_mu) / y_std,
        "y_test_n": (y_test - y_mu) / y_std,
        "x_mu": x_mu,
        "x_std": x_std,
        "y_mu": y_mu,
        "y_std": y_std,
    }


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(in_dim: int, out_dim: int, use_modulus: bool = True) -> Tuple[nn.Module, str]:
    if use_modulus:
        try:
            from modulus.sym.key import Key
            from modulus.sym.models.fully_connected import FullyConnectedArch

            input_keys = [Key(f"x_{i}") for i in range(in_dim)]
            output_keys = [Key(f"y_{i}") for i in range(out_dim)]
            model = FullyConnectedArch(
                input_keys=input_keys,
                output_keys=output_keys,
                nr_layers=3,
                layer_size=64,
            )
            return model, "modulus"
        except Exception:
            local_modulus = Path(__file__).resolve().parent / "modulus-sym"
            if local_modulus.is_dir() and str(local_modulus) not in sys.path:
                sys.path.insert(0, str(local_modulus))
            try:
                from modulus.sym.key import Key
                from modulus.sym.models.fully_connected import FullyConnectedArch

                input_keys = [Key(f"x_{i}") for i in range(in_dim)]
                output_keys = [Key(f"y_{i}") for i in range(out_dim)]
                model = FullyConnectedArch(
                    input_keys=input_keys,
                    output_keys=output_keys,
                    nr_layers=3,
                    layer_size=64,
                )
                return model, "modulus"
            except Exception:
                pass

    return TorchMLP(in_dim, out_dim), "torch"


def _forward(model: nn.Module, x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "modulus":
        in_dict = {f"x_{i}": x[:, i: i + 1] for i in range(x.shape[1])}
        out_dict = model(in_dict)
        out_tensors = [out_dict[f"y_{i}"] for i in range(len(out_dict))]
        return torch.cat(out_tensors, dim=1)
    return model(x)


def train_surrogate(
    x_train_n: np.ndarray,
    y_train_n: np.ndarray,
    epochs: int = 500,
    lr: float = 1e-3,
    use_modulus: bool = True,
    device: str | None = None,
) -> Dict[str, object]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_t = torch.tensor(x_train_n, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train_n, dtype=torch.float32, device=device)

    model, backend = build_model(
        in_dim=x_t.shape[1],
        out_dim=y_t.shape[1],
        use_modulus=use_modulus,
    )
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        pred = _forward(model, x_t, backend)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optim.step()

        history.append(float(loss.detach().cpu().item()))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1}/{epochs}, train_loss={history[-1]:.6f}")

    return {"model": model, "backend": backend, "history": history, "device": device}


@torch.no_grad()
def predict_surrogate(model: nn.Module, backend: str, x_n: np.ndarray, device: str) -> np.ndarray:
    x_t = torch.tensor(x_n, dtype=torch.float32, device=device)
    model.eval()
    pred = _forward(model, x_t, backend)
    return pred.detach().cpu().numpy()


def plot_outputs_vs_prediction(
    t_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_path: str | Path | None = None,
    show_plot: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes = np.atleast_1d(axes)

    for i, name in enumerate(target_names):
        axes[i].plot(t_test[:, 0], y_test[:, i], label=f"true {name}", linewidth=2)
        axes[i].plot(t_test[:, 0], y_pred[:, i], "--", label=f"mlp {name}", linewidth=2)
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("time t [s]")
    fig.suptitle("OP data vs MLP prediction")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_op_surrogate_pipeline(
    op_id: int = 1,
    dataset_root: str | Path = r"c:\Users\M0245635\Downloads",
    train_ratio: float = 0.8,
    epochs: int = 500,
    lr: float = 1e-3,
    use_modulus: bool = True,
    show_plot: bool = True,
    save_plot_path: str | Path | None = None,
    save_debug_path: str | Path | None = None,
) -> Dict[str, object]:
    data = load_op_surrogate_data(op_id=op_id, dataset_root=dataset_root)
    split = time_series_split(data.x, data.y, train_ratio=train_ratio)
    std = standardize_train_test(
        split["x_train"],
        split["x_test"],
        split["y_train"],
        split["y_test"],
    )

    train_out = train_surrogate(
        x_train_n=std["x_train_n"],
        y_train_n=std["y_train_n"],
        epochs=epochs,
        lr=lr,
        use_modulus=use_modulus,
    )

    y_pred_test_n = predict_surrogate(
        model=train_out["model"],
        backend=train_out["backend"],
        x_n=std["x_test_n"],
        device=str(train_out["device"]),
    )
    y_pred_test = y_pred_test_n * std["y_std"] + std["y_mu"]

    rmse = np.sqrt(np.mean((y_pred_test - split["y_test"]) ** 2, axis=0))
    print(f"Backend: {train_out['backend']}")
    print(f"Test RMSE {data.target_names[0]}: {rmse[0]:.6f}")
    print(f"Test RMSE {data.target_names[1]}: {rmse[1]:.6f}")

    plot_outputs_vs_prediction(
        t_test=split["x_test"][:, :1],
        y_test=split["y_test"],
        y_pred=y_pred_test,
        target_names=data.target_names,
        save_path=save_plot_path,
        show_plot=show_plot,
    )

    if save_debug_path is not None:
        save_debug_path = Path(save_debug_path)
        save_debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_debug_path,
            t_train=split["x_train"][:, :1],
            y_train=split["y_train"],
            t_test=split["x_test"][:, :1],
            y_test=split["y_test"],
            y_pred_test=y_pred_test,
            target_names=np.array(data.target_names),
            train_ratio=np.array([train_ratio], dtype=np.float32),
        )
        print(f"Saved debug arrays to: {save_debug_path}")

    return {
        "data": data,
        "split": split,
        "std": std,
        "train": train_out,
        "y_pred_test": y_pred_test,
        "rmse": rmse,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/test MLP surrogate for bc_SOC and bc_Q")
    parser.add_argument("--op-id", type=int, default=1)
    parser.add_argument("--dataset-root", type=str, default=r"c:\Users\M0245635\Downloads")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-modulus", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-path", type=str, default="outputs/op_surrogate_compare.png")
    parser.add_argument("--dump-path", type=str, default="outputs/op_surrogate_debug.npz")
    args = parser.parse_args()

    run_op_surrogate_pipeline(
        op_id=args.op_id,
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        lr=args.lr,
        use_modulus=not args.no_modulus,
        show_plot=not args.no_show,
        save_plot_path=args.plot_path,
        save_debug_path=args.dump_path,
    )
