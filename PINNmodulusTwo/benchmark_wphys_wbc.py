#!/usr/bin/env python3
"""Benchmark: 2D sweep of physics loss weight (w_phys) and BC loss weight (w_bc).

Sweeps both w_phys and w_bc in a grid and scores every combination by the
free-running autoregressive rollout MAE on held-out test OP.

Everything except w_phys and w_bc is fixed so the comparison is apples-to-apples.

Fixed hyperparameters (override on CLI if desired):
- architecture: width=128, depth=4, per-layer learnable swish, weight-norm
- recurrence: k_max=2, history_mode=hybrid, rate_lags=[5.0, 20.0]
- optimization: Adam, lr=2e-3, epochs=60, seed=0, device=auto (CUDA when available)
- data: train=OP01-OP06, test=OP07, subsample=2 (CFL-stable Δt=0.2s)
- loss weights: w_data=1.0 (fixed), w_phys and w_bc swept

Default sweep grid:
- w_phys: [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0] (10 points)
- w_bc: [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1.0] (10 points)
- Total: 100 combinations
- Quasi-log spacing samples densely in promising region (0.01-0.3)

Run (CPU, in WSL):
    cd /mnt/c/Users/M0245635/batterysurrogatemodell
    source modulus_env/bin/activate
    python3 PINNmodulusTwo/benchmark_wphys_wbc.py

⚠️  WARNING: Expected runtime with default settings (OP01-OP06 train, 60 epochs, CPU):
    ~17 minutes per sweep point × 100 points = ~28 HOURS total!
    
For faster testing:
    --epochs 30 → ~14 hours
    --w-phys 0.0 0.01 0.05 0.1 0.3 --w-bc 0.0 0.01 0.05 0.1 0.3 → 25 points, ~7 hours

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_wphys_wbc.csv - one row per (w_phys, w_bc) with rollout MAEs
    benchmark_wphys_wbc_heatmap.png - 2D heatmap of held-out MAE
    benchmark_wphys_wbc_best.txt - best combination + summary table
    checkpoints_wphys_wbc/*.pt - per-sweep model checkpoints
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from data import build_op
from device_utils import resolve_device
from model import rollout
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Default 2D sweep grid (5×5 = 25 points, ~7 hours on CPU)
# Quasi-logarithmic spacing with dense sampling in promising region (0.01-0.3)
# Based on existing benchmarks: best around w_phys~0.1-0.2, w_bc~0.1-0.7
DEFAULT_W_PHYS = [0.0, 0.01, 0.05, 0.1, 0.3]
DEFAULT_W_BC = [0.0, 0.01, 0.05, 0.1, 0.3]

# Extended 10×10 grid (100 points, ~28 hours) - use with --extended-grid
EXTENDED_W_PHYS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
EXTENDED_W_BC = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1.0]


def _w_tag(w_phys: float, w_bc: float) -> str:
    """Filesystem-safe tag for (w_phys, w_bc) pair."""
    p = f"{float(w_phys):.6g}".replace("-", "m").replace(".", "p")
    b = f"{float(w_bc):.6g}".replace("-", "m").replace(".", "p")
    return f"p{p}_b{b}"


def _load_yaml_defaults() -> dict:
    cfg_path = THIS_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    d = _load_yaml_defaults()
    p = argparse.ArgumentParser(description="2D physics+BC loss-weight benchmark")
    # Data
    p.add_argument("--ops", nargs="+", default=["OP01", "OP02", "OP03", "OP04", "OP05", "OP06"])
    p.add_argument("--test-op", default="OP07")
    p.add_argument("--subsample", type=int, default=2, help="CFL-stable default: 2 -> Δt=0.2s")
    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N (auto = cuda when available)")
    # Architecture (FIXED for fair comparison)
    p.add_argument("--width", type=int, default=128, help="MLP width (FIXED)")
    p.add_argument("--depth", type=int, default=4, help="MLP depth (FIXED)")
    p.add_argument("--k-max", type=int, default=2, help="history lags (FIXED)")
    p.add_argument("--history-mode", choices=["raw", "hybrid"], default="hybrid",
                   help="history mode (FIXED)")
    p.add_argument("--rate-lags", nargs="+", type=float, default=[5.0, 20.0],
                   help="hybrid rate segments in seconds (FIXED initial values, learned)")
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"], default="bdf2",
                   help="time derivative method (FIXED)")
    p.add_argument("--use-static", action="store_true", default=True,
                   help="use static features (FIXED)")
    p.add_argument("--use-forcing", action="store_true", default=True,
                   help="use forcing features (FIXED)")
    # Loss weights (SWEPT)
    p.add_argument("--w-phys", type=float, nargs="+", default=None,
                   help="physics weights to sweep (default: 5-point grid)")
    p.add_argument("--w-bc", type=float, nargs="+", default=None,
                   help="BC weights to sweep (default: 5-point grid)")
    p.add_argument("--extended-grid", action="store_true",
                   help="use extended 10×10 grid (~28h) instead of default 5×5 (~7h)")
    # Batching
    p.add_argument("--batch-data", type=int, default=2048)
    p.add_argument("--batch-phys", type=int, default=256)
    p.add_argument("--batch-bc", type=int, default=128)
    p.add_argument("--phys-norm", type=float, default=0.0,
                   help="L_phys divisor: 0=adaptive EMA, >0=fixed divisor")
    # Checkpoints
    p.add_argument("--save-models", dest="save_models", action="store_true", default=True)
    p.add_argument("--no-save-models", dest="save_models", action="store_false")
    p.add_argument("--save-best-only", action="store_true",
                   help="save only the best held-out model instead of all sweep points")
    p.add_argument("--model-dir", default=str(ART_DIR / "checkpoints_wphys_wbc"))
    
    args = p.parse_args()
    
    # Apply grid defaults
    if args.extended_grid:
        if args.w_phys is None:
            args.w_phys = EXTENDED_W_PHYS
        if args.w_bc is None:
            args.w_bc = EXTENDED_W_BC
    else:
        if args.w_phys is None:
            args.w_phys = DEFAULT_W_PHYS
        if args.w_bc is None:
            args.w_bc = DEFAULT_W_BC
    
    return args


def _make_args(cli: argparse.Namespace, w_phys: float, w_bc: float) -> Namespace:
    """Build the Namespace that train.fit expects for one sweep point."""
    return Namespace(
        ops=cli.ops, subsample=cli.subsample, epochs=cli.epochs,
        k_max=cli.k_max, time_deriv=cli.time_deriv,
        history_mode=cli.history_mode, rate_lags=cli.rate_lags,
        width=cli.width, depth=cli.depth, lr=cli.lr,
        w_data=1.0, w_phys=float(w_phys), w_bc=float(w_bc),
        batch_data=cli.batch_data, batch_phys=cli.batch_phys,
        batch_bc=cli.batch_bc, delta_init_steps=1.0,
        weight_decay=cli.weight_decay, grad_clip=cli.grad_clip,
        early_stopping_patience=cli.early_stopping_patience,
        phys_norm=cli.phys_norm,
        use_static=cli.use_static, use_forcing=cli.use_forcing,
        seed=cli.seed, device=cli.device,
        test_op=cli.test_op,
    )


@torch.no_grad()
def _rollout_phys(model, op, bundle, device) -> np.ndarray:
    """Free-running rollout for one OPData -> physical temperature (n_t, P)."""
    xn = torch.as_tensor(op.xn, dtype=torch.float32, device=device)
    static = torch.as_tensor(op.static_feat, dtype=torch.float32, device=device)
    forcing = torch.as_tensor(op.forcing_feat, dtype=torch.float32, device=device)
    cfg = torch.as_tensor(op.config_feat, dtype=torch.float32, device=device)
    tn = torch.as_tensor(op.tn, dtype=torch.float32, device=device)
    Tn_ic = torch.as_tensor(op.Tn_ic, dtype=torch.float32, device=device)
    static = static[:, :model.n_static]
    forcing = forcing[:, :model.n_forcing]
    buf = rollout(model, xn, static, cfg, forcing, Tn_ic, tn, op.dtn)
    return buf.cpu().numpy() * bundle.T_sigma + bundle.T_mu


def _mae(pred, true, lo, hi) -> float:
    return float(np.abs(pred[lo:hi] - true[lo:hi]).mean())


def _timepoint_maes(pred: np.ndarray, true: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
    """Mean absolute error at each selected time index, averaged over space."""
    return np.array([float(np.abs(pred[i] - true[i]).mean()) for i in time_idx], dtype=float)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli = parse_args()
    device = resolve_device(cli.device)
    cli.device = str(device)  # hand the resolved device down to fit()
    dt_s = 0.1 * cli.subsample
    model_dir = Path(cli.model_dir)
    if cli.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "2D Physics+BC loss-weight benchmark (free-running rollout, NO teacher forcing)",
        f"train = {'+'.join(cli.ops)}   held-out test = {cli.test_op}",
        "FIXED ARCHITECTURE (for fair comparison):",
        f"  width={cli.width}  depth={cli.depth}  k_max={cli.k_max}  "
        f"history_mode={cli.history_mode}  rate_lags_init={cli.rate_lags}s",
        f"  time_deriv={cli.time_deriv}  use_static={cli.use_static}  use_forcing={cli.use_forcing}",
        "TRAINING SETTINGS:",
        f"  lr={cli.lr}  epochs={cli.epochs}  dt={dt_s:.1f}s  seed={cli.seed}  grad_clip={cli.grad_clip}",
        "LOSS WEIGHTS (SWEPT):",
        f"  w_data=1.0 (fixed)   phys_norm={cli.phys_norm} (adaptive EMA)",
        f"  w_phys sweep = {cli.w_phys}",
        f"  w_bc sweep = {cli.w_bc}",
        f"Grid size: {len(cli.w_phys)} × {len(cli.w_bc)} = {len(cli.w_phys)*len(cli.w_bc)} points",
        "",
    ]
    print("\n".join(header), flush=True)

    results = []
    histories = []  # Store epoch histories for convergence plotting
    total_points = len(cli.w_phys) * len(cli.w_bc)
    start_time_total = time.time()

    for idx, (w_phys, w_bc) in enumerate(
        [(p, b) for p in cli.w_phys for b in cli.w_bc], start=1
    ):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_points}] Training w_phys={w_phys}, w_bc={w_bc}")
        print(f"{'='*60}")
        start_time = time.time()

        args = _make_args(cli, w_phys, w_bc)
        model, bundle, _packed, _dtn, hist = fit(args)
        model.eval()
        train_time = time.time() - start_time

        intime_maes = [
            _mae(_rollout_phys(model, op, bundle, device), op.T_lab, op.split_t, op.n_t)
            for op in bundle.ops
        ]
        held = build_op(cli.test_op, bundle, subsample_time=cli.subsample)
        held_pred = _rollout_phys(model, held, bundle, device)
        held_mae = _mae(held_pred, held.T_lab, 0, held.n_t)
        
        # Sample 10 time points for boxplot
        test_time_idx = np.linspace(0, held.n_t - 1, num=10, dtype=int)
        test_time_maes = _timepoint_maes(held_pred, held.T_lab, test_time_idx)

        L_data_raw = float(hist["L_data"][-1])
        L_phys_raw = float(hist["L_phys"][-1])
        L_bc_raw = float(hist["L_bc"][-1])
        intime = float(np.mean(intime_maes))

        delta_s = float(model.delta.detach()) * bundle.T_span_ref
        src_gain = float(model.src_gain.detach())
        diff_gain = float(model.diff_gain.detach())
        n_params = sum(p.numel() for p in model.parameters())
        learned_rate_lags_s = (
            np.array(model.rate_lags.detach().cpu().numpy()) * bundle.T_span_ref
        ).astype(float).tolist()

        checkpoint = ""
        if cli.save_models and not cli.save_best_only:
            checkpoint_path = model_dir / f"model_{_w_tag(w_phys, w_bc)}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "n_config": bundle.n_config,
                        "n_static": model.n_static,
                        "n_forcing": model.n_forcing,
                        "k_max": args.k_max,
                        "history_mode": args.history_mode,
                        "rate_lags": [float(v) / bundle.T_span_ref for v in args.rate_lags],
                        "layer_size": args.width,
                        "num_layers": args.depth,
                        "delta_seconds": 1.0,
                        "dtn": float(_dtn),
                        "use_autograd_time": (args.time_deriv == "autograd"),
                    },
                    "bundle_stats": {
                        "T_mu": float(bundle.T_mu),
                        "T_sigma": float(bundle.T_sigma),
                        "T_span_ref": float(bundle.T_span_ref),
                    },
                    "benchmark_context": {
                        "w_phys": float(w_phys),
                        "w_bc": float(w_bc),
                        "ops": list(args.ops),
                        "test_op": args.test_op,
                        "epochs": int(args.epochs),
                        "subsample": int(args.subsample),
                        "seed": int(args.seed),
                    },
                },
                checkpoint_path,
            )
            checkpoint = str(checkpoint_path)

        results.append({
            "w_phys": float(w_phys),
            "w_bc": float(w_bc),
            "intime_mae": intime,
            "held_mae": held_mae,
            "L_data": L_data_raw,
            "L_phys": L_phys_raw,
            "L_bc": L_bc_raw,
            "delta_s": delta_s,
            "src_gain": src_gain,
            "diff_gain": diff_gain,
            "n_params": n_params,
            "rate_lags_s": learned_rate_lags_s,
            "train_time": train_time,
            "checkpoint": checkpoint,
            "test_time_maes": test_time_maes,
        })
        histories.append({"w_phys": w_phys, "w_bc": w_bc, "hist": hist})

        elapsed = time.time() - start_time_total
        avg_time = elapsed / idx
        eta = avg_time * (total_points - idx)
        print(f"  MAE(in-time)={intime:.3f}°C  MAE(held {cli.test_op})={held_mae:.3f}°C")
        print(f"  L_data={L_data_raw:.4g}  L_phys={L_phys_raw:.4g}  L_bc={L_bc_raw:.4g}")
        print(f"  Train time: {train_time/60:.1f} min | ETA: {eta/60:.1f} min", flush=True)

    total_time = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n")

    # ---- CSV ----------------------------------------------------------------
    csv_lines = [
        "w_phys,w_bc,L_data,L_phys,L_bc,MAE_in_C,MAE_test_C,"
        "delta_s,src_gain,diff_gain,rate_lags_s,train_time_min,checkpoint"
    ]
    for r in results:
        lags_str = ";".join(f"{v:.6g}" for v in r["rate_lags_s"])
        csv_lines.append(
            f"{r['w_phys']},{r['w_bc']},{r['L_data']:.6f},{r['L_phys']:.6f},{r['L_bc']:.6f},"
            f"{r['intime_mae']:.4f},{r['held_mae']:.4f},"
            f"{r['delta_s']:.6f},{r['src_gain']:.6f},{r['diff_gain']:.6f},"
            f"\"{lags_str}\",{r['train_time']/60:.2f},{r['checkpoint']}"
        )
    (ART_DIR / "benchmark_wphys_wbc.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- best pick + summary ------------------------------------------------
    best = min(results, key=lambda r: r["held_mae"])
    if cli.save_models and cli.save_best_only:
        best_ckpt_path = model_dir / f"model_best_{_w_tag(best['w_phys'], best['w_bc'])}.pt"
        args_best = _make_args(cli, best["w_phys"], best["w_bc"])
        model_best, bundle_best, _, dtn_best, _ = fit(args_best)
        torch.save(
            {
                "model_state_dict": model_best.state_dict(),
                "model_config": {
                    "n_config": bundle_best.n_config,
                    "n_static": model_best.n_static,
                    "n_forcing": model_best.n_forcing,
                    "k_max": args_best.k_max,
                    "history_mode": args_best.history_mode,
                    "rate_lags": [float(v) / bundle_best.T_span_ref for v in args_best.rate_lags],
                    "layer_size": args_best.width,
                    "num_layers": args_best.depth,
                    "delta_seconds": 1.0,
                    "dtn": float(dtn_best),
                    "use_autograd_time": (args_best.time_deriv == "autograd"),
                },
                "bundle_stats": {
                    "T_mu": float(bundle_best.T_mu),
                    "T_sigma": float(bundle_best.T_sigma),
                    "T_span_ref": float(bundle_best.T_span_ref),
                },
                "benchmark_context": {
                    "w_phys": float(best["w_phys"]),
                    "w_bc": float(best["w_bc"]),
                    "ops": list(args_best.ops),
                    "test_op": args_best.test_op,
                    "epochs": int(args_best.epochs),
                    "subsample": int(args_best.subsample),
                    "seed": int(args_best.seed),
                },
            },
            best_ckpt_path,
        )
        best["checkpoint"] = str(best_ckpt_path)

    th = f"{'w_phys':>7} {'w_bc':>7} | {'L_data':>10} {'L_phys':>10} {'L_bc':>10} | {'MAE_in':>7} {'MAE_test':>8}"
    summary = header + [th, "-" * len(th)]
    for r in results:
        summary.append(
            f"{r['w_phys']:>7.3f} {r['w_bc']:>7.3f} | {r['L_data']:>10.4g} {r['L_phys']:>10.4g} "
            f"{r['L_bc']:>10.4g} | {r['intime_mae']:>7.3f} {r['held_mae']:>8.3f}"
        )
    summary += [
        "",
        "MAE = mean |true - predicted| (°C) from free-running rollout.",
        f"BEST (by held-out MAE): w_phys={best['w_phys']}, w_bc={best['w_bc']}  "
        f"-> held-out {best['held_mae']:.3f}°C, in-time {best['intime_mae']:.3f}°C",
        f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} min)",
    ]
    if cli.save_models:
        summary.append(f"Checkpoints dir: {model_dir}")
    (ART_DIR / "benchmark_wphys_wbc_best.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[len(header):]), flush=True)

    # ---- 2D heatmap ---------------------------------------------------------
    w_phys_vals = sorted(set(r["w_phys"] for r in results))
    w_bc_vals = sorted(set(r["w_bc"] for r in results))
    heatmap = np.full((len(w_bc_vals), len(w_phys_vals)), np.nan)
    for r in results:
        i = w_bc_vals.index(r["w_bc"])
        j = w_phys_vals.index(r["w_phys"])
        heatmap[i, j] = r["held_mae"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(w_phys_vals)))
    ax.set_xticklabels([f"{v:.3g}" for v in w_phys_vals])
    ax.set_yticks(range(len(w_bc_vals)))
    ax.set_yticklabels([f"{v:.3g}" for v in w_bc_vals])
    ax.set_xlabel("w_phys (physics loss weight)")
    ax.set_ylabel("w_bc (boundary condition loss weight)")
    ax.set_title(f"Held-out {cli.test_op} MAE (°C) — lower is better")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE [°C]")

    # Mark best point
    best_i = w_bc_vals.index(best["w_bc"])
    best_j = w_phys_vals.index(best["w_phys"])
    ax.scatter([best_j], [best_i], s=200, facecolors="none", edgecolors="red", linewidths=2.5, label="Best")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ART_DIR / "benchmark_wphys_wbc_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved heatmap: {ART_DIR / 'benchmark_wphys_wbc_heatmap.png'}", flush=True)

    # ---- Convergence plots (show subset to avoid cluttering) ----------------
    # Plot only corner points + best point for clarity on large grids
    if len(histories) <= 9:
        plot_histories = histories  # Show all if small grid
    else:
        # Show corners + best
        corners = [
            histories[0],  # (min w_phys, min w_bc)
            histories[len(cli.w_bc) - 1],  # (min w_phys, max w_bc)
            histories[-len(cli.w_bc)],  # (max w_phys, min w_bc)
            histories[-1],  # (max w_phys, max w_bc)
        ]
        best_hist = next(h for h in histories if h["w_phys"] == best["w_phys"] and h["w_bc"] == best["w_bc"])
        plot_histories = corners + ([best_hist] if best_hist not in corners else [])
    
    fig_conv, axes_conv = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for item in plot_histories:
        wp = item["w_phys"]
        wb = item["w_bc"]
        h = item["hist"]
        epochs = h["epoch"]
        is_best = (wp == best["w_phys"] and wb == best["w_bc"])
        label = f"p={wp:.3g}, b={wb:.3g}" + (" ★BEST" if is_best else "")
        lw = 2.5 if is_best else 1.5
        
        # Top: data loss
        axes_conv[0].plot(epochs, h["L_data"], marker="o", markersize=3, label=label, linewidth=lw)
        axes_conv[0].set_ylabel("L_data (MSE)", fontsize=11)
        axes_conv[0].set_yscale("log")
        axes_conv[0].grid(True, alpha=0.3)
        axes_conv[0].legend(fontsize=8, ncol=2)
        axes_conv[0].set_title(f"Convergence (subset): {'+'.join(cli.ops)} train, {cli.test_op} test", fontsize=12)
        
        # Middle: physics loss
        if "L_phys" in h and len(h["L_phys"]) > 0:
            axes_conv[1].plot(epochs, h["L_phys"], marker="s", markersize=3, label=label, linewidth=lw, alpha=0.8)
        axes_conv[1].set_ylabel("L_phys (unweighted)", fontsize=11)
        axes_conv[1].set_yscale("log")
        axes_conv[1].grid(True, alpha=0.3)
        axes_conv[1].legend(fontsize=8, ncol=2)
        
        # Bottom: BC loss
        if "L_bc" in h and len(h["L_bc"]) > 0:
            axes_conv[2].plot(epochs, h["L_bc"], marker="^", markersize=3, label=label, linewidth=lw, alpha=0.8)
    
    axes_conv[2].set_xlabel("Epoch", fontsize=11)
    axes_conv[2].set_ylabel("L_bc (unweighted)", fontsize=11)
    axes_conv[2].set_yscale("log")
    axes_conv[2].grid(True, alpha=0.3)
    axes_conv[2].legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    convergence_plot = ART_DIR / "benchmark_wphys_wbc_convergence.png"
    plt.savefig(convergence_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot: {convergence_plot}", flush=True)

    # ---- test-op boxplot (10 time points) ----------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    
    # Create labels for each (w_phys, w_bc) combination
    box_labels = [f"p{r['w_phys']:.3g}\nb{r['w_bc']:.3g}" for r in results]
    box_data = [r["test_time_maes"] for r in results]
    
    bp = ax2.boxplot(box_data, labels=box_labels, showmeans=True, whis=(0, 100), patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e8ff")
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#b00020")
        median.set_linewidth(2.0)
    for flier in bp.get("fliers", []):
        flier.set(markerfacecolor="#444444", markeredgecolor="#444444", markersize=4)

    # Overlay the 10 time-point MAEs
    for i, vals in enumerate(box_data, start=1):
        jitter = np.linspace(-0.15, 0.15, num=len(vals))
        ax2.scatter(np.full(len(vals), i) + jitter, vals, s=20, color="#1f77b4", alpha=0.6, zorder=3)

    # Mark best point
    best_idx = next(i for i, r in enumerate(results) if r["w_phys"] == best["w_phys"] and r["w_bc"] == best["w_bc"])
    ax2.scatter([best_idx + 1], [best["held_mae"]], s=300, marker="*", color="red", 
                edgecolors="darkred", linewidths=1.5, zorder=10, label="Best (overall MAE)")

    ax2.set_xlabel("(w_phys, w_bc) combination", fontsize=11)
    ax2.set_ylabel("Held-out test-op MAE across 10 time points [°C]", fontsize=11)
    ax2.set_title(f"Test-op MAE distribution from 10 uniformly spaced time points ({cli.test_op})", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right")
    
    # Rotate labels if too many
    if len(results) > 16:
        ax2.tick_params(axis="x", labelrotation=45, labelsize=8)
    
    fig2.tight_layout()
    fig2.savefig(ART_DIR / "benchmark_wphys_wbc_boxplot.png", dpi=150)
    plt.close(fig2)

    print(f"\n  Saved: {ART_DIR/'benchmark_wphys_wbc.csv'}")
    print(f"         {ART_DIR/'benchmark_wphys_wbc_heatmap.png'}")
    print(f"         {ART_DIR/'benchmark_wphys_wbc_boxplot.png'}")
    print(f"         {ART_DIR/'benchmark_wphys_wbc_best.txt'}", flush=True)


if __name__ == "__main__":
    main()
