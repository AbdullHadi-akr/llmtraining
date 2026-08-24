#!/usr/bin/env python3
"""Benchmark: 2D sweep of physics loss weight (w_phys) and BC loss weight (w_bc).

Sweeps both w_phys and w_bc in a grid and scores every combination by the
free-running autoregressive rollout MAE on a held-out VALIDATION OP; a second,
never-selected-on TEST OP is reported alongside it.

Everything except w_phys and w_bc is fixed so the comparison is apples-to-apples.

Fixed hyperparameters (override on CLI if desired):
- architecture: width=128, depth=4, per-layer learnable swish, weight-norm
- recurrence: k_max=2, history_mode=hybrid, rate_lags=[5.0, 20.0]
- optimization: Adam, lr=2e-3, epochs=60, device=auto (CUDA when available)
- seeds: one training run per seed per grid point (--seeds, default [0]).
  Each point is scored by the MEAN over its seeds and carries the standard
  deviation, so a difference between points can be read against the spread
  the initialisation alone produces. Runtime scales linearly with the seed
  count: 100 points x 3 seeds = 300 trainings.
- data: train=OP01-OP05, val=OP06 (selection), test=OP07 (report only),
  subsample=2 (CFL-stable Δt=0.2s)
- loss weights: w_data=1.0 (fixed), w_phys and w_bc swept

Default sweep grid:
- w_phys: [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0] (10 points)
- w_bc: [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1.0] (10 points)
- Total: 100 combinations
- Quasi-log spacing samples densely in promising region (0.01-0.3)

Run:
    source .venv/bin/activate
    python3 PINNmodulusTwo/benchmark_wphys_wbc.py --extended-grid --device cuda

RUNTIME -- read this before starting a long run.

At subsample=2 the rollout is ~7000 sequential steps per OP per epoch, and that
dominates everything. One epoch over 5 training OPs costs roughly 1.5-2.5 min on
an RTX 5090 Laptop, so ONE grid point at 60 epochs is 1.5-2.5 HOURS:
    5x5 grid,   1 seed  ->  25 trainings  ~1.5-2 days
    10x10 grid, 1 seed  -> 100 trainings  ~6-8 days   (--extended-grid)
Multiply by the seed count on top of that.

Do not take those numbers on faith: the log prints the measured seconds per
epoch from the first epoch on ("[12.4s/epoch, this run ~124 min left]"). Read it
once and compute the real total before committing days of GPU time.

To bring it down, in order of effect: fewer --epochs, a coarser grid, or a
larger --subsample (which shortens the rollout quadratically in wall time but
changes the time resolution).

The full test sequence, in the order that makes sense, is in
README_GPU_SERVER.md section 7 -- run the smoke test and the seed-spread check
before committing hours to this one.

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_wphys_wbc.csv - one row per (w_phys, w_bc): mean rollout MAEs
        over the seeds, their standard deviation, and how many seeds survived
    benchmark_wphys_wbc_heatmap.png - 2D heatmap of validation MAE
    benchmark_wphys_wbc_best.txt - best combination + summary table
    checkpoints_wphys_wbc/*.pt - per-sweep model checkpoints
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from bench_common import (
    EMPTY_HIST, aggregate_seeds, failed_result, make_train_args, noise_verdict,
    print_eta, train_one_seed,
)
from data import require_ops
from device_utils import resolve_device
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
    p.add_argument("--ops", nargs="+", default=["OP01", "OP02", "OP03", "OP04", "OP05"])
    p.add_argument("--val-op", default="OP06",
                   help="OP used to SELECT the best (w_phys, w_bc)")
    p.add_argument("--test-op", default="OP07",
                   help="OP used only to REPORT the chosen point; never selected on")
    p.add_argument("--subsample", type=int, default=2, help="CFL-stable default: 2 -> Δt=0.2s")
    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gain-lr-mult", type=float, default=25.0,
                   help="LR multiplier for src_gain/diff_gain (FIXED)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one training run per seed per grid point; the point is "
                        "scored by the MEAN over seeds. Runtime scales with the "
                        "number of seeds. Use >=3 to tell a real effect from "
                        "init noise.")
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
                   help="save only the best (by validation MAE) model instead of all points")
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


def _make_args(cli: argparse.Namespace, w_phys: float, w_bc: float,
               seed: int):
    """Namespace for one grid point; the swept weights go in as overrides."""
    return make_train_args(
        cli,
        {"w_phys": float(w_phys), "w_bc": float(w_bc),
         "rate_lags": list(cli.rate_lags)},
        seed,
    )


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli = parse_args()
    require_ops(*cli.ops, cli.val_op, cli.test_op)
    device = resolve_device(cli.device)
    cli.device = str(device)  # hand the resolved device down to fit()
    dt_s = 0.1 * cli.subsample
    model_dir = Path(cli.model_dir)
    if cli.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "2D Physics+BC loss-weight benchmark (free-running rollout, NO teacher forcing)",
        f"train = {'+'.join(cli.ops)}   val (selection) = {cli.val_op}   "
        f"test (report only) = {cli.test_op}",
        "FIXED ARCHITECTURE (for fair comparison):",
        f"  width={cli.width}  depth={cli.depth}  k_max={cli.k_max}  "
        f"history_mode={cli.history_mode}  rate_lags_init={cli.rate_lags}s",
        f"  time_deriv={cli.time_deriv}  use_static={cli.use_static}  use_forcing={cli.use_forcing}",
        "TRAINING SETTINGS:",
        f"  lr={cli.lr}  epochs={cli.epochs}  dt={dt_s:.1f}s  "
        f"seeds={cli.seeds}  grad_clip={cli.grad_clip}",
        f"  runs = {len(cli.w_phys)*len(cli.w_bc)} grid points x {len(cli.seeds)} "
        f"seed(s) = {len(cli.w_phys)*len(cli.w_bc)*len(cli.seeds)} trainings",
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
        print(f"[{idx}/{total_points}] Training w_phys={w_phys}, w_bc={w_bc}"
              f"  ({len(cli.seeds)} seed{'s' if len(cli.seeds) > 1 else ''}: "
              f"{', '.join(str(s) for s in cli.seeds)})")
        print(f"{'='*60}")
        start_time = time.time()

        per_seed, first_hist = [], None
        for seed in cli.seeds:
            # Only the first seed writes a checkpoint: the grid point is scored by
            # the mean over seeds, so no single seed's weights are "the" result.
            save_ckpt = (cli.save_models and not cli.save_best_only
                         and seed == cli.seeds[0])
            one, hist = train_one_seed(
                cli,
                {"w_phys": float(w_phys), "w_bc": float(w_bc),
                 "rate_lags": list(cli.rate_lags)},
                seed, device, fit,
                checkpoint_path=(model_dir / f"model_{_w_tag(w_phys, w_bc)}.pt"
                                 if save_ckpt else None),
                context={"w_phys": float(w_phys), "w_bc": float(w_bc)},
            )
            if first_hist is None:
                first_hist = hist
            if one is not None:
                per_seed.append(one)

        train_time = time.time() - start_time

        if not per_seed:
            print(f"  [SKIP] w_phys={w_phys}, w_bc={w_bc}: every seed diverged or "
                  f"crashed - recorded as NaN, sweep continues", flush=True)
            results.append(failed_result(
                {"w_phys": float(w_phys), "w_bc": float(w_bc)},
                train_time, len(cli.seeds)))
            histories.append({"w_phys": w_phys, "w_bc": w_bc,
                              "hist": first_hist or EMPTY_HIST})
            print_eta(idx, total_points, start_time_total, train_time)
            continue

        row = aggregate_seeds({"w_phys": float(w_phys), "w_bc": float(w_bc)},
                              per_seed, len(cli.seeds), train_time)
        results.append(row)
        histories.append({"w_phys": w_phys, "w_bc": w_bc, "hist": first_hist})

        n_ok, n_all = row["n_seeds_ok"], row["n_seeds"]
        spread = ""
        if n_ok > 1:
            spread = (f"  (+/-{row['val_mae_std']:.3f} val, "
                      f"+/-{row['test_mae_std']:.3f} test over {n_ok} seeds)")
        elif n_all > 1:
            spread = f"  ({n_ok}/{n_all} seeds usable)"
        print(f"  MAE(in-time)={row['intime_mae']:.3f}°C  "
              f"MAE(val {cli.val_op})={row['val_mae']:.3f}°C  "
              f"MAE(test {cli.test_op})={row['test_mae']:.3f}°C{spread}")
        print(f"  L_data={row['L_data']:.4g}  L_phys={row['L_phys']:.4g}  "
              f"L_bc={row['L_bc']:.4g}")
        if n_ok < n_all:
            print(f"  note: {n_all - n_ok}/{n_all} seeds diverged and were left out "
                  f"of the mean", flush=True)
        print_eta(idx, total_points, start_time_total, train_time)

    total_time = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n")

    # ---- CSV ----------------------------------------------------------------
    csv_lines = [
        "w_phys,w_bc,L_data,L_phys,L_bc,MAE_in_C,MAE_val_C,MAE_val_std_C,"
        "MAE_test_C,MAE_test_std_C,n_seeds,n_seeds_ok,"
        "delta_s,src_gain,diff_gain,rate_lags_s,train_time_min,checkpoint"
    ]
    for r in results:
        lags_str = ";".join(f"{v:.6g}" for v in r["rate_lags_s"])
        csv_lines.append(
            f"{r['w_phys']},{r['w_bc']},{r['L_data']:.6f},{r['L_phys']:.6f},{r['L_bc']:.6f},"
            f"{r['intime_mae']:.4f},{r['val_mae']:.4f},{r['val_mae_std']:.4f},"
            f"{r['test_mae']:.4f},{r['test_mae_std']:.4f},"
            f"{r['n_seeds']},{r['n_seeds_ok']},"
            f"{r['delta_s']:.6f},{r['src_gain']:.6f},{r['diff_gain']:.6f},"
            f"\"{lags_str}\",{r['train_time']/60:.2f},{r['checkpoint']}"
        )
    (ART_DIR / "benchmark_wphys_wbc.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- best pick + summary ------------------------------------------------
    # Diverged points carry NaN and must not win the min() comparison.
    usable = [r for r in results if np.isfinite(r["val_mae"])]
    n_failed = len(results) - len(usable)
    if not usable:
        print(f"\nAll {len(results)} grid points diverged - no result to rank.", flush=True)
        print(f"Raw values are in {ART_DIR / 'benchmark_wphys_wbc.csv'}.", flush=True)
        print("Check the [DATA WARN]/[ABORT] lines above, then retry with a smaller "
              "--subsample or a stricter --grad-clip.", flush=True)
        return
    if n_failed:
        print(f"\n{n_failed}/{len(results)} grid points diverged and are recorded as NaN.",
              flush=True)
    best = min(usable, key=lambda r: r["val_mae"])
    if cli.save_models and cli.save_best_only:
        best_ckpt_path = model_dir / f"model_best_{_w_tag(best['w_phys'], best['w_bc'])}.pt"
        args_best = _make_args(cli, best["w_phys"], best["w_bc"], cli.seeds[0])
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
                    "t_span_ref": float(bundle_best.T_span_ref),
                    "rate_scale": float(bundle_best.dTdt_scale),
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
                    "val_op": cli.val_op,
                    "test_op": args_best.test_op,
                    "epochs": int(args_best.epochs),
                    "subsample": int(args_best.subsample),
                    "seed": int(args_best.seed),
                },
            },
            best_ckpt_path,
        )
        best["checkpoint"] = str(best_ckpt_path)

    th = (f"{'w_phys':>7} {'w_bc':>7} | {'L_data':>10} {'L_phys':>10} {'L_bc':>10} | "
          f"{'MAE_in':>7} {'MAE_val':>8} {'+/-':>6} {'MAE_test':>9} {'+/-':>6}")
    summary = header + [th, "-" * len(th)]
    for r in results:
        summary.append(
            f"{r['w_phys']:>7.3f} {r['w_bc']:>7.3f} | {r['L_data']:>10.4g} {r['L_phys']:>10.4g} "
            f"{r['L_bc']:>10.4g} | {r['intime_mae']:>7.3f} {r['val_mae']:>8.3f} "
            f"{r['val_mae_std']:>6.3f} {r['test_mae']:>9.3f} {r['test_mae_std']:>6.3f}"
        )
    summary += [
        "",
        "MAE = mean |true - predicted| (°C) from free-running rollout.",
        f"Selection ran on {cli.val_op} (MAE_val); {cli.test_op} (MAE_test) was never "
        f"used to choose anything.",
        f"BEST (by MAE_val): w_phys={best['w_phys']}, w_bc={best['w_bc']}",
        f"  -> val {best['val_mae']:.3f}°C, test {best['test_mae']:.3f}°C, "
        f"in-time {best['intime_mae']:.3f}°C",
        "  Report the test number. MAE_val is optimistic: it is the minimum over "
        f"{len(results)} grid points.",
        *noise_verdict(usable, best, len(cli.seeds),
                       lambda r: f"w_phys={r['w_phys']}, w_bc={r['w_bc']}"),
        f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} min)",
    ]
    if n_failed:
        summary.append(
            f"Diverged (recorded as NaN, excluded from the ranking): {n_failed}/{len(results)}"
        )
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
        heatmap[i, j] = r["val_mae"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(w_phys_vals)))
    ax.set_xticklabels([f"{v:.3g}" for v in w_phys_vals])
    ax.set_yticks(range(len(w_bc_vals)))
    ax.set_yticklabels([f"{v:.3g}" for v in w_bc_vals])
    ax.set_xlabel("w_phys (physics loss weight)")
    ax.set_ylabel("w_bc (boundary condition loss weight)")
    ax.set_title(f"Validation {cli.val_op} MAE (°C) — selection surface, lower is better")
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
        axes_conv[0].set_title(f"Convergence (subset): {'+'.join(cli.ops)} train, "
                               f"{cli.val_op} val", fontsize=12)
        
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
    box_labels = [f"p{r['w_phys']:.3g}\nb{r['w_bc']:.3g}" for r in usable]
    box_data = [r["test_time_maes"] for r in usable]
    
    # set_xticklabels instead of the boxplot(labels=...) kwarg, which was
    # removed in matplotlib 3.11 (renamed to tick_labels in 3.9).
    bp = ax2.boxplot(box_data, showmeans=True, whis=(0, 100), patch_artist=True)
    ax2.set_xticks(range(1, len(box_data) + 1))
    ax2.set_xticklabels(box_labels)
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
    best_idx = next(i for i, r in enumerate(usable) if r["w_phys"] == best["w_phys"] and r["w_bc"] == best["w_bc"])
    # The axis shows test-op MAE, so mark the test value of the point that was
    # selected on the validation OP -- not its validation MAE.
    ax2.scatter([best_idx + 1], [best["test_mae"]], s=300, marker="*", color="red",
                edgecolors="darkred", linewidths=1.5, zorder=10,
                label=f"Selected on {cli.val_op} (test MAE shown)")

    ax2.set_xlabel("(w_phys, w_bc) combination", fontsize=11)
    ax2.set_ylabel("Held-out test-op MAE across 10 time points [°C]", fontsize=11)
    ax2.set_title(f"Test-op MAE distribution from 10 uniformly spaced time points ({cli.test_op})", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right")
    
    # Rotate labels if too many
    if len(usable) > 16:
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
