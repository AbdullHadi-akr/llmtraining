
#!/usr/bin/env python3
"""Benchmark: how strongly should we weight the physics loss?

Sweeps the physics-loss weight ``w_phys`` (relative to a fixed data weight
``w_data = 1.0``) and scores every setting by the HONEST metric: free-running
autoregressive rollout MAE (no teacher forcing), both on the in-time TEST split
of the training OPs and on a fully held-out OP.

Everything except ``w_phys`` is fixed and listed below so the comparison is
apples-to-apples. The training itself is the free-running loop in ``train.fit``
(the data loss is taken on the model's OWN rollout -- there is no teacher
forcing anywhere in train or eval).

Chosen, fixed hyperparameters (override on the CLI if desired)
-------------------------------------------------------------
* architecture : width = 128, depth = 4 hidden FCLayers, per-layer learnable
                 swish, weight-norm on hidden layers.
* recurrence   : k_max = 2 history lags, fixed delta (init = 1 step),
                 learnable src/diff gains.
* optimisation : Adam, lr = 2e-3, epochs = 80, seed = 0, device = cpu.
* data         : train = OP01+OP02+OP03, held-out test = OP16,
                 subsample from config (default 10 -> dt ~ 1 s), train_frac = 0.8.
* loss weights : w_data = 1.0 (fixed), w_phys swept over
                 {0.0, 0.01, 0.03, 0.1, 0.3, 1.0}.

TODO: Consider sweeping these hyperparameters in future benchmarks:
  - subsample_time (Δgrid): {10, 20, 40} -> {1s, 2s, 4s} time resolution
    * Δgrid is FIXED by user, not learnable
    * Smaller = finer resolution but 4× slower training
  - rate_lags (LEARNABLE initial values):
      * [5, 20]   -> 5s recent + 20s trend (default)
      * [3, 15]   -> faster dynamics
      * [10, 30]  -> slower dynamics / longer memory
      * [5, 15, 30] -> 3 segments (k=4 with anchor)
    * Network learns optimal values via gradient descent
    * Log learned lags after training to see what it found
  - history_mode: raw vs hybrid
  - delta_init_steps: history spacing for raw mode

Run (CPU-first, in the repo's WSL env):
    cd /mnt/c/Users/M0245635/batterysurrogatemodell
    source modulus_env/bin/activate
    python3 PINNmodulusTwo/benchmark_wphys.py

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_wphys.csv   - one row per w_phys with rollout MAEs + learned params
    benchmark_wphys.png   - rollout MAE vs w_phys (in-time test + held-out)
    benchmark_wphys_test_boxplot.png - held-out test-op MAE boxplots over 10
                                       uniformly sampled time points
    benchmark_wphys_params.png - learned recurrence/physics params vs w_phys
    benchmark_wphys_losses.png - unweighted losses + weighted-loss balance trends
    benchmark_wphys.txt   - human-readable summary incl. the fixed settings
    checkpoints_wphys/*.pt - per-sweep model checkpoints for later reuse

This benchmark uses the same recurrence knobs as train.py: raw vs hybrid
history, rate_lags, and the configured time derivative method.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from data import build_op
from model import rollout
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# The physics-weight sweep (relative to w_data = 1.0). 0.0 = data-only baseline.
# Wider sweep around the balanced regime so the curve can show a clear optimum.
DEFAULT_W_PHYS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]


def _w_tag(w_phys: float) -> str:
    """Filesystem-safe tag for a physics weight value."""
    return f"{float(w_phys):.6g}".replace("-", "m").replace(".", "p")


def _load_yaml_defaults() -> dict:
    cfg_path = THIS_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    d = _load_yaml_defaults()
    p = argparse.ArgumentParser(description="Physics-loss-weight benchmark")
    p.add_argument("--ops", nargs="+", default=d.get("ops", ["OP01", "OP02", "OP03"]))
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 40))
    p.add_argument("--epochs", type=int, default=d.get("epochs", 80))
    p.add_argument("--test-op", default=d.get("test_op", "OP07"))
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "raw"))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 25.0]))
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", True))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", True))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 0.0))
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0))
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    p.add_argument("--phys-norm", type=float, default=0.0,
                   help="L_phys divisor: 0 = adaptive EMA auto-balance, >0 = "
                        "fixed divisor")
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", default=d.get("device", "cpu"))
    p.add_argument("--w-phys", type=float, nargs="+", default=DEFAULT_W_PHYS,
                   help="physics weights to sweep (w_data is fixed at 1.0)")
    p.add_argument("--save-models", dest="save_models", action="store_true", default=True,
                   help="save checkpoints for later reuse (default: on)")
    p.add_argument("--no-save-models", dest="save_models", action="store_false",
                   help="disable checkpoint saving")
    p.add_argument("--save-best-only", action="store_true",
                   help="save only the best held-out model instead of every sweep point")
    p.add_argument("--model-dir", default=str(ART_DIR / "checkpoints_wphys"),
                   help="checkpoint output directory")
    return p.parse_args()


def _make_args(cli: argparse.Namespace, w_phys: float) -> Namespace:
    """Build the Namespace that ``train.fit`` expects for one sweep point."""
    return Namespace(
        ops=cli.ops, subsample=cli.subsample, epochs=cli.epochs,
        k_max=cli.k_max, time_deriv=cli.time_deriv,
        history_mode=cli.history_mode, rate_lags=cli.rate_lags,
        width=cli.width, depth=cli.depth, lr=cli.lr,
        w_data=1.0, w_phys=float(w_phys), w_bc=cli.w_bc,
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
    static = static[:, : model.n_static]
    forcing = forcing[:, : model.n_forcing]
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
    device = torch.device(cli.device)
    dt_s = 0.1 * cli.subsample
    model_dir = Path(cli.model_dir)
    if cli.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "Physics-loss-weight benchmark (free-running rollout, NO teacher forcing)",
        f"train = {'+'.join(cli.ops)}   held-out test = {cli.test_op}",
        f"width={cli.width}  depth={cli.depth}  k_max={cli.k_max}  "
        f"history_mode={cli.history_mode}  rate_lags_s={cli.rate_lags}  "
        f"time_deriv={cli.time_deriv}  use_static={cli.use_static} "
        f"use_forcing={cli.use_forcing}",
        f"lr={cli.lr}  epochs={cli.epochs}  dt={dt_s:.0f}s  seed={cli.seed}",
        f"w_data=1.0 (fixed)   "
        f"phys_norm={cli.phys_norm} ({'raw' if cli.phys_norm > 0 else 'adaptive-EMA'})   "
        f"w_phys sweep = {cli.w_phys}",
        "",
    ]
    print("\n".join(header), flush=True)

    results = []
    histories = []  # Store epoch histories for convergence plotting
    for w_phys in cli.w_phys:
        print(f"=== training w_phys = {w_phys} ===", flush=True)
        args = _make_args(cli, w_phys)
        model, bundle, _packed, _dtn, hist = fit(args)
        model.eval()

        intime_maes = [
            _mae(_rollout_phys(model, op, bundle, device), op.T_lab, op.split_t, op.n_t)
            for op in bundle.ops
        ]
        held = build_op(cli.test_op, bundle, subsample_time=cli.subsample)
        held_pred = _rollout_phys(model, held, bundle, device)
        held_mae = _mae(held_pred, held.T_lab, 0, held.n_t)

        test_time_idx = np.linspace(0, held.n_t - 1, num=10, dtype=int)
        test_time_maes = _timepoint_maes(held_pred, held.T_lab, test_time_idx)

        # Raw (un-weighted) losses at the final epoch, plus the WEIGHTED terms so
        # w_phys*L_phys can be compared against w_data*L_data (same range = balanced).
        L_data_raw = float(hist["L_data"][-1])
        L_phys_raw = float(hist["L_phys"][-1])
        wL_data = 1.0 * L_data_raw
        wL_phys = float(w_phys) * L_phys_raw

        delta_s = float(model.delta.detach()) * bundle.T_span_ref
        gates = np.round(model.gates().detach().cpu().numpy(), 3).tolist()
        src_gain = float(model.src_gain.detach())
        diff_gain = float(model.diff_gain.detach())
        n_params = sum(p.numel() for p in model.parameters())
        learned_rate_lags_s = (
            np.array(model.rate_lags.detach().cpu().numpy()) * bundle.T_span_ref
        ).astype(float).tolist()
        intime = float(np.mean(intime_maes))

        checkpoint = ""
        if cli.save_models and not cli.save_best_only:
            checkpoint_path = model_dir / f"model_wphys_{_w_tag(w_phys)}.pt"
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
                        "ops": list(args.ops),
                        "test_op": args.test_op,
                        "epochs": int(args.epochs),
                        "subsample": int(args.subsample),
                        "time_deriv": str(args.time_deriv),
                        "history_mode": str(args.history_mode),
                        "rate_lags_init_s": [float(v) for v in args.rate_lags],
                        "seed": int(args.seed),
                    },
                },
                checkpoint_path,
            )
            checkpoint = str(checkpoint_path)

        results.append(dict(
            w_phys=float(w_phys), intime_mae=intime, held_mae=held_mae,
            L_data=L_data_raw, L_phys=L_phys_raw, wL_data=wL_data, wL_phys=wL_phys,
            delta_s=delta_s, gates=gates, src_gain=src_gain, diff_gain=diff_gain,
            n_params=n_params, test_time_maes=test_time_maes,
            rate_lags_s=learned_rate_lags_s, checkpoint=checkpoint,
        ))
        histories.append({"w_phys": w_phys, "hist": hist})
        print(f"  MAE(in-time)={intime:.3f} C  MAE(held {cli.test_op})={held_mae:.3f} C  "
              f"| L_data={L_data_raw:.4g}  L_phys={L_phys_raw:.4g}  "
              f"| w*L_data={wL_data:.4g}  w*L_phys={wL_phys:.4g}", flush=True)

    # ---- CSV -----------------------------------------------------------------
    csv_lines = [
        "w_phys,L_data,L_phys_unweighted,MAE_in_C,MAE_test_C,"
        "delta_s,src_gain,diff_gain,rate_lags_s,checkpoint"
    ]
    for r in results:
        lags_str = ";".join(f"{v:.6g}" for v in r["rate_lags_s"])
        csv_lines.append(
            f"{r['w_phys']},{r['L_data']:.6f},{r['L_phys']:.6f},"
            f"{r['intime_mae']:.4f},{r['held_mae']:.4f},"
            f"{r['delta_s']:.6f},{r['src_gain']:.6f},{r['diff_gain']:.6f},"
            f"\"{lags_str}\",{r['checkpoint']}"
        )
    (ART_DIR / "benchmark_wphys.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- best pick + summary table -------------------------------------------
    best = min(results, key=lambda r: r["held_mae"])
    if cli.save_models and cli.save_best_only:
        best_ckpt_path = model_dir / f"model_best_wphys_{_w_tag(best['w_phys'])}.pt"
        best_rec = next(r for r in results if r["w_phys"] == best["w_phys"])
        args_best = _make_args(cli, best_rec["w_phys"])
        model_best, bundle_best, _packed_best, dtn_best, _hist_best = fit(args_best)
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
                    "w_phys": float(best_rec["w_phys"]),
                    "ops": list(args_best.ops),
                    "test_op": args_best.test_op,
                    "epochs": int(args_best.epochs),
                    "subsample": int(args_best.subsample),
                    "time_deriv": str(args_best.time_deriv),
                    "history_mode": str(args_best.history_mode),
                    "rate_lags_init_s": [float(v) for v in args_best.rate_lags],
                    "seed": int(args_best.seed),
                },
            },
            best_ckpt_path,
        )
        best["checkpoint"] = str(best_ckpt_path)

    th = (f"{'w_phys':>7} | {'L_data':>10} {'L_phys(unw)':>12} | "
          f"{'MAE_in':>7} {'MAE_test':>8}")
    summary = header + [th, "-" * len(th)]
    for r in results:
        summary.append(
            f"{r['w_phys']:>7} | {r['L_data']:>10.4g} {r['L_phys']:>12.4g} | "
            f"{r['intime_mae']:>7.3f} {r['held_mae']:>8.3f}"
        )
    summary += [
        "",
        "MAE = mean |true - predicted| (deg C) from the free-running rollout.",
        "L_data / L_phys(unw) = final-epoch UN-weighted losses.",
        f"BEST (by held-out MAE): w_phys = {best['w_phys']}  "
        f"-> held-out {best['held_mae']:.3f} C, in-time {best['intime_mae']:.3f} C",
    ]
    if cli.save_models:
        summary.append(f"Checkpoints dir: {model_dir}")
        if cli.save_best_only and best.get("checkpoint", ""):
            summary.append(f"Saved best checkpoint: {best['checkpoint']}")
    (ART_DIR / "benchmark_wphys.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[len(header):]), flush=True)

    # ---- Convergence plots ---------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for item in histories:
        w = item["w_phys"]
        h = item["hist"]
        epochs = h["epoch"]
        label = f"w_phys={w:.3f}"
        
        # Top: data loss
        axes[0].plot(epochs, h["L_data"], marker="o", markersize=3, label=label)
        axes[0].set_ylabel("L_data (MSE)", fontsize=11)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].set_title(f"Convergence: {'+'.join(cli.ops)} train, {cli.test_op} test, {cli.epochs} epochs", fontsize=12)
        
        # Middle: physics loss (raw, unweighted)
        if "L_phys" in h and len(h["L_phys"]) > 0:
            axes[1].plot(epochs, h["L_phys"], marker="s", markersize=3, label=label, alpha=0.8)
        axes[1].set_ylabel("L_phys (unweighted)", fontsize=11)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8, ncol=2)
        
        # Bottom: balanced losses (fair comparison)
        if "L_phys_bal" in h and len(h["L_phys_bal"]) > 0:
            axes[2].plot(epochs, h["L_phys_bal"], marker="^", markersize=3, 
                        label=f"{label} L_phys_bal", linestyle="--", alpha=0.7)
        if "L_bc_bal" in h and len(h["L_bc_bal"]) > 0:
            axes[2].plot(epochs, h["L_bc_bal"], marker="v", markersize=3, 
                        label=f"{label} L_bc_bal", linestyle=":", alpha=0.7)
    
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Balanced Loss (~O(1))", fontsize=11)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, ncol=3)
    
    plt.tight_layout()
    convergence_plot = ART_DIR / "benchmark_wphys_convergence.png"
    plt.savefig(convergence_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved convergence plot: {convergence_plot}", flush=True)

    # ---- plot ----------------------------------------------------------------
    labels = [str(r["w_phys"]) for r in results]
    x = np.arange(len(results))
    intime = [r["intime_mae"] for r in results]
    held = [r["held_mae"] for r in results]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax.plot(x, intime, "o-", color="C0", label="in-time TEST rollout MAE")
    ax.plot(x, held, "s--", color="C3", label=f"held-out {cli.test_op} rollout MAE")
    bi = int(np.argmin(held))
    ax.scatter([x[bi]], [held[bi]], s=140, facecolors="none", edgecolors="k",
               zorder=5, label="best (held-out)")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("physics-loss weight  w_phys  (w_data = 1.0)")
    ax.set_ylabel("free-running rollout MAE  [C]")
    ax.set_title("MAE = mean |true - predicted|  vs  w_phys")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # right panel: weighted loss terms -> shows whether w_phys*L_phys lands in
    # the same range as w_data*L_data (i.e. balanced).
    wl_data = [r["wL_data"] for r in results]
    wl_phys = [r["wL_phys"] for r in results]
    ax2.plot(x, wl_data, "o-", color="C2", label="w_data * L_data (=L_data)")
    ax2.plot(x, wl_phys, "^-", color="C1", label="w_phys * L_phys")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("physics-loss weight  w_phys")
    ax2.set_ylabel("weighted loss contribution")
    ax2.set_yscale("log")
    ax2.set_title("weighted loss terms (same range = balanced)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Physics-weight benchmark  (train={'+'.join(cli.ops)}, "
                 f"test={cli.test_op}, width={cli.width}, depth={cli.depth}, "
                 f"k={cli.k_max}, mode={cli.history_mode}, {cli.epochs}ep, dt={dt_s:.0f}s)")
    fig.tight_layout()
    fig.savefig(ART_DIR / "benchmark_wphys.png", dpi=130)
    plt.close(fig)

    # ---- test-op boxplot -----------------------------------------------------
    fig2, ax3 = plt.subplots(1, 1, figsize=(10.5, 4.8))
    box_data = [r["test_time_maes"] for r in results]
    bp = ax3.boxplot(box_data, labels=labels, showmeans=True, whis=(0, 100), patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e8ff")
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#b00020")
        median.set_linewidth(2.0)
    for flier in bp.get("fliers", []):
        flier.set(markerfacecolor="#444444", markeredgecolor="#444444", markersize=4)

    # Overlay the 10 time-point MAEs so the sampled points are visible.
    for i, vals in enumerate(box_data, start=1):
        jitter = np.linspace(-0.08, 0.08, num=len(vals))
        ax3.scatter(np.full(len(vals), i) + jitter, vals, s=16, color="#1f77b4", alpha=0.7, zorder=3)

    ax3.set_xlabel("physics-loss weight  w_phys  (w_data = 1.0)")
    ax3.set_ylabel("held-out test-op MAE across 10 time points  [C]")
    ax3.set_title(f"Test-op MAE distribution from 10 uniformly spaced time points ({cli.test_op})")
    ax3.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(ART_DIR / "benchmark_wphys_test_boxplot.png", dpi=130)
    plt.close(fig2)

    # ---- parameters vs physics weight ----------------------------------------
    fig3, (axp1, axp2, axp3) = plt.subplots(1, 3, figsize=(16, 4.8))
    delta_vals = [r["delta_s"] for r in results]
    src_vals = [r["src_gain"] for r in results]
    diff_vals = [r["diff_gain"] for r in results]
    axp1.plot(x, delta_vals, "o-", color="C4")
    axp1.set_xticks(x)
    axp1.set_xticklabels(labels)
    axp1.set_xlabel("w_phys")
    axp1.set_ylabel("delta [s]")
    axp1.set_title("learned delta")
    axp1.grid(True, alpha=0.3)

    axp2.plot(x, src_vals, "o-", color="C5", label="src_gain")
    axp2.plot(x, diff_vals, "s--", color="C6", label="diff_gain")
    axp2.set_xticks(x)
    axp2.set_xticklabels(labels)
    axp2.set_xlabel("w_phys")
    axp2.set_ylabel("gain")
    axp2.set_title("physics gains")
    axp2.legend()
    axp2.grid(True, alpha=0.3)

    max_lags = max((len(r["rate_lags_s"]) for r in results), default=0)
    for li in range(max_lags):
        y = [r["rate_lags_s"][li] if li < len(r["rate_lags_s"]) else np.nan for r in results]
        axp3.plot(x, y, "o-", label=f"lag_{li + 1}")
    axp3.set_xticks(x)
    axp3.set_xticklabels(labels)
    axp3.set_xlabel("w_phys")
    axp3.set_ylabel("learned lag [s]")
    axp3.set_title("learned hybrid rate lags")
    if max_lags > 0:
        axp3.legend()
    axp3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(ART_DIR / "benchmark_wphys_params.png", dpi=130)
    plt.close(fig3)

    # ---- loss diagnostics ----------------------------------------------------
    fig4, (axl1, axl2) = plt.subplots(1, 2, figsize=(13, 4.8))
    l_data = [max(float(r["L_data"]), 1e-30) for r in results]
    l_phys = [max(float(r["L_phys"]), 1e-30) for r in results]
    balance = [float(r["wL_phys"]) / max(float(r["wL_data"]), 1e-30) for r in results]

    axl1.plot(x, l_data, "o-", color="C2", label="L_data (unweighted)")
    axl1.plot(x, l_phys, "^-", color="C1", label="L_phys (unweighted)")
    axl1.set_xticks(x)
    axl1.set_xticklabels(labels)
    axl1.set_yscale("log")
    axl1.set_xlabel("w_phys")
    axl1.set_ylabel("loss (log scale)")
    axl1.set_title("final unweighted losses")
    axl1.legend()
    axl1.grid(True, which="both", alpha=0.3)

    axl2.plot(x, balance, "o-", color="C0")
    axl2.axhline(1.0, color="k", ls=":", lw=1.0)
    axl2.set_xticks(x)
    axl2.set_xticklabels(labels)
    axl2.set_yscale("log")
    axl2.set_xlabel("w_phys")
    axl2.set_ylabel("(w_phys * L_phys) / (w_data * L_data)")
    axl2.set_title("weighted-term balance (1.0 is balanced)")
    axl2.grid(True, which="both", alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(ART_DIR / "benchmark_wphys_losses.png", dpi=130)
    plt.close(fig4)

    print(f"  wrote {ART_DIR/'benchmark_wphys.csv'}, .png and .txt", flush=True)
    print(f"  wrote {ART_DIR/'benchmark_wphys_test_boxplot.png'}", flush=True)
    print(f"  wrote {ART_DIR/'benchmark_wphys_params.png'}", flush=True)
    print(f"  wrote {ART_DIR/'benchmark_wphys_losses.png'}", flush=True)
    if cli.save_models:
        print(f"  wrote checkpoints to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
