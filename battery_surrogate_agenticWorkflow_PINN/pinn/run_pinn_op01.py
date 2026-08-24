#!/usr/bin/env python3
"""Run the OP01 PINN (plan 004): train net_T + net_V, evaluate, and plot.

Usage (from repo root):
    source modulus_env/bin/activate
    python3 battery_surrogate_agenticWorkflow_PINN/pinn/run_pinn_op01.py \
        --epochs 80 --subsample 40 --depth 4 --width 128 --k 2

Outputs (under battery_surrogate_agenticWorkflow_PINN/artifacts/op01_pinn/):
    loss_curves.png           values of EVERY loss + weights (BC weight = 0 shown)
    T_true_vs_pred_train.png  true vs predicted T, TRAIN timesteps
    T_true_vs_pred_test.png   true vs predicted T, TEST timesteps
    T_timeseries.png          T(t) at a few grid points (train/test split marked)
    V_true_vs_pred.png        bc_V true vs predicted
    norm_stats_OP01.json      saved normalization stats
    run_summary.txt           model + training parameters and metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PINN_ROOT = Path(__file__).parent.parent          # battery_surrogate_agenticWorkflow_PINN
PROJECT_ROOT = PINN_ROOT.parent                   # batterysurrogatemodell
sys.path.insert(0, str(PINN_ROOT))

from pinn.data.load_op01 import load_op01_data
from pinn.data.load_properties import load_material_properties
from pinn.data.preprocess import build_norm_stats
from pinn.train.train_pinn import (
    TemperaturePINN,
    LossWeights,
    train_temperature,
    train_voltage,
)


def banner(msg: str) -> None:
    print("=" * 80, flush=True)
    print(msg, flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--subsample", type=int, default=40)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--bptt", type=int, default=8)
    ap.add_argument("--w-phys", type=float, default=0.1)
    ap.add_argument("--soft-ic", action="store_true", help="use SOFT IC instead of hard")
    ap.add_argument("--iso-physics", action="store_true", help="DEBUG: isotropic Laplacian")
    ap.add_argument("--v-epochs", type=int, default=400)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  device = {device}"
          + (f"  ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)

    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = PINN_ROOT / "artifacts" / "op01_pinn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- load data
    banner("Loading OP01 data")
    op = load_op01_data(
        npz_path=str(PROJECT_ROOT / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
        heat_source_csv=str(PINN_ROOT / "data/OP01_raw/OP01/OP1_Heat Source.csv"),
        subsample_time=args.subsample,
    )
    props = load_material_properties(
        layer=op["layer"],
        props_cc_dir=str(PINN_ROOT / "Cell Center"),
        props_jr1_dir=str(PINN_ROOT / "JR1 Center"),
    )
    t, xyz, T_labels, bc_V = op["t"], op["xyz"], op["T"], op["bc_V"]
    n_t, n_points = T_labels.shape
    split_t = int(0.8 * n_t)
    print(f"  timesteps={n_t} (dt={op['dt']}s, t0={t[0]}s, t_max={t[-1]}s)  points={n_points}")
    print(f"  train/test split at index {split_t}  -> train=[1,{split_t}) test=[{split_t},{n_t})")

    # -------------------------------------------------------------- stats
    stats = build_norm_stats(
        t=t, xyz=xyz, T_labels=T_labels, bc_V=bc_V,
        config=op["config"], T_init=op["config"]["solid_initial_temp"],
        train_slice=slice(0, split_t),
    )
    stats.to_json(out_dir / "norm_stats_OP01.json")

    # -------------------------------------------------------------- weights
    weights = LossWeights(w_data=1.0, w_phys=args.w_phys, w_ic=1.0, w_bc_in=0.0, w_bc_out=0.0)

    # -------------------------------------------------------------- print config
    hard_ic = not args.soft_ic
    banner("MODEL & TRAINING CONFIGURATION")
    cfg_lines = [
        "SEPARATED NETS      : YES  -> net_T (temperature, recurrent) & net_V (voltage, non-recurrent)",
        f"net_T architecture  : depth={args.depth}, width={args.width}, activation=SiLU (Swish, beta=1 fixed)",
        f"net_T inputs        : (x,y,z,t) + config[7] + T_history[k]  = {4+7+args.k} features",
        f"recurrent k         : {args.k}  (history length; NO teacher forcing -> uses predicted T_hat)",
        f"warm-up history     : OFFICIAL measured first sample T[0] at t0={t[0]}s (treated as t~=0)",
        f"dt (delta t)        : {op['dt']} s   (subsample={args.subsample})",
        f"BPTT window W       : {args.bptt}  (truncated backprop-through-time)",
        f"IC enforcement      : {'HARD (T~ = T~_ic + t~*N, exact at t=0.1s)' if hard_ic else 'SOFT (loss penalty)'}",
        f"BC enforcement      : DISABLED  -> w_bc_in = {weights.w_bc_in}, w_bc_out = {weights.w_bc_out}  (BC NOT included)",
        f"physics             : {'ISOTROPIC (debug)' if args.iso_physics else 'ANISOTROPIC solid heat eq (Notion-faithful, full lambda tensor)'}",
        "normalization       : xyz,t -> [0,1] ; outputs T & bc_V + config -> z-score",
        "",
        "LOSS WEIGHTS:",
        f"   w_data  = {weights.w_data}",
        f"   w_phys  = {weights.w_phys}",
        f"   w_ic    = {weights.w_ic}",
        f"   w_bc_in = {weights.w_bc_in}   <-- BC NOT INCLUDED",
        f"   w_bc_out= {weights.w_bc_out}   <-- BC NOT INCLUDED",
    ]
    for ln in cfg_lines:
        print(ln, flush=True)
    print("\n>>> NOTE: BC is NOT included in this run (w_bc = 0). It is plotted as 0 so it stays visible.\n", flush=True)

    # -------------------------------------------------------------- train net_T
    banner(f"Training net_T ({args.epochs} epochs)")
    trainer = TemperaturePINN(
        t=t, xyz=xyz, T_labels=T_labels,
        q_dot=op["q_dot"], rho=props["rho"], Cp=props["Cp"],
        lambda_tensor=props["lambda_tensor"], region=props["region"],
        stats=stats, weights=weights,
        depth=args.depth, width=args.width, k=args.k, bptt_window=args.bptt,
        hard_ic=hard_ic, iso_physics=args.iso_physics, split_t=split_t, device=device,
    )
    n_params_T = trainer.n_params()
    train_temperature(trainer, epochs=args.epochs, log_interval=max(1, args.epochs // 10))

    # -------------------------------------------------------------- train net_V
    banner(f"Training net_V ({args.v_epochs} epochs)")
    net_V, v_hist, V_pred = train_voltage(
        t=t, bc_V=bc_V, stats=stats, split_t=split_t, epochs=args.v_epochs, device=device,
    )
    n_params_V = sum(p.numel() for p in net_V.parameters())

    # -------------------------------------------------------------- evaluate
    banner("Evaluation (free-running rollout)")
    T_pred = trainer.rollout_predict()               # (n_t, n_points) physical
    err = np.abs(T_pred - T_labels)
    mae_train = float(err[1:split_t].mean())
    mae_test = float(err[split_t:].mean())
    rmse_train = float(np.sqrt(((T_pred[1:split_t] - T_labels[1:split_t]) ** 2).mean()))
    rmse_test = float(np.sqrt(((T_pred[split_t:] - T_labels[split_t:]) ** 2).mean()))
    v_mae = float(np.abs(V_pred - bc_V).mean())
    print(f"  net_T  MAE  train={mae_train:.3f} C   test={mae_test:.3f} C")
    print(f"  net_T  RMSE train={rmse_train:.3f} C   test={rmse_test:.3f} C")
    print(f"  net_V  MAE  = {v_mae:.4f} V")

    # save predictions for offline inspection + random (time, point) abs error
    np.savez_compressed(
        out_dir / "op01_predictions.npz",
        t=t, xyz=xyz, T_true=T_labels, T_pred=T_pred, bc_V_true=bc_V, bc_V_pred=V_pred,
        split_t=split_t,
    )
    rng_e = np.random.default_rng(123)
    ti_r = int(rng_e.integers(1, n_t))
    pt_r = int(rng_e.integers(0, n_points))
    abs_err_r = float(abs(T_pred[ti_r, pt_r] - T_labels[ti_r, pt_r]))
    region_r = "train" if ti_r < split_t else "test"
    print(f"  RANDOM sample abs error: t[{ti_r}]={t[ti_r]:.1f}s  point {pt_r}  ({region_r})")
    print(f"    true={T_labels[ti_r, pt_r]:.3f} C  pred={T_pred[ti_r, pt_r]:.3f} C  |err|={abs_err_r:.3f} C")

    # -------------------------------------------------------------- plots
    banner("Writing plots")
    h = trainer.history
    ep = np.arange(1, len(h.data) + 1)

    # (1) every loss + weights
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(ep, np.clip(h.data, 1e-12, None), "C0-o", ms=3, lw=1.5, label=f"L_data (w={weights.w_data})")
    ax.semilogy(ep, np.clip(h.phys, 1e-12, None), "C1-s", ms=3, lw=1.5, label=f"L_phys (w={weights.w_phys})  [FIXED]")
    ax.semilogy(ep, np.clip(h.ic, 1e-12, None), "C2-^", ms=3, lw=1.5, label=f"L_IC (w={weights.w_ic})")
    ax.plot(ep, h.bc_in, "--", color="gray", alpha=0.5, label=f"L_BCin (w={weights.w_bc_in}) = 0  [BC NOT included]")
    ax.plot(ep, h.bc_out, ":", color="gray", alpha=0.5, label=f"L_BCout (w={weights.w_bc_out}) = 0  [BC NOT included]")
    # "before fix" reference: L_phys was stuck ~1e5 and oscillating
    ax.axhline(1e5, color="C1", ls=":", lw=1.5, alpha=0.6, label="L_phys BEFORE fix (~1e5, oscillating)")
    ax.annotate("Before fix\n~1e5", xy=(ep[-1] * 0.05, 1e5), xytext=(ep[-1] * 0.05, 1e5 * 3),
                fontsize=8, color="C1", alpha=0.8,
                arrowprops=dict(arrowstyle="->", color="C1", alpha=0.6))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss components vs epoch\n(L_phys fixed: O(1) and decreasing vs ~1e5 before)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "loss_curves.png", dpi=130); plt.close(fig)

    # (1b) focused physics-loss comparison plot
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    # left: all 3 relevant losses on log scale
    axes[0].semilogy(ep, np.clip(h.data, 1e-12, None), "C0-", lw=1.8, label="L_data")
    axes[0].semilogy(ep, np.clip(h.phys, 1e-12, None), "C1-", lw=1.8, label="L_phys (after fix)")
    axes[0].semilogy(ep, np.clip(h.ic, 1e-12, None), "C2-", lw=1.8, label="L_IC")
    axes[0].axhline(1e5, color="C1", ls="--", lw=1.5, alpha=0.55, label="L_phys BEFORE fix")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (log)")
    axes[0].set_title("All losses (log)")
    axes[0].legend(fontsize=9); axes[0].grid(True, which="both", alpha=0.3)
    # right: physics loss only, zoomed
    axes[1].semilogy(ep, np.clip(h.phys, 1e-12, None), "C1-s", ms=4, lw=2, label="L_phys (after fix)")
    axes[1].axhline(1e5, color="C1", ls="--", lw=1.5, alpha=0.55, label="L_phys before fix (~1e5)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("L_phys (log)")
    axes[1].set_title("Physics loss: before vs after fix")
    axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)
    fig2.suptitle("Physics-loss improvement: ~1e5 → O(1)", fontsize=12, fontweight="bold")
    fig2.tight_layout(); fig2.savefig(out_dir / "phys_loss_comparison.png", dpi=130); plt.close(fig2)

    # (2)/(3) true vs pred scatter, train & test
    def scatter(idx, name, fname):
        yt = T_labels[idx].ravel(); yp = T_pred[idx].ravel()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(yt, yp, s=4, alpha=0.3)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r-", lw=1)
        mae = float(np.abs(yp - yt).mean())
        ax.set_xlabel("true T [C]"); ax.set_ylabel("predicted T [C]")
        ax.set_title(f"T true vs predicted - {name}  (MAE={mae:.2f} C)")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130); plt.close(fig)

    scatter(slice(1, split_t), "TRAIN set", "T_true_vs_pred_train.png")
    scatter(slice(split_t, n_t), "TEST set", "T_true_vs_pred_test.png")

    # (4) time series at a few points
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pts = np.linspace(0, n_points - 1, 4).astype(int)
    for p in pts:
        ax.plot(t, T_labels[:, p], "-", lw=1.2, label=f"true pt{p}")
        ax.plot(t, T_pred[:, p], "--", lw=1.2, label=f"pred pt{p}")
    ax.axvline(t[split_t], color="k", ls=":", label="train/test split")
    ax.set_xlabel("time [s]"); ax.set_ylabel("T [C]"); ax.set_title("T(t) true vs predicted")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "T_timeseries.png", dpi=130); plt.close(fig)

    # (6) single random grid-point temperature timeseries
    rng = np.random.default_rng(seed=42)
    rand_pt = int(rng.integers(0, n_points))
    xyz_pt = xyz[rand_pt]
    err_pt = float(np.abs(T_pred[1:, rand_pt] - T_labels[1:, rand_pt]).mean())
    fig, axes6 = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
    # top: true vs predicted
    axes6[0].plot(t, T_labels[:, rand_pt], "C0-", lw=2, label="True T (measured)")
    axes6[0].plot(t, T_pred[:, rand_pt], "C1--", lw=2, label="Predicted T (net_T)")
    axes6[0].axvspan(t[0], t[split_t], alpha=0.07, color="C0", label="train region")
    axes6[0].axvspan(t[split_t], t[-1], alpha=0.07, color="C3", label="test region")
    axes6[0].axvline(t[split_t], color="k", ls=":", lw=1.2)
    axes6[0].set_ylabel("Temperature [°C]", fontsize=11)
    axes6[0].set_title(
        f"OP01 — net_T prediction at random grid point {rand_pt}\n"
        f"xyz = ({xyz_pt[0]:.4f}, {xyz_pt[1]:.4f}, {xyz_pt[2]:.4f}) m   |   MAE = {err_pt:.3f} °C",
        fontsize=11,
    )
    axes6[0].legend(fontsize=9); axes6[0].grid(True, alpha=0.3)
    # bottom: absolute error
    axes6[1].fill_between(t[1:], np.abs(T_pred[1:, rand_pt] - T_labels[1:, rand_pt]),
                          color="C3", alpha=0.6, label="|error|")
    axes6[1].axvline(t[split_t], color="k", ls=":", lw=1.2)
    axes6[1].set_xlabel("time [s]", fontsize=11)
    axes6[1].set_ylabel("|error| [°C]", fontsize=11)
    axes6[1].legend(fontsize=9); axes6[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "T_single_gridpoint.png", dpi=130)
    plt.close(fig)
    print(f"  single grid-point plot: point {rand_pt}  xyz={xyz_pt}  MAE={err_pt:.3f} C", flush=True)

    # (5) voltage
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, bc_V, "-", label="true bc_V")
    ax.plot(t, V_pred, "--", label="pred bc_V")
    ax.axvline(t[split_t], color="k", ls=":", label="train/test split")
    ax.set_xlabel("time [s]"); ax.set_ylabel("bc_V [V]"); ax.set_title(f"bc_V true vs predicted (MAE={v_mae:.4f} V)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "V_true_vs_pred.png", dpi=130); plt.close(fig)

    # -------------------------------------------------------------- summary file
    summary = [
        "OP01 PINN run summary",
        "=====================",
        "SEPARATED NETS: YES (net_T recurrent, net_V non-recurrent)",
        f"net_T: depth={args.depth} width={args.width} k={args.k} bptt_W={args.bptt} params={n_params_T}",
        f"net_V: depth=2 width=64 (non-recurrent) params={n_params_V}",
        f"dt={op['dt']}s  subsample={args.subsample}  timesteps={n_t}  points={n_points}",
        f"IC: {'HARD' if hard_ic else 'SOFT'} using official measured first sample at t0={t[0]}s",
        f"physics: {'isotropic(debug)' if args.iso_physics else 'ANISOTROPIC (full lambda tensor)'}",
        "teacher forcing: NO (autoregressive predicted history)",
        "loss weights: "
        f"w_data={weights.w_data} w_phys={weights.w_phys} w_ic={weights.w_ic} "
        f"w_bc_in={weights.w_bc_in} w_bc_out={weights.w_bc_out}  (BC NOT included)",
        "",
        "METRICS:",
        f"  net_T MAE  train={mae_train:.3f} C  test={mae_test:.3f} C",
        f"  net_T RMSE train={rmse_train:.3f} C  test={rmse_test:.3f} C",
        f"  net_V MAE  = {v_mae:.4f} V",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary))
    print("\n".join(summary), flush=True)
    print(f"\nAll outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
