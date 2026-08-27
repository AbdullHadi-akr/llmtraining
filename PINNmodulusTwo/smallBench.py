#!/usr/bin/env python3
"""Small convergence benchmark: quick sanity check before full sweeps.

Runs a short training (few epochs, 2 w_phys values) to verify:
1. CFL stability (no inf/NaN losses)
2. Loss convergence (L_data decreasing)
3. Balanced losses are ~O(1)
4. Test MAE is reasonable (< 20°C)

This is also "step A" of PINNmodulusTwo/README_MODEL_CRITIQUE.md: run it once as
it stands and once as

    --inner-steps 1 --no-residual-output --learn-gains

which is the configuration from before the training-budget, residual-output and
physics-residual fixes. Comparing the two Test MAEs is what decides whether those
fixes actually helped -- so far they are only verified mathematically -- and the
critique file turns each possible outcome into the next step to take.

Run:
    cd /mnt/c/Users/M0245635/batterysurrogatemodell
    source modulus_env/bin/activate
    python3 PINNmodulusTwo/smallBench.py

Expected runtime: ~2-5 minutes on CPU.
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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from data import build_op
from device_utils import resolve_device
from model import rollout
from train import fit

ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


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
    p = argparse.ArgumentParser(description="Small convergence benchmark")
    p.add_argument("--ops", nargs="+", default=d.get("ops", ["OP01", "OP02"]))
    p.add_argument("--test-op", default=d.get("test_op", "OP07"))
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 2))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--width", type=int, default=d.get("layer_size", 64))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 3))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--history-mode", default=d.get("history_mode", "hybrid"))
    p.add_argument("--rate-lags", nargs="+", type=float, default=d.get("rate_lags", [5.0, 20.0]))
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2),
                   help="anchor lag of the hybrid history in seconds")
    p.add_argument("--time-deriv", default=d.get("time_deriv", "bdf2"))
    p.add_argument("--lr", type=float, default=d.get("lr", 0.002))
    p.add_argument("--w-phys", type=float, nargs="+", default=[0.0, 0.1])
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 1024))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 128))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 64))
    p.add_argument("--inner-steps", type=int, default=d.get("inner_steps", 100),
                   help="optimiser steps per OP per epoch against that epoch's "
                        "frozen rollout")
    p.add_argument("--rollout-clamp", type=float,
                   default=d.get("rollout_clamp", 50.0),
                   help="saturate the rollout buffer at +/-this many normalised "
                        "temperature units; 0 disables")
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0),
                   help="cap the hybrid history amplification A by raising "
                        "rate_scale; 0 = leave it at dTdt_scale")
    p.add_argument("--residual-output", action=argparse.BooleanOptionalAction,
                   default=d.get("residual_output", True))
    p.add_argument("--learn-gains", action=argparse.BooleanOptionalAction,
                   default=d.get("learn_gains", False))
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", True))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", True))
    p.add_argument("--loss-balance", choices=["ema", "legacy", "fixed"],
                   default=d.get("loss_balance", "ema"))
    p.add_argument("--ema-decay", type=float, default=d.get("ema_decay", 0.9))
    p.add_argument("--balance-warmup", type=int, default=d.get("balance_warmup", 1))
    p.add_argument("--data-floor", type=float, default=d.get("data_floor", 1e-8))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--bc-norm", type=float, default=d.get("bc_norm", 0.0))
    p.add_argument("--residual-norm", choices=["rms", "legacy"],
                   default=d.get("residual_norm", "rms"))
    p.add_argument("--zero-weight-terms", choices=["skip", "compute"],
                   default=d.get("zero_weight_terms", "skip"))
    p.add_argument("--subsample-mode", choices=["stride", "mean"],
                   default=d.get("subsample_mode", "stride"))
    p.add_argument("--forcing-energy", action="store_true",
                   default=d.get("forcing_energy", False))
    p.add_argument("--config-rates", action="store_true",
                   default=d.get("config_rates", False))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N (auto = cuda when available)")
    return p.parse_args()


class Args:
    """Namespace-like class for fit() compatibility."""
    pass


def _make_args(cli, w_phys: float) -> Args:
    args = Args()
    args.ops = cli.ops
    args.subsample = cli.subsample
    args.epochs = cli.epochs
    args.k_max = cli.k_max
    args.history_mode = cli.history_mode
    args.rate_lags = cli.rate_lags
    # fit() liest beide nur via getattr. Ohne diese Zeilen faellt delta_grid
    # still auf den Datenschritt zurueck statt auf den konfigurierten Wert --
    # bei subsample=2 zufaellig identisch, bei jedem anderen Wert nicht.
    args.delta_grid = cli.delta_grid
    args.gain_lr_mult = cli.gain_lr_mult
    args.time_deriv = cli.time_deriv
    args.width = cli.width
    args.depth = cli.depth
    args.lr = cli.lr
    args.w_data = 1.0
    args.w_phys = float(w_phys)
    args.w_bc = cli.w_bc
    args.batch_data = cli.batch_data
    args.batch_phys = cli.batch_phys
    args.batch_bc = cli.batch_bc
    args.inner_steps = cli.inner_steps
    args.rollout_clamp = cli.rollout_clamp
    args.max_rate_amp = cli.max_rate_amp
    args.residual_output = cli.residual_output
    args.learn_gains = cli.learn_gains
    args.weight_decay = 0.0
    args.grad_clip = cli.grad_clip
    args.early_stopping_patience = 0
    # Same trap as delta_grid above: fit() reads all of these via getattr, so a
    # field missing here does not raise -- the smoke test would quietly run a
    # different balancing than config.yaml asks for, which is exactly what this
    # file exists to rule out.
    args.phys_norm = cli.phys_norm
    args.bc_norm = cli.bc_norm
    args.loss_balance = cli.loss_balance
    args.ema_decay = cli.ema_decay
    args.balance_warmup = cli.balance_warmup
    args.data_floor = cli.data_floor
    args.residual_norm = cli.residual_norm
    args.zero_weight_terms = cli.zero_weight_terms
    args.subsample_mode = cli.subsample_mode
    args.forcing_energy = cli.forcing_energy
    args.config_rates = cli.config_rates
    args.use_static = cli.use_static
    args.use_forcing = cli.use_forcing
    args.seed = cli.seed
    args.device = cli.device
    args.test_op = cli.test_op
    return args


@torch.no_grad()
def _rollout_mae(model, op, bundle, device) -> float:
    """Free-running rollout MAE in physical °C."""
    xn = torch.as_tensor(op.xn, dtype=torch.float32, device=device)
    static = torch.as_tensor(op.static_feat, dtype=torch.float32, device=device)
    forcing = torch.as_tensor(op.forcing_feat, dtype=torch.float32, device=device)
    cfg = torch.as_tensor(op.config_feat, dtype=torch.float32, device=device)
    tn = torch.as_tensor(op.tn, dtype=torch.float32, device=device)
    Tn_ic = torch.as_tensor(op.Tn_ic, dtype=torch.float32, device=device)
    static = static[:, :model.n_static]
    forcing = forcing[:, :model.n_forcing]
    buf = rollout(model, xn, static, cfg, forcing, Tn_ic, tn, op.dtn)
    T_pred = buf.cpu().numpy() * bundle.T_sigma + bundle.T_mu
    return float(np.abs(T_pred - op.T_lab).mean())


def main():
    cli = parse_args()
    device = resolve_device(cli.device)
    cli.device = str(device)  # hand the resolved device down to fit()
    dt_s = 0.1 * cli.subsample

    print("=" * 70)
    print("SMALL CONVERGENCE BENCHMARK")
    print("=" * 70)
    print(f"  Training OPs:  {cli.ops}")
    print(f"  Test OP:       {cli.test_op}")
    print(f"  Δt:            {dt_s:.1f}s (subsample={cli.subsample})")
    print(f"  Epochs:        {cli.epochs}")
    print(f"  Architecture:  width={cli.width}, depth={cli.depth}")
    print(f"  History:       mode={cli.history_mode}, rate_lags={cli.rate_lags}")
    print(f"  Physics sweep: w_phys={cli.w_phys}")
    print(f"  Grad clip:     {cli.grad_clip}")
    print("=" * 70)
    print()

    results = []
    all_passed = True
    all_histories = []  # Store histories for convergence plot

    for w_phys in cli.w_phys:
        print(f"\n{'='*50}")
        print(f"Training with w_phys = {w_phys}")
        print(f"{'='*50}")

        args = _make_args(cli, w_phys)
        model, bundle, ops_packed, dtn, hist = fit(args)
        model.eval()

        # A term whose weight is 0 is not computed at all (train.py
        # --zero-weight-terms), and is recorded as NaN on purpose. That is an
        # absent measurement, not a failed one: checking it would fail the very
        # w_phys=0 reference point this smoke test needs in order to show what
        # the physics term is worth.
        phys_on = float(w_phys) != 0.0
        bc_on = float(cli.w_bc) != 0.0

        # Check 1: No inf/NaN in final losses
        L_data_final = hist["L_data"][-1] if hist["L_data"] else float("nan")
        L_phys_final = hist["L_phys"][-1] if hist["L_phys"] else float("nan")
        stable = np.isfinite(L_data_final) and (not phys_on
                                                or np.isfinite(L_phys_final))

        # Check 2: Loss decreased (convergence)
        if len(hist["L_data"]) >= 2:
            L_data_first = hist["L_data"][0]
            converged = L_data_final < L_data_first
        else:
            converged = False

        # Check 3: Balanced losses are ~O(1) -- only where a term is switched on
        L_phys_bal = hist.get("L_phys_bal", [1.0])[-1]
        L_bc_bal = hist.get("L_bc_bal", [1.0])[-1]
        balanced = ((not phys_on or 0.01 < L_phys_bal < 100)
                    and (not bc_on or 0.01 < L_bc_bal < 100))

        # Check 4: Test MAE is reasonable
        held = build_op(cli.test_op, bundle, subsample_time=cli.subsample)
        test_mae = _rollout_mae(model, held, bundle, device)
        mae_ok = test_mae < 20.0

        # In-time MAE for training OPs
        train_maes = [_rollout_mae(model, op, bundle, device) 
                      for op in [build_op(oid, bundle, cli.subsample) for oid in cli.ops]]
        train_mae = float(np.mean(train_maes))

        passed = stable and converged and balanced and mae_ok

        results.append({
            "w_phys": w_phys,
            "L_data_final": L_data_final,
            "L_phys_bal": L_phys_bal,
            "L_bc_bal": L_bc_bal,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "stable": stable,
            "converged": converged,
            "balanced": balanced,
            "mae_ok": mae_ok,
            "passed": passed,
        })
        all_histories.append({"w_phys": w_phys, "hist": hist})

        if not passed:
            all_passed = False

        print(f"\n  Results for w_phys={w_phys}:")
        print(f"    L_data(final):  {L_data_final:.4e}  {'✓' if stable else '✗ (inf/NaN)'}")
        phys_note = ("(w_phys=0: term not computed)" if not phys_on
                     else "✓" if balanced else "✗ (not ~O(1))")
        print(f"    L_phys_bal:     {L_phys_bal:.4e}  {phys_note}")
        print(f"    L_bc_bal:       {L_bc_bal:.4e}")
        print(f"    Train MAE:      {train_mae:.2f}°C")
        print(f"    Test MAE:       {test_mae:.2f}°C  {'✓' if mae_ok else '✗ (>20°C)'}")
        print(f"    Converged:      {'✓' if converged else '✗'}")
        print(f"    Overall:        {'✓ PASS' if passed else '✗ FAIL'}")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'w_phys':>8} | {'L_data':>10} | {'L_phys_bal':>10} | {'Train MAE':>10} | {'Test MAE':>10} | {'Status':>8}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['w_phys']:>8.3f} | {r['L_data_final']:>10.4e} | {r['L_phys_bal']:>10.4e} | "
              f"{r['train_mae']:>9.2f}C | {r['test_mae']:>9.2f}C | {status:>8}")
    print("-" * 70)

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Ready for full benchmark!")
        # NOT the 10x10 grid: that is 100 trainings (~6-8 days) and it would
        # sweep weights before anything has established what a weight means
        # here. The balancing benchmark is ~4 h and settles that first.
        print("  Next: python3 PINNmodulusTwo/benchmark_balance.py --part 1 "
              "--epochs 20 --device cuda")
        print("        (settles the loss balancing; the weight probe in "
              "benchmark_wphys_wbc.py comes after it)")
    else:
        print("\n✗ SOME CHECKS FAILED - Review issues above before full benchmark")

    # Plot convergence curves
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for item in all_histories:
        w = item["w_phys"]
        h = item["hist"]
        epochs = h["epoch"]
        label = f"w_phys={w:.3f}"
        
        # Top: data loss
        axes[0].plot(epochs, h["L_data"], marker="o", label=label)
        axes[0].set_ylabel("L_data (MSE)", fontsize=11)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        axes[0].set_title(f"Convergence: {len(cli.ops)} OPs, {cli.epochs} epochs", fontsize=12)
        
        # Bottom: balanced physics and BC losses (if applicable)
        if "L_phys_bal" in h and len(h["L_phys_bal"]) > 0:
            axes[1].plot(epochs, h["L_phys_bal"], marker="s", linestyle="--", 
                        label=f"{label} L_phys_bal", alpha=0.7)
        if "L_bc_bal" in h and len(h["L_bc_bal"]) > 0:
            axes[1].plot(epochs, h["L_bc_bal"], marker="^", linestyle=":", 
                        label=f"{label} L_bc_bal", alpha=0.7)
    
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Balanced Loss", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    convergence_plot = ART_DIR / "smallBench_convergence.png"
    plt.savefig(convergence_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved convergence plot to {convergence_plot}")

    # Save results
    out_file = ART_DIR / "smallBench_results.txt"
    with open(out_file, "w") as f:
        f.write("Small Benchmark Results\n")
        f.write(f"OPs: {cli.ops}, Test: {cli.test_op}\n")
        f.write(f"Δt: {dt_s}s, Epochs: {cli.epochs}\n")
        f.write(f"Architecture: width={cli.width}, depth={cli.depth}\n\n")
        for r in results:
            f.write(f"w_phys={r['w_phys']}: L_data={r['L_data_final']:.4e}, "
                    f"test_mae={r['test_mae']:.2f}C, {'PASS' if r['passed'] else 'FAIL'}\n")
    print(f"  Saved results to {out_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
