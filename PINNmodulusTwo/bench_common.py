"""Shared machinery for the PINNmodulusTwo benchmarks.

A benchmark here is always the same shape: walk a list of configurations, train
each one under one or more seeds, score every configuration by the MEAN over its
seeds on a validation OP, and report a second OP that never took part in any
selection. Only the configuration axis differs -- loss weights in
``benchmark_wphys_wbc.py``, architecture and history lags in
``benchmark_arch.py`` -- so everything except that axis lives here.

A configuration is expressed as ``overrides``: a dict of ``train.fit`` argument
names to values, applied on top of the CLI defaults. That keeps the sweep axis
data rather than code, so a new benchmark only has to describe its grid.
"""

from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from data import build_op
from model import rollout

# How many uniformly spaced time points the per-configuration boxplots sample.
N_BOX_POINTS = 10

# History shape returned by fit(); used as a placeholder when fit() itself failed.
EMPTY_HIST = {"epoch": [], "L_data": [], "L_phys": [], "L_bc": [],
              "L_phys_bal": [], "L_bc_bal": [], "delta": [], "aborted": True}


def make_train_args(cli, overrides: dict, seed: int) -> Namespace:
    """Build the Namespace ``train.fit`` expects for one configuration.

    ``w_phys``/``w_bc``/``rate_lags`` deliberately have no meaningful default
    here: whichever benchmark sweeps them passes them through ``overrides``, and
    whichever holds them fixed passes its fixed value the same way. That way the
    swept axis is never silently inherited from a CLI attribute that happens to
    be a list.
    """
    spec = dict(
        ops=list(cli.ops), subsample=cli.subsample, epochs=cli.epochs,
        k_max=cli.k_max, time_deriv=cli.time_deriv,
        history_mode=cli.history_mode, rate_lags=[],
        delta_grid=getattr(cli, "delta_grid", 0.2),
        width=cli.width, depth=cli.depth, lr=cli.lr,
        w_data=1.0, w_phys=0.0, w_bc=0.0,
        batch_data=cli.batch_data, batch_phys=cli.batch_phys,
        batch_bc=cli.batch_bc,
        weight_decay=cli.weight_decay, grad_clip=cli.grad_clip,
        gain_lr_mult=cli.gain_lr_mult,
        early_stopping_patience=cli.early_stopping_patience,
        phys_norm=cli.phys_norm,
        use_static=cli.use_static, use_forcing=cli.use_forcing,
        seed=int(seed), device=cli.device,
        test_op=cli.test_op,
    )
    spec.update(overrides)
    return Namespace(**spec)


@torch.no_grad()
def rollout_phys(model, op, bundle, device) -> np.ndarray:
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


def mae(pred, true, lo, hi) -> float:
    return float(np.abs(pred[lo:hi] - true[lo:hi]).mean())


def timepoint_maes(pred: np.ndarray, true: np.ndarray,
                   time_idx: np.ndarray) -> np.ndarray:
    """Mean absolute error at each selected time index, averaged over space."""
    return np.array([float(np.abs(pred[i] - true[i]).mean()) for i in time_idx],
                    dtype=float)


def failed_result(extra: dict, train_time: float, n_seeds: int = 1) -> dict:
    """Result row for a configuration where every seed diverged or crashed.

    Every configuration must produce exactly one entry, because the convergence
    plots pick their corner points by position.
    """
    nan = float("nan")
    row = {
        "intime_mae": nan, "val_mae": nan, "test_mae": nan,
        "val_mae_std": nan, "test_mae_std": nan,
        "n_seeds": int(n_seeds), "n_seeds_ok": 0,
        "L_data": nan, "L_phys": nan, "L_bc": nan,
        "delta_s": nan, "src_gain": nan, "diff_gain": nan,
        "n_params": 0,
        "rate_lags_s": [],
        "train_time": train_time,
        "checkpoint": "",
        "test_time_maes": np.full(N_BOX_POINTS, nan),
    }
    row.update(extra)
    return row


def train_one_seed(cli, overrides: dict, seed: int, device, fit,
                   checkpoint_path: Path | None = None,
                   context: dict | None = None):
    """Train one configuration at one seed.

    Returns ``(row_or_None, history)``. ``None`` means this seed produced nothing
    usable -- it crashed or its loss diverged -- and the caller drops it from the
    mean instead of letting a NaN poison the whole configuration.
    """
    args = make_train_args(cli, overrides, seed)
    try:
        model, bundle, _packed, dtn, hist = fit(args)
    except Exception as exc:  # one bad seed must not kill the sweep
        print(f"    seed {seed}: training failed ({exc})", flush=True)
        return None, EMPTY_HIST
    model.eval()

    # fit() aborts on NaN/Inf and records the failed epoch, so an empty history
    # only happens for epochs=0. Either way this seed has no usable result.
    L_data = float(hist["L_data"][-1]) if hist["L_data"] else float("nan")
    L_phys = float(hist["L_phys"][-1]) if hist["L_phys"] else float("nan")
    L_bc = float(hist["L_bc"][-1]) if hist["L_bc"] else float("nan")
    if not np.isfinite(L_data):
        print(f"    seed {seed}: loss diverged (L_data={L_data})", flush=True)
        return None, hist

    intime_maes = [
        mae(rollout_phys(model, op, bundle, device), op.T_lab, op.split_t, op.n_t)
        for op in bundle.ops
    ]

    # Selection OP: this is what the ranking uses.
    val_data = build_op(cli.val_op, bundle, subsample_time=cli.subsample)
    val_mae = mae(rollout_phys(model, val_data, bundle, device),
                  val_data.T_lab, 0, val_data.n_t)

    # Reporting OP: never used to choose anything, so the number stays an honest
    # held-out estimate for whichever configuration selection lands on.
    test_data = build_op(cli.test_op, bundle, subsample_time=cli.subsample)
    test_pred = rollout_phys(model, test_data, bundle, device)
    test_mae = mae(test_pred, test_data.T_lab, 0, test_data.n_t)

    idx = np.linspace(0, test_data.n_t - 1, num=N_BOX_POINTS, dtype=int)
    test_time_maes = timepoint_maes(test_pred, test_data.T_lab, idx)

    checkpoint = ""
    if checkpoint_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_config": bundle.n_config,
                    "n_static": model.n_static,
                    "n_forcing": model.n_forcing,
                    "k_max": args.k_max,
                    "history_mode": args.history_mode,
                    "rate_lags": [float(v) / bundle.T_span_ref
                                  for v in args.rate_lags],
                    "layer_size": args.width,
                    "num_layers": args.depth,
                    "delta_seconds": 1.0,
                    "dtn": float(dtn),
                    "t_span_ref": float(bundle.T_span_ref),
                    "rate_scale": float(bundle.dTdt_scale),
                    "delta_grid": float(args.delta_grid) / bundle.T_span_ref,
                    "use_autograd_time": (args.time_deriv == "autograd"),
                },
                "bundle_stats": {
                    "T_mu": float(bundle.T_mu),
                    "T_sigma": float(bundle.T_sigma),
                    "T_span_ref": float(bundle.T_span_ref),
                },
                "benchmark_context": {
                    **(context or {}),
                    "ops": list(args.ops),
                    "val_op": cli.val_op,
                    "test_op": args.test_op,
                    "epochs": int(args.epochs),
                    "subsample": int(args.subsample),
                    "seed": int(seed),
                },
            },
            checkpoint_path,
        )
        checkpoint = str(checkpoint_path)

    return (
        {
            "intime_mae": float(np.mean(intime_maes)),
            "val_mae": val_mae,
            "test_mae": test_mae,
            "L_data": L_data,
            "L_phys": L_phys,
            "L_bc": L_bc,
            "delta_s": float(model.delta.detach()) * bundle.T_span_ref,
            "src_gain": float(model.src_gain.detach()),
            "diff_gain": float(model.diff_gain.detach()),
            "n_params": sum(p.numel() for p in model.parameters()),
            "rate_lags_s": (
                np.array(model.rate_lags.detach().cpu().numpy()) * bundle.T_span_ref
            ).astype(float).tolist(),
            "checkpoint": checkpoint,
            "test_time_maes": test_time_maes,
        },
        hist,
    )


def aggregate_seeds(extra: dict, per_seed: list, n_seeds: int,
                    train_time: float) -> dict:
    """Collapse the per-seed rows of one configuration into a single result row.

    The configuration is scored by the MEAN over seeds; the standard deviation
    travels with it so a difference between two configurations can be read
    against the spread the seeds alone produce. With one seed the std is 0 by
    construction -- that is not evidence of stability, only of a single sample.
    """
    def col(key):
        return np.array([r[key] for r in per_seed], dtype=float)

    def spread(values):
        return float(values.std(ddof=1)) if len(values) > 1 else 0.0

    val, test = col("val_mae"), col("test_mae")
    first = per_seed[0]
    row = {
        "intime_mae": float(col("intime_mae").mean()),
        "val_mae": float(val.mean()),
        "test_mae": float(test.mean()),
        "val_mae_std": spread(val),
        "test_mae_std": spread(test),
        "n_seeds": int(n_seeds),
        "n_seeds_ok": len(per_seed),
        "L_data": float(col("L_data").mean()),
        "L_phys": float(col("L_phys").mean()),
        "L_bc": float(col("L_bc").mean()),
        "delta_s": first["delta_s"],
        "src_gain": float(col("src_gain").mean()),
        "diff_gain": float(col("diff_gain").mean()),
        "n_params": first["n_params"],
        "rate_lags_s": first["rate_lags_s"],
        "train_time": train_time,
        "checkpoint": first["checkpoint"],
        "test_time_maes": np.mean([r["test_time_maes"] for r in per_seed], axis=0),
    }
    row.update(extra)
    return row


def noise_verdict(usable: list, best: dict, n_seeds: int, label_of) -> list:
    """Say whether the winning configuration is distinguishable from noise.

    A sweep reports the minimum over all its configurations, so the winner is
    partly whichever one got the luckiest initialisation. Comparing the gap to
    the runner-up against the seed-to-seed spread is what turns the ranking into
    a claim one can defend.
    """
    if n_seeds < 2:
        return ["  NOTE: one seed per configuration - the ranking cannot be "
                "separated from init noise. Re-run with --seeds 0 1 2 to find out."]
    if len(usable) < 2:
        return []
    runner_up = sorted(usable, key=lambda r: r["val_mae"])[1]
    gap = runner_up["val_mae"] - best["val_mae"]
    spread = max(best["val_mae_std"], runner_up["val_mae_std"])
    if spread <= 0:
        return []
    if gap < spread:
        return [f"  WARNING: the gap to the runner-up ({label_of(runner_up)}) is "
                f"{gap:.3f} °C, smaller than the seed spread ({spread:.3f} °C).",
                "  The ranking is within noise - treat the winner as 'one of "
                "several equally good', not as the optimum."]
    return [f"  The gap to the runner-up ({gap:.3f} °C) exceeds the seed spread "
            f"({spread:.3f} °C), so the ranking is meaningful at this grid size."]


def print_eta(idx: int, total: int, start_time_total: float,
              train_time: float) -> None:
    """Report the time this configuration took and extrapolate the remainder."""
    elapsed = time.time() - start_time_total
    eta = (elapsed / idx) * (total - idx) if idx else 0.0
    print(f"  Train time: {train_time/60:.1f} min | ETA: {eta/60:.1f} min",
          flush=True)
