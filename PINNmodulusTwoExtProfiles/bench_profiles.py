"""Shared machinery for the profile benchmark (``profileBench.py``).

Same shape as ``PINNmodulusTwo/bench_common.py`` -- walk a list of
configurations, train each under one or more seeds, score by the mean over
seeds, report a set that never took part in selection -- with the two changes
the profile OPs force.

**Selection is over a SET of validation OPs, not one.** The base project ranks
on a single held-out OP. With profiles there are two different things a
configuration has to get right, and they trade off: staying accurate on the
constant OPs, and following a driver that moves. One OP can only measure one of
them. ``profileBench`` therefore selects on the MEAN validation MAE over
``--val-ops`` (OP06 for the constant case, OP09 for the profile case by
default), and every per-OP number travels with it so a configuration that wins
the mean by wrecking one of the two is visible rather than hidden. The mean is
unweighted on purpose: weighting it would be a second tuning knob smuggled into
the metric.

**Results are grouped by tier.** ``op_registry`` grades every held-out OP by
what it asks of the model, and the per-tier means are reported separately.
A single averaged "test MAE" over OP13, OP15 and OP16 would mix a C-rate
extrapolation with an unseen profile type and a 3x flow, and the average of
three different questions answers none of them.

Everything else is deliberately unchanged from ``bench_common``: a configuration
is expressed as ``overrides`` (a dict of ``train.fit`` argument names), a
failed seed is dropped rather than allowed to poison the mean with NaN, every
configuration produces exactly one row, and the seed-noise verdict decides
whether the ranking can be defended at all.
"""

from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
from data import build_op, effective_rate_scale
from op_metrics import box_errors, box_time_idx, op_metrics, rollout_phys
from op_registry import TIER_ORDER, tier_of

# History shape returned by fit(); placeholder when fit() itself failed.
EMPTY_HIST = {"epoch": [], "L_data": [], "L_phys": [], "L_bc": [],
              "L_phys_bal": [], "L_bc_bal": [], "delta": [], "aborted": True}

# The per-OP metrics carried into the CSV and the aggregation.
METRIC_KEYS = ("mae", "rmse", "max_abs_err", "peak_err", "mae_transient",
               "mae_quiescent", "late_mae")


def make_train_args(cli, overrides: dict, seed: int) -> Namespace:
    """Build the Namespace ``train.fit`` expects for one configuration.

    Every key ``fit`` reads has to be present here -- ``fit`` uses ``getattr``
    with defaults in a few places, and a silently-defaulted key is how a swept
    axis stops being swept. Axes a benchmark sweeps arrive through
    ``overrides``; axes it holds fixed arrive the same way, so nothing is ever
    inherited by accident from a CLI attribute that happens to share a name.
    """
    spec = dict(
        ops=list(cli.ops), subsample=cli.subsample, epochs=cli.epochs,
        train_frac=cli.train_frac,
        val_ops=list(cli.val_ops), test_ops=list(cli.test_ops),
        resample=cli.resample,
        driver_history=cli.driver_history,
        driver_rate_lags=list(cli.driver_rate_lags),
        k_max=cli.k_max, time_deriv=cli.time_deriv,
        history_mode=cli.history_mode, rate_lags=list(cli.rate_lags),
        delta_grid=cli.delta_grid, max_rate_amp=cli.max_rate_amp,
        width=cli.width, depth=cli.depth, lr=cli.lr,
        w_data=1.0, w_phys=0.0, w_bc=0.0,
        batch_data=cli.batch_data, batch_phys=cli.batch_phys,
        batch_bc=cli.batch_bc,
        weight_decay=cli.weight_decay, grad_clip=cli.grad_clip,
        gain_lr_mult=cli.gain_lr_mult,
        early_stopping_patience=cli.early_stopping_patience,
        phys_norm=cli.phys_norm,
        use_static=cli.use_static, use_forcing=cli.use_forcing,
        shuffle_ops=cli.shuffle_ops, holdout_tail=cli.holdout_tail,
        seed=int(seed), device=cli.device, tf32=getattr(cli, "tf32", False),
    )
    spec.update(overrides)
    return Namespace(**spec)


def _eval_op(model, op_data, bundle, device, *, late_is_holdout: bool,
             want_box: bool) -> dict:
    pred = rollout_phys(model, op_data, bundle, device)
    row = dict(op_metrics(pred, op_data, late_is_holdout=late_is_holdout))
    if want_box:
        idx = box_time_idx(op_data.n_t)
        row["box_errors"] = box_errors(pred, op_data.T_lab, idx)
        row["box_times_s"] = np.asarray(op_data.t, dtype=float)[idx]
    return row


def failed_result(extra: dict, train_time: float, cli, n_seeds: int = 1) -> dict:
    """Result row for a configuration where every seed diverged or crashed.

    Every configuration must produce exactly one row, because the per-axis plots
    pick their points by position.
    """
    nan = float("nan")
    row = {
        "n_seeds": int(n_seeds), "n_seeds_ok": 0,
        "L_data": nan, "L_phys": nan, "L_bc": nan,
        "src_gain": nan, "diff_gain": nan, "rate_scale": nan,
        "n_params": 0, "train_time": train_time, "checkpoint": "",
        "val_mae": nan, "val_mae_std": nan,
        "per_op": {}, "per_tier": {},
        # Empty rather than NaN-filled: the sensor count is not known here and a
        # diverged point is excluded from the plots anyway.
        "box_errors": np.zeros((0, 0)), "box_times_s": np.zeros(0),
        "box_op": (cli.test_ops[0] if cli.test_ops else ""),
    }
    row.update(extra)
    return row


def train_one_seed(cli, overrides: dict, seed: int, device, fit,
                   checkpoint_path: Path | None = None,
                   context: dict | None = None):
    """Train one configuration at one seed. Returns ``(row_or_None, history)``.

    ``None`` means the seed produced nothing usable -- it crashed or its loss
    diverged -- and the caller drops it instead of letting a NaN poison the
    configuration's mean.
    """
    args = make_train_args(cli, overrides, seed)
    try:
        model, bundle, _packed, dtn, hist = fit(args)
    except Exception as exc:  # one bad seed must not kill the sweep
        print(f"    seed {seed}: training failed ({exc})", flush=True)
        return None, EMPTY_HIST
    model.eval()

    L_data = float(hist["L_data"][-1]) if hist["L_data"] else float("nan")
    L_phys = float(hist["L_phys"][-1]) if hist["L_phys"] else float("nan")
    L_bc = float(hist["L_bc"][-1]) if hist["L_bc"] else float("nan")
    if not np.isfinite(L_data):
        print(f"    seed {seed}: loss diverged (L_data={L_data})", flush=True)
        return None, hist

    per_op: Dict[str, dict] = {}
    box_op = cli.test_ops[0] if cli.test_ops else (
        cli.val_ops[0] if cli.val_ops else "")

    # Training OPs: in-sample unless the run held out the tail.
    for op_data in bundle.ops:
        per_op[op_data.op_id] = _eval_op(
            model, op_data, bundle, device,
            late_is_holdout=bool(args.holdout_tail), want_box=False)
        per_op[op_data.op_id]["role"] = "train"

    # Held-out OPs. build_op reuses the bundle's constants, so nothing about the
    # preprocessing is refitted on data the model must not have seen.
    box_errors_arr = np.zeros((0, 0))
    box_times = np.zeros(0)
    for role, op_ids in (("val", list(cli.val_ops)), ("test", list(cli.test_ops))):
        for op_id in op_ids:
            op_data = build_op(op_id, bundle, subsample_time=cli.subsample,
                               train_frac=cli.train_frac)
            row = _eval_op(model, op_data, bundle, device,
                           late_is_holdout=True, want_box=(op_id == box_op))
            row["role"] = role
            if op_id == box_op:
                box_errors_arr = row.pop("box_errors")
                box_times = row.pop("box_times_s")
            per_op[op_id] = row

    val_mae = float(np.mean([per_op[o]["mae"] for o in cli.val_ops])) \
        if cli.val_ops else float("nan")

    checkpoint = ""
    if checkpoint_path is not None:
        rate_scale, _ = effective_rate_scale(
            bundle.dTdt_scale,
            [float(v) / bundle.T_span_ref for v in args.rate_lags],
            float(args.max_rate_amp))
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
                    "rate_scale": float(rate_scale),
                    "delta_grid": float(args.delta_grid) / bundle.T_span_ref,
                    "use_autograd_time": (args.time_deriv == "autograd"),
                },
                # Everything a checkpoint needs to be reloadable INCLUDING the
                # preprocessing: a profile model restored with a different
                # resample mode or driver-rate layout is silently a different
                # model, and the weights give no hint of it.
                "bundle_stats": {
                    "T_mu": float(bundle.T_mu),
                    "T_sigma": float(bundle.T_sigma),
                    "T_span_ref": float(bundle.T_span_ref),
                    "q_mu": float(bundle.q_mu),
                    "q_sigma": float(bundle.q_sigma),
                    "config_mu": bundle.config_mu.tolist(),
                    "config_sigma": bundle.config_sigma.tolist(),
                    "config_active": bundle.config_active.tolist(),
                    "driver_rate_rms": bundle.driver_rate_rms.tolist(),
                    "driver_rate_active": bundle.driver_rate_active.tolist(),
                },
                "preprocessing": {
                    "resample": bundle.resample,
                    "use_driver_history": bundle.use_driver_history,
                    "driver_rate_lags": list(bundle.driver_rate_lags),
                    "driver_names": list(bundle.driver_names),
                    "subsample": int(args.subsample),
                    "train_frac": float(args.train_frac),
                },
                "benchmark_context": {
                    **(context or {}),
                    "ops": list(args.ops),
                    "val_ops": list(cli.val_ops),
                    "test_ops": list(cli.test_ops),
                    "epochs": int(args.epochs),
                    "seed": int(seed),
                    "holdout_tail": bool(args.holdout_tail),
                },
            },
            checkpoint_path,
        )
        checkpoint = str(checkpoint_path)

    return (
        {
            "val_mae": val_mae,
            "per_op": per_op,
            "L_data": L_data, "L_phys": L_phys, "L_bc": L_bc,
            "src_gain": float(model.src_gain.detach()),
            "diff_gain": float(model.diff_gain.detach()),
            "rate_scale": float(model.rate_scale),
            "n_params": sum(p.numel() for p in model.parameters()),
            "checkpoint": checkpoint,
            "box_errors": box_errors_arr,
            "box_times_s": box_times,
            "box_op": box_op,
        },
        hist,
    )


def tier_means(per_op: Dict[str, dict], roles=("val", "test")) -> Dict[str, dict]:
    """Mean of each metric over the held-out OPs of each tier.

    Training OPs are excluded: their numbers are in-sample unless the run used
    ``--holdout-tail``, and averaging them into a tier would put an in-sample
    number next to held-out ones under one heading.
    """
    out: Dict[str, dict] = {}
    for tier in TIER_ORDER:
        members = [o for o, r in per_op.items()
                   if r.get("role") in roles and tier_of(o) == tier]
        if not members:
            continue
        out[tier] = {
            k: float(np.nanmean([per_op[o][k] for o in members]))
            for k in METRIC_KEYS
        }
        out[tier]["ops"] = members
    return out


def aggregate_seeds(extra: dict, per_seed: List[dict], n_seeds: int,
                    train_time: float, cli) -> dict:
    """Collapse the per-seed rows of one configuration into a single row.

    The configuration is scored by the MEAN over seeds; the standard deviation
    travels with it so a difference between two configurations can be read
    against the spread the seeds alone produce. With one seed the std is 0 by
    construction -- that is not evidence of stability, only of a single sample.
    """
    def col(key):
        return np.array([r[key] for r in per_seed], dtype=float)

    def spread(values):
        return float(values.std(ddof=1)) if len(values) > 1 else 0.0

    first = per_seed[0]
    op_ids = list(first["per_op"])
    per_op = {}
    for op_id in op_ids:
        per_op[op_id] = {"role": first["per_op"][op_id]["role"]}
        for k in METRIC_KEYS:
            vals = np.array([r["per_op"][op_id][k] for r in per_seed], float)
            per_op[op_id][k] = float(np.nanmean(vals))
            per_op[op_id][k + "_std"] = spread(vals)

    val = col("val_mae")
    boxes = [r["box_errors"] for r in per_seed if r["box_errors"].size]
    row = {
        "val_mae": float(val.mean()),
        "val_mae_std": spread(val),
        "n_seeds": int(n_seeds),
        "n_seeds_ok": len(per_seed),
        "L_data": float(col("L_data").mean()),
        "L_phys": float(col("L_phys").mean()),
        "L_bc": float(col("L_bc").mean()),
        "src_gain": float(col("src_gain").mean()),
        "diff_gain": float(col("diff_gain").mean()),
        "rate_scale": first["rate_scale"],
        "n_params": first["n_params"],
        "train_time": train_time,
        "checkpoint": first["checkpoint"],
        "per_op": per_op,
        "per_tier": tier_means(per_op),
        "box_errors": (np.mean(boxes, axis=0) if boxes else np.zeros((0, 0))),
        "box_times_s": first["box_times_s"],
        "box_op": first["box_op"],
    }
    row.update(extra)
    return row


def noise_verdict(usable: List[dict], best: dict, n_seeds: int, label_of) -> List[str]:
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
                f"{gap:.3f} C, smaller than the seed spread ({spread:.3f} C).",
                "  The ranking is within noise - treat the winner as 'one of "
                "several equally good', not as the optimum."]
    return [f"  The gap to the runner-up ({gap:.3f} C) exceeds the seed spread "
            f"({spread:.3f} C), so the ranking is meaningful at this grid size."]


def split_verdict(best: dict, val_ops: Sequence[str]) -> List[str]:
    """Flag a winner that bought its mean validation score from one OP.

    Selection uses the unweighted mean over ``--val-ops``, which is fair only
    while the configuration is reasonable on both. A configuration that halves
    the error on the constant OP and doubles it on the profile OP can still win
    that mean, and it is the wrong answer to the question this extension exists
    to ask.
    """
    if len(val_ops) < 2:
        return []
    maes = {o: best["per_op"][o]["mae"] for o in val_ops if o in best["per_op"]}
    if len(maes) < 2:
        return []
    worst, bestv = max(maes, key=maes.get), min(maes, key=maes.get)
    if maes[bestv] <= 0:
        return []
    ratio = maes[worst] / maes[bestv]
    line = "  selection MAE per val OP: " + ", ".join(
        f"{o} ({tier_of(o)}) {maes[o]:.3f} C" for o in val_ops if o in maes)
    if ratio > 3.0:
        return [line,
                f"  WARNING: {worst} is {ratio:.1f}x worse than {bestv}. The "
                f"winning configuration is carried by one half of the selection "
                f"set; check the per-OP columns before adopting it."]
    return [line]


def print_eta(idx: int, total: int, start_time_total: float,
              train_time: float) -> None:
    """Report the time this configuration took and extrapolate the remainder."""
    elapsed = time.time() - start_time_total
    eta = (elapsed / idx) * (total - idx) if idx else 0.0
    print(f"  Train time: {train_time/60:.1f} min | ETA: {eta/60:.1f} min",
          flush=True)
