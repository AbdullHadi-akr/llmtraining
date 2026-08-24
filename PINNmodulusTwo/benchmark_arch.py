#!/usr/bin/env python3
"""Benchmark: architecture (width, depth) and history lags.

Why this exists
---------------
The loss-weight sweep spends its whole budget on two numbers while width, depth
and the history lags stay fixed at values nobody measured. If the weight sweep
turns out to rank within seed noise, that budget bought nothing; the same hours
spent here answer questions that were never asked.

Axes: width, depth, lags (the rate segments) and dgrid (the anchor lag of the
hybrid history -- where the whole history block is rooted).

This benchmark walks one axis at a time rather than a full product grid: the
point is to find out which knob moves the error at all, not to optimise all of
them jointly. A joint grid over all four axes is several hundred trainings; one
axis at a time is sixteen and tells you where the leverage is.

Everything not on the swept axis is held at the CLI baseline, so every
configuration is compared against the same reference point.

RUNTIME: one configuration at 60 epochs costs 1.5-2.5 h (see
benchmark_wphys_wbc.py for why), so the 16-configuration default is 1-1.5 days
per seed. --epochs 20 thirds that and is usually enough to see which axis moves
the error. Use --axes to walk one axis at a time, and read the measured
seconds-per-epoch the training log prints before planning a long run.

Run:
    python3 PINNmodulusTwo/benchmark_arch.py --device cuda --seeds 0 1 2

    # only the history layout, at the weights the loss sweep picked
    python3 PINNmodulusTwo/benchmark_arch.py --axes lags dgrid \
        --w-phys 0.05 --w-bc 0.1 --epochs 20

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_arch.csv       - one row per configuration, mean over seeds + std
    benchmark_arch.png       - one panel per axis, MAE with seed spread as error bars
    benchmark_arch_best.txt  - ranking, baseline comparison and the noise verdict
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bench_common import (
    aggregate_seeds, failed_result, noise_verdict, print_eta, train_one_seed,
)
from data import require_ops
from device_utils import resolve_device
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Swept values per axis. Each is explored against the CLI baseline for the others.
DEFAULT_WIDTHS = [64, 128, 256]
DEFAULT_DEPTHS = [2, 3, 4, 6]
# Hybrid history: cumulative segment lengths in seconds. The first entry is the
# current default, the rest probe shorter and longer memory.
DEFAULT_DELTA_GRIDS = [0.2, 0.5, 1.0, 2.0]
DEFAULT_LAG_SETS = [
    [5.0, 20.0],
    [2.0, 10.0],
    [10.0, 60.0],
    [5.0, 20.0, 60.0],
    [30.0],
]


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
    p = argparse.ArgumentParser(description="Architecture and history-lag benchmark")
    # Data
    p.add_argument("--ops", nargs="+",
                   default=["OP01", "OP02", "OP03", "OP04", "OP05"])
    p.add_argument("--val-op", default="OP06",
                   help="OP used to SELECT the best configuration")
    p.add_argument("--test-op", default="OP07",
                   help="OP used only to REPORT; never selected on")
    p.add_argument("--subsample", type=int, default=2,
                   help="CFL-stable default: 2 -> dt=0.2s")
    # What to sweep
    p.add_argument("--axes", nargs="+",
                   choices=["width", "depth", "lags", "dgrid"],
                   default=["width", "depth", "lags", "dgrid"],
                   help="which axes to walk; each is explored against the baseline")
    p.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    p.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    p.add_argument("--delta-grids", type=float, nargs="+",
                   default=DEFAULT_DELTA_GRIDS,
                   help="anchor lags in seconds to try on the dgrid axis")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one training run per seed per configuration; each "
                        "configuration is scored by the MEAN over seeds")
    # Baseline held fixed off-axis
    p.add_argument("--width", type=int, default=128, help="baseline MLP width")
    p.add_argument("--depth", type=int, default=4, help="baseline MLP depth")
    p.add_argument("--rate-lags", nargs="+", type=float, default=[5.0, 20.0],
                   help="baseline hybrid rate segments in seconds")
    p.add_argument("--delta-grid", type=float, default=0.2,
                   help="baseline anchor lag in seconds")
    p.add_argument("--w-phys", type=float, default=0.05,
                   help="physics weight, FIXED here (sweep it with benchmark_wphys_wbc)")
    p.add_argument("--w-bc", type=float, default=0.1, help="BC weight, FIXED here")
    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gain-lr-mult", type=float, default=25.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--k-max", type=int, default=2,
                   help="raw-mode lag count; ignored in hybrid mode")
    p.add_argument("--history-mode", choices=["raw", "hybrid"], default="hybrid")
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default="bdf2")
    p.add_argument("--use-static", action="store_true", default=True)
    p.add_argument("--use-forcing", action="store_true", default=True)
    p.add_argument("--batch-data", type=int, default=2048)
    p.add_argument("--batch-phys", type=int, default=256)
    p.add_argument("--batch-bc", type=int, default=128)
    p.add_argument("--phys-norm", type=float, default=0.0)
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N")
    return p.parse_args()


def _lag_label(lags) -> str:
    return "+".join(f"{v:g}" for v in lags)


def build_configs(cli) -> list:
    """One entry per configuration: (axis, label, overrides).

    The baseline appears once per axis it sits on, which is deliberate: each
    axis panel needs its own reference point, and re-training it per axis also
    gives a free read on how much the seeds alone move the same configuration.
    """
    base_lags = list(cli.rate_lags)
    base = {"w_phys": float(cli.w_phys), "w_bc": float(cli.w_bc),
            "delta_grid": float(cli.delta_grid)}
    weights = base
    configs = []
    if "width" in cli.axes:
        for w in cli.widths:
            configs.append(("width", str(w),
                            {"width": int(w), "depth": cli.depth,
                             "rate_lags": base_lags, **weights}))
    if "depth" in cli.axes:
        for dpt in cli.depths:
            configs.append(("depth", str(dpt),
                            {"width": cli.width, "depth": int(dpt),
                             "rate_lags": base_lags, **weights}))
    if "lags" in cli.axes:
        for lags in DEFAULT_LAG_SETS:
            configs.append(("lags", _lag_label(lags),
                            {"width": cli.width, "depth": cli.depth,
                             "rate_lags": [float(v) for v in lags], **weights}))
    if "dgrid" in cli.axes:
        for dg in cli.delta_grids:
            ov = {"width": cli.width, "depth": cli.depth,
                  "rate_lags": base_lags, **weights}
            ov["delta_grid"] = float(dg)
            configs.append(("dgrid", f"{float(dg):g}", ov))
    return configs


def _is_baseline(row, cli) -> bool:
    if row["axis"] == "width":
        return row["label"] == str(cli.width)
    if row["axis"] == "depth":
        return row["label"] == str(cli.depth)
    if row["axis"] == "dgrid":
        return row["label"] == f"{float(cli.delta_grid):g}"
    return row["label"] == _lag_label(cli.rate_lags)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli = parse_args()
    require_ops(*cli.ops, cli.val_op, cli.test_op)
    device = resolve_device(cli.device)
    cli.device = str(device)

    configs = build_configs(cli)
    total = len(configs)
    n_runs = total * len(cli.seeds)

    header = [
        "Architecture / history-lag benchmark (free-running rollout, NO teacher forcing)",
        f"train = {'+'.join(cli.ops)}   val (selection) = {cli.val_op}   "
        f"test (report only) = {cli.test_op}",
        f"baseline: width={cli.width} depth={cli.depth} "
        f"rate_lags={_lag_label(cli.rate_lags)}s",
        f"loss weights held FIXED: w_phys={cli.w_phys}  w_bc={cli.w_bc}",
        f"axes = {', '.join(cli.axes)}   seeds = {cli.seeds}",
        f"runs = {total} configurations x {len(cli.seeds)} seed(s) = {n_runs} trainings",
        f"epochs = {cli.epochs}  dt = {0.1*cli.subsample:.1f}s  lr = {cli.lr}",
        "",
    ]
    print("\n".join(header), flush=True)

    results = []
    start_time_total = time.time()

    for idx, (axis, label, overrides) in enumerate(configs, start=1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {axis}={label}"
              f"  ({len(cli.seeds)} seed{'s' if len(cli.seeds) > 1 else ''})")
        print(f"{'='*60}", flush=True)
        start = time.time()

        extra = {"axis": axis, "label": label,
                 "width": overrides["width"], "depth": overrides["depth"],
                 "lags_s": _lag_label(overrides["rate_lags"]),
                 "dgrid_s": float(overrides["delta_grid"])}

        per_seed = []
        for seed in cli.seeds:
            row, _hist = train_one_seed(
                cli, overrides, seed, device, fit,
                context={"axis": axis, "label": label,
                         "w_phys": float(cli.w_phys), "w_bc": float(cli.w_bc)},
            )
            if row is not None:
                per_seed.append(row)

        train_time = time.time() - start
        if not per_seed:
            print(f"  [SKIP] {axis}={label}: every seed diverged or crashed - "
                  f"recorded as NaN, sweep continues", flush=True)
            results.append(failed_result(extra, train_time, len(cli.seeds)))
            print_eta(idx, total, start_time_total, train_time)
            continue

        row = aggregate_seeds(extra, per_seed, len(cli.seeds), train_time)
        results.append(row)
        spread = (f"  (+/-{row['val_mae_std']:.3f} val)"
                  if row["n_seeds_ok"] > 1 else "")
        print(f"  params={row['n_params']}  "
              f"MAE(val {cli.val_op})={row['val_mae']:.3f}°C  "
              f"MAE(test {cli.test_op})={row['test_mae']:.3f}°C{spread}", flush=True)
        print_eta(idx, total, start_time_total, train_time)

    total_time = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n", flush=True)

    # ---- CSV ----------------------------------------------------------------
    csv_lines = [
        "axis,label,width,depth,lags_s,dgrid_s,n_params,L_data,L_phys,L_bc,"
        "MAE_in_C,MAE_val_C,MAE_val_std_C,MAE_test_C,MAE_test_std_C,"
        "n_seeds,n_seeds_ok,src_gain,diff_gain,train_time_min"
    ]
    for r in results:
        csv_lines.append(
            f"{r['axis']},{r['label']},{r['width']},{r['depth']},\"{r['lags_s']}\","
            f"{r['dgrid_s']:g},"
            f"{r['n_params']},{r['L_data']:.6f},{r['L_phys']:.6f},{r['L_bc']:.6f},"
            f"{r['intime_mae']:.4f},{r['val_mae']:.4f},{r['val_mae_std']:.4f},"
            f"{r['test_mae']:.4f},{r['test_mae_std']:.4f},"
            f"{r['n_seeds']},{r['n_seeds_ok']},"
            f"{r['src_gain']:.6f},{r['diff_gain']:.6f},{r['train_time']/60:.2f}"
        )
    (ART_DIR / "benchmark_arch.csv").write_text("\n".join(csv_lines) + "\n")

    usable = [r for r in results if np.isfinite(r["val_mae"])]
    n_failed = len(results) - len(usable)
    if not usable:
        print(f"All {len(results)} configurations diverged - no result to rank.",
              flush=True)
        print(f"Raw values are in {ART_DIR / 'benchmark_arch.csv'}.", flush=True)
        return

    best = min(usable, key=lambda r: r["val_mae"])
    label_of = lambda r: f"{r['axis']}={r['label']}"

    th = (f"{'axis':>6} {'value':>12} {'params':>8} | {'MAE_in':>7} "
          f"{'MAE_val':>8} {'+/-':>6} {'MAE_test':>9} {'+/-':>6}")
    summary = header + [th, "-" * len(th)]
    for r in results:
        summary.append(
            f"{r['axis']:>6} {r['label']:>12} {r['n_params']:>8} | "
            f"{r['intime_mae']:>7.3f} {r['val_mae']:>8.3f} {r['val_mae_std']:>6.3f} "
            f"{r['test_mae']:>9.3f} {r['test_mae_std']:>6.3f}"
        )

    # How much does each axis move the error at all? An axis whose whole span is
    # narrower than the seed spread is not worth tuning further.
    summary += ["", "Span per axis (max - min validation MAE over the axis):"]
    for axis in cli.axes:
        vals = [r["val_mae"] for r in usable if r["axis"] == axis]
        if len(vals) < 2:
            continue
        span = max(vals) - min(vals)
        noise = max((r["val_mae_std"] for r in usable if r["axis"] == axis),
                    default=0.0)
        verdict = ("below the seed spread - this knob does not matter here"
                   if len(cli.seeds) > 1 and span < noise
                   else "worth tuning" if len(cli.seeds) > 1
                   else "seed spread unknown (single seed)")
        summary.append(f"  {axis:>6}: {span:6.3f} °C   ({verdict})")

    baseline = next((r for r in usable if _is_baseline(r, cli)), None)
    summary += [
        "",
        "MAE = mean |true - predicted| (°C) from free-running rollout.",
        f"Selection ran on {cli.val_op}; {cli.test_op} was never used to choose.",
        f"BEST (by MAE_val): {label_of(best)}  (width={best['width']}, "
        f"depth={best['depth']}, lags={best['lags_s']}s, {best['n_params']} params)",
        f"  -> val {best['val_mae']:.3f}°C, test {best['test_mae']:.3f}°C, "
        f"in-time {best['intime_mae']:.3f}°C",
    ]
    if baseline is not None and baseline is not best:
        summary.append(
            f"  baseline {label_of(baseline)}: val {baseline['val_mae']:.3f}°C "
            f"-> improvement {baseline['val_mae'] - best['val_mae']:.3f}°C"
        )
    summary += noise_verdict(usable, best, len(cli.seeds), label_of)
    summary.append(f"Total runtime: {total_time/3600:.2f} hours "
                   f"({total_time/60:.1f} min)")
    if n_failed:
        summary.append(f"Diverged (recorded as NaN, excluded): "
                       f"{n_failed}/{len(results)}")
    (ART_DIR / "benchmark_arch_best.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[len(header):]), flush=True)

    # ---- one panel per axis -------------------------------------------------
    axes_present = [a for a in cli.axes
                    if any(r["axis"] == a for r in results)]
    if not axes_present:
        return
    fig, panels = plt.subplots(1, len(axes_present),
                               figsize=(5.2 * len(axes_present), 4.4),
                               squeeze=False)
    for col, axis in enumerate(axes_present):
        rows = [r for r in results if r["axis"] == axis]
        ax = panels[0][col]
        x = np.arange(len(rows))
        val = [r["val_mae"] for r in rows]
        test = [r["test_mae"] for r in rows]
        ax.errorbar(x, val, yerr=[r["val_mae_std"] for r in rows], fmt="o-",
                    capsize=4, label=f"val ({cli.val_op})", color="#1f77b4")
        ax.errorbar(x, test, yerr=[r["test_mae_std"] for r in rows], fmt="s--",
                    capsize=4, label=f"test ({cli.test_op})", color="#b00020",
                    alpha=0.8)
        for i, r in enumerate(rows):
            if _is_baseline(r, cli):
                ax.axvline(i, color="gray", ls=":", lw=1.5, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right")
        ax.set_xlabel(axis)
        ax.set_ylabel("rollout MAE [°C]")
        ax.set_title(f"{axis} (dotted = baseline)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Architecture sweep — {len(cli.seeds)} seed(s), error bars = seed std",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ART_DIR / "benchmark_arch.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {ART_DIR / 'benchmark_arch.png'}", flush=True)


if __name__ == "__main__":
    main()
