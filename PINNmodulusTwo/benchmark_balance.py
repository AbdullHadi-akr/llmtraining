#!/usr/bin/env python3
"""Benchmark: how the loss terms are balanced, and which input features help.

Why this runs BEFORE the loss-weight sweep
------------------------------------------
``w_phys`` and ``w_bc`` only mean something relative to how the terms they
weight have been scaled. Under the historical balancing (``--loss-balance
legacy``) ``L_phys`` and ``L_bc`` are divided by their own running average while
``L_data`` is not, so the two normalised terms sit near 1 for the whole run
while ``L_data`` falls by orders of magnitude. The mixture the optimiser
actually sees therefore drifts steadily towards physics as training proceeds,
and the best ``w_phys`` becomes a function of ``--epochs``: a weight tuned in a
20-epoch probe is too large for a 60-epoch run.

Sweeping the weights on top of that measures the drift as much as the weights.
This benchmark settles the balancing first, on a single fixed weight point, and
only then is a weight sweep worth its hours.

Axes
----
balance   ema | legacy | fixed        -- which terms get divided by their own
                                        magnitude, and whether the divisor keeps
                                        tracking or is frozen after warm-up.
resnorm   rms | legacy                -- whether each residual term is divided by
                                        its training RMS (unit scale) or by the
                                        square root of it, which leaves the three
                                        terms with their original size gap.
feats     none | energy | rates | both -- extra input channels: cumulative
                                        injected heat, and d(config)/dt for the
                                        config channels that are real profiles.

RUNTIME: one 20-epoch configuration on five OPs costs ~40 min, so each part
below stays under ~4 h. Read the measured seconds-per-epoch from chapter 6.3
before planning; --part exists so a session never has to run longer than that.

Run (two sessions):
    python3 PINNmodulusTwo/benchmark_balance.py --part 1 --epochs 20 --device cuda
    # then, with the winning balance from part 1:
    python3 PINNmodulusTwo/benchmark_balance.py --part 2 --epochs 20 --device cuda \
        --loss-balance ema --residual-norm rms

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_balance.csv       - one row per configuration, mean over seeds + std
    benchmark_balance.png       - one panel per axis, MAE with seed spread
    benchmark_balance_ratio.png - w*L_term / w_data*L_data per epoch, per config
    benchmark_balance_best.txt  - ranking, per-axis span and the noise verdict
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench_common import (
    EMPTY_HIST, aggregate_seeds, failed_result, noise_verdict, print_eta,
    train_one_seed,
)
from data import require_ops
from device_utils import resolve_device
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = "balance_parts.json"

BALANCE_MODES = ["ema", "legacy", "fixed"]
RESIDUAL_NORMS = ["rms", "legacy"]
FEATURE_SETS = {
    "none":   {"forcing_energy": False, "config_rates": False},
    "energy": {"forcing_energy": True, "config_rates": False},
    "rates":  {"forcing_energy": False, "config_rates": True},
    "both":   {"forcing_energy": True, "config_rates": True},
}
# Part 1 answers "how should the terms be scaled", part 2 "which inputs help".
# Splitting them is not only about session length: part 2 is only meaningful
# once part 1 has picked a balance, because otherwise a feature would be
# measured through a mixture that drifts differently for every configuration.
PART_AXES = {"1": ["balance", "resnorm"], "2": ["feats"], "all": ["balance", "resnorm", "feats"]}


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
    p = argparse.ArgumentParser(description="Loss-balancing and input-feature benchmark")
    p.add_argument("--ops", nargs="+",
                   default=d.get("ops", ["OP01", "OP02", "OP03", "OP04", "OP05"]))
    p.add_argument("--val-op", default=d.get("val_op", "OP06"),
                   help="OP used to SELECT the best configuration")
    p.add_argument("--test-op", default=d.get("test_op", "OP07"),
                   help="OP reported but never used for selection")
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 2))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one seed per configuration is enough to see a large "
                        "effect; 3 seeds are needed before a small one is real")
    p.add_argument("--part", choices=["1", "2", "all"], default="all",
                   help="1 = balance + residual norm (~5 trainings), "
                        "2 = input features (~4), all = both in one session")

    # Fixed point the axes are measured at. NOT swept here on purpose: the whole
    # point is to hold the weights still while the scaling underneath them moves.
    p.add_argument("--w-phys", type=float, default=d.get("w_phys", 0.1))
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))

    # Baselines for the axes not being swept in this part.
    p.add_argument("--loss-balance", choices=BALANCE_MODES,
                   default=d.get("loss_balance", "ema"))
    p.add_argument("--residual-norm", choices=RESIDUAL_NORMS,
                   default=d.get("residual_norm", "rms"))
    p.add_argument("--ema-decay", type=float, default=d.get("ema_decay", 0.9))
    p.add_argument("--balance-warmup", type=int, default=d.get("balance_warmup", 1))
    p.add_argument("--data-floor", type=float, default=d.get("data_floor", 1e-8))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--bc-norm", type=float, default=d.get("bc_norm", 0.0))
    p.add_argument("--zero-weight-terms", choices=["skip", "compute"],
                   default=d.get("zero_weight_terms", "skip"))
    p.add_argument("--subsample-mode", choices=["stride", "mean"],
                   default=d.get("subsample_mode", "stride"))
    p.add_argument("--forcing-energy", action="store_true",
                   default=d.get("forcing_energy", False))
    p.add_argument("--config-rates", action="store_true",
                   default=d.get("config_rates", False))

    # Held fixed, matching config.yaml so this is comparable with the other sweeps.
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "hybrid"))
    p.add_argument("--rate-lags", type=float, nargs="+",
                   default=d.get("rate_lags", [200.0, 600.0]))
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2))
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0))
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", True))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", True))
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N")
    return p.parse_args()


def build_configs(cli) -> list:
    """One entry per configuration: ``(axis, label, overrides)``.

    The CLI baseline is re-trained once per axis it appears on, exactly as
    ``benchmark_arch.py`` does: each panel needs its own reference point, and the
    repeat is a free read on how much the same configuration moves between axes.
    """
    axes = PART_AXES[cli.part]
    base = {
        "w_phys": float(cli.w_phys), "w_bc": float(cli.w_bc),
        "rate_lags": [float(v) for v in cli.rate_lags],
        "delta_grid": float(cli.delta_grid),
        "loss_balance": cli.loss_balance, "residual_norm": cli.residual_norm,
        "forcing_energy": bool(cli.forcing_energy),
        "config_rates": bool(cli.config_rates),
    }
    configs = []
    if "balance" in axes:
        for mode in BALANCE_MODES:
            configs.append(("balance", mode, {**base, "loss_balance": mode}))
    if "resnorm" in axes:
        for rn in RESIDUAL_NORMS:
            configs.append(("resnorm", rn, {**base, "residual_norm": rn}))
    if "feats" in axes:
        for name, flags in FEATURE_SETS.items():
            configs.append(("feats", name, {**base, **flags}))
    return configs


def _is_baseline(row, cli) -> bool:
    if row["axis"] == "balance":
        return row["label"] == cli.loss_balance
    if row["axis"] == "resnorm":
        return row["label"] == cli.residual_norm
    return row["label"] == ("both" if cli.forcing_energy and cli.config_rates
                            else "energy" if cli.forcing_energy
                            else "rates" if cli.config_rates else "none")


def _ratio_summary(hist) -> tuple[float, float, float]:
    """(first, last, drift) of the physics-to-data contribution ratio.

    Two different things are being reported here and both matter:

    * the LEVEL (``first``, ``last``). This is what ``w_phys`` nominally sets.
      Under ``ema`` it starts at ``w_phys/w_data`` by construction; under
      ``legacy`` it starts at ``w_phys/(w_data * L_data)``, so the same
      ``w_phys`` can mean something orders of magnitude different.
    * the DRIFT (``last/first``): how far the mixture wanders during the run.
      Near 1 a weight tuned in a short probe transfers to a long run; far from
      1 it does not.

    Epoch 1 is deliberately excluded from the baseline when there is more than
    one epoch: the first free-running rollout starts from an untrained network
    and its ``L_data`` can be many orders of magnitude above the converged
    value. Anchoring the drift there would measure that transient, not the
    balancing.
    """
    vals = [v for v in hist.get("ratio_phys", []) if np.isfinite(v) and v > 0]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    first = vals[1] if len(vals) > 2 else vals[0]
    return first, vals[-1], vals[-1] / first


def _merge_parts(cli, results, histories) -> tuple[list, dict]:
    """Persist this part's rows and merge with any earlier part.

    Same idea as the weight probe's ``probe_parts.json``: a session that only
    ran part 1 must not lose its five trainings when part 2 runs a day later.
    Rows are keyed by (axis, label), so re-running a part replaces its own rows
    and leaves the other part's alone.
    """
    path = ART_DIR / STATE_FILE
    signature = {
        "ops": list(cli.ops), "val_op": cli.val_op, "test_op": cli.test_op,
        "epochs": int(cli.epochs), "subsample": int(cli.subsample),
        "seeds": [int(s) for s in cli.seeds],
        "w_phys": float(cli.w_phys), "w_bc": float(cli.w_bc),
        "width": int(cli.width), "depth": int(cli.depth),
    }
    state = {"signature": signature, "rows": {}, "ratios": {}}
    if path.exists():
        try:
            old = json.loads(path.read_text())
            # Settings must match or the parts are not comparable; a mismatch
            # discards the stored part rather than quietly mixing two setups.
            if old.get("signature") == signature:
                state = old
            else:
                print("  [balance] stored part(s) ran with different settings - "
                      "discarding them; the other part has to be re-run to match "
                      "this one.", flush=True)
        except Exception:
            pass

    for row, hist in zip(results, histories):
        key = f"{row['axis']}|{row['label']}"
        state["rows"][key] = {k: v for k, v in row.items()
                              if not isinstance(v, np.ndarray)}
        state["ratios"][key] = {
            "ratio_phys": [float(v) for v in hist.get("ratio_phys", [])],
            "ratio_bc": [float(v) for v in hist.get("ratio_bc", [])],
            "L_data": [float(v) for v in hist.get("L_data", [])],
        }
    path.write_text(json.dumps(state, indent=1))
    print(f"  [balance] part {cli.part} stored in {path}", flush=True)
    return list(state["rows"].values()), state["ratios"]


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
        "Loss-balancing / input-feature benchmark "
        "(free-running rollout, NO teacher forcing)",
        f"train = {'+'.join(cli.ops)}   val (selection) = {cli.val_op}   "
        f"test (report only) = {cli.test_op}",
        f"loss weights held FIXED: w_phys={cli.w_phys}  w_bc={cli.w_bc}  "
        f"(this benchmark moves the SCALING under them, not the weights)",
        f"baseline: loss_balance={cli.loss_balance} "
        f"residual_norm={cli.residual_norm} ema_decay={cli.ema_decay}",
        f"part = {cli.part} -> axes {', '.join(PART_AXES[cli.part])}   "
        f"seeds = {cli.seeds}",
        f"runs = {total} configurations x {len(cli.seeds)} seed(s) = {n_runs} trainings",
        f"epochs = {cli.epochs}  dt = {0.1*cli.subsample:.1f}s  lr = {cli.lr}",
        "",
    ]
    print("\n".join(header), flush=True)

    results, histories = [], []
    start_time_total = time.time()

    for idx, (axis, label, overrides) in enumerate(configs, start=1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {axis}={label}"
              f"  ({len(cli.seeds)} seed{'s' if len(cli.seeds) > 1 else ''})")
        print(f"{'='*60}", flush=True)
        start = time.time()

        extra = {"axis": axis, "label": label,
                 "loss_balance": overrides["loss_balance"],
                 "residual_norm": overrides["residual_norm"],
                 "forcing_energy": int(overrides["forcing_energy"]),
                 "config_rates": int(overrides["config_rates"])}

        per_seed, first_hist = [], EMPTY_HIST
        for seed in cli.seeds:
            row, hist = train_one_seed(
                cli, overrides, seed, device, fit,
                context={"axis": axis, "label": label},
            )
            if row is not None:
                per_seed.append(row)
                if first_hist is EMPTY_HIST:
                    first_hist = hist

        train_time = time.time() - start
        if not per_seed:
            print(f"  [SKIP] {axis}={label}: every seed diverged or crashed - "
                  f"recorded as NaN, sweep continues", flush=True)
            results.append(failed_result(extra, train_time, len(cli.seeds)))
            histories.append(EMPTY_HIST)
            print_eta(idx, total, start_time_total, train_time)
            continue

        row = aggregate_seeds(extra, per_seed, len(cli.seeds), train_time)
        r0, r1, drift = _ratio_summary(first_hist)
        row.update({"ratio_first": r0, "ratio_last": r1, "ratio_drift": drift})
        results.append(row)
        histories.append(first_hist)
        spread = (f"  (+/-{row['val_mae_std']:.3f} val)"
                  if row["n_seeds_ok"] > 1 else "")
        print(f"  MAE(val {cli.val_op})={row['val_mae']:.3f}°C  "
              f"MAE(test {cli.test_op})={row['test_mae']:.3f}°C{spread}", flush=True)
        print(f"  phys/data contribution: epoch 1 = {r0:.3g}  "
              f"epoch {cli.epochs} = {r1:.3g}  drift = {drift:.3g}x", flush=True)
        print_eta(idx, total, start_time_total, train_time)

    total_time = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n", flush=True)

    merged, ratios = _merge_parts(cli, results, histories)

    # ---- CSV ----------------------------------------------------------------
    csv_lines = [
        "axis,label,loss_balance,residual_norm,forcing_energy,config_rates,"
        "n_params,L_data,L_phys,L_bc,MAE_in_C,MAE_val_C,MAE_val_std_C,"
        "MAE_test_C,MAE_test_std_C,n_seeds,n_seeds_ok,"
        "ratio_first,ratio_last,ratio_drift,train_time_min"
    ]
    for r in merged:
        csv_lines.append(
            f"{r['axis']},{r['label']},{r.get('loss_balance','')},"
            f"{r.get('residual_norm','')},{r.get('forcing_energy',0)},"
            f"{r.get('config_rates',0)},"
            f"{r['n_params']},{r['L_data']:.6f},{r['L_phys']:.6f},{r['L_bc']:.6f},"
            f"{r['intime_mae']:.4f},{r['val_mae']:.4f},{r['val_mae_std']:.4f},"
            f"{r['test_mae']:.4f},{r['test_mae_std']:.4f},"
            f"{r['n_seeds']},{r['n_seeds_ok']},"
            f"{r.get('ratio_first', float('nan')):.6g},"
            f"{r.get('ratio_last', float('nan')):.6g},"
            f"{r.get('ratio_drift', float('nan')):.6g},"
            f"{r['train_time']/60:.2f}"
        )
    (ART_DIR / "benchmark_balance.csv").write_text("\n".join(csv_lines) + "\n")

    usable = [r for r in merged if np.isfinite(r["val_mae"])]
    if not usable:
        print(f"All {len(merged)} configurations diverged - no result to rank.",
              flush=True)
        return

    best = min(usable, key=lambda r: r["val_mae"])
    label_of = lambda r: f"{r['axis']}={r['label']}"

    th = (f"{'axis':>8} {'value':>8} | {'MAE_in':>7} {'MAE_val':>8} {'+/-':>6} "
          f"{'MAE_test':>9} | {'ratio_1':>8} {'ratio_N':>8} {'drift':>7}")
    summary = header + [th, "-" * len(th)]
    for r in merged:
        summary.append(
            f"{r['axis']:>8} {r['label']:>8} | {r['intime_mae']:>7.3f} "
            f"{r['val_mae']:>8.3f} {r['val_mae_std']:>6.3f} {r['test_mae']:>9.3f} | "
            f"{r.get('ratio_first', float('nan')):>8.3g} "
            f"{r.get('ratio_last', float('nan')):>8.3g} "
            f"{r.get('ratio_drift', float('nan')):>7.3g}"
        )

    summary += ["",
                "ratio = w_phys*L_phys_bal / (w_data*L_data_bal), i.e. what the "
                "optimiser actually mixed.",
                "ratio_1 is taken at epoch 2 (epoch 1 still carries the untrained "
                "rollout transient),",
                "ratio_N at the last epoch, drift = ratio_N / ratio_1.",
                "  LEVEL  under loss_balance=ema this starts at w_phys/w_data by "
                "construction. Under",
                "         legacy it starts at w_phys/(w_data*L_data), so the same "
                "w_phys can mean",
                "         something orders of magnitude different depending on how "
                "well the fit started.",
                "  DRIFT  near 1 the mixture is stable and a weight tuned in a short "
                "probe transfers to",
                "         a long run. Far from 1 it does not, and the best w_phys is "
                "a function of --epochs.",
                ""]

    summary.append("Span per axis (max - min validation MAE over the axis):")
    for axis in ("balance", "resnorm", "feats"):
        vals = [r["val_mae"] for r in usable if r["axis"] == axis]
        if len(vals) < 2:
            continue
        span = max(vals) - min(vals)
        noise = max((r["val_mae_std"] for r in usable if r["axis"] == axis),
                    default=0.0)
        verdict = ("below the seed spread - this choice does not matter here"
                   if len(cli.seeds) > 1 and span < noise
                   else "worth acting on" if len(cli.seeds) > 1
                   else "seed spread unknown (single seed)")
        summary.append(f"  {axis:>8}: {span:6.3f} °C   ({verdict})")

    missing = [a for a in ("balance", "resnorm", "feats")
               if not any(r["axis"] == a for r in merged)]
    summary += [
        "",
        "MAE = mean |true - predicted| (°C) from free-running rollout.",
        f"Selection ran on {cli.val_op}; {cli.test_op} was never used to choose.",
        f"BEST (by MAE_val): {label_of(best)} -> val {best['val_mae']:.3f}°C, "
        f"test {best['test_mae']:.3f}°C",
    ]
    summary += noise_verdict(usable, best, len(cli.seeds), label_of)
    if missing:
        summary.append(f"Axes not yet run: {', '.join(missing)} "
                       f"(--part {'2' if 'feats' in missing else '1'})")
    summary += [
        "",
        "Next: run the weight probe with the balance this picked, e.g.",
        f"  python3 benchmark_wphys_wbc.py --probe --probe-part 1 --epochs "
        f"{cli.epochs} --loss-balance {best.get('loss_balance', cli.loss_balance)} "
        f"--device cuda",
        f"Total runtime this session: {total_time/3600:.2f} hours",
    ]
    (ART_DIR / "benchmark_balance_best.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[len(header):]), flush=True)

    # ---- per-axis MAE panels ------------------------------------------------
    axes_present = [a for a in ("balance", "resnorm", "feats")
                    if any(r["axis"] == a for r in merged)]
    fig, panels = plt.subplots(1, len(axes_present),
                               figsize=(5.0 * len(axes_present), 4.3), squeeze=False)
    for col, axis in enumerate(axes_present):
        rows = [r for r in merged if r["axis"] == axis]
        ax = panels[0][col]
        x = np.arange(len(rows))
        ax.errorbar(x, [r["val_mae"] for r in rows],
                    yerr=[r["val_mae_std"] for r in rows], fmt="o-", capsize=4,
                    label=f"val ({cli.val_op})", color="#1f77b4")
        ax.errorbar(x, [r["test_mae"] for r in rows],
                    yerr=[r["test_mae_std"] for r in rows], fmt="s--", capsize=4,
                    label=f"test ({cli.test_op})", color="#b00020", alpha=0.8)
        for i, r in enumerate(rows):
            if _is_baseline(r, cli):
                ax.axvline(i, color="gray", ls=":", lw=1.5, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows], rotation=20, ha="right")
        ax.set_xlabel(axis)
        ax.set_ylabel("rollout MAE [°C]")
        ax.set_title(f"{axis} (dotted = baseline)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Loss balancing — error bars = seed std", fontsize=12)
    fig.tight_layout()
    fig.savefig(ART_DIR / "benchmark_balance.png", dpi=130)
    plt.close(fig)

    # ---- the drift itself ---------------------------------------------------
    # The MAE panels say which configuration won; this one says WHY, by showing
    # the mixture the optimiser actually saw over the run.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = 0
    for key, series in ratios.items():
        vals = [v for v in series.get("ratio_phys", []) if np.isfinite(v)]
        if len(vals) < 2:
            continue
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", ms=3,
                label=key.replace("|", "="))
        plotted += 1
    if plotted:
        ax.set_yscale("log")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$w_{phys}L_{phys}^{bal}\ /\ w_{data}L_{data}^{bal}$")
        ax.set_title("What the optimiser actually mixed (flat = stable balance)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(ART_DIR / "benchmark_balance_ratio.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {ART_DIR / 'benchmark_balance.png'} and _ratio.png", flush=True)


if __name__ == "__main__":
    main()
