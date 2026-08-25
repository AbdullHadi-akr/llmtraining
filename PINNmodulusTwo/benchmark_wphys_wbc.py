#!/usr/bin/env python3
"""Benchmark: 2D sweep of physics loss weight (w_phys) and BC loss weight (w_bc).

Sweeps both w_phys and w_bc in a grid and scores every combination by the
free-running autoregressive rollout MAE on a held-out VALIDATION OP; a second,
never-selected-on TEST OP is reported alongside it.

Everything except w_phys and w_bc is fixed so the comparison is apples-to-apples.

Fixed hyperparameters (override on CLI if desired):
- architecture: width=128, depth=4, per-layer learnable swish, weight-norm
- recurrence: history_mode=hybrid, delta_grid=0.2s (anchor lag),
  rate_lags=[5.0, 20.0] (cumulative segment lengths, all fixed)
- optimization: Adam, lr=2e-3, epochs=60, device=auto (CUDA when available)
- seeds: one training run per seed per grid point (--seeds, default [0]).
  Each point is scored by the MEAN over its seeds and carries the standard
  deviation, so a difference between points can be read against the spread
  the initialisation alone produces. Runtime scales linearly with the seed
  count: 100 points x 3 seeds = 300 trainings.
- data: train=OP01-OP05, val=OP06 (selection), test=OP07 (report only),
  subsample=2 (CFL-stable Δt=0.2s)
- loss weights: w_data=1.0 (fixed), w_phys and w_bc swept

Two modes:

--probe   RANGE PROBE, 9 points. A decade-spaced CROSS through a shared centre
          rather than a grid: each weight is walked over [0, 1e-3, 1e-2, 1e-1, 1]
          while the other sits at the centre. Answers "does this weight move the
          error at all, and in which decade" -- run it BEFORE the grid, because
          resolving a grid inside a range that turns out to be flat is the
          expensive way to learn nothing.

          --probe-part 1|2 splits those 9 points into two shorter sessions along
          the arms of the cross: part 1 walks w_phys (5 points, the shared centre
          among them), part 2 walks w_bc (4 points). Same trainings, same total
          time, in two sittings instead of one. A part run TRAINS AND SAVES --
          the CSV, the settings block and the raw rows are written every time --
          but it does not evaluate: the per-axis verdict weighs each arm against
          the shared centre, which arm 2 does not contain, so a half cross would
          compare an axis against itself.

          --report-only is the evaluation step. It trains nothing, merges the
          stored parts and writes the summary, the per-axis verdict and the
          plots. It refuses an incomplete cross. The parts must agree on every
          training setting; a mismatch is refused rather than merged, so that no
          verdict is ever assembled from two different experiments.

          The three steps, matching README_GPU_SERVER.md sections 7.1-7.3:
              --probe --probe-part 1 --epochs 20 --device cuda   # 5 trainings
              --probe --probe-part 2 --epochs 20 --device cuda   # 4 trainings
              --probe --report-only  --epochs 20 --device cuda   # plots, no GPU

(default) GRID, 5x5 = 25 points over [0, 0.01, 0.05, 0.1, 0.3] per weight.
          Use --w-phys/--w-bc to centre it on the decade the probe found.
          --extended-grid gives 10x10; see the runtime note below first.

Run:
    source .venv/bin/activate
    python3 PINNmodulusTwo/benchmark_wphys_wbc.py --probe --epochs 20 --device cuda
    python3 PINNmodulusTwo/benchmark_wphys_wbc.py --device cuda

RUNTIME -- read this before starting a long run.

At subsample=2 the rollout is ~7000 sequential steps per OP per epoch, and that
dominates everything. One epoch over 5 training OPs costs roughly 1.5-2.5 min on
an RTX 5090 Laptop, so ONE grid point at 60 epochs is 1.5-2.5 HOURS:
    probe,      1 seed  ->   9 trainings  ~14-22 h   (~5-7 h at --epochs 20)
        split as --probe-part 1 -> 5 trainings (~3-4 h at --epochs 20)
             and --probe-part 2 -> 4 trainings (~2.5-3 h at --epochs 20)
    5x5 grid,   1 seed  ->  25 trainings  ~1.5-2 days
    10x10 grid, 1 seed  -> 100 trainings  ~6-8 days   (--extended-grid)
Multiply by the seed count on top of that.

Do not take those numbers on faith: the log prints the measured seconds per
epoch from the first epoch on ("[12.4s/epoch, this run ~124 min left]"). Read it
once and compute the real total before committing days of GPU time.

To bring it down, in order of effect: fewer --epochs, a coarser grid, or a
larger --subsample (which shortens the rollout quadratically in wall time but
changes the time resolution). Note that --subsample is capped by CFL stability
at 2 (dt=0.2s against a limit of ~0.241s), so on this data it is not actually
available. --probe-part does not shorten the probe, it only splits it.

The full test sequence, in the order that makes sense, is in
README_GPU_SERVER.md section 7 -- run the smoke test and the seed-spread check
before committing hours to this one.

Outputs (in PINNmodulusTwo/artifacts/):
    benchmark_wphys_wbc.csv - one row per trained (w_phys, w_bc): mean rollout
        MAEs over the seeds, their standard deviation, and how many seeds
        survived. Written after every run, including an unfinished probe part.
    benchmark_wphys_wbc_settings.txt - the full hyperparameter block that
        produced those rows. Written after every run.
    probe_parts.json - split-probe bookkeeping: the rows of each finished part,
        waiting for the other one. Delete it to start a split probe over.
    benchmark_wphys_wbc_best.txt - best combination + summary table + per-axis
        verdict. Evaluation step only.
    benchmark_wphys_wbc_probe.png - per-axis MAE over the decades (probe).
    benchmark_wphys_wbc_probe_boxplot.png - one panel per probe point; per
        sampled time point, the spread of the absolute error across the sensors.
    benchmark_wphys_wbc_probe_convergence.png - loss curves per probe point.
    benchmark_wphys_wbc_heatmap.png - 2D heatmap of validation MAE (grid only).
    benchmark_wphys_wbc_boxplot.png - per-sensor error per configuration (grid).
    checkpoints_wphys_wbc/*.pt - per-sweep model checkpoints
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bench_common import (
    EMPTY_HIST, N_BOX_TIMES, aggregate_seeds, failed_result, make_train_args,
    noise_verdict, print_eta, train_one_seed,
)
from data import require_ops
from device_utils import resolve_device
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Default 2D sweep grid: 5x5 = 25 points. Centre it on whatever decade the
# range probe found; the values below are only a starting spread.
# (The old comment claimed 'best around w_phys~0.1-0.2' from earlier runs --
#  those ran with the broken L_phys and delta, so the number meant nothing.)
#
# Under --loss-balance ema every term is divided by its own magnitude, so a
# weight is a ratio between TERMS: 0.1 means "physics contributes a tenth of
# what data does", and it means that in every epoch. The spread below therefore
# walks from off to roughly a third -- which is where a regulariser belongs.
DEFAULT_W_PHYS = [0.0, 0.01, 0.05, 0.1, 0.3]
DEFAULT_W_BC = [0.0, 0.01, 0.05, 0.1, 0.3]

# Range probe: a CROSS through the baseline, not a grid. Decade-spaced, so it
# answers "which order of magnitude matters at all" before a grid commits hours
# to resolving differences inside a range that may be entirely flat.
#
# These values assume --loss-balance ema, where every term is divided by its own
# magnitude and the weights are therefore genuine ratios: w_phys = 1 means "the
# physics term contributes as much as the data term", and the decades below walk
# from "off" to "equal footing".
#
# Under --loss-balance legacy the same numbers mean something else entirely,
# because there L_data stays raw: the ratio is w_phys/(w_data * L_data), so it
# depends on how far the fit has converged and keeps moving during the run. If
# you deliberately probe in legacy mode, shift this list DOWN by roughly the
# decade L_data converges to (~1e-2..1e-3 here, i.e. probe 1e-5..1e-2) -- and
# expect the answer not to transfer to a run with a different --epochs.
# benchmark_balance.py measures which of the two regimes you are in.
PROBE_W_PHYS = [0.0, 0.001, 0.01, 0.1, 1.0]
PROBE_W_BC = [0.0, 0.001, 0.01, 0.1, 1.0]
# The two baselines must be members of their own lists, so the arms of the cross
# meet in a single shared point. Otherwise each axis is measured against a
# different reference and the two are not comparable -- and the cross costs one
# training more than it needs to.
PROBE_BASE_W_PHYS = 0.01
PROBE_BASE_W_BC = 0.1
assert PROBE_BASE_W_PHYS in PROBE_W_PHYS and PROBE_BASE_W_BC in PROBE_W_BC

# Extended 10×10 grid - use with --extended-grid
EXTENDED_W_PHYS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
EXTENDED_W_BC = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.7, 1.0]


def _report_probe(results, usable, cli, summary) -> None:
    """Per-axis verdict for the range probe: which decade, and does it matter.

    A weight only deserves a grid if moving it across decades moves the error by
    more than the seeds do on their own. Otherwise the honest conclusion is that
    this weight is not a useful knob here, and the grid budget belongs elsewhere.
    """
    lines = ["", "RANGE PROBE - per-axis verdict:"]
    for axis, base_other, key, other_key in (
        ("w_phys", PROBE_BASE_W_BC, "w_phys", "w_bc"),
        ("w_bc", PROBE_BASE_W_PHYS, "w_bc", "w_phys"),
    ):
        rows = [r for r in usable if r[other_key] == base_other]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[key])
        vals = [r["val_mae"] for r in rows]
        span = max(vals) - min(vals)
        noise = max((r["val_mae_std"] for r in rows), default=0.0)
        best_row = min(rows, key=lambda r: r["val_mae"])
        lines.append(f"  {axis} (at {other_key}={base_other}):")
        lines.append("    " + "  ".join(
            f"{r[key]:g}->{r['val_mae']:.2f}" for r in rows))
        lines.append(f"    best {axis}={best_row[key]:g} "
                     f"(val {best_row['val_mae']:.3f} °C), "
                     f"span over the decades = {span:.3f} °C")
        if len(cli.seeds) < 2:
            lines.append("    seed spread unknown (single seed) - re-run the probe "
                         "with --seeds 0 1 2 before trusting this.")
        elif span < noise:
            lines.append(f"    span is BELOW the seed spread ({noise:.3f} °C): this "
                         f"weight does not move the error. Skip the grid for it.")
        else:
            lines.append(f"    span exceeds the seed spread ({noise:.3f} °C) - worth "
                         f"a grid, centred on the best decade above.")
    lines += ["",
              "Next: run the 5x5 grid only over the decade(s) that mattered, e.g.",
              "  --w-phys 0.01 0.03 0.05 0.1 0.3  --w-bc 0.01 0.03 0.05 0.1 0.3"]
    (ART_DIR / "benchmark_wphys_wbc_best.txt").write_text(
        "\n".join(summary + lines) + "\n")
    print("\n".join(lines), flush=True)


def probe_arms(cli) -> tuple[list, list]:
    """The two arms of the probe cross, in the order they are trained.

    Arm 1 walks w_phys at the w_bc baseline; arm 2 walks w_bc at the w_phys
    baseline. The shared centre point belongs to arm 1 -- arm 2 skips it -- so
    the arms are disjoint and 5 + 4 = 9 points, not 10.

    Splitting here is what makes ``--probe-part`` possible: each arm is a
    self-contained run, and the per-axis verdict needs both of them.
    """
    base_p, base_b = PROBE_BASE_W_PHYS, PROBE_BASE_W_BC
    arm_phys = [(p, base_b) for p in cli.w_phys]
    arm_bc = [(base_p, b) for b in cli.w_bc if (base_p, b) not in arm_phys]
    return arm_phys, arm_bc


def build_pairs(cli) -> list:
    """The (w_phys, w_bc) pairs to train.

    Normally the full product of both lists. In probe mode a CROSS instead: each
    axis is walked through the decades while the other sits at its baseline. That
    is 2*n-1 points rather than n^2 and answers a different question -- not "which
    cell is best" but "does this weight move the error at all, and in which
    decade". Resolving a grid inside a range that turns out to be flat is the
    expensive way to learn nothing.

    ``--probe-part`` narrows this to one arm so the probe fits into two shorter
    sessions instead of one long one. The evaluation still needs the whole cross,
    so a part run stores its rows and defers the report (see ``save_probe_part``).
    """
    if not cli.probe:
        return [(p, b) for p in cli.w_phys for b in cli.w_bc]
    arm_phys, arm_bc = probe_arms(cli)
    part = getattr(cli, "probe_part", "all")
    if part == "1":
        return arm_phys
    if part == "2":
        return arm_bc
    return arm_phys + arm_bc


# --- Split probe: storing one part until the other one lands ---------------
#
# The per-axis verdict compares each arm against the SHARED centre point, so it
# cannot be computed from one arm alone -- arm 2 does not even contain the
# centre. A part run therefore writes its rows here and stops before the report;
# whichever part completes the cross merges everything and reports once.
PROBE_STATE_FILE = "probe_parts.json"
PROBE_PART_LABEL = {"1": "1/2 (w_phys arm)", "2": "2/2 (w_bc arm)"}


def _probe_signature(cli) -> dict:
    """Settings the two parts must agree on for their rows to be comparable.

    Merging an arm trained at 20 epochs with one trained at 10 would produce a
    verdict built from two different experiments, and nothing in the output would
    say so. Anything that changes what a training run means belongs in here.
    """
    return {
        "ops": list(cli.ops), "val_op": cli.val_op, "test_op": cli.test_op,
        "subsample": int(cli.subsample), "epochs": int(cli.epochs),
        "seeds": list(cli.seeds), "lr": float(cli.lr),
        "weight_decay": float(cli.weight_decay),
        "gain_lr_mult": float(cli.gain_lr_mult),
        "grad_clip": float(cli.grad_clip),
        "width": int(cli.width), "depth": int(cli.depth),
        "k_max": int(cli.k_max), "history_mode": str(cli.history_mode),
        "rate_lags": [float(v) for v in cli.rate_lags],
        "delta_grid": float(cli.delta_grid),
        "time_deriv": str(cli.time_deriv),
        "phys_norm": float(cli.phys_norm),
        "batch_data": int(cli.batch_data),
        "batch_phys": int(cli.batch_phys), "batch_bc": int(cli.batch_bc),
        "w_phys": [float(v) for v in cli.w_phys],
        "w_bc": [float(v) for v in cli.w_bc],
        # The balancing decides what a weight MEANS, so two arms balanced
        # differently are not one experiment. Without these keys part 2 could be
        # run with another --loss-balance and silently merge into part 1.
        "loss_balance": str(cli.loss_balance),
        "ema_decay": float(cli.ema_decay),
        "balance_warmup": int(cli.balance_warmup),
        "data_floor": float(cli.data_floor),
        "bc_norm": float(cli.bc_norm),
        "residual_norm": str(cli.residual_norm),
        "subsample_mode": str(cli.subsample_mode),
        "forcing_energy": bool(cli.forcing_energy),
        "config_rates": bool(cli.config_rates),
    }


def _jsonable(obj):
    """Recursively turn numpy scalars/arrays into plain JSON types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _pair_key(w_phys, w_bc) -> str:
    """Stable dict key for a (w_phys, w_bc) pair across a JSON round-trip."""
    return f"{float(w_phys):.6g}|{float(w_bc):.6g}"


def save_probe_part(cli, part: str, results: list, histories: list) -> dict:
    """Persist one arm's rows and return the full state (this part included).

    A signature mismatch discards the stored parts rather than merging across
    settings: a stale arm is worse than a missing one, because the merge would
    still produce a plausible-looking verdict.
    """
    path = ART_DIR / PROBE_STATE_FILE
    signature = _probe_signature(cli)
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [probe] could not read {path.name} ({exc}) - starting over.",
                  flush=True)
            state = {}
    if state.get("signature") not in (None, signature):
        print("  [probe] stored part(s) ran with different settings - discarding "
              "them; the other part has to be re-run to match this one.", flush=True)
        state = {}
    state["signature"] = signature
    state.setdefault("parts", {})[part] = {
        "results": [_jsonable(r) for r in results],
        "histories": [_jsonable(h) for h in histories],
    }
    path.write_text(json.dumps(state, indent=1) + "\n")
    print(f"  [probe] part {PROBE_PART_LABEL[part]} stored in {path}", flush=True)
    return state


def merge_probe_parts(cli, state: dict) -> tuple[list, list, list]:
    """Collect the stored arms into cross order -> (results, histories, missing).

    Order matches ``build_pairs`` with no part selected, so a split probe reports
    exactly what a single-run probe would have.
    """
    rows, hists = {}, {}
    for blob in state.get("parts", {}).values():
        for r in blob.get("results", []):
            row = dict(r)
            row["test_box_errors"] = np.asarray(row["test_box_errors"],
                                                dtype=float)
            row["test_box_times_s"] = np.asarray(row["test_box_times_s"],
                                                 dtype=float)
            rows[_pair_key(row["w_phys"], row["w_bc"])] = row
        for h in blob.get("histories", []):
            hists[_pair_key(h["w_phys"], h["w_bc"])] = h
    arm_phys, arm_bc = probe_arms(cli)
    results, histories, missing = [], [], []
    for w_phys, w_bc in arm_phys + arm_bc:
        key = _pair_key(w_phys, w_bc)
        if key in rows:
            results.append(rows[key])
            histories.append(hists.get(
                key, {"w_phys": w_phys, "w_bc": w_bc, "hist": EMPTY_HIST}))
        else:
            missing.append((w_phys, w_bc))
    return results, histories, missing


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
                   help="hybrid rate segments in seconds (FIXED)")
    p.add_argument("--delta-grid", type=float, default=0.2,
                   help="anchor lag of the hybrid history in seconds (FIXED)")
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
                   help="10×10 grid instead of the default 5×5. Costs days - read "
                        "the runtime note in the module docstring first.")
    p.add_argument("--probe", action="store_true",
                   help="range probe instead of a grid: a decade-spaced CROSS "
                        "through the baseline (9 points, not 25). Run this before "
                        "the 5x5 grid to find out which decade of each weight "
                        "matters at all - a grid inside a flat range buys nothing.")
    p.add_argument("--report-only", action="store_true",
                   help="train nothing: merge the stored probe parts and write "
                        "the CSV, the summary, the per-axis verdict and the "
                        "plots. This is the evaluation step of a split probe; it "
                        "refuses to run on an incomplete cross. Probe mode only.")
    p.add_argument("--probe-part", choices=["1", "2", "all"], default="all",
                   help="split the probe into two shorter sessions: 1 = w_phys "
                        "arm (5 points), 2 = w_bc arm (4 points), all = both in "
                        "one run. A part run stores its rows and reports nothing; "
                        "whichever part completes the cross writes the CSV, the "
                        "summary and the per-axis verdict. Probe mode only.")
    # Batching
    p.add_argument("--batch-data", type=int, default=2048)
    p.add_argument("--batch-phys", type=int, default=256)
    p.add_argument("--batch-bc", type=int, default=128)
    p.add_argument("--phys-norm", type=float, default=0.0,
                   help="L_phys divisor: 0=adaptive EMA, >0=fixed divisor")
    # Checkpoints
    # Loss balancing / preprocessing. These change what w_phys and w_bc MEAN,
    # so a sweep is only comparable with another one that used the same values.
    # benchmark_balance.py is what decides them.
    p.add_argument("--loss-balance", choices=["ema", "legacy", "fixed"],
                   default=d.get("loss_balance", "ema"), help="see train.py --loss-balance")
    p.add_argument("--ema-decay", type=float, default=d.get("ema_decay", 0.9))
    p.add_argument("--balance-warmup", type=int, default=d.get("balance_warmup", 1))
    p.add_argument("--data-floor", type=float, default=d.get("data_floor", 1e-08))
    p.add_argument("--bc-norm", type=float, default=d.get("bc_norm", 0.0))
    p.add_argument("--residual-norm", choices=["rms", "legacy"], default=d.get("residual_norm", "rms"))
    p.add_argument("--zero-weight-terms", choices=["skip", "compute"], default=d.get("zero_weight_terms", "skip"))
    p.add_argument("--subsample-mode", choices=["stride", "mean"], default=d.get("subsample_mode", "stride"))
    p.add_argument("--forcing-energy", action="store_true", default=d.get("forcing_energy", False))
    p.add_argument("--config-rates", action="store_true", default=d.get("config_rates", False))
    p.add_argument("--save-models", dest="save_models", action="store_true", default=True)
    p.add_argument("--no-save-models", dest="save_models", action="store_false")
    p.add_argument("--save-best-only", action="store_true",
                   help="save only the best (by validation MAE) model instead of all points")
    p.add_argument("--model-dir", default=str(ART_DIR / "checkpoints_wphys_wbc"))
    
    args = p.parse_args()
    
    if args.probe_part != "all" and not args.probe:
        p.error("--probe-part only applies to --probe (the grid is not a cross)")
    if args.report_only and not args.probe:
        p.error("--report-only only applies to --probe (a grid reports in one run)")
    if args.report_only and args.probe_part != "all":
        p.error("--report-only evaluates the whole cross; drop --probe-part")

    # Apply grid defaults
    if args.probe:
        if args.w_phys is None:
            args.w_phys = PROBE_W_PHYS
        if args.w_bc is None:
            args.w_bc = PROBE_W_BC
    elif args.extended_grid:
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


def build_header(cli, n_points: int, part: str | None = None) -> list:
    """The settings block printed at the start and stored with every result.

    Every hyperparameter that shapes a training run is listed here, including the
    ones that are never swept. A result file that does not say what produced it
    cannot be compared against anything later, and a split probe makes that worse:
    its two halves are only meaningful together if they ran identically.

    ``part`` names one arm of a split probe; pass None for the whole cross, which
    is also what the merged report is written with.
    """
    dt_s = 0.1 * cli.subsample
    mode = "RANGE PROBE (cross through the baseline)" if cli.probe else "GRID"
    if part is not None:
        mode += f" - PART {PROBE_PART_LABEL[part]}"
    lines = [
        f"Physics+BC loss-weight benchmark - {mode}",
        "free-running rollout, NO teacher forcing",
        f"train = {'+'.join(cli.ops)}   val (selection) = {cli.val_op}   "
        f"test (report only) = {cli.test_op}",
        "DATA / TIME GRID:",
        f"  subsample={cli.subsample} -> dt={dt_s:.1f}s   "
        f"delta_grid={cli.delta_grid}s   rate_lags={cli.rate_lags}s",
        "FIXED ARCHITECTURE (for fair comparison):",
        f"  width={cli.width}  depth={cli.depth}  k_max={cli.k_max}  "
        f"history_mode={cli.history_mode}",
        f"  time_deriv={cli.time_deriv}  use_static={cli.use_static}  "
        f"use_forcing={cli.use_forcing}",
        "TRAINING SETTINGS:",
        f"  lr={cli.lr}  epochs={cli.epochs}  weight_decay={cli.weight_decay}  "
        f"grad_clip={cli.grad_clip}",
        f"  gain_lr_mult={cli.gain_lr_mult}  seeds={cli.seeds}  "
        f"early_stopping_patience={cli.early_stopping_patience}",
        f"  batch_data={cli.batch_data}  batch_phys={cli.batch_phys}  "
        f"batch_bc={cli.batch_bc}",
        f"  runs = {n_points} points x {len(cli.seeds)} "
        f"seed(s) = {n_points*len(cli.seeds)} trainings",
        "LOSS BALANCING (decides what a weight MEANS - see benchmark_balance.py):",
        f"  loss_balance={cli.loss_balance}  ema_decay={cli.ema_decay}  "
        f"residual_norm={cli.residual_norm}",
        f"  phys_norm={cli.phys_norm}  bc_norm={cli.bc_norm}  "
        f"data_floor={cli.data_floor}  balance_warmup={cli.balance_warmup}",
        f"  forcing_energy={cli.forcing_energy}  config_rates={cli.config_rates}  "
        f"subsample_mode={cli.subsample_mode}",
        "LOSS WEIGHTS (SWEPT):",
        "  w_data=1.0 (fixed)",
        f"  w_phys sweep = {cli.w_phys}",
        f"  w_bc sweep = {cli.w_bc}",
        (f"Probe: w_phys swept at w_bc={PROBE_BASE_W_BC}, "
         f"w_bc swept at w_phys={PROBE_BASE_W_PHYS}"
         + ("  |  this run trains one arm only; evaluation is a separate step"
            if part is not None else "")
         if cli.probe else
         f"Grid size: {len(cli.w_phys)} x {len(cli.w_bc)} = "
         f"{len(cli.w_phys)*len(cli.w_bc)} points"),
        "",
    ]
    return lines


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
    if cli.report_only:
        run_report_only(cli, plt)
        return
    dt_s = 0.1 * cli.subsample
    model_dir = Path(cli.model_dir)
    if cli.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    pairs_preview = build_pairs(cli)
    split_probe = cli.probe and cli.probe_part != "all"
    header = build_header(cli, len(pairs_preview),
                          part=cli.probe_part if split_probe else None)
    print("\n".join(header), flush=True)

    results = []
    histories = []  # Store epoch histories for convergence plotting
    pairs = build_pairs(cli)
    total_points = len(pairs)
    start_time_total = time.time()

    for idx, (w_phys, w_bc) in enumerate(pairs, start=1):
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
    print(f"Part time: {total_time/3600:.2f} hours" if split_probe
          else f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*60}\n")

    if split_probe:
        # A part run trains and SAVES, and stops there. Evaluation is its own
        # step because the per-axis verdict weighs each arm against the shared
        # centre point: arm 2 does not contain it, so a half cross would compare
        # an axis against itself and still print a plausible number.
        state = save_probe_part(cli, cli.probe_part, results, histories)
        merged, _, missing = merge_probe_parts(cli, state)
        csv_path = write_csv(merged)
        set_path = write_settings(cli, len(merged), cli.probe_part)
        print(f"Part {PROBE_PART_LABEL[cli.probe_part]} done - "
              f"{len(merged)} of {len(merged) + len(missing)} points trained.",
              flush=True)
        print(f"  results saved: {csv_path}", flush=True)
        print(f"  settings saved: {set_path}", flush=True)
        if missing:
            other = "2" if cli.probe_part == "1" else "1"
            print("  still to train: " + ", ".join(
                f"(w_phys={a:g}, w_bc={b:g})" for a, b in missing), flush=True)
            # Echo the balancing flags explicitly: they are part of the
            # signature now, so a part 2 started without them lands on the
            # config.yaml defaults and discards part 1 instead of merging.
            extra = f" --loss-balance {cli.loss_balance}"
            if cli.residual_norm != "rms":
                extra += f" --residual-norm {cli.residual_norm}"
            if cli.forcing_energy:
                extra += " --forcing-energy"
            if cli.config_rates:
                extra += " --config-rates"
            print(f"\nNext - the other part, with these same flags:\n"
                  f"  python3 {Path(__file__).name} --probe --probe-part {other} "
                  f"--epochs {cli.epochs} --device {cli.device}{extra}", flush=True)
            print("  (a settings mismatch discards the stored part instead of "
                  "merging it)", flush=True)
        else:
            print(f"\nThe cross is complete. Plots and verdict:\n"
                  f"  python3 {Path(__file__).name} --probe --report-only "
                  f"--epochs {cli.epochs} --device {cli.device}", flush=True)
        return

    evaluate(cli, results, histories, total_time, plt)



def write_csv(results: list) -> Path:
    """One row per trained point. Written after EVERY run, complete or not.

    This is the raw record: a part run that trains five points and then stops
    still has five real results, and losing them to a crash before the second
    part would cost hours. The evaluation files are the ones that wait for the
    full cross -- the CSV does not.
    """
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
    path = ART_DIR / "benchmark_wphys_wbc.csv"
    path.write_text("\n".join(csv_lines) + "\n")
    return path


def write_settings(cli, n_points: int, part: str | None) -> Path:
    """Drop the full hyperparameter block next to the results, as a file.

    The console scrolls away and a CSV carries no settings, so without this a
    result folder cannot say what produced it. Rewritten by every run, including
    the parts, which is safe because a mismatch between parts is refused earlier.
    """
    path = ART_DIR / "benchmark_wphys_wbc_settings.txt"
    path.write_text("\n".join(build_header(cli, n_points, part)) + "\n")
    return path


def run_report_only(cli, plt) -> None:
    """Evaluate stored probe parts without training anything (README step 7.3).

    Refuses an incomplete cross rather than reporting on it: the per-axis verdict
    weighs each arm against the shared centre point, so a half cross would
    compare an axis against itself and still print a confident-looking number.
    """
    path = ART_DIR / PROBE_STATE_FILE
    if not path.exists():
        print(f"No stored probe parts at {path} - run --probe-part 1 and 2 first.",
              flush=True)
        raise SystemExit(1)
    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Cannot read {path}: {exc}", flush=True)
        raise SystemExit(1)
    stored = state.get("signature") or {}
    current = _probe_signature(cli)
    differing = [k for k in current if stored.get(k) != current[k]]
    if stored and differing:
        print("The stored parts ran with different settings than this command:",
              flush=True)
        for k in differing:
            print(f"  {k}: stored {stored.get(k)!r} vs here {current[k]!r}", flush=True)
        print("Pass the same flags the parts ran with (they are recorded in "
              f"{ART_DIR / 'benchmark_wphys_wbc_settings.txt'}).", flush=True)
        raise SystemExit(1)
    results, histories, missing = merge_probe_parts(cli, state)
    if missing:
        print(f"{len(missing)} of {len(results) + len(missing)} points are still "
              f"missing - nothing to evaluate yet:", flush=True)
        for w_phys, w_bc in missing:
            print(f"  (w_phys={w_phys:g}, w_bc={w_bc:g})", flush=True)
        arm_phys, _ = probe_arms(cli)
        todo = "1" if set(missing) & set(arm_phys) else "2"
        # Deliberately NOT cli.device: this step is documented to run on the CPU
        # because it computes nothing, and echoing that into a TRAINING command
        # would quietly put a multi-hour run on the CPU.
        print(f"\nRun the missing part first (needs the GPU):\n"
              f"  python3 {Path(__file__).name} --probe --probe-part {todo} "
              f"--epochs {cli.epochs} --device cuda", flush=True)
        raise SystemExit(1)
    total_time = sum(r["train_time"] for r in results)
    print("\n".join(build_header(cli, len(results))), flush=True)
    print(f"Evaluating {len(results)} stored points (no training in this step). "
          f"Combined training time of the parts: {total_time/3600:.2f} hours.\n",
          flush=True)
    evaluate(cli, results, histories, total_time, plt)


def draw_time_boxes(ax, row: dict, cli, show_ylabel: bool = True) -> bool:
    """One box per sampled time point; each box summarises the SENSORS.

    At a given moment the held-out OP has one absolute error per sensor, so the
    box is the middle 50% of sensors (25% above the median, 25% below), the red
    line is the median sensor, and the dots above the whisker are the sensors the
    model gets worst at that moment.

    Whiskers are the matplotlib default of 1.5 IQR, not min-max: with a few
    hundred sensors, min-max whiskers stretch to the single worst one and flatten
    every box into a line, hiding the spread the plot exists to show.
    """
    err = np.asarray(row.get("test_box_errors", np.zeros((0, 0))), dtype=float)
    times = np.asarray(row.get("test_box_times_s", np.zeros(0)), dtype=float)
    if err.ndim != 2 or err.size == 0:
        ax.text(0.5, 0.5, "no usable rollout", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        return False
    data = [e[np.isfinite(e)] for e in err]          # one entry per time point
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e8ff")
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("#b00020")
        median.set_linewidth(1.8)
    for flier in bp.get("fliers", []):
        flier.set(markerfacecolor="#444444", markeredgecolor="#444444",
                  markersize=3, alpha=0.45)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([f"{t:.0f}" for t in times], fontsize=8)
    if show_ylabel:
        ax.set_ylabel(f"|error| per sensor [°C]  (n={err.shape[1]})", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    return True


def draw_config_boxes(ax, rows: list, labels: list, cli, best: dict | None = None):
    """One box per configuration, pooling every sampled (time, sensor) error.

    Used where a panel per configuration would not fit -- the 5x5 grid has 25 of
    them. Pooling loses the time structure that :func:`draw_time_boxes` shows, so
    it answers only the comparison question: does this configuration's error
    distribution sit lower than that one's.
    """
    data, kept = [], []
    for row, label in zip(rows, labels):
        err = np.asarray(row.get("test_box_errors", np.zeros((0, 0))), dtype=float)
        flat = err.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            data.append(flat)
            kept.append((row, label))
    if not data:
        ax.text(0.5, 0.5, "no usable rollout", ha="center", va="center",
                transform=ax.transAxes)
        return
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([lbl for _, lbl in kept])
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e8ff")
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("#b00020")
        median.set_linewidth(1.8)
    for flier in bp.get("fliers", []):
        flier.set(markerfacecolor="#444444", markeredgecolor="#444444",
                  markersize=2.5, alpha=0.35)
    if best is not None:
        for i, (row, _) in enumerate(kept, start=1):
            if row["w_phys"] == best["w_phys"] and row["w_bc"] == best["w_bc"]:
                # The axis shows test-op error, so mark the TEST value of the
                # point selection picked on the validation OP -- not its val MAE.
                ax.scatter([i], [row["test_mae"]], s=280, marker="*", color="red",
                           edgecolors="darkred", linewidths=1.2, zorder=10,
                           label=f"selected on {cli.val_op} (mean test MAE)")
                ax.legend(loc="upper right", fontsize=9)
                break
    ax.set_ylabel("|error| per sensor [°C]", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)


def plot_probe(results: list, histories: list, cli, best: dict, plt) -> list:
    """Plots for the cross: one panel per axis, plus the convergence curves.

    The cross has no surface to draw, so the grid's heatmap makes no sense here.
    What the probe actually answers is per-axis -- "does this weight move the
    error, and where" -- so each arm gets its own panel, read against the shared
    centre point that both arms have in common.

    Decade values include 0, which no log axis can show, so the points sit at
    even positions and the real values become the tick labels. That keeps the
    zero-weight case on the same plot as the rest instead of dropping it.
    """
    saved = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (axis, other_key, base_other) in zip(axes, (
            ("w_phys", "w_bc", PROBE_BASE_W_BC),
            ("w_bc", "w_phys", PROBE_BASE_W_PHYS))):
        rows = [r for r in results if r[other_key] == base_other]
        rows.sort(key=lambda r: r[axis])
        if not rows:
            continue
        pos = list(range(len(rows)))
        val = [r["val_mae"] for r in rows]
        test = [r["test_mae"] for r in rows]
        val_err = [r["val_mae_std"] for r in rows]
        ax.errorbar(pos, val, yerr=val_err, marker="o", markersize=6, linewidth=2,
                    capsize=4, label=f"val {cli.val_op} (selection)")
        ax.plot(pos, test, marker="s", markersize=5, linewidth=1.5, alpha=0.7,
                linestyle="--", label=f"test {cli.test_op} (report only)")
        finite = [(i, v) for i, v in zip(pos, val) if np.isfinite(v)]
        if finite:
            bi, bv = min(finite, key=lambda t: t[1])
            ax.scatter([bi], [bv], s=220, facecolors="none", edgecolors="red",
                       linewidths=2.5, zorder=5, label="best on this axis")
            span = max(v for _, v in finite) - bv
            ax.set_title(f"{axis} at {other_key}={base_other:g}\n"
                         f"span over the decades = {span:.3f} °C", fontsize=11)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{r[axis]:g}" for r in rows])
        ax.set_xlabel(axis, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("rollout MAE [°C], lower is better", fontsize=11)
    fig.suptitle(f"Range probe - {'+'.join(cli.ops)} train, {cli.epochs} epochs, "
                 f"dt={0.1*cli.subsample:.1f}s, seeds={cli.seeds}", fontsize=12)
    fig.tight_layout()
    path = ART_DIR / "benchmark_wphys_wbc_probe.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # Small multiples: every probe point gets its own time axis. Nine panels is
    # what makes the per-time-point box readable at all -- pooled into one axis
    # per configuration the time structure disappears, and that structure is
    # where a bad weight shows itself (error growing along the rollout rather
    # than being uniformly worse).
    # Panels follow the arms of the cross, not a plain numeric sort: the w_phys
    # arm first in decade order, then the w_bc arm. A sort by (w_phys, w_bc)
    # interleaves the two arms and the rows stop meaning anything.
    by_pair = {_pair_key(r["w_phys"], r["w_bc"]): r for r in results}
    arm_phys, arm_bc = probe_arms(cli)
    ordered = [by_pair[k] for k in
               (_pair_key(a, b) for a, b in arm_phys + arm_bc) if k in by_pair]
    n_cols = 3
    n_rows = int(np.ceil(len(ordered) / n_cols)) or 1
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows),
                               sharey=True, squeeze=False)
    flat_axes = [ax for row in axes3 for ax in row]
    for ax, row in zip(flat_axes, ordered):
        drawn = draw_time_boxes(ax, row, cli, show_ylabel=ax in axes3[:, 0])
        is_best = (row["w_phys"] == best["w_phys"] and row["w_bc"] == best["w_bc"])
        title = f"w_phys={row['w_phys']:g}, w_bc={row['w_bc']:g}"
        if drawn and np.isfinite(row["test_mae"]):
            title += f"   (mean {row['test_mae']:.2f} °C)"
        ax.set_title(title + ("  * BEST" if is_best else ""), fontsize=10,
                     fontweight="bold" if is_best else "normal")
        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor("#b00020")
                spine.set_linewidth(2.0)
    for ax in flat_axes[len(ordered):]:
        ax.axis("off")
    for ax in axes3[-1]:
        ax.set_xlabel("time [s]", fontsize=10)
    fig3.suptitle(
        f"Held-out {cli.test_op}: error spread across the sensors, at "
        f"{N_BOX_TIMES} random time points\n"
        f"one box = one moment; box = middle 50% of the sensors, red line = "
        f"median sensor, dots = sensors beyond 1.5 IQR",
        fontsize=12)
    fig3.tight_layout(rect=(0, 0, 1, 0.94))
    path3 = ART_DIR / "benchmark_wphys_wbc_probe_boxplot.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    saved.append(path3)

    usable_hist = [h for h in histories if h.get("hist", {}).get("epoch")]
    if usable_hist:
        fig2, axc = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for item in usable_hist:
            h = item["hist"]
            is_best = (item["w_phys"] == best["w_phys"]
                       and item["w_bc"] == best["w_bc"])
            label = (f"p={item['w_phys']:.3g}, b={item['w_bc']:.3g}"
                     + (" *BEST" if is_best else ""))
            lw = 2.5 if is_best else 1.2
            for ax, key, marker in ((axc[0], "L_data", "o"),
                                    (axc[1], "L_phys", "s"),
                                    (axc[2], "L_bc", "^")):
                if h.get(key):
                    ax.plot(h["epoch"], h[key], marker=marker, markersize=3,
                            linewidth=lw, alpha=0.85, label=label)
        for ax, name in zip(axc, ("L_data (MSE)", "L_phys (unweighted)",
                                  "L_bc (unweighted)")):
            ax.set_ylabel(name, fontsize=11)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=3)
        axc[0].set_title(f"Range-probe convergence - {'+'.join(cli.ops)} train, "
                         f"{cli.val_op} val", fontsize=12)
        axc[2].set_xlabel("Epoch", fontsize=11)
        fig2.tight_layout()
        path2 = ART_DIR / "benchmark_wphys_wbc_probe_convergence.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        saved.append(path2)
    return saved


def evaluate(cli, results: list, histories: list, total_time: float, plt) -> None:
    """Turn a complete set of result rows into CSV, summary, verdict and plots.

    Separated from the training loop so a split probe can be evaluated in its
    own step, from stored rows, without retraining anything. The header is
    rebuilt here from the full row count, so a report merged out of two parts
    describes the whole cross rather than whichever part happened to run last.
    """
    model_dir = Path(cli.model_dir)
    header = build_header(cli, len(results))
    write_csv(results)
    write_settings(cli, len(results), None)

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

    if cli.probe:
        _report_probe(results, usable, cli, summary)
        plots = plot_probe(results, histories, cli, best, plt)
        print(f"\n  Saved: {ART_DIR/'benchmark_wphys_wbc.csv'}")
        print(f"         {ART_DIR/'benchmark_wphys_wbc_best.txt'}")
        print(f"         {ART_DIR/'benchmark_wphys_wbc_settings.txt'}")
        for path in plots:
            print(f"         {path}")
        print("", flush=True)
        return

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

    # ---- test-op boxplot: one box per configuration -------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    labels = [f"p{r['w_phys']:.3g}\nb{r['w_bc']:.3g}" for r in usable]
    draw_config_boxes(ax2, usable, labels, cli, best)
    ax2.set_xlabel("(w_phys, w_bc) combination", fontsize=11)
    ax2.set_title(f"Held-out {cli.test_op}: per-sensor error pooled over "
                  f"{N_BOX_TIMES} random time points", fontsize=12)
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
