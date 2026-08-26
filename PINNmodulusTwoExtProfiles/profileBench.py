#!/usr/bin/env python3
"""profileBench - the Profile Tier Benchmark for the PINNmodulusTwo extension.

What it measures
----------------
The base project's two benchmarks sweep loss weights (``benchmark_wphys_wbc``)
and architecture (``benchmark_arch``) against a single held-out constant OP.
Neither question survives the move to profiles unchanged:

* The loss weights were tuned against ``phys_scale`` / ``Qsrc_scale`` fitted on
  OP01-OP05. Those divisors are refitted here on OP01-OP16, so the same
  ``w_phys`` is a different mixing ratio. The numbers do not transfer and are
  not inherited -- they are swept again.
* The architecture answers were found on data where every driver was constant.
  Whether a wider net helps is a different question once the net has to track a
  CC-CV taper.

And there are questions that did not exist before: does anti-aliased driver
resampling matter, do the driver-rate channels earn their input width, and how
far back does driver memory have to reach.

So this benchmark walks these axes, one at a time against a common baseline:

    resample   mean vs point           - does driver aliasing cost anything
    drivhist   on vs off               - do the driver rate channels earn their keep
    drlags     driver rate segments    - how far back driver memory must reach
    wphys      physics weight          - re-tuned against the NEW normalisation
    wbc        boundary-condition weight
    ratelags   temperature history segments
    width / depth                      - architecture, re-asked with profiles present

One axis at a time rather than a full product grid: the point is to find which
knob moves the error at all. A joint grid over seven axes is several thousand
trainings; this is a few dozen and tells you where the leverage is.

How it scores
-------------
Selection is the MEAN rollout MAE over ``--val-ops`` (default OP06, a constant
OP, and OP09, a fluid-temperature-profile OP). Reporting is per TIER over
``--test-ops``, which are never selected on. See ``op_registry`` for the tier
definitions and ``bench_profiles`` for why selection is a set rather than a
single OP.

RUNTIME. This trains a full model per configuration per seed. At ``subsample 2``
and 60 epochs one training is hours, and the default grid is ~30 configurations,
so the full default sweep is days per seed. Read the seconds-per-epoch the
training log prints before planning a long run, cut ``--epochs`` to 20 for a
first pass, and use ``--axes`` to walk one axis at a time.

Run:
    python3 PINNmodulusTwoExtProfiles/smokeBench.py            # ALWAYS first
    python3 PINNmodulusTwoExtProfiles/profileBench.py --device cuda --seeds 0 1 2

    # only the two questions that are new, at a short epoch budget
    python3 PINNmodulusTwoExtProfiles/profileBench.py \
        --axes resample drivhist drlags --epochs 20

Outputs (in PINNmodulusTwoExtProfiles/artifacts/):
    profileBench.csv        - one row per configuration: mean over seeds + std
    profileBench_perop.csv  - the same runs broken out per OP
    profileBench.png        - one panel per axis, tiers as separate series
    profileBench_box.png    - per-sensor error spread on the first test OP
    profileBench_best.txt   - ranking, per-tier table, and the noise verdicts
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from bench_profiles import (
    METRIC_KEYS, aggregate_seeds, failed_result, noise_verdict, print_eta,
    split_verdict, train_one_seed,
)
from data import require_ops
from device_utils import resolve_device
from op_registry import (
    DEFAULT_TEST_OPS, DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS, TIER_ORDER,
    split_summary, tier_of,
)
from train import fit

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Swept values per axis; each is explored against the CLI baseline for the rest.
DEFAULT_DRIVER_LAG_SETS = [
    [5.0, 20.0],     # the baseline
    [2.0, 10.0],     # shorter memory: only the immediate driver move
    [10.0, 60.0],    # longer memory: reaches across a CC->CV transition
    [1.0],           # one very short segment: essentially a derivative
    [5.0, 20.0, 120.0],  # a third, slow segment on top of the baseline
]
DEFAULT_RATE_LAG_SETS = [
    [5.0, 20.0],
    [2.0, 10.0],
    [10.0, 60.0],
    [30.0],
]
DEFAULT_W_PHYS = [0.0, 0.01, 0.05, 0.1, 0.3]
DEFAULT_W_BC = [0.0, 0.05, 0.1, 0.3]
DEFAULT_WIDTHS = [64, 128, 256]
DEFAULT_DEPTHS = [2, 4, 6]

AXES = ("resample", "drivhist", "drlags", "wphys", "wbc", "ratelags",
        "width", "depth")


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
    bo = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="Profile Tier Benchmark (profileBench)")
    # ---- data / split -------------------------------------------------------
    p.add_argument("--ops", nargs="+", default=d.get("ops", list(DEFAULT_TRAIN_OPS)))
    p.add_argument("--val-ops", nargs="+", default=d.get("val_ops", list(DEFAULT_VAL_OPS)),
                   help="SELECTION set; the ranking uses the mean MAE over these")
    p.add_argument("--test-ops", nargs="+", default=d.get("test_ops", list(DEFAULT_TEST_OPS)),
                   help="reported per tier, never selected on")
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 2))
    p.add_argument("--train-frac", type=float, default=d.get("train_frac", 0.8))
    # ---- what to sweep ------------------------------------------------------
    p.add_argument("--axes", nargs="+", choices=list(AXES),
                   default=["resample", "drivhist", "drlags", "wphys", "wbc"],
                   help="which axes to walk; each is explored against the "
                        "baseline. Default is the five that the profiles make "
                        "new or invalidate; add width/depth/ratelags to re-ask "
                        "the base project's architecture questions.")
    p.add_argument("--driver-lag-sets", nargs="+", type=str, default=None,
                   help="comma-separated driver rate segment sets, e.g. "
                        "'5,20' '10,60'. Default: see DEFAULT_DRIVER_LAG_SETS")
    p.add_argument("--w-phys-values", nargs="+", type=float, default=DEFAULT_W_PHYS)
    p.add_argument("--w-bc-values", nargs="+", type=float, default=DEFAULT_W_BC)
    p.add_argument("--widths", nargs="+", type=int, default=DEFAULT_WIDTHS)
    p.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="one training per seed per configuration; a "
                        "configuration is scored by the MEAN over its seeds")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="write the first seed's weights per configuration")
    # ---- baseline held fixed off-axis --------------------------------------
    p.add_argument("--resample", choices=["mean", "point"],
                   default=d.get("resample", "mean"))
    p.add_argument("--driver-history", action=bo,
                   default=d.get("use_driver_history", True))
    p.add_argument("--driver-rate-lags", nargs="+", type=float,
                   default=d.get("driver_rate_lags", [5.0, 20.0]))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 20.0]))
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2))
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0))
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--w-phys", type=float, default=d.get("w_phys", 0.1),
                   help="baseline physics weight, held fixed off the wphys axis")
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "hybrid"))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--use-static", action=bo, default=d.get("use_static", True))
    p.add_argument("--use-forcing", action=bo, default=d.get("use_forcing", True))
    p.add_argument("--shuffle-ops", action=bo, default=d.get("shuffle_ops", True))
    p.add_argument("--holdout-tail", action=bo, default=d.get("holdout_tail", False))
    # ---- training -----------------------------------------------------------
    p.add_argument("--epochs", type=int, default=d.get("epochs", 60))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    p.add_argument("--device", default=d.get("device", "auto"))
    p.add_argument("--tf32", action="store_true", default=d.get("tf32", False))
    return p.parse_args()


def _lag_label(lags) -> str:
    return "+".join(f"{float(v):g}" for v in lags)


def _parse_lag_sets(raw, fallback):
    if not raw:
        return [list(s) for s in fallback]
    return [[float(v) for v in item.replace(" ", "").split(",") if v]
            for item in raw]


def build_configs(cli) -> list:
    """One entry per configuration: ``(axis, label, overrides)``.

    The baseline appears once per axis it sits on. That is deliberate: each
    panel needs its own reference point, and re-training the same configuration
    on several axes also gives a free read on how much the seeds alone move it.
    """
    base = {
        "resample": cli.resample,
        "driver_history": bool(cli.driver_history),
        "driver_rate_lags": [float(v) for v in cli.driver_rate_lags],
        "rate_lags": [float(v) for v in cli.rate_lags],
        "w_phys": float(cli.w_phys), "w_bc": float(cli.w_bc),
        "width": int(cli.width), "depth": int(cli.depth),
    }
    configs = []

    def add(axis, label, **changes):
        ov = dict(base)
        ov.update(changes)
        configs.append((axis, label, ov))

    if "resample" in cli.axes:
        for mode in ("mean", "point"):
            add("resample", mode, resample=mode)
    if "drivhist" in cli.axes:
        for on in (True, False):
            add("drivhist", "on" if on else "off", driver_history=on)
    if "drlags" in cli.axes:
        for lags in _parse_lag_sets(cli.driver_lag_sets, DEFAULT_DRIVER_LAG_SETS):
            add("drlags", _lag_label(lags), driver_rate_lags=lags,
                driver_history=True)
    if "wphys" in cli.axes:
        for w in cli.w_phys_values:
            add("wphys", f"{float(w):g}", w_phys=float(w))
    if "wbc" in cli.axes:
        for w in cli.w_bc_values:
            add("wbc", f"{float(w):g}", w_bc=float(w))
    if "ratelags" in cli.axes:
        for lags in DEFAULT_RATE_LAG_SETS:
            add("ratelags", _lag_label(lags), rate_lags=list(lags))
    if "width" in cli.axes:
        for w in cli.widths:
            add("width", str(int(w)), width=int(w))
    if "depth" in cli.axes:
        for dpt in cli.depths:
            add("depth", str(int(dpt)), depth=int(dpt))
    return configs


def _is_baseline(row, cli) -> bool:
    axis, label = row["axis"], row["label"]
    if axis == "resample":
        return label == cli.resample
    if axis == "drivhist":
        return label == ("on" if cli.driver_history else "off")
    if axis == "drlags":
        return label == _lag_label(cli.driver_rate_lags)
    if axis == "wphys":
        return label == f"{float(cli.w_phys):g}"
    if axis == "wbc":
        return label == f"{float(cli.w_bc):g}"
    if axis == "ratelags":
        return label == _lag_label(cli.rate_lags)
    if axis == "width":
        return label == str(int(cli.width))
    if axis == "depth":
        return label == str(int(cli.depth))
    return False


def _tier_cell(row, tier, key="mae"):
    t = row.get("per_tier", {}).get(tier)
    return float(t[key]) if t else float("nan")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli = parse_args()
    require_ops(*cli.ops, *cli.val_ops, *cli.test_ops)
    device = resolve_device(cli.device)
    cli.device = str(device)

    configs = build_configs(cli)
    total = len(configs)
    n_runs = total * len(cli.seeds)

    header = [
        "profileBench - Profile Tier Benchmark",
        "free-running rollout, NO teacher forcing; MAE in physical degrees C",
        "",
    ] + split_summary(cli.ops, cli.val_ops, cli.test_ops) + [
        "",
        f"selection = MEAN rollout MAE over {', '.join(cli.val_ops)}",
        f"baseline: resample={cli.resample} driver_history={cli.driver_history} "
        f"driver_rate_lags={_lag_label(cli.driver_rate_lags)}s "
        f"rate_lags={_lag_label(cli.rate_lags)}s w_phys={cli.w_phys} "
        f"w_bc={cli.w_bc} width={cli.width} depth={cli.depth}",
        f"axes = {', '.join(cli.axes)}   seeds = {cli.seeds}",
        f"runs = {total} configurations x {len(cli.seeds)} seed(s) = {n_runs} trainings",
        f"epochs = {cli.epochs}  dt = {0.1*cli.subsample:.1f}s  lr = {cli.lr}  "
        f"holdout_tail = {bool(cli.holdout_tail)}",
        "",
    ]
    print("\n".join(header), flush=True)

    results = []
    start_time_total = time.time()

    for idx, (axis, label, overrides) in enumerate(configs, start=1):
        print(f"\n{'='*66}")
        print(f"[{idx}/{total}] {axis}={label}"
              f"  ({len(cli.seeds)} seed{'s' if len(cli.seeds) > 1 else ''})")
        print(f"{'='*66}", flush=True)
        start = time.time()

        extra = {
            "axis": axis, "label": label,
            "resample": overrides["resample"],
            "drivhist": bool(overrides["driver_history"]),
            "drlags_s": _lag_label(overrides["driver_rate_lags"]),
            "ratelags_s": _lag_label(overrides["rate_lags"]),
            "w_phys": float(overrides["w_phys"]),
            "w_bc": float(overrides["w_bc"]),
            "width": int(overrides["width"]), "depth": int(overrides["depth"]),
        }

        per_seed = []
        for si, seed in enumerate(cli.seeds):
            ckpt = None
            if cli.save_checkpoints and si == 0:
                ckpt = ART_DIR / f"ckpt_{axis}_{label.replace('+', '-')}.pt"
            row, _hist = train_one_seed(
                cli, overrides, seed, device, fit, checkpoint_path=ckpt,
                context={"axis": axis, "label": label},
            )
            if row is not None:
                per_seed.append(row)

        train_time = time.time() - start
        if not per_seed:
            print(f"  [SKIP] {axis}={label}: every seed diverged or crashed - "
                  f"recorded as NaN, sweep continues", flush=True)
            results.append(failed_result(extra, train_time, cli, len(cli.seeds)))
            print_eta(idx, total, start_time_total, train_time)
            continue

        row = aggregate_seeds(extra, per_seed, len(cli.seeds), train_time, cli)
        results.append(row)
        spread = (f" (+/-{row['val_mae_std']:.3f})" if row["n_seeds_ok"] > 1 else "")
        tiers = "  ".join(
            f"{t}={_tier_cell(row, t):.3f}" for t in TIER_ORDER
            if t in row.get("per_tier", {}))
        print(f"  params={row['n_params']}  "
              f"MAE(val)={row['val_mae']:.3f} C{spread}   {tiers}", flush=True)
        print_eta(idx, total, start_time_total, train_time)

    total_time = time.time() - start_time_total
    print(f"\n{'='*66}")
    print(f"Total benchmark time: {total_time/3600:.2f} hours")
    print(f"{'='*66}\n", flush=True)

    # ---- CSVs ---------------------------------------------------------------
    tier_cols = [t for t in TIER_ORDER
                 if any(t in r.get("per_tier", {}) for r in results)]
    csv = ["axis,label,resample,driver_history,driver_rate_lags_s,rate_lags_s,"
           "w_phys,w_bc,width,depth,n_params,L_data,L_phys,L_bc,rate_scale,"
           "MAE_val_C,MAE_val_std_C,"
           + ",".join(f"MAE_{t}_C" for t in tier_cols)
           + ",n_seeds,n_seeds_ok,src_gain,diff_gain,train_time_min"]
    for r in results:
        csv.append(
            f"{r['axis']},{r['label']},{r['resample']},{int(r['drivhist'])},"
            f"\"{r['drlags_s']}\",\"{r['ratelags_s']}\","
            f"{r['w_phys']:g},{r['w_bc']:g},{r['width']},{r['depth']},"
            f"{r['n_params']},{r['L_data']:.6g},{r['L_phys']:.6g},"
            f"{r['L_bc']:.6g},{r['rate_scale']:.6g},"
            f"{r['val_mae']:.4f},{r['val_mae_std']:.4f},"
            + ",".join(f"{_tier_cell(r, t):.4f}" for t in tier_cols)
            + f",{r['n_seeds']},{r['n_seeds_ok']},"
            f"{r['src_gain']:.6g},{r['diff_gain']:.6g},{r['train_time']/60:.2f}"
        )
    (ART_DIR / "profileBench.csv").write_text("\n".join(csv) + "\n")

    # Per-OP breakdown: the tier means are a summary, and a summary of three
    # different extrapolations is exactly the thing worth being able to open up.
    perop = ["axis,label,op,tier,role," + ",".join(METRIC_KEYS) + ",mae_std"]
    for r in results:
        for op_id, m in sorted(r.get("per_op", {}).items()):
            perop.append(
                f"{r['axis']},{r['label']},{op_id},{tier_of(op_id)},{m['role']},"
                + ",".join(f"{m.get(k, float('nan')):.4f}" for k in METRIC_KEYS)
                + f",{m.get('mae_std', float('nan')):.4f}"
            )
    (ART_DIR / "profileBench_perop.csv").write_text("\n".join(perop) + "\n")

    usable = [r for r in results if np.isfinite(r["val_mae"])]
    n_failed = len(results) - len(usable)
    if not usable:
        print(f"All {len(results)} configurations diverged - no result to rank.",
              flush=True)
        print(f"Raw values are in {ART_DIR / 'profileBench.csv'}.", flush=True)
        print("If L_data went non-finite in epoch 1, try --max-rate-amp 50: "
              "the wider pooled normalisation raises the factor by which the "
              "hybrid history magnifies the opening rollout steps.", flush=True)
        return

    best = min(usable, key=lambda r: r["val_mae"])
    label_of = lambda r: f"{r['axis']}={r['label']}"

    th = (f"{'axis':>9} {'value':>12} {'params':>8} | {'MAE_val':>8} {'+/-':>6} | "
          + " ".join(f"{t:>12}" for t in tier_cols))
    summary = header + [th, "-" * len(th)]
    for r in results:
        summary.append(
            f"{r['axis']:>9} {r['label']:>12} {r['n_params']:>8} | "
            f"{r['val_mae']:>8.3f} {r['val_mae_std']:>6.3f} | "
            + " ".join(f"{_tier_cell(r, t):>12.3f}" for t in tier_cols)
        )

    # How much does each axis move the error at all? An axis whose whole span is
    # narrower than the seed spread is not worth tuning further.
    summary += ["", "Span per axis (max - min selection MAE over the axis):"]
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
        summary.append(f"  {axis:>9}: {span:6.3f} C   ({verdict})")

    baseline = next((r for r in usable if _is_baseline(r, cli)), None)
    summary += [
        "",
        f"BEST (by mean MAE over {', '.join(cli.val_ops)}): {label_of(best)}",
        f"  -> val {best['val_mae']:.3f} C",
    ]
    for tier in tier_cols:
        t = best.get("per_tier", {}).get(tier)
        if t:
            summary.append(
                f"  -> {tier} ({', '.join(t['ops'])}): MAE {t['mae']:.3f} C, "
                f"peak_err {t['peak_err']:.3f} C, "
                f"transient {t['mae_transient']:.3f} C, "
                f"quiescent {t['mae_quiescent']:.3f} C"
            )
    if baseline is not None and baseline is not best:
        summary.append(
            f"  baseline {label_of(baseline)}: val {baseline['val_mae']:.3f} C "
            f"-> improvement {baseline['val_mae'] - best['val_mae']:.3f} C")
    summary += split_verdict(best, cli.val_ops)
    summary += noise_verdict(usable, best, len(cli.seeds), label_of)
    summary += [
        "",
        "Reading the tiers: T1 is generalisation inside the trained envelope, T2",
        "is a profile whose TYPE was trained on, T3 is outside the envelope",
        "entirely. A configuration that wins T1/T2 and loses T3 is not broken --",
        "T3 is an extrapolation and was never selected on. Do not quote a T3",
        "number as the model's accuracy without saying which OP it came from;",
        "profileBench_perop.csv has them separately for exactly that reason.",
        f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} min)",
    ]
    if n_failed:
        summary.append(f"Diverged (recorded as NaN, excluded): "
                       f"{n_failed}/{len(results)}")
    (ART_DIR / "profileBench_best.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[len(header):]), flush=True)

    # ---- one panel per axis -------------------------------------------------
    axes_present = [a for a in cli.axes if any(r["axis"] == a for r in results)]
    if axes_present:
        fig, panels = plt.subplots(1, len(axes_present),
                                   figsize=(5.0 * len(axes_present), 4.4),
                                   squeeze=False)
        for col, axis in enumerate(axes_present):
            rows = [r for r in results if r["axis"] == axis]
            ax = panels[0][col]
            x = np.arange(len(rows))
            ax.errorbar(x, [r["val_mae"] for r in rows],
                        yerr=[r["val_mae_std"] for r in rows], fmt="o-",
                        capsize=4, color="#1f77b4", lw=2,
                        label=f"selection ({'+'.join(cli.val_ops)})")
            for t, style in zip(tier_cols, ("s--", "^:", "v-.", "d--")):
                vals = [_tier_cell(r, t) for r in rows]
                if not np.isfinite(vals).any():
                    continue
                ax.plot(x, vals, style, alpha=0.75, label=t)
            for i, r in enumerate(rows):
                if _is_baseline(r, cli):
                    ax.axvline(i, color="gray", ls=":", lw=1.5, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right")
            ax.set_xlabel(axis)
            ax.set_ylabel("rollout MAE [C]")
            ax.set_title(f"{axis} (dotted = baseline)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(
            f"profileBench - {len(cli.seeds)} seed(s), error bars = seed std",
            fontsize=12)
        fig.tight_layout()
        fig.savefig(ART_DIR / "profileBench.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {ART_DIR / 'profileBench.png'}", flush=True)

    # ---- per-sensor error spread on the first test OP -----------------------
    # One box per configuration, pooling every SENSOR at the fixed sampled time
    # points. The box is the middle 50% of those errors, the line the median and
    # the dots what lies past 1.5 IQR -- so a configuration whose mean looks
    # acceptable while a handful of sensors are far off shows up as a long upper
    # whisker instead of hiding inside the mean. Every configuration is scored at
    # the SAME instants (fixed seed in op_metrics.box_time_idx); boxes taken from
    # different points of the trajectory would not be comparable and nothing in
    # the plot would reveal it.
    boxed = [r for r in usable if r["box_errors"].size]
    if boxed:
        show = sorted(boxed, key=lambda r: r["val_mae"])[:6]
        fig, ax = plt.subplots(figsize=(1.6 * max(len(show), 3) + 3, 4.4))
        data = [r["box_errors"].ravel() for r in show]
        labels = [f"{r['axis']}={r['label']}" for r in show]
        # positional labels moved from `labels=` to `tick_labels=` in
        # matplotlib 3.9; setting the ticks afterwards works on both.
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("absolute error [C]")
        ax.set_title(f"per-sensor error spread on {show[0]['box_op']} "
                     f"({tier_of(show[0]['box_op'])}), "
                     f"{show[0]['box_errors'].shape[0]} sampled times, "
                     f"best {len(show)} configurations")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(ART_DIR / "profileBench_box.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {ART_DIR / 'profileBench_box.png'}", flush=True)


if __name__ == "__main__":
    main()
