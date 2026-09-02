#!/usr/bin/env python3
"""Run one configuration axis across several seeds and report the spread.

Why this exists
---------------
Every result this project has produced so far came from a single run, and twice
that was enough to draw a conclusion the next run overturned -- ``spread`` after
three epochs (5b), ``ratio_bc`` after one (O12). On 02.09. it nearly happened a
third time: Axis 0 put the no-physics run 0.74 C ahead of Step 6 on OP06, and
nobody can say whether 0.74 C is a difference or a seed.

That is the whole job here. Not a framework -- a loop around ``train.fit`` that
runs the same configuration more than once and prints the two numbers side by
side:

    spread BETWEEN configurations   vs   spread BETWEEN seeds

FAHRPLAN §10 already states the verdict that follows: if the first is smaller
than the second, there is no ranking, and any table claiming one is noise.

Usage
-----
    python3 PINNmodulusTwo/sweep.py --axis w_phys --values 0 0.1 --seeds 0 1 2
    python3 PINNmodulusTwo/sweep.py --axis delta_phys --values 1.0 0.4 0.2 --seeds 0 1 2

Anything not named by ``--axis`` comes from ``config.yaml`` and the usual
``train.py`` flags, which are accepted here unchanged and passed through.

Cost, before you start it
-------------------------
One 60-epoch run is ~2 h on CPU. ``--values 0 0.1 --seeds 0 1 2`` is six of
them. Rows are appended to ``artifacts/sweep.csv`` as each run FINISHES, and a
run whose row is already there is skipped, so a crash at hour nine costs one run
and not nine. That is not the "resume" the roadmap defers -- there is no
mid-training checkpoint here, only the refusal to throw away finished work.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics as st
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "tools"))

import train as train_mod  # noqa: E402
from data import build_op  # noqa: E402
from op_metrics import op_metrics, rollout_phys  # noqa: E402

ART_DIR = train_mod.ART_DIR
SWEEP_CSV = ART_DIR / "sweep.csv"
RUN_DIR = ART_DIR / "sweep_runs"

# The window the per-epoch series are summarised over. Never the last epoch --
# see FAHRPLAN §11.6 and tools/analyse_history.py, which applies the same rule.
TAIL_EPOCHS = 30

# FAHRPLAN §10: spread is a SIDE CONDITION. A configuration outside this band has
# stopped predicting rather than won, whatever its MAE says.
SPREAD_OK = (0.7, 1.3)


def _summarise_tail(series, last=TAIL_EPOCHS):
    """Median over the trailing window, or nan. The rule, applied."""
    vals = [float(v) for v in series[-last:]
            if v is not None and float(v) == float(v)]
    return st.median(vals) if vals else float("nan")


def _mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x == x]
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def _rows_done(path: Path) -> set:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["axis"], r["value"], r["seed"]) for r in csv.DictReader(fh)}


def run_one(args, axis, value, seed, val_ops, test_ops):
    """Train once and score every held-out OP. Returns a flat dict for the CSV."""
    setattr(args, axis, value)
    args.seed = int(seed)

    t0 = time.time()
    model, bundle, _ops, _dtn, history = train_mod.fit(args)
    device = next(model.parameters()).device

    row = {"axis": axis, "value": value, "seed": seed,
           "epochs": int(args.epochs),
           "spread_time": _summarise_tail(history["spread_time"]),
           "spread_space": _summarise_tail(history["spread_space"]),
           "L_data": _summarise_tail(history["L_data"]),
           "aborted": bool(history.get("aborted", False))}

    for op_id in list(val_ops) + list(test_ops):
        op_data = build_op(op_id, bundle, subsample_time=args.subsample)
        pred = rollout_phys(model, op_data, bundle, device)
        m = op_metrics(pred, op_data, late_is_holdout=True)
        row[f"mae_{op_id}"] = m["mae"]
        row[f"late_{op_id}"] = m["late_mae"]
        row[f"biasfrac_{op_id}"] = m["late_bias_frac"]

    row["val_mean"] = _mean([row[f"mae_{o}"] for o in val_ops])
    row["test_mean"] = _mean([row[f"mae_{o}"] for o in test_ops])
    row["wall_s"] = round(time.time() - t0, 1)

    # Keep the per-epoch evidence: fit() overwrites artifacts/history.csv every
    # epoch, so without this every run but the last would be unreadable
    # afterwards -- and the per-epoch series is where O15 and the excursions live.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    src = ART_DIR / "history.csv"
    if src.is_file():
        shutil.copyfile(src, RUN_DIR / f"{axis}={value}_seed{seed}_history.csv")
    return row


def report(rows, val_ops, test_ops) -> None:
    """Print the comparison FAHRPLAN §10 asks for, and its verdict."""
    by_value: dict = {}
    for r in rows:
        by_value.setdefault(str(r["value"]), []).append(r)

    print()
    print("=" * 78)
    print(f"{'value':>10}{'n':>4}{'val mean':>11}{'val std':>10}"
          f"{'test mean':>11}{'test std':>10}{'spread_t':>10}")
    print("-" * 78)
    means = {}
    seed_stds = []
    for value, rs in by_value.items():
        vm = [r["val_mean"] for r in rs]
        tm = [r["test_mean"] for r in rs]
        sp = [r["spread_time"] for r in rs]
        means[value] = _mean(vm)
        seed_stds.append(_std(vm))
        print(f"{value:>10}{len(rs):>4}{_mean(vm):>11.3f}{_std(vm):>10.3f}"
              f"{_mean(tm):>11.3f}{_std(tm):>10.3f}{_mean(sp):>10.3f}")
    print("-" * 78)

    # The verdict. Two spans, one comparison, and it is the only thing that
    # licenses a ranking.
    if len(means) > 1:
        config_span = max(means.values()) - min(means.values())
        seed_span = max(seed_stds) if seed_stds else 0.0
        print(f"\nspan between configurations: {config_span:.3f} C")
        print(f"largest std between seeds:   {seed_span:.3f} C")
        if config_span < seed_span:
            print("\n  NO RANKING. The configurations differ by less than one "
                  "configuration\n  differs from itself across seeds. Report the "
                  "numbers, do not order them.\n  (FAHRPLAN §10)")
        else:
            best = min(means, key=lambda k: means[k])
            print(f"\n  {axis_label(rows)} = {best} has the lower val mean, and the "
                  f"gap exceeds the seed\n  spread. Check the side condition before "
                  f"calling it a winner:")
            for value, rs in by_value.items():
                sp = _mean([r["spread_time"] for r in rs])
                ok = SPREAD_OK[0] <= sp <= SPREAD_OK[1]
                print(f"    {value:>8}: spread_time {sp:.3f} "
                      f"{'ok' if ok else '<-- OUTSIDE [0.7, 1.3], not a winner'}")

    # Held-out OPs one by one: a mean over tiers mixes questions (FAHRPLAN §10).
    print(f"\nper-OP val mean{'':<2}" + "".join(f"{o:>10}" for o in val_ops)
          + "   |" + "".join(f"{o:>10}" for o in test_ops))
    for value, rs in by_value.items():
        cells = "".join(f"{_mean([r[f'mae_{o}'] for r in rs]):>10.3f}" for o in val_ops)
        cells += "   |" + "".join(f"{_mean([r[f'mae_{o}'] for r in rs]):>10.3f}"
                                  for o in test_ops)
        print(f"{value:>15}  " + cells)
    print("=" * 78)


def axis_label(rows):
    return rows[0]["axis"] if rows else "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", required=True,
                    help="the train.py argument to vary, e.g. w_phys, delta_phys, w_bc")
    ap.add_argument("--values", nargs="+", required=True,
                    help="values for that axis; parsed as float")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(SWEEP_CSV))
    ap.add_argument("--rerun", action="store_true",
                    help="ignore rows already in the CSV and run everything again")
    known, rest = ap.parse_known_args(argv)

    # Everything else is a train.py flag, parsed by train.py itself so this file
    # never has to know or duplicate the option list.
    old_argv = sys.argv
    sys.argv = ["train.py"] + rest
    try:
        args = train_mod.parse_args()
    finally:
        sys.argv = old_argv

    if not hasattr(args, known.axis):
        raise SystemExit(f"--axis {known.axis!r} is not a train.py argument. "
                         f"Try one of: w_phys, w_bc, w_data, delta_phys, delta_grid, lr")

    val_ops = list(getattr(args, "val_ops", []) or [])
    test_ops = list(getattr(args, "test_ops", []) or [])
    if not val_ops:
        raise SystemExit("no --val-ops: there is nothing to select on, and a sweep "
                         "scored on training OPs measures memorisation.")

    out = Path(known.out)
    fields = (["axis", "value", "seed", "epochs", "val_mean", "test_mean",
               "spread_time", "spread_space", "L_data", "aborted", "wall_s"]
              + [f"{p}_{o}" for o in val_ops + test_ops
                 for p in ("mae", "late", "biasfrac")])
    done = set() if known.rerun else _rows_done(out)
    todo = [(v, s) for v in known.values for s in known.seeds
            if (known.axis, str(v), str(s)) not in done]

    print(f"sweep: {known.axis} = {known.values} x seeds {known.seeds} "
          f"= {len(known.values) * len(known.seeds)} runs, {len(todo)} to go "
          f"({len(known.values) * len(known.seeds) - len(todo)} already in {out.name})")
    print(f"val={val_ops}  test={test_ops}  epochs={args.epochs}")

    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()

    for i, (value, seed) in enumerate(todo, 1):
        print(f"\n----- [{i}/{len(todo)}] {known.axis}={value} seed={seed} -----",
              flush=True)
        row = run_one(args, known.axis, float(value), seed, val_ops, test_ops)
        row["value"] = value          # keep the label the user typed
        # Append as soon as it exists: the next run is another two hours, and a
        # finished run that only lives in RAM is a finished run that can be lost.
        with out.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore").writerow(row)
        print(f"  val_mean={row['val_mean']:.3f} C  test_mean={row['test_mean']:.3f} C  "
              f"spread_time={row['spread_time']:.3f}  ({row['wall_s']:.0f}s)", flush=True)

    with out.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["axis"] == known.axis]
    for r in rows:
        for k, v in r.items():
            if k not in ("axis", "value", "aborted"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
    report(rows, val_ops, test_ops)
    print(f"\nrows in {out}, per-epoch series in {RUN_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
