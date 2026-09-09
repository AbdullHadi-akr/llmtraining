#!/usr/bin/env python3
"""Run a grid of train.py configurations, several at a time, and pool the seeds.

The tool Teil I of the Fahrplan asks for. It exists because **a MAE difference
between two runs is not readable without the spread over seeds next to it**:
6.27 against 6.51 is not a result if the same configuration wanders between 5.9
and 6.8 when only the seed changes. Every result this project has produced so
far was missing that number.

    python3 PINNmodulusTwo/sweep.py --seeds 0 1 2 \\
        --vary delta-phys 1.0 0.4 0.2 -- --epochs 20

    -> artifacts/sweep/<point>/   one directory per (configuration, seed):
                                  history.csv, op_metrics.csv, metrics.txt,
                                  model.pt, the plots and train.log
    -> artifacts/sweep.csv        one row per (configuration, seed)
    -> stdout                     mean and std per configuration over val_ops

Everything after a bare ``--`` is handed to train.py untouched, so the sweep
needs no knowledge of train.py's flags and cannot fall out of step with them.

Why it runs the points in PARALLEL
----------------------------------
A single run of this model leaves the GPU almost entirely idle, and not for want
of tuning: it is ~7000 sequential rollout steps per OP per epoch, each one a
363x128 matmul. That is ~50 CUDA kernels of ~5 us launch latency against ~1.5 us
of arithmetic, so the card spends its time waiting for Python to hand it the next
tiny matmul. Peak VRAM is ~0.4 GB of the 15 available and the SMs are nearly
empty. No batch size fixes this -- ``batch_data``/``batch_phys``/``batch_bc`` are
experiment knobs, and turning them up to fill memory changes the gradient noise,
which is the one thing a sweep must hold still.

What DOES fill the card is more independent work resident at once, and a sweep is
made of nothing else: the points and the seeds are separate experiments that must
not influence each other. Running them as separate processes gives that for free
and gives it exactly: no shared RNG, no shared allocator, no shared optimiser
state. That property is the whole reason this is subprocesses and not one
process looping over ``train.fit()``.

One caveat, measured rather than assumed. The runs are bit-identical to a
standalone ``train.py`` **at the same thread count**, and this script pins its
children to one thread (see below), so a sweep row will not match a hand-started
run that used the machine's default. Verified on the synthetic fixture: with
``OMP_NUM_THREADS=1`` on both sides every weight tensor and every prediction
array matched bit for bit; against a default-threaded run the numbers drifted in
the 8th significant digit, because multi-threaded CPU reductions do not fix their
summation order. Within a sweep this cannot bite -- every point gets the same
environment, which is what makes the points comparable to each other. To compare
against an existing hand-run number, use ``--threads`` to match it.

The ceiling is the CPU, not the GPU. Each run is one Python loop issuing millions
of kernel launches, so it wants a core to itself; on a g4dn.2xlarge (8 vCPU, 4
physical) about 4-6 concurrent runs is the useful range, landing around 4-6 GB of
VRAM. Two things matter and are handled here: the children get
``OMP_NUM_THREADS=1`` so they do not each spawn a thread pool and fight over the
same cores, and each gets its own ``--artifacts-dir`` -- without that they would
all write ``artifacts/model.pt``, which is exactly the collision the Fahrplan
warns about.

Worth doing on the instance, once, outside this script: start the CUDA MPS daemon
(``nvidia-cuda-mps-control -d``). Without it the driver time-slices between
processes rather than letting their kernels share the SMs.

What this deliberately does NOT do
----------------------------------
No plots, no resume, no checkpoint merge -- the Fahrplan puts those last, and
only for the axis that really costs hours. A crashed or diverged point is
recorded as a failure and the sweep carries on; it is never worth losing eleven
finished points to the twelfth.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRAIN_PY = THIS_DIR / "train.py"
ART_DIR = THIS_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="grid of train.py runs, executed in parallel, pooled over seeds",
        epilog="everything after a bare -- is passed to train.py unchanged",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="one run per configuration per seed. Three is the "
                        "minimum that says anything about the spread")
    p.add_argument("--vary", nargs="+", action="append", default=[],
                   metavar=("FLAG", "VALUE"),
                   help="an axis: the train.py flag WITHOUT its leading dashes, "
                        "then its values. Repeat for a grid, e.g. "
                        "--vary delta-phys 1.0 0.4 0.2 --vary w-bc 0.1 0.0")
    p.add_argument("--jobs", "-j", type=int, default=0,
                   help="runs in flight at once. 0 = min(4, cores/2). Each run "
                        "wants a core to itself; the GPU is not the limit")
    p.add_argument("--out", default=str(ART_DIR / "sweep"),
                   help="root for the per-run artifact directories")
    p.add_argument("--csv", default=str(ART_DIR / "sweep.csv"),
                   help="where the one-row-per-run summary goes")
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads per run (OMP_NUM_THREADS and friends). 1 is "
                        "right when several runs share the machine, and it also "
                        "makes the runs reproducible: multi-threaded CPU "
                        "reductions do not fix their summation order, so the "
                        "thread count moves the last digits. Raise it only to "
                        "reproduce a number from a run that used more")
    p.add_argument("--dry-run", action="store_true",
                   help="print the runs that would be started, then stop")
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    args = p.parse_args(argv)
    args.passthrough = passthrough
    return args


def build_points(args) -> list[dict]:
    """The cartesian product of the axes, times the seeds. One dict per run."""
    axes: list[tuple[str, list[str]]] = []
    for spec in args.vary:
        if len(spec) < 2:
            raise SystemExit(f"--vary needs a flag and at least one value, got {spec!r}")
        axes.append((spec[0].lstrip("-"), [str(v) for v in spec[1:]]))

    combos = [dict(zip([a for a, _ in axes], vals))
              for vals in itertools.product(*[v for _, v in axes])] or [{}]
    points = []
    for combo in combos:
        for seed in args.seeds:
            points.append({"config": combo, "seed": seed})
    return points


def slug(config: dict, seed: int) -> str:
    """Directory name for one run. Stable, sorted, filesystem-safe."""
    parts = [f"{k}={v}" for k, v in sorted(config.items())] or ["base"]
    safe = "__".join(parts).replace("/", "_").replace(" ", "")
    return f"{safe}__seed={seed}"


def config_key(config: dict) -> str:
    """Identifies a CONFIGURATION, i.e. a point pooled over its seeds."""
    return ", ".join(f"{k}={v}" for k, v in sorted(config.items())) or "base"


def run_one(point: dict, out_root: Path, passthrough: list[str],
            threads: int = 1) -> dict:
    """Start one train.py, wait for it, return what happened.

    Never raises on a failing run. A point that diverges or crashes is a result
    (``ok=False``) and the sweep keeps going -- losing the finished points to the
    unfinished one would be the expensive mistake here.
    """
    run_dir = out_root / slug(point["config"], point["seed"])
    cmd = [sys.executable, str(TRAIN_PY),
           "--seed", str(point["seed"]),
           "--artifacts-dir", str(run_dir)]
    for flag, value in sorted(point["config"].items()):
        cmd += [f"--{flag}", str(value)]
    cmd += passthrough

    env = dict(os.environ)
    # Each run is one Python loop driving millions of tiny kernel launches, so it
    # wants ONE core. Left at the default, every child would open a thread pool
    # the width of the machine and the runs would spend their time preempting
    # each other rather than training.
    #
    # It is also what makes a run reproducible: multi-threaded CPU reductions do
    # not fix their summation order, so the thread count alone moves the last
    # digits. Every point in a sweep gets the same value, so points stay
    # comparable to each other whatever it is set to.
    threads = str(max(1, int(threads)))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads)

    t0 = time.time()
    log = run_dir / "train.log"
    # A non-zero exit is an ordinary result and needs no handler (no check=True).
    # This guard is for the launch itself failing -- an unwritable directory, a
    # missing interpreter, the OS refusing a process. Letting that escape would
    # abort ThreadPoolExecutor.map and throw away every finished point with it,
    # which is the one outcome a sweep must never have.
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        rc = proc.returncode
    except KeyboardInterrupt:
        raise
    except Exception as exc:                       # noqa: BLE001 - deliberate
        rc, secs = -1, time.time() - t0
        print(f"  [FAIL] {run_dir.name}  could not be started: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {**point, "dir": run_dir, "ok": False, "seconds": secs,
                "returncode": rc}
    secs = time.time() - t0
    ok = rc == 0 and (run_dir / "op_metrics.csv").exists()
    print(f"  [{'ok  ' if ok else 'FAIL'}] {run_dir.name}  {secs / 60:.1f} min"
          + ("" if ok else f"  (exit {rc}, see {log})"), flush=True)
    return {**point, "dir": run_dir, "ok": ok, "seconds": secs, "returncode": rc}


def read_val_mae(run_dir: Path) -> dict:
    """``{op_id: mae}`` for the val OPs of one finished run.

    Reads op_metrics.csv, which train.py writes next to metrics.txt for exactly
    this purpose: the human table is fixed-width and carries footnotes, and
    parsing it here would break the first time a column is widened.
    """
    path = run_dir / "op_metrics.csv"
    if not path.exists():
        return {}
    # Tolerant on purpose: a run killed mid-write leaves a truncated file, and
    # that is a point to report as unusable, never a reason to lose the sweep's
    # other results at the very last step.
    try:
        rows = path.read_text().strip().splitlines()
        if len(rows) < 2:
            return {}
        header = rows[0].split(",")
        i_op, i_role, i_mae = header.index("op"), header.index("role"), header.index("mae")
        out = {}
        for line in rows[1:]:
            cells = line.split(",")
            if len(cells) > max(i_op, i_role, i_mae) and cells[i_role] == "val":
                out[cells[i_op]] = float(cells[i_mae])
        return out
    except (OSError, ValueError, IndexError) as exc:
        print(f"  [WARN] {path} is not readable as a metrics table "
              f"({type(exc).__name__}): treating the point as unusable.", flush=True)
        return {}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    """Population std. With three seeds the sample correction is noise about noise."""
    if len(xs) < 2:
        return float("nan")
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def write_csv(results: list[dict], path: Path) -> list[str]:
    """One row per (configuration, seed). Returns the val OP ids it found."""
    val_ops: list[str] = []
    for r in results:
        for op in r.get("val_mae", {}):
            if op not in val_ops:
                val_ops.append(op)
    val_ops.sort()

    axes = sorted({k for r in results for k in r["config"]})
    header = (["config", "seed", "ok", "minutes", "val_mae_mean"]
              + [f"mae_{op}" for op in val_ops] + axes)
    lines = [",".join(header)]
    for r in results:
        vm = r.get("val_mae", {})
        mean = _mean([vm[op] for op in val_ops if op in vm])
        cells = [config_key(r["config"]).replace(",", ";"), str(r["seed"]),
                 "1" if r["ok"] else "0", f"{r['seconds'] / 60:.2f}",
                 f"{mean:.6g}"]
        cells += [f"{vm[op]:.6g}" if op in vm else "" for op in val_ops]
        cells += [str(r["config"].get(a, "")) for a in axes]
        lines.append(",".join(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return val_ops


def report(results: list[dict], val_ops: list[str]) -> None:
    """Mean and std per configuration over its seeds. The point of the exercise."""
    by_config: dict = {}
    for r in results:
        by_config.setdefault(config_key(r["config"]), []).append(r)

    print("\n" + "=" * 78)
    print("val-MAE per configuration, pooled over seeds (C)")
    print("=" * 78)
    print(f"{'configuration':<34} {'n':>2} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    ranked = []
    for key, runs in by_config.items():
        means = [_mean([r["val_mae"][op] for op in val_ops if op in r["val_mae"]])
                 for r in runs if r["ok"] and r.get("val_mae")]
        means = [m for m in means if m == m]          # drop NaN
        if not means:
            print(f"{key:<34} {len(runs):>2}   no finished run")
            continue
        ranked.append((_mean(means), key, means, len(runs)))
        print(f"{key:<34} {len(means):>2} {_mean(means):>8.3f} {_std(means):>8.3f} "
              f"{min(means):>8.3f} {max(means):>8.3f}")

    if len(ranked) > 1:
        ranked.sort()
        best, second = ranked[0], ranked[1]
        gap = second[0] - best[0]
        # NaN is dropped rather than passed to max(): _std returns it for a
        # single finished seed, and ``max(nan, 0.5)`` and ``max(0.5, nan)``
        # disagree in Python -- the verdict would then depend on which
        # configuration happened to rank first.
        spreads = [v for v in (_std(best[2]), _std(second[2])) if v == v]
        print()
        # The one sentence this whole tool exists to be able to say.
        if not spreads:
            print(f"  [NO SPREAD] {best[1]!r} leads {second[1]!r} by {gap:.3f} C, "
                  f"and neither has more than one finished seed. Two single "
                  f"measurements are not a comparison -- there is nothing here "
                  f"to read yet. Run more seeds.")
        elif gap < max(spreads):
            print(f"  [NOT SEPARATED] {best[1]!r} leads {second[1]!r} by "
                  f"{gap:.3f} C, but the seed spread is {max(spreads):.3f} C. "
                  f"That is not a difference -- it is the same configuration "
                  f"measured twice. More seeds, or a wider axis.")
        else:
            print(f"  {best[1]!r} beats {second[1]!r} by {gap:.3f} C against a "
                  f"seed spread of {max(spreads):.3f} C.")


def main() -> None:
    args = parse_args()
    if not TRAIN_PY.exists():
        raise SystemExit(f"train.py not found next to sweep.py ({TRAIN_PY})")
    points = build_points(args)
    out_root = Path(args.out).expanduser().resolve()
    jobs = args.jobs or max(1, min(4, (os.cpu_count() or 2) // 2))
    jobs = min(jobs, len(points))

    # Wall time is ceil(n / jobs) run-durations, not n/jobs: the pool refills as
    # runs finish, but a last wave with fewer runs than workers still costs a
    # whole duration. With 9 runs (3 points x 3 seeds, one Fahrplan axis) that
    # makes -j 4 exactly as fast as -j 3 and a core busy for nothing.
    waves = -(-len(points) // jobs)
    print(f"{len(points)} runs ({len(points) // max(1, len(args.seeds))} "
          f"configuration(s) x {len(args.seeds)} seed(s)), {jobs} at a time "
          f"-> {waves} run-duration(s) of wall time")
    if len(points) % jobs:
        same = min(j for j in range(1, jobs + 1) if -(-len(points) // j) == waves)
        if same < jobs:
            print(f"  [HINT] -j {same} finishes in the same {waves} run-duration(s) "
                  f"and leaves {jobs - same} core(s) free -- with {len(points)} runs "
                  f"the last wave is ragged. A -j that divides {len(points)} wastes "
                  f"nothing.")
    print(f"artifacts: {out_root}")
    if args.passthrough:
        print(f"passed to train.py: {' '.join(args.passthrough)}")
    if args.dry_run:
        for pt in points:
            print(f"  {slug(pt['config'], pt['seed'])}")
        return

    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"--out {out_root} cannot be created: "
                         f"{type(exc).__name__}: {exc}")
    t0 = time.time()
    # Threads, not processes: each worker only waits on a subprocess, so the GIL
    # is never held and the real parallelism is the train.py processes themselves.
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(
                lambda pt: run_one(pt, out_root, args.passthrough, args.threads),
                points))
    except KeyboardInterrupt:
        print("\n[interrupted] finished runs keep their artifacts; rerun to "
              "redo the rest.", flush=True)
        raise SystemExit(130)

    for r in results:
        r["val_mae"] = read_val_mae(r["dir"]) if r["ok"] else {}

    val_ops = write_csv(results, Path(args.csv))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} runs finished in {(time.time() - t0) / 60:.1f} "
          f"min wall time (sum of the runs: "
          f"{sum(r['seconds'] for r in results) / 60:.1f} min)")
    print(f"wrote {args.csv}")
    if not val_ops:
        print("  [NOTE] no val OPs in any op_metrics.csv -- pass --val-ops to "
              "train.py, or every number here is in-sample.")
        return
    report(results, val_ops)


if __name__ == "__main__":
    main()
