#!/usr/bin/env python3
"""Write a synthetic OP cache so the local scripts run on a bare checkout.

Why this exists
---------------
``data_cache/`` and ``material_properties/`` are both in ``.gitignore``, so a
fresh clone can run the unit tests and ``selftest.py`` and nothing else:
``smallBench.py``, ``train.py`` and every benchmark stop at the first
``np.load`` of a missing ``OP*.npz``. That is also why the MAE table in
``README_ERSTER_TEST.md`` chapter 6 could never be reproduced anywhere -- the
synthetic bundle those numbers came from lived on one machine and was never
written down.

This script writes that bundle back into the repo's own preferred cache
location, in the exact ``.npz`` layout ``data.py:_read_raw`` expects. It is a
SMOKE fixture, not a physics model: it exists so that "does the pipeline run
end to end, and do the diagnostics report what they claim" can be answered
without the measured data. Absolute MAE numbers from it mean nothing about the
real OPs -- see ``README_ERSTER_TEST.md`` chapter 9.

Two properties are deliberately built in, because they are what the smoke test
actually checks:

* ``dT/dx = 0`` at ``x = 0``. The field is a cosine in x with its maximum on
  the ``x = 0`` plane, so the Neumann BC ``physics.py:boundary_condition_loss``
  imposes is one the synthetic labels genuinely satisfy. A BC term that cannot
  come down on this data is a bug in the term, not in the data.
* A source-driven transient. ``T`` relaxes towards a steady state on a time
  constant, driven by the same ``q_source`` column the loader reads, so
  ``dTdt_scale`` and the hybrid history amplification ``A`` are finite and
  non-trivial rather than an artefact of noise.

Usage
-----
    python3 PINNmodulusTwo/tools/make_synthetic_cache.py
    python3 PINNmodulusTwo/tools/make_synthetic_cache.py --ops OP01 OP02 OP07
    python3 PINNmodulusTwo/tools/make_synthetic_cache.py --seconds 300

Then the ordinary local commands work unchanged:

    python3 PINNmodulusTwo/smallBench.py --ops OP01 OP02 --test-op OP07

Delete ``data_cache/`` again before touching the real data -- a synthetic
bundle and a measured one must never be mixed in the same cache directory,
which is why every file written here carries ``synthetic: True`` and
``synthetic_note`` and why ``smallBench.py`` prints a banner when it loads one.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_DIR = THIS_DIR.parent
PROJECT_ROOT = PKG_DIR.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from data import CONFIG_ORDER, N_JR1_POINTS, PREFERRED_DATA_CACHE  # noqa: E402

# Three x planes, one per layer: 121 points each by default, which is the split
# materials.py demonstrates in its own __main__ and the JR1 count data.py
# hardcodes as N_JR1_POINTS.
#
# The count is a DEFAULT, not a constant. materials.py reads one value per point
# out of the per-layer CSVs and assigns it into a mask of that many points, so a
# grid whose layers do not match the CSVs' column count raises a shape error.
# On the machine that has the measured material_properties/ those column counts
# are whatever the real grid has -- so when that folder is present the grid is
# sized from it rather than from this number.
DEFAULT_N_PER_LAYER = 121
LAYERS = ("cc", "jr1c", "g")          # x plane 0, 1, 2

# Cell geometry in metres. x is the thin direction and x=0 is the cell-centre
# symmetry plane the BC is imposed on.
LX, LY, LZ = 0.030, 0.100, 0.150

RAW_DT = 0.1                          # data.py assumes the raw grid is 0.1 s


def _csv_columns(path: Path) -> int | None:
    """Number of data columns in one of materials.py's row CSVs, or None."""
    try:
        return len(path.read_text().splitlines()[0].split(","))
    except Exception:
        return None


def points_per_layer() -> int:
    """How many points each layer gets.

    Derived from the installed ``material_properties/`` when it is there, so a
    synthetic cache built on the machine with the measured properties still
    lines up with them, and falls back to 121 on a bare checkout.
    """
    mat_dir = PKG_DIR / "material_properties"
    counts = [
        _csv_columns(mat_dir / "Cell Center" / "Density_Grid_CellCenter.csv"),
        _csv_columns(mat_dir / "JR1 Center"
                     / "ThermalConductivityXX_Grid_JR1Center.csv"),
    ]
    counts = [c for c in counts if c]
    if not counts:
        return DEFAULT_N_PER_LAYER
    if len(set(counts)) != 1:
        raise SystemExit(
            f"material_properties/ has {counts[0]} cell-centre columns but "
            f"{counts[1]} JR1 columns. This script lays one point per column on "
            f"each layer and cannot resolve a mismatch -- pass --force-materials "
            f"to replace the folder with the stand-in, or fix the CSVs.")
    return counts[0]


def _plane(n: int) -> np.ndarray:
    """``n`` points on a roughly square lattice covering the (y, z) face.

    A perfect square (the 121 = 11x11 default) comes out as one; anything else
    is padded along z and the surplus trimmed, so any per-layer count works.
    """
    ny = int(np.ceil(np.sqrt(n)))
    nz = int(np.ceil(n / ny))
    yy, zz = np.meshgrid(np.linspace(0.0, LY, ny),
                         np.linspace(0.0, LZ, nz), indexing="ij")
    return np.stack([yy.ravel(), zz.ravel()], axis=1)[:n]


def _grid(n_per_layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, layer)`` for the synthetic grid."""
    plane = _plane(n_per_layer)
    xs = np.linspace(0.0, LX, len(LAYERS))
    xyz, layer = [], []
    for x, name in zip(xs, LAYERS):
        xyz.append(np.column_stack([np.full(len(plane), x), plane]))
        layer.append(np.full(len(plane), name))
    return np.concatenate(xyz).astype(np.float64), np.concatenate(layer)


def _config_for(op_id: str, rng: np.random.Generator) -> dict:
    """Seven config scalars that genuinely differ between OPs.

    They have to differ: ``data.py`` marks a channel inactive when its pooled
    training variance is below 1e-6 and zeroes the feature, so a cache where
    every OP shares one config trains the network on an all-zero config block
    and the held-out OP becomes indistinguishable from the training ones.
    """
    idx = int(op_id[2:])
    return {
        "c_rate": 0.5 + 0.25 * idx,
        "cell_current": 20.0 + 5.0 * idx,
        "fluid_initial_temp": 293.15 + 1.5 * idx,
        "fluid_inlet_temp": 293.15 + 1.0 * idx,
        "fluid_mass_flow": 0.010 + 0.002 * idx,
        "soc_start": min(0.95, 0.30 + 0.08 * idx),
        "solid_initial_temp": 293.15 + 1.5 * idx + 0.1 * rng.standard_normal(),
    }


def _field(t: np.ndarray, xyz: np.ndarray, layer: np.ndarray, cfg: dict,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T, q_source)`` for one OP.

    ``T`` is a separable transient: one time profile per point, scaled by a
    spatial shape whose x-derivative vanishes at x=0. The JR1 plane runs
    hottest because that is where the source sits.
    """
    n_t, n_points = len(t), xyz.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Both time constants are fractions of the trajectory, not fixed seconds, so
    # the shape stays the same in NORMALISED time and the fixture keeps its
    # calibration at any ``--seconds``. What is being calibrated is
    #     A = T_span_ref / (lag * dTdt_scale)
    # -- the hybrid history amplification, which is what decides whether the
    # rollout sits in the regime that actually destabilises (README_ERSTER_TEST
    # chapter 8). Trajectory length alone does not: a short window lands at
    # A ~ 10 and never exercises it.
    #
    # ``span / 2.5`` is fitted, not guessed. Measured with tools/data_probe.py
    # on the seven default OPs at the default 1445 s:
    #
    #     tau_T      dTdt_scale     A(5 s)   A(20 s)
    #     span/2.5      2.512        115.0     28.8     <- this one
    #     span/3.5      2.685        107.6     26.9
    #     span/6        3.123         92.5     23.1
    #     span/12       4.126         70.0     17.5
    #
    # against the measured data's dTdt_scale = 2.479 and A = 118.9 / 29.7, i.e.
    # 1% on the scale and 3% on both amplifications.
    span = float(t[-1] - t[0]) or 1.0

    # ---- source: a C-rate dependent ramp that settles, in watts -------------
    c_rate = cfg["c_rate"]
    tau_q = span / 36.0
    jr1_w = 4.0 * c_rate * (1.0 - np.exp(-t / tau_q))
    jr2_w = 0.15 * jr1_w
    q_source = np.column_stack([jr1_w, jr2_w, jr1_w + jr2_w])

    # ---- spatial shape: dT/dx = 0 at x = 0 by construction -----------------
    # cos(pi/2 * x/LX) has zero slope at x=0 and falls to 0 at the housing, so
    # the cell centre is the hot plane and heat leaves through the outside.
    shape_x = np.cos(0.5 * np.pi * x / LX)
    shape_yz = (1.0
                + 0.18 * np.sin(2.0 * np.pi * y / LY)
                + 0.12 * np.cos(np.pi * z / LZ))
    hot = np.where(layer == "jr1c", 1.25, 1.0)      # the source plane runs hotter
    shape = shape_x * shape_yz * hot                # (P,)

    # ---- time profile: first-order relaxation towards the source -----------
    tau_T = span / 2.5
    rise = 1.0 - np.exp(-t / tau_T)                 # (n_t,)
    amplitude = 14.0 * c_rate                       # K at steady state

    T0 = cfg["solid_initial_temp"] - 273.15         # °C
    T = T0 + amplitude * rise[:, None] * shape[None, :]

    # A little measurement-scale noise so dTdt_scale is not exactly analytic.
    # Small against the 14 K swing: the hybrid history amplifies short-lag
    # differences by A ~ 100, and noise above ~1e-3 K would dominate that
    # channel and make the smoke test measure the noise instead of the model.
    T += 5e-4 * rng.standard_normal((n_t, n_points))
    return T.astype(np.float64), q_source.astype(np.float64)


def write_op(op_id: str, out_dir: Path, seconds: float, seed: int) -> Path:
    # crc32, not hash(): Python randomises string hashing per process unless
    # PYTHONHASHSEED is set, so hash(op_id) would give a different bundle on
    # every invocation and the fixture would not be reproducible at all.
    rng = np.random.default_rng(zlib.crc32(op_id.encode()) + seed)
    xyz, layer = _grid(points_per_layer())
    t = np.arange(0.0, seconds, RAW_DT, dtype=np.float64)
    cfg = _config_for(op_id, rng)
    T, q_source = _field(t, xyz, layer, cfg, rng)

    names = list(CONFIG_ORDER)
    scalars = np.array([cfg[n] for n in names], dtype=np.float64)

    # One genuine time profile, so the profile branch of _config_timeseries and
    # the config_profile / config_label split in load_ops are both exercised.
    # Everything else stays a per-OP constant, which is what OP01-03 look like.
    ramp_t = np.array([t[0], t[-1]], dtype=np.float64)
    ramp_v = np.array([cfg["c_rate"] * 0.8, cfg["c_rate"]], dtype=np.float64)

    payload = {
        "t_fast": t,
        "T": T,
        "q_source": q_source,
        "xyz": xyz,
        "layer": layer.astype("U8"),
        "sim_config_scalar": scalars,
        "sim_config_scalar_names_json": json.dumps(names),
        "sim_config_ts_names_json": json.dumps(["c_rate"]),
        "sim_config_ts_c_rate_t": ramp_t,
        "sim_config_ts_c_rate_v": ramp_v,
        # Loud provenance: these arrays must never be mistaken for measured data.
        "synthetic": np.array(True),
        "synthetic_note": np.array(
            "written by tools/make_synthetic_cache.py -- smoke fixture, "
            "absolute MAE from it says nothing about the real OPs"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{op_id}.npz"
    np.savez_compressed(path, **payload)
    return path


# --------------------------------------------------------------------------
# material_properties/ fallback
# --------------------------------------------------------------------------
# materials.py reads per-point CSVs and a constants.yaml from a folder that is
# also gitignored. On the machine with the real data that folder exists and is
# left alone; on a bare checkout there is nothing for load_material_properties
# to read, so the synthetic cache alone would still not load. These numbers are
# ordinary orders of magnitude for a prismatic cell, not measurements.
_CONSTANTS = {
    "jr1": {"density": 2100.0, "specific_heat": 1100.0, "lambda_zz": 1.2},
    "housing": {"density": 2700.0, "specific_heat": 900.0, "lambda_iso": 200.0},
}

_CC_CSVS = {
    "Density_Grid_CellCenter.csv": 2200.0,
    "SpecificHeat_Grid_CellCenter.csv": 1050.0,
    "ThermalConductivityXX_Grid_CellCenter.csv": 0.9,
    "ThermalConductivityYY_Grid_CellCenter.csv": 18.0,
    "ThermalConductivityZZ_Grid_CellCenter.csv": 18.0,
}

_JR1_CSVS = {
    "ThermalConductivityXX_Grid_JR1Center.csv": 0.8,
    "ThermalConductivityXY_Grid_JR1Center.csv": 0.05,
    "ThermalConductivityYY_Grid_JR1Center.csv": 16.0,
}


def _write_row_csv(path: Path, value: float, n: int, rng) -> None:
    """One header row + one data row, one column per point (materials.py)."""
    vals = value * (1.0 + 0.02 * rng.standard_normal(n))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"p{i}" for i in range(n))
    row = ",".join(f"{v:.6g}" for v in vals)
    path.write_text(header + "\n" + row + "\n")


def ensure_material_properties(force: bool = False) -> bool:
    """Write a stand-in ``material_properties/`` tree when none is present.

    Returns True when files were written. An existing folder is never touched
    without ``force``: on the machine that has the measured properties, this
    script must not overwrite them.
    """
    mat_dir = PKG_DIR / "material_properties"
    if mat_dir.exists() and not force:
        return False
    rng = np.random.default_rng(0)
    mat_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    (mat_dir / "constants.yaml").write_text(yaml.safe_dump(_CONSTANTS,
                                                           sort_keys=True))
    for name, value in _CC_CSVS.items():
        _write_row_csv(mat_dir / "Cell Center" / name, value,
                       DEFAULT_N_PER_LAYER, rng)
    for name, value in _JR1_CSVS.items():
        _write_row_csv(mat_dir / "JR1 Center" / name, value,
                       DEFAULT_N_PER_LAYER, rng)
    (mat_dir / "SYNTHETIC.txt").write_text(
        "Stand-in properties written by tools/make_synthetic_cache.py.\n"
        "Plausible orders of magnitude for a prismatic cell, NOT measurements.\n"
        "Delete this folder before using the real material data.\n")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ops", nargs="+",
                   default=["OP01", "OP02", "OP03", "OP04", "OP05", "OP06", "OP07"])
    p.add_argument("--seconds", type=float, default=1445.0,
                   help="trajectory length in seconds. The default is the real "
                        "OP01 length (14450 raw steps at 0.1 s), which puts "
                        "T_span_ref -- and therefore the hybrid amplification A "
                        "-- in the same regime as the measured data")
    p.add_argument("--quick", action="store_true",
                   help="120 s instead. Runs in seconds, but A drops by ~12x "
                        "and the fixture then only answers 'does it run', not "
                        "'is the rollout stable'")
    p.add_argument("--out", default=str(PREFERRED_DATA_CACHE),
                   help="cache directory to write into")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-materials", action="store_true",
                   help="overwrite an existing material_properties/ folder")
    args = p.parse_args()

    if args.quick:
        args.seconds = 120.0
    if args.seconds < 60.0:
        raise SystemExit(
            f"--seconds {args.seconds:g} is shorter than the 20 s rate lag needs; "
            "use at least 60")

    wrote = ensure_material_properties(force=args.force_materials)
    print(f"material_properties/: {'written (stand-in)' if wrote else 'left as is'}")

    out_dir = Path(args.out)
    n_steps = int(args.seconds / RAW_DT)
    n_layer = points_per_layer()
    n_points = n_layer * len(LAYERS)
    print(f"grid: {n_points} points ({n_layer} per layer, "
          f"JR1={n_layer} vs data.N_JR1_POINTS={N_JR1_POINTS})")
    print(f"time: {n_steps} raw steps at {RAW_DT:g}s = {args.seconds:g}s")
    for op_id in args.ops:
        path = write_op(op_id, out_dir, args.seconds, args.seed)
        print(f"  wrote {path}")
    print(f"\nSYNTHETIC cache in {out_dir}. Smoke fixture only -- absolute MAE "
          f"from it says nothing about the real OPs.")
    if args.quick:
        print("--quick: A is ~12x below the measured regime. Use this to check "
              "that a run completes, never to judge rollout stability.")
    print("Check the amplification the loader derives from it with:\n"
          "  python3 PINNmodulusTwo/tools/data_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
