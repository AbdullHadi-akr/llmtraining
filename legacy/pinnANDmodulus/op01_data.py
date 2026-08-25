"""Shared OP01 non-dimensional dataset builder for the pinnANDmodulus experiments.

Both the (true) Modulus-Sym Solver script and the runnable continuous-time PINN use
this so the physics, normalisation and Fourier tensor are identical to the current
PyTorch model in ``battery_surrogate_agenticWorkflow_PINN``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
PINN_ROOT = PROJECT_ROOT / "battery_surrogate_agenticWorkflow_PINN"
if str(PINN_ROOT) not in sys.path:
    sys.path.insert(0, str(PINN_ROOT))

from pinn.data.load_op01 import load_op01_data  # noqa: E402
from pinn.data.load_properties import load_material_properties  # noqa: E402
from pinn.data.preprocess import build_norm_stats  # noqa: E402


@dataclass
class OP01NonDim:
    """Non-dimensional OP01 arrays (all float64 unless noted)."""

    t: np.ndarray          # physical time (n_t,)
    xn: np.ndarray         # scaled coords (n_points, 3)
    tn: np.ndarray         # scaled time (n_t,)
    Tn: np.ndarray         # z-scored labels (n_t, n_points)
    Tn_ic: np.ndarray      # z-scored IC (n_points,)
    Fo: np.ndarray         # Fourier tensor (n_points, 3, 3)
    q_dot: np.ndarray      # source (n_t,) W/m^3
    q_mask: np.ndarray     # JR1 mask (n_points,)
    rho: np.ndarray        # (n_points,)
    Cp: np.ndarray         # (n_points,)
    T_mu: float
    T_sigma: float
    T_span: float
    phys_scale: float
    n_t: int
    n_points: int
    split_t: int
    T_lab: np.ndarray      # physical labels (n_t, n_points)


def load_nondim(subsample_time: int = 40) -> OP01NonDim:
    """Load OP01 and reproduce the non-dimensionalisation of ``TemperaturePINN``."""
    op = load_op01_data(
        npz_path=str(PROJECT_ROOT / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
        heat_source_csv=str(PINN_ROOT / "data/OP01_raw/OP01/OP1_Heat Source.csv"),
        subsample_time=subsample_time,
    )
    props = load_material_properties(
        layer=op["layer"],
        props_cc_dir=str(PINN_ROOT / "Cell Center"),
        props_jr1_dir=str(PINN_ROOT / "JR1 Center"),
    )
    t = op["t"].astype(np.float64)
    xyz = op["xyz"].astype(np.float64)
    T_lab = op["T"].astype(np.float64)
    q_dot = op["q_dot"].astype(np.float64)
    n_t, n_points = T_lab.shape
    split_t = int(0.8 * n_t)

    stats = build_norm_stats(
        t=op["t"], xyz=op["xyz"], T_labels=op["T"], bc_V=op["bc_V"],
        config=op["config"], T_init=op["config"]["solid_initial_temp"],
        train_slice=slice(0, split_t),
    )

    xyz_min = np.asarray(stats.xyz_min, dtype=np.float64)
    L_axis = np.asarray(stats.L_axis, dtype=np.float64)
    L_ref = float(np.prod(L_axis) ** (1.0 / 3.0))
    T_mu, T_sigma = float(stats.T_mu), float(stats.T_sigma)
    t0, t_max = float(stats.t0), float(stats.t_max)
    T_span = t_max - t0

    xn = (xyz - xyz_min) / L_ref
    tn = (t - t0) / (T_span + 1e-12)
    Tn = (T_lab - T_mu) / T_sigma
    Tn_ic = (np.asarray(stats.T_ic, dtype=np.float64) - T_mu) / T_sigma

    rho = np.asarray(props["rho"], dtype=np.float64)
    Cp = np.asarray(props["Cp"], dtype=np.float64)
    lam = np.asarray(props["lambda_tensor"], dtype=np.float64)
    region = np.asarray(props["region"])
    rc = (rho * Cp).reshape(-1, 1, 1)
    Fo = lam * T_span / (rc * (L_ref ** 2) + 1e-30)
    q_mask = (region == 1).astype(np.float64)

    dTdt_true = (Tn[2:] - Tn[:-2]) / (tn[2:] - tn[:-2])[:, None]
    phys_scale = float(np.sqrt((dTdt_true ** 2).mean())) + 1e-6

    return OP01NonDim(
        t=t, xn=xn, tn=tn, Tn=Tn, Tn_ic=Tn_ic, Fo=Fo, q_dot=q_dot, q_mask=q_mask,
        rho=rho, Cp=Cp, T_mu=T_mu, T_sigma=T_sigma, T_span=T_span, phys_scale=phys_scale,
        n_t=n_t, n_points=n_points, split_t=split_t, T_lab=T_lab,
    )
