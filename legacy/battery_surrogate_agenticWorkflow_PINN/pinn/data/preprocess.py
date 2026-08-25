"""Preprocessing / normalization for the OP01 PINN.

Convention (per plan 004, user-confirmed):
- Spatial coords (x, y, z) and time t  -> min-max to [0, 1]  (needed for the
  non-dimensional Fourier-number PDE).
- OUTPUTS T and bc_V, plus the config scalars -> z-score (standardize) using
  train-set statistics; stats are saved so inference can invert exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class NormStats:
    """All normalization statistics needed to (de)normalize a run."""

    # spatial min-max (per axis) -> [0, 1]
    xyz_min: list
    xyz_max: list
    L_axis: list          # per-axis extent (max - min); used by the PDE chain rule
    # time
    t0: float             # first OFFICIAL sample time (=0.1 s); treated as t~=0
    t_max: float
    # z-score for temperature output
    T_mu: float
    T_sigma: float
    # z-score for voltage output
    V_mu: float
    V_sigma: float
    # z-score for config scalars
    config_mu: list
    config_sigma: list
    # physics helpers
    T_init: float
    # OFFICIAL initial condition = measured first sample (per grid point), physical units.
    # Notion: the IC is the true temperature given by the data; the first sample is at t=0.1 s.
    T_ic: list
    V_ic: float

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "NormStats":
        return NormStats(**json.loads(Path(path).read_text()))


def _std_guard(sigma: np.ndarray | float, eps: float = 1e-8):
    """Guard a standard deviation so we never divide by (near) zero."""
    return np.maximum(sigma, eps)


def build_norm_stats(
    t: np.ndarray,
    xyz: np.ndarray,
    T_labels: np.ndarray,
    bc_V: np.ndarray,
    config: Dict[str, float],
    T_init: float,
    train_slice: slice | None = None,
) -> NormStats:
    """Compute normalization statistics from the TRAIN portion of the data.

    Args:
        t:          (n_t,) time in seconds.
        xyz:        (n_points, 3) coordinates.
        T_labels:   (n_t, n_points) temperature labels.
        bc_V:       (n_t,) voltage.
        config:     dict of the 7 config scalars.
        T_init:     initial solid temperature (physical units).
        train_slice: which timesteps count as "train" for computing stats.
                     Defaults to all timesteps.
    """
    if train_slice is None:
        train_slice = slice(0, len(t))

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    L_axis = xyz_max - xyz_min

    T_train = T_labels[train_slice]
    V_train = bc_V[train_slice]

    # Official IC = the very first measured sample (t = t0 = 0.1 s), per grid point.
    T_ic = T_labels[0].astype(float)
    V_ic = float(bc_V[0])

    T_mu = float(T_train.mean())
    T_sigma = float(_std_guard(T_train.std()))
    V_mu = float(V_train.mean())
    V_sigma = float(_std_guard(V_train.std()))

    cfg_vals = np.array(
        [
            config["c_rate"],
            config["cell_current"],
            config["fluid_initial_temp"],
            config["fluid_inlet_temp"],
            config["fluid_mass_flow"],
            config["soc_start"],
            config["solid_initial_temp"],
        ],
        dtype=np.float64,
    )
    # Single OP -> sigma is 0; guard so config normalizes to 0 (constant, no signal).
    config_mu = cfg_vals.copy()
    config_sigma = _std_guard(np.zeros_like(cfg_vals))

    return NormStats(
        xyz_min=xyz_min.astype(float).tolist(),
        xyz_max=xyz_max.astype(float).tolist(),
        L_axis=L_axis.astype(float).tolist(),
        t0=float(t[0]),
        t_max=float(t.max()),
        T_mu=T_mu,
        T_sigma=T_sigma,
        V_mu=V_mu,
        V_sigma=V_sigma,
        config_mu=config_mu.tolist(),
        config_sigma=config_sigma.tolist(),
        T_init=float(T_init),
        T_ic=T_ic.tolist(),
        V_ic=V_ic,
    )


if __name__ == "__main__":
    # Invertibility self-check.
    rng = np.random.default_rng(0)
    T = rng.normal(30, 5, size=(50, 10)).astype(np.float32)
    mu, sig = T.mean(), T.std()
    Tn = (T - mu) / sig
    Tback = Tn * sig + mu
    assert np.allclose(T, Tback, atol=1e-4), "z-score inverse failed"
    print("preprocess self-check OK: z-score inverse round-trips.")
