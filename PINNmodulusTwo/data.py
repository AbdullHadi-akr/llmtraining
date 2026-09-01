"""Profile-aware loader + preprocessing for the PINNmodulusTwo profile extension.

This is ``PINNmodulusTwo/data.py`` with the changes the *profile* operating
points force. Everything the base loader does that still holds is kept
byte-for-byte in spirit: pooled z-scored temperature, shared ``L_ref`` /
``T_span_ref`` non-dimensionalisation, the anisotropic Fourier tensor, the
per-point static features, and the same ``load_ops`` / ``build_op`` split
between "fit the constants on training data" and "reuse them for a held-out OP".

What the profiles change
------------------------
1. **Drivers are resampled, not point-sampled.**
   The base loader takes ``[::subsample]`` of everything. For a constant OP that
   is exact -- a constant equals its own mean. For a profile it is aliasing: at
   ``subsample=2`` nineteen of every twenty raw samples of the CC-CV current
   taper are discarded, and which twentieth survives decides what the model is
   told the current was. Every *driver* (the heat source and the config
   profiles) is therefore reduced with a **backward window mean**: sample ``j``
   is the average over the raw interval that ends at it. That is the quantity
   the step from ``t_{j-1}`` to ``t_j`` is actually driven by, and for the heat
   source it preserves the energy that went into the window. Temperature stays
   point-sampled: it is a state, and the model's own rollout has to match the
   state at the sample instants, not an average over them. ``resample: point``
   restores the base behaviour exactly.

2. **The drivers get their own history channels.**
   The recurrence feeds back the model's own past *temperature*. That is what
   disambiguates two OPs sharing an instantaneous config, and it stays. But the
   drivers themselves are exogenous and fully known in advance, so their history
   costs nothing to compute and does not have to be predicted: for every driver
   ``d`` we append causal rate channels over the same cumulative segments the
   hybrid temperature history uses,

       rate_1 = [d(t)      - d(t-L1)]      / L1
       rate_2 = [d(t-L1)   - d(t-L1-L2)]   / L2

   with ``d(t) := d(t_0)`` for ``t < t_0`` and each rate divided by its own
   NOMINAL segment length -- the same choice, for the same reason, as
   ``model._history_hybrid``: dividing by the clamped elapsed span is a
   singularity at the start of the trajectory. A constant OP gets rates that are
   identically zero, which is itself informative ("nothing is moving").
   These are plain extra columns of ``forcing_feat``: the recurrence, the model
   and the physics residual are untouched.

3. **The normalisation constants are different, and not by a little.**
   Pooling OP01-OP16 instead of OP01-OP05 widens every pooled statistic:
   ``T_sigma`` now spans a 0 C start (OP14) to a 4 C charge (OP13), and
   ``Qsrc_scale`` is dominated by the high-C-rate OPs. ``phys_scale``,
   ``dTdt_scale``, ``aniso_scale`` and ``Qsrc_scale`` all move, and they are the
   divisors of the physics residual -- so ``w_phys`` / ``w_bc`` tuned against the
   base project's constants do **not** carry over. That is the reason this
   extension needs its own benchmark rather than inheriting numbers.
   ``normalisation_report()`` prints the constants next to the per-OP spread so
   the shift is visible instead of assumed.

4. **Held-out OPs are checked for extrapolation.**
   ``build_op`` reuses the training constants, which is what makes a held-out OP
   a genuine test -- and also what makes an out-of-range driver silently
   z-score to something the network never saw. ``coverage_report()`` says which
   channel of which OP leaves the trained range, and by how much.

Scope is unchanged from the base project: temperature only, ``bc_V`` out of
scope, JR1 heat = ``q_source[:, 0]``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from env_check import require_training_env

require_training_env()   # a useful sentence instead of a pandas ImportError

import numpy as np  # noqa: E402

from materials import load_material_properties  # noqa: E402
from op_registry import PROFILE_CHANNELS  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


# Where the OP*.npz bundles may live, most specific first. The bundles are NOT
# in git (see .gitignore), so they stay wherever they already are on a given
# machine; first existing hit wins. The two extra entries keep a cache that was
# built before the folders were restructured working without being moved.
_CACHE_CANDIDATES = (
    THIS_DIR / "data_cache",                    # project-local override
    PROJECT_ROOT / "PINNmodulusTwoExtProfiles" / "data_cache",  # pre-merge
    PROJECT_ROOT / "data_cache",                # preferred: shared, top level
    PROJECT_ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "data_cache",
    PROJECT_ROOT / "battery_surrogate_agenticWorkflow" / "data_cache",  # pre-move
)
PREFERRED_DATA_CACHE = PROJECT_ROOT / "data_cache"


def _resolve_data_cache() -> Path:
    for candidate in _CACHE_CANDIDATES:
        if candidate.exists():
            return candidate
    return PREFERRED_DATA_CACHE


DATA_CACHE = _resolve_data_cache()

# Volume of the heated JR1 region, in m^3. The cached heat column
# ``q_source[:, 0]`` (upstream name ``jr1_w``) is a TOTAL POWER in watts for
# that region, so the volumetric source density the PDE needs is
#
#     q [W/m^3] = P_total [W] / V_JR1 [m^3]
#
# and nothing else. Until 31.08.2026 this line also divided by the NUMBER OF
# JR1 GRID POINTS (121), which is a category error: "each point's share of the
# total power" and "the power density at a point" are different quantities, and
# only the second one belongs in ``dTn/dtn = Fo : grad^2 Tn + Qsrc``. Every
# point in the heated region carries the SAME W/m^3.
#
# The consequence was a source 121x too small, i.e. a residual describing a cell
# heated by almost nothing. It was invisible everywhere it could have been
# caught: the EMA loss balancer divides a uniform factor straight back out,
# L_phys still lands at O(1), and ``phys_scale``/``Qsrc_scale`` are built from
# the same understated numbers, so they agreed with it.
#
# What found it was an energy argument (``energy_balance_report``): on OP07 and
# OP14 the coolant flow is ZERO, so almost nothing can be carried away and
# <dTn/dtn> ~ <Qsrc> has to hold. Measured, it was short by 110x and 107x --
# against a bug factor of 121. Removing the division lands both at 0.91 and
# 0.88, just under 1 as they must be (a little heat still reaches the housing),
# and every cooled OP between 0.45 and 0.63, ordered by its flow rate.
V_JR1 = 4.394793e-04

# Order of the 7 config scalars in ``sim_config_scalar`` (unchanged).
CONFIG_ORDER = [
    "c_rate",
    "cell_current",
    "fluid_initial_temp",
    "fluid_inlet_temp",
    "fluid_mass_flow",
    "soc_start",
    "solid_initial_temp",
]

# Signals that get causal rate channels. The heat source first (it is the term
# that actually appears in the PDE), then the four config channels the upstream
# assembly can deliver as a profile. A channel that is constant in a given OP
# contributes exactly-zero rates there; the columns are kept anyway so every OP
# has the same feature width.
DRIVER_NAMES = ("q_dot",) + PROFILE_CHANNELS

DEFAULT_DRIVER_RATE_LAGS = (5.0, 20.0)

# A driver sample counts as "transient" when at least one of its shortest-window
# normalised rates exceeds this many times its own pooled training RMS. Used
# only for reporting (the benchmark splits the error into transient vs.
# quiescent); it never touches training.
TRANSIENT_TAU = 1.0

# Below this the within-OP temporal spread of a channel is treated as numerical
# noise rather than a profile, in the channel's own physical units.
PROFILE_DETECT_TOL = 1e-6

# Temperature-history segment lengths the report quotes A for. Only a reporting
# default -- the value that trains comes from config.yaml / --rate-lags, and
# train.py prints A for whatever was actually asked for.
TEMPERATURE_RATE_LAGS_FOR_REPORT = (5.0, 20.0)


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def _keep_indices(n_raw: int, subsample: int) -> np.ndarray:
    return np.arange(0, n_raw, max(int(subsample), 1))


def _backward_window_mean(v: np.ndarray, subsample: int,
                          keep: np.ndarray) -> np.ndarray:
    """Average each driver over the raw interval that ENDS at its kept sample.

    ``out[j]`` is the mean of raw samples ``(keep[j]-subsample, keep[j]]``, i.e.
    the window the step from ``keep[j-1]`` to ``keep[j]`` is driven by. ``out[0]``
    is the raw value at ``t_0``: nothing precedes it, and the initial condition
    is imposed there anyway.

    Backward rather than centred or forward on purpose. A forward window lets a
    step that happens *after* a sample influence that sample, which hands the
    rollout information from its own future -- harmless for a plot, not harmless
    for a free-running autoregressive model.
    """
    v = np.asarray(v, dtype=np.float64)
    s = max(int(subsample), 1)
    n_keep = keep.shape[0]
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    out = np.empty((n_keep, v.shape[1]), dtype=np.float64)
    out[0] = v[keep[0]]
    if n_keep > 1:
        if s == 1:
            out[1:] = v[keep[1:]]
        else:
            # rows keep[j]-s+1 .. keep[j]; every window has the same length, and
            # keep[j] >= s for j >= 1, so no index runs off the front.
            offsets = np.arange(-s + 1, 1)
            rows = keep[1:, None] + offsets[None, :]
            out[1:] = v[rows].mean(axis=1)
    return out[:, 0] if squeeze else out


def _resample_driver(v: np.ndarray, subsample: int, keep: np.ndarray,
                     mode: str) -> np.ndarray:
    """``mean`` = anti-aliased backward window mean, ``point`` = base behaviour."""
    if mode == "point":
        return np.asarray(v, dtype=np.float64)[keep]
    if mode != "mean":
        raise SystemExit(
            f"unknown resample mode {mode!r}; expected 'mean' or 'point'"
        )
    return _backward_window_mean(v, subsample, keep)


def _causal_rate_block(t: np.ndarray, drivers: np.ndarray,
                       lags: Sequence[float]) -> np.ndarray:
    """(n_t, n_drivers, n_lags) cumulative backward rates of every driver.

    Segment ``i`` starts where segment ``i-1`` ended, exactly like the hybrid
    temperature history, and each difference is divided by its OWN nominal
    segment length. Before ``t_0`` the driver is held at ``d(t_0)`` (the same
    flat padding ``model._padded_lookup`` uses), so early in the trajectory a
    partially-filled window yields a damped rate that converges to the true one
    once the window has filled -- rather than the ~1/dt blow-up that dividing by
    the clamped elapsed span would give.
    """
    t = np.asarray(t, dtype=np.float64)
    drivers = np.asarray(drivers, dtype=np.float64)
    n_t, n_d = drivers.shape
    n_l = len(lags)
    out = np.zeros((n_t, n_d, n_l), dtype=np.float64)
    if n_t == 0 or n_l == 0:
        return out
    t0 = t[0]
    boundary = t.copy()
    for i, lag in enumerate(lags):
        seg = float(lag)
        t_next = boundary - seg
        if seg <= 0.0:
            boundary = t_next
            continue
        for c in range(n_d):
            d_end = np.interp(np.maximum(boundary, t0), t, drivers[:, c])
            d_start = np.interp(np.maximum(t_next, t0), t, drivers[:, c])
            out[:, c, i] = (d_end - d_start) / seg
        boundary = t_next
    return out


def _std_guard(sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.maximum(sigma, eps)


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class OPData:
    """Per-OP non-dimensional arrays (float32 unless noted)."""

    op_id: str
    t: np.ndarray            # physical time (n_t,)
    tn: np.ndarray           # scaled time ~[0, 1] (n_t,)
    xn: np.ndarray           # scaled coords (n_points, 3)
    Tn: np.ndarray           # z-scored temperature (n_t, n_points)
    Tn_ic: np.ndarray        # z-scored initial condition (n_points,)
    config_feat: np.ndarray  # z-scored config features (n_t, n_config)
    static_feat: np.ndarray  # per-point static feats (n_points, n_static)
    forcing_feat: np.ndarray  # [q_dot_z, driver rates...] (n_t, n_forcing)
    Fo: np.ndarray           # Fourier tensor (n_points, 3, 3)
    Qsrc: np.ndarray         # nondim source term (n_t, n_points)
    q_mask: np.ndarray       # JR1 mask (n_points,)
    region: np.ndarray       # 0=CC, 1=JR1, 2=Housing (n_points,)
    split_t: int             # train/test boundary along time
    n_t: int
    n_points: int
    dtn: float               # uniform spacing of tn
    T_lab: np.ndarray        # physical temperature labels (n_t, n_points)
    # ---- profile bookkeeping (reporting only, never a model input) ----------
    cfg_phys: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_dot: np.ndarray = field(default_factory=lambda: np.zeros(0))
    transient: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    profiles_detected: tuple = ()
    profile_flags_meta: tuple = ()
    profile_coverage: dict = field(default_factory=dict)

    @property
    def transient_frac(self) -> float:
        return float(self.transient.mean()) if self.transient.size else 0.0


@dataclass
class NormBundle:
    """Shared normalisation constants + the per-OP datasets."""

    ops: List[OPData]
    T_mu: float
    T_sigma: float
    T_span_ref: float
    L_ref: float
    xyz_min: np.ndarray
    config_mu: np.ndarray
    config_sigma: np.ndarray
    config_active: np.ndarray   # bool mask: channels with real training variance
    n_config: int
    phys_scale: float
    dTdt_scale: float
    aniso_scale: float
    Qsrc_scale: float
    bc_scale: float
    static_feat: np.ndarray
    n_static: int
    q_mu: float
    q_sigma: float
    n_forcing: int
    xn: np.ndarray
    Fo: np.ndarray
    q_mask: np.ndarray
    region: np.ndarray
    rho: np.ndarray
    Cp: np.ndarray
    train_frac: float
    # ---- profile extension --------------------------------------------------
    resample: str
    use_driver_history: bool
    driver_rate_lags: tuple
    driver_names: tuple
    driver_rate_rms: np.ndarray      # (n_drivers, n_lags) pooled train RMS
    driver_rate_active: np.ndarray   # (n_drivers, n_lags) bool
    n_driver_rate: int
    config_time_std: np.ndarray      # max within-OP temporal std, per channel
    config_across_std: np.ndarray    # std of per-OP means, per channel
    config_min: np.ndarray           # pooled train range, physical units
    config_max: np.ndarray
    q_min: float
    q_max: float
    T_min: float
    T_max: float
    train_ops: tuple
    trained_profiles: tuple
    per_op_Qsrc_rms: dict
    # How many x-neighbour pairs bc_scale was measured from. 0 means the grid
    # was not structured enough and the 1/L_ref fallback was used, which is a
    # guess that never looked at a temperature -- worth knowing before reading
    # anything into L_bc.
    bc_pairs: int


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _config_vector(config: Dict[str, float]) -> np.ndarray:
    # Missing scalars (present only as a profile, or derived upstream) -> NaN.
    return np.array([float(config.get(k, np.nan)) for k in CONFIG_ORDER],
                    dtype=np.float64)


def _config_timeseries_full(npz, t_full: np.ndarray,
                            config: Dict[str, float]):
    """(n_raw, n_config) config values on the RAW time grid.

    Interpolating on the raw grid and reducing afterwards -- rather than
    interpolating straight onto the subsampled grid, as the base loader does --
    is what makes the window mean see the profile's real shape between two kept
    samples instead of a straight line across the gap.

    ``np.interp`` holds the first/last profile value outside the profile's own
    time range. That is the right fallback, but it is a silent one, so the range
    each profile covers travels back with the array.
    """
    ts_names = json.loads(str(npz["sim_config_ts_names_json"].item()))
    base = _config_vector(config)
    feat = np.tile(base, (t_full.shape[0], 1))
    profiles, coverage = [], {}
    for name in ts_names:
        if name not in CONFIG_ORDER:
            continue
        col = CONFIG_ORDER.index(name)
        t_src = np.asarray(npz[f"sim_config_ts_{name}_t"], dtype=np.float64)
        v_src = np.asarray(npz[f"sim_config_ts_{name}_v"], dtype=np.float64)
        feat[:, col] = np.interp(t_full, t_src, v_src)
        profiles.append(name)
        coverage[name] = (float(t_src.min()), float(t_src.max()))
    return feat, tuple(profiles), coverage


def _meta_profile_flags(npz) -> tuple:
    """Channels the upstream assembly recorded as profiles, if it recorded any."""
    if "meta_json" not in getattr(npz, "files", []):
        return ()
    try:
        meta = json.loads(str(npz["meta_json"].item()))
    except Exception:
        return ()
    flags = meta.get("profile_flags") or {}
    return tuple(sorted(k for k, v in flags.items() if v))


def _read_raw(op_id: str, subsample_time: int, resample: str) -> dict:
    npz = np.load(DATA_CACHE / f"{op_id}.npz", allow_pickle=True)
    names = json.loads(str(npz["sim_config_scalar_names_json"].item()))
    config = dict(zip(names, [float(v) for v in npz["sim_config_scalar"]]))

    t_full = np.asarray(npz["t_fast"], dtype=np.float64)
    keep = _keep_indices(t_full.shape[0], subsample_time)
    t = t_full[keep]

    # State: point-sampled. The rollout has to reproduce T at the sample
    # instants, so averaging the label over a window would ask it for something
    # else entirely.
    T = np.asarray(npz["T"], dtype=np.float64)[keep]

    # Drivers: window-reduced (see _backward_window_mean).
    jr1_full = np.asarray(npz["q_source"], dtype=np.float64)[:, 0]
    # Total watts over the JR1 region -> volumetric source density. See V_JR1
    # for why there is no point-count factor here any more.
    q_dot_full = jr1_full / V_JR1                      # W/m^3
    q_dot = _resample_driver(q_dot_full, subsample_time, keep, resample)

    cfg_full, profiles, coverage = _config_timeseries_full(npz, t_full, config)
    cfg_ts = _resample_driver(cfg_full, subsample_time, keep, resample)

    return dict(
        op_id=op_id, t=t, T=T, q_dot=q_dot, cfg_ts=cfg_ts,
        profiles=profiles, profile_coverage=coverage,
        profile_flags_meta=_meta_profile_flags(npz),
        t_range=(float(t_full[0]), float(t_full[-1])),
        xyz=np.asarray(npz["xyz"], dtype=np.float64),
        layer=np.asarray(npz["layer"]),
    )


def _grid_arrays(layer, xyz, T_span_ref, L_ref):
    props = load_material_properties(layer=layer)
    rho = np.asarray(props["rho"], dtype=np.float64)
    Cp = np.asarray(props["Cp"], dtype=np.float64)
    lam = np.asarray(props["lambda_tensor"], dtype=np.float64)
    region = np.asarray(props["region"])
    rc = (rho * Cp).reshape(-1, 1, 1)
    Fo = lam * T_span_ref / (rc * (L_ref ** 2) + 1e-30)
    q_mask = (region == 1).astype(np.float64)
    return rho, Cp, lam, Fo, region, q_mask


def _static_features(rho, Cp, lam, region, xn):
    """Per-point, time-independent features (unchanged from the base project)."""
    lam_iso = (lam[:, 0, 0] + lam[:, 1, 1] + lam[:, 2, 2]) / 3.0
    alpha = lam_iso / (rho * Cp + 1e-30)
    alpha_z = (alpha - alpha.mean()) / (alpha.std() + 1e-12)
    jr1 = (region == 1).astype(np.float64)
    xcol = xn[:, 0]
    x_z = (xcol - xcol.mean()) / (xcol.std() + 1e-12)
    return np.stack([alpha_z, jr1, x_z], axis=1).astype(np.float32)


def _normalise_config(cfg_ts, config_mu, config_sigma, config_active):
    """z-score active config channels; force dead channels to 0.

    Unchanged from the base project, including the NaN fill: a channel that is
    absent for a held-out OP degrades to a neutral 0 instead of exploding
    through the epsilon guard. ``config_nan_channels`` reports when that happens
    so a silently-neutral channel is at least visible.
    """
    cfg = np.where(np.isnan(cfg_ts), config_mu[None, :], cfg_ts)
    feat = (cfg - config_mu) / config_sigma
    feat[:, ~config_active] = 0.0
    return feat.astype(np.float32)


def _driver_matrix(r) -> np.ndarray:
    """(n_t, n_drivers) physical driver signals, in ``DRIVER_NAMES`` order."""
    cols = [np.asarray(r["q_dot"], dtype=np.float64)]
    for name in PROFILE_CHANNELS:
        cols.append(r["cfg_ts"][:, CONFIG_ORDER.index(name)])
    return np.stack(cols, axis=1)


def _forcing_features(r, *, q_mu, q_sigma, driver_rate_lags, driver_rate_rms,
                      driver_rate_active, use_driver_history):
    """[q_dot_z, driver rate channels...] -> (n_t, n_forcing).

    The rates are divided by their pooled TRAINING RMS, not by a per-OP scale:
    a per-OP scale would make "the current is changing fast" mean something
    different in every OP, which is precisely the comparison the model has to be
    able to make. Rate columns that are flat across the whole training set are
    forced to 0 for the same reason ``config_active`` exists -- an all-zero
    channel divided by an epsilon guard is noise amplification, not a feature.
    """
    q_z = ((np.asarray(r["q_dot"], dtype=np.float64) - q_mu) / q_sigma)
    if not use_driver_history or not len(driver_rate_lags):
        return q_z.astype(np.float32).reshape(-1, 1)
    rates = _causal_rate_block(r["t"], _driver_matrix(r), driver_rate_lags)
    scaled = rates / driver_rate_rms[None, :, :]
    scaled[:, ~driver_rate_active] = 0.0
    flat = scaled.reshape(rates.shape[0], -1)
    return np.concatenate([q_z.reshape(-1, 1), flat], axis=1).astype(np.float32)


def _transient_mask(forcing_feat: np.ndarray, n_drivers: int, n_lags: int,
                    tau: float = TRANSIENT_TAU) -> np.ndarray:
    """Samples where some driver moves faster than its own training RMS rate.

    Uses the SHORTEST segment only: that is the window that resolves a step, and
    a longer one keeps reporting a transient long after the driver has settled.
    Reporting only -- the benchmark splits its error by this, training never
    sees it.
    """
    if forcing_feat.shape[1] <= 1 or n_lags == 0:
        return np.zeros(forcing_feat.shape[0], dtype=bool)
    block = forcing_feat[:, 1:].reshape(-1, n_drivers, n_lags)
    return (np.abs(block[:, :, 0]) > tau).any(axis=1)


def _measure_bc_scale(xn, ops, fallback: float, max_times: int = 200):
    """RMS of the spatial temperature gradient dTn/dxn, measured on train data.

    The BC drives dT/dx to zero at x=0, so "how large is a gradient here" cannot
    be read off the boundary itself -- it is supposed to vanish there. What makes
    a useful yardstick is the gradient the data actually carries elsewhere: a BC
    residual of 1 then means "as steep as a typical gradient in this cell".

    The previous ``1 / L_ref`` was a pure guess that did not involve the
    temperature at all, yet it set the scale of ``L_bc`` and with it the range of
    ``w_bc`` that means anything.

    Returns ``(scale, n_pairs)``; ``n_pairs == 0`` means the grid was not
    structured enough to find x-neighbours and ``fallback`` is returned.
    """
    columns: Dict[tuple, List[int]] = {}
    for i, (y, z) in enumerate(np.round(xn[:, 1:], 9)):
        columns.setdefault((float(y), float(z)), []).append(i)

    lo, hi, dx = [], [], []
    for idxs in columns.values():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs, key=lambda i: xn[i, 0])
        for a, b in zip(idxs[:-1], idxs[1:]):
            step = float(xn[b, 0] - xn[a, 0])
            if step > 1e-12:
                lo.append(a)
                hi.append(b)
                dx.append(step)
    if len(dx) < 8:
        return float(fallback), 0

    lo_i, hi_i, dx_a = np.asarray(lo), np.asarray(hi), np.asarray(dx)
    pooled = []
    for op in ops:
        n = max(1, int(op.split_t))
        stride = max(1, n // max_times)
        Tn = op.Tn[:n:stride].astype(np.float64)
        pooled.append(((Tn[:, hi_i] - Tn[:, lo_i]) / dx_a[None, :]).ravel())
    g = np.concatenate(pooled)
    return float(np.sqrt((g ** 2).mean())) + 1e-12, len(dx_a)


def _assemble_op(r, split_t, *, T_mu, T_sigma, T_span_ref, xn, Fo, q_mask,
                 region, rho, Cp, config_mu, config_sigma, config_active,
                 static_feat, q_mu, q_sigma, driver_rate_lags, driver_rate_rms,
                 driver_rate_active, use_driver_history) -> OPData:
    t = r["t"]
    n_t = t.shape[0]
    tn = (t - t[0]) / (T_span_ref + 1e-12)
    dtn = float(tn[1] - tn[0]) if n_t > 1 else 1.0
    Tn = (r["T"] - T_mu) / T_sigma
    Tn_ic = Tn[0].copy()
    cfg_feat = _normalise_config(r["cfg_ts"], config_mu, config_sigma,
                                 config_active)
    forcing_feat = _forcing_features(
        r, q_mu=q_mu, q_sigma=q_sigma, driver_rate_lags=driver_rate_lags,
        driver_rate_rms=driver_rate_rms, driver_rate_active=driver_rate_active,
        use_driver_history=use_driver_history,
    )
    Qsrc = (
        r["q_dot"][:, None] * q_mask[None, :] * T_span_ref
        / (rho[None, :] * Cp[None, :] * T_sigma)
    )
    detected = tuple(
        name for name in CONFIG_ORDER
        if r["cfg_ts"][:, CONFIG_ORDER.index(name)].std() > PROFILE_DETECT_TOL
    )
    return OPData(
        op_id=r["op_id"], t=t.astype(np.float32), tn=tn.astype(np.float32),
        xn=xn.astype(np.float32), Tn=Tn.astype(np.float32),
        Tn_ic=Tn_ic.astype(np.float32), config_feat=cfg_feat,
        static_feat=static_feat, forcing_feat=forcing_feat,
        Fo=Fo.astype(np.float32), Qsrc=Qsrc.astype(np.float32),
        q_mask=q_mask.astype(np.float32), region=region.astype(np.int64),
        split_t=split_t, n_t=n_t, n_points=xn.shape[0], dtn=dtn,
        T_lab=r["T"].astype(np.float32),
        cfg_phys=r["cfg_ts"], q_dot=r["q_dot"],
        transient=_transient_mask(forcing_feat, len(DRIVER_NAMES),
                                  len(driver_rate_lags)),
        profiles_detected=detected,
        profile_flags_meta=r.get("profile_flags_meta", ()),
        profile_coverage=r.get("profile_coverage", {}),
    )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_ops(
    op_ids: Sequence[str] | None = None,
    subsample_time: int = 2,
    train_frac: float = 0.8,
    resample: str = "mean",
    driver_rate_lags: Sequence[float] = DEFAULT_DRIVER_RATE_LAGS,
    use_driver_history: bool = True,
) -> NormBundle:
    """Load the requested OPs and build a shared non-dimensional dataset.

    Args:
        op_ids: training OPs, e.g. ``op_registry.DEFAULT_TRAIN_OPS``.
        subsample_time: keep every N-th raw timestep (raw dt = 0.1 s).
        train_frac: fraction of each OP's timeline used for the pooled
            statistics and for the in-time metric split.
        resample: ``mean`` (anti-aliased backward window) or ``point``
            (the base project's ``[::N]``).
        driver_rate_lags: cumulative segment lengths in SECONDS for the driver
            rate channels.
        use_driver_history: append the driver rate channels at all.
    """
    if op_ids is None:
        from op_registry import DEFAULT_TRAIN_OPS
        op_ids = list(DEFAULT_TRAIN_OPS)
    op_ids = list(op_ids)
    driver_rate_lags = tuple(float(v) for v in driver_rate_lags)

    raw = [_read_raw(op_id, subsample_time, resample) for op_id in op_ids]

    # ---- shared geometry (grid identical across OPs) ------------------------
    xyz = raw[0]["xyz"]
    xyz_min = xyz.min(axis=0)
    L_axis = xyz.max(axis=0) - xyz_min
    L_ref = float(np.prod(L_axis) ** (1.0 / 3.0))
    xn = (xyz - xyz_min) / L_ref

    # ---- shared time scale ---------------------------------------------------
    # Still the longest TRAINING trajectory. With CC-CV OPs in the set the spread
    # of trajectory lengths is much wider than it was for OP01-OP05, so tn no
    # longer reaches ~1 for the short OPs -- that is fine (tn is a scale, not a
    # progress bar) but a held-out OP longer than this reference gets tn > 1,
    # which build_op warns about.
    T_span_ref = float(max(r["t"].max() - r["t"][0] for r in raw))

    # ---- pooled train statistics --------------------------------------------
    T_pool, cfg_pool, q_pool, splits = [], [], [], []
    cfg_time_std, cfg_op_mean = [], []
    for r in raw:
        n_t = r["T"].shape[0]
        split_t = max(int(train_frac * n_t), 2)
        splits.append(split_t)
        T_pool.append(r["T"][:split_t].ravel())
        cfg_pool.append(r["cfg_ts"][:split_t])
        q_pool.append(r["q_dot"][:split_t])
        with np.errstate(invalid="ignore"):
            cfg_time_std.append(np.nanstd(r["cfg_ts"][:split_t], axis=0))
            cfg_op_mean.append(np.nanmean(r["cfg_ts"][:split_t], axis=0))

    T_flat = np.concatenate(T_pool)
    T_mu = float(T_flat.mean())
    T_sigma = float(_std_guard(T_flat.std()))
    T_min, T_max = float(T_flat.min()), float(T_flat.max())

    cfg_all = np.concatenate(cfg_pool, axis=0)
    config_mu = np.nanmean(cfg_all, axis=0)
    config_raw_sigma = np.nan_to_num(np.nanstd(cfg_all, axis=0), nan=0.0)
    config_active = config_raw_sigma > 1e-6
    config_sigma = _std_guard(config_raw_sigma)
    config_min = np.nan_to_num(np.nanmin(cfg_all, axis=0), nan=np.nan)
    config_max = np.nan_to_num(np.nanmax(cfg_all, axis=0), nan=np.nan)
    # Two very different questions, kept apart on purpose: does this channel
    # differ BETWEEN OPs, and does it move WITHIN an OP? The base project only
    # ever needed the first; a profile is the second.
    config_time_std = np.nan_to_num(np.max(np.stack(cfg_time_std), axis=0))
    config_across_std = np.nan_to_num(np.std(np.stack(cfg_op_mean), axis=0))

    q_all = np.concatenate(q_pool)
    q_mu = float(q_all.mean())
    q_sigma = float(_std_guard(np.array(q_all.std())))
    q_min, q_max = float(q_all.min()), float(q_all.max())

    # ---- driver rate scales (pooled over the training split) ----------------
    n_d, n_l = len(DRIVER_NAMES), len(driver_rate_lags)
    if use_driver_history and n_l:
        rate_pool = [
            _causal_rate_block(r["t"], _driver_matrix(r), driver_rate_lags)[:s]
            for r, s in zip(raw, splits)
        ]
        rates_all = np.concatenate(rate_pool, axis=0)
        driver_rate_rms = np.sqrt(np.nanmean(rates_all ** 2, axis=0))
        driver_rate_rms = np.nan_to_num(driver_rate_rms, nan=0.0)
        # A rate column counts as real if it is above a relative floor set by the
        # rate the driver WOULD show if it swung its own spread over one segment.
        # An absolute floor cannot work here: q_dot rates are ~1e4 W/m^3/s while
        # a mass-flow rate is ~1e-3 kg/s/s, and one absolute threshold would
        # either keep every rounding wobble of the first or discard all of the
        # second.
        drivers_all = np.concatenate([_driver_matrix(r)[:s]
                                      for r, s in zip(raw, splits)], axis=0)
        driver_spread = np.nan_to_num(np.nanstd(drivers_all, axis=0))
        ref = driver_spread[:, None] / np.asarray(driver_rate_lags)[None, :]
        driver_rate_active = driver_rate_rms > np.maximum(1e-6 * ref, 1e-30)
        driver_rate_rms = np.maximum(driver_rate_rms, 1e-30)
    else:
        driver_rate_rms = np.ones((n_d, max(n_l, 1)))
        driver_rate_active = np.zeros((n_d, max(n_l, 1)), dtype=bool)
    n_driver_rate = int(n_d * n_l) if (use_driver_history and n_l) else 0

    # ---- material properties / Fourier tensor / static feats -----------------
    rho, Cp, lam, Fo, region, q_mask = _grid_arrays(
        raw[0]["layer"], xyz, T_span_ref, L_ref)
    static_feat = _static_features(rho, Cp, lam, region, xn)

    consts = dict(
        T_mu=T_mu, T_sigma=T_sigma, T_span_ref=T_span_ref, xn=xn, Fo=Fo,
        q_mask=q_mask, region=region, rho=rho, Cp=Cp, config_mu=config_mu,
        config_sigma=config_sigma, config_active=config_active,
        static_feat=static_feat, q_mu=q_mu, q_sigma=q_sigma,
        driver_rate_lags=driver_rate_lags, driver_rate_rms=driver_rate_rms,
        driver_rate_active=driver_rate_active,
        use_driver_history=use_driver_history,
    )

    ops, dTdt_pool, Qsrc_pool, per_op_Qsrc_rms = [], [], [], {}
    for r, split_t in zip(raw, splits):
        op = _assemble_op(r, split_t, **consts)
        ops.append(op)
        tn = op.tn.astype(np.float64)
        Tn = op.Tn.astype(np.float64)
        dTdt = (Tn[2:] - Tn[:-2]) / (tn[2:] - tn[:-2])[:, None]
        dTdt_pool.append(dTdt[:split_t].ravel())
        Qsrc_pool.append(op.Qsrc[:split_t].ravel())
        per_op_Qsrc_rms[op.op_id] = float(
            np.sqrt((op.Qsrc[:split_t].astype(np.float64) ** 2).mean()))

    dTdt_scale = float(np.sqrt((np.concatenate(dTdt_pool) ** 2).mean())) + 1e-6
    aniso_scale = float(np.sqrt(np.mean(np.sum(Fo ** 2, axis=(1, 2))))) + 1e-6
    Qsrc_scale = float(np.sqrt((np.concatenate(Qsrc_pool) ** 2).mean())) + 1e-6
    # BC scale: the RMS spatial gradient the training data actually carries, so
    # a BC residual of 1 means "as steep as a typical gradient in this cell".
    # The BC drives dT/dx to zero AT x=0, so the yardstick cannot be read off
    # the boundary -- it is supposed to vanish there. 1/L_ref stays only as the
    # fallback for a grid with no x-neighbour pairs; it is a guess that does not
    # involve the temperature at all, yet it sets the range of w_bc that means
    # anything.
    bc_scale, bc_pairs = _measure_bc_scale(xn, ops, fallback=1.0 / L_ref)
    # phys_scale: the ONE scale physics.py divides the assembled residual by, so
    # L_phys lands at O(1) without altering the equation. Built from the two
    # genuine term magnitudes -- dTdt_scale and Qsrc_scale are RMS values of
    # terms that actually appear in the residual, both already in the shared
    # nondimensional units.
    #
    # aniso_scale is deliberately NOT in here. It is the RMS of the Fourier
    # tensor alone, with the grad^2 T factor missing, so it is not the magnitude
    # of the diffusion TERM; mixing it in sets the scale by a quantity the
    # residual never contains. It stays on the bundle for logging only.
    phys_scale = float(np.sqrt(dTdt_scale**2 + Qsrc_scale**2)) + 1e-6

    n_forcing = 1 + n_driver_rate
    trained_profiles = tuple(sorted({p for op in ops
                                     for p in op.profiles_detected
                                     if p in PROFILE_CHANNELS}))

    return NormBundle(
        ops=ops, T_mu=T_mu, T_sigma=T_sigma, T_span_ref=T_span_ref, L_ref=L_ref,
        xyz_min=xyz_min.astype(np.float64), config_mu=config_mu,
        config_sigma=config_sigma, config_active=config_active,
        n_config=len(CONFIG_ORDER), phys_scale=phys_scale,
        dTdt_scale=dTdt_scale, aniso_scale=aniso_scale, Qsrc_scale=Qsrc_scale,
        bc_scale=bc_scale, bc_pairs=bc_pairs, static_feat=static_feat,
        n_static=static_feat.shape[1], q_mu=q_mu, q_sigma=q_sigma,
        n_forcing=n_forcing, xn=xn, Fo=Fo, q_mask=q_mask, region=region,
        rho=rho, Cp=Cp, train_frac=train_frac,
        resample=resample, use_driver_history=bool(use_driver_history),
        driver_rate_lags=driver_rate_lags, driver_names=DRIVER_NAMES,
        driver_rate_rms=driver_rate_rms, driver_rate_active=driver_rate_active,
        n_driver_rate=n_driver_rate, config_time_std=config_time_std,
        config_across_std=config_across_std, config_min=config_min,
        config_max=config_max, q_min=q_min, q_max=q_max,
        T_min=T_min, T_max=T_max, train_ops=tuple(op_ids),
        trained_profiles=trained_profiles, per_op_Qsrc_rms=per_op_Qsrc_rms,
    )


def build_op(op_id: str, bundle: NormBundle, subsample_time: int = 2,
             train_frac: float | None = None) -> OPData:
    """Build one OPData for a HELD-OUT OP using ``bundle``'s training constants.

    Nothing is re-fitted -- temperature/config normalisation, ``L_ref``,
    ``T_span_ref``, the Fourier tensor and now also the driver-rate scales all
    come from the training bundle, so the held-out OP is a genuine out-of-sample
    test. The resampling mode and driver-history layout follow the bundle too:
    a held-out OP preprocessed differently from the training set would not be
    measuring the model.
    """
    r = _read_raw(op_id, subsample_time, bundle.resample)
    tf = bundle.train_frac if train_frac is None else train_frac
    split_t = max(int(tf * r["T"].shape[0]), 2)
    return _assemble_op(
        r, split_t, T_mu=bundle.T_mu, T_sigma=bundle.T_sigma,
        T_span_ref=bundle.T_span_ref, xn=bundle.xn, Fo=bundle.Fo,
        q_mask=bundle.q_mask, region=bundle.region, rho=bundle.rho,
        Cp=bundle.Cp, config_mu=bundle.config_mu,
        config_sigma=bundle.config_sigma, config_active=bundle.config_active,
        static_feat=bundle.static_feat, q_mu=bundle.q_mu,
        q_sigma=bundle.q_sigma, driver_rate_lags=bundle.driver_rate_lags,
        driver_rate_rms=bundle.driver_rate_rms,
        driver_rate_active=bundle.driver_rate_active,
        use_driver_history=bundle.use_driver_history,
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def cache_is_synthetic() -> bool:
    """True when the loaded OP bundles were written by make_synthetic_cache.py.

    An absolute MAE measured off that fixture says nothing about the real OPs,
    so a run on it must never be quoted as a result. Cheap to check and easy to
    forget, which is why ``train.py`` prints a banner rather than leaving it to
    a line in a README.

    ``DATA_CACHE`` is re-read on every call, so a test that repoints it is seen.
    Any synthetic bundle in the cache is enough to disqualify the run: a folder
    holding both kinds is not a dataset, and the banner has to fire on the
    mixture too, not only when the file that happens to sort first is the
    synthetic one.
    """
    try:
        for path in sorted(Path(DATA_CACHE).glob("OP*.npz")):
            with np.load(path, allow_pickle=True) as npz:
                if "synthetic" in npz.files:
                    return True
    except Exception:
        pass
    return False


def available_ops() -> List[str]:
    """OP ids that actually have a cached ``.npz`` bundle, sorted."""
    return sorted(p.stem for p in DATA_CACHE.glob("*.npz"))


def require_ops(*op_ids: str) -> None:
    """Fail fast if a requested OP has no cached bundle.

    A benchmark resolves its held-out OPs only after training the first grid
    point, so without this a typo costs a full training run before it surfaces.
    """
    have = set(available_ops())
    missing = [op for op in dict.fromkeys(op_ids) if op and op not in have]
    if missing:
        searched = "\n".join(f"                {c}" for c in _CACHE_CANDIDATES)
        raise SystemExit(
            f"missing cached OP bundle(s): {', '.join(missing)}\n"
            f"  using cache : {DATA_CACHE}"
            f"{' (does not exist)' if not DATA_CACHE.exists() else ''}\n"
            f"  available   : {', '.join(available_ops()) or '(none)'}\n"
            f"  searched    :\n{searched}\n"
            f"  preferred   : {PREFERRED_DATA_CACHE}\n"
            f"  build them  : python3 PINNmodulusTwo/generate_cache.py "
            f"{' '.join(missing)}\n"
            f"  NOTE: the data cache is not in git. A fresh clone never has it; "
            f"it has to be generated or copied onto this machine first."
        )


def normalisation_report(bundle: NormBundle) -> List[str]:
    """The constants every downstream number depends on, and their spread.

    Printed at the top of every run because these are exactly the quantities
    that changed when the profile OPs joined the training set -- and because
    ``w_phys`` / ``w_bc`` are only meaningful relative to them.
    """
    lines = [
        "normalisation constants (fitted on the training split ONLY):",
        f"  T_mu={bundle.T_mu:.4g} C  T_sigma={bundle.T_sigma:.4g} C  "
        f"(train range {bundle.T_min:.1f} .. {bundle.T_max:.1f} C)",
        f"  L_ref={bundle.L_ref:.5g} m  T_span_ref={bundle.T_span_ref:.1f} s",
        f"  q_mu={bundle.q_mu:.4g}  q_sigma={bundle.q_sigma:.4g} W/m^3  "
        f"(train range {bundle.q_min:.4g} .. {bundle.q_max:.4g})",
        f"  phys_scale={bundle.phys_scale:.4g}  dTdt_scale={bundle.dTdt_scale:.4g}"
        f"  aniso_scale={bundle.aniso_scale:.4g}  Qsrc_scale={bundle.Qsrc_scale:.4g}"
        f"  bc_scale={bundle.bc_scale:.4g} "
        f"(from {bundle.bc_pairs} x-neighbour pairs)"
        + ("  [FALLBACK 1/L_ref -- a guess, not a measurement]"
           if bundle.bc_pairs == 0 else ""),
        f"  resample={bundle.resample}  driver_history="
        f"{'on' if bundle.use_driver_history else 'off'}  "
        f"rate_lags={list(bundle.driver_rate_lags)} s  "
        f"n_forcing={bundle.n_forcing} (1 + {bundle.n_driver_rate} rate channels)",
    ]

    # The amplification of the hybrid TEMPERATURE history. It used to be printed
    # only by train.py, which meant the one number a data check is supposed to
    # produce could not be read off a data check.
    #
    # It is reported against the subsample this bundle was loaded at, because
    # dTdt_scale is an RMS of a central difference ON THAT GRID: a coarse grid
    # smooths the derivative and a fine one does not, so A is not a property of
    # the data alone. Quoting a value without its subsample is how two people
    # end up comparing different numbers under the same name.
    dt_s = bundle.ops[0].dtn * bundle.T_span_ref if bundle.ops else float("nan")
    amps = [(lag, 1.0 / ((lag / bundle.T_span_ref) * bundle.dTdt_scale + 1e-30))
            for lag in TEMPERATURE_RATE_LAGS_FOR_REPORT]
    lines.append(
        "  A = 1/(lag_n * dTdt_scale), the factor the hybrid temperature history "
        "multiplies a one-step jump by:"
    )
    lines.append(
        "    " + "   ".join(f"lag {lag:g}s -> A = {a:.1f}" for lag, a in amps)
        + f"      (at dt = {dt_s:.3g} s; A DEPENDS ON THIS)"
    )
    rms = bundle.per_op_Qsrc_rms
    if rms:
        lo = min(rms, key=rms.get)
        hi = max(rms, key=rms.get)
        ratio = rms[hi] / (rms[lo] + 1e-30)
        lines.append(
            f"  Qsrc RMS per OP spans {ratio:.1f}x "
            f"({lo}={rms[lo]:.4g} .. {hi}={rms[hi]:.4g}). Qsrc_scale is ONE "
            f"pooled divisor for all of them, so w_phys means something "
            f"different here than in the base project."
        )
    lines.append("  config channels (physical units, pooled over the train split):")
    for i, name in enumerate(CONFIG_ORDER):
        kind = ("profile" if bundle.config_time_std[i] > PROFILE_DETECT_TOL
                else "constant")
        state = "active" if bundle.config_active[i] else "DEAD -> forced to 0"
        lines.append(
            f"    {name:<20} {kind:<9} {state:<20} "
            f"mu={bundle.config_mu[i]:>10.4g}  sigma={bundle.config_sigma[i]:>10.4g}"
            f"  within-OP std={bundle.config_time_std[i]:>10.4g}"
            f"  between-OP std={bundle.config_across_std[i]:>10.4g}"
        )
    if bundle.use_driver_history and bundle.n_driver_rate:
        lines.append("  driver rate channels (divisor = pooled train RMS):")
        for d, name in enumerate(bundle.driver_names):
            cells = "  ".join(
                f"L={lag:g}s: {bundle.driver_rate_rms[d, i]:.4g}"
                f"{'' if bundle.driver_rate_active[d, i] else ' (DEAD->0)'}"
                for i, lag in enumerate(bundle.driver_rate_lags)
            )
            lines.append(f"    {name:<20} {cells}")
    return lines


def hybrid_rate_amplification(rate_scale: float,
                              rate_lags_n: Sequence[float]) -> np.ndarray:
    """How much the hybrid TEMPERATURE history magnifies a one-step level jump.

    ``model._history_hybrid`` feeds the network
    ``(T_end - T_start) / (lag_n * rate_scale)``. For a genuine rate that is the
    right normalisation and lands the channel at O(1). Early in a free-running
    rollout it is not a genuine rate: step 1 differences the untrained network's
    first output against the imposed initial condition, so the numerator is a
    LEVEL jump and the channel returns that jump multiplied by

        A = 1 / (lag_n * rate_scale).

    That product is the number to watch, and this extension changes it. The base
    project's ``rate_scale`` is ``dTdt_scale`` -- the RMS of ``dTn/dtn`` on
    z-scored temperature -- so widening the pooled ``T_sigma`` shrinks ``Tn``,
    shrinks its time derivative, and therefore *raises* A. Pooling OP01-OP16
    does exactly that: OP14 starts at 0 C and OP05 at 40 C, so a large part of
    ``T_sigma`` is now between-OP offset that contributes nothing to any OP's own
    rate. The level jumps do not shrink with ``T_sigma`` -- they are what
    ``T_sigma`` is made of -- so A grows and the first few rollout steps can be
    amplified into saturation and then into NaN.

    Returned per lag so the shortest segment, which is the largest A, is visible.
    """
    lags = np.asarray([float(v) for v in rate_lags_n], dtype=np.float64)
    if lags.size == 0:
        return np.zeros(0)
    return 1.0 / (np.maximum(lags, 1e-30) * max(float(rate_scale), 1e-30))


def effective_rate_scale(dTdt_scale: float, rate_lags_n: Sequence[float],
                         max_amp: float = 0.0):
    """``(rate_scale, report_lines)`` for the hybrid temperature history.

    ``max_amp <= 0`` returns ``dTdt_scale`` unchanged -- byte-for-byte the base
    project's behaviour, which is the default here on purpose: silently
    rescaling a channel would make this extension's model quietly different from
    the one the base results were produced with.

    With ``max_amp > 0`` the scale is raised just far enough that the worst-case
    amplification above stays at ``max_amp``. That damps the opening steps of the
    rollout and leaves the converged rate channel scaled by a constant the
    network can absorb into its first layer. It is a guard against divergence,
    not a tuning knob: reach for it when a run aborts with a non-finite
    ``L_data`` in the first epoch, and record that you used it.
    """
    amps = hybrid_rate_amplification(dTdt_scale, rate_lags_n)
    lines = []
    if amps.size:
        lines.append(
            "  hybrid history amplification A = 1/(lag_n * rate_scale) per lag: "
            + ", ".join(f"{a:.4g}" for a in amps)
            + f"   (rate_scale = dTdt_scale = {dTdt_scale:.4g})"
        )
    scale = float(dTdt_scale)
    if max_amp and max_amp > 0 and amps.size and amps.max() > max_amp:
        lags = np.asarray([float(v) for v in rate_lags_n], dtype=np.float64)
        scale = 1.0 / (max(lags.min(), 1e-30) * float(max_amp))
        lines.append(
            f"  A exceeds --max-rate-amp {max_amp:g}; raising rate_scale "
            f"{dTdt_scale:.4g} -> {scale:.4g} so the worst lag amplifies by "
            f"{max_amp:g}. This makes the model differ from an unguarded run - "
            f"say so when reporting the result."
        )
    elif amps.size and amps.max() > 100.0 and not (max_amp and max_amp > 0):
        lines.append(
            f"  [WARN] A reaches {amps.max():.4g}. The opening steps of the "
            f"free-running rollout difference the untrained network against the "
            f"initial condition, and that level jump is magnified by A. If "
            f"training aborts with a non-finite L_data in epoch 1, this is the "
            f"first thing to try: --max-rate-amp 50 (or --history-mode raw to "
            f"remove the rate channels entirely)."
        )
    return scale, lines


def energy_balance_report(bundle: NormBundle) -> List[str]:
    """Can the heat source account for the temperature the cell actually reaches?

    The nondimensional equation is ``dTn/dtn = Fo : grad^2 Tn + Qsrc``. Averaged
    over the cell the diffusion term integrates to the boundary flux, so for an
    operating point with **no coolant flow** almost nothing leaves and

        <dTn/dtn>  ~  <Qsrc>        (over the heated region)

    has to hold to within the cooling that is left. That makes it a real test of
    the SOURCE, and it needs no model at all -- only the bundle.

    This check is why the source bug was found at all, so it is worth keeping
    even now that the bug is fixed. A UNIFORM error in ``Qsrc`` is invisible
    everywhere else: the EMA loss balancer divides it straight back out,
    ``L_phys`` still lands at O(1), and ``phys_scale``/``Qsrc_scale`` are built
    from the same wrong numbers, so they agree with it. Only an energy argument
    sees it, and only a no-flow OP makes the argument airtight.

    What it caught on 31.08.2026: ``_read_raw`` divided the total JR1 power by
    ``V_JR1 * N_JR1_POINTS``, i.e. by the grid-point count as well as the
    volume. Measured shortfall 110x on OP07 and 107x on OP14, against a bug
    factor of 121; removing the point count lands them at 0.91 and 0.88, just
    under 1 as they must be, and every cooled OP between 0.45 and 0.63 ordered
    by its flow rate.

    What it would mean if it fires again: the residual reduces towards
    ``dTdt = Fo : grad^2 Tn``, i.e. the physics term telling the network the
    cell is heated by nothing and must reach its temperature by conduction
    alone. That is a different PDE from the one the data obeys, and no weight on
    it can be right.

    Reported per OP, with the ratio; a zero-flow OP whose ratio is far from 1 is
    the conclusive case.
    """
    try:
        flow_idx = CONFIG_ORDER.index("fluid_mass_flow")
    except ValueError:                                   # pragma: no cover
        flow_idx = None

    lines = [
        "energy balance (does the source explain the temperature rise?):",
        "  <dTn/dtn> against <Qsrc>, both over the heated JR1 region and the",
        "  training split. A no-flow OP has almost nowhere to lose heat, so its",
        "  ratio has to be near 1 -- see energy_balance_report for what a large",
        "  ratio would mean.",
    ]
    worst = 0.0
    worst_noflow = 0.0
    for op in bundle.ops:
        jr1 = np.asarray(bundle.q_mask) > 0.5
        if not jr1.any():
            continue
        n = max(2, int(op.split_t))
        tn = np.asarray(op.tn, dtype=np.float64)[:n]
        Tn = np.asarray(op.Tn, dtype=np.float64)[:n][:, jr1]
        if tn.shape[0] < 3:
            continue
        # central difference, same stencil the scales are built from
        dTdt = (Tn[2:] - Tn[:-2]) / (tn[2:] - tn[:-2])[:, None]
        mean_dTdt = float(np.abs(dTdt).mean())
        mean_Qsrc = float(np.abs(np.asarray(op.Qsrc, dtype=np.float64)[:n][:, jr1]).mean())
        ratio = mean_dTdt / (mean_Qsrc + 1e-30)

        flow = float("nan")
        if flow_idx is not None and np.asarray(op.cfg_phys).size:
            flow = float(np.nanmean(np.asarray(op.cfg_phys)[:n, flow_idx]))
        no_flow = np.isfinite(flow) and abs(flow) < 1e-9
        tag = "  <- NO FLOW: this one has to balance" if no_flow else ""
        lines.append(
            f"  {op.op_id}  <|dTn/dtn|>={mean_dTdt:9.4g}  <|Qsrc|>={mean_Qsrc:9.4g}"
            f"  ratio={ratio:8.1f}x  flow={flow:.4g}{tag}"
        )
        worst = max(worst, ratio)
        if no_flow:
            worst_noflow = max(worst_noflow, ratio)

    ref = worst_noflow or worst
    if ref > 10.0:
        basis = ("on the no-flow OP, where almost nothing can be carried away, so"
                 " there is nowhere else for that energy to have come from"
                 if worst_noflow else
                 "on the worst OP. No zero-flow OP was in this set, so cooling"
                 " could in principle explain part of it -- but not a factor this"
                 " large, and OP07/OP14 settle it outright")
        lines += [
            "",
            f"  [ENERGY] the source is short by a factor of ~{ref:.0f} {basis}.",
            "  The heated region rises far faster than Qsrc can drive it, so Qsrc",
            "  is too SMALL rather than the temperature too large.",
            "",
            "  NOTE: the known cause of exactly this symptom is already fixed. Until",
            "  31.08.2026 _read_raw divided the total JR1 power by V_JR1 *and* by the",
            "  grid-point count (121), which put the source 121x low. If this warning",
            "  is still firing, it is something ELSE -- do not re-apply a point-count",
            "  or volume factor on top.",
            "  Worth checking next: the unit of q_source[:, 0] as exported, the value",
            "  of V_JR1 against the real geometry, and whether rho*Cp for the heated",
            "  region is right.",
            "  Until it is settled, w_phys is not tunable: the residual would describe",
            "  a cell heated by ~nothing, which is not the cell the data came from.",
        ]
    elif ref:
        lines.append(f"  balance holds to within {ref:.1f}x on the binding OP.")
    return lines


def profile_report(bundle: NormBundle, extra_ops: Sequence[OPData] = ()) -> List[str]:
    """What each OP's bundle actually contains, next to what the sheet claims.

    The plan sheet in ``op_registry`` is a transcription and can be wrong; the
    ``.npz`` is the ground truth. Disagreement is worth seeing, so both are
    printed and mismatches are marked.

    On a mismatch there is a THIRD source worth consulting, and it is what
    separates the two very different causes: ``meta_json.profile_flags``, which
    the upstream assembly wrote to record what it BELIEVED it was producing.

    * the sheet claims a profile, the assembly flagged it, the data is constant
      -> the assembly ran but the profile file was missing or empty, and the
         channel silently fell back to its scalar. A data-build problem.
    * the sheet claims a profile and the assembly never flagged it
      -> the sheet is wrong for this OP, or this OP was exported without it.

    Guessing between those two costs a rebuild of the wrong thing, so the flags
    are printed whenever they add information.
    """
    from op_registry import OPS, profiles_of

    lines = ["per-OP profiles (detected from the bundle vs. the plan sheet):"]
    for op in list(bundle.ops) + list(extra_ops):
        detected = tuple(p for p in op.profiles_detected if p in PROFILE_CHANNELS)
        claimed = tuple(profiles_of(op.op_id)) if op.op_id in OPS else ()
        flag = "" if set(detected) == set(claimed) else "   <-- MISMATCH"
        role = "train" if op in bundle.ops else "held out"
        lines.append(
            f"  {op.op_id} [{role:<8}] n_t={op.n_t:<6} "
            f"detected={','.join(detected) or '-':<45} "
            f"sheet={','.join(claimed) or '-':<45} "
            f"transient={op.transient_frac*100:5.1f}%{flag}"
        )
        if flag:
            meta = tuple(m for m in op.profile_flags_meta if m in PROFILE_CHANNELS)
            missing = tuple(c for c in claimed if c not in detected)
            extra = tuple(c for c in detected if c not in claimed)
            lines.append(
                f"      upstream assembly flagged: "
                f"{','.join(meta) or '(nothing, or no meta_json in the bundle)'}"
            )
            for chan in missing:
                if chan in meta:
                    lines.append(
                        f"      ! {chan}: the assembly flagged it as a profile but "
                        f"the values do not vary. The profile file was missing or "
                        f"empty and the channel fell back to its scalar -- rebuild "
                        f"this OP and check its raw export."
                    )
                else:
                    lines.append(
                        f"      ! {chan}: the sheet claims it, the assembly never "
                        f"flagged it, and the values do not vary. Either the sheet "
                        f"is wrong for {op.op_id} or it was exported without this "
                        f"profile. Believe the bundle, then fix whichever is wrong."
                    )
            for chan in extra:
                lines.append(
                    f"      ! {chan}: varies in the bundle but the sheet does not "
                    f"claim it. The sheet is a transcription -- check it against "
                    f"the source before trusting the tier for {op.op_id}."
                )
        for name, (lo, hi) in sorted(op.profile_coverage.items()):
            t_lo, t_hi = float(op.t[0]), float(op.t[-1])
            if lo > t_lo + 1e-6 or hi < t_hi - 1e-6:
                lines.append(
                    f"      ! profile {name} covers {lo:.1f}..{hi:.1f} s but the "
                    f"OP runs {t_lo:.1f}..{t_hi:.1f} s; outside that range the "
                    f"first/last profile value is held flat."
                )
    return lines


def coverage_report(bundle: NormBundle, op: OPData) -> List[str]:
    """Does this held-out OP stay inside the ranges the constants were fitted on?

    A held-out OP outside the trained range is not testing generalisation, it is
    testing extrapolation, and the two deserve different words in a result
    table. Reported per channel, in physical units, with the overshoot expressed
    in training sigmas because that is what the network actually sees.
    """
    lines = []
    for i, name in enumerate(CONFIG_ORDER):
        if not bundle.config_active[i]:
            continue
        col = op.cfg_phys[:, i]
        if not np.isfinite(col).any():
            lines.append(f"  {name}: no value in this bundle -> filled with the "
                         f"training mean and fed as a neutral feature")
            continue
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        below = (bundle.config_min[i] - lo) / bundle.config_sigma[i]
        above = (hi - bundle.config_max[i]) / bundle.config_sigma[i]
        if below > 1e-3 or above > 1e-3:
            lines.append(
                f"  {name}: {lo:.4g} .. {hi:.4g} leaves the trained "
                f"{bundle.config_min[i]:.4g} .. {bundle.config_max[i]:.4g} "
                f"by {max(below, 0):.2f} sigma below / {max(above, 0):.2f} "
                f"sigma above"
            )
    q_lo, q_hi = float(op.q_dot.min()), float(op.q_dot.max())
    if q_lo < bundle.q_min - 1e-9 or q_hi > bundle.q_max + 1e-9:
        lines.append(
            f"  q_dot: {q_lo:.4g} .. {q_hi:.4g} W/m^3 leaves the trained "
            f"{bundle.q_min:.4g} .. {bundle.q_max:.4g} by "
            f"{max(bundle.q_min - q_lo, 0) / bundle.q_sigma:.2f} sigma below / "
            f"{max(q_hi - bundle.q_max, 0) / bundle.q_sigma:.2f} sigma above"
        )
    new_profiles = [p for p in op.profiles_detected
                    if p in PROFILE_CHANNELS and p not in bundle.trained_profiles]
    if new_profiles:
        lines.append(
            f"  profile type(s) {', '.join(new_profiles)} vary in time here but "
            f"in no training OP -- the matching rate channels were dead during "
            f"training and are being asked to mean something for the first time."
        )
    span = float(op.t[-1] - op.t[0])
    if span > bundle.T_span_ref + 1e-6:
        lines.append(
            f"  trajectory is {span:.1f} s but T_span_ref is "
            f"{bundle.T_span_ref:.1f} s, so tn runs past 1.0 -- outside the "
            f"normalised time range every training OP lived in."
        )
    if not lines:
        lines.append("  inside the trained range on every active channel")
    return lines


if __name__ == "__main__":
    from op_registry import DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS, DEFAULT_TEST_OPS

    require_ops(*DEFAULT_TRAIN_OPS, *DEFAULT_VAL_OPS, *DEFAULT_TEST_OPS)
    b = load_ops(op_ids=list(DEFAULT_TRAIN_OPS), subsample_time=40)
    print("\n".join(normalisation_report(b)))
    print("\n".join(energy_balance_report(b)))
    held = [build_op(o, b, subsample_time=40)
            for o in list(DEFAULT_VAL_OPS) + list(DEFAULT_TEST_OPS)]
    print("\n".join(profile_report(b, held)))
    for op in held:
        print(f"coverage {op.op_id}:")
        print("\n".join(coverage_report(b, op)))
