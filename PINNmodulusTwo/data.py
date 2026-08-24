"""Multi-OP data loader + preprocessing for the Approach-2 PINN (PINNmodulusTwo).

Approach 2 (see Notion "Battery Model with NVIDIA MODULUS"):
    Use Modulus as a tool but bring our own *recurrence* (built in PyTorch).
    Spatial derivatives come from autograd; the time derivative is a finite
    difference over the temperature history  T_{t-delta}, T_{t-2 delta}, ...

Scope for this module
---------------------
* Trains on OP01, OP02, OP03 (temperature only -- bc_V is intentionally ignored).
* Reads everything from the cached ``.npz`` bundles (no CSV dependency): the JR1
  heat source is ``q_source[:, 0]`` (column order ``[jr1_w, jr2_w, total_w]``).
* Non-dimensionalisation matches the existing pinnANDmodulus pipeline so the
  physics residual is directly comparable:
    - coords scaled by a single ``L_ref`` (geometric mean of the axis extents),
    - time scaled by a shared ``T_span_ref`` to ``tn`` in ~[0, 1],
    - temperature z-scored with pooled train statistics over OP01-03,
    - anisotropic Fourier tensor ``Fo = lambda * T_span_ref / (rho Cp L_ref^2)``.

Profiles in the configs
-----------------------
The seven simulation configs *can* vary in time (a "profile"). Two OPs may share
the same instantaneous config at some time ``t`` yet have very different
temperatures because their *history* differed -- this is exactly why recurrence
is needed. We therefore expose the configs as a per-timestep feature block
``config_feat(t)`` (constant in time for OP01-03, but the shape already supports
time-varying profiles via the ``sim_config_ts_*`` arrays when present).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


# Where the OP*.npz bundles may live, most specific first. The cache is not in
# git (see .gitignore), so it stays wherever it already is on a given machine --
# hence the fallbacks: a checkout that moved the legacy folders must keep finding
# a cache that did not move with them. First existing hit wins; if none exists,
# the preferred top-level location is reported so errors name the right path.
_CACHE_CANDIDATES = (
    THIS_DIR / "data_cache",                    # project-local override
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

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from materials import load_material_properties  # noqa: E402  (local, folder-driven)

# JR1 volume + Gleichverteilung factor (matches load_op01._load_heat_source).
V_JR1 = 4.394793e-04
N_JR1_POINTS = 121

# Order of the 7 config scalars in ``sim_config_scalar``.
CONFIG_ORDER = [
    "c_rate",
    "cell_current",
    "fluid_initial_temp",
    "fluid_inlet_temp",
    "fluid_mass_flow",
    "soc_start",
    "solid_initial_temp",
]


@dataclass
class OPData:
    """Per-OP non-dimensional arrays (all float32 unless noted)."""

    op_id: str
    t: np.ndarray            # physical time (n_t,)
    tn: np.ndarray           # scaled time ~[0, 1] (n_t,)
    xn: np.ndarray           # scaled coords (n_points, 3)
    Tn: np.ndarray           # z-scored temperature (n_t, n_points)
    Tn_ic: np.ndarray        # z-scored initial condition (n_points,)
    config_feat: np.ndarray  # z-scored config features (n_t, n_config)
    static_feat: np.ndarray  # per-point static material/geometry feats (n_points, n_static)
    forcing_feat: np.ndarray  # normalised heat source q_dot(t) (n_t, n_forcing)
    Fo: np.ndarray           # Fourier tensor (n_points, 3, 3)
    Qsrc: np.ndarray         # nondim source term (n_t, n_points)
    q_mask: np.ndarray       # JR1 mask (n_points,)
    region: np.ndarray       # 0=CC, 1=JR1, 2=Housing (n_points,)
    split_t: int             # train/test boundary along time
    n_t: int
    n_points: int
    dtn: float               # uniform spacing of tn
    T_lab: np.ndarray        # physical temperature labels (n_t, n_points)


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
    # extra input features (material/geometry + forcing)
    static_feat: np.ndarray     # (n_points, n_static), grid-shared
    n_static: int
    q_mu: float                 # heat-source normalisation (pooled train)
    q_sigma: float
    n_forcing: int
    # grid-level arrays (identical across OPs), reused for held-out OPs
    xn: np.ndarray
    Fo: np.ndarray
    q_mask: np.ndarray
    region: np.ndarray
    rho: np.ndarray
    Cp: np.ndarray
    train_frac: float


def _config_vector(config: Dict[str, float]) -> np.ndarray:
    # Missing scalars (present only as a time-series profile) -> NaN, filled later.
    return np.array([float(config.get(k, np.nan)) for k in CONFIG_ORDER], dtype=np.float64)


def _config_timeseries(npz, t: np.ndarray, config: Dict[str, float]) -> np.ndarray:
    """Return (n_t, n_config) config features, honouring time-varying profiles.

    For OP01-03 there are no ``sim_config_ts_*`` arrays, so every channel is the
    constant scalar broadcast over time. When a profile *is* present it is linearly
    interpolated onto the (subsampled) time grid.
    """
    ts_names = json.loads(str(npz["sim_config_ts_names_json"].item()))
    base = _config_vector(config)
    feat = np.tile(base, (t.shape[0], 1))
    for name in ts_names:
        if name not in CONFIG_ORDER:
            continue
        col = CONFIG_ORDER.index(name)
        t_src = np.asarray(npz[f"sim_config_ts_{name}_t"], dtype=np.float64)
        v_src = np.asarray(npz[f"sim_config_ts_{name}_v"], dtype=np.float64)
        feat[:, col] = np.interp(t, t_src, v_src)
    return feat


def _std_guard(sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.maximum(sigma, eps)


def _read_raw(op_id: str, subsample_time: int) -> dict:
    npz = np.load(DATA_CACHE / f"{op_id}.npz", allow_pickle=True)
    names = json.loads(str(npz["sim_config_scalar_names_json"].item()))
    config = dict(zip(names, [float(v) for v in npz["sim_config_scalar"]]))
    t = np.asarray(npz["t_fast"], dtype=np.float64)[::subsample_time]
    T = np.asarray(npz["T"], dtype=np.float64)[::subsample_time]
    jr1_w = np.asarray(npz["q_source"], dtype=np.float64)[::subsample_time, 0]
    q_dot = jr1_w / (V_JR1 * N_JR1_POINTS)  # W/m^3, Gleichverteilung
    cfg_ts = _config_timeseries(npz, t, config)
    return dict(op_id=op_id, t=t, T=T, q_dot=q_dot, cfg_ts=cfg_ts,
                xyz=np.asarray(npz["xyz"], dtype=np.float64),
                layer=np.asarray(npz["layer"]))


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


def _normalise_config(cfg_ts, config_mu, config_sigma, config_active):
    """z-score active config channels; force dead (zero-variance) channels to 0.

    Missing values (NaN) are replaced by the training mean first, so an active
    channel that is absent for a held-out OP degrades to a neutral 0 feature
    instead of exploding through the epsilon guard.
    """
    cfg = np.where(np.isnan(cfg_ts), config_mu[None, :], cfg_ts)
    feat = (cfg - config_mu) / config_sigma
    feat[:, ~config_active] = 0.0
    return feat.astype(np.float32)


def _static_features(rho, Cp, lam, region, xn):
    """Per-point, time-independent features so the net can tell materials apart.

    Channels: [thermal diffusivity (z-scored), JR1 indicator, x-plane (z-scored)].
    Grid-shared, so the same array is reused for every OP (train or held-out).
    """
    lam_iso = (lam[:, 0, 0] + lam[:, 1, 1] + lam[:, 2, 2]) / 3.0
    alpha = lam_iso / (rho * Cp + 1e-30)          # thermal diffusivity [m^2/s]
    alpha_z = (alpha - alpha.mean()) / (alpha.std() + 1e-12)
    jr1 = (region == 1).astype(np.float64)
    xcol = xn[:, 0]
    x_z = (xcol - xcol.mean()) / (xcol.std() + 1e-12)
    return np.stack([alpha_z, jr1, x_z], axis=1).astype(np.float32)


def _assemble_op(r, split_t, *, T_mu, T_sigma, T_span_ref, xn, Fo, q_mask, region,
                 rho, Cp, config_mu, config_sigma, config_active,
                 static_feat, q_mu, q_sigma) -> OPData:
    t = r["t"]
    n_t = t.shape[0]
    tn = (t - t[0]) / (T_span_ref + 1e-12)
    dtn = float(tn[1] - tn[0]) if n_t > 1 else 1.0
    Tn = (r["T"] - T_mu) / T_sigma
    Tn_ic = Tn[0].copy()
    cfg_feat = _normalise_config(r["cfg_ts"], config_mu, config_sigma, config_active)
    forcing_feat = ((r["q_dot"] - q_mu) / q_sigma).astype(np.float32).reshape(-1, 1)
    Qsrc = (
        r["q_dot"][:, None] * q_mask[None, :] * T_span_ref
        / (rho[None, :] * Cp[None, :] * T_sigma)
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
    )


def load_ops(
    op_ids: List[str] | None = None,
    subsample_time: int = 40,
    train_frac: float = 0.8,
) -> NormBundle:
    """Load the requested OPs and build a shared non-dimensional dataset.

    Args:
        op_ids: e.g. ``["OP01", "OP02", "OP03"]`` (default).
        subsample_time: keep every N-th raw timestep (raw dt = 0.1 s).
        train_frac: fraction of each OP's timeline used for train stats/metrics.
    """
    if op_ids is None:
        op_ids = ["OP01", "OP02", "OP03"]

    raw = [_read_raw(op_id, subsample_time) for op_id in op_ids]

    # ---- shared geometry (grid identical across OPs) --------------------------
    xyz = raw[0]["xyz"]
    xyz_min = xyz.min(axis=0)
    L_axis = xyz.max(axis=0) - xyz_min
    L_ref = float(np.prod(L_axis) ** (1.0 / 3.0))
    xn = (xyz - xyz_min) / L_ref

    # ---- shared time scale ----------------------------------------------------
    T_span_ref = float(max(r["t"].max() - r["t"][0] for r in raw))

    # ---- pooled train statistics (temperature + configs + heat source) --------
    T_train_pool, cfg_train_pool, q_train_pool, splits = [], [], [], []
    for r in raw:
        n_t = r["T"].shape[0]
        split_t = int(train_frac * n_t)
        splits.append(split_t)
        T_train_pool.append(r["T"][:split_t].ravel())
        cfg_train_pool.append(np.where(np.isnan(r["cfg_ts"][:split_t]),
                                       np.nan, r["cfg_ts"][:split_t]))
        q_train_pool.append(r["q_dot"][:split_t])
    T_mu = float(np.concatenate(T_train_pool).mean())
    T_sigma = float(_std_guard(np.concatenate(T_train_pool).std()))
    cfg_all = np.concatenate(cfg_train_pool, axis=0)
    config_mu = np.nanmean(cfg_all, axis=0)
    config_raw_sigma = np.nan_to_num(np.nanstd(cfg_all, axis=0), nan=0.0)
    config_active = config_raw_sigma > 1e-6   # channels that actually vary in train
    config_sigma = _std_guard(config_raw_sigma)
    q_all = np.concatenate(q_train_pool)
    q_mu = float(q_all.mean())
    q_sigma = float(_std_guard(np.array(q_all.std())))

    # ---- material properties / Fourier tensor / static feats (grid-level) -----
    rho, Cp, lam, Fo, region, q_mask = _grid_arrays(raw[0]["layer"], xyz, T_span_ref, L_ref)
    static_feat = _static_features(rho, Cp, lam, region, xn)

    consts = dict(
        T_mu=T_mu, T_sigma=T_sigma, T_span_ref=T_span_ref, xn=xn, Fo=Fo,
        q_mask=q_mask, region=region, rho=rho, Cp=Cp, config_mu=config_mu,
        config_sigma=config_sigma, config_active=config_active,
        static_feat=static_feat, q_mu=q_mu, q_sigma=q_sigma,
    )

    ops, dTdt_pool, Qsrc_pool = [], [], []
    for r, split_t in zip(raw, splits):
        op = _assemble_op(r, split_t, **consts)
        ops.append(op)
        tn = op.tn.astype(np.float64)
        Tn = op.Tn.astype(np.float64)
        dTdt = (Tn[2:] - Tn[:-2]) / (tn[2:] - tn[:-2])[:, None]
        dTdt_pool.append(dTdt[:split_t].ravel())
        Qsrc_pool.append(op.Qsrc[:split_t].ravel())

    dTdt_scale = float(np.sqrt((np.concatenate(dTdt_pool) ** 2).mean())) + 1e-6
    aniso_scale = float(np.sqrt(np.mean(np.sum(Fo ** 2, axis=(1, 2))))) + 1e-6
    Qsrc_scale = float(np.sqrt((np.concatenate(Qsrc_pool) ** 2).mean())) + 1e-6
    # BC scale: expected magnitude of temperature spatial gradient dT/dx
    # Heuristic: typical temperature variation (T_sigma) over typical length (L_ref)
    # Since T is z-scored, T_sigma~1 in normalized units, and x is in [0, L_axis/L_ref]
    bc_scale = 1.0 / L_ref  # normalized temperature per normalized length
    # phys_scale: RMS of the combined residual magnitude to scale entire PDE loss
    phys_scale = float(np.sqrt(dTdt_scale**2 + aniso_scale**2 + Qsrc_scale**2)) + 1e-6

    return NormBundle(
        ops=ops, T_mu=T_mu, T_sigma=T_sigma, T_span_ref=T_span_ref, L_ref=L_ref,
        xyz_min=xyz_min.astype(np.float64), config_mu=config_mu,
        config_sigma=config_sigma, config_active=config_active,
        n_config=len(CONFIG_ORDER), phys_scale=phys_scale,
        dTdt_scale=dTdt_scale, aniso_scale=aniso_scale, Qsrc_scale=Qsrc_scale,
        bc_scale=bc_scale,
        static_feat=static_feat, n_static=static_feat.shape[1],
        q_mu=q_mu, q_sigma=q_sigma, n_forcing=1,
        xn=xn, Fo=Fo, q_mask=q_mask, region=region, rho=rho, Cp=Cp,
        train_frac=train_frac,
    )


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
            f"  build them  : python3 PINNmodulusTwo/generate_cache.py"
        )


def build_op(op_id: str, bundle: NormBundle, subsample_time: int = 40,
             train_frac: float | None = None) -> OPData:
    """Build one OPData for a HELD-OUT OP using ``bundle``'s training constants.

    Nothing is re-fitted: temperature/config normalisation, ``L_ref``,
    ``T_span_ref`` and the Fourier tensor all come from the training bundle, so
    the held-out OP is a genuine out-of-sample test. ``train_frac`` defaults to
    the bundle's value only for the split marker; the whole OP is unseen.
    """
    r = _read_raw(op_id, subsample_time)
    tf = bundle.train_frac if train_frac is None else train_frac
    split_t = int(tf * r["T"].shape[0])
    return _assemble_op(
        r, split_t, T_mu=bundle.T_mu, T_sigma=bundle.T_sigma,
        T_span_ref=bundle.T_span_ref, xn=bundle.xn, Fo=bundle.Fo,
        q_mask=bundle.q_mask, region=bundle.region, rho=bundle.rho, Cp=bundle.Cp,
        config_mu=bundle.config_mu, config_sigma=bundle.config_sigma,
        config_active=bundle.config_active, static_feat=bundle.static_feat,
        q_mu=bundle.q_mu, q_sigma=bundle.q_sigma,
    )


if __name__ == "__main__":
    b = load_ops(subsample_time=40)
    print(f"loaded {len(b.ops)} OPs | T_mu={b.T_mu:.3f} T_sigma={b.T_sigma:.3f} "
          f"L_ref={b.L_ref:.4g} T_span_ref={b.T_span_ref:.1f}s "
          f"phys_scale={b.phys_scale:.4g} dTdt_scale={b.dTdt_scale:.4g} "
          f"aniso_scale={b.aniso_scale:.4g} Qsrc_scale={b.Qsrc_scale:.4g} "
          f"bc_scale={b.bc_scale:.4g} n_config={b.n_config}")
    print(f"  config_active={b.config_active.tolist()}  names={CONFIG_ORDER}")
    for op in b.ops:
        print(f"  {op.op_id}: n_t={op.n_t} n_points={op.n_points} "
              f"split_t={op.split_t} dtn={op.dtn:.4g} "
              f"config_feat[0]={np.round(op.config_feat[0], 3)}")
    held = build_op("OP16", b, subsample_time=40)
    print(f"  HELD-OUT {held.op_id}: n_t={held.n_t} "
          f"config_feat[0]={np.round(held.config_feat[0], 3)}")
