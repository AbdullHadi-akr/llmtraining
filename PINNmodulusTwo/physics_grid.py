"""The same heat residual as ``physics.py``, differenced on the grid.

Why this file has to exist
--------------------------
``physics.py`` gets its spatial derivatives from ``torch.autograd`` with respect
to the coordinate input ``xn``. That works because the MLP is a continuous
function OF ``xn``. A convolution is not: it reads a fixed lattice. It does take
``xn`` as three input channels -- deliberately, so it sees what the MLP sees --
so autograd on it returns a finite number rather than an error, and that is the
danger. The number answers "how does the prediction change if this pixel is
relabelled with different coordinates, its neighbours' temperatures unchanged?",
which is not the spatial variation of the field: the conduction lives in the
kernel and this derivative never reaches it. Handing ``ConvRecurrentField`` to
``physics.heat_residual`` would not fail, and ``L_phys`` would fall like any
other run. So ``physics.py`` refuses a model carrying a ``grid`` attribute, and
the grid model gets grid stencils here.

The equation, the nondimensionalisation, the ``residual_norm`` handling and the
ONE-divisor rule are unchanged from ``physics.py``; only how the derivatives are
obtained differs. Coordinates are ``xn = (xyz - xyz_min) / L_ref``, exactly the
units ``Fo`` is built in, so the stencils are taken in ``xn`` and no extra factor
appears.

What the discretisation costs, stated plainly
---------------------------------------------
* **y and z**: 11 uniform levels. Second-order central differences, evaluated on
  the 9 interior levels of each. The 2 outer rings carry no residual -- there is
  no boundary condition on the cell's outer faces to close them with, and a
  one-sided stencil there would be inventing one.
* **x**: only THREE planes, and unevenly spaced. Three samples determine one
  quadratic and no more, so ``d^2T/dx^2`` is the second derivative of that
  quadratic -- a constant in x. This is the real accuracy limit of the grid
  formulation, and it is a limit of the DATA, not of the method: no scheme
  recovers a third x-mode from three planes. The MLP's autograd Laplacian looks
  more accurate in x, but only because it is free to invent curvature between
  planes that nothing measured.
* The residual therefore covers ``nx * (ny-2) * (nz-2) = 3 * 9 * 9 = 243`` of the
  363 points at every sampled time -- against ``physics.py``, which can sample
  any point at all. In exchange every sampled time evaluates 243 residual points
  from ONE forward pass instead of one point per pass.

The ``dT/dx = 0`` boundary at the cell centre is kept as its own term, exactly as
in ``physics.py``: it is a statement about the solution, not part of the interior
stencil, and folding it into a ghost node would contradict the quadratic fit.
"""

from __future__ import annotations

import torch

from physics import ResidualNorm, TimeDerivMethod, _term_norm


def _poly_diff_matrices(nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First and second derivative matrices of the interpolant through ``nodes``.

    ``D1 @ T`` and ``D2 @ T`` give ``dT/dx`` and ``d^2T/dx^2`` at every node, for
    the unique degree-``n-1`` polynomial through the ``n`` samples. With ``n = 3``
    that is the quadratic the three x-planes determine; ``D2`` then has identical
    rows, which is the statement "three planes cannot resolve a varying second
    derivative" written as a matrix.

    Nodes are shifted and scaled to ``[-1, 1]`` before the Vandermonde solve and
    the derivatives are scaled back, so conditioning does not depend on where the
    plane happens to sit in absolute coordinates.
    """
    n = nodes.shape[0]
    if n < 3:
        raise ValueError(
            f"a second derivative needs at least 3 levels along the axis, got {n}"
        )
    if n > 4:
        raise ValueError(
            f"{n} x-planes: the global polynomial fit is only intended for the "
            "3 planes this dataset has. Add a local stencil before using more."
        )
    x = nodes.to(torch.float64)
    span = float(x[-1] - x[0])
    scale = span / 2.0
    u = (x - (x[0] + x[-1]) / 2.0) / scale                # nodes in [-1, 1]

    k = torch.arange(n, dtype=torch.float64, device=x.device)
    V = u[:, None] ** k[None, :]                          # p(u_i) = sum_k c_k u^k
    zero = torch.zeros_like(V)
    V1 = torch.where(
        k[None, :] >= 1,
        k[None, :] * u[:, None] ** (k[None, :] - 1).clamp(min=0),
        zero,
    )
    V2 = torch.where(
        k[None, :] >= 2,
        k[None, :] * (k[None, :] - 1) * u[:, None] ** (k[None, :] - 2).clamp(min=0),
        zero,
    )
    Vinv = torch.linalg.inv(V)
    D1 = (V1 @ Vinv) / scale
    D2 = (V2 @ Vinv) / (scale ** 2)
    return D1.to(nodes.dtype), D2.to(nodes.dtype)


class GridStencils:
    """Cached derivative operators for one :class:`grid.GridSpec`.

    Built once per run: the matrices depend only on the coordinates, never on the
    weights or the time step.
    """

    def __init__(self, spec, dtype=torch.float32, device=None) -> None:
        self.spec = spec
        dx, dy, dz = spec.spacing()
        self.hy = float(dy)
        self.hz = float(dz)
        D1x, D2x = _poly_diff_matrices(spec.x.to(torch.float64))
        self.D1x = D1x.to(dtype=dtype, device=device)
        self.D2x = D2x.to(dtype=dtype, device=device)

    def to(self, device=None, dtype=None) -> "GridStencils":
        if device is not None or dtype is not None:
            self.D1x = self.D1x.to(device=device, dtype=dtype)
            self.D2x = self.D2x.to(device=device, dtype=dtype)
        return self

    @property
    def n_residual_points(self) -> int:
        """Points per time step where the interior residual is defined."""
        s = self.spec
        return s.nx * (s.ny - 2) * (s.nz - 2)

    @property
    def n_bc_points(self) -> int:
        """Points on the ``x = 0`` plane where the Neumann term is evaluated."""
        return self.spec.ny * self.spec.nz

    # -- axis operators -----------------------------------------------------
    def _dx(self, T: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Apply an x-direction matrix to ``(B, nx, ny, nz)``, keeping the shape."""
        return torch.einsum("ij,bjyz->biyz", mat, T)

    def dT_dx(self, T: torch.Tensor) -> torch.Tensor:
        return self._dx(T, self.D1x)

    def d2T_dx2(self, T: torch.Tensor) -> torch.Tensor:
        return self._dx(T, self.D2x)

    @staticmethod
    def _central(T: torch.Tensor, dim: int, h: float) -> torch.Tensor:
        """Second-order central first difference; the axis loses its 2 edges."""
        fwd = T.narrow(dim, 2, T.shape[dim] - 2)
        bwd = T.narrow(dim, 0, T.shape[dim] - 2)
        return (fwd - bwd) / (2.0 * h)

    @staticmethod
    def _second(T: torch.Tensor, dim: int, h: float) -> torch.Tensor:
        """Second-order central second difference; the axis loses its 2 edges."""
        fwd = T.narrow(dim, 2, T.shape[dim] - 2)
        mid = T.narrow(dim, 1, T.shape[dim] - 2)
        bwd = T.narrow(dim, 0, T.shape[dim] - 2)
        return (fwd - 2.0 * mid + bwd) / (h * h)

    def interior(self, T: torch.Tensor) -> torch.Tensor:
        """Crop ``(B, nx, ny, nz)`` to the region the residual is defined on."""
        return T[:, :, 1:-1, 1:-1]

    def hessian(self, T: torch.Tensor) -> dict[str, torch.Tensor]:
        """All six independent second derivatives on the interior (B, nx, ny-2, nz-2).

        ``Txx`` and the mixed x-terms come from the quadratic through the three
        x-planes; ``Tyy``/``Tzz``/``Tyz`` from central differences.
        """
        Tx = self.dT_dx(T)
        Txx = self.d2T_dx2(T)
        Ty = self._central(T, 2, self.hy)                    # (B,nx,ny-2,nz)

        return {
            "Txx": Txx[:, :, 1:-1, 1:-1],
            "Tyy": self._second(T, 2, self.hy)[:, :, :, 1:-1],
            "Tzz": self._second(T, 3, self.hz)[:, :, 1:-1, :],
            "Txy": self._central(Tx, 2, self.hy)[:, :, :, 1:-1],
            "Txz": self._central(Tx, 3, self.hz)[:, :, 1:-1, :],
            "Tyz": self._central(Ty, 3, self.hz),
        }


def _history_all_points(model, Tn_seq, dtn, tn_q, n_points):
    """Query times and point indices covering every point at every sampled time.

    Returns ``(tq_rep, p_rep)`` of length ``B * P`` in row-major (time-major)
    order, so a result reshaped to ``(B, P)`` lines up with ``tn_q``.
    """
    B = tn_q.shape[0]
    tq_rep = tn_q.repeat_interleave(n_points)
    p_rep = torch.arange(n_points, device=tn_q.device).repeat(B)
    return tq_rep, p_rep


def field_and_history(model, xn, static, cfg, forcing, Tn_seq, dtn, tn_q):
    """Predicted field at ``tn_q``, plus the history block it was built from.

    Returns ``(T (B, P), hist (B, P, k), tq_rep, p_rep)`` -- the last two so the
    caller can fetch further lags at the same (time, point) pairs without
    rebuilding the index arithmetic.
    """
    P = model.n_points
    tq_rep, p_rep = _history_all_points(model, Tn_seq, dtn, tn_q, P)
    hist = model._history(Tn_seq, dtn, tq_rep, p_rep).reshape(tn_q.shape[0], P, -1)
    level = model.level(Tn_seq, dtn, tn_q)
    T = model.field_batch(xn, static, cfg, forcing, hist, level)
    return T, hist, tq_rep, p_rep


def heat_residual_grid(
    model,
    stencils: GridStencils,
    xn: torch.Tensor,        # (P, 3)
    static: torch.Tensor,    # (P, n_static)
    cfg: torch.Tensor,       # (B, n_config)
    forcing: torch.Tensor,   # (B, n_forcing)
    Fo: torch.Tensor,        # (P, 3, 3)
    Qsrc: torch.Tensor,      # (B, P) source at the sampled times
    Tn_seq: torch.Tensor,    # (n_t, P) the frozen rollout
    dtn: float,
    tn_q: torch.Tensor,      # (B,) sampled times, normalised
    phys_scale: float,
    time_deriv: TimeDerivMethod = "bdf2",
    residual_norm: ResidualNorm = "rms",
) -> torch.Tensor:
    """Anisotropic heat residual on the interior lattice. Returns (B, nx, ny-2, nz-2).

    Mirrors ``physics.heat_residual`` term for term:
    ``residual = dT/dt - diff_gain * (Fo : grad^2 T) - src_gain * Qsrc``, divided
    by ONE scale at the end.
    """
    if time_deriv == "autograd":
        raise NotImplementedError(
            "the convolutional model has no continuous-time input; "
            "use --time-deriv bdf1 or bdf2 with --arch cnn"
        )

    spec = model.grid
    B, P = tn_q.shape[0], model.n_points
    T, hist, tq_rep, p_rep = field_and_history(
        model, xn, static, cfg, forcing, Tn_seq, dtn, tn_q
    )

    # ---- time derivative: identical stencil to physics.py --------------------
    raw_hist = model.history_mode != "hybrid"

    def _lag(n: int) -> torch.Tensor:
        if raw_hist and model.k_max >= n:
            return hist[:, :, n - 1]
        return model.history_at(Tn_seq, dtn, tq_rep, p_rep, lag=n).reshape(B, P)

    if time_deriv == "bdf2":
        dTdt = (3.0 * T - 4.0 * _lag(1) + _lag(2)) / (2.0 * model.delta + 1e-8)
    else:
        dTdt = (T - _lag(1)) / (model.delta + 1e-8)

    # ---- space: finite differences on the lattice ---------------------------
    Tg = model._as_grid(T)                                  # (B, nx, ny, nz)
    d2 = stencils.hessian(Tg)

    fo = model._as_grid(Fo.permute(1, 2, 0).reshape(9, P))  # (9, nx, ny, nz)
    fo = fo[:, :, 1:-1, 1:-1].unsqueeze(1)                  # (9, 1, nx, ny-2, nz-2)
    aniso = (
        fo[0] * d2["Txx"] + fo[4] * d2["Tyy"] + fo[8] * d2["Tzz"]
        + 2.0 * (fo[1] * d2["Txy"] + fo[2] * d2["Txz"] + fo[5] * d2["Tyz"])
    )

    dTdt_i = stencils.interior(model._as_grid(dTdt))
    Qsrc_i = stencils.interior(model._as_grid(Qsrc))

    residual = dTdt_i - model.diff_gain * aniso - model.src_gain * Qsrc_i
    if residual_norm == "legacy":
        return residual / (phys_scale ** 0.5 + 1e-30)
    return residual / (phys_scale + 1e-30)


def boundary_condition_loss_grid(
    model,
    stencils: GridStencils,
    xn: torch.Tensor,
    static: torch.Tensor,
    cfg: torch.Tensor,       # (B, n_config)
    forcing: torch.Tensor,   # (B, n_forcing)
    Tn_seq: torch.Tensor,
    dtn: float,
    tn_q: torch.Tensor,      # (B,)
    bc_scale: float = 1.0,
    residual_norm: ResidualNorm = "rms",
) -> torch.Tensor:
    """``dT/dx = 0`` at the cell centre, on every point of that plane. (B, ny, nz).

    ``x = 0`` is the FIRST x level (``xn`` is shifted by ``xyz_min``), so the
    derivative comes from the quadratic's one-sided 3-point row -- second-order
    accurate, and the same interpolant the interior term uses. Unlike
    ``physics.boundary_condition_loss`` this evaluates every one of the 121 plane
    points per sampled time rather than a random subset, because the field they
    live on was computed in one pass anyway.
    """
    T, _, _, _ = field_and_history(
        model, xn, static, cfg, forcing, Tn_seq, dtn, tn_q
    )
    dT_dx = stencils.dT_dx(model._as_grid(T))[:, 0]          # (B, ny, nz) at x=0
    return dT_dx / _term_norm(bc_scale, residual_norm)
