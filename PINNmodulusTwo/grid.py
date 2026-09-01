"""The measurement points as a structured grid -- the prerequisite for a CNN.

The 363 points this project predicts are not a point cloud. They are three
x-planes (cell centre, JR1 centre, housing wall), each carrying the SAME regular
11 x 11 raster in (y, z). ``legacy/.../coordinates/*.csv`` holds all three tables
and they agree to the last digit on y and z; only x differs.

So the temperature field is literally a small image: ``(nx=3, ny=11, nz=11)``.
That is what makes a convolution applicable at all, and it is why this module
exists separately from ``cnn_model.py`` -- the claim "the points form a grid" is
a claim about the DATA, and it is checked against the data rather than assumed.
:func:`build_grid_spec` derives the layout from ``bundle.xn`` at runtime and
raises if the points do not actually form a full tensor product.

Spacings (normalised by ``L_ref``, i.e. the same units ``physics.py`` takes its
autograd derivatives in):

* ``y``: 11 levels, uniform, 19.81 mm apart
* ``z``: 11 levels, uniform, 10.44 mm apart
* ``x``:  3 levels, NON-uniform -- 10.79 mm then 11.11 mm

Three x-levels is the whole reason ``physics_grid.py`` fits one quadratic
through the x direction instead of differencing it: with three samples a
quadratic is the highest-order interpolant available, and its second derivative
is constant in x.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Two coordinates count as the same grid level when they are closer than this
# fraction of the axis extent. The exports agree to ~1e-9 within an axis and the
# levels are ~5 % of the extent apart, so anything in between separates them.
_LEVEL_TOL = 1e-4


def _levels(v: torch.Tensor, tol: float) -> torch.Tensor:
    """Distinct grid levels along one axis, ascending.

    Clusters values that differ by less than ``tol`` times the axis extent, so a
    float32 round-trip through the ``.npz`` cannot split one physical plane into
    two levels. The level is the cluster MEAN, not its first member: the 121
    points of a plane all carry the same coordinate in exact arithmetic and
    differ only by float32 rounding, so averaging them recovers about a digit
    that picking one of them throws away.
    """
    vs = torch.sort(v)[0]
    extent = float(vs[-1] - vs[0])
    eps = tol * (extent if extent > 0.0 else 1.0)
    groups: list[list[torch.Tensor]] = [[vs[0]]]
    for value in vs[1:]:
        if float(value - groups[-1][-1]) > eps:
            groups.append([value])
        else:
            groups[-1].append(value)
    return torch.stack([torch.stack(g).mean() for g in groups])


@dataclass(frozen=True)
class GridSpec:
    """Bidirectional map between the (P,) point order and an (nx, ny, nz) raster.

    ``to_raster`` and ``to_points`` are inverse permutations:

    * ``values[..., to_points].reshape(..., nx, ny, nz)`` lays a per-point vector
      out as an image (:meth:`as_grid`),
    * ``grid.reshape(..., nx * ny * nz)[..., to_raster]`` puts it back in the
      point order everything else in the project uses (:meth:`as_points`).

    Both are plain index reads, so they are differentiable and cost one gather.
    """

    nx: int
    ny: int
    nz: int
    x: torch.Tensor          # (nx,) normalised level positions
    y: torch.Tensor          # (ny,)
    z: torch.Tensor          # (nz,)
    to_raster: torch.Tensor  # (P,) long: raster slot of point p
    to_points: torch.Tensor  # (P,) long: point index sitting in raster slot r

    @property
    def n_points(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def to(self, device) -> "GridSpec":
        return GridSpec(
            self.nx, self.ny, self.nz,
            self.x.to(device), self.y.to(device), self.z.to(device),
            self.to_raster.to(device), self.to_points.to(device),
        )

    def to_dict(self) -> dict:
        """Plain-python form for a checkpoint.

        The permutation travels with the weights rather than being re-derived
        from a bundle at load time: a model trained on one point ordering and
        reloaded against another would produce a scrambled field and no error.
        """
        return {
            "nx": int(self.nx), "ny": int(self.ny), "nz": int(self.nz),
            "x": [float(v) for v in self.x],
            "y": [float(v) for v in self.y],
            "z": [float(v) for v in self.z],
            "to_raster": [int(v) for v in self.to_raster],
            "to_points": [int(v) for v in self.to_points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridSpec":
        return cls(
            nx=int(d["nx"]), ny=int(d["ny"]), nz=int(d["nz"]),
            x=torch.tensor(d["x"], dtype=torch.float64),
            y=torch.tensor(d["y"], dtype=torch.float64),
            z=torch.tensor(d["z"], dtype=torch.float64),
            to_raster=torch.tensor(d["to_raster"], dtype=torch.long),
            to_points=torch.tensor(d["to_points"], dtype=torch.long),
        )

    def as_grid(self, values: torch.Tensor) -> torch.Tensor:
        """(..., P) -> (..., nx, ny, nz). Differentiable; a single gather."""
        lead = values.shape[:-1]
        return values[..., self.to_points].reshape(*lead, self.nx, self.ny, self.nz)

    def as_points(self, grid: torch.Tensor) -> torch.Tensor:
        """(..., nx, ny, nz) -> (..., P). Inverse of :meth:`as_grid`."""
        lead = grid.shape[:-3]
        return grid.reshape(*lead, self.n_points)[..., self.to_raster]

    def spacing(self) -> tuple[torch.Tensor, float, float]:
        """``(dx_levels, dy, dz)`` in normalised units.

        ``dx`` stays a vector because the three x-planes are NOT equally spaced;
        y and z are checked for uniformity in :func:`build_grid_spec` and are
        returned as scalars.

        The uniform spacings are the MEAN step across the axis, not the first
        one. The exports are float32, so consecutive steps disagree in the last
        digit or two; a central difference is only exact on a truly uniform mesh,
        and taking the mean is the best estimate of the spacing the mesh actually
        has. The difference is small -- it moved the second derivative of an
        exactly-quadratic test field from ~1e-4 to ~1e-9 -- but it is free.
        """
        dx = self.x[1:] - self.x[:-1]
        dy = float(self.y[-1] - self.y[0]) / (self.ny - 1)
        dz = float(self.z[-1] - self.z[0]) / (self.nz - 1)
        return dx, dy, dz


def _regularise(lev: torch.Tensor, name: str, tol: float) -> torch.Tensor:
    """Snap uniformly-spaced levels onto the exact arithmetic progression.

    The y and z rasters are uniform by construction -- StarCCM+ laid them out
    that way -- but the exports are float32, so consecutive steps disagree in the
    last digit or two. A central difference is only exact on a truly uniform
    mesh, and the error it makes is the mis-spacing amplified by ``1/h^2``: on
    this grid that turns ~2e-7 of coordinate rounding into ~3e-5 of relative
    error in every second derivative. Least-squares fitting ``lev_i = a + b*i``
    removes it (measured: 8e-5 -> 1e-13 on an exactly quadratic test field).

    Raises if the levels are NOT an arithmetic progression to within ``tol``,
    which is the check this replaced: a genuinely non-uniform axis must not be
    silently straightened.
    """
    n = lev.shape[0]
    i = torch.arange(n, dtype=lev.dtype, device=lev.device)
    # Closed-form least squares for a straight line through (i, lev_i).
    i_mu, l_mu = i.mean(), lev.mean()
    b = ((i - i_mu) * (lev - l_mu)).sum() / ((i - i_mu) ** 2).sum()
    fitted = (l_mu - b * i_mu) + b * i
    if float((fitted - lev).abs().max()) > tol * float(b.abs()):
        raise NotAGridError(
            f"the {name} levels are not uniformly spaced; the finite-difference "
            f"stencils in physics_grid.py assume they are"
        )
    return fitted


class NotAGridError(ValueError):
    """The points do not form a full (nx, ny, nz) tensor product."""


def build_grid_spec(xn: torch.Tensor, *, tol: float = _LEVEL_TOL,
                    require_uniform_yz: bool = True) -> GridSpec:
    """Derive the raster layout from the normalised coordinates.

    ``xn`` is ``(P, 3)`` exactly as ``data.py`` builds it: ``(xyz - xyz_min) /
    L_ref``. Nothing about the ordering of the rows is assumed -- the permutation
    is read off the coordinates -- so a bundle whose points arrive in a different
    order still maps correctly, and one whose points are NOT a grid fails here
    with the reason rather than silently training on a scrambled image.
    """
    if xn.dim() != 2 or xn.shape[1] != 3:
        raise NotAGridError(f"xn must be (P, 3), got {tuple(xn.shape)}")
    xn = xn.detach().to(torch.float64)

    x, y, z = (_levels(xn[:, i], tol) for i in range(3))
    nx, ny, nz = len(x), len(y), len(z)
    n_points = xn.shape[0]
    if nx * ny * nz != n_points:
        raise NotAGridError(
            f"{n_points} points do not fill a {nx}x{ny}x{nz} raster "
            f"({nx * ny * nz} slots). The points are not a tensor-product grid, "
            "so a convolution has no lattice to run on."
        )

    def _index(col: torch.Tensor, lev: torch.Tensor) -> torch.Tensor:
        # Nearest level per coordinate; the distance is checked below, so a point
        # that sits between two levels is caught rather than snapped to one.
        d = (col[:, None] - lev[None, :]).abs()
        near, idx = d.min(dim=1)
        extent = float(lev[-1] - lev[0])
        if float(near.max()) > tol * (extent if extent > 0.0 else 1.0):
            raise NotAGridError("a coordinate does not lie on any grid level")
        return idx

    ix, iy, iz = (_index(xn[:, i], lev) for i, lev in enumerate((x, y, z)))
    to_raster = (ix * ny + iy) * nz + iz          # (P,) raster slot of point p

    if len(torch.unique(to_raster)) != n_points:
        raise NotAGridError(
            "two points map to the same raster slot: the coordinates are "
            "degenerate, not a grid"
        )
    to_points = torch.empty_like(to_raster)
    to_points[to_raster] = torch.arange(n_points, device=xn.device)

    if require_uniform_yz:
        y = _regularise(y, "y", tol)
        z = _regularise(z, "z", tol)

    # The levels stay float64. They are geometry, not data: the stencil
    # coefficients are built from them once and a float32 level position shows up
    # directly as a relative error in every second derivative.
    return GridSpec(
        nx=nx, ny=ny, nz=nz, x=x, y=y, z=z,
        to_raster=to_raster, to_points=to_points,
    )


def describe(spec: GridSpec, L_ref: float | None = None) -> list[str]:
    """Human-readable summary, for the training log."""
    dx, dy, dz = spec.spacing()
    scale = 1.0 if L_ref is None else L_ref
    unit = "" if L_ref is None else " m"
    return [
        f"grid {spec.nx} x {spec.ny} x {spec.nz} = {spec.n_points} points "
        f"(x-planes x y x z)",
        f"  dx (non-uniform): "
        + ", ".join(f"{float(v) * scale:.6g}" for v in dx) + unit,
        f"  dy = {dy * scale:.6g}{unit}   dz = {dz * scale:.6g}{unit}",
    ]


if __name__ == "__main__":  # pragma: no cover - a hand check, needs the data
    import numpy as np
    from data import load_ops

    bundle = load_ops(["OP01"])
    spec = build_grid_spec(torch.from_numpy(np.asarray(bundle.xn)))
    for line in describe(spec, bundle.L_ref):
        print(line)
