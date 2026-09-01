"""Guards for the convolutional architecture: the grid, the stencils, the net.

Runs in seconds, needs neither Modulus (conftest stubs it), the data cache, nor
a GPU. Four claims are load-bearing and each has a test here:

1. The 363 measurement points really are a 3 x 11 x 11 raster, and the
   permutation between point order and raster order round-trips.
2. The finite-difference stencils reproduce every second derivative of a
   quadratic exactly -- including the three-plane x direction, where "exactly"
   is the most that three samples allow.
3. ``field`` (what ``rollout`` calls, one time step) and ``field_batch`` (what
   training and ``physics_grid`` call) are the same function.
4. Autograd on ``xn`` does NOT give the spatial derivative of a conv field -- it
   gives a finite number that answers a different question. That is the reason
   ``physics_grid.py`` exists, and because it is silent rather than an error,
   ``physics.py`` refuses a grid model outright and the refusal is tested.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

import physics
from cnn_model import ConvRecurrentField
from grid import GridSpec, NotAGridError, build_grid_spec
from model import rollout
from physics_grid import GridStencils, boundary_condition_loss_grid, heat_residual_grid

COORD_DIR = (Path(__file__).resolve().parents[2]
             / "legacy" / "battery_surrogate_agenticWorkflow" / "coordinates")
COORD_FILES = (
    "Coordinates - Grid Cell Center.csv",
    "Coordinates - Grid JR1 Center.csv",
    "Coordinates - Grid Gehäusewand.csv",
)


def _synthetic_xn(nx=3, ny=11, nz=11, shuffle=True, seed=0):
    """A tensor-product grid in a deliberately scrambled point order."""
    x = torch.tensor([0.0, 0.41, 0.83], dtype=torch.float64)[:nx]
    y = torch.linspace(0.0, 2.578, ny, dtype=torch.float64)
    z = torch.linspace(0.0, 1.359, nz, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    xn = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        xn = xn[torch.randperm(xn.shape[0], generator=g)]
    return xn


# ---------------------------------------------------------------------------
# 1. the grid
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not COORD_DIR.exists(), reason="coordinate CSVs not checked out")
def test_the_real_measurement_points_form_a_3x11x11_raster():
    """The claim the whole CNN rests on, checked against the shipped coordinates.

    If this ever fails, a convolution has no lattice to run on and ``--arch cnn``
    is not applicable to the new geometry -- which is exactly what
    ``build_grid_spec`` is supposed to say out loud rather than train through.
    """
    rows = []
    for name in COORD_FILES:
        with open(COORD_DIR / name, encoding="utf-8") as fh:
            rows += [[float(c) for c in r] for r in list(csv.reader(fh))[1:]]
    xyz = torch.tensor(rows, dtype=torch.float64)
    assert xyz.shape == (363, 3)

    # data.py's transform, verbatim.
    xyz_min = xyz.min(0).values
    L_ref = float(((xyz.max(0).values - xyz_min).clamp(min=1e-12)).prod() ** (1 / 3))
    spec = build_grid_spec((xyz - xyz_min) / L_ref)

    assert spec.shape == (3, 11, 11)
    dx, dy, dz = spec.spacing()
    # y and z uniform, x is not -- the two x gaps differ by ~3 %.
    assert dy > 0 and dz > 0
    assert float((dx[1] - dx[0]).abs()) > 0.01 * float(dx[0])


def test_permutation_round_trips_and_places_coordinates_on_their_levels():
    xn = _synthetic_xn()
    spec = build_grid_spec(xn)
    assert spec.shape == (3, 11, 11)

    v = torch.arange(spec.n_points, dtype=torch.float64)
    assert torch.equal(spec.as_points(spec.as_grid(v)), v)

    # Each coordinate column, laid out as an image, must vary along its own axis
    # only -- the check that the permutation is the right one and not merely a
    # permutation.
    g = spec.as_grid(xn.T)
    assert torch.allclose(g[0], spec.x[:, None, None].expand_as(g[0]))
    assert torch.allclose(g[1], spec.y[None, :, None].expand_as(g[1]))
    assert torch.allclose(g[2], spec.z[None, None, :].expand_as(g[2]))


def test_a_point_cloud_is_rejected_rather_than_reshaped():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(NotAGridError):
        build_grid_spec(torch.rand(363, 3, generator=g, dtype=torch.float64))


def test_a_grid_with_a_hole_is_rejected():
    xn = _synthetic_xn(shuffle=False)[:-1]          # 362 points, one slot empty
    with pytest.raises(NotAGridError):
        build_grid_spec(xn)


def test_grid_spec_survives_a_checkpoint_round_trip():
    spec = build_grid_spec(_synthetic_xn())
    back = GridSpec.from_dict(spec.to_dict())
    assert back.shape == spec.shape
    assert torch.equal(back.to_points, spec.to_points)
    assert torch.equal(back.to_raster, spec.to_raster)


# ---------------------------------------------------------------------------
# 2. the stencils
# ---------------------------------------------------------------------------
def _analytic(spec):
    X = spec.x[:, None, None].expand(*spec.shape)
    Y = spec.y[None, :, None].expand(*spec.shape)
    Z = spec.z[None, None, :].expand(*spec.shape)
    return X, Y, Z


def test_every_second_derivative_of_a_quadratic_is_exact():
    """All six, including the mixed ones and the three-plane x direction.

    A quadratic is the highest order the x direction can represent at all (three
    samples), so exactness on quadratics is the strongest statement available --
    and it is the one that fails first if a stencil coefficient, a spacing or the
    raster permutation is wrong.
    """
    spec = build_grid_spec(_synthetic_xn())
    st = GridStencils(spec, dtype=torch.float64)
    X, Y, Z = _analytic(spec)
    T = (0.5 * X ** 2 + 1.5 * Y ** 2 - 0.75 * Z ** 2
         + 2.0 * X * Y + 3.0 * X * Z - 1.25 * Y * Z + 7.0)

    d2 = st.hessian(T[None])
    ones = torch.ones_like(d2["Txx"])
    for key, exact in (("Txx", 1.0), ("Tyy", 3.0), ("Tzz", -1.5),
                       ("Txy", 2.0), ("Txz", 3.0), ("Tyz", -1.25)):
        assert torch.allclose(d2[key], exact * ones, atol=1e-9), key


def test_the_x_second_derivative_is_constant_across_the_three_planes():
    """Three samples determine one quadratic, whose curvature does not vary in x.

    Not a defect to be fixed later: no scheme recovers a third x-mode from three
    planes. The test pins the property so nobody reads a varying ``Txx`` off this
    model and believes it.
    """
    spec = build_grid_spec(_synthetic_xn())
    st = GridStencils(spec, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    T = torch.rand(1, *spec.shape, generator=g, dtype=torch.float64)
    Txx = st.d2T_dx2(T)
    assert torch.allclose(Txx[:, 0], Txx[:, 1], atol=1e-9)
    assert torch.allclose(Txx[:, 1], Txx[:, 2], atol=1e-9)


def test_the_neumann_derivative_at_x0_is_exact_on_a_quadratic():
    spec = build_grid_spec(_synthetic_xn())
    st = GridStencils(spec, dtype=torch.float64)
    X, _, _ = _analytic(spec)
    # (x - x0)^2 has dT/dx = 0 at x0 exactly; a linear term makes it exactly 4.
    flat = st.dT_dx(((X - spec.x[0]) ** 2 * 3.0 + 5.0)[None])[:, 0]
    assert torch.allclose(flat, torch.zeros_like(flat), atol=1e-9)
    sloped = st.dT_dx((4.0 * X + 1.0)[None])[:, 0]
    assert torch.allclose(sloped, torch.full_like(sloped, 4.0), atol=1e-9)


def test_the_residual_covers_the_interior_and_says_how_many_points():
    spec = build_grid_spec(_synthetic_xn())
    st = GridStencils(spec)
    assert st.n_residual_points == 3 * 9 * 9
    assert st.n_bc_points == 11 * 11


# ---------------------------------------------------------------------------
# 3. the model
# ---------------------------------------------------------------------------
def _model(spec, **kw):
    torch.manual_seed(0)
    kw.setdefault("n_config", 4)
    kw.setdefault("n_static", 2)
    kw.setdefault("n_forcing", 3)
    kw.setdefault("k_max", 2)
    kw.setdefault("history_mode", "raw")
    kw.setdefault("rate_lags", ())
    kw.setdefault("width", 16)
    kw.setdefault("num_layers", 3)
    kw.setdefault("dtn", 1e-3)
    kw.setdefault("t_span_ref", 1000.0)
    return ConvRecurrentField(grid=spec, **kw)


def test_field_and_field_batch_are_the_same_function():
    """``rollout`` uses one, training and physics_grid the other."""
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    P = spec.n_points
    g = torch.Generator().manual_seed(1)
    xn = torch.rand(P, 3, generator=g)
    static = torch.rand(P, m.n_static, generator=g)
    cfg = torch.rand(1, m.n_config, generator=g)
    forcing = torch.rand(1, m.n_forcing, generator=g)
    hist = torch.rand(P, m.k_max, generator=g)

    one = m.field(xn, static, cfg.expand(P, -1), forcing.expand(P, -1), hist)
    many = m.field_batch(xn, static, cfg, forcing, hist.unsqueeze(0))[0]
    assert torch.equal(one, many)


def test_field_refuses_a_subset_of_points():
    """A convolution on scattered pixels is meaningless, so it must not be silent."""
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    P = spec.n_points
    with pytest.raises(ValueError, match="whole field"):
        m.field_batch(torch.zeros(P - 1, 3), torch.zeros(P - 1, m.n_static),
                      torch.zeros(1, m.n_config), torch.zeros(1, m.n_forcing),
                      torch.zeros(1, P - 1, m.k_max))


def test_the_convolution_is_translation_equivariant_in_y():
    """The inductive bias the whole approach is chosen for, stated as a test.

    Shifting the temperature history one cell in y shifts the prediction one cell
    in y. Only away from the padded edge, and only with the geometry channels
    held constant -- which is exactly the sense in which "the same local pattern
    means the same thing anywhere" is true of a convolution and false of the
    coordinate MLP.
    """
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec, n_static=0, n_config=0, n_forcing=0)
    P = spec.n_points
    xn = torch.zeros(P, 3)          # constant geometry, so only the shift acts
    static = torch.zeros(P, 0)
    cfg = torch.zeros(1, 0)
    forcing = torch.zeros(1, 0)

    g = torch.Generator().manual_seed(2)
    hist_img = torch.rand(1, m.k_max, *spec.shape, generator=g)
    shifted = torch.roll(hist_img, shifts=1, dims=3)

    def predict(img):
        pts = spec.as_points(img).permute(0, 2, 1)      # (1, P, k)
        return spec.as_grid(m.field_batch(xn, static, cfg, forcing, pts))

    base = predict(hist_img)
    moved = predict(shifted)
    # Only where NEITHER output can see an edge: the roll wraps at y=0 and the
    # padding replicates at both ends, and each corrupts everything within the
    # stack's reach of them.
    r = m.net.receptive_field
    ny = spec.ny
    assert torch.allclose(moved[:, :, r + 1:ny - r], base[:, :, r:ny - r - 1],
                          atol=1e-6)


def test_rollout_keeps_the_initial_condition_and_stays_finite():
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    P, n_t = spec.n_points, 24
    g = torch.Generator().manual_seed(4)
    xn = torch.rand(P, 3, generator=g)
    static = torch.rand(P, m.n_static, generator=g)
    cfg = torch.rand(n_t, m.n_config, generator=g)
    forcing = torch.rand(n_t, m.n_forcing, generator=g)
    ic = torch.rand(P, generator=g)
    tn = torch.arange(n_t, dtype=torch.float32) * 1e-3

    buf = rollout(m, xn, static, cfg, forcing, ic, tn, 1e-3, clamp=50.0)
    assert buf.shape == (n_t, P)
    assert torch.equal(buf[0], ic)        # imposed, never predicted
    assert torch.isfinite(buf).all()


def test_checkpoint_config_round_trips_to_the_same_predictions():
    """``ConvRecurrentField(**model_config)`` + ``load_state_dict``, as train.py claims."""
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    cfg_kwargs = dict(
        grid=spec.to_dict(), n_config=m.n_config, n_static=m.n_static,
        n_forcing=m.n_forcing, k_max=m.k_max, history_mode=m.history_mode,
        rate_lags=[], width=m.width, num_layers=m.num_layers,
        kernel_size=m.kernel_size, padding_mode=m.padding_mode,
        dtn=float(m._dtn), t_span_ref=1000.0,
    )
    twin = ConvRecurrentField(**cfg_kwargs)
    twin.load_state_dict(m.state_dict())

    P = spec.n_points
    g = torch.Generator().manual_seed(5)
    args = (torch.rand(P, 3, generator=g), torch.rand(P, m.n_static, generator=g),
            torch.rand(1, m.n_config, generator=g),
            torch.rand(1, m.n_forcing, generator=g),
            torch.rand(1, P, m.k_max, generator=g))
    assert torch.equal(m.field_batch(*args), twin.field_batch(*args))


# ---------------------------------------------------------------------------
# 4. why physics_grid.py exists
# ---------------------------------------------------------------------------
def test_autograd_on_xn_returns_a_number_and_the_number_is_not_a_derivative():
    """The trap ``physics_grid.py`` is the answer to -- and why it is dangerous.

    ``xn`` reaches the conv as three input CHANNELS (so it sees what the MLP
    sees), which means autograd through it does NOT return zero. It returns a
    finite, plausible-looking gradient that answers a different question: how the
    prediction responds to relabelling a pixel's coordinates while its
    neighbours' temperatures stay put. Nothing about conduction is in it.

    A silent wrong number is worse than a loud zero, so the test pins both
    halves: the gradient exists, and the guard refuses it anyway.
    """
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    P = spec.n_points
    g = torch.Generator().manual_seed(6)
    xn = torch.rand(P, 3, generator=g, requires_grad=True)
    out = m.field_batch(xn, torch.rand(P, m.n_static, generator=g),
                        torch.rand(1, m.n_config, generator=g),
                        torch.rand(1, m.n_forcing, generator=g),
                        torch.rand(1, P, m.k_max, generator=g))
    grad = torch.autograd.grad(out.sum(), xn, allow_unused=True)[0]
    assert grad is not None and float(grad.abs().max()) > 0.0


def test_physics_py_refuses_a_grid_model_instead_of_differentiating_it():
    """The guard itself: both autograd terms, not just the one."""
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    P, n_t = spec.n_points, 8
    Tn_seq = torch.zeros(n_t, P)
    common = (torch.zeros(P, 3), torch.zeros(P, m.n_static),
              torch.zeros(2, m.n_config), torch.zeros(2, m.n_forcing))

    with pytest.raises(NotImplementedError, match="physics_grid"):
        physics.heat_residual(
            m, *common, torch.zeros(P, 3, 3), torch.zeros(2), Tn_seq, 1e-3,
            torch.zeros(2), torch.zeros(2, dtype=torch.long), phys_scale=1.0,
        )
    with pytest.raises(NotImplementedError, match="physics_grid"):
        physics.boundary_condition_loss(
            m, *common, Tn_seq, 1e-3, torch.zeros(2),
            torch.ones(P, dtype=torch.bool),
        )


def test_grid_residual_vanishes_on_a_constant_field_with_no_source():
    """Constant in space and time, zero source -> every term is zero.

    The same triviality ``train.py``'s ``spread`` diagnostic exists to catch, and
    the cheapest end-to-end check that the three terms are assembled with the
    right signs and shapes.
    """
    spec = build_grid_spec(_synthetic_xn())
    st = GridStencils(spec)
    m = _model(spec)

    class Flat(type(m)):                      # a field that ignores its inputs
        def field_batch(self, xn, static, cfg, forcing, hist, level=None):
            return torch.full((hist.shape[0], self.n_points), 0.5)

    m.__class__ = Flat
    P, n_t = spec.n_points, 12
    Tn_seq = torch.full((n_t, P), 0.5)
    tn = torch.arange(n_t, dtype=torch.float32) * 1e-3
    res = heat_residual_grid(
        m, st, torch.zeros(P, 3), torch.zeros(P, m.n_static),
        torch.zeros(4, m.n_config), torch.zeros(4, m.n_forcing),
        torch.zeros(P, 3, 3), torch.zeros(4, P), Tn_seq, 1e-3, tn[5:9],
        phys_scale=1.0, time_deriv="bdf2",
    )
    assert res.shape == (4, 3, 9, 9)
    assert float(res.abs().max()) < 1e-5

    bc = boundary_condition_loss_grid(
        m, st, torch.zeros(P, 3), torch.zeros(P, m.n_static),
        torch.zeros(4, m.n_config), torch.zeros(4, m.n_forcing),
        Tn_seq, 1e-3, tn[5:9], bc_scale=1.0,
    )
    assert bc.shape == (4, 11, 11)
    assert float(bc.abs().max()) < 1e-5


def test_autograd_time_is_refused_rather_than_silently_wrong():
    spec = build_grid_spec(_synthetic_xn())
    m = _model(spec)
    with pytest.raises(NotImplementedError):
        heat_residual_grid(
            m, GridStencils(spec), torch.zeros(spec.n_points, 3),
            torch.zeros(spec.n_points, m.n_static), torch.zeros(1, m.n_config),
            torch.zeros(1, m.n_forcing), torch.zeros(spec.n_points, 3, 3),
            torch.zeros(1, spec.n_points), torch.zeros(4, spec.n_points), 1e-3,
            torch.zeros(1), phys_scale=1.0, time_deriv="autograd",
        )
