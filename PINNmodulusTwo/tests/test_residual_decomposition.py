"""The three claims ``tools/residual_decomposition.py`` rests on, held at BEHAVIOUR.

FAHRPLAN 11.10 (23.09.) moves the plan off O18 on an argument, and an argument
that is only written down can quietly stop being true. These pin it:

1. ``return_parts`` is a window, not a second implementation: the default call is
   unchanged, and the parts add up to the residual.
2. A spatial term does not depend on delta at all, while the jump between the
   live network and the frozen buffer scales EXACTLY like 1/delta^2. That is the
   whole case against O18 as the cause of Achse 1's ``L_phys ~ 1/delta^2``.
3. A field that carries its spatial structure through the history anchor is flat
   to autograd (``[BLIND]``); the same field written explicitly in ``xn`` is not.

Plus the two things that went wrong while building the tool: float32 line keys
that merged grid lines (``nan``), and ``--time-deriv autograd`` training a
network the rollout never uses (O20).

No data cache, no Modulus (conftest.py stubs it), seconds on a CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(PKG_DIR / "tools"))

from model import RecurrentField, rollout  # noqa: E402
from physics import heat_residual  # noqa: E402
import residual_decomposition as rd  # noqa: E402

NY, NZ = 5, 5
N_T = 60
DTN = 0.01


def _grid() -> torch.Tensor:
    """3 x-planes (non-equidistant, like the real 0 / 10.8 / 21.9 mm) x 5 x 5."""
    xs = torch.tensor([0.0, 0.049, 0.100])
    ys = torch.linspace(0.0, 1.0, NY)
    zs = torch.linspace(0.0, 0.5, NZ)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)


def _op(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    xn = _grid()
    P = xn.shape[0]
    Fo = torch.zeros(P, 3, 3)
    Fo[:, 0, 0], Fo[:, 1, 1], Fo[:, 2, 2] = 0.3, 0.5, 0.7
    Fo[:, 0, 1] = Fo[:, 1, 0] = 0.05
    tn = torch.arange(N_T, dtype=torch.float32) * DTN
    shape = 1.0 + xn[:, 1] ** 2 + 0.5 * xn[:, 2] ** 2
    Tn = (0.1 + tn[:, None]) * shape[None, :]
    return dict(
        op_id="OPT", xn=xn, static=torch.zeros(P, 0),
        cfg=torch.randn(N_T, 2, generator=g), forcing=torch.randn(N_T, 1, generator=g),
        Fo=Fo, Qsrc=0.1 * torch.randn(N_T, P, generator=g), Tn=Tn, Tn_ic=Tn[0].clone(),
        tn=tn, n_t=N_T, n_points=P, split_t=int(0.8 * N_T), dtn=DTN,
    )


def _model(**kw) -> RecurrentField:
    """HYBRID history, as in Achse 0/1: its channels sit at ``delta_grid`` and
    ``rate_lags``, not at ``delta``, so the network input -- and with it ``T`` --
    does not move when delta does. (In raw mode the history IS the BDF lag
    stack, delta changes the input, and the argument below would not hold as
    stated.) ``rate_scale`` keeps the untrained rollout bounded: at 1.0 the rate
    channels amplify a one-step difference ~20x and it runs away (ARCHITECTURE
    3.1), which would measure the divergence instead of the jump."""
    torch.manual_seed(0)
    cfg = dict(n_config=2, n_static=0, n_forcing=1, history_mode="hybrid",
               rate_lags=(5 * DTN, 10 * DTN), layer_size=16, num_layers=2,
               delta_seconds=2 * DTN, dtn=DTN, t_span_ref=1.0, delta_grid=DTN,
               rate_scale=50.0)
    cfg.update(kw)
    return RecurrentField(**cfg)


def _residual(model, op, buf, seed=1, n=200, return_parts=False):
    g = torch.Generator().manual_seed(seed)
    pt = torch.randint(1, N_T, (n,), generator=g)
    pp = torch.randint(0, op["n_points"], (n,), generator=g)
    return heat_residual(model, op["xn"], op["static"], op["cfg"][pt],
                         op["forcing"][pt], op["Fo"], op["Qsrc"][pt, pp], buf, DTN,
                         op["tn"][pt], pp, 2.5, return_parts=return_parts)


# --------------------------------------------------------------------------
# 1. return_parts is a window
# --------------------------------------------------------------------------

def test_return_parts_leaves_the_default_call_bit_for_bit_unchanged():
    op, model = _op(), _model()
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    plain = _residual(model, op, buf)
    windowed, _ = _residual(model, op, buf, return_parts=True)
    assert torch.equal(plain, windowed)


def test_the_parts_add_up_to_the_residual():
    op, model = _op(), _model()
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    res, p = _residual(model, op, buf, return_parts=True)
    aniso = p["aniso_xx"] + p["aniso_yy"] + p["aniso_zz"] + p["aniso_cross"]
    rebuilt = (p["dTdt"] - p["diff_gain"] * aniso - p["src_gain"] * p["Qsrc"]) / p["divisor"]
    rel = float((rebuilt - res).detach().norm() / res.detach().norm())
    assert rel < 1e-6
    assert set(p) >= {"T", "T_1", "T_2", "Txx", "Tyy", "Tzz"}


# --------------------------------------------------------------------------
# 2. the case against O18 as the cause of L_phys ~ 1/delta^2
# --------------------------------------------------------------------------

def _set_delta(model, delta_n):
    model._delta.fill_(float(delta_n))


def test_a_spatial_term_does_not_depend_on_delta():
    """``aniso`` is added to the residual, never divided by delta. Whatever is
    wrong inside it -- a missing ``(grad lambda).(grad T)`` included -- therefore
    cannot produce a 1/delta^2 signature. Same samples, two deltas, identical
    conduction terms."""
    op, model = _op(), _model()
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    parts = []
    for d in (5 * DTN, 1 * DTN):
        _set_delta(model, d)
        parts.append(_residual(model, op, buf, return_parts=True)[1])
    for k in ("aniso_xx", "aniso_yy", "aniso_zz", "aniso_cross", "Qsrc"):
        assert torch.equal(parts[0][k], parts[1][k]), k


def test_the_jump_is_zero_on_the_rollout_the_weights_produced():
    op, model = _op(), _model()
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    out = rd.decompose(model, op, buf, 2 * DTN, 2.5, n_samples=300,
                       gen=torch.Generator().manual_seed(3))
    assert out["jump_rms_z"] < 1e-5
    assert out["L_jump"] < 1e-6 * max(out["L"], 1.0)


def test_the_jump_scales_exactly_like_one_over_delta_squared():
    """Move the weights after the rollout -- which is what every inner step of
    train.py does -- and the live network leaves its own frozen trajectory by a
    delta-INDEPENDENT amount. Shifting the output bias by 0.1 makes that amount
    exactly 0.1 everywhere, so L_jump * delta^2 must come out constant."""
    op, model = _op(), _model()
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    with torch.no_grad():
        model.mlp.out.linear.bias += 0.1
    scaled = []
    for d in (1 * DTN, 2 * DTN, 5 * DTN):
        out = rd.decompose(model, op, buf, d, 2.5, n_samples=300,
                           gen=torch.Generator().manual_seed(3))
        assert out["jump_rms_z"] == pytest.approx(0.1, rel=1e-4)
        scaled.append(out["L_jump"] * d ** 2)
    assert max(scaled) / min(scaled) - 1.0 < 1e-3


# --------------------------------------------------------------------------
# 3. [BLIND]: autograd cannot see structure that arrives through the history
# --------------------------------------------------------------------------

def _q(xn):
    return 1.0 + 3.0 * xn[:, 1] ** 2 + 2.0 * xn[:, 2] ** 2


class _Explicit(RecurrentField):
    """The field written in xn: T_yy = 6, T_zz = 4, visible to autograd."""

    def field(self, xn, static, cfg, forcing, hist, level=None):
        return _q(xn) + 0.0 * hist.sum(dim=1)


class _ThroughAnchor(RecurrentField):
    """The SAME values, carried by the history anchor -- what a network that
    learnt T(t) ~ T(t - delta_grid) does. Autograd sees a constant in xn."""

    def field(self, xn, static, cfg, forcing, hist, level=None):
        # 0 * xn**2 keeps xn in the graph (autograd needs something to
        # differentiate twice) and contributes exactly nothing.
        return hist[:, 0] + 0.0 * (xn ** 2).sum(dim=1)


@pytest.mark.parametrize("cls,expected", [(_Explicit, 1.0), (_ThroughAnchor, 0.0)])
def test_laplacian_visibility_tells_explicit_from_carried(cls, expected):
    op = _op()
    model = _model()
    model.__class__ = cls
    buf = _q(op["xn"]).expand(N_T, -1).clone()        # every row the same field
    vis = rd.laplacian_visibility(model, op, buf, rows=[20, 40], phys_scale=2.5)
    for k in ("yy", "zz"):
        assert vis[k]["fd_rms"] == pytest.approx({"yy": 6.0, "zz": 4.0}[k], rel=1e-4)
        assert vis[k]["ratio"] == pytest.approx(expected, abs=1e-4)


def test_fd_line_keys_survive_float32_coordinates():
    """The first version keyed lines in float32; the keys rounded together, lines
    merged, and the stencil divided by a zero spacing (nan). The real grid:
    3 x 11 x 11, non-equidistant x, float32 as the loader delivers it."""
    xs = torch.tensor([0.0, 0.0544470, 0.1105540])
    ys = torch.linspace(0.0, 1.0, 11) * 0.9999
    zs = torch.linspace(0.0, 0.5272, 11)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    xn = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1).float()
    f = 3.0 * xn[:, 1] ** 2
    r = lambda v: torch.round(v.double() * 1e6).long()  # noqa: E731 -- as the tool
    d2, idx = rd._fd_second(f.double(), xn[:, 1].double(), r(xn[:, 0]) * 10**9 + r(xn[:, 2]))
    assert torch.isfinite(d2).all()
    assert idx.numel() == 3 * 11 * 9                 # 33 lines, 9 interior points each
    assert torch.allclose(d2, torch.full_like(d2, 6.0), rtol=1e-3)


# --------------------------------------------------------------------------
# the read-out
# --------------------------------------------------------------------------

def test_fit_inverse_square_recovers_its_coefficients():
    d = [1.0, 0.4, 0.2]
    A, B, err = rd.fit_inverse_square(d, [3.0 + 7.0 / x ** 2 for x in d])
    assert A == pytest.approx(3.0) and B == pytest.approx(7.0) and err < 1e-12


def test_the_achse1_numbers_are_almost_all_one_over_delta_squared():
    """The documented medians (FAHRPLAN 11.9): the fit leaves at most 5 %, and
    the 1/delta^2 part is > 99 % at delta = 0.2 s. Pinned, because FAHRPLAN 11.10
    quotes exactly these two numbers."""
    d, L = [1.0, 0.4, 0.2], [4.6e4, 2.5e5, 1.0e6]
    _, _, err = rd.fit_inverse_square(d, L)
    assert err < 0.05
    assert rd.share_inverse_square(d, L) > 0.99


def test_tags_can_hold_together():
    d = [1.0, 0.4, 0.2]
    flat = [1.0, 1.0, 1.0]
    inv = [1.0 / x ** 2 for x in d]
    assert rd.tags(d, flat, inv, 0.9, 1.0, 1.0) == ["[STALE]"]
    assert rd.tags(d, inv, inv, 0.1, 1.0, 0.01) == ["[JITTER]", "[BLIND]"]
    assert rd.tags(d, flat, flat, 0.9, 50.0, 1.0) == ["[NO 1/delta^2]", "[STALE]", "[JITTER]"]


def test_child_processes_see_the_parents_options():
    args = rd._parser().parse_args(["a.pt", "b.pt", "-j", "4", "--deltas", "1", "0.2",
                                    "--device", "cpu", "--samples", "64"])
    argv = rd._child_argv(args, Path("a.pt"))
    again = rd._parser().parse_args(argv)
    assert again.checkpoints == [Path("a.pt")] and again.jobs == 1
    for name in rd._PASS:
        assert getattr(again, name) == getattr(args, name), name


# --------------------------------------------------------------------------
# O20
# --------------------------------------------------------------------------

def test_train_refuses_the_autograd_time_derivative_before_reading_data():
    import train as train_mod
    with pytest.raises(SystemExit, match="mlp_with_time"):
        train_mod.fit(SimpleNamespace(time_deriv="autograd"))


# --------------------------------------------------------------------------
# --phys-stencil live (O21): the repair for [STALE], behind a switch
# --------------------------------------------------------------------------

def _live(op, pt, rows):
    import train as train_mod
    return train_mod._live_lag_inputs(op, pt, rows)


def _residual_at(model, op, buf, pt, pp, live_lags=None):
    return heat_residual(model, op["xn"], op["static"], op["cfg"][pt],
                         op["forcing"][pt], op["Fo"], op["Qsrc"][pt, pp], buf, DTN,
                         op["tn"][pt], pp, 2.5, live_lags=live_lags).detach()


def test_live_stencil_equals_the_buffer_stencil_on_a_fresh_rollout():
    """Right after the rollout the live network IS the buffer at grid rows, so
    the switch must not change anything there -- it only removes the drift."""
    op, model = _op(), _model()
    rows = 2
    _set_delta(model, rows * DTN)
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    g = torch.Generator().manual_seed(5)
    pt = torch.randint(0, N_T, (300,), generator=g)      # row 0 and early rows included
    pp = torch.randint(0, op["n_points"], (300,), generator=g)
    a = _residual_at(model, op, buf, pt, pp)
    b = _residual_at(model, op, buf, pt, pp, _live(op, pt, rows))
    assert float((a - b).norm() / a.norm()) < 1e-5


def test_live_stencil_cancels_the_drift_the_buffer_stencil_amplifies():
    """Move the weights after the rollout (output bias + 0.1, as in the 1/delta^2
    test). The buffer stencil picks up 3 * 0.1 / (2 delta); the live stencil
    shifts all three BDF points together and the offset cancels (3 - 4 + 1)."""
    op, model = _op(), _model()
    rows = 2
    _set_delta(model, rows * DTN)
    buf = rollout(model, op["xn"], op["static"], op["cfg"], op["forcing"],
                  op["Tn_ic"], op["tn"], DTN)
    g = torch.Generator().manual_seed(6)
    pt = torch.randint(2 * rows + 1, N_T, (300,), generator=g)   # both lags predicted
    pp = torch.randint(0, op["n_points"], (300,), generator=g)
    fresh = _residual_at(model, op, buf, pt, pp)
    with torch.no_grad():
        model.mlp.out.linear.bias += 0.1
    stale = _residual_at(model, op, buf, pt, pp)
    live = _residual_at(model, op, buf, pt, pp, _live(op, pt, rows))
    jump = 3.0 * 0.1 / (2.0 * rows * DTN) / 2.5
    assert torch.allclose(stale - fresh, torch.full_like(fresh, jump), rtol=1e-3)
    assert float((live - fresh).norm() / fresh.norm()) < 1e-4


def test_live_stencil_refuses_a_lag_between_rows():
    import train as train_mod
    assert train_mod.phys_lag_rows_for(1.0, 0.2) == 5
    assert train_mod.phys_lag_rows_for(0.2, 0.2) == 1
    with pytest.raises(SystemExit, match="whole number"):
        train_mod.phys_lag_rows_for(0.3, 0.2)
    with pytest.raises(SystemExit, match="whole number"):
        train_mod.phys_lag_rows_for(0.1, 0.2)
