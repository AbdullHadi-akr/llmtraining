"""The rollout history fast path must be BIT-identical to the general path.

These optimisations are only worth having if they are pure bookkeeping: same
history values in, same weights out, same seed reproducing the same run. So every
assertion here is ``torch.equal``, never ``allclose`` -- a one-ulp drift would mean
the fast path is a different model, not a faster one.

Two failure modes are specifically hunted:

* **Future leakage.** The general path passes the slice ``buf[:ti]``, and that
  slice is the only thing stopping step ``ti`` from reading its own prediction.
  The fast path reads the whole buffer and moves that bound into the plan's cap,
  so a wrong cap would feed the future back in -- training would look BETTER while
  being meaningless. ``test_plan_is_causal`` asserts the bound directly.
* **Stale cache.** ``n_t`` differs per OP and the benchmarks sweep ``rate_lags``,
  so a plan keyed too loosely would silently read the wrong rows.
"""

from __future__ import annotations

import pytest
import torch

from model import RecurrentField, rollout

DTN = 0.02
N_T = 40
P = 7

# Layouts worth covering: fractional lags (so interpolation actually blends),
# windows that run off the start of the buffer (so padding is exercised), a lag
# below the grid step (so the clamp binds), and the degenerate widths.
LAYOUTS = [
    pytest.param(dict(history_mode="raw", k_max=2), id="raw-k2"),
    pytest.param(dict(history_mode="raw", k_max=4), id="raw-k4"),
    pytest.param(dict(history_mode="raw", k_max=1), id="raw-k1"),
    pytest.param(dict(history_mode="raw", k_max=0), id="raw-k0"),
    pytest.param(
        dict(history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03),
        id="hybrid-fractional",
    ),
    pytest.param(
        dict(history_mode="hybrid", rate_lags=(0.04, 0.06, 0.5), delta_grid=DTN),
        id="hybrid-3lags-longwindow",
    ),
    pytest.param(
        dict(history_mode="hybrid", rate_lags=(0.05,), delta_grid=DTN / 4.0),
        id="hybrid-subgrid-anchor",
    ),
]


# float32 is what training actually runs in, and it is where a hoisted
# computation is most likely to drift by an ulp. float64 is kept alongside it so a
# failure separates "genuinely different" from "float32 got lucky".
DTYPES = [
    pytest.param(torch.float32, id="f32"),
    pytest.param(torch.float64, id="f64"),
]


def _model(dtype=torch.float64, **over) -> RecurrentField:
    torch.manual_seed(0)
    kw = dict(
        n_config=2, n_static=0, n_forcing=0, k_max=2, history_mode="raw",
        rate_lags=(0.05, 0.2), layer_size=8, num_layers=2,
        delta_seconds=1.0, dtn=DTN, t_span_ref=30.0, rate_scale=1.3,
        delta_grid=0.03, weight_norm=False,
    )
    kw.update(over)
    model = RecurrentField(**kw).to(dtype)
    # Damp the output layer so the free-running rollout below stays FINITE.
    #
    # Every assertion in this file compares the fast path against the general
    # path over a rollout buffer, and ``torch.equal(nan, nan)`` is False -- so a
    # buffer that reached inf does not weaken these tests, it voids them. The
    # untrained recurrence really does diverge at these settings:
    # ``hybrid-subgrid-anchor`` divides a temperature difference by
    # ``0.05 * 1.3``, feeding a 15x-amplified value back into the next step, and
    # 40 steps of that in float32 is enough. That divergence is the subject of
    # ``test_untrained_rollout_stays_finite``, which measures it deliberately;
    # here it is only noise, so the loop gain is taken below 1 and the equality
    # claims stay meaningful.
    with torch.no_grad():
        model.mlp.out.linear.weight.mul_(0.05)
    return model


def _grid(dtype=torch.float64) -> torch.Tensor:
    return torch.arange(N_T, dtype=dtype) * DTN


def _reference_history(model, buf, ti, tn):
    """What the loop used to call: general path, on the causal SLICE."""
    p_idx = torch.arange(buf.shape[1])
    return model._history(buf[:ti], DTN, tn[ti].expand(buf.shape[1]), p_idx)


# --------------------------------------------------------------------------
# 1. the fast path reproduces the general path exactly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("over", LAYOUTS)
def test_history_rollout_matches_general_path(over, dtype):
    model = _model(dtype=dtype, **over)
    tn = _grid(dtype)
    plan = model.rollout_plan(tn, DTN)
    torch.manual_seed(1)
    buf = torch.randn(N_T, P, dtype=dtype)

    for ti in range(1, N_T):
        fast = model.history_rollout(buf, ti, plan)
        ref = _reference_history(model, buf, ti, tn)
        assert fast.shape == ref.shape, f"step {ti}: {fast.shape} vs {ref.shape}"
        assert torch.equal(fast, ref), (
            f"step {ti} differs by up to {(fast - ref).abs().max()}"
        )


@pytest.mark.parametrize("over", LAYOUTS)
def test_history_block_width_is_k_max(over):
    model = _model(**over)
    plan = model.rollout_plan(_grid(), DTN)
    buf = torch.zeros(N_T, P, dtype=torch.float64)
    assert model.history_rollout(buf, 5, plan).shape == (P, model.k_max)


# --------------------------------------------------------------------------
# 2. causality: no step may read at or beyond its own row
# --------------------------------------------------------------------------

@pytest.mark.parametrize("over", LAYOUTS)
def test_plan_is_causal(over):
    """The cap must reproduce what the ``buf[:ti]`` slice used to enforce."""
    model = _model(**over)
    plan = model.rollout_plan(_grid(), DTN)
    if plan["n_off"] == 0:
        pytest.skip("no history channels")
    for ti in range(1, N_T):
        assert int(plan["lo"][ti].max()) <= ti - 1, f"step {ti} reads its future"
        assert int(plan["hi"][ti].max()) <= ti - 1, f"step {ti} reads its future"
        assert int(plan["lo"][ti].min()) >= 0
        assert torch.all(plan["frac"][ti] >= 0.0)
        assert torch.all(plan["frac"][ti] <= 1.0)


def test_step_one_reads_only_the_initial_condition():
    """Step 1 has exactly one row of history, so every channel must resolve to it."""
    model = _model(history_mode="raw", k_max=3)
    plan = model.rollout_plan(_grid(), DTN)
    assert int(plan["lo"][1].max()) == 0
    assert int(plan["hi"][1].max()) == 0


# --------------------------------------------------------------------------
# 3. end-to-end: the rollouts themselves are unchanged
# --------------------------------------------------------------------------

def _reference_rollout(model, xn, static, cfg_seq, forcing_seq, Tn_ic, tn, dtn):
    """The pre-optimisation loop, kept as the oracle for the fast path.

    It reads the causal SLICE ``buf[:ti]`` through the general history path, which
    is exactly what ``history_rollout`` replaces. The residual ``level`` is passed
    the same slice, so the only difference left between this and ``rollout`` is the
    index arithmetic under test.
    """
    n_t, n_p = tn.shape[0], xn.shape[0]
    buf = torch.zeros(n_t, n_p, dtype=xn.dtype, device=xn.device)
    buf[0] = Tn_ic
    p_idx = torch.arange(n_p, device=xn.device)
    for ti in range(1, n_t):
        past = buf[:ti]
        tq = tn[ti].expand(n_p)
        hist = model._history(past, dtn, tq, p_idx)
        cfg = cfg_seq[ti].expand(n_p, -1)
        forcing = forcing_seq[ti].expand(n_p, -1)
        buf[ti] = model.field(xn, static, cfg, forcing, hist,
                              model.level(past, dtn, tq))
    return buf


def _rollout_inputs(dtype=torch.float64):
    torch.manual_seed(2)
    return dict(
        xn=torch.randn(P, 3, dtype=dtype),
        static=torch.zeros(P, 0, dtype=dtype),
        cfg_seq=torch.randn(N_T, 2, dtype=dtype),
        forcing_seq=torch.zeros(N_T, 0, dtype=dtype),
        Tn_ic=torch.randn(P, dtype=dtype),
        tn=_grid(dtype),
        dtn=DTN,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("over", LAYOUTS)
def test_rollout_matches_reference_loop(over, dtype):
    model = _model(dtype=dtype, **over)
    args = _rollout_inputs(dtype)
    with torch.no_grad():
        got = rollout(model, **args)
        want = _reference_rollout(model, **args)
    assert torch.equal(got, want)


def test_rollout_carries_no_gradient():
    """``rollout`` is the frozen-trajectory generator, not a differentiable path.

    Its gradient-keeping twin ``rollout_train`` was removed: the history was
    detached between steps anyway, so ~7000 sequential steps bought exactly the
    gradient a minibatch against a FROZEN trajectory gives. ``train.py`` now takes
    that cheaper equivalent, and this asserts the buffer really is frozen -- a
    buffer that still required grad would mean the no_grad guard was lost again.
    """
    model = _model()
    buf = rollout(model, **_rollout_inputs())
    assert not buf.requires_grad


def test_history_is_detached_in_training_rollout():
    """The gradient argument for bit-exactness rests on history carrying no grad."""
    model = _model(history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03)
    plan = model.rollout_plan(_grid(), DTN)
    buf = torch.randn(N_T, P, dtype=torch.float64, requires_grad=True)
    assert not model.history_rollout(buf.detach(), 10, plan).requires_grad


# --------------------------------------------------------------------------
# 4. the hybrid dedupe changed nothing
# --------------------------------------------------------------------------

def _history_hybrid_original(model, Tn_seq, dtn, tn_q, p_idx, rate_lags):
    """Pre-dedupe implementation, kept verbatim as the oracle (1 + 2k lookups)."""
    dgrid = model._delta_grid.to(tn_q.dtype)
    T_anchor = model._padded_lookup(Tn_seq, dtn, tn_q - dgrid, p_idx)
    rates = []
    t_boundary = tn_q - dgrid
    for i in range(len(rate_lags)):
        seg_len = rate_lags[i]
        t_next = t_boundary - seg_len
        T_end = model._padded_lookup(Tn_seq, dtn, t_boundary, p_idx)
        T_start = model._padded_lookup(Tn_seq, dtn, t_next, p_idx)
        span = torch.clamp(seg_len, min=float(dtn))
        rates.append((T_end - T_start) / (span * model.rate_scale))
        t_boundary = t_next
    return torch.cat([T_anchor.unsqueeze(1), torch.stack(rates, dim=1)], dim=1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lags", [(0.05, 0.2), (0.04, 0.06, 0.5), (0.05,)])
def test_hybrid_dedupe_matches_original(lags, dtype):
    model = _model(dtype=dtype, history_mode="hybrid", rate_lags=lags,
                   delta_grid=0.03)
    torch.manual_seed(3)
    Tn_seq = torch.randn(N_T, P, dtype=dtype)
    # Random query times AND random point indices: the physics path, where the
    # bracketing indices genuinely vary across the batch.
    tn_q = torch.rand(64, dtype=dtype) * (N_T - 1) * DTN
    p_idx = torch.randint(0, P, (64,))

    got = model._history_hybrid(Tn_seq, DTN, tn_q, p_idx, model.rate_lags)
    want = _history_hybrid_original(model, Tn_seq, DTN, tn_q, p_idx, model.rate_lags)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------
# 5. the BDF2 reuse really is the same tensor
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k_max", [2, 4])
def test_raw_history_columns_equal_history_at(k_max, dtype):
    """What ``heat_residual`` now relies on instead of re-fetching."""
    model = _model(dtype=dtype, history_mode="raw", k_max=k_max)
    torch.manual_seed(4)
    Tn_seq = torch.randn(N_T, P, dtype=dtype)
    tn_q = torch.rand(64, dtype=dtype) * (N_T - 1) * DTN
    p_idx = torch.randint(0, P, (64,))

    hist = model._history(Tn_seq, DTN, tn_q, p_idx)
    for lag in (1, 2):
        want = model.history_at(Tn_seq, DTN, tn_q, p_idx, lag=lag)
        assert torch.equal(hist[:, lag - 1], want)


def test_hybrid_history_columns_are_not_lags():
    """Guards the mode check: hybrid columns are [anchor, rates], not T(t-i*delta)."""
    model = _model(history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03)
    torch.manual_seed(5)
    Tn_seq = torch.randn(N_T, P, dtype=torch.float64)
    tn_q = torch.rand(64, dtype=torch.float64) * (N_T - 1) * DTN
    p_idx = torch.randint(0, P, (64,))

    hist = model._history(Tn_seq, DTN, tn_q, p_idx)
    lag2 = model.history_at(Tn_seq, DTN, tn_q, p_idx, lag=2)
    assert not torch.equal(hist[:, 1], lag2)


# --------------------------------------------------------------------------
# 6. cache invalidation
# --------------------------------------------------------------------------

def test_plan_rebuilds_for_a_different_trajectory_length():
    """n_t is per-OP: a plan built for one OP must not be reused for a longer one."""
    model = _model(history_mode="raw", k_max=2)
    short = model.rollout_plan(_grid(), DTN)
    assert short["lo"].shape[0] == N_T

    long_tn = torch.arange(N_T * 2, dtype=torch.float64) * DTN
    long_plan = model.rollout_plan(long_tn, DTN)
    assert long_plan["lo"].shape[0] == N_T * 2

    # And back again -- the cache must not pin the first shape it ever saw.
    assert model.rollout_plan(_grid(), DTN)["lo"].shape[0] == N_T


def test_plan_rebuilds_when_rate_lags_change():
    """benchmark_arch.py sweeps rate_lags on models that have already rolled out."""
    model = _model(history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03)
    tn = _grid()
    before = model.rollout_plan(tn, DTN)["lo"].clone()

    with torch.no_grad():
        model._rate_lags.copy_(torch.tensor([0.1, 0.4], dtype=model._rate_lags.dtype))
    after = model.rollout_plan(tn, DTN)["lo"]

    assert not torch.equal(before, after), "stale plan survived a layout change"


def test_plan_is_reused_when_nothing_changed():
    model = _model(history_mode="raw", k_max=2)
    tn = _grid()
    assert model.rollout_plan(tn, DTN) is model.rollout_plan(tn, DTN)


def test_plan_stays_out_of_the_checkpoint():
    """The plan is derived state; leaking it into state_dict would break loading."""
    model = _model(history_mode="raw", k_max=2)
    keys_before = set(model.state_dict())
    model.rollout_plan(_grid(), DTN)
    assert set(model.state_dict()) == keys_before


# --------------------------------------------------------------------------
# 7. the point of the whole exercise: gradients are unchanged
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("over", LAYOUTS)
def test_gradients_match_reference_loop(over, dtype):
    """Identical forward values are only worth something if the grads follow.

    Since the training loop optimises against a FROZEN buffer, the gradient that
    matters is the one through a single ``field()`` evaluation fed by the history
    the fast path produced. History carries no gradient either way, so it enters
    ``field()`` as a constant: equal constants plus equal weights must give equal
    parameter gradients -- this asserts it rather than arguing it.
    """
    args = _rollout_inputs(dtype)
    tn, dtn, n_p = args["tn"], args["dtn"], args["xn"].shape[0]
    ti = N_T - 1

    fast = _model(dtype=dtype, **over)
    buf = rollout(fast, **args)
    plan = fast.rollout_plan(tn, dtn)
    tq = tn[ti].expand(n_p)
    cfg = args["cfg_seq"][ti].expand(n_p, -1)
    forcing = args["forcing_seq"][ti].expand(n_p, -1)
    fast.field(args["xn"], args["static"], cfg, forcing,
               fast.history_rollout(buf, ti, plan),
               fast.level(buf[:ti], dtn, tq)).pow(2).sum().backward()

    ref = _model(dtype=dtype, **over)          # same seed -> same init
    p_idx = torch.arange(n_p)
    ref.field(args["xn"], args["static"], cfg, forcing,
              ref._history(buf[:ti], dtn, tq, p_idx),
              ref.level(buf[:ti], dtn, tq)).pow(2).sum().backward()

    pairs = list(zip(fast.named_parameters(), ref.named_parameters()))
    assert pairs
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), f"{name}: initial weights differ, test is void"
        if a.grad is not None:
            assert torch.isfinite(a.grad).all(), (
                f"{name}: gradient is not finite, so the equality below would "
                f"compare nan against nan and assert nothing"
            )
        if a.grad is None and b.grad is None:
            continue
        assert torch.equal(a.grad, b.grad), f"{name}: gradient changed"


# --------------------------------------------------------------------------
# 8. heat_residual still runs, and the reuse did not change it
# --------------------------------------------------------------------------

def _residual(model, dtype, time_deriv, Tn_seq, tn_q, p_idx, xn):
    from physics import heat_residual
    B = tn_q.shape[0]
    return heat_residual(
        model, xn, torch.zeros(P, 0, dtype=dtype),
        torch.randn(B, 2, dtype=dtype, generator=torch.Generator().manual_seed(9)),
        torch.zeros(B, 0, dtype=dtype),
        torch.eye(3, dtype=dtype).expand(P, 3, 3).contiguous(),
        torch.zeros(B, dtype=dtype), Tn_seq, DTN, tn_q, p_idx,
        phys_scale=1.0, time_deriv=time_deriv,
    )


@pytest.mark.parametrize("time_deriv", ["bdf1", "bdf2"])
@pytest.mark.parametrize("mode", ["raw", "hybrid"])
def test_heat_residual_runs_and_is_finite(mode, time_deriv):
    """Covers both branches of the new _lag(): reuse (raw) and fallback (hybrid)."""
    dtype = torch.float64
    over = dict(history_mode="raw", k_max=2) if mode == "raw" else dict(
        history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03)
    model = _model(dtype=dtype, **over)
    torch.manual_seed(6)
    res = _residual(model, dtype, time_deriv,
                    torch.randn(N_T, P, dtype=dtype),
                    torch.rand(16, dtype=dtype) * (N_T - 1) * DTN,
                    torch.randint(0, P, (16,)),
                    torch.randn(P, 3, dtype=dtype))
    assert res.shape == (16,) and torch.isfinite(res).all()


def _heat_residual_original(model, xn, static, cfg, forcing, Fo, Qsrc, Tn_seq,
                            dtn, tn_q, p_idx, phys_scale, time_deriv="bdf2"):
    """Pre-optimisation ``heat_residual`` body, kept verbatim as the oracle.

    Carries both of the things that changed: the ``.clone()`` on the coordinate
    slice, and the explicit ``history_at`` re-lookups instead of reusing the
    history block that was just built. Everything else -- the residual level, and
    the assembly of the equation before ONE division by ``phys_scale`` -- is the
    current formulation, because those are deliberate changes to what the residual
    IS. What this asserts is only that the lookup reuse changed nothing.
    """
    from physics import _grad

    xb = xn[p_idx].clone().requires_grad_(True)
    hist = model._history(Tn_seq, dtn, tn_q, p_idx)
    level = model.level(Tn_seq, dtn, tn_q)
    T = model.field(xb, static[p_idx], cfg, forcing, hist, level)
    if time_deriv == "bdf2":
        T_1 = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=1)
        T_2 = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=2)
        dTdt = (3.0 * T - 4.0 * T_1 + T_2) / (2.0 * model.delta + 1e-8)
    else:
        T_prev = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=1)
        dTdt = (T - T_prev) / (model.delta + 1e-8)

    grad1 = _grad(T, xb)
    Txx_row, Tyy_row, Tzz_row = (_grad(grad1[:, i], xb) for i in range(3))
    Txx, Txy, Txz = Txx_row[:, 0], Txx_row[:, 1], Txx_row[:, 2]
    Tyy, Tyz = Tyy_row[:, 1], Tyy_row[:, 2]
    Tzz = Tzz_row[:, 2]

    fo = Fo[p_idx]
    aniso = (
        fo[:, 0, 0] * Txx + fo[:, 1, 1] * Tyy + fo[:, 2, 2] * Tzz
        + 2.0 * (fo[:, 0, 1] * Txy + fo[:, 0, 2] * Txz + fo[:, 1, 2] * Tyz)
    )
    residual = dTdt - model.diff_gain * aniso - model.src_gain * Qsrc
    return residual / (phys_scale + 1e-30)


@pytest.mark.parametrize("time_deriv", ["bdf1", "bdf2"])
@pytest.mark.parametrize("mode", ["raw", "hybrid"])
def test_heat_residual_matches_original(mode, time_deriv):
    """Reuse (raw, k_max>=2) and fallback (hybrid) must both reproduce the original."""
    from physics import heat_residual

    dtype = torch.float64
    over = dict(history_mode="raw", k_max=4) if mode == "raw" else dict(
        history_mode="hybrid", rate_lags=(0.05, 0.2), delta_grid=0.03)
    model = _model(dtype=dtype, **over)

    torch.manual_seed(7)
    B = 16
    shared = dict(
        xn=torch.randn(P, 3, dtype=dtype),
        static=torch.zeros(P, 0, dtype=dtype),
        cfg=torch.randn(B, 2, dtype=dtype),
        forcing=torch.zeros(B, 0, dtype=dtype),
        Fo=torch.eye(3, dtype=dtype).expand(P, 3, 3).contiguous(),
        Qsrc=torch.randn(B, dtype=dtype),
        Tn_seq=torch.randn(N_T, P, dtype=dtype),
        dtn=DTN,
        tn_q=torch.rand(B, dtype=dtype) * (N_T - 1) * DTN,
        p_idx=torch.randint(0, P, (B,)),
        phys_scale=1.0,
    )
    got = heat_residual(model, **shared, time_deriv=time_deriv)
    want = _heat_residual_original(model, **shared, time_deriv=time_deriv)
    assert torch.equal(got, want)


def test_raw_history_reuse_is_actually_taken():
    """Without this, the reuse could silently regress to the fallback and still pass."""
    from unittest.mock import patch

    dtype = torch.float64
    model = _model(dtype=dtype, history_mode="raw", k_max=4)
    torch.manual_seed(8)
    B = 8
    with patch.object(RecurrentField, "history_at",
                      side_effect=AssertionError("fell back to history_at")):
        from physics import heat_residual
        heat_residual(
            model, torch.randn(P, 3, dtype=dtype), torch.zeros(P, 0, dtype=dtype),
            torch.randn(B, 2, dtype=dtype), torch.zeros(B, 0, dtype=dtype),
            torch.eye(3, dtype=dtype).expand(P, 3, 3).contiguous(),
            torch.zeros(B, dtype=dtype), torch.randn(N_T, P, dtype=dtype), DTN,
            torch.rand(B, dtype=dtype) * (N_T - 1) * DTN,
            torch.randint(0, P, (B,)), phys_scale=1.0, time_deriv="bdf2",
        )


def test_plan_follows_the_time_grid_dtype_not_the_weights():
    """The general path derives query times from tn, so the plan must too."""
    model = _model(dtype=torch.float32, history_mode="hybrid",
                   rate_lags=(0.05, 0.2), delta_grid=0.03)
    tn64 = _grid(torch.float64)
    buf = torch.randn(N_T, P, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(11))
    plan = model.rollout_plan(tn64, DTN)
    for ti in range(1, N_T):
        assert torch.equal(model.history_rollout(buf, ti, plan),
                           _reference_history(model, buf, ti, tn64))


def test_plan_rebuilds_when_only_the_dtype_changes():
    model = _model(dtype=torch.float32, history_mode="raw", k_max=2)
    a = model.rollout_plan(_grid(torch.float32), DTN)["frac"]
    b = model.rollout_plan(_grid(torch.float64), DTN)["frac"]
    assert a.dtype != b.dtype, "stale plan survived a dtype change"
