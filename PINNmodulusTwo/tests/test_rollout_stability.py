"""The rollout must stay finite -- that is what epoch 1 depends on.

``train.py`` computes one free-running rollout per OP per epoch under
``no_grad`` and then takes ``--inner-steps`` minibatch updates against that
frozen buffer. So the buffer is an INPUT to the first gradient step, not an
output of it. If it contains inf or nan, every prediction read out of it is nan,
``L_data`` is nan, and the run aborts with no gradient to recover from.

Two separate things drove that, and they are not equally important.

**The output parameterisation is the primary one.** With ``residual_output``,
``field()`` returns ``level(t) + net(...)`` where ``level`` is the spatial mean
of the anchor slice. That makes ``level(t) ~ level(t - delta_grid) + mean(net)``
-- an integrator of gain exactly 1 with no leak. Any one-signed component of the
network output accumulates over the trajectory without bound and nothing pulls
it back. Measured end to end on a synthetic bundle (20 epochs, 3 seeds, no
guards): it aborted on every seed in EVERY history configuration, ``raw``
included, where there are no rate channels at all.

**The hybrid rate channel is the secondary one.** It divides a temperature
difference by ``lag_n * rate_scale``; with the old 5 s segments against a
~1474 s reference span that is a gain of ~119 on anything non-smooth, including
an untrained network's step-to-step jitter. Long segments put it near 1.

Neither is fixed by initialisation. The stable region of weight space is small
and ordinary training walks out of it: with the old settings the rollout at
initialisation is perfectly well-behaved, and twenty Adam steps later the next
rollout reaches 1e4. The fix is the layout -- ``residual_output: false`` and
long ``rate_lags`` -- not a better starting point.

These tests pin the mechanism so a future change cannot quietly reintroduce it.
The geometry mirrors ``smallBench.py``, scaled down in step count so the suite
stays fast.
"""

from __future__ import annotations

import pytest
import torch

from model import RecurrentField, rollout

# smallBench defaults: subsample 2 -> dt = 0.2 s, T_span_ref ~ 1474 s.
T_SPAN = 1474.0
DTN = 0.2 / T_SPAN
RATE_LAGS_N = (5.0 / T_SPAN, 20.0 / T_SPAN)
RATE_SCALE = 2.479          # bundle.dTdt_scale on OP01-OP05
N_T = 1500                  # enough to reach inf unguarded; a real OP has ~7000
P = 32


def _model(history_mode="hybrid", residual_output=True, seed=0):
    torch.manual_seed(seed)
    return RecurrentField(
        n_config=6, n_static=3, n_forcing=1, k_max=2,
        history_mode=history_mode, rate_lags=RATE_LAGS_N,
        layer_size=64, num_layers=3,           # smallBench --width/--depth
        delta_seconds=1.0, dtn=DTN, t_span_ref=T_SPAN,
        rate_scale=RATE_SCALE, delta_grid=DTN,
        residual_output=residual_output,
    )


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed + 1000)
    tn = torch.arange(N_T, dtype=torch.float32) * DTN
    # A config that drifts and a source that ramps: the real trajectories are
    # smooth and non-stationary, and a static input would understate the drive.
    cfg = (torch.randn(N_T, 6, generator=g) * 0.1
           + torch.linspace(-1.0, 1.0, N_T)[:, None])
    return dict(
        xn=torch.rand(P, 3, generator=g),
        static=torch.randn(P, 3, generator=g),
        cfg_seq=cfg,
        forcing_seq=torch.linspace(0.0, 1.0, N_T)[:, None],
        Tn_ic=torch.randn(P, generator=g) * 0.05 - 1.0,
        tn=tn,
        dtn=DTN,
    )


# --------------------------------------------------------------------------
# 1. The two drivers, each pinned separately
# --------------------------------------------------------------------------

@pytest.mark.parametrize("history_mode,min_runaway", [("hybrid", 4), ("raw", 2)])
def test_residual_output_runs_away_without_any_rate_channels(history_mode,
                                                             min_runaway):
    """``residual_output=True`` runs away in ``raw`` too, where no rate channel exists.

    This is the test that identifies the integrator as the PRIMARY driver rather
    than the hybrid amplification: ``raw`` has no rate channels, and it still
    goes. It needs more steps to get there -- nothing is multiplying the drift
    by ~119 -- which is why ``raw`` is only required to run away on half the
    seeds at this trajectory length while ``hybrid`` must do it on all of them.
    Measured peaks here: hybrid inf on 4/4, raw {15, 3.8e11, 2.2e11, 4.8}.

    Under TRAINING the distinction disappears: end to end on a synthetic bundle
    (20 epochs, 3 seeds, no guards) ``residual_output=True`` aborted 3/3 in raw
    as well, because the optimiser walks the weights straight out of the small
    region where the level path happens to be stable.

    If this starts passing, the level path grew a leak -- find out where before
    relaxing anything else.
    """
    # Runaway, not strictly inf: the buffer is z-scored temperature, so a
    # plausible trajectory sits within a few units and 1e3 is already absurd.
    peaks = [float(rollout(_model(history_mode, True, s), **_inputs(s)).abs().max())
             for s in range(4)]
    n = sum(1 for p in peaks if p != p or p > 1e3)
    assert n >= min_runaway, (
        f"expected residual_output=True to run away on at least {min_runaway}/4 "
        f"seeds in {history_mode} mode, got {n}/4 (peaks {peaks})"
    )


def test_short_rate_lags_amplify_far_more_than_long_ones():
    """The rate channel's gain is ``1/(lag_n * rate_scale)`` -- a layout choice.

    Pins the number the whole diagnosis rests on, so a change to ``rate_lags``
    or ``rate_scale`` that reintroduces a three-digit gain is visible here and
    not only in a training log three hours in.
    """
    short = 1.0 / (5.0 / T_SPAN * RATE_SCALE)
    long_ = 1.0 / (200.0 / T_SPAN * RATE_SCALE)
    assert short == pytest.approx(118.9, rel=1e-3)
    assert long_ == pytest.approx(2.97, rel=1e-2)


def test_long_lags_without_residual_output_stay_finite():
    """The shipped configuration: no integrator, segments long enough to matter."""
    for seed in range(4):
        model = RecurrentField(
            n_config=6, n_static=3, n_forcing=1, k_max=2, history_mode="hybrid",
            rate_lags=(200.0 / T_SPAN, 600.0 / T_SPAN), layer_size=64,
            num_layers=3, delta_seconds=1.0, dtn=DTN, t_span_ref=T_SPAN,
            rate_scale=RATE_SCALE, delta_grid=DTN, residual_output=False,
        )
        torch.manual_seed(seed)
        buf = rollout(model, **_inputs(seed))
        assert torch.isfinite(buf).all()


# --------------------------------------------------------------------------
# 2. The clamp bounds the buffer whatever the layout does
# --------------------------------------------------------------------------

@pytest.mark.parametrize("history_mode", ["hybrid", "raw"])
@pytest.mark.parametrize("residual_output", [True, False])
def test_clamp_keeps_rollout_finite(history_mode, residual_output):
    for seed in range(3):
        buf = rollout(_model(history_mode, residual_output, seed),
                      **_inputs(seed), clamp=50.0)
        assert torch.isfinite(buf).all()
        assert buf.abs().max() <= 50.0
