"""Per-OP rollout metrics for ``train.py``.

A separate module rather than part of ``train.py`` so that whatever compares
configurations later can import the metrics without importing the training loop.

Why more than one number
------------------------
The base project reports a mean absolute error over the whole rollout. On a
constant OP that is a fair summary: the trajectory is one slow ramp, so the mean
is representative of every part of it. On a profile OP it is not. A CC-CV OP
spends most of its samples in the CC phase, where the drivers are flat and the
model has an easy time, and a short window in the CV taper, where the current
collapses and the temperature turns over. A mean over the whole trajectory is
dominated by the easy part, so a model that gets the turnover completely wrong
can still post a respectable MAE.

So each OP is scored with:

* ``mae`` / ``rmse``   -- the familiar whole-trajectory numbers, kept so results
  stay comparable with the base project.
* ``mae_transient`` / ``mae_quiescent`` -- the same error split by whether the
  drivers were moving (``data._transient_mask``). The gap between the two is the
  profile-specific quantity: on a constant OP the transient set is nearly empty
  and the split says nothing, which is itself the correct answer.
* ``max_abs_err`` -- the worst single sensor at the worst moment.
* ``peak_err``    -- error in the PEAK temperature the cell reaches. This is the
  number an aging model consumes; being right on average while missing the
  hot-spot peak by several degrees is the failure mode that matters and the one
  a mean hides.
* ``late_mae``    -- error over the last part of the trajectory (after
  ``split_t``). For a HELD-OUT OP this is a genuine held-out-in-time number. For
  a TRAINING OP it is in-sample unless training ran with ``--holdout-tail``,
  because the data loss covers the whole rollout. ``late_is_holdout`` carries
  that distinction so no report has to guess.
* ``bias_early`` / ``bias_mid`` / ``bias_end`` and ``late_bias`` --
  the SIGNED mean error, over equal thirds and over the same window as
  ``late_mae``. See below.

Why a signed error as well as an absolute one (O13)
---------------------------------------------------
The error on this model roughly doubles towards the end of a trajectory: OP06 is
at 6.270 C over the whole rollout and 13.248 C after ``split_t``, and OP03 and
OP16 show the same shape. Two very different faults produce that, and every
number above is an ABSOLUTE error, so they look identical in all of them:

* **Drift.** ``field`` predicts ``level(t - delta_grid) + net(...)``, where
  ``level`` is the spatial mean of the model's OWN previous prediction. Over
  ~7000 rollout steps that is an integrator of gain 1: a small systematic bias
  is added at every step and nothing pulls it back. The late error is then
  signed -- consistently too warm or consistently too cold.
* **A genuinely harder regime.** The end of a charge is the hottest phase with
  the steepest gradients, and the pooled statistics are fitted on ``[:split_t]``
  only. The late error is then large but UNSIGNED -- wrong in both directions.

The two call for opposite responses, so the report has to tell them apart.
``late_bias_frac = |late_bias| / late_mae`` is the one number that does: near 1
the late error is almost entirely directional (drift), near 0 it is scatter.
Nothing here decides O13 on its own -- it makes the run able to answer it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from model import rollout

# Boxplot sampling: one box per sampled time point summarising the SENSORS at
# that moment, drawn from a fixed seed so every configuration is scored at the
# same instants. Nothing draws these yet -- the benchmark that did is deleted --
# but the sampling rule belongs with the metrics it applies to.
N_BOX_TIMES = 10
BOX_TIME_SEED = 20240517


@torch.no_grad()
def rollout_phys(model, op, bundle, device) -> np.ndarray:
    """Free-running rollout for one OPData -> physical temperature (n_t, P).

    No teacher forcing: seeded with the measured initial condition, every later
    step reads history from the model's own predictions.
    """
    xn = torch.as_tensor(op.xn, dtype=torch.float32, device=device)
    static = torch.as_tensor(op.static_feat, dtype=torch.float32, device=device)
    forcing = torch.as_tensor(op.forcing_feat, dtype=torch.float32, device=device)
    cfg = torch.as_tensor(op.config_feat, dtype=torch.float32, device=device)
    tn = torch.as_tensor(op.tn, dtype=torch.float32, device=device)
    Tn_ic = torch.as_tensor(op.Tn_ic, dtype=torch.float32, device=device)
    # The model may have been built with the extra feature blocks switched off;
    # slicing here keeps one loader serving every feature configuration.
    static = static[:, :model.n_static]
    forcing = forcing[:, :model.n_forcing]
    buf = rollout(model, xn, static, cfg, forcing, Tn_ic, tn, op.dtn)
    return buf.cpu().numpy() * bundle.T_sigma + bundle.T_mu


def box_time_idx(n_t: int) -> np.ndarray:
    """The fixed random time points every configuration is scored at."""
    rng = np.random.default_rng(BOX_TIME_SEED)
    n = min(N_BOX_TIMES, int(n_t))
    return np.sort(rng.choice(int(n_t), size=n, replace=False))


def box_errors(pred: np.ndarray, true: np.ndarray,
               time_idx: np.ndarray) -> np.ndarray:
    """Absolute error at the sampled times, kept per sensor (n_times, n_sensors).

    Deliberately not reduced: at a moment where the mean error looks acceptable
    a handful of sensors can still be far off, and that shows as a long upper
    whisker rather than in the mean.
    """
    return np.abs(pred[time_idx] - true[time_idx]).astype(float)


def op_metrics(pred: np.ndarray, op, *, late_is_holdout: bool) -> Dict[str, float]:
    """All per-OP numbers from one rollout. ``pred``/``op.T_lab`` in degrees C.

    Step 0 is excluded everywhere: it is the imposed initial condition, so its
    error is identically zero and including it only dilutes the average by a
    sample the model never predicted.
    """
    true = np.asarray(op.T_lab, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    resid = (pred - true)[1:]          # SIGNED: positive = prediction too warm
    err = np.abs(resid)
    diff2 = resid ** 2
    transient = np.asarray(op.transient, dtype=bool)
    transient = transient[1:] if transient.size == op.n_t else np.zeros(len(err), bool)
    split = max(int(op.split_t) - 1, 1)

    def _mean(a):
        return float(a.mean()) if a.size else float("nan")

    # Equal thirds rather than the split_t boundary: the shape of the bias over
    # time is what separates an integrator running away (monotone in one
    # direction) from a hard final phase (no ordering). ``late_bias`` then uses
    # the SAME window as ``late_mae`` so the pair can be read as a ratio.
    n_err = err.shape[0]
    third = max(n_err // 3, 1)
    late_mae = _mean(err[split:])
    late_bias = _mean(resid[split:])

    return {
        "mae": _mean(err),
        "rmse": float(np.sqrt(_mean(diff2))),
        "max_abs_err": float(err.max()) if err.size else float("nan"),
        # Peak temperature over the whole field and the whole run: the hot-spot
        # number the aging model downstream actually consumes.
        "peak_err": float(abs(pred.max() - true.max())),
        "peak_true": float(true.max()),
        "peak_pred": float(pred.max()),
        "mae_transient": _mean(err[transient]),
        "mae_quiescent": _mean(err[~transient]),
        "transient_frac": float(transient.mean()) if transient.size else 0.0,
        "late_mae": late_mae,
        "late_is_holdout": bool(late_is_holdout),
        # Signed means. A sign flip between thirds is scatter; the same sign
        # growing in magnitude is an integrator.
        "bias_early": _mean(resid[:third]),
        "bias_mid": _mean(resid[third:2 * third]),
        "bias_end": _mean(resid[2 * third:]),
        "late_bias": late_bias,
        # In [0, 1] by construction: |mean| can never exceed mean|.|.
        "late_bias_frac": (abs(late_bias) / late_mae
                           if late_mae and np.isfinite(late_mae) else float("nan")),
    }


def format_op_metrics(op_id: str, tier: str, m: Dict[str, float]) -> str:
    late = "held out" if m["late_is_holdout"] else "in-sample"
    tr = (f"  transient={m['mae_transient']:.3f}"
          if np.isfinite(m["mae_transient"]) else "  transient=n/a")
    return (
        f"  {op_id} [{tier:<11}] MAE={m['mae']:.3f} C  RMSE={m['rmse']:.3f} C"
        f"  max={m['max_abs_err']:.3f} C  peak_err={m['peak_err']:.3f} C"
        f"{tr}  quiescent={m['mae_quiescent']:.3f}"
        f"  late({late})={m['late_mae']:.3f}"
        f"  late_bias={m['late_bias']:+.3f} ({m['late_bias_frac']:.0%} of it)"
        f"  bias/3={m['bias_early']:+.2f}/{m['bias_mid']:+.2f}/{m['bias_end']:+.2f}"
    )
