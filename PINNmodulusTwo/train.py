#!/usr/bin/env python3
"""Train the Approach-2 recurrent Modulus PINN on the full OP set.

Scope note (31.08.2026): there is no longer a constant-driver project and a
profile extension. There is one project, and it trains on the whole plan sheet
-- OP01-OP16, constant drivers and profiles together -- because the profile
pipeline is a strict superset: a constant driver is a profile that does not
move, and ``--resample point --no-driver-history`` reproduces the old
constant-only preprocessing exactly if it is ever needed for a comparison.

OP17-OP19 are NOT part of that set and cannot be: they are the mini-module
MEASUREMENT comparison, partly discharge, with drivers read from test data
rather than the plan sheet, and only OP19 has a bundle at all. ``--measurement-
ops`` rolls them out and reports them, never trains or selects on them.

Temperature only (bc_V is intentionally out of scope). The model uses a Modulus
``FCLayer`` MLP with a per-layer learnable swish, wrapped in a PyTorch recurrence
whose history spacing ``delta`` and lag count ``k`` are FIXED hyperparameters --
they are configured, not learned. See ``model.RecurrentField`` for what the
recurrence does and does not learn.

Run (the device defaults to ``auto`` = CUDA when a GPU is available):
    source .venv/bin/activate
    python3 PINNmodulusTwo/train.py --epochs 60 --subsample 40

For the GPU server setup see ``PINNmodulusTwo/README_GPU_SERVER.md``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data import (
    build_op, cache_is_synthetic, coverage_report, effective_rate_scale,
    available_ops, load_ops, normalisation_report, profile_report,
    require_ops,
)
from device_utils import enable_tf32, resolve_device, seed_everything
from model import RecurrentField, rollout
from op_metrics import format_op_metrics, op_metrics, rollout_phys
from op_registry import (
    DEFAULT_TEST_OPS, DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS, MEASUREMENT_OPS,
    TIER_IN, split_summary, tier_or_unknown,
)
from physics import heat_residual, boundary_condition_loss

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Every per-epoch series ``fit`` records. One tuple rather than a dict literal
# buried in ``fit``, because the abort path appends to each series by hand: a
# series added in one place and forgotten in the other misaligns the CSV and the
# plots by a row, silently. ``fit`` builds the dict from this and asserts the
# lengths match, and tests/test_local_smoke.py asserts the set.
#
#   L_*_bal        balanced losses -- what the optimiser actually saw
#   ratio_*        w * L_term / (w_data * L_data), i.e. the mix the weights set
#   div_*          the divisors those balanced losses were produced with. Without
#                  them a small L_phys_bal is unreadable: either the term fell or
#                  the EMA divisor is stale from an earlier regime, and those call
#                  for opposite responses.
#   spread_*       the model's own rollout spread against the labels', in space
#                  and in time. The heat residual and the Neumann BC are both
#                  satisfied EXACTLY by a field constant in space and time, so
#                  "L_phys went to zero" is only good news while these stay near
#                  1. They are what separates a converged physics term from a
#                  collapsed one.
HISTORY_KEYS = (
    "epoch", "L_data", "L_phys", "L_bc",
    "L_phys_bal", "L_bc_bal", "ratio_phys", "ratio_bc",
    "div_data", "div_phys", "div_bc",
    "spread_space", "spread_time", "delta",
)


def _check_cfl_stability(bundle, delta_s, device):
    """Warn if the effective time step violates a simple CFL estimate."""
    alpha_max = float(bundle.Fo.max()) * bundle.L_ref**2 / (bundle.T_span_ref + 1e-30)
    L_axis = bundle.xn.max(axis=0) - bundle.xn.min(axis=0)
    vol = float(np.prod(L_axis * bundle.L_ref))
    dx_est = (vol / bundle.xn.shape[0]) ** (1.0 / 3.0)
    dt_max_cfl = dx_est**2 / (6.0 * alpha_max + 1e-30)
    stable = float(delta_s) <= dt_max_cfl
    status = "CFL OK" if stable else "CFL WARN"
    print(
        f"[{status}] Δt={float(delta_s):.3f}s, "
        f"Δt_max≈{dt_max_cfl:.3f}s -> "
        f"{'STABLE' if stable else 'POTENTIALLY UNSTABLE'}",
        flush=True,
    )
    return stable


def _load_yaml_defaults() -> dict:
    cfg_path = THIS_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    d = _load_yaml_defaults()
    p = argparse.ArgumentParser(description="Approach-2 recurrent Modulus PINN")
    p.add_argument("--ops", nargs="+",
                   default=d.get("ops", list(DEFAULT_TRAIN_OPS)),
                   help="training OPs. The default is the whole plan sheet's "
                        "training tier, constant drivers and profiles together")
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 40))
    p.add_argument("--epochs", type=int, default=d.get("epochs", 60))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 4))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "raw"))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 20.0]),
                   help="hybrid rate segments in SECONDS. The number that "
                        "decides stability is A = 1/(lag_n * rate_scale), "
                        "printed at startup: a short segment divides a small "
                        "temperature difference by a small number and so "
                        "amplifies everything non-smooth by A. 5 s against a "
                        "~1474 s reference span gives A ~ 119 and the rollout "
                        "diverges; 200 s gives A ~ 3")
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2),
                   help="anchor lag of the hybrid history in SECONDS: the block is "
                        "[T(t-delta_grid), rate_1, ...] and the rate segments "
                        "cascade back from there. Independent of --subsample.")
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0),
                   help="LR multiplier for src_gain/diff_gain; 1.0 = same LR as the "
                        "rest (they then tend to stay stuck at their 1.0 init)")
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 0.0),
                   help="maximum gradient norm; 0 disables clipping")
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0),
                   help="epochs without training-loss improvement; 0 disables it")
    p.add_argument("--w-data", type=float, default=d.get("w_data", 1.0))
    p.add_argument("--w-phys", type=float, default=d.get("w_phys", 0.1))
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--loss-balance", choices=["ema", "legacy", "fixed"],
                   default=d.get("loss_balance", "ema"),
                   help="which loss terms are divided by their own magnitude. "
                        "ema = all three (w_* are then true ratios that mean the "
                        "same in epoch 1 and 60); legacy = only phys and bc, so "
                        "the mix drifts towards physics as L_data falls and the "
                        "best w_phys depends on --epochs; fixed = freeze the "
                        "divisors after --balance-warmup epochs. NOTE: legacy is "
                        "the historical SCHEME, not a byte-identical replay -- it "
                        "keeps the corrected divisor (previous estimate, not one "
                        "that includes the current sample), so a long run drifts "
                        "slightly from pre-fix numbers")
    p.add_argument("--ema-decay", type=float, default=d.get("ema_decay", 0.9),
                   help="EMA decay PER EPOCH for the loss balancing. Corrected "
                        "internally for the number of OPs and --inner-steps, so "
                        "the horizon no longer changes when either does")
    p.add_argument("--balance-warmup", type=int, default=d.get("balance_warmup", 1),
                   help="epochs the divisors track before --loss-balance fixed "
                        "freezes them; ignored in the other modes")
    p.add_argument("--data-floor", type=float, default=d.get("data_floor", 1e-8),
                   help="lower bound on the L_data divisor, so a nearly perfect "
                        "fit cannot amplify its own gradient without bound")
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0),
                   help="fixed divisor for L_phys; 0 = follow --loss-balance")
    p.add_argument("--bc-norm", type=float, default=d.get("bc_norm", 0.0),
                   help="fixed divisor for L_bc; 0 = follow --loss-balance")
    p.add_argument("--residual-norm", choices=["rms", "legacy"],
                   default=d.get("residual_norm", "rms"),
                   help="rms divides each residual term by its own training RMS, "
                        "which is what puts them at unit scale. legacy keeps the "
                        "original division by sqrt(RMS), which leaves the three "
                        "terms with their size gap intact")
    p.add_argument("--inner-steps", type=int, default=d.get("inner_steps", 100),
                   help="optimiser steps per OP per epoch, all against that "
                        "epoch's frozen rollout. The rollout is the expensive "
                        "part, so this raises the update count at roughly "
                        "constant cost; 1 reproduces the old one-step-per-OP "
                        "behaviour")
    p.add_argument("--rollout-clamp", type=float,
                   default=d.get("rollout_clamp", 50.0),
                   help="saturate the rollout buffer at +/-this many normalised "
                        "temperature units; 0 disables. A working trajectory "
                        "stays within a few units, so this never binds on a sane "
                        "model -- it exists so a diverging one produces a finite "
                        "loss that training can still move, instead of an inf "
                        "that makes every downstream term NaN. It is reported "
                        "whenever it binds")
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0),
                   help="cap the hybrid history's amplification A = 1/(lag_n * "
                        "rate_scale) by raising rate_scale; 0 = leave rate_scale "
                        "at dTdt_scale. A is printed at startup either way. This "
                        "CHANGES THE MODEL. Prefer longer --rate-lags, which lower "
                        "A by making the segment a real span rather than by "
                        "rescaling a channel; record it when you use this")
    # Default FALSE, and it has to be stated here as well as in config.yaml.
    # _load_yaml_defaults() swallows every error (a missing pyyaml included) and
    # returns {}, so this literal is what a machine without pyyaml trains with --
    # and with True there every run aborts in epoch 1 with L_data=nan. See the
    # --residual-output note in README.md and ARCHITECTURE.md 3.1.
    p.add_argument("--residual-output", action=argparse.BooleanOptionalAction,
                   default=d.get("residual_output", False),
                   help="predict the deviation from the spatially averaged "
                        "temperature level of the anchor slice instead of the "
                        "absolute value, so the level is carried through the "
                        "rollout rather than re-predicted at every step")
    p.add_argument("--learn-gains", action=argparse.BooleanOptionalAction,
                   default=d.get("learn_gains", False),
                   help="let src_gain/diff_gain train. Off: they are pinned at "
                        "1.0, because the residual no longer needs them to undo "
                        "a per-term normalisation -- and free gains can be driven "
                        "to 0, which satisfies L_phys with a constant field")
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    # These four were in config.yaml, documented, and unreachable: fit() called
    # load_ops(op_ids, subsample_time) and nothing else, and zero_weight_terms
    # was read with getattr() off an attribute no argparse entry created. Every
    # default below equals what load_ops / the getattr fell back to, so wiring
    # them changes no result -- it only makes the knobs work.
    p.add_argument("--resample", choices=["mean", "point"],
                   default=d.get("resample", "mean"),
                   help="how DRIVERS are reduced onto the subsampled grid. "
                        "'mean' averages each driver over the raw interval that "
                        "ends at the kept sample -- anti-aliased, and for the "
                        "heat source energy-preserving. 'point' is a plain "
                        "[::N] and is exact only while the drivers are constant, "
                        "which is why it cannot be the default any more. "
                        "Temperature is always point-sampled: it is a state, "
                        "not a rate")
    p.add_argument("--driver-history", action=argparse.BooleanOptionalAction,
                   default=d.get("use_driver_history", True),
                   help="append causal rate channels for q_dot and the four "
                        "profile-capable config channels. Off leaves the model "
                        "with instantaneous driver values only")
    p.add_argument("--driver-rate-lags", nargs="+", type=float,
                   default=d.get("driver_rate_lags", [5.0, 20.0]),
                   help="cumulative segment lengths in SECONDS for the driver "
                        "rate channels. Exogenous, so unlike --rate-lags these "
                        "carry no feedback and no amplification risk")
    p.add_argument("--zero-weight-terms", choices=["skip", "compute"],
                   default=d.get("zero_weight_terms", "skip"),
                   help="skip = a term with weight 0 costs no forward pass and "
                        "is logged as NaN (an absent measurement, not a "
                        "measurement of zero); compute = evaluate it anyway")
    p.add_argument("--train-frac", type=float, default=d.get("train_frac", 0.8),
                   help="fraction of each OP's timeline used for the pooled "
                        "statistics and for split_t")
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", False))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", False))
    p.add_argument("--shuffle-ops", action=argparse.BooleanOptionalAction,
                   default=d.get("shuffle_ops", True),
                   help="reshuffle the OP order every epoch. With a dozen "
                        "heterogeneous OPs a fixed order lets the same OP always "
                        "take the last optimiser step of every epoch")
    p.add_argument("--holdout-tail", action=argparse.BooleanOptionalAction,
                   default=d.get("holdout_tail", False),
                   help="truncate the TRAINING rollout at split_t so the late "
                        "window of a training OP is genuinely held out. Off by "
                        "default: with CC-CV OPs the late window IS the CV "
                        "taper, and dropping it removes the hardest part of the "
                        "trajectory from training")
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", default=d.get("device", "ask"),
                   help="ask | auto | cpu | cuda | cuda:N. 'ask' lists what this "
                        "machine has and prompts; it falls back to 'auto' "
                        "without blocking when there is no terminal (CI, nohup, "
                        "a pipe). 'auto' = cuda when available. An explicit "
                        "'cuda' FAILS if the card is not there rather than "
                        "silently running on the CPU")
    p.add_argument("--tf32", action="store_true", default=d.get("tf32", False),
                   help="allow TF32 matmuls on Ampere+ GPUs; off by default because "
                        "the physics residual needs precise second derivatives")
    # Held-out OPs, evaluated after training with the TRAINING bundle's
    # normalisation constants (data.build_op re-fits nothing). Neither takes
    # part in training; the split between them is one of intent, and only this
    # file enforces it:
    #   --val-ops   what you are allowed to look at while tuning. Every
    #               hyperparameter you pick by comparing runs is chosen on these,
    #               so their MAE is optimistic by exactly as much as you tuned.
    #   --test-ops  the report number. Look at it once, at the end. Choosing
    #               anything on it turns it into a second validation set and
    #               there is no held-out estimate left.
    p.add_argument("--save-checkpoint", default=d.get("save_checkpoint", "model.pt"),
                   help="filename under artifacts/ for the trained weights; "
                        "empty string disables. The file carries everything "
                        "RecurrentField and the de-normalisation need, so a run "
                        "is reloadable without config.yaml")
    p.add_argument("--val-ops", nargs="*",
                   default=d.get("val_ops", list(DEFAULT_VAL_OPS)),
                   help="held-out OPs to rank configurations on (not trained on)")
    p.add_argument("--test-ops", nargs="*",
                   default=d.get("test_ops", list(DEFAULT_TEST_OPS)),
                   help="held-out OPs reported once and never selected on")
    # The mini-module MEASUREMENT comparison. A different exercise from every
    # OP above: measured data rather than a Batemo/StarCCM+ simulation, partly
    # discharge where the training block is all charge, and OP19 is a synthetic
    # drive cycle. It is a report, never a training or selection input -- and of
    # OP17/OP18/OP19 only OP19 has a bundle in this pipeline at all.
    p.add_argument("--measurement-ops", nargs="*",
                   default=d.get("measurement_ops", []),
                   help=f"mini-module measurement OPs to roll out and report; "
                        f"never trained or selected on. Candidates: "
                        f"{', '.join(MEASUREMENT_OPS)}")
    return p.parse_args()


def _to_tensor_ops(bundle, device):
    """Move the per-OP arrays used in the hot loop onto ``device`` as tensors."""
    packed = []
    for op in bundle.ops:
        packed.append(
            dict(
                op_id=op.op_id,
                xn=torch.as_tensor(op.xn, dtype=torch.float32, device=device),
                cfg=torch.as_tensor(op.config_feat, dtype=torch.float32, device=device),
                static=torch.as_tensor(op.static_feat, dtype=torch.float32, device=device),
                forcing=torch.as_tensor(op.forcing_feat, dtype=torch.float32, device=device),
                Tn=torch.as_tensor(op.Tn, dtype=torch.float32, device=device),
                Tn_ic=torch.as_tensor(op.Tn_ic, dtype=torch.float32, device=device),
                Fo=torch.as_tensor(op.Fo, dtype=torch.float32, device=device),
                Qsrc=torch.as_tensor(op.Qsrc, dtype=torch.float32, device=device),
                tn=torch.as_tensor(op.tn, dtype=torch.float32, device=device),
                n_t=op.n_t, n_points=op.n_points, split_t=op.split_t, dtn=op.dtn,
                T_lab=op.T_lab, t=np.asarray(op.t),
            )
        )
    return packed


def _check_finite_inputs(ops) -> None:
    """Warn once if any per-OP input tensor holds NaN/Inf.

    A loss that is already NaN in epoch 1 usually comes from the data, not from
    the optimisation - this says which OP and which field, instead of leaving the
    generic "CFL violation" guess as the only hint.
    """
    fields = ("xn", "cfg", "static", "forcing", "Tn", "Tn_ic", "Fo", "Qsrc", "tn")
    bad = []
    for op in ops:
        for name in fields:
            t = op[name]
            if t.numel() and not torch.isfinite(t).all():
                n_bad = int((~torch.isfinite(t)).sum())
                bad.append(f"{op['op_id']}.{name} ({n_bad}/{t.numel()} non-finite)")
    if bad:
        print("  [DATA WARN] non-finite values in the inputs: " + ", ".join(bad),
              flush=True)
        print("  Training will very likely produce NaN losses. Check the cached "
              ".npz bundles for these OPs.", flush=True)


class _LossBalancer:
    """Divides each loss term by an estimate of its own magnitude.

    The point is to make ``w_data``, ``w_phys`` and ``w_bc`` mean the same thing
    throughout a run. A term divided by its own running average sits near 1, so
    the weights express a ratio between terms rather than a ratio between their
    accidental units -- but only for the terms that actually get divided, which
    is what the ``mode`` selects (see the block in ``fit``).

    Two details that look like nitpicks and are not:

    * The divisor is the estimate from BEFORE this step. Folding the current
      value in first (as the original code did) lets a spike partly cancel
      itself: a term jumping 10x would be reported as ~5x. The signal being
      damped here is exactly the one worth seeing.
    * A non-finite sample never enters the average. ``decay * nan`` is ``nan``,
      so a single bad step would otherwise pin the divisor at nan for the rest
      of the run and silently poison every later epoch.
    """

    KEYS = ("data", "phys", "bc")

    def __init__(self, *, mode: str, decay: float, warmup_steps: int,
                 phys_norm: float, bc_norm: float, data_floor: float) -> None:
        self.mode = mode
        self.decay = decay
        self.warmup_steps = max(1, warmup_steps)
        self.data_floor = data_floor
        self.override = {"data": 0.0, "phys": phys_norm, "bc": bc_norm}
        self._ema: dict = {k: None for k in self.KEYS}
        self._frozen: dict = {k: None for k in self.KEYS}
        self.last: dict = {k: 1.0 for k in self.KEYS}
        self._steps = 0

    def divisor(self, key: str, value: float) -> float:
        override = self.override.get(key, 0.0)
        if override > 0.0:
            self.last[key] = override
            return override
        if key == "data" and self.mode == "legacy":
            self.last[key] = 1.0          # historical behaviour: L_data stays raw
            return 1.0
        if self._frozen[key] is not None:
            self.last[key] = self._frozen[key]
            return self._frozen[key]

        prev = self._ema[key]
        den = value if prev is None else prev
        if np.isfinite(value):
            self._ema[key] = (value if prev is None
                              else self.decay * prev + (1.0 - self.decay) * value)
        if not np.isfinite(den) or den <= 0.0:
            den = 1.0
        if key == "data":
            den = max(den, self.data_floor)
        self.last[key] = den
        return den

    def end_step(self) -> None:
        """Advance the step counter and, in ``fixed`` mode, freeze after warm-up."""
        self._steps += 1
        if self.mode != "fixed" or self._steps < self.warmup_steps:
            return
        for key in self.KEYS:
            if self._frozen[key] is None and self._ema[key] is not None:
                floor = self.data_floor if key == "data" else 1e-30
                self._frozen[key] = max(self._ema[key], floor)


def fit(args):
    """Train on ``args.ops`` and return ``(model, bundle, ops_packed, dtn, history)``."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    enable_tf32(getattr(args, "tf32", False))

    bundle = load_ops(
        op_ids=args.ops, subsample_time=args.subsample,
        train_frac=float(getattr(args, "train_frac", 0.8)),
        resample=str(getattr(args, "resample", "mean")),
        driver_rate_lags=[float(v) for v in getattr(args, "driver_rate_lags", [])],
        use_driver_history=bool(getattr(args, "driver_history", True)),
    )
    ops = _to_tensor_ops(bundle, device)
    _check_finite_inputs(ops)
    # Optional extra input features (default OFF: the richer features empirically
    # hurt the free-running rollout on this data). Zero-width when disabled.
    n_static = bundle.n_static if args.use_static else 0
    n_forcing = bundle.n_forcing if args.use_forcing else 0
    for op in ops:
        if not args.use_static:
            op["static"] = op["static"][:, :0]
        if not args.use_forcing:
            op["forcing"] = op["forcing"][:, :0]
    dtn = ops[0]["dtn"]
    dt_s = dtn * bundle.T_span_ref
    _check_cfl_stability(bundle, dt_s, device)
    phys_scale = bundle.phys_scale
    rate_lags_s = [float(v) for v in getattr(args, "rate_lags", [])]
    rate_lags_n = [v / bundle.T_span_ref for v in rate_lags_s]
    delta_grid_s = float(getattr(args, "delta_grid", 0.0)) or dt_s
    delta_grid_n = delta_grid_s / bundle.T_span_ref
    if args.history_mode == "hybrid" and delta_grid_s < dt_s - 1e-9:
        # The anchor would sit between two samples that the rollout has not
        # produced yet, so the lookup clamps back to the last available step --
        # silently making delta_grid behave as if it were dt_s.
        print(
            f"  [WARN] --delta-grid {delta_grid_s:g}s is below the data step "
            f"{dt_s:g}s; the anchor cannot resolve finer than the grid and will "
            f"effectively act as {dt_s:g}s.",
            flush=True,
        )
    # What the pooled normalisation actually came out at, which OPs really carry
    # a profile (the plan sheet is a transcription and can be wrong), and the
    # amplification A. All three are cheap and all three have been the thing a
    # failed run turned out to hinge on.
    print("\n".join(normalisation_report(bundle)), flush=True)
    print("\n".join(profile_report(bundle)), flush=True)
    print(
        f"OPs={args.ops} n_config={bundle.n_config} n_static={n_static} "
        f"n_forcing={n_forcing} dtn={dtn:.4g} "
        f"phys_scale={phys_scale:.4g} dTdt_scale={bundle.dTdt_scale:.4g} "
        f"aniso_scale={bundle.aniso_scale:.4g} Qsrc_scale={bundle.Qsrc_scale:.4g} "
        f"bc_scale={bundle.bc_scale:.4g} T_sigma={bundle.T_sigma:.3f} "
        f"time_deriv={args.time_deriv} history_mode={args.history_mode} "
        f"rate_lags_s={rate_lags_s}",
        flush=True,
    )

    # The hybrid history divides a temperature DIFFERENCE by ``lag_n *
    # rate_scale``. That product is tiny (5 s out of a ~1474 s reference span),
    # so the channel multiplies whatever the previous step produced by A ~ 119
    # and feeds it back in. Print A always; --max-rate-amp caps it by raising
    # rate_scale, at the cost of no longer being the same model.
    rate_scale = bundle.dTdt_scale
    if args.history_mode == "hybrid":
        # Pooling OP01-OP16 widens T_sigma with between-OP offset that
        # contributes nothing to any single OP's own rate, which SHRINKS
        # dTdt_scale and so RAISES A. Watch the line this prints: the base
        # project's measured 119/30 was on OP01-OP05 only.
        rate_scale, amp_lines = effective_rate_scale(
            bundle.dTdt_scale, rate_lags_n, float(getattr(args, "max_rate_amp", 0.0))
        )
        for line in amp_lines:
            print(line, flush=True)

    model = RecurrentField(
        n_config=bundle.n_config, n_static=n_static, n_forcing=n_forcing,
        k_max=args.k_max, history_mode=args.history_mode, rate_lags=rate_lags_n,
        layer_size=args.width, num_layers=args.depth,
        delta_seconds=1.0, dtn=dtn, t_span_ref=bundle.T_span_ref,
        rate_scale=rate_scale, delta_grid=delta_grid_n,
        use_autograd_time=(args.time_deriv == "autograd"),
        residual_output=bool(getattr(args, "residual_output", False)),
        learn_gains=bool(getattr(args, "learn_gains", False)),
    ).to(device)
    # src_gain / diff_gain used to carry a ~100x scale gap that ``physics.py``
    # created itself, by dividing each residual term by a different RMS. The
    # residual is now assembled in its own consistent units and divided by one
    # scale, so there is no gap left for the gains to close: they stay pinned at
    # 1.0 and never reach the optimiser. --learn-gains restores the old free
    # gains (and with them their own high-LR group, since at the base LR they
    # barely move at all).
    gain_params = [p for p in (model.log_src_gain, model.log_diff_gain)
                   if isinstance(p, torch.nn.Parameter)]
    gain_ids = {id(p) for p in gain_params}
    base_params = [p for p in model.parameters() if id(p) not in gain_ids]
    gain_lr_mult = float(getattr(args, "gain_lr_mult", 25.0))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    groups = [{"params": base_params, "lr": args.lr, "weight_decay": weight_decay}]
    if gain_params:
        groups.append(
            {"params": gain_params, "lr": args.lr * gain_lr_mult, "weight_decay": 0.0}
        )
    opt = torch.optim.Adam(groups)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model params={n_params} k_max={model.k_max} (fixed) "
        f"delta=1.0s = {float(model.delta):.4g} normalised (fixed) "
        f"delta_grid={delta_grid_s:g}s gates=all-on "
        f"history_mode={model.history_mode} rate_lags_s={rate_lags_s} "
        f"width={args.width} depth={args.depth}",
        flush=True,
    )

    # Create boundary condition mask (x ≈ 0 for cell center)
    bc_mask = torch.tensor(np.abs(bundle.xn[:, 0]) < 1e-6, dtype=torch.bool, device=device)
    n_bc = int(bc_mask.sum().item())
    print(f"BC points (x=0): {n_bc}/{len(bc_mask)}", flush=True)
    # The BC term is silent about its own failure modes, and both of them look
    # like an ordinary run in the loss curve.
    #
    # ``boundary_condition_loss`` returns a bare 0.0 when the mask is empty, so
    # L_bc is then identically zero, its EMA divisor collapses, and L_bc_bal is
    # 0 for the whole run -- which fails the "balanced ~ O(1)" check every time
    # without anything in the log saying the BC was never evaluated.
    if n_bc == 0 and args.w_bc != 0.0:
        print("  [WARN] the BC mask is EMPTY: no grid point satisfies "
              "|xn[:, 0]| < 1e-6, so L_bc is identically 0 and w_bc buys "
              "nothing. xn = (xyz - xyz_min) / L_ref puts the minimum x plane "
              "at exactly 0, so an empty mask means the coordinates did not "
              "come through that transform.", flush=True)
    # And it samples min(n_bc, batch_bc) points, so a thin boundary plane
    # silently shrinks the BC batch rather than sampling it repeatedly.
    batch_bc = int(getattr(args, "batch_bc", 128))
    if 0 < n_bc < batch_bc and args.w_bc != 0.0:
        print(f"  [WARN] only {n_bc} boundary points against --batch-bc "
              f"{batch_bc}: the BC term is estimated from {n_bc} samples per "
              f"step, not {batch_bc}. Its gradient is correspondingly noisier "
              f"than the weight suggests.", flush=True)

    # One series per HISTORY_KEYS entry; see that constant for what each is and
    # why the list lives at module scope. The EMA horizon behind div_* is
    # ~1/(1-ema_decay) EPOCHS, so a term that drops by orders of magnitude inside
    # that horizon reports a ratio far below 1 with nothing wrong at all.
    history: dict = {key: [] for key in HISTORY_KEYS}
    history["aborted"] = False
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0))

    # Optimiser steps taken per OP per epoch, all against the one frozen rollout
    # that epoch computed. The rollout is what costs time (~7000 sequential steps
    # that cannot be parallelised); a minibatch step is cheap, so this is where the
    # update count comes from.
    inner_steps = max(1, int(getattr(args, "inner_steps", 100)))

    rollout_clamp = float(getattr(args, "rollout_clamp", 50.0) or 0.0)
    if rollout_clamp > 0.0:
        print(f"rollout saturation guard: |Tn| <= {rollout_clamp:g}", flush=True)
    batch_data = int(getattr(args, "batch_data", 2048))
    print(
        f"optimiser steps = {inner_steps} per OP per epoch x {len(ops)} OPs "
        f"x {args.epochs} epochs = {inner_steps * len(ops) * args.epochs} total "
        f"(batch_data={batch_data})",
        flush=True,
    )

    # Loss balancing. Each enabled term is divided by an estimate of its own
    # magnitude so w_data:w_phys:w_bc is a ratio between TERMS and not between
    # their accidental units. --phys-norm/--bc-norm override the corresponding
    # divisor with a constant in any mode, which is also how "fixed" is expressed
    # once warm-up has ended.
    residual_norm = str(getattr(args, "residual_norm", "rms"))
    balance_mode = str(getattr(args, "loss_balance", "ema"))
    if balance_mode not in ("ema", "legacy", "fixed"):
        raise SystemExit(f"unknown --loss-balance {balance_mode!r} "
                         f"(expected ema|legacy|fixed)")
    ema_decay = float(getattr(args, "ema_decay", 0.9))
    if not 0.0 <= ema_decay < 1.0:
        raise SystemExit(f"--ema-decay must be in [0, 1), got {ema_decay}")
    # The EMA is updated once per optimiser step, so its horizon would silently
    # depend on how many steps an epoch happens to contain: 1/(1-decay) steps is
    # ~2 epochs at 5 OPs but ~10 epochs at one, and --inner-steps multiplies that
    # again. Correcting the per-step decay by the steps an epoch actually takes
    # makes the horizon what it claims to be -- a number of EPOCHS -- for any
    # --ops and any --inner-steps.
    n_ops = max(1, len(ops))
    steps_per_epoch = n_ops * inner_steps
    step_decay = ema_decay ** (1.0 / steps_per_epoch)
    warmup_steps = int(getattr(args, "balance_warmup", 0)) * steps_per_epoch
    phys_norm = float(getattr(args, "phys_norm", 0.0))
    bc_norm = float(getattr(args, "bc_norm", 0.0))
    # Floor for the data divisor: once L_data approaches zero, dividing by it
    # would amplify its gradient without bound. The floor turns the data term
    # back into an ordinary MSE long before that happens.
    data_floor = float(getattr(args, "data_floor", 1e-8))
    balance = _LossBalancer(
        mode=balance_mode, decay=step_decay, warmup_steps=warmup_steps,
        phys_norm=phys_norm, bc_norm=bc_norm, data_floor=data_floor,
    )
    print(
        f"loss balance: mode={balance_mode} ema_decay={ema_decay:g}/epoch "
        f"(={step_decay:.6g}/step over {steps_per_epoch} steps/epoch) "
        f"phys_norm={phys_norm:g} bc_norm={bc_norm:g} "
        f"residual_norm={residual_norm}",
        flush=True,
    )

    # A term with weight 0 contributes nothing but still costs a full forward
    # pass plus an autograd Hessian -- that is most of the runtime of the
    # w_phys=0 / w_bc=0 sweep points. Skipped terms are logged as NaN rather than
    # 0.0, so the convergence plots show a gap instead of a flat line that never
    # happened. Constant for the whole run, hence hoisted out of both loops.
    skip_zero_terms = str(getattr(args, "zero_weight_terms", "skip")) == "skip"
    want_phys = args.w_phys != 0.0 or not skip_zero_terms
    want_bc = args.w_bc != 0.0 or not skip_zero_terms
    if not (want_phys and want_bc):
        skipped = [n for n, w in (("L_phys", want_phys), ("L_bc", want_bc)) if not w]
        print(f"zero-weight terms skipped, logged as NaN: {', '.join(skipped)}",
              flush=True)

    # Its own generator, so toggling --shuffle-ops does not shift the sampling
    # stream that draws the minibatches and thereby change the run for a reason
    # that has nothing to do with the OP order.
    order_rng = np.random.default_rng(int(args.seed))
    shuffle_ops = bool(getattr(args, "shuffle_ops", True))
    holdout_tail = bool(getattr(args, "holdout_tail", False))
    print(f"OP order per epoch: {'shuffled' if shuffle_ops else 'fixed'}   "
          f"training rollout: {'truncated at split_t' if holdout_tail else 'full'}",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        ep_data, ep_phys, ep_bc = 0.0, 0.0, 0.0
        ep_ratio_phys, ep_ratio_bc = 0.0, 0.0
        ep_spread_space, ep_spread_time = 0.0, 0.0
        # Split the epoch's wall time into its two halves. Every runtime estimate
        # in README_GPU_SERVER chapters 7 and 8 is derived from one measured
        # seconds-per-epoch, and --inner-steps moves only the second half, so the
        # split is what makes the budget plannable instead of guessed.
        t_roll_s, t_inner_s = 0.0, 0.0
        ep_saturated: list[tuple[str, int, int]] = []
        aborted_epoch = False
        order = (order_rng.permutation(len(ops)) if shuffle_ops
                 else np.arange(len(ops)))
        for oi in order:
            op = ops[int(oi)]
            n_t, n_pts = op["n_t"], op["n_points"]
            # With --holdout-tail the training rollout stops at split_t, so the
            # late window is never fitted and op_metrics' `late` column becomes a
            # real held-out number instead of a relabelled training error.
            t_end = max(int(op["split_t"]), 2) if holdout_tail else n_t
            Tn_seq = op["Tn"]

            # ---- the model's OWN trajectory, computed ONCE per OP per epoch ---
            # No teacher forcing: this is the free-running rollout, seeded only by
            # the measured initial condition.
            #
            # Why it is frozen and then reused. The recurrence always detached its
            # history between steps (truncated BPTT), so the gradient at time t
            # never left that step's own field evaluation even before this change.
            # The gradient of the old full-sequence L_data is therefore EXACTLY a
            # sum of independent per-(t, point) gradients against a trajectory the
            # weights are treated as constant in -- which is the same quantity a
            # minibatch of (t, point) pairs estimates without bias.
            #
            # What that buys: the old loop paid one ~7000-step sequential rollout
            # for ONE optimiser step, so a 60-epoch run on 5 OPs finished after
            # 300 Adam updates -- far too few for a 70k-parameter MLP, which is the
            # single biggest reason the rollout error stayed large. Now the same
            # rollout carries ``inner_steps`` updates.
            #
            # The cost of freezing: after a few updates the buffer is no longer
            # quite the trajectory the current weights would produce. It is
            # refreshed every epoch, so inner_steps trades update count against
            # how stale the trajectory may get -- keep it in the hundreds, not the
            # tens of thousands.
            _t0 = time.time()
            with torch.no_grad():
                own_hist = rollout(
                    model, op["xn"], op["static"], op["cfg"], op["forcing"],
                    op["Tn_ic"], op["tn"][:t_end], dtn, clamp=rollout_clamp,
                )
            t_roll_s += time.time() - _t0
            # The guard binding means the rollout tried to run away and was held
            # back -- the loss stays finite, but the trajectory is not a
            # prediction any more. Silence here would look like ordinary slow
            # convergence, so it is counted and reported.
            if rollout_clamp > 0.0:
                n_sat = int((own_hist.abs() >= rollout_clamp).any(dim=1).sum())
                if n_sat:
                    ep_saturated.append((op["op_id"], n_sat, own_hist.shape[0]))

            # How much structure is left in the model's own trajectory, against
            # how much the labels carry. Both residual terms vanish identically
            # on a field that is constant in space (Laplacian 0, dT/dx 0) and in
            # time (dT/dt 0), so a physics loss falling towards zero is only
            # evidence of physics while these ratios stay near 1. Near 0 means
            # the optimiser found the trivial solution instead, which no loss
            # curve in this run would show.
            with torch.no_grad():
                # Against the SAME window the rollout covers: with
                # --holdout-tail own_hist stops at split_t, and comparing its
                # spread to the labels' over the whole trajectory would report a
                # ratio below 1 for a reason that has nothing to do with a flat
                # field -- which is the one thing this diagnostic exists to say.
                lab = Tn_seq[:own_hist.shape[0]]
                s_pred = float(own_hist.std(dim=1).mean())
                s_lab = float(lab.std(dim=1).mean())
                t_pred = float(own_hist.std(dim=0).mean())
                t_lab = float(lab.std(dim=0).mean())
                ep_spread_space += s_pred / (s_lab + 1e-12)
                ep_spread_time += t_pred / (t_lab + 1e-12)
            _t0 = time.time()

            op_data = op_phys = op_bc = 0.0
            op_ratio_phys = op_ratio_bc = 0.0
            # Placeholder for a term that is switched off: NaN, not 0.0, so the
            # convergence plot shows a gap instead of a flat line that never
            # happened.
            nan = torch.tensor(float("nan"), device=device)
            for _ in range(inner_steps):
                # ---- data term on a minibatch of (t, point) ------------------
                # Labels are only read up to split_t. The rollout still covers the
                # whole trajectory -- the recurrence needs it, and the physics and
                # BC terms below are unsupervised and use all of it -- but fitting
                # LABELS past split_t would train on the very rows that
                # ``metrics.txt`` and the benchmarks' MAE_in column report as a
                # held-out in-time check. data.py already treats that tail as
                # held out: T_mu/T_sigma and every config/source statistic are
                # pooled over [:split_t] only.
                # t starts at 1: row 0 is the imposed initial condition, never a
                # prediction.
                bt = torch.randint(1, op["split_t"], (batch_data,), device=device)
                bp = torch.randint(0, n_pts, (batch_data,), device=device)
                tq = op["tn"][bt]
                hist = model._history(own_hist, dtn, tq, bp)
                pred = model.field(
                    op["xn"][bp], op["static"][bp], op["cfg"][bt],
                    op["forcing"][bt], hist, model.level(own_hist, dtn, tq),
                )
                L_data = torch.mean((pred - Tn_seq[bt, bp]) ** 2)

                # ---- physics term (autograd space + FD time) -----------------
                # History for the residual comes from the same frozen rollout.
                if want_phys:
                    pt = torch.randint(0, t_end, (args.batch_phys,), device=device)
                    pp = torch.randint(0, n_pts, (args.batch_phys,), device=device)
                    res = heat_residual(
                        model, op["xn"], op["static"], op["cfg"][pt], op["forcing"][pt],
                        op["Fo"], op["Qsrc"][pt, pp], own_hist, dtn, op["tn"][pt], pp,
                        phys_scale, time_deriv=args.time_deriv,
                        residual_norm=residual_norm,
                    )
                    L_phys = torch.mean(res ** 2)
                else:
                    L_phys = nan

                # ---- boundary condition term (dT/dx = 0 at x=0) --------------
                if want_bc:
                    pt_bc = torch.randint(0, t_end, (args.batch_bc,), device=device)
                    bc_res = boundary_condition_loss(
                        model, op["xn"], op["static"], op["cfg"][pt_bc],
                        op["forcing"][pt_bc], own_hist, dtn, op["tn"][pt_bc],
                        bc_mask, bundle.bc_scale, residual_norm=residual_norm,
                    )
                    L_bc = torch.mean(bc_res ** 2)
                else:
                    L_bc = nan

                # Every enabled term is divided by an estimate of its own
                # magnitude, so w_data:w_phys:w_bc is a ratio between terms and
                # not between their units. Which terms get divided depends on
                # --loss-balance; the divisor is always the estimate from BEFORE
                # this step, and a non-finite sample never enters it.
                L_data_bal = L_data / balance.divisor("data", float(L_data.detach()))
                L_phys_bal = (L_phys / balance.divisor("phys", float(L_phys.detach()))
                              if want_phys else nan)
                L_bc_bal = (L_bc / balance.divisor("bc", float(L_bc.detach()))
                            if want_bc else nan)
                balance.end_step()

                # Only add terms that are actually switched on: ``0.0 * nan`` is
                # nan, so a zero weight does NOT neutralise a non-finite term -- it
                # would poison the whole loss, the gradients, and every later
                # epoch. This is what made even the w_phys=0, w_bc=0 sweep point
                # report L_data=nan.
                loss = args.w_data * L_data_bal
                if args.w_phys != 0.0:
                    loss = loss + args.w_phys * L_phys_bal
                if args.w_bc != 0.0:
                    loss = loss + args.w_bc * L_bc_bal

                # Never let a non-finite loss reach the optimiser: clip_grad_norm_
                # does not rescue it (total_norm=nan -> clip_coef=nan -> all grads
                # nan), so one bad step would permanently destroy the weights.
                if not torch.isfinite(loss):
                    # Only terms that were actually computed can be to blame; a
                    # term skipped for having weight 0 is NaN on purpose and must
                    # not be reported as the cause.
                    candidates = [("L_data", L_data)]
                    if want_phys:
                        candidates.append(("L_phys", L_phys))
                    if want_bc:
                        candidates.append(("L_bc", L_bc))
                    bad = [n for n, v in candidates if not torch.isfinite(v)]
                    print(
                        f"  [ABORT] epoch {epoch}, {op['op_id']}: non-finite loss; "
                        f"first offending term(s): {', '.join(bad) or 'weighted sum'}",
                        flush=True,
                    )
                    ep_data = float("nan")
                    aborted_epoch = True
                    break

                opt.zero_grad()
                loss.backward()
                grad_clip = float(getattr(args, "grad_clip", 0.0))
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
                op_data += float(L_data.detach())
                op_phys += float(L_phys.detach())
                op_bc += float(L_bc.detach())
                # Ratio each weighted term actually contributes against the data
                # term. This is the number the loss weights are really setting,
                # and without it the balance can only be guessed at from w_phys
                # alone. ``benchmark_balance.py`` reads exactly this series.
                base = args.w_data * float(L_data_bal.detach())
                if np.isfinite(base) and base > 0.0:
                    if want_phys and args.w_phys != 0.0:
                        op_ratio_phys += args.w_phys * float(L_phys_bal.detach()) / base
                    if want_bc and args.w_bc != 0.0:
                        op_ratio_bc += args.w_bc * float(L_bc_bal.detach()) / base

            t_inner_s += time.time() - _t0
            if aborted_epoch:
                break
            # Mean over the inner steps, so the logged epoch loss stays a
            # per-step number and does not change meaning with --inner-steps.
            ep_data += op_data / inner_steps
            ep_phys += op_phys / inner_steps
            ep_bc += op_bc / inner_steps
            ep_ratio_phys += op_ratio_phys / inner_steps
            ep_ratio_bc += op_ratio_bc / inner_steps

        if ep_saturated:
            worst = ", ".join(
                f"{op_id} {n}/{tot} steps" for op_id, n, tot in ep_saturated
            )
            print(f"  [SATURATED] epoch {epoch}: rollout hit the "
                  f"|Tn| <= {rollout_clamp:g} guard ({worst}). The loss stays "
                  f"finite, but the trajectory ran away and was held back -- it "
                  f"is not a prediction, and a run that only survives because "
                  f"of this is not trained. Watch the count: falling is the "
                  f"model pulling itself together, flat or rising is not.",
                  flush=True)

        ep_ratio_phys /= len(ops)
        ep_ratio_bc /= len(ops)
        ep_data /= len(ops)
        ep_phys /= len(ops)
        ep_bc /= len(ops)
        ep_spread_space /= len(ops)
        ep_spread_time /= len(ops)

        # The trivial solution: a field flat in space and time satisfies the
        # heat residual and the Neumann BC exactly, so both physics losses can
        # be driven to zero by giving up on the data. That failure is invisible
        # in every curve this run plots -- L_phys falling looks like success --
        # which is why it gets its own line the moment the rollout carries less
        # than a fifth of the structure the labels do.
        # Not on an aborted epoch: the sum then covers only the OPs that ran but
        # is still divided by all of them, so the ratio is low for a reason that
        # has nothing to do with a flat field.
        if (want_phys or want_bc) and not aborted_epoch:
            flat = [name for name, val in (("space", ep_spread_space),
                                           ("time", ep_spread_time)) if val < 0.2]
            if flat:
                print(f"  [FLAT] epoch {epoch}: the rollout carries "
                      f"{ep_spread_space:.2g}x the labels' spatial spread and "
                      f"{ep_spread_time:.2g}x their temporal spread. A field "
                      f"that is constant in {' and '.join(flat)} satisfies the "
                      f"residual and the BC for free, so a falling L_phys/L_bc "
                      f"here is the trivial solution, not physics. Lower "
                      f"--w-phys/--w-bc, or check that the data term is "
                      f"reaching the optimiser at all.", flush=True)

        # Early NaN/inf detection - abort before wasting epochs. A term that was
        # deliberately skipped is NaN on purpose and must not trigger this.
        phys_broken = want_phys and not np.isfinite(ep_phys)
        if not np.isfinite(ep_data) or phys_broken:
            print(f"  [ABORT] epoch {epoch}: loss exploded (L_data={ep_data:.4g}, L_phys={ep_phys:.4g})")
            print("  L_data non-finite means the rollout that feeds it diverged.")
            if epoch == 1:
                print("  This is epoch 1, so it happened BEFORE any optimiser step: the")
                print("  untrained network's own output is what the history channels fed")
                print("  back, and training cannot learn its way out of a NaN it starts in.")
            if bool(getattr(args, "residual_output", False)):
                print("  --residual-output is ON, and that is the first thing to turn off.")
                print("  T(t) = level(t) + net(...) carries the level through an integrator")
                print("  of gain exactly 1 with no leak, so any one-signed component of the")
                print("  network output accumulates over the ~7000 steps without bound.")
                print("  Measured on a synthetic bundle it aborts on every seed in EVERY")
                print("  history configuration, raw included; --no-residual-output does not.")
            print("  The hybrid rate channel is the usual amplifier -- see the A = ... line")
            print("  at startup; A above ~100 means a one-step level jump comes back into")
            print("  the net magnified that many times (--history-mode raw isolates it).")
            print("  L_phys non-finite alone points at the residual: --time-deriv bdf1 or --w-phys 0.")
            print("  Also worth trying: --grad-clip 1.0, a lower --lr, or a larger --subsample.")
            # Record the failed epoch so callers never see an empty history and
            # the divergence stays visible downstream (CSV, plots) as NaN.
            history["epoch"].append(epoch)
            history["L_data"].append(ep_data)
            history["L_phys"].append(ep_phys)
            history["L_bc"].append(ep_bc)
            history["L_phys_bal"].append(float("nan"))
            history["L_bc_bal"].append(float("nan"))
            history["ratio_phys"].append(float("nan"))
            history["ratio_bc"].append(float("nan"))
            # Every series stays the same length as history["epoch"], or the
            # plots and CSV writers downstream silently misalign an aborted run
            # by one row. The divisors are real values and worth keeping; the
            # spreads describe the epoch that just blew up, so they go in too.
            history["div_data"].append(balance.last["data"])
            history["div_phys"].append(balance.last["phys"] if want_phys else float("nan"))
            history["div_bc"].append(balance.last["bc"] if want_bc else float("nan"))
            history["spread_space"].append(ep_spread_space)
            history["spread_time"].append(ep_spread_time)
            history["delta"].append(float(model.delta.detach()))
            history["aborted"] = True
            break

        # Balanced losses for fair logging (all ~O(1) when stable). They use the
        # divisors the balancer last handed out, so the logged numbers are the
        # ones the optimiser actually saw -- not a second, differently computed
        # estimate that would drift away from them under --loss-balance.
        ep_data_bal = ep_data / balance.last["data"]
        ep_phys_bal = ep_phys / balance.last["phys"] if want_phys else float("nan")
        ep_bc_bal = ep_bc / balance.last["bc"] if want_bc else float("nan")

        history["epoch"].append(epoch)
        history["L_data"].append(ep_data)
        history["L_phys"].append(ep_phys)
        history["L_bc"].append(ep_bc)
        history["L_phys_bal"].append(ep_phys_bal)
        history["L_bc_bal"].append(ep_bc_bal)
        history["ratio_phys"].append(ep_ratio_phys if want_phys else float("nan"))
        history["ratio_bc"].append(ep_ratio_bc if want_bc else float("nan"))
        history["div_data"].append(balance.last["data"])
        history["div_phys"].append(balance.last["phys"] if want_phys else float("nan"))
        history["div_bc"].append(balance.last["bc"] if want_bc else float("nan"))
        history["spread_space"].append(ep_spread_space)
        history["spread_time"].append(ep_spread_time)
        history["delta"].append(float(model.delta.detach()))
        # The two append blocks above and in the abort path are written out by
        # hand, so a series added to HISTORY_KEYS and forgotten in one of them
        # would misalign the CSV and the plots by a row without any error.
        assert all(len(history[k]) == epoch for k in HISTORY_KEYS), \
            "a HISTORY_KEYS series was not appended this epoch"
        if ep_data < best_train_loss:
            best_train_loss = ep_data
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            betas = np.round(model.mlp.betas(), 2)
            sg = float(model.src_gain.detach())
            dg = float(model.diff_gain.detach())
            # Log learned rate_lags (hybrid mode only)
            if hasattr(model, '_raw_rate_lags'):
                learned_lags = np.round(model.rate_lags.detach().cpu().numpy(), 2)
                lags_str = f"  rate_lags={learned_lags}"
            else:
                lags_str = ""
            # Show BALANCED losses (all ~O(1)) so user can compare fairly
            # Per-epoch wall time and the resulting projection: a benchmark
            # extrapolates its ETA from whole grid points, which is useless while
            # the first one is still running -- and badly misleading if a run
            # aborts early, because then the "per point" time is one epoch, not
            # all of them.
            epoch_s = time.time() - epoch_start
            eta_min = (args.epochs - epoch) * epoch_s / 60.0
            print(
                f"  epoch {epoch:3d}  L_data={ep_data:.4e}  L_data_bal={ep_data_bal:.4e}  "
                f"L_phys_bal={ep_phys_bal:.4e}  L_bc_bal={ep_bc_bal:.4e}  "
                f"ratio phys/bc={ep_ratio_phys:.3g}/{ep_ratio_bc:.3g}  "
                f"spread s/t={ep_spread_space:.3g}/{ep_spread_time:.3g}  "
                f"delta={float(model.delta.detach()):.4g}  "
                f"src_gain={sg:.3g}  diff_gain={dg:.3g}{lags_str}  betas={betas}  "
                f"[{epoch_s:.1f}s/epoch = {t_roll_s:.1f}s rollout + "
                f"{t_inner_s:.1f}s x{inner_steps} inner, "
                f"this run ~{eta_min:.0f} min left]",
                flush=True,
            )
            if epoch == 1 and device.type == "cuda":
                # Peak VRAM, once. The batch sizes are the only real GPU knob here
                # (see README_GPU_SERVER 6.4) and guessing how much headroom is
                # left is exactly the thing a measurement should answer.
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                total = torch.cuda.get_device_properties(device).total_memory / 1e9
                print(
                    f"  peak VRAM {peak:.2f} GB of {total:.1f} GB "
                    f"(batch_data={batch_data} batch_phys={args.batch_phys} "
                    f"batch_bc={args.batch_bc})",
                    flush=True,
                )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"  early stopping after epoch {epoch}: training data loss did not improve "
                f"for {early_stopping_patience} epochs",
                flush=True,
            )
            break

    return model, bundle, ops, dtn, history


def save_checkpoint(model, bundle, args, dtn, history, path: Path) -> None:
    """Write the trained weights plus everything needed to use them again.

    Until 31.08.2026 the ONLY ``torch.save`` in this project lived in
    ``bench_common.py``, so a plain ``train.py`` run left the finished model in
    RAM and nothing else -- a multi-hour run whose result evaporated. Deleting
    the benchmarks took that with it, hence this.

    A bare ``state_dict`` is not enough to reload: ``RecurrentField`` needs its
    layout (widths, ``k_max``, history mode, the lags in NORMALISED time) and
    turning a prediction back into degrees C needs ``T_mu``/``T_sigma``. Both
    travel in the file, so the checkpoint does not depend on ``config.yaml``
    still saying what it said at training time.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # Exactly the RecurrentField constructor arguments, so reloading is
            # RecurrentField(**ckpt["model_config"]) followed by load_state_dict.
            "model_config": {
                "n_config": bundle.n_config,
                "n_static": model.n_static,
                "n_forcing": model.n_forcing,
                "k_max": int(args.k_max),
                "history_mode": args.history_mode,
                "rate_lags": [float(v) / bundle.T_span_ref
                              for v in getattr(args, "rate_lags", [])],
                "layer_size": int(args.width),
                "num_layers": int(args.depth),
                "delta_seconds": 1.0,
                "dtn": float(dtn),
                "t_span_ref": float(bundle.T_span_ref),
                "rate_scale": float(model.rate_scale),
                "delta_grid": float(model.delta_grid),
                "use_autograd_time": (args.time_deriv == "autograd"),
                "residual_output": bool(model.residual_output),
                "learn_gains": bool(model.learn_gains),
            },
            # T_pred_C = Tn * T_sigma + T_mu. Without these the weights predict
            # a z-score against a normalisation nothing recorded.
            "bundle_stats": {
                "T_mu": float(bundle.T_mu),
                "T_sigma": float(bundle.T_sigma),
                "T_span_ref": float(bundle.T_span_ref),
                "L_ref": float(bundle.L_ref),
                "phys_scale": float(bundle.phys_scale),
                "dTdt_scale": float(bundle.dTdt_scale),
            },
            # The preprocessing is part of the model in everything but name:
            # a checkpoint replayed with a different --resample sees driver
            # channels built by a different rule than the ones it trained on.
            "preprocessing": {
                "resample": str(getattr(args, "resample", "mean")),
                "driver_rate_lags": [float(v) for v in bundle.driver_rate_lags],
                "use_driver_history": bool(bundle.use_driver_history),
                "train_frac": float(getattr(args, "train_frac", 0.8)),
            },
            "run": {
                "ops": list(args.ops),
                "val_ops": list(getattr(args, "val_ops", []) or []),
                "test_ops": list(getattr(args, "test_ops", []) or []),
                "measurement_ops": list(getattr(args, "measurement_ops", []) or []),
                "holdout_tail": bool(getattr(args, "holdout_tail", False)),
                "epochs": int(args.epochs),
                "epochs_run": len(history["epoch"]),
                "aborted": bool(history.get("aborted", False)),
                "subsample": int(args.subsample),
                "seed": int(args.seed),
                "synthetic_cache": cache_is_synthetic(),
            },
        },
        path,
    )


def trivial_baselines(op, bundle) -> tuple[float, float]:
    """MAE of the two trivial predictors on ``op``, in physical degrees C.

    Without these a MAE is a number with no scale. The model has to beat the
    BETTER of the two to have learned anything at all:

    * persistence -- ``T(t) = T(0)``: the field never changes.
    * mean -- the constant mean of the training labels. That constant IS
      ``bundle.T_mu`` by construction (``data.py`` pools it over the training
      portion of the training OPs), so this is not a second definition of the
      same quantity.

    Computed on whatever ``op`` is actually in hand, never quoted from an
    earlier run: the magnitudes do not transfer between OP sets, and a flat
    held-out OP has a persistence baseline of 0 C, which is unbeatable rather
    than merely hard.
    """
    persistence = float(np.abs(op.T_lab - op.T_lab[0][None, :]).mean())
    mean = float(np.abs(op.T_lab - bundle.T_mu).mean())
    return persistence, mean


def train(args) -> None:
    # Resolve the OPs the run DEPENDS on before training, not after: a typo in
    # --val-ops would otherwise cost the whole run before it surfaces.
    require_ops(*args.ops, *getattr(args, "val_ops", []),
                *getattr(args, "test_ops", []))
    # --measurement-ops is deliberately NOT in that list. It is a bonus report,
    # not part of the evaluation: OP17/OP18 have not been simulated yet, so a
    # config that names them must degrade to a warning rather than refuse to
    # train. A missing measurement bundle is a fact about the simulation
    # backlog; it is never a reason to abort a GPU run.
    meas = list(getattr(args, "measurement_ops", []) or [])
    if meas:
        have = set(available_ops())
        missing = [op for op in meas if op not in have]
        if missing:
            print(f"  [SKIP] measurement OP(s) with no cached bundle: "
                  f"{', '.join(missing)}. Not simulated yet, or not built with "
                  f"generate_cache.py. Training and the val/test report are "
                  f"unaffected.", flush=True)
            args.measurement_ops = [op for op in meas if op in have]
    if cache_is_synthetic():
        print("=" * 72, flush=True)
        print("  *** SYNTHETIC DATA CACHE (tools/make_synthetic_cache.py) ***",
              flush=True)
        print("  Every MAE below is measured on a fixture, not on the simulation.",
              flush=True)
        print("  The numbers are usable for comparing two runs against each other",
              flush=True)
        print("  and for nothing else. Do not quote them as a result.", flush=True)
        print("=" * 72, flush=True)
    model, bundle, ops, dtn, history = fit(args)
    device = next(model.parameters()).device
    # Before evaluate(), not after: evaluation rolls out every OP and a
    # non-finite trajectory there must not cost the weights that produced it.
    name = str(getattr(args, "save_checkpoint", "") or "")
    if name:
        path = ART_DIR / name
        save_checkpoint(model, bundle, args, dtn, history, path)
        print(f"  wrote {path}", flush=True)
    evaluate(model, bundle, ops, dtn, device, history, args)


def _report_op(model, bundle, device, op_data, tier, role, *, late_is_holdout,
               lines, rows, with_coverage):
    """Roll one OP out free-running, print and record every metric for it.

    One function for all four roles (train / val / test / measurement) so a
    metric can never be reported for one group and quietly missing from
    another -- which is exactly how a held-out number ends up being compared
    against an in-sample one.
    """
    pred = rollout_phys(model, op_data, bundle, device)
    if not np.isfinite(pred).all():
        bad = ~np.isfinite(pred).all(axis=1)
        msg = (f"  [DIVERGED] {op_data.op_id}: the eval rollout is non-finite "
               f"from step {int(np.argmax(bad))} on ({int(bad.sum())}/"
               f"{pred.shape[0]} steps). Every metric for this OP is nan. The "
               f"eval rollout is unclamped on purpose -- a saturated trajectory "
               f"is not a prediction, and reporting the clamped MAE would dress "
               f"a diverged model up as a merely bad one.")
        print(msg, flush=True)
        lines.append(msg.strip())
    m = op_metrics(pred, op_data, late_is_holdout=late_is_holdout)
    rows.append((op_data.op_id, tier, role, m))
    print(format_op_metrics(op_data.op_id, tier, m), flush=True)

    # The bar: a MAE has no scale without it, and losing to "the field never
    # changes" means the run learned nothing -- invisible in any loss curve.
    persistence, mean_base = trivial_baselines(op_data, bundle)
    best = min(persistence, mean_base)
    verdict = "beats" if m["mae"] < best else "LOSES TO"
    bar = (f"     baseline: {verdict} the trivial predictors "
           f"(persistence={persistence:.3f} C, train-mean={mean_base:.3f} C)")
    print(bar, flush=True)
    lines.append(f"{op_data.op_id} {bar.strip()}")

    if with_coverage:
        cov = coverage_report(bundle, op_data)
        for line in cov:
            print("     coverage:" + line, flush=True)
        lines.append(f"coverage {op_data.op_id}:")
        lines.extend(cov)

    np.savez_compressed(
        ART_DIR / f"pred_{op_data.op_id}.npz",
        t=np.asarray(op_data.t), T_true=op_data.T_lab, T_pred=pred,
        split_t=op_data.split_t, transient=op_data.transient,
    )
    return op_data


@torch.no_grad()
def evaluate(model, bundle, ops, dtn, device, history, args) -> None:
    """Free-running rollout (NO teacher forcing) on every OP; MAE in physical C.

    Four groups, and the difference between them is the whole point:

    * ``--ops``          -- trained on. In-sample, unless ``--holdout-tail``.
    * ``--val-ops``      -- whole unseen OPs. This is what a hyperparameter
      choice may look at, and it is optimistic by exactly as much as you tuned
      against it.
    * ``--test-ops``     -- whole unseen OPs nothing selected on. Read once.
    * ``--measurement-ops`` -- the mini-module comparison. Not a held-out
      simulation but measured data, so it answers a different question:
      does a model trained on StarCCM+ agree with a real cell? Never trained
      or selected on either way.

    Every group is scored by the same ``op_metrics`` and printed next to the
    two trivial baselines computed on that OP.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    late_is_holdout = bool(getattr(args, "holdout_tail", False))
    subsample = int(getattr(args, "subsample", 2))
    lines = [
        "PINNmodulusTwo -- recurrent Modulus PINN, temperature only",
        "evaluation = FREE-RUNNING ROLLOUT (no teacher forcing)",
        f"history_mode(final) = {model.history_mode}",
        f"rate_lags(final, s) = "
        f"{np.round(model.rate_lags.cpu().numpy() * bundle.T_span_ref, 3).tolist()}",
        f"driver_rate_lags(s) = {list(bundle.driver_rate_lags)} "
        f"({'on' if bundle.use_driver_history else 'off'})",
        f"resample = {bundle.resample}",
        f"rate_scale = {float(model.rate_scale):.5g} "
        f"(dTdt_scale = {bundle.dTdt_scale:.5g})",
        f"delta(final) = {float(model.delta):.5g} (normalised time)",
        f"src_gain(final)  = {float(model.src_gain):.4g}",
        f"diff_gain(final) = {float(model.diff_gain):.4g}",
        f"betas(final) = {np.round(model.mlp.betas(), 3).tolist()}",
        "",
    ]
    lines += normalisation_report(bundle) + [""]
    lines += split_summary(list(args.ops), list(args.val_ops),
                           list(args.test_ops)) + [""]

    rows: list = []
    held: list = []

    print("\ntraining OPs (in-sample unless --holdout-tail):", flush=True)
    for op_data in bundle.ops:
        _report_op(model, bundle, device, op_data, TIER_IN, "train",
                   late_is_holdout=late_is_holdout, lines=lines, rows=rows,
                   with_coverage=False)

    for role, op_ids in (("val", list(getattr(args, "val_ops", []) or [])),
                         ("test", list(getattr(args, "test_ops", []) or []))):
        if not op_ids:
            continue
        print(f"\nheld-out OPs ({role}):", flush=True)
        for op_id in op_ids:
            op_data = build_op(op_id, bundle, subsample_time=subsample)
            held.append(op_data)
            # tier_of raises for an OP outside the plan sheet. The DATA path
            # accepts one happily -- profiles are detected from the bundle, not
            # looked up -- so a new OP is a label problem, never a reason to
            # lose a finished training run.
            # Held out entirely, so the late window is genuinely unseen whatever
            # --holdout-tail was set to.
            _report_op(model, bundle, device, op_data, tier_or_unknown(op_id),
                       role, late_is_holdout=True, lines=lines, rows=rows,
                       with_coverage=True)

    meas = list(getattr(args, "measurement_ops", []) or [])
    if meas:
        print("\nmini-module MEASUREMENT comparison (never trained or selected "
              "on; measured data, not a simulation):", flush=True)
        for op_id in meas:
            op_data = build_op(op_id, bundle, subsample_time=subsample)
            _report_op(model, bundle, device, op_data, "measurement",
                       "measure", late_is_holdout=True, lines=lines, rows=rows,
                       with_coverage=True)

    if not (getattr(args, "val_ops", None) or getattr(args, "test_ops", None)):
        note = ("no held-out OPs evaluated: pass --val-ops / --test-ops. Every "
                "number above is in-sample, which is not a generalisation "
                "estimate.")
        print(f"  [NOTE] {note}", flush=True)
        lines.append(f"NOTE: {note}")

    lines += [""] + profile_report(bundle, held) + [""]
    lines.append(f"{'OP':<6} {'tier':<11} {'role':<8} {'MAE':>8} {'RMSE':>8} "
                 f"{'max':>8} {'peak_err':>9} {'transient':>10} {'quiescent':>10} "
                 f"{'late':>8}")
    for op_id, tier, role, m in rows:
        lines.append(
            f"{op_id:<6} {tier:<11} {role:<8} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
            f"{m['max_abs_err']:>8.3f} {m['peak_err']:>9.3f} "
            f"{m['mae_transient']:>10.3f} {m['mae_quiescent']:>10.3f} "
            f"{m['late_mae']:>8.3f}"
        )
    lines.append("")
    lines.append("MAE/RMSE/max are over the free-running rollout, step 0 excluded "
                 "(it is the imposed IC).")
    lines.append("'transient' = samples where a driver moves faster than its own "
                 "pooled training RMS rate; 'quiescent' = the rest.")
    lines.append(f"'late' = after split_t; for a TRAINING OP that is "
                 f"{'held out (--holdout-tail)' if late_is_holdout else 'IN-SAMPLE'}.")

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(history["epoch"], history["L_data"], label="L_data")
    ax[0].semilogy(history["epoch"], history["L_phys"], label="L_phys (raw)")
    ax[0].semilogy(history["epoch"], history["L_bc"], label="L_bc (raw)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (log scale)")
    ax[0].legend(); ax[0].set_title("raw losses (may span orders of magnitude)")
    ax[1].plot(history["epoch"], history["L_data"], label="L_data")
    ax[1].plot(history["epoch"], history["L_phys_bal"], label="L_phys (balanced)")
    ax[1].plot(history["epoch"], history["L_bc_bal"], label="L_bc (balanced)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("balanced loss (~O(1))")
    ax[1].legend(); ax[1].set_title("balanced losses (all ~same scale)")
    fig.tight_layout()
    fig.savefig(ART_DIR / "training_curves.png", dpi=130)
    plt.close(fig)

    all_ops = list(bundle.ops) + held
    fig, axes = plt.subplots(len(all_ops), 1, figsize=(10, 2.6 * len(all_ops)),
                             squeeze=False)
    rng = np.random.default_rng(42)
    for row, op_data in enumerate(all_ops):
        d = np.load(ART_DIR / f"pred_{op_data.op_id}.npz")
        t, T_true, T_pred = d["t"], d["T_true"], d["T_pred"]
        split_t = int(d["split_t"])
        pt = int(rng.integers(0, T_true.shape[1]))
        a = axes[row][0]
        a.plot(t, T_true[:, pt], "k-", lw=2, label="true")
        a.plot(t, T_pred[:, pt], "C3--", lw=1.4, label="pred")
        a.axvline(t[split_t], color="gray", ls=":")
        a.set_title(f"{op_data.op_id} - point {pt}")
        a.set_ylabel("T [C]"); a.legend()
    axes[-1][0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(ART_DIR / "timeseries.png", dpi=130)
    plt.close(fig)

    (ART_DIR / "metrics.txt").write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {ART_DIR/'metrics.txt'} and plots", flush=True)



if __name__ == "__main__":
    train(parse_args())
