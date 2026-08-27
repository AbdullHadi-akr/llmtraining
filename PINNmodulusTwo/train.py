#!/usr/bin/env python3
"""Train the Approach-2 recurrent Modulus PINN on OP01, OP02, OP03.

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

from data import load_ops
from device_utils import enable_tf32, resolve_device, seed_everything
from model import RecurrentField, rollout
from physics import heat_residual, boundary_condition_loss

THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


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
    p.add_argument("--ops", nargs="+", default=d.get("ops", ["OP01", "OP02", "OP03"]))
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 40))
    p.add_argument("--epochs", type=int, default=d.get("epochs", 60))
    p.add_argument("--k-max", type=int, default=d.get("k_max", 4))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "raw"))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 25.0]))
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
    p.add_argument("--residual-output", action=argparse.BooleanOptionalAction,
                   default=d.get("residual_output", True),
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
    p.add_argument("--use-static", action="store_true", default=d.get("use_static", False))
    p.add_argument("--use-forcing", action="store_true", default=d.get("use_forcing", False))
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N (auto = cuda when available)")
    p.add_argument("--tf32", action="store_true", default=d.get("tf32", False),
                   help="allow TF32 matmuls on Ampere+ GPUs; off by default because "
                        "the physics residual needs precise second derivatives")
    p.add_argument("--test-op", default=d.get("test_op", "OP16"))
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

    bundle = load_ops(op_ids=args.ops, subsample_time=args.subsample)
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

    model = RecurrentField(
        n_config=bundle.n_config, n_static=n_static, n_forcing=n_forcing,
        k_max=args.k_max, history_mode=args.history_mode, rate_lags=rate_lags_n,
        layer_size=args.width, num_layers=args.depth,
        delta_seconds=1.0, dtn=dtn, t_span_ref=bundle.T_span_ref,
        rate_scale=bundle.dTdt_scale, delta_grid=delta_grid_n,
        use_autograd_time=(args.time_deriv == "autograd"),
        residual_output=bool(getattr(args, "residual_output", True)),
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
    print(f"BC points (x=0): {bc_mask.sum().item()}/{len(bc_mask)}", flush=True)

    # Keys mirror bench_common.EMPTY_HIST: a benchmark that aggregates a failed
    # run against a successful one needs both to carry the same series.
    history = {"epoch": [], "L_data": [], "L_phys": [], "L_bc": [], "delta": [],
               "L_phys_bal": [], "L_bc_bal": [],   # balanced losses, fair comparison
               "ratio_phys": [], "ratio_bc": []}   # read by benchmark_balance.py
    history["aborted"] = False
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0))

    # Optimiser steps taken per OP per epoch, all against the one frozen rollout
    # that epoch computed. The rollout is what costs time (~7000 sequential steps
    # that cannot be parallelised); a minibatch step is cheap, so this is where the
    # update count comes from.
    inner_steps = max(1, int(getattr(args, "inner_steps", 100)))
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

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        ep_data, ep_phys, ep_bc = 0.0, 0.0, 0.0
        ep_ratio_phys, ep_ratio_bc = 0.0, 0.0
        # Split the epoch's wall time into its two halves. Every runtime estimate
        # in README_GPU_SERVER chapters 7 and 8 is derived from one measured
        # seconds-per-epoch, and --inner-steps moves only the second half, so the
        # split is what makes the budget plannable instead of guessed.
        t_roll_s, t_inner_s = 0.0, 0.0
        aborted_epoch = False
        for op in ops:
            n_t, n_pts = op["n_t"], op["n_points"]
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
                    op["Tn_ic"], op["tn"], dtn,
                )
            t_roll_s += time.time() - _t0
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
                    pt = torch.randint(0, n_t, (args.batch_phys,), device=device)
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
                    pt_bc = torch.randint(0, n_t, (args.batch_bc,), device=device)
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

        ep_ratio_phys /= len(ops)
        ep_ratio_bc /= len(ops)
        ep_data /= len(ops)
        ep_phys /= len(ops)
        ep_bc /= len(ops)

        # Early NaN/inf detection - abort before wasting epochs. A term that was
        # deliberately skipped is NaN on purpose and must not trigger this.
        phys_broken = want_phys and not np.isfinite(ep_phys)
        if not np.isfinite(ep_data) or phys_broken:
            print(f"  [ABORT] epoch {epoch}: loss exploded (L_data={ep_data:.4g}, L_phys={ep_phys:.4g})")
            print("  L_data non-finite means the free-running rollout diverged; check the")
            print("  history channels feeding back into the net (--history-mode raw isolates it).")
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
        history["delta"].append(float(model.delta.detach()))
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


def train(args) -> None:
    model, bundle, ops, dtn, history = fit(args)
    device = next(model.parameters()).device
    evaluate(model, bundle, ops, dtn, device, history)


@torch.no_grad()
def evaluate(model, bundle, ops, dtn, device, history) -> None:
    """Free-running rollout (NO teacher forcing); report MAE per OP (physical C)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    T_mu, T_sigma = bundle.T_mu, bundle.T_sigma
    lines = ["Approach-2 recurrent Modulus PINN (temperature only)\n"]
    lines.append("evaluation = FREE-RUNNING ROLLOUT (no teacher forcing)\n")
    lines.append(f"history_mode(final) = {model.history_mode}\n")
    rate_lags_final = model.rate_lags.detach().cpu().numpy() * bundle.T_span_ref
    lines.append(f"rate_lags(final, s) = {np.round(rate_lags_final, 3).tolist()}\n")
    lines.append(f"delta(final) = {float(model.delta):.5g} (normalised time)\n")
    lines.append(f"src_gain(final)  = {float(model.src_gain):.4g}\n")
    lines.append(f"diff_gain(final) = {float(model.diff_gain):.4g}\n")
    lines.append(f"gates(final) = {np.round(model.gates().cpu().numpy(), 3).tolist()}\n")
    lines.append(f"betas(final) = {np.round(model.mlp.betas(), 3).tolist()}\n\n")

    summary = {}
    for op in ops:
        split_t = op["split_t"]
        T_lab = op["T_lab"]
        # Autoregressive rollout: seed the buffer with the measured IC and feed
        # the model's OWN predictions back as history (no ground-truth history).
        buf = rollout(
            model, op["xn"], op["static"], op["cfg"], op["forcing"],
            op["Tn_ic"], op["tn"], dtn,
        )
        T_pred = (buf.cpu().numpy() * T_sigma + T_mu).astype(np.float64)
        err = np.abs(T_pred - T_lab)
        mae_tr = float(err[1:split_t].mean())
        mae_te = float(err[split_t:].mean())
        rmse_te = float(np.sqrt(((T_pred[split_t:] - T_lab[split_t:]) ** 2).mean()))
        summary[op["op_id"]] = (mae_tr, mae_te, rmse_te)
        lines.append(f"{op['op_id']}: MAE train={mae_tr:.3f} C  test={mae_te:.3f} C  "
                     f"RMSE test={rmse_te:.3f} C\n")
        print(f"  {op['op_id']}: MAE train={mae_tr:.3f} C  test={mae_te:.3f} C  "
              f"RMSE test={rmse_te:.3f} C", flush=True)
        np.savez_compressed(
            ART_DIR / f"pred_{op['op_id']}.npz",
            t=op["t"], T_true=T_lab, T_pred=T_pred, split_t=split_t,
        )

    # loss curves (left: raw losses, right: balanced losses on same scale)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(history["epoch"], history["L_data"], label="L_data")
    ax[0].semilogy(history["epoch"], history["L_phys"], label="L_phys (raw)")
    ax[0].semilogy(history["epoch"], history["L_bc"], label="L_bc (raw)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss (log scale)")
    ax[0].legend()
    ax[0].set_title("raw losses (may span orders of magnitude)")
    # Right panel: balanced losses - all should be ~O(1)
    ax[1].plot(history["epoch"], history["L_data"], label="L_data")
    if "L_phys_bal" in history and history["L_phys_bal"]:
        ax[1].plot(history["epoch"], history["L_phys_bal"], label="L_phys (balanced)")
        ax[1].plot(history["epoch"], history["L_bc_bal"], label="L_bc (balanced)")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("balanced loss (~O(1))")
    ax[1].legend()
    ax[1].set_title("balanced losses (all ~same scale)")
    fig.tight_layout()
    fig.savefig(ART_DIR / "training_curves.png", dpi=130)
    plt.close(fig)

    # per-OP single-point timeseries
    fig, axes = plt.subplots(len(ops), 1, figsize=(10, 3 * len(ops)), squeeze=False)
    rng = np.random.default_rng(42)
    for row, op in enumerate(ops):
        d = np.load(ART_DIR / f"pred_{op['op_id']}.npz")
        t, T_true, T_pred, split_t = d["t"], d["T_true"], d["T_pred"], int(d["split_t"])
        p = int(rng.integers(0, op["n_points"]))
        a = axes[row][0]
        a.plot(t, T_true[:, p], "k-", lw=2, label="true")
        a.plot(t, T_pred[:, p], "C3--", lw=1.4, label="pred")
        a.axvline(t[split_t], color="gray", ls=":")
        a.set_title(f"{op['op_id']} - point {p}")
        a.set_ylabel("T [C]")
        a.legend()
    axes[-1][0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(ART_DIR / "timeseries.png", dpi=130)
    plt.close(fig)

    (ART_DIR / "metrics.txt").write_text("".join(lines))
    print(f"  wrote {ART_DIR/'metrics.txt'} and plots", flush=True)


if __name__ == "__main__":
    train(parse_args())
