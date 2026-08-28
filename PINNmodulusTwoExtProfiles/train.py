#!/usr/bin/env python3
"""Train the profile extension on the full OP01-OP16 set.

Same model, same physics, same recurrence as ``PINNmodulusTwo/train.py``. What
differs is everything that a heterogeneous, profile-carrying OP set forces:

1. **Per-OP loss balancing.** The base loop keeps ONE EMA of ``L_phys`` and one
   of ``L_bc``, updated as it walks the OPs and stepped after each. With five
   near-identical constant OPs that is harmless. With eleven OPs spanning 0 C to
   4 C the physics term's magnitude differs by more than an order of magnitude
   between them, so a single EMA balances each OP against whichever OP happened
   to come before it -- an OP-order-dependent weighting nobody chose. Each OP
   now carries its own EMA, so ``w_phys`` means the same thing everywhere.

2. **Shuffled OP order.** One optimiser step per OP per epoch means the last OP
   in the list always gets the last word of every epoch. Harmless among five
   similar OPs, a systematic bias among sixteen dissimilar ones. The order is
   reshuffled every epoch from a seeded RNG, so runs stay reproducible.

3. **Honest late-window metric.** The data loss covers the whole rollout, so the
   base project's "MAE test" on a training OP is in-sample -- it is a split of
   the metric, not of the training data. This is now labelled as such, and
   ``--holdout-tail`` makes it real by truncating the training rollout at
   ``split_t``. Default off, so a plain run matches the base project's
   behaviour; see the README for when turning it on is and is not a good idea.

4. **Held-out OPs are evaluated here too.** ``--val-ops`` / ``--test-ops`` are
   rolled out and reported at the end of a run, with the tier from
   ``op_registry`` and a coverage check against the training ranges, so a single
   ``train.py`` run already answers "and how does it do on the profiles it never
   saw".

Run (device defaults to ``auto`` = CUDA when available):
    source .venv/bin/activate
    python3 PINNmodulusTwoExtProfiles/train.py --epochs 60

For the GPU server setup see ``PINNmodulusTwo/README_GPU_SERVER.md`` -- the
environment is the same one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from data import (
    build_op, coverage_report, effective_rate_scale, load_ops,
    normalisation_report, profile_report, require_ops,
)
from device_utils import enable_tf32, resolve_device, seed_everything
from model import RecurrentField, rollout
from op_metrics import format_op_metrics, op_metrics, rollout_phys
from op_registry import (
    DEFAULT_TEST_OPS, DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS, TIER_IN,
    split_summary, tier_of,
)
from physics import boundary_condition_loss, heat_residual

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
        f"[{status}] dt={float(delta_s):.3f}s, dt_max~{dt_max_cfl:.3f}s -> "
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
    bo = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(
        description="PINNmodulusTwo profile extension: OP01-OP16 with profiles")
    # ---- data ---------------------------------------------------------------
    p.add_argument("--ops", nargs="+", default=d.get("ops", list(DEFAULT_TRAIN_OPS)))
    p.add_argument("--val-ops", nargs="*", default=d.get("val_ops", list(DEFAULT_VAL_OPS)),
                   help="held-out OPs evaluated after training; the benchmark "
                        "SELECTS on these, train.py only reports them")
    p.add_argument("--test-ops", nargs="*", default=d.get("test_ops", list(DEFAULT_TEST_OPS)),
                   help="held-out OPs reported but never selected on")
    p.add_argument("--subsample", type=int, default=d.get("subsample_time", 2))
    p.add_argument("--train-frac", type=float, default=d.get("train_frac", 0.8),
                   help="fraction of each OP's timeline used for the pooled "
                        "normalisation statistics and the late-window split")
    p.add_argument("--resample", choices=["mean", "point"],
                   default=d.get("resample", "mean"),
                   help="how DRIVERS are reduced to the subsampled grid: 'mean' "
                        "= anti-aliased backward window (needed for profiles), "
                        "'point' = the base project's [::N]")
    # ---- profile feature block ----------------------------------------------
    p.add_argument("--driver-history", action=bo,
                   default=d.get("use_driver_history", True),
                   help="append causal rate channels for q_dot and the four "
                        "profile-capable config channels")
    p.add_argument("--driver-rate-lags", nargs="+", type=float,
                   default=d.get("driver_rate_lags", [5.0, 20.0]),
                   help="cumulative segment lengths in SECONDS for the driver "
                        "rate channels")
    # ---- model --------------------------------------------------------------
    p.add_argument("--k-max", type=int, default=d.get("k_max", 2))
    p.add_argument("--time-deriv", choices=["bdf1", "bdf2", "autograd"],
                   default=d.get("time_deriv", "bdf2"))
    p.add_argument("--history-mode", choices=["raw", "hybrid"],
                   default=d.get("history_mode", "hybrid"))
    p.add_argument("--rate-lags", nargs="+", type=float,
                   default=d.get("rate_lags", [5.0, 20.0]),
                   help="TEMPERATURE history segments in SECONDS (the "
                        "recurrence); distinct from --driver-rate-lags, which "
                        "are exogenous and unaffected. What matters is "
                        "A = 1/(lag_n * rate_scale), printed at startup")
    p.add_argument("--delta-grid", type=float, default=d.get("delta_grid", 0.2),
                   help="anchor lag of the hybrid temperature history in SECONDS")
    p.add_argument("--residual-output", action=argparse.BooleanOptionalAction,
                   default=d.get("residual_output", False),
                   help="OFF by default, and it should stay off. On, field() "
                        "returns level(t) + net(...) and the level is carried "
                        "through an integrator of gain exactly 1 with no leak, "
                        "so any one-signed component of the network output "
                        "accumulates over the trajectory without bound. Measured "
                        "in the base project it aborted on every seed in every "
                        "history configuration, raw included. See "
                        "PINNmodulusTwo/ARCHITECTURE.md 3.1")
    p.add_argument("--rollout-clamp", type=float,
                   default=d.get("rollout_clamp", 50.0),
                   help="saturate |Tn| in the rollout buffer; 0 disables. Keeps "
                        "a runaway rollout finite so the loss stays a number the "
                        "optimiser can move, instead of an inf that makes every "
                        "downstream term NaN. Load-bearing once w_phys > 0")
    p.add_argument("--max-rate-amp", type=float, default=d.get("max_rate_amp", 0.0),
                   help="cap on 1/(lag_n * rate_scale), the factor by which the "
                        "hybrid history magnifies a one-step LEVEL jump. 0 = off "
                        "(base-project behaviour). Raise the scale instead of "
                        "diverging when the pooled normalisation makes this large "
                        "-- see data.effective_rate_scale")
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--use-static", action=bo, default=d.get("use_static", True))
    p.add_argument("--use-forcing", action=bo, default=d.get("use_forcing", True))
    # ---- optimisation -------------------------------------------------------
    p.add_argument("--epochs", type=int, default=d.get("epochs", 60))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--gain-lr-mult", type=float, default=d.get("gain_lr_mult", 25.0))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 1.0))
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0))
    p.add_argument("--shuffle-ops", action=bo, default=d.get("shuffle_ops", True),
                   help="reshuffle the OP order every epoch so no OP always "
                        "takes the last optimiser step of the epoch")
    p.add_argument("--holdout-tail", action=bo, default=d.get("holdout_tail", False),
                   help="truncate the TRAINING rollout at split_t so the late "
                        "window of a training OP is genuinely held out")
    # ---- loss ---------------------------------------------------------------
    p.add_argument("--w-data", type=float, default=d.get("w_data", 1.0))
    p.add_argument("--w-phys", type=float, default=d.get("w_phys", 0.1))
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0))
    p.add_argument("--inner-steps", type=int, default=d.get("inner_steps", 100),
                   help="optimiser steps per OP per epoch, all against that "
                        "epoch's frozen rollout. The rollout is the expensive "
                        "part, so this raises the update count at roughly "
                        "constant cost; 1 reproduces the old one-step-per-OP "
                        "behaviour")
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    # ---- runtime ------------------------------------------------------------
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", default=d.get("device", "auto"),
                   help="auto | cpu | cuda | cuda:N")
    p.add_argument("--tf32", action="store_true", default=d.get("tf32", False))
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

    A loss that is already NaN in epoch 1 usually comes from the data. With
    profiles there is one more way in than the base project had: a config
    channel that is neither a scalar nor a time series in the bundle stays NaN
    until ``_normalise_config`` fills it, and a profile whose own time base does
    not overlap the OP would surface here.
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


def fit(args):
    """Train on ``args.ops``; returns ``(model, bundle, ops_packed, dtn, history)``."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    enable_tf32(getattr(args, "tf32", False))

    bundle = load_ops(
        op_ids=args.ops, subsample_time=args.subsample,
        train_frac=float(getattr(args, "train_frac", 0.8)),
        resample=getattr(args, "resample", "mean"),
        driver_rate_lags=[float(v) for v in getattr(args, "driver_rate_lags", [])],
        use_driver_history=bool(getattr(args, "driver_history", True)),
    )
    ops = _to_tensor_ops(bundle, device)
    _check_finite_inputs(ops)

    # Optional extra input feature blocks. Zero-width when disabled -- note that
    # switching forcing off now also removes every driver rate channel, which is
    # the whole profile feature block, not just q_dot.
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
        print(
            f"  [WARN] --delta-grid {delta_grid_s:g}s is below the data step "
            f"{dt_s:g}s; the anchor cannot resolve finer than the grid and will "
            f"effectively act as {dt_s:g}s.",
            flush=True,
        )

    rate_scale, rate_scale_lines = effective_rate_scale(
        bundle.dTdt_scale, rate_lags_n, float(getattr(args, "max_rate_amp", 0.0)))

    print("\n".join(normalisation_report(bundle)), flush=True)
    print("\n".join(rate_scale_lines), flush=True)
    print("\n".join(profile_report(bundle)), flush=True)
    print(
        f"OPs={list(args.ops)} n_config={bundle.n_config} n_static={n_static} "
        f"n_forcing={n_forcing} dtn={dtn:.4g} T_sigma={bundle.T_sigma:.3f} "
        f"time_deriv={args.time_deriv} history_mode={args.history_mode} "
        f"rate_lags_s={rate_lags_s} holdout_tail={bool(args.holdout_tail)}",
        flush=True,
    )

    model = RecurrentField(
        n_config=bundle.n_config, n_static=n_static, n_forcing=n_forcing,
        k_max=args.k_max, history_mode=args.history_mode, rate_lags=rate_lags_n,
        layer_size=args.width, num_layers=args.depth,
        delta_seconds=1.0, dtn=dtn, t_span_ref=bundle.T_span_ref,
        rate_scale=rate_scale, delta_grid=delta_grid_n,
        use_autograd_time=(args.time_deriv == "autograd"),
        # Was never passed here, so this extension silently ran with the model's
        # old default (True) and had no way to switch it off -- the same
        # integrator that aborts every run in the base project.
        residual_output=bool(getattr(args, "residual_output", False)),
    ).to(device)

    # src_gain / diff_gain correct the scale gap between the source and the
    # diffusion term. At the base LR they barely move; no weight decay on them,
    # because decaying log_gain towards 0 pulls the gain back to the 1.0 init
    # that is the very bias they exist to escape.
    gain_params = [model.log_src_gain, model.log_diff_gain]
    gain_ids = {id(p) for p in gain_params}
    base_params = [p for p in model.parameters() if id(p) not in gain_ids]
    gain_lr_mult = float(getattr(args, "gain_lr_mult", 25.0))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    opt = torch.optim.Adam([
        {"params": base_params, "lr": args.lr, "weight_decay": weight_decay},
        {"params": gain_params, "lr": args.lr * gain_lr_mult, "weight_decay": 0.0},
    ])
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model params={n_params} k_max={model.k_max} (fixed) "
        f"delta=1.0s = {float(model.delta):.4g} normalised (fixed) "
        f"delta_grid={delta_grid_s:g}s gates=all-on "
        f"history_mode={model.history_mode} width={args.width} depth={args.depth}",
        flush=True,
    )

    bc_mask = torch.tensor(np.abs(bundle.xn[:, 0]) < 1e-6, dtype=torch.bool,
                           device=device)
    print(f"BC points (x=0): {bc_mask.sum().item()}/{len(bc_mask)}", flush=True)

    history = {"epoch": [], "L_data": [], "L_phys": [], "L_bc": [], "delta": [],
               "L_phys_bal": [], "L_bc_bal": [], "aborted": False}
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0))

    # Per-OP balancing EMAs. One shared EMA would balance each OP against
    # whichever OP preceded it in the walk; across a set spanning 0 C to 4 C
    # that is an OP-order-dependent weighting nobody asked for.
    phys_norm = float(getattr(args, "phys_norm", 0.0))
    inner_steps = max(1, int(getattr(args, "inner_steps", 100)))
    phys_ema: dict = {}
    bc_ema: dict = {}
    order_rng = np.random.default_rng(int(args.seed))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        ep_data, ep_phys, ep_bc = 0.0, 0.0, 0.0
        ep_phys_bal, ep_bc_bal = 0.0, 0.0
        order = (order_rng.permutation(len(ops)) if args.shuffle_ops
                 else np.arange(len(ops)))
        for oi in order:
            op = ops[int(oi)]
            op_id = op["op_id"]
            n_t, n_pts = op["n_t"], op["n_points"]
            # With --holdout-tail the training rollout stops at split_t, so the
            # late window is never fitted and the late-window metric becomes a
            # real held-out number rather than a relabelled training error.
            t_end = int(op["split_t"]) if args.holdout_tail else n_t
            t_end = max(t_end, 2)
            Tn_seq = op["Tn"]

            # ---- one frozen rollout, then `inner_steps` updates against it ---
            # The base project replaced its differentiable rollout with this:
            # the history was detached between steps anyway, so ~7000 sequential
            # steps bought exactly the gradient a minibatch of (t, point) pairs
            # against a FROZEN trajectory gives -- for ONE optimiser step. The
            # rollout is refreshed every epoch, so inner_steps trades update
            # count against how stale the trajectory may get.
            with torch.no_grad():
                own_hist = rollout(
                    model, op["xn"], op["static"], op["cfg"], op["forcing"],
                    op["Tn_ic"], op["tn"][:t_end], dtn,
                    clamp=float(getattr(args, "rollout_clamp", 50.0) or 0.0),
                )

            op_data = op_phys = op_bc = 0.0
            op_phys_bal = op_bc_bal = 0.0
            aborted = False
            for _ in range(inner_steps):
                # ---- data term on a minibatch of (t, point) ------------------
                # t starts at 1: row 0 is the imposed initial condition, never a
                # prediction. With --holdout-tail the rollout already stops at
                # split_t, so the late window is never fitted.
                bt = torch.randint(1, t_end, (args.batch_data,), device=device)
                bp = torch.randint(0, n_pts, (args.batch_data,), device=device)
                tq = op["tn"][bt]
                hist = model._history(own_hist, dtn, tq, bp)
                pred = model.field(
                    op["xn"][bp], op["static"][bp], op["cfg"][bt],
                    op["forcing"][bt], hist, model.level(own_hist, dtn, tq),
                )
                L_data = torch.mean((pred - Tn_seq[bt, bp]) ** 2)

                # ---- physics term (autograd space + FD time) ----------------
                pt = torch.randint(0, t_end, (args.batch_phys,), device=device)
                pp = torch.randint(0, n_pts, (args.batch_phys,), device=device)
                res = heat_residual(
                    model, op["xn"], op["static"], op["cfg"][pt], op["forcing"][pt],
                    op["Fo"], op["Qsrc"][pt, pp], own_hist, dtn, op["tn"][pt], pp,
                    phys_scale, time_deriv=args.time_deriv,
                )
                L_phys = torch.mean(res ** 2)

                # ---- boundary condition term (dT/dx = 0 at x=0) -------------
                pt_bc = torch.randint(0, t_end, (args.batch_bc,), device=device)
                bc_res = boundary_condition_loss(
                    model, op["xn"], op["static"], op["cfg"][pt_bc],
                    op["forcing"][pt_bc], own_hist, dtn, op["tn"][pt_bc], bc_mask,
                    bundle.bc_scale,
                )
                L_bc = torch.mean(bc_res ** 2)

                # Balance onto the data scale so the weights are fair 0-1 knobs.
                # Only finite values update an EMA: one non-finite sample would
                # otherwise pin it at nan for the rest of the run (0.9*nan == nan).
                if phys_norm > 0.0:
                    phys_den = phys_norm
                else:
                    cur = float(L_phys.detach())
                    if np.isfinite(cur):
                        phys_ema[op_id] = (cur if op_id not in phys_ema
                                           else 0.9 * phys_ema[op_id] + 0.1 * cur)
                    phys_den = phys_ema.get(op_id, 1.0)
                L_phys_bal = L_phys / (phys_den + 1e-8)

                bc_cur = float(L_bc.detach())
                if np.isfinite(bc_cur):
                    bc_ema[op_id] = (bc_cur if op_id not in bc_ema
                                     else 0.9 * bc_ema[op_id] + 0.1 * bc_cur)
                L_bc_bal = L_bc / (bc_ema.get(op_id, 1.0) + 1e-8)

                # Only add terms that are switched on: ``0.0 * nan`` is nan, so a
                # zero weight does NOT neutralise a non-finite term -- it poisons
                # the whole loss, the gradients and every later epoch.
                loss = args.w_data * L_data
                if args.w_phys != 0.0:
                    loss = loss + args.w_phys * L_phys_bal
                if args.w_bc != 0.0:
                    loss = loss + args.w_bc * L_bc_bal

                if not torch.isfinite(loss):
                    bad = [n for n, v in (("L_data", L_data), ("L_phys", L_phys),
                                          ("L_bc", L_bc)) if not torch.isfinite(v)]
                    print(
                        f"  [ABORT] epoch {epoch}, {op_id}: non-finite loss; "
                        f"first offending term(s): {', '.join(bad) or 'weighted sum'}",
                        flush=True,
                    )
                    ep_data = float("nan")
                    aborted = True
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
                op_phys_bal += float(L_phys_bal.detach())
                op_bc_bal += float(L_bc_bal.detach())

            if aborted:
                break

            # Per-OP means over the inner steps, so the epoch numbers stay
            # comparable to the one-update-per-OP runs they are logged against.
            ep_data += op_data / inner_steps
            ep_phys += op_phys / inner_steps
            ep_bc += op_bc / inner_steps
            ep_phys_bal += op_phys_bal / inner_steps
            ep_bc_bal += op_bc_bal / inner_steps

        n_ops = len(ops)
        ep_data /= n_ops
        ep_phys /= n_ops
        ep_bc /= n_ops
        ep_phys_bal /= n_ops
        ep_bc_bal /= n_ops

        if not np.isfinite(ep_data) or not np.isfinite(ep_phys):
            print(f"  [ABORT] epoch {epoch}: loss exploded "
                  f"(L_data={ep_data:.4g}, L_phys={ep_phys:.4g})")
            print("  L_data non-finite means the free-running rollout diverged; check")
            print("  the history channels feeding back in (--history-mode raw isolates it).")
            print("  L_phys non-finite alone points at the residual: --time-deriv bdf1")
            print("  or --w-phys 0. Also worth trying: --grad-clip 1.0, a lower --lr,")
            print("  a larger --subsample, --no-driver-history to rule out the new")
            print("  driver rate channels, or --max-rate-amp 50: pooling OP01-OP16")
            print("  widens T_sigma, which shrinks dTdt_scale and so RAISES the")
            print("  factor by which the hybrid history magnifies the opening")
            print("  steps of the rollout (see data.effective_rate_scale).")
            history["epoch"].append(epoch)
            history["L_data"].append(ep_data)
            history["L_phys"].append(ep_phys)
            history["L_bc"].append(ep_bc)
            history["L_phys_bal"].append(float("nan"))
            history["L_bc_bal"].append(float("nan"))
            history["delta"].append(float(model.delta.detach()))
            history["aborted"] = True
            break

        history["epoch"].append(epoch)
        history["L_data"].append(ep_data)
        history["L_phys"].append(ep_phys)
        history["L_bc"].append(ep_bc)
        history["L_phys_bal"].append(ep_phys_bal)
        history["L_bc_bal"].append(ep_bc_bal)
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
            epoch_s = time.time() - epoch_start
            eta_min = (args.epochs - epoch) * epoch_s / 60.0
            print(
                f"  epoch {epoch:3d}  L_data={ep_data:.4e}  "
                f"L_phys_bal={ep_phys_bal:.4e}  L_bc_bal={ep_bc_bal:.4e}  "
                f"src_gain={sg:.3g}  diff_gain={dg:.3g}  betas={betas}  "
                f"[{epoch_s:.1f}s/epoch, this run ~{eta_min:.0f} min left]",
                flush=True,
            )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"  early stopping after epoch {epoch}: training data loss did "
                  f"not improve for {early_stopping_patience} epochs", flush=True)
            break

    return model, bundle, ops, dtn, history


def train(args) -> None:
    model, bundle, ops, dtn, history = fit(args)
    device = next(model.parameters()).device
    evaluate(model, bundle, ops, dtn, device, history, args)


@torch.no_grad()
def evaluate(model, bundle, ops, dtn, device, history, args) -> None:
    """Free-running rollout on training AND held-out OPs; MAE in physical C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    late_is_holdout = bool(getattr(args, "holdout_tail", False))
    lines = [
        "PINNmodulusTwo profile extension (temperature only)",
        "evaluation = FREE-RUNNING ROLLOUT (no teacher forcing)",
        f"history_mode(final) = {model.history_mode}",
        f"rate_lags(final, s) = "
        f"{np.round(model.rate_lags.cpu().numpy() * bundle.T_span_ref, 3).tolist()}",
        f"driver_rate_lags(s) = {list(bundle.driver_rate_lags)} "
        f"({'on' if bundle.use_driver_history else 'off'})",
        f"resample = {bundle.resample}",
        f"rate_scale = {float(model.rate_scale):.5g} "
        f"(dTdt_scale = {bundle.dTdt_scale:.5g}, "
        f"--max-rate-amp {float(getattr(args, 'max_rate_amp', 0.0)):g})",
        f"delta(final) = {float(model.delta):.5g} (normalised time)",
        f"src_gain(final)  = {float(model.src_gain):.4g}",
        f"diff_gain(final) = {float(model.diff_gain):.4g}",
        f"betas(final) = {np.round(model.mlp.betas(), 3).tolist()}",
        "",
    ]
    lines += normalisation_report(bundle) + [""]
    lines += split_summary(list(args.ops), list(args.val_ops),
                           list(args.test_ops)) + [""]

    rows = []
    # ---- training OPs --------------------------------------------------------
    print("\ntraining OPs (in-sample unless --holdout-tail):", flush=True)
    for op_data in bundle.ops:
        pred = rollout_phys(model, op_data, bundle, device)
        m = op_metrics(pred, op_data, late_is_holdout=late_is_holdout)
        rows.append((op_data.op_id, TIER_IN, "train", m))
        print(format_op_metrics(op_data.op_id, TIER_IN, m), flush=True)
        np.savez_compressed(
            ART_DIR / f"pred_{op_data.op_id}.npz",
            t=np.asarray(op_data.t), T_true=op_data.T_lab, T_pred=pred,
            split_t=op_data.split_t, transient=op_data.transient,
        )

    # ---- held-out OPs --------------------------------------------------------
    held = []
    for role, op_ids in (("val", list(args.val_ops)), ("test", list(args.test_ops))):
        if not op_ids:
            continue
        print(f"\nheld-out OPs ({role}):", flush=True)
        for op_id in op_ids:
            op_data = build_op(op_id, bundle, subsample_time=args.subsample)
            held.append(op_data)
            pred = rollout_phys(model, op_data, bundle, device)
            # Held out entirely, so the late window is genuinely unseen whatever
            # --holdout-tail was set to.
            m = op_metrics(pred, op_data, late_is_holdout=True)
            rows.append((op_id, tier_of(op_id), role, m))
            print(format_op_metrics(op_id, tier_of(op_id), m), flush=True)
            cov = coverage_report(bundle, op_data)
            for line in cov:
                print("     coverage:" + line, flush=True)
            lines.append(f"coverage {op_id}:")
            lines += cov
            np.savez_compressed(
                ART_DIR / f"pred_{op_id}.npz",
                t=np.asarray(op_data.t), T_true=op_data.T_lab, T_pred=pred,
                split_t=op_data.split_t, transient=op_data.transient,
            )

    lines += [""] + profile_report(bundle, held) + [""]
    lines.append(f"{'OP':<6} {'tier':<11} {'role':<6} {'MAE':>8} {'RMSE':>8} "
                 f"{'max':>8} {'peak_err':>9} {'transient':>10} {'quiescent':>10} "
                 f"{'late':>8}")
    for op_id, tier, role, m in rows:
        lines.append(
            f"{op_id:<6} {tier:<11} {role:<6} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
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

    # ---- plots ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(history["epoch"], history["L_data"], label="L_data")
    ax[0].semilogy(history["epoch"], history["L_phys"], label="L_phys (raw)")
    ax[0].semilogy(history["epoch"], history["L_bc"], label="L_bc (raw)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss (log scale)")
    ax[0].legend()
    ax[0].set_title("raw losses (may span orders of magnitude)")
    ax[1].plot(history["epoch"], history["L_data"], label="L_data")
    ax[1].plot(history["epoch"], history["L_phys_bal"], label="L_phys (balanced)")
    ax[1].plot(history["epoch"], history["L_bc_bal"], label="L_bc (balanced)")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("balanced loss (~O(1))")
    ax[1].legend()
    ax[1].set_title("balanced losses (per-OP EMA)")
    fig.tight_layout()
    fig.savefig(ART_DIR / "training_curves.png", dpi=130)
    plt.close(fig)

    # One timeseries panel per OP, at the sensor with the largest error: the
    # random sensor the base project picks usually lands somewhere easy, which
    # makes a profile OP look fine exactly where it is not.
    all_ops = list(bundle.ops) + held
    fig, axes = plt.subplots(len(all_ops), 1, figsize=(10, 2.6 * len(all_ops)),
                             squeeze=False)
    for row, op_data in enumerate(all_ops):
        d = np.load(ART_DIR / f"pred_{op_data.op_id}.npz")
        t, T_true, T_pred = d["t"], d["T_true"], d["T_pred"]
        p = int(np.abs(T_pred - T_true).mean(axis=0).argmax())
        a = axes[row][0]
        a.plot(t, T_true[:, p], "k-", lw=2, label="true")
        a.plot(t, T_pred[:, p], "C3--", lw=1.4, label="pred")
        tr = np.asarray(d["transient"], dtype=bool)
        if tr.any():
            a.fill_between(t, *a.get_ylim(), where=tr, color="C0", alpha=0.12,
                           step="mid", label="driver transient")
        a.set_title(f"{op_data.op_id} [{tier_of(op_data.op_id)}] - worst sensor {p}")
        a.set_ylabel("T [C]")
        a.legend(fontsize=8)
    axes[-1][0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(ART_DIR / "timeseries.png", dpi=130)
    plt.close(fig)

    (ART_DIR / "metrics.txt").write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {ART_DIR/'metrics.txt'} and plots", flush=True)


if __name__ == "__main__":
    _args = parse_args()
    require_ops(*_args.ops, *_args.val_ops, *_args.test_ops)
    train(_args)
