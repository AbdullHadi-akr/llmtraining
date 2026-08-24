#!/usr/bin/env python3
"""Train the Approach-2 recurrent Modulus PINN on OP01, OP02, OP03.

Temperature only (bc_V is intentionally out of scope). The model uses a Modulus
``FCLayer`` MLP with a per-layer learnable swish, wrapped in a PyTorch recurrence
whose history spacing ``delta`` and per-lag gates (variable ``k``) are learned.

Run (the device defaults to ``auto`` = CUDA when a GPU is available):
    source .venv/bin/activate
    python3 PINNmodulusTwo/train.py --epochs 60 --subsample 40

For the GPU server setup see ``PINNmodulusTwo/README_GPU_SERVER.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import load_ops
from device_utils import enable_tf32, resolve_device, seed_everything
from model import RecurrentField, rollout, rollout_train
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
    p.add_argument("--width", type=int, default=d.get("layer_size", 128))
    p.add_argument("--depth", type=int, default=d.get("num_layers", 4))
    p.add_argument("--lr", type=float, default=d.get("lr", 2e-3))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.0))
    p.add_argument("--grad-clip", type=float, default=d.get("grad_clip", 0.0),
                   help="maximum gradient norm; 0 disables clipping")
    p.add_argument("--early-stopping-patience", type=int,
                   default=d.get("early_stopping_patience", 0),
                   help="epochs without training-loss improvement; 0 disables it")
    p.add_argument("--w-data", type=float, default=d.get("w_data", 1.0))
    p.add_argument("--w-phys", type=float, default=d.get("w_phys", 0.1))
    p.add_argument("--w-bc", type=float, default=d.get("w_bc", 0.1))
    p.add_argument("--phys-norm", type=float, default=d.get("phys_norm", 0.0),
                   help="scale L_phys down before weighting: 0 = adaptive EMA "
                        "(auto-balance to ~data scale), >0 = fixed divisor")
    p.add_argument("--batch-data", type=int, default=d.get("batch_data", 2048))
    p.add_argument("--batch-phys", type=int, default=d.get("batch_phys", 256))
    p.add_argument("--batch-bc", type=int, default=d.get("batch_bc", 128))
    p.add_argument("--delta-init-steps", type=float, default=d.get("delta_init_steps", 1.0))
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


def fit(args):
    """Train on ``args.ops`` and return ``(model, bundle, ops_packed, dtn, history)``."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    enable_tf32(getattr(args, "tf32", False))

    bundle = load_ops(op_ids=args.ops, subsample_time=args.subsample)
    ops = _to_tensor_ops(bundle, device)
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
    _check_cfl_stability(bundle, dtn * bundle.T_span_ref, device)
    phys_scale = bundle.phys_scale
    rate_lags_s = [float(v) for v in getattr(args, "rate_lags", [])]
    rate_lags_n = [v / bundle.T_span_ref for v in rate_lags_s]
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
        delta_seconds=1.0, dtn=dtn,
        use_autograd_time=(args.time_deriv == "autograd"),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model params={n_params} k_max={model.k_max} delta=1.0s (fixed) "
        f"history_mode={model.history_mode} rate_lags_s={rate_lags_s} "
        f"width={args.width} depth={args.depth}",
        flush=True,
    )

    # Create boundary condition mask (x ≈ 0 for cell center)
    bc_mask = torch.tensor(np.abs(bundle.xn[:, 0]) < 1e-6, dtype=torch.bool, device=device)
    print(f"BC points (x=0): {bc_mask.sum().item()}/{len(bc_mask)}", flush=True)

    history = {"epoch": [], "L_data": [], "L_phys": [], "L_bc": [], "delta": [],
                "L_phys_bal": [], "L_bc_bal": []}  # balanced losses for fair comparison
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0))

    # Physics-loss balancing: L_phys typically lands ~100x above L_data, so a raw
    # w_phys is not a fair mixing weight. phys_norm=0 -> adaptive EMA of L_phys's
    # own magnitude (keeps the weighted term ~O(1)); phys_norm>0 -> fixed divisor.
    phys_norm = float(getattr(args, "phys_norm", 0.0))
    phys_ema = None
    bc_ema = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_data, ep_phys, ep_bc = 0.0, 0.0, 0.0
        for op in ops:
            n_t, n_pts = op["n_t"], op["n_points"]
            Tn_seq = op["Tn"]

            # ---- data term: free-running on the model's OWN past -------------
            # No teacher forcing: history is the model's rolled-out predictions,
            # seeded only by the measured initial condition.
            buf = rollout_train(
                model, op["xn"], op["static"], op["cfg"], op["forcing"],
                op["Tn_ic"], op["tn"], dtn,
            )
            L_data = torch.mean((buf[1:] - Tn_seq[1:]) ** 2)

            # ---- physics term (autograd space + FD time) ---------------------
            # History for the residual also comes from the model's OWN rollout.
            own_hist = buf.detach()
            pt = torch.randint(0, n_t, (args.batch_phys,), device=device)
            pp = torch.randint(0, n_pts, (args.batch_phys,), device=device)
            res = heat_residual(
                model, op["xn"], op["static"], op["cfg"][pt], op["forcing"][pt],
                op["Fo"], op["Qsrc"][pt, pp], own_hist, dtn, op["tn"][pt], pp,
                phys_scale, bundle.dTdt_scale, bundle.aniso_scale, bundle.Qsrc_scale,
                time_deriv=args.time_deriv,
            )
            L_phys = torch.mean(res ** 2)

            # ---- boundary condition term (dT/dx = 0 at x=0) ------------------
            pt_bc = torch.randint(0, n_t, (args.batch_bc,), device=device)
            bc_res = boundary_condition_loss(
                model, op["xn"], op["static"], op["cfg"][pt_bc], op["forcing"][pt_bc],
                own_hist, dtn, op["tn"][pt_bc], bc_mask, bundle.bc_scale,
            )
            L_bc = torch.mean(bc_res ** 2)

            # Balance L_phys and L_bc onto the data scale so weights are fair 0-1 knobs.
            if phys_norm > 0.0:
                phys_den = phys_norm
            else:
                cur = float(L_phys.detach())
                phys_ema = cur if phys_ema is None else 0.9 * phys_ema + 0.1 * cur
                phys_den = phys_ema
            L_phys_bal = L_phys / (phys_den + 1e-8)

            # BC balancing similar to physics
            bc_cur = float(L_bc.detach())
            bc_ema = bc_cur if bc_ema is None else 0.9 * bc_ema + 0.1 * bc_cur
            L_bc_bal = L_bc / (bc_ema + 1e-8)

            loss = args.w_data * L_data + args.w_phys * L_phys_bal + args.w_bc * L_bc_bal

            opt.zero_grad()
            loss.backward()
            grad_clip = float(getattr(args, "grad_clip", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            ep_data += float(L_data.detach())
            ep_phys += float(L_phys.detach())
            ep_bc += float(L_bc.detach())

        ep_data /= len(ops)
        ep_phys /= len(ops)
        ep_bc /= len(ops)

        # Early NaN/inf detection - abort before wasting epochs
        if not np.isfinite(ep_data) or not np.isfinite(ep_phys):
            print(f"  [ABORT] epoch {epoch}: loss exploded (L_data={ep_data:.4g}, L_phys={ep_phys:.4g})")
            print("  Possible causes: CFL violation (Δt too large), unstable IC, or bad hyperparams")
            print("  Try: --subsample 2 (for CFL-stable Δt=0.2s) or --grad-clip 1.0")
            break

        # Compute balanced losses for fair logging (all ~O(1) when stable)
        ep_phys_bal = ep_phys / (phys_ema + 1e-8) if phys_ema else 1.0
        ep_bc_bal = ep_bc / (bc_ema + 1e-8) if bc_ema else 1.0

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
            gates = np.round(model.gates().detach().cpu().numpy(), 2)
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
            print(
                f"  epoch {epoch:3d}  L_data={ep_data:.4e}  L_phys_bal={ep_phys_bal:.4e}  "
                f"L_bc_bal={ep_bc_bal:.4e}  delta={float(model.delta.detach()):.4g}  "
                f"src_gain={sg:.3g}  diff_gain={dg:.3g}{lags_str}  betas={betas}",
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
