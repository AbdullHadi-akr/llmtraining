#!/usr/bin/env python3
"""Runnable continuous-time PINN for OP01 (Option A modelling paradigm).

This is the *executable* counterpart to ``run_modulus_op01.py``. The true
end-to-end ``physicsnemo.sym`` Solver cannot run in this environment (the
workspace ``modulus-sym/`` is the physicsnemo-branded rewrite and needs a
``physicsnemo`` core package that is not installed; only ``nvidia-modulus 0.9.0``
is present, which has no ``modulus.sym``). See ``README.md`` / prompt 009.

To still produce real comparison numbers, this script implements the SAME
modelling paradigm as the Solver script — a continuous field ``T(x,y,z,t)`` with
autograd time/space derivatives, a soft IC and an anisotropic-heat physics
residual — using the installed ``modulus.models.mlp.FullyConnected`` and a plain
PyTorch training loop. It deliberately mirrors the non-dimensionalisation and
Fourier tensor of the current recurrent model so the losses are comparable.

Differences from the current model that this script isolates:
  * time is a continuous input (NO recurrence, NO history, NO teacher forcing)
  * dT/dt is autograd (partial), not a finite difference along a rollout
  * IC is soft (penalty at t=0), not hard

Run:
    source modulus_env/bin/activate
    python3 pinnANDmodulus/run_continuous_op01.py --steps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from modulus.models.mlp import FullyConnected  # noqa: E402
from op01_data import load_nondim  # noqa: E402

OUT_DIR = THIS_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x with graph retained (y, x same shape (N,1))."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-batch", type=int, default=8192)
    ap.add_argument("--phys-batch", type=int, default=4096)
    ap.add_argument("--w-phys", type=float, default=0.1)
    ap.add_argument("--w-ic", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cpu")

    ds = load_nondim(subsample_time=40)
    print(f"  n_t={ds.n_t} n_points={ds.n_points} split_t={ds.split_t} "
          f"phys_scale={ds.phys_scale:.4g}", flush=True)

    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    xn = T(ds.xn)                      # (P,3)
    tn = T(ds.tn)                      # (n_t,)
    Tn = T(ds.Tn)                      # (n_t,P)
    Tn_ic = T(ds.Tn_ic)               # (P,)
    Fo = T(ds.Fo)                      # (P,3,3)
    q_dot = T(ds.q_dot)               # (n_t,)
    q_mask = T(ds.q_mask)             # (P,)
    rho = T(ds.rho)
    Cp = T(ds.Cp)
    T_span, T_sigma = ds.T_span, ds.T_sigma
    P, split_t, n_t = ds.n_points, ds.split_t, ds.n_t
    phys_scale2 = ds.phys_scale ** 2

    # training index pools (train timesteps only, exclude t0 for data)
    ti_data = np.arange(1, split_t)
    TT, PP = np.meshgrid(ti_data, np.arange(P), indexing="ij")
    data_ti = torch.as_tensor(TT.ravel(), device=dev)
    data_pp = torch.as_tensor(PP.ravel(), device=dev)
    n_data = data_ti.numel()

    ti_phys = np.arange(1, split_t)   # collocation over train times

    net = FullyConnected(
        in_features=4, out_features=1, layer_size=args.width,
        num_layers=args.depth, activation_fn="silu",
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  net params = {n_params}", flush=True)

    def forward_T(x, y, z, t):
        return net(torch.cat([x, y, z, t], dim=1))

    rng = np.random.default_rng(0)
    phys_table = []

    for step in range(1, args.steps + 1):
        opt.zero_grad()

        # ---- DATA loss (continuous field vs labels) ----
        di = torch.as_tensor(rng.integers(0, n_data, size=args.data_batch), device=dev)
        ti_b, pp_b = data_ti[di], data_pp[di]
        xd = xn[pp_b]
        td = tn[ti_b].unsqueeze(1)
        pred = forward_T(xd[:, 0:1], xd[:, 1:2], xd[:, 2:3], td)
        target = Tn[ti_b, pp_b].unsqueeze(1)
        loss_data = ((pred - target) ** 2).mean()

        # ---- PHYSICS loss (autograd anisotropic heat residual) ----
        pi = rng.integers(0, ti_phys.size, size=args.phys_batch)
        pp = rng.integers(0, P, size=args.phys_batch)
        ti_p = torch.as_tensor(ti_phys[pi], device=dev)
        pp_p = torch.as_tensor(pp, device=dev)
        xp = xn[pp_p]
        x = xp[:, 0:1].clone().requires_grad_(True)
        y = xp[:, 1:2].clone().requires_grad_(True)
        z = xp[:, 2:3].clone().requires_grad_(True)
        t = tn[ti_p].unsqueeze(1).clone().requires_grad_(True)
        Tp = forward_T(x, y, z, t)
        Tt = grad(Tp, t)
        Tx, Ty, Tz = grad(Tp, x), grad(Tp, y), grad(Tp, z)
        Txx = grad(Tx, x)
        Tyy = grad(Ty, y)
        Tzz = grad(Tz, z)
        Txy = grad(Tx, y)
        Txz = grad(Tx, z)
        Tyz = grad(Ty, z)
        Fp = Fo[pp_p]
        div = (Fp[:, 0, 0:1] * Txx + Fp[:, 1, 1:2] * Tyy + Fp[:, 2, 2:3] * Tzz
               + 2.0 * (Fp[:, 0, 1:2] * Txy + Fp[:, 0, 2:3] * Txz + Fp[:, 1, 2:3] * Tyz))
        Q = (q_dot[ti_p] * q_mask[pp_p] * T_span
             / (rho[pp_p] * Cp[pp_p] * T_sigma)).unsqueeze(1)
        resid = Tt - div - Q
        loss_phys_norm = (resid ** 2).mean() / phys_scale2   # comparable to current L_phys
        loss_phys = args.w_phys * loss_phys_norm

        # ---- IC loss (soft, t=0) ----
        t0 = torch.zeros((P, 1), device=dev)
        pred_ic = forward_T(xn[:, 0:1], xn[:, 1:2], xn[:, 2:3], t0)
        loss_ic = args.w_ic * ((pred_ic.squeeze(1) - Tn_ic) ** 2).mean()

        loss = loss_data + loss_phys + loss_ic
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:5d}  L={loss.item():.4e}  data={loss_data.item():.4e}"
                  f"  phys_norm={loss_phys_norm.item():.4e}  ic={loss_ic.item():.4e}",
                  flush=True)
            phys_table.append((step, float(loss_phys_norm.item())))

    # ---------------------------------------------------------------- evaluate
    net.eval()
    T_pred = np.zeros((n_t, P), dtype=np.float64)
    with torch.no_grad():
        for ti in range(n_t):
            tcol = torch.full((P, 1), float(tn[ti]), device=dev)
            out = forward_T(xn[:, 0:1], xn[:, 1:2], xn[:, 2:3], tcol).squeeze(1).cpu().numpy()
            T_pred[ti] = out * T_sigma + ds.T_mu

    err = np.abs(T_pred - ds.T_lab)
    mae_train = float(err[1:split_t].mean())
    mae_test = float(err[split_t:].mean())
    rmse_train = float(np.sqrt(((T_pred[1:split_t] - ds.T_lab[1:split_t]) ** 2).mean()))
    rmse_test = float(np.sqrt(((T_pred[split_t:] - ds.T_lab[split_t:]) ** 2).mean()))

    print("=" * 70, flush=True)
    print("CONTINUOUS-TIME PINN (Option A paradigm, runnable) - OP01", flush=True)
    print(f"  net_T  MAE  train={mae_train:.3f} C   test={mae_test:.3f} C", flush=True)
    print(f"  net_T  RMSE train={rmse_train:.3f} C   test={rmse_test:.3f} C", flush=True)
    print("  L_phys(normalised) by step:", flush=True)
    for s, v in phys_table:
        print(f"    step {s:5d}:  {v:.4f}", flush=True)
    print("=" * 70, flush=True)

    np.savez_compressed(
        OUT_DIR / "op01_continuous_predictions.npz",
        t=ds.t, T_true=ds.T_lab, T_pred=T_pred, split_t=split_t,
    )

    # single random grid point (seed 42, matching the current run convention)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = int(np.random.default_rng(42).integers(0, P))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax0.plot(ds.t, ds.T_lab[:, p], "k-", lw=2, label="true")
    ax0.plot(ds.t, T_pred[:, p], "C0--", lw=1.5, label="continuous-PINN pred")
    ax0.axvline(ds.t[split_t], color="gray", ls=":", label="train/test split")
    ax0.set_ylabel("T [C]")
    ax0.set_title(f"Continuous-time PINN OP01 - grid point {p}  (MAE test={mae_test:.2f} C)")
    ax0.legend()
    ax1.plot(ds.t, np.abs(T_pred[:, p] - ds.T_lab[:, p]), "C1-", lw=1.2)
    ax1.axvline(ds.t[split_t], color="gray", ls=":")
    ax1.set_ylabel("|error| [C]")
    ax1.set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "T_single_gridpoint_continuous.png", dpi=130)
    plt.close(fig)

    with open(OUT_DIR / "continuous_metrics.txt", "w") as fh:
        fh.write("Continuous-time PINN (Option A paradigm) OP01 heat\n")
        fh.write(f"steps={args.steps} width={args.width} depth={args.depth} "
                 f"w_phys={args.w_phys} w_ic={args.w_ic}\n")
        fh.write(f"params     = {n_params}\n")
        fh.write(f"MAE  train = {mae_train:.4f} C\n")
        fh.write(f"MAE  test  = {mae_test:.4f} C\n")
        fh.write(f"RMSE train = {rmse_train:.4f} C\n")
        fh.write(f"RMSE test  = {rmse_test:.4f} C\n")
        fh.write(f"phys_scale = {ds.phys_scale:.6g}\n")
    print(f"  wrote {OUT_DIR/'continuous_metrics.txt'}", flush=True)


if __name__ == "__main__":
    main()
