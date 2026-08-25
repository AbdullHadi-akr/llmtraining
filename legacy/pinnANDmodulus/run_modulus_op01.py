#!/usr/bin/env python3
"""End-to-end Modulus-Sym (PhysicsNeMo) PINN for the OP01 solid heat equation.

This is the "Option A" counterpart to the hand-written PyTorch PINN in
``battery_surrogate_agenticWorkflow_PINN/pinn``. It uses the *full* Modulus-Sym
training stack — ``Solver`` + ``Domain`` + ``PointwiseConstraint`` + a custom
``PDE`` — instead of only borrowing ``modulus.models``.

Key modelling differences vs. the current PyTorch approach
----------------------------------------------------------
* Time ``t`` is a *continuous input coordinate*; there is NO recurrence, NO history
  channels and therefore NO "teacher forcing" concept. The time derivative is taken
  by autograd (``T__t``) rather than a finite difference along a rollout.
* The initial condition is enforced as a *soft* pointwise constraint (t=0 -> T_ic).
* The anisotropic diffusion tensor (Fourier numbers) and the volumetric source are
  supplied per collocation point as input Keys, matching the constant-per-point
  coefficient assumption of the current model (no div(lambda) term).

Run:
    source modulus_env/bin/activate
    python3 pinnANDmodulus/run_modulus_op01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
THIS_DIR = Path(__file__).resolve().parent            # .../pinnANDmodulus
PROJECT_ROOT = THIS_DIR.parent                        # .../batterysurrogatemodell
PINN_ROOT = PROJECT_ROOT / "battery_surrogate_agenticWorkflow_PINN"
MODULUS_SYM_SRC = PROJECT_ROOT / "modulus-sym"

# Prefer the classic Modulus-Sym Solver framework from the workspace source
# (the pip-installed physicsnemo core only ships the slim modern ``sym`` module).
sys.path.insert(0, str(MODULUS_SYM_SRC))
# reuse the existing data loaders / normalisation
sys.path.insert(0, str(PINN_ROOT))

import torch  # noqa: E402

# CPU shim: the classic Modulus-Sym trainer emits CUDA-only NVTX range markers
# unconditionally. Neutralise them (and other CUDA-only profiling hooks) when no
# GPU is present so the solver can run on a CPU-only PyTorch build.
if not torch.cuda.is_available():
    torch.cuda.nvtx.range_push = lambda *a, **k: None  # type: ignore[assignment]
    torch.cuda.nvtx.range_pop = lambda *a, **k: None  # type: ignore[assignment]

import physicsnemo.sym  # noqa: E402
from physicsnemo.sym.hydra import PhysicsNeMoConfig  # noqa: E402
from physicsnemo.sym.solver import Solver  # noqa: E402
from physicsnemo.sym.domain import Domain  # noqa: E402
from physicsnemo.sym.domain.constraint import PointwiseConstraint  # noqa: E402
from physicsnemo.sym.models.fully_connected import FullyConnectedArch  # noqa: E402
from physicsnemo.sym.key import Key  # noqa: E402

from op01_data import load_nondim  # noqa: E402
from heat_pde import AnisotropicHeatNonDim  # noqa: E402

# weighting to mirror the current run (w_phys = 0.1)
W_PHYS = 0.1
OUT_DIR = THIS_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- data
def build_nondim_dataset():
    """Load OP01 (shared builder) and return flat numpy arrays for the constraints."""
    d = load_nondim(subsample_time=40)
    xn, tn, Tn, Tn_ic, Fo = d.xn, d.tn, d.Tn, d.Tn_ic, d.Fo
    q_dot, q_mask, rho, Cp = d.q_dot, d.q_mask, d.rho, d.Cp
    T_span, T_sigma = d.T_span, d.T_sigma
    n_t, n_points, split_t, phys_scale = d.n_t, d.n_points, d.split_t, d.phys_scale

    col = lambda a: np.asarray(a, dtype=np.float32).reshape(-1, 1)

    # ---- DATA constraint: train timesteps [1, split_t) x all points ------------
    ti_tr = np.arange(1, split_t)
    TT, PP = np.meshgrid(ti_tr, np.arange(n_points), indexing="ij")
    ti_f = TT.ravel()
    pp_f = PP.ravel()
    data_invar = {
        "x": col(xn[pp_f, 0]), "y": col(xn[pp_f, 1]),
        "z": col(xn[pp_f, 2]), "t": col(tn[ti_f]),
    }
    data_outvar = {"T": col(Tn[ti_f, pp_f])}

    # ---- PHYSICS constraint: same train times x all points ---------------------
    Qsrc = q_dot[ti_f] * q_mask[pp_f] * T_span / (rho[pp_f] * Cp[pp_f] * T_sigma)
    phys_invar = {
        "x": col(xn[pp_f, 0]), "y": col(xn[pp_f, 1]),
        "z": col(xn[pp_f, 2]), "t": col(tn[ti_f]),
        "Fo_xx": col(Fo[pp_f, 0, 0]), "Fo_yy": col(Fo[pp_f, 1, 1]),
        "Fo_zz": col(Fo[pp_f, 2, 2]), "Fo_xy": col(Fo[pp_f, 0, 1]),
        "Fo_xz": col(Fo[pp_f, 0, 2]), "Fo_yz": col(Fo[pp_f, 1, 2]),
        "Qsrc": col(Qsrc),
    }
    phys_outvar = {"heat_residual": np.zeros((ti_f.size, 1), dtype=np.float32)}
    # weight the physics residual to match the current run:
    #   L_phys = w_phys * mean( (residual / phys_scale)^2 )
    phys_lambda = {
        "heat_residual": np.full((ti_f.size, 1), W_PHYS / (phys_scale ** 2), dtype=np.float32)
    }

    # ---- IC constraint: t=0 (tn=0) x all points --------------------------------
    ic_invar = {
        "x": col(xn[:, 0]), "y": col(xn[:, 1]), "z": col(xn[:, 2]),
        "t": np.zeros((n_points, 1), dtype=np.float32),
    }
    ic_outvar = {"T": col(Tn_ic)}

    return {
        "data": (data_invar, data_outvar),
        "phys": (phys_invar, phys_outvar, phys_lambda),
        "ic": (ic_invar, ic_outvar),
        # de-normalisation + eval bookkeeping
        "xn": xn, "tn": tn, "Tn": Tn, "T_lab": d.T_lab,
        "T_mu": d.T_mu, "T_sigma": T_sigma,
        "n_t": n_t, "n_points": n_points, "split_t": split_t,
        "t_phys": d.t, "phys_scale": phys_scale,
    }


# --------------------------------------------------------------------------- eval
def evaluate_and_plot(net, ds):
    """Run inference with the trained arch, print MAE, save predictions + a plot."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xn, tn = ds["xn"], ds["tn"]
    n_t, n_points, split_t = ds["n_t"], ds["n_points"], ds["split_t"]
    T_mu, T_sigma, T_lab = ds["T_mu"], ds["T_sigma"], ds["T_lab"]

    net.eval()
    T_pred = np.zeros((n_t, n_points), dtype=np.float64)
    dev = next(net.parameters()).device
    with torch.no_grad():
        for ti in range(n_t):
            invar = {
                "x": torch.as_tensor(xn[:, 0:1], dtype=torch.float32, device=dev),
                "y": torch.as_tensor(xn[:, 1:2], dtype=torch.float32, device=dev),
                "z": torch.as_tensor(xn[:, 2:3], dtype=torch.float32, device=dev),
                "t": torch.full((n_points, 1), float(tn[ti]), dtype=torch.float32, device=dev),
            }
            out = net(invar)["T"].detach().cpu().numpy().ravel()
            T_pred[ti] = out * T_sigma + T_mu

    err = np.abs(T_pred - T_lab)
    mae_train = float(err[1:split_t].mean())
    mae_test = float(err[split_t:].mean())
    rmse_train = float(np.sqrt(((T_pred[1:split_t] - T_lab[1:split_t]) ** 2).mean()))
    rmse_test = float(np.sqrt(((T_pred[split_t:] - T_lab[split_t:]) ** 2).mean()))

    print("=" * 70, flush=True)
    print("MODULUS-SYM (Option A) evaluation", flush=True)
    print(f"  net_T  MAE  train={mae_train:.3f} C   test={mae_test:.3f} C", flush=True)
    print(f"  net_T  RMSE train={rmse_train:.3f} C   test={rmse_test:.3f} C", flush=True)
    print("=" * 70, flush=True)

    np.savez_compressed(
        OUT_DIR / "op01_modulus_predictions.npz",
        t=ds["t_phys"], T_true=T_lab, T_pred=T_pred, split_t=split_t,
    )

    # single random grid point (same seed convention as the current run)
    rng = np.random.default_rng(42)
    p = int(rng.integers(0, n_points))
    t_phys = ds["t_phys"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax0.plot(t_phys, T_lab[:, p], "k-", lw=2, label="true")
    ax0.plot(t_phys, T_pred[:, p], "C3--", lw=1.5, label="Modulus pred")
    ax0.axvline(t_phys[split_t], color="gray", ls=":", label="train/test split")
    ax0.set_ylabel("T [C]")
    ax0.set_title(f"Modulus-Sym OP01 - grid point {p}  (MAE test={mae_test:.2f} C)")
    ax0.legend()
    ax1.plot(t_phys, np.abs(T_pred[:, p] - T_lab[:, p]), "C1-", lw=1.2)
    ax1.axvline(t_phys[split_t], color="gray", ls=":")
    ax1.set_ylabel("|error| [C]")
    ax1.set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "T_single_gridpoint_modulus.png", dpi=130)
    plt.close(fig)

    with open(OUT_DIR / "modulus_metrics.txt", "w") as fh:
        fh.write("Modulus-Sym (Option A) OP01 heat PINN\n")
        fh.write(f"MAE  train = {mae_train:.4f} C\n")
        fh.write(f"MAE  test  = {mae_test:.4f} C\n")
        fh.write(f"RMSE train = {rmse_train:.4f} C\n")
        fh.write(f"RMSE test  = {rmse_test:.4f} C\n")
        fh.write(f"phys_scale = {ds['phys_scale']:.6g}\n")
    print(f"  wrote {OUT_DIR/'modulus_metrics.txt'}", flush=True)


# --------------------------------------------------------------------------- main
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    ds = build_nondim_dataset()
    print(f"  dataset: n_t={ds['n_t']} n_points={ds['n_points']} split_t={ds['split_t']}"
          f"  phys_scale={ds['phys_scale']:.4g}", flush=True)

    # continuous field T(x,y,z,t)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("T")],
        nr_layers=4,
        layer_size=128,
    )
    pde = AnisotropicHeatNonDim()
    nodes = pde.make_nodes() + [net.make_node(name="net_T")]

    domain = Domain()

    data_invar, data_outvar = ds["data"]
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes, invar=data_invar, outvar=data_outvar,
            batch_size=cfg.batch_size.data,
        ),
        "data",
    )

    phys_invar, phys_outvar, phys_lambda = ds["phys"]
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes, invar=phys_invar, outvar=phys_outvar,
            batch_size=cfg.batch_size.physics, lambda_weighting=phys_lambda,
        ),
        "physics",
    )

    ic_invar, ic_outvar = ds["ic"]
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes, invar=ic_invar, outvar=ic_outvar,
            batch_size=cfg.batch_size.ic,
        ),
        "ic",
    )

    slv = Solver(cfg, domain)
    slv.solve()

    evaluate_and_plot(net, ds)


if __name__ == "__main__":
    run()
