from __future__ import annotations

from typing import Literal
import torch

from model import RecurrentField


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs.sum())/d(inputs), keeping the graph for higher-order derivs."""
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
    )[0]


TimeDerivMethod = Literal["bdf1", "bdf2", "autograd"]


def boundary_condition_loss(
    model: RecurrentField,
    xn: torch.Tensor,        # (P, 3) all grid coords (normalised)
    static: torch.Tensor,    # (P, n_static) per-point static features
    cfg: torch.Tensor,       # (B, n_config)
    forcing: torch.Tensor,   # (B, n_forcing) forcing at the sampled times
    Tn_seq: torch.Tensor,    # (n_t, n_points) history sequence (model's own predictions)
    dtn: float,
    tn_q: torch.Tensor,      # (B,) query times (normalised)
    bc_mask: torch.Tensor,   # (P,) boolean mask for boundary points (x=0)
    bc_scale: float = 1.0,
) -> torch.Tensor:
    """Boundary condition: dT/dx = 0 at cell center (x=0).
    
    Returns residual for sampled boundary points and times.
    """
    # Find boundary point indices where x ≈ 0
    bc_indices = torch.where(bc_mask)[0]
    if len(bc_indices) == 0:
        return torch.tensor(0.0, device=xn.device)
    
    # Sample random boundary points and times
    n_samples = min(len(bc_indices), len(tn_q))
    p_idx = bc_indices[torch.randint(0, len(bc_indices), (n_samples,), device=xn.device)]
    t_idx = torch.randint(0, len(tn_q), (n_samples,), device=xn.device)
    tn_sample = tn_q[t_idx]
    cfg_sample = cfg[t_idx]
    forcing_sample = forcing[t_idx]
    
    # Evaluate model at boundary points
    xb = xn[p_idx].clone().requires_grad_(True)
    hist = model._history(Tn_seq, dtn, tn_sample, p_idx)
    T = model.field(xb, static[p_idx], cfg_sample, forcing_sample, hist)
    
    # Compute dT/dx (gradient wrt first coordinate)
    grad_T = _grad(T, xb)  # (n_samples, 3)
    dT_dx = grad_T[:, 0]   # derivative wrt x (first coordinate)
    
    # BC residual: dT/dx should be 0, normalized by sqrt(training RMS)
    # We use sqrt because the loss squares this residual
    return dT_dx / (bc_scale**0.5 + 1e-8)


def heat_residual(
    model: RecurrentField,
    xn: torch.Tensor,        # (P, 3) all grid coords (normalised)
    static: torch.Tensor,    # (P, n_static) per-point static features
    cfg: torch.Tensor,       # (B, n_config)
    forcing: torch.Tensor,   # (B, n_forcing) forcing at the sampled times
    Fo: torch.Tensor,        # (P, 3, 3) Fourier tensor per point
    Qsrc: torch.Tensor,      # (B,) nondim source at the sampled (t, point)
    Tn_seq: torch.Tensor,    # (n_t, n_points) history sequence (model's own predictions)
    dtn: float,
    tn_q: torch.Tensor,      # (B,) query times (normalised)
    p_idx: torch.Tensor,     # (B,) point index per sample
    phys_scale: float,
    dTdt_scale: float = 1.0,
    aniso_scale: float = 1.0,
    Qsrc_scale: float = 1.0,
    time_deriv: TimeDerivMethod = "bdf2",
) -> torch.Tensor:
    """Return the scaled heat-equation residual at the sampled points.
    
    Time derivative methods:
      - bdf1: 1st-order backward difference, O(Δt) error
      - bdf2: 2nd-order backward difference, O(Δt²) error (recommended)
      - autograd: continuous autograd derivative, O(ε_machine) error
    """
    xb = xn[p_idx].clone().requires_grad_(True)   # (B, 3)
    hist = model._history(Tn_seq, dtn, tn_q, p_idx)
    
    if time_deriv == "autograd":
        # Continuous time derivative via autograd
        # Time as additional input, requires_grad=True for dT/dt
        t_input = tn_q.clone().requires_grad_(True)
        T = model.field_with_time(xb, static[p_idx], cfg, forcing, hist, t_input)
        dTdt = _grad(T, t_input)
    else:
        T = model.field(xb, static[p_idx], cfg, forcing, hist)
        
        if time_deriv == "bdf2":
            # BDF2: 2nd-order backward difference, O(Δt²) error
            # dT/dt ≈ (3*T - 4*T_{-1} + T_{-2}) / (2*Δt)
            T_1 = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=1)
            T_2 = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=2)
            dTdt = (3.0 * T - 4.0 * T_1 + T_2) / (2.0 * model.delta + 1e-8)
        else:
            # BDF1: 1st-order backward difference, O(Δt) error
            T_prev = model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=1)
            dTdt = (T - T_prev) / (model.delta + 1e-8)

    # Spatial derivatives via autograd (always continuous)
    grad1 = _grad(T, xb)                           # (B, 3) -> [Tx, Ty, Tz]
    Txx_row = _grad(grad1[:, 0], xb)               # [Txx, Txy, Txz]
    Tyy_row = _grad(grad1[:, 1], xb)               # [Tyx, Tyy, Tyz]
    Tzz_row = _grad(grad1[:, 2], xb)               # [Tzx, Tzy, Tzz]

    Txx, Txy, Txz = Txx_row[:, 0], Txx_row[:, 1], Txx_row[:, 2]
    Tyy, Tyz = Tyy_row[:, 1], Tyy_row[:, 2]
    Tzz = Tzz_row[:, 2]

    fo = Fo[p_idx]                                 # (B, 3, 3)
    aniso = (
        fo[:, 0, 0] * Txx + fo[:, 1, 1] * Tyy + fo[:, 2, 2] * Tzz
        + 2.0 * (fo[:, 0, 1] * Txy + fo[:, 0, 2] * Txz + fo[:, 1, 2] * Tyz)
    )

    # Normalize each term with sqrt of its training RMS so the squared loss
    # lands at the right scale. The learnable gains then adjust relative strength.
    dTdt_n = dTdt / (dTdt_scale**0.5 + 1e-8)
    aniso_n = model.diff_gain * (aniso / (aniso_scale**0.5 + 1e-8))
    src_n = model.src_gain * (Qsrc / (Qsrc_scale**0.5 + 1e-8))
    residual = dTdt_n - aniso_n - src_n
    return residual / (phys_scale**0.5)
