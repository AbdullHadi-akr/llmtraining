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
ResidualNorm = Literal["rms", "legacy"]


def _term_norm(scale: float, mode: ResidualNorm) -> float:
    """Divisor that takes a residual term to unit RMS.

    Every ``*_scale`` handed in here is already an RMS (``data.py`` builds them as
    ``sqrt(mean(x**2))``), so unit RMS means dividing by the scale itself:
    ``mean((x / s)**2) == mean(x**2) / s**2 == 1``.

    ``legacy`` reproduces the original ``x / sqrt(s)``, which leaves the term at
    ``mean(x**2) == s`` instead of 1. That is not a normalisation at all: the
    three residual terms then keep their relative size gap, which is precisely
    what the scales exist to remove. Kept only so old runs stay reproducible.
    """
    return (scale ** 0.5 if mode == "legacy" else scale) + 1e-8


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
    residual_norm: ResidualNorm = "rms",
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
    
    # BC residual: dT/dx should be 0, measured against the RMS spatial gradient
    # the training data actually shows (data.py: _measure_bc_scale). Dividing by
    # that scale puts a "typical" gradient at 1, so L_bc reads as a fraction of
    # the gradients present in the data rather than in units of nothing.
    return dT_dx / _term_norm(bc_scale, residual_norm)


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
    residual_norm: ResidualNorm = "rms",
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

    # Each term is divided by its own training RMS, so all three enter the
    # residual at unit scale and the equation is balanced BEFORE the learnable
    # gains touch it. src_gain/diff_gain then express a genuine physical
    # correction rather than having to travel decades just to undo a scaling
    # mistake -- and dTdt, which has no gain at all, could never be corrected.
    dTdt_n = dTdt / _term_norm(dTdt_scale, residual_norm)
    aniso_n = model.diff_gain * (aniso / _term_norm(aniso_scale, residual_norm))
    src_n = model.src_gain * (Qsrc / _term_norm(Qsrc_scale, residual_norm))
    residual = dTdt_n - aniso_n - src_n
    # In "rms" mode the terms are already unit-scale, so a further division by a
    # combined scale would only re-introduce an arbitrary factor -- and under EMA
    # balancing in train.py any constant here cancels out anyway.
    if residual_norm == "legacy":
        return residual / (phys_scale ** 0.5)
    return residual
