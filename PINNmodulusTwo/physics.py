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
    # ``level`` is spatially constant, so it drops out of d/dx exactly -- it is
    # passed anyway so T is the same absolute field the data term sees.
    T = model.field(xb, static[p_idx], cfg_sample, forcing_sample, hist,
                    model.level(Tn_seq, dtn, tn_sample))
    
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
    time_deriv: TimeDerivMethod = "bdf2",
) -> torch.Tensor:
    """Return the scaled heat-equation residual at the sampled points.

    All three terms -- ``dT/dt``, the anisotropic Laplacian ``Fo : grad^2 T`` and
    the source ``Qsrc`` -- are already expressed in the SAME nondimensional units
    by ``data.py`` (shared ``T_span_ref``, ``L_ref``, ``T_sigma``). The residual
    is therefore assembled first and divided by ONE scale at the end.

    Dividing each term by its own RMS instead, as this did before, does not
    rescale the equation -- it changes it. ``dTdt/sqrt(a) - aniso/sqrt(b) -
    Qsrc/sqrt(c)`` is only equivalent to ``dTdt - aniso - Qsrc`` when
    ``a == b == c``, and here they differ by orders of magnitude (``aniso_scale``
    was not even a term magnitude: it is the RMS of the Fourier tensor, with the
    ``grad^2 T`` factor missing). The learnable ``src_gain``/``diff_gain`` existed
    to undo that damage, which is why they needed a 25x learning rate -- and why
    the optimiser could instead drive both to 0 and satisfy the residual with a
    constant field. One scale, no gains to collapse.

    Time derivative methods:
      - bdf1: 1st-order backward difference, O(Δt) error
      - bdf2: 2nd-order backward difference, O(Δt²) error (recommended)
      - autograd: continuous autograd derivative, O(ε_machine) error
    """
    xb = xn[p_idx].clone().requires_grad_(True)   # (B, 3)
    hist = model._history(Tn_seq, dtn, tn_q, p_idx)
    level = model.level(Tn_seq, dtn, tn_q)
    
    if time_deriv == "autograd":
        # Continuous time derivative via autograd
        # Time as additional input, requires_grad=True for dT/dt
        t_input = tn_q.clone().requires_grad_(True)
        T = model.field_with_time(xb, static[p_idx], cfg, forcing, hist, t_input,
                                  level)
        dTdt = _grad(T, t_input)
    else:
        T = model.field(xb, static[p_idx], cfg, forcing, hist, level)
        
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

    # The nondimensional heat equation, assembled in its own units. The gains are
    # 1.0 unless --learn-gains restores the old free-gain behaviour.
    residual = dTdt - model.diff_gain * aniso - model.src_gain * Qsrc
    # One scale for the assembled residual: an equation is not rescaled by
    # dividing its terms by different numbers. phys_scale is the RMS magnitude a
    # term of this equation has on the training data, so this lands L_phys at
    # O(1) without touching what the equation says.
    return residual / (phys_scale + 1e-30)
