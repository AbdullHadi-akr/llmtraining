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
    bc_mask: torch.Tensor,   # (P,) bool mask OR (n_bc,) long index for x=0
    bc_scale: float = 1.0,
    residual_norm: ResidualNorm = "rms",
) -> torch.Tensor:
    """Boundary condition: dT/dx = 0 at cell center (x=0).
    
    Returns residual for sampled boundary points and times.
    """
    # Find boundary point indices where x ≈ 0.
    #
    # A bool mask has to go through torch.where() to become indices, and that is
    # a DATA-DEPENDENT shape: the very next line asks for its length, which makes
    # the CPU wait for the GPU to finish. The mask is constant for a whole run,
    # so paying that stall on every optimiser step bought nothing. train.py now
    # hands the indices in directly. A bool mask is still accepted -- the tests
    # and every other caller pass one -- and then behaves exactly as before.
    if bc_mask.dtype == torch.bool:
        bc_indices = torch.where(bc_mask)[0]
    else:
        bc_indices = bc_mask
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
    # ``xn[p_idx]`` is advanced indexing, which already allocates a fresh
    # tensor sharing no storage with xn -- the clone was a second copy.
    xb = xn[p_idx].requires_grad_(True)
    hist = model._history(Tn_seq, dtn, tn_sample, p_idx)
    # ``level`` is spatially constant, so it drops out of d/dx exactly -- it is
    # passed anyway so T is the same absolute field the data term sees.
    T = model.field(xb, static[p_idx], cfg_sample, forcing_sample, hist,
                    model.level(Tn_seq, dtn, tn_sample))
    
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
    time_deriv: TimeDerivMethod = "bdf2",
    residual_norm: ResidualNorm = "rms",
    return_parts: bool = False,
    live_lags: dict | None = None,
):
    """Return the scaled heat-equation residual at the sampled points.

    ``live_lags`` switches the BDF stencil to ``--phys-stencil live`` (O21,
    FAHRPLAN 11.10). By default ``T_1``/``T_2`` are READ from ``Tn_seq`` -- the
    rollout frozen at the start of the epoch -- while ``T`` is the LIVE network.
    Every inner step moves the live network towards the labels and leaves the
    buffer where it was, and that delta-independent jump sits in the numerator
    and is divided by delta: ``L_phys ~ 1/delta^2``. With ``live_lags`` the two
    lags are evaluated by the SAME live network instead, each with its own
    history from the buffer, so a common offset cancels (3 - 4 + 1 = 0). The
    dict holds, for n = 1 and 2: ``tq_n`` (normalised time t - n*delta),
    ``cfg_n``, ``forcing_n`` (the inputs at that row) and ``valid_n`` (bool:
    the row is a prediction -- row 0 is the imposed initial condition, and
    before it the buffer is padded; there the buffer value is kept). ``None``
    is the old stencil, unchanged.

    With ``return_parts=True`` the call returns ``(residual, parts)`` instead,
    where ``parts`` holds every UNSCALED term the residual was assembled from
    (``T``, the BDF lags ``T_1``/``T_2``, ``dTdt``, the four pieces of ``aniso``,
    ``Qsrc``) plus the divisor that was applied. It exists for
    ``tools/residual_decomposition.py``, which has to split ``L_phys`` into its
    terms WITHOUT a second copy of this assembly that could drift from it. The
    residual itself is computed by the same lines either way.

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
    xb = xn[p_idx].requires_grad_(True)   # (B, 3); indexing already copies
    hist = model._history(Tn_seq, dtn, tn_q, p_idx)
    level = model.level(Tn_seq, dtn, tn_q)
    T_1 = T_2 = None

    if time_deriv == "autograd":
        # Continuous time derivative via autograd
        # Time as additional input, requires_grad=True for dT/dt
        t_input = tn_q.clone().requires_grad_(True)
        T = model.field_with_time(xb, static[p_idx], cfg, forcing, hist, t_input,
                                  level)
        dTdt = _grad(T, t_input)
    else:
        T = model.field(xb, static[p_idx], cfg, forcing, hist, level)
        
        # In RAW mode the history block already IS the BDF stencil: column i-1 is
        # ``interp_history(tn_q - i*delta)``, the same tensor and the same call
        # ``history_at(lag=i)`` would make, so re-fetching it is duplicate work.
        # Hybrid packs [anchor, rates...] instead, and a raw run with too few
        # columns has nothing to reuse -- both fall back to the explicit lookup.
        raw_hist = model.history_mode != "hybrid"

        def _lag(n: int) -> torch.Tensor:
            if raw_hist and model.k_max >= n:
                return hist[:, n - 1]
            return model.history_at(Tn_seq, dtn, tn_q, p_idx, lag=n)

        def _stencil_lag(n: int) -> torch.Tensor:
            buffered = _lag(n)
            if live_lags is None:
                return buffered
            # xn WITHOUT grad: only T's own spatial derivatives enter the
            # conduction term; the lags only enter dT/dt.
            tq_n = live_lags[f"tq_{n}"]
            live = model.field(
                xn[p_idx], static[p_idx], live_lags[f"cfg_{n}"],
                live_lags[f"forcing_{n}"], model._history(Tn_seq, dtn, tq_n, p_idx),
                model.level(Tn_seq, dtn, tq_n),
            )
            return torch.where(live_lags[f"valid_{n}"], live, buffered)

        if time_deriv == "bdf2":
            # BDF2: 2nd-order backward difference, O(Δt²) error
            # dT/dt ≈ (3*T - 4*T_{-1} + T_{-2}) / (2*Δt)
            T_1 = _stencil_lag(1)
            T_2 = _stencil_lag(2)
            dTdt = (3.0 * T - 4.0 * T_1 + T_2) / (2.0 * model.delta + 1e-8)
        else:
            # BDF1: 1st-order backward difference, O(Δt) error
            T_1 = _stencil_lag(1)
            dTdt = (T - T_1) / (model.delta + 1e-8)

    # Spatial derivatives via autograd (always continuous)
    grad1 = _grad(T, xb)                           # (B, 3) -> [Tx, Ty, Tz]
    Txx_row = _grad(grad1[:, 0], xb)               # [Txx, Txy, Txz]
    Tyy_row = _grad(grad1[:, 1], xb)               # [Tyx, Tyy, Tyz]
    Tzz_row = _grad(grad1[:, 2], xb)               # [Tzx, Tzy, Tzz]

    Txx, Txy, Txz = Txx_row[:, 0], Txx_row[:, 1], Txx_row[:, 2]
    Tyy, Tyz = Tyy_row[:, 1], Tyy_row[:, 2]
    Tzz = Tzz_row[:, 2]

    fo = Fo[p_idx]                                 # (B, 3, 3)
    aniso_xx = fo[:, 0, 0] * Txx
    aniso_yy = fo[:, 1, 1] * Tyy
    aniso_zz = fo[:, 2, 2] * Tzz
    aniso_cross = 2.0 * (fo[:, 0, 1] * Txy + fo[:, 0, 2] * Txz + fo[:, 1, 2] * Tyz)
    aniso = aniso_xx + aniso_yy + aniso_zz + aniso_cross

    # The nondimensional heat equation, assembled in its own units. The gains are
    # 1.0 unless --learn-gains restores the old free-gain behaviour.
    residual = dTdt - model.diff_gain * aniso - model.src_gain * Qsrc
    # One scale for the assembled residual: an equation is not rescaled by
    # dividing its terms by different numbers. phys_scale is the RMS magnitude a
    # term of this equation has on the training data, so this lands L_phys at
    # O(1) without touching what the equation says.
    #
    # ``residual_norm`` can only still act on that ONE divisor. The per-term
    # variant it used to select is gone on purpose (see the note above: it
    # changed the equation rather than scaling it), so ``legacy`` no longer
    # restores the old size gap between dTdt / aniso / Qsrc -- it restores only
    # the old OVERALL divisor, ``sqrt(phys_scale)``, which leaves
    # ``mean(res**2) == phys_scale`` instead of 1.
    divisor = (phys_scale ** 0.5 if residual_norm == "legacy" else phys_scale) + 1e-30
    scaled = residual / divisor
    if not return_parts:
        return scaled
    parts = {
        "T": T, "T_1": T_1, "T_2": T_2, "dTdt": dTdt,
        "Txx": Txx, "Tyy": Tyy, "Tzz": Tzz,
        "aniso_xx": aniso_xx, "aniso_yy": aniso_yy, "aniso_zz": aniso_zz,
        "aniso_cross": aniso_cross,
        "diff_gain": model.diff_gain, "src_gain": model.src_gain,
        "Qsrc": Qsrc, "divisor": divisor,
    }
    return scaled, parts
