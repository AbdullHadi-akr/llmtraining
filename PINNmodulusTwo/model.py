"""Approach-2 network: Modulus MLP backbone + PyTorch recurrence.

Split of responsibilities (the Notion "~50:50 Modulus:PyTorch" method #2):

* Modulus provides the function approximator:
    - ``modulus.models.module.Module`` base class (save/load, device, meta),
    - ``modulus.models.layers.FCLayer`` linear blocks with optional weight-norm.
* PyTorch provides everything the recurrence needs:
        - a per-layer *learnable swish*  ``x * sigmoid(beta * x)``  (beta learned per
            layer, exactly as described in the Notion page),
    - the temperature *history* channels  ``T_{t-delta}, ..., T_{t-k delta}``,
    - a **learnable delta** (history spacing) via differentiable time
      interpolation, and a soft **variable-k** gate so the model can switch lags
      on/off instead of us hard-coding how many are useful.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modulus.models.layers import FCLayer
    from modulus.models.meta import ModelMetaData
    from modulus.models.module import Module
except ModuleNotFoundError as exc:
    if exc.name == "modulus":
        raise ModuleNotFoundError(
            "Modulus is not installed in the current Python environment.\n"
            "  Activate the project virtualenv first, e.g.:\n"
            "    source .venv/bin/activate\n"
            "  and install it if needed:\n"
            "    pip install nvidia-modulus\n"
            "  Full setup (driver, CUDA torch, Modulus): "
            "PINNmodulusTwo/README_GPU_SERVER.md"
        ) from exc
    raise


class LearnableSwish(nn.Module):
    """Swish / SiLU with a learnable slope: ``x * sigmoid(beta * x)``.

    ``beta`` is a single learnable scalar per instance, so using one instance per
    hidden layer yields a distinct learned ``beta`` for every layer.
    """

    def __init__(self, beta_init: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class _MLPMeta(ModelMetaData):
    name: str = "ModulusMLP_Approach2"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = False


class ModulusMLP(Module):
    """Plain MLP built from Modulus ``FCLayer`` blocks with a learnable swish.

    Input  : ``[xn(3), config_feat(n_config), history(k_max)]``.
    Output : normalised temperature ``Tn`` (1 channel).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        layer_size: int | Sequence[int] = 128,
        num_layers: int = 4,
        weight_norm: bool = True,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__(meta=_MLPMeta())
        # ``layer_size`` may be a single int (uniform width) or a per-layer list;
        # a list makes the widths -- and the depth (len) -- fully variable.
        if isinstance(layer_size, int):
            widths = [layer_size] * num_layers
        else:
            widths = [int(w) for w in layer_size]
        layers = []
        prev = in_features
        for width in widths:
            layers.append(
                FCLayer(
                    in_features=prev,
                    out_features=width,
                    activation_fn=LearnableSwish(beta_init),
                    weight_norm=weight_norm,
                )
            )
            prev = width
        self.hidden = nn.ModuleList(layers)
        self.out = FCLayer(prev, out_features, activation_fn=None, weight_norm=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

    def betas(self) -> list[float]:
        """Current learned ``beta`` for every hidden layer (for logging)."""
        return [float(layer.activation_fn.beta.detach()) for layer in self.hidden]


def interp_history(
    Tn_seq: torch.Tensor,
    dtn: float,
    tq: torch.Tensor,
    p_idx: torch.Tensor,
) -> torch.Tensor:
    """Differentiable linear interpolation of a temperature history.

    Args:
        Tn_seq: (n_t, n_points) normalised temperature sequence for ONE OP.
        dtn: uniform spacing of the normalised time grid.
        tq: (B,) query times in normalised units (may depend on a learnable delta).
        p_idx: (B,) point index for each query (history is per grid point).

    Returns:
        (B,) interpolated normalised temperature at ``(tq, p_idx)``.

    The bracketing indices are detached (non-differentiable), but the linear blend
    weight is a smooth function of ``tq`` -- hence gradients flow into ``delta``.
    """
    n_t = Tn_seq.shape[0]
    pos = torch.clamp(tq / dtn, min=0.0, max=float(n_t - 1))
    lo = torch.clamp(torch.floor(pos).long(), 0, n_t - 1)
    hi = torch.clamp(lo + 1, 0, n_t - 1)
    frac = pos - lo.to(pos.dtype)
    t_lo = Tn_seq[lo, p_idx]
    t_hi = Tn_seq[hi, p_idx]
    return t_lo * (1.0 - frac) + t_hi * frac


class RecurrentField(nn.Module):
    """Temperature field with PyTorch recurrence around a Modulus MLP.

    The network input is ``[xn(3), static(S), config(C), forcing(F), history(k)]``:

    * ``static``  -- per-point, time-independent material/geometry features
      (thermal diffusivity, JR1 indicator, x-plane) so the net can tell the
      layers/materials apart.
    * ``forcing`` -- the instantaneous, normalised heat source ``q_dot(t)`` that
      actually drives the temperature rise.
    * ``config``  -- the (possibly time-varying) operating-point features.
    * ``history`` -- history channels built in either raw or hybrid mode.

    History is read through the differentiable :func:`interp_history` so that
    ``delta`` is learnable in raw mode. The history is ALWAYS the model's own
    past predictions (never a ground-truth teacher signal): training feeds the
    free-running rollout buffer back in.
    """

    def __init__(
        self,
        n_config: int,
        n_static: int = 0,
        n_forcing: int = 0,
        k_max: int = 2,
        history_mode: str = "raw",
        rate_lags: Sequence[float] = (5.0, 25.0),
        layer_size: int | Sequence[int] = 128,
        num_layers: int = 4,
        delta_seconds: float = 1.0,
        dtn: float = 1.0,
        weight_norm: bool = True,
        beta_init: float = 1.0,
        use_autograd_time: bool = False,
    ) -> None:
        super().__init__()
        self.history_mode = history_mode
        self._n_lags = len(rate_lags)
        self.k_max = 1 + self._n_lags if history_mode == "hybrid" else k_max
        
        # Learnable rate_lags: store raw values, use softplus to ensure positive
        # Initialize so that softplus(raw) ≈ desired lag values
        raw_lags = [float(lag) for lag in rate_lags]  # softplus(x) ≈ x for x > 3
        self._raw_rate_lags = nn.Parameter(torch.tensor(raw_lags))
        self.n_config = n_config
        self.n_static = n_static
        self.n_forcing = n_forcing
        self.use_autograd_time = use_autograd_time

        in_features = 3 + n_static + n_config + n_forcing + self.k_max
        self.mlp = ModulusMLP(
            in_features=in_features, out_features=1, layer_size=layer_size,
            num_layers=num_layers, weight_norm=weight_norm, beta_init=beta_init,
        )

        if use_autograd_time:
            self.mlp_with_time = ModulusMLP(
                in_features=in_features + 1, out_features=1, layer_size=layer_size,
                num_layers=num_layers, weight_norm=weight_norm, beta_init=beta_init,
            )
        else:
            self.mlp_with_time = None

        self.register_buffer("_delta", torch.tensor(float(delta_seconds)))

        self.log_src_gain = nn.Parameter(torch.zeros(()))
        self.log_diff_gain = nn.Parameter(torch.zeros(()))
        self.raw_cool = nn.Parameter(torch.tensor(-2.0))
        self.amb = nn.Parameter(torch.zeros(()))

    @property
    def delta(self) -> torch.Tensor:
        return self._delta

    @property
    def rate_lags(self) -> torch.Tensor:
        """Learnable segment lengths (always positive via softplus)."""
        return F.softplus(self._raw_rate_lags)

    @property
    def src_gain(self) -> torch.Tensor:
        return torch.exp(self.log_src_gain)

    @property
    def diff_gain(self) -> torch.Tensor:
        return torch.exp(self.log_diff_gain)

    @property
    def cool(self) -> torch.Tensor:
        return F.softplus(self.raw_cool)

    def gates(self) -> torch.Tensor:
        return torch.ones(self.k_max, dtype=self._delta.dtype, device=self._delta.device)

    def _padded_lookup(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        t_query: torch.Tensor,
        p_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup with per-endpoint padding: T(t) := T(0) for t < 0."""
        return interp_history(Tn_seq, dtn, torch.clamp(t_query, min=0.0), p_idx)

    def _history_hybrid(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
        rate_lags: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid history: anchor T(t-Δgrid) plus disjoint rate segments.

        rate_lags are CUMULATIVE segment lengths (not absolute boundaries):
          - rate_lags = [5, 20] with Δgrid=1s gives:
            - Anchor: T(t-1)
            - Rate 1: (T(t-1) - T(t-1-5)) / span = (T(t-1) - T(t-6)) / 5
            - Rate 2: (T(t-6) - T(t-6-20)) / span = (T(t-6) - T(t-26)) / 20

        Per-endpoint padding: T(t) := T(0) if t < 0.
        Effective span: uses actual elapsed time after padding (not nominal).
        """
        if self.k_max == 0:
            return tn_q.new_zeros((tn_q.shape[0], 0))

        dtn_t = tn_q.new_tensor(float(dtn))
        T_anchor = self._padded_lookup(Tn_seq, dtn, tn_q - dtn_t, p_idx)

        rates = []
        t_boundary = tn_q - dtn_t  # start at t - Δgrid (anchor point)
        for i in range(len(rate_lags)):
            seg_len = rate_lags[i]  # now a tensor (learnable)
            t_next = t_boundary - seg_len  # cumulative: subtract segment length

            T_end = self._padded_lookup(Tn_seq, dtn, t_boundary, p_idx)
            T_start = self._padded_lookup(Tn_seq, dtn, t_next, p_idx)

            # Effective span after per-endpoint padding
            t_end_actual = torch.clamp(t_boundary, min=0.0)
            t_start_actual = torch.clamp(t_next, min=0.0)
            actual_span = t_end_actual - t_start_actual

            rate = (T_end - T_start) / (actual_span + 1e-8)
            rates.append(rate)
            t_boundary = t_next  # next segment starts here

        return torch.cat([T_anchor.unsqueeze(1), torch.stack(rates, dim=1)], dim=1)

    def _history_raw(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build the raw (B, k_max) history block for query times ``tn_q``."""
        if self.k_max == 0:
            return tn_q.new_zeros((tn_q.shape[0], 0))

        delta = self.delta
        cols = []
        for i in range(1, self.k_max + 1):
            tq = tn_q - i * delta
            cols.append(interp_history(Tn_seq, dtn, tq, p_idx))
        return torch.stack(cols, dim=1)

    def field(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        hist: torch.Tensor,
    ) -> torch.Tensor:
        feats = torch.cat([xn, static, cfg, forcing, hist], dim=1)
        return self.mlp(feats).squeeze(-1)

    def field_with_time(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        hist: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_col = t.unsqueeze(-1) if t.dim() == 1 else t
        feats = torch.cat([xn, static, cfg, forcing, hist, t_col], dim=1)
        return self.mlp_with_time(feats).squeeze(-1)

    def _history(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.history_mode == "hybrid":
            return self._history_hybrid(Tn_seq, dtn, tn_q, p_idx, self.rate_lags)
        return self._history_raw(Tn_seq, dtn, tn_q, p_idx)

    def forward(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
    ) -> torch.Tensor:
        hist = self._history(Tn_seq, dtn, tn_q, p_idx)
        return self.field(xn, static, cfg, forcing, hist)

    def history_at(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
        lag: int = 1,
    ) -> torch.Tensor:
        """Ungated history at a single ``lag`` (used by the FD time derivative)."""
        tq = tn_q - lag * self.delta
        return interp_history(Tn_seq, dtn, tq, p_idx)


def rollout_train(
    model: RecurrentField,
    xn: torch.Tensor,        # (P, 3) normalised coords
    static: torch.Tensor,    # (P, n_static) per-point static features
    cfg_seq: torch.Tensor,   # (n_t, n_config) config features over time
    forcing_seq: torch.Tensor,  # (n_t, n_forcing) forcing (q_dot) over time
    Tn_ic: torch.Tensor,     # (P,) normalised initial condition (seed)
    tn: torch.Tensor,        # (n_t,) normalised time grid
    dtn: float,
) -> torch.Tensor:
    """Free-running autoregressive rollout that KEEPS gradients (for training).

    Identical stepping to :func:`rollout`, but differentiable so the data loss can
    be taken directly on the model's own trajectory -- there is NO teacher forcing.
    The buffer is seeded with the measured IC and every step reads history from the
    model's OWN earlier predictions. History values are detached between steps
    (truncated BPTT): each step's gradient flows through its own field evaluation
    and through the learnable ``delta``/gates, keeping memory bounded.
    """
    n_t, P = tn.shape[0], xn.shape[0]
    p_idx = torch.arange(P, device=xn.device)
    buf = torch.zeros(n_t, P, dtype=xn.dtype, device=xn.device)
    buf[0] = Tn_ic
    for ti in range(1, n_t):
        hist = model._history(buf[:ti].detach(), dtn, tn[ti].expand(P), p_idx)
        cfg = cfg_seq[ti].expand(P, -1)
        forcing = forcing_seq[ti].expand(P, -1)
        buf[ti] = model.field(xn, static, cfg, forcing, hist)
    return buf


@torch.no_grad()
def rollout(
    model: RecurrentField,
    xn: torch.Tensor,        # (P, 3) normalised coords
    static: torch.Tensor,    # (P, n_static) per-point static features
    cfg_seq: torch.Tensor,   # (n_t, n_config) config features over time
    forcing_seq: torch.Tensor,  # (n_t, n_forcing) forcing (q_dot) over time
    Tn_ic: torch.Tensor,     # (P,) normalised initial condition (seed)
    tn: torch.Tensor,        # (n_t,) normalised time grid
    dtn: float,
) -> torch.Tensor:
    """Free-running autoregressive rollout (no teacher forcing).

    The buffer is **seeded with the measured initial condition** and never
    predicts ``t=0``; from then on the model's own predictions are fed back as the
    temperature history. Because the history at step ``ti`` is read only from
    strictly earlier, already-computed times (``hi`` clamped to ``ti-1``), the IC
    is always satisfied exactly -- it is imposed, not learned.
    """
    n_t, P = tn.shape[0], xn.shape[0]
    buf = torch.zeros(n_t, P, dtype=xn.dtype, device=xn.device)
    buf[0] = Tn_ic
    p_idx = torch.arange(P, device=xn.device)
    for ti in range(1, n_t):
        hist = model._history(buf[:ti], dtn, tn[ti].expand(P), p_idx)
        cfg = cfg_seq[ti].expand(P, -1)
        forcing = forcing_seq[ti].expand(P, -1)
        buf[ti] = model.field(xn, static, cfg, forcing, hist)
    return buf
