"""Approach-2 network: Modulus MLP backbone + PyTorch recurrence.

Split of responsibilities (the Notion "~50:50 Modulus:PyTorch" method #2):

* Modulus provides the function approximator:
    - ``modulus.models.module.Module`` base class (save/load, device, meta),
    - ``modulus.models.layers.FCLayer`` linear blocks with optional weight-norm.
* PyTorch provides everything the recurrence needs:
        - a per-layer *learnable swish*  ``x * sigmoid(beta * x)``  (beta learned per
            layer, exactly as described in the Notion page),
    - the temperature *history* channels  ``T_{t-delta}, ..., T_{t-k delta}``,
    - differentiable time interpolation of that history, so a history value can be
      read at any time between two grid points.

What is learned and what is not
-------------------------------
Deliberately fixed (configured once, never trained):

* ``delta``  -- history spacing, a registered buffer, NOT a parameter.
* ``k_max``  -- number of history channels. In hybrid mode it follows directly
  from the number of ``rate_lags`` (k_max = 1 anchor + one channel per lag).
* ``gates()`` -- returns all-ones. There is no soft lag gating: every history
  channel is always fully on. The method is kept so logging and checkpoints keep
  a stable shape.
* ``rate_lags`` -- hybrid-mode segment lengths, also a buffer.
* ``src_gain`` / ``diff_gain`` -- pinned at 1.0. They only ever existed to undo a
  per-term normalisation that ``physics.py`` no longer does; ``learn_gains=True``
  brings the old free gains back.

Learned: the MLP weights and the per-layer swish ``beta`` (plus the gains, when
``learn_gains`` asks for it). Nothing about the history layout is trained.

Output parameterisation
-----------------------
With ``residual_output`` (the default) the network predicts the deviation from
the spatially averaged temperature level of the anchor slice, and :meth:`field`
adds that level back. The level is spatially constant, so the autograd Laplacian
in ``physics.py`` and the ``dT/dx = 0`` boundary term are untouched, while the
rollout carries the overall temperature level instead of re-deriving it at every
one of its ~7000 steps.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

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
        tq: (B,) query times in normalised units.
        p_idx: (B,) point index for each query (history is per grid point).

    Returns:
        (B,) interpolated normalised temperature at ``(tq, p_idx)``.

    The bracketing indices are detached (non-differentiable), but the linear blend
    weight is a smooth function of ``tq``. The history layout is fixed, so no
    gradient flows into the query time itself; the interpolation exists so a lag
    may land between two grid points, not to train the lag.
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

    History is read through the differentiable :func:`interp_history`, which lets a
    lag land between two grid points. ``delta`` and ``k_max`` are fixed
    hyperparameters and ``gates()`` is all-ones -- see the module docstring for
    the full list of what is learned. The history is ALWAYS the model's own past
    predictions (never a ground-truth teacher signal): training feeds the
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
        t_span_ref: float = 1.0,
        rate_scale: float = 1.0,
        delta_grid: float | None = None,
        weight_norm: bool = True,
        beta_init: float = 1.0,
        use_autograd_time: bool = False,
        residual_output: bool = True,
        learn_gains: bool = False,
    ) -> None:
        super().__init__()
        self.history_mode = history_mode
        self.residual_output = bool(residual_output)
        self.learn_gains = bool(learn_gains)
        self._n_lags = len(rate_lags)
        # In hybrid mode the channel count follows from the history layout itself
        # (1 anchor + one rate per lag), so the ``k_max`` argument does not apply
        # there -- callers passing one get it overridden on purpose.
        self.k_max = 1 + self._n_lags if history_mode == "hybrid" else k_max
        
        # Fixed rate_lags: the segment lengths are configured, not trained, in line
        # with delta / k_max / gates. Stored as a plain buffer in NORMALISED time,
        # so what the CLI asks for is exactly what the history layout uses. (The
        # earlier learnable version ran them through softplus, which silently
        # turned a requested 5 s lag into 1024 s because softplus(x) ~ x does not
        # hold for the tiny normalised values -- no softplus, no such failure.)
        self.register_buffer(
            "_rate_lags", torch.tensor([float(lag) for lag in rate_lags])
        )
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

        # ``_delta`` is consumed as NORMALISED time everywhere (history_at,
        # heat_residual, and benchmark readback via delta * T_span_ref), so the
        # seconds value has to be normalised on the way in. Storing the raw 1.0
        # meant "one whole trajectory" (~1474 s) instead of one second: every
        # BDF stencil then read the initial condition and divided by a step
        # ~1500x too large, which silently invalidated L_phys.
        self.register_buffer(
            "_delta", torch.tensor(float(delta_seconds) / (float(t_span_ref) + 1e-30))
        )
        self.register_buffer("_dtn", torch.tensor(float(dtn)))
        # Anchor lag of the hybrid history, in NORMALISED time. It used to be
        # hardwired to the data grid step, which silently tied "how far back is
        # the anchor" to "how finely is the trajectory sampled" -- two unrelated
        # questions. Now it is its own knob; None keeps the old coupling.
        self.register_buffer(
            "_delta_grid",
            torch.tensor(float(dtn if delta_grid is None else delta_grid)),
        )
        self.rate_scale = float(rate_scale)

        # The three residual terms (dT/dt, the anisotropic Laplacian, the source)
        # are already in the SAME nondimensional units -- that is what the shared
        # ``T_span_ref`` / ``L_ref`` / ``T_sigma`` scaling in ``data.py`` buys.
        # ``physics.py`` therefore divides the assembled residual by one scale
        # instead of each term by its own, and these gains have nothing left to
        # correct: they stay pinned at exactly 1.0 unless ``learn_gains`` asks
        # for the old behaviour. Learnable gains multiply two of the three terms,
        # which lets the optimiser drive both towards 0 and satisfy L_phys with a
        # constant field -- the physics term switching itself off.
        if self.learn_gains:
            self.log_src_gain = nn.Parameter(torch.zeros(()))
            self.log_diff_gain = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_src_gain", torch.zeros(()))
            self.register_buffer("log_diff_gain", torch.zeros(()))

    @property
    def delta(self) -> torch.Tensor:
        """Lag of the physics BDF stencil, NORMALISED. Fixed, not learned.

        Distinct from :attr:`delta_grid`: this one only feeds ``history_at`` and
        therefore the finite-difference time derivative in ``physics.py``.
        """
        return self._delta

    @property
    def delta_grid(self) -> torch.Tensor:
        """Anchor lag of the hybrid history, NORMALISED. Fixed, not learned.

        The hybrid history is ``[T(t-delta_grid), rate_1, rate_2, ...]`` and the
        rate segments cascade backwards from that same anchor point, so this sets
        where the whole history block is rooted.
        """
        return self._delta_grid

    @property
    def rate_lags(self) -> torch.Tensor:
        """Hybrid-history segment lengths in normalised time. Fixed, not learned."""
        return self._rate_lags

    @property
    def src_gain(self) -> torch.Tensor:
        return torch.exp(self.log_src_gain)

    @property
    def diff_gain(self) -> torch.Tensor:
        return torch.exp(self.log_diff_gain)

    def gates(self) -> torch.Tensor:
        """All-ones: every history channel is always on. There is no lag gating.

        Kept as a method (rather than dropped) so the training log, ``metrics.txt``
        and the benchmark checkpoints keep a stable, k_max-shaped field.
        """
        return torch.ones(self.k_max, dtype=self._delta.dtype, device=self._delta.device)

    def _causal(self, tn_q: torch.Tensor, t_query: torch.Tensor,
                dtn: float) -> torch.Tensor:
        """Clamp a history query to at most one full grid step before ``tn_q``.

        The history must never read at or after the time being predicted. That
        used to be guaranteed implicitly, by handing the recurrence the truncated
        view ``buf[:ti]`` so a too-recent query simply clamped to the last row.
        The training loop now hands over the whole frozen rollout buffer -- one
        rollout, many minibatch steps -- and in that buffer the row at ``tn_q``
        exists, so an unclamped lookup with ``delta_grid < dt`` would interpolate
        the very value the data term is fitting. Clamping here restores the old
        semantics for every caller and every buffer view.
        """
        return torch.minimum(t_query, tn_q - float(dtn))

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

rate_lags are CUMULATIVE segment lengths (not absolute boundaries), and each
        segment starts where the previous one ended. Each rate is divided by its
        own segment length -- the actual distance between the two points being
        differenced. With delta_grid=1s and rate_lags=[5, 20]:
            Anchor: T(t-1)
            Rate 1: (T(t-1)  - T(t-6))  / 5
            Rate 2: (T(t-6)  - T(t-26)) / 20

        ``delta_grid`` shifts where the whole window sits but is not part of any
        span: the endpoints of rate 1 are 5 s apart no matter how far back the
        anchor is.

        Per-endpoint padding: T(t) := T(0) if t < 0.

        The rate is divided by the NOMINAL segment length, never by the clamped
        elapsed span. Dividing by the clamped span looks more "honest" early in
        the rollout, but it is a singularity: at step 2 the clamped span is one
        grid step (dtn = 1.4e-4 normalised at Δt = 0.2 s), so the one-step
        prediction difference gets amplified ~7000x, fed back into the net, and
        the rollout diverges to inf -> nan within a handful of steps. Using the
        nominal length instead spreads a partially-filled window over its full
        segment, which damps the rate towards 0 exactly where the history is
        unknown and converges to the true rate once the window has filled.
        """
        if self.k_max == 0:
            return tn_q.new_zeros((tn_q.shape[0], 0))

        dgrid = self._delta_grid.to(tn_q.dtype)
        # Causal anchor; every rate segment then runs backwards from here, so
        # clamping this one point makes the whole block causal.
        t_anchor = self._causal(tn_q, tn_q - dgrid, dtn)

        # Walk the boundary points ONCE. The segments are disjoint and cascade, so
        # the anchor is also segment 0's upper endpoint and every segment's lower
        # endpoint is the next segment's upper one -- looking each one up per
        # segment meant 1 + 2k lookups for 1 + k distinct times. Same times, same
        # order of subtraction, so the values are unchanged bit for bit.
        t_boundary = t_anchor  # start at t - delta_grid (the anchor point)
        T_bounds = [self._padded_lookup(Tn_seq, dtn, t_boundary, p_idx)]
        spans = []
        for i in range(len(rate_lags)):
            seg_len = rate_lags[i]
            t_boundary = t_boundary - seg_len  # cumulative: subtract segment length
            T_bounds.append(self._padded_lookup(Tn_seq, dtn, t_boundary, p_idx))

            # Span = the segment's own length. That is exactly how far apart the
            # two endpoints of this difference are, so it is the divisor that
            # turns the difference into a rate. delta_grid only shifts WHERE the
            # window sits; it is not part of the window. Floored at one grid
            # step: a span below the time resolution is not resolvable.
            spans.append(torch.clamp(seg_len, min=float(dtn)))

        # Normalised d T / d t: rate_scale keeps this channel O(1) so it sits on
        # the same scale as the z-scored anchor channel next to it.
        rates = [
            (T_bounds[i] - T_bounds[i + 1]) / (spans[i] * self.rate_scale)
            for i in range(len(rate_lags))
        ]
        return torch.cat([T_bounds[0].unsqueeze(1), torch.stack(rates, dim=1)], dim=1)

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
            tq = self._causal(tn_q, tn_q - i * delta, dtn)
            cols.append(self._padded_lookup(Tn_seq, dtn, tq, p_idx))
        return torch.stack(cols, dim=1)

    def level(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
    ) -> torch.Tensor | None:
        """Spatially CONSTANT reference level of the field, one value per query time.

        This is the spatial mean of the anchor slice ``T(t - delta_grid)``, read
        with the same interpolation as the history channels. ``field`` adds it
        back, so the network only has to produce the deviation from the current
        temperature level instead of re-deriving the absolute level at every one
        of the ~7000 rollout steps. That is what keeps the free-running rollout
        from drifting: the level is carried, not re-predicted.

        Why the SPATIAL MEAN and not the per-point anchor ``hist[:, 0]``:
        ``physics.py`` takes the Laplacian of ``field``'s output by autograd with
        respect to ``xn``. A per-point anchor is read from a discrete buffer and
        is therefore invisible to autograd, so ``nabla^2 T`` would silently come
        back as the Laplacian of the deviation alone -- missing the anchor's own
        curvature, which is most of it. A spatially constant level has Laplacian
        zero exactly, so the residual and the ``dT/dx = 0`` boundary term stay
        correct with no correction term. It also carries the drift-prone part:
        what wanders over a long rollout is the overall level, not the shape.

        Returns ``None`` when the residual parameterisation is off, which makes
        ``field`` fall back to predicting the absolute value.
        """
        if not self.residual_output:
            return None
        dgrid = self._delta_grid.to(tn_q.dtype)
        mean_seq = Tn_seq.mean(dim=1, keepdim=True)     # (n_t, 1)
        p_zero = torch.zeros_like(tn_q, dtype=torch.long)
        t_anchor = self._causal(tn_q, tn_q - dgrid, dtn)
        return self._padded_lookup(mean_seq, dtn, t_anchor, p_zero)

    def field(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        hist: torch.Tensor,
        level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = torch.cat([xn, static, cfg, forcing, hist], dim=1)
        out = self.mlp(feats).squeeze(-1)
        if level is None:
            return out
        return level + out

    def field_with_time(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        hist: torch.Tensor,
        t: torch.Tensor,
        level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_col = t.unsqueeze(-1) if t.dim() == 1 else t
        feats = torch.cat([xn, static, cfg, forcing, hist, t_col], dim=1)
        out = self.mlp_with_time(feats).squeeze(-1)
        if level is None:
            return out
        return level + out

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

    def rollout_plan(self, tn: torch.Tensor, dtn: float) -> dict:
        """Precomputed bracketing rows/weights for every rollout step.

        A rollout query is special in a way the general path cannot assume: the
        time is ONE scalar broadcast over all P points and ``p_idx`` is
        ``arange(P)``. So ``interp_history``'s bracketing indices do not depend on
        the point at all -- ``lo``/``hi`` collapse to whole ROWS of the buffer and
        ``frac`` to a single scalar. Both are pure functions of the step index on a
        uniform grid, yet the general path recomputed the whole clamp/floor/gather
        chain ``k_max`` times per step, every step, every epoch, every OP.

        This hoists that arithmetic out of the loop and evaluates it once.

        Bit-exactness is deliberate, not incidental: the tables are built by
        running the SAME expressions the general path runs, in the same dtype and
        with the same rounding at every step -- successive
        subtraction for the hybrid boundaries (never a pre-summed offset, which is
        algebraically equal but a different sequence of rounding steps) and the
        same clamp order. ``tests/test_history_fastpath.py`` asserts equality with
        ``torch.equal``, not ``allclose``.

        Causality: the general path passes the SLICE ``buf[:ti]``, and that slice
        is the only thing stopping a step from reading its own future. The fast
        path reads the whole buffer, so the bound moves into ``cap = ti - 1`` here.
        A wrong cap would leak future temperature into the history and quietly
        make training look better, so the test asserts the bound directly.

        The plan is cached and rebuilt whenever the layout it was derived from
        changes (n_t differs per OP, and the benchmarks sweep rate_lags).
        """
        device = self._delta.device
        n_t = int(tn.shape[0])
        key = (
            n_t, float(dtn), str(device), str(tn.dtype), self.history_mode,
            int(self.k_max), float(self._delta), float(self._delta_grid),
            tuple(float(v) for v in self._rate_lags),
        )
        cached = getattr(self, "_hist_plan_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        # Follow ``tn``'s dtype, not the module's. The general path derives its
        # query times from tn_q = tn[ti], so tying the plan to the parameter dtype
        # instead would silently diverge for a model whose weights and time grid
        # are not the same precision.
        tn_c = tn.to(device=device)
        # Every query the general path builds runs through ``_causal`` first, so
        # the plan has to as well -- and through the SAME expression, because the
        # equality asserted by tests/test_history_fastpath.py is torch.equal and
        # not allclose. With ``delta_grid < dtn`` the clamp is what stops the
        # anchor from reading the very row being predicted; without it here, the
        # fast path would read a row the general path never would.
        if self.history_mode == "hybrid":
            # Same walk as _history_hybrid, vectorised over all steps at once:
            # elementwise ops, so per-element results are untouched, and the
            # subtraction stays SUCCESSIVE rather than becoming a pre-summed offset.
            t_boundary = self._causal(tn_c, tn_c - self._delta_grid.to(tn_c.dtype), dtn)
            times = [t_boundary]
            for i in range(self._n_lags):
                t_boundary = t_boundary - self._rate_lags[i]
                times.append(t_boundary)
            # _padded_lookup's clamp: T(t) := T(0) for t < 0.
            times = [torch.clamp(t, min=0.0) for t in times]
            denoms = [
                torch.clamp(self._rate_lags[i], min=float(dtn)) * self.rate_scale
                for i in range(self._n_lags)
            ]
        else:
            times = [self._causal(tn_c, tn_c - i * self._delta, dtn)
                     for i in range(1, self.k_max + 1)]
            denoms = []
        dtype = times[0].dtype if times else tn_c.dtype

        plan: dict = {"n_off": len(times), "denoms": denoms}
        if not times:
            self._hist_plan_cache = (key, plan)
            return plan

        # ``cap`` is float(n_t - 1) of the SLICE the general path would have seen,
        # i.e. ti - 1. Step 0 never queries; clamping keeps its row well-formed.
        steps = torch.arange(n_t, device=device)
        cap_i = (steps - 1).clamp(min=0).unsqueeze(1)
        cap_f = cap_i.to(dtype)

        tq = torch.stack(times, dim=1)                       # (n_t, n_off)
        pos = torch.minimum(torch.clamp(tq / dtn, min=0.0), cap_f)
        lo = torch.minimum(torch.floor(pos).long().clamp(min=0), cap_i)
        hi = torch.minimum(lo + 1, cap_i)
        plan["lo"], plan["hi"] = lo, hi
        plan["frac"] = pos - lo.to(dtype)

        self._hist_plan_cache = (key, plan)
        return plan

    def history_rollout(self, buf: torch.Tensor, ti: int, plan: dict) -> torch.Tensor:
        """History block at rollout step ``ti``, using a :meth:`rollout_plan`.

        Equivalent to ``self._history(buf[:ti], dtn, tn[ti].expand(P), arange(P))``
        but without rebuilding the index arithmetic, and reading rows of ``buf``
        directly instead of running a two-index gather whose indices are constant
        across the batch.
        """
        n_off = plan["n_off"]
        if n_off == 0:
            return buf.new_zeros((buf.shape[1], 0))

        lo, hi, frac = plan["lo"], plan["hi"], plan["frac"]
        vals = []
        for j in range(n_off):
            # frac and the row indices stay 0-dim TENSORS. Calling ``.item()`` /
            # ``float()`` on them would be numerically identical (the values round
            # the same either way) but forces a device-to-host sync per channel per
            # step -- on CUDA that alone would undo the point of the fast path.
            f = frac[ti, j]
            vals.append(buf[lo[ti, j]] * (1.0 - f) + buf[hi[ti, j]] * f)

        if self.history_mode != "hybrid":
            return torch.stack(vals, dim=1)

        denoms = plan["denoms"]
        rates = [(vals[i] - vals[i + 1]) / denoms[i] for i in range(len(denoms))]
        return torch.cat([vals[0].unsqueeze(1), torch.stack(rates, dim=1)], dim=1)

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
        return self.field(xn, static, cfg, forcing, hist,
                          self.level(Tn_seq, dtn, tn_q))

    def history_at(
        self,
        Tn_seq: torch.Tensor,
        dtn: float,
        tn_q: torch.Tensor,
        p_idx: torch.Tensor,
        lag: int = 1,
    ) -> torch.Tensor:
        """Ungated history at a single ``lag`` (used by the FD time derivative)."""
        tq = self._causal(tn_q, tn_q - lag * self.delta, dtn)
        return self._padded_lookup(Tn_seq, dtn, tq, p_idx)


# ``rollout_train`` -- a gradient-keeping twin of ``rollout`` -- used to live here.
# It detached its history between steps anyway (truncated BPTT), so the gradient at
# time t never left that step's own field evaluation: the whole differentiable
# rollout produced exactly the gradient a minibatch of (t, point) pairs against a
# FROZEN trajectory produces, at ~7000 sequential steps for a single optimiser
# update. ``train.py`` now takes that cheaper equivalent -- one ``rollout`` under
# no_grad per OP per epoch, then ``--inner-steps`` minibatch updates against it.


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
    plan = model.rollout_plan(tn, dtn)
    for ti in range(1, n_t):
        tq = tn[ti].expand(P)
        hist = model.history_rollout(buf, ti, plan)
        cfg = cfg_seq[ti].expand(P, -1)
        forcing = forcing_seq[ti].expand(P, -1)
        buf[ti] = model.field(xn, static, cfg, forcing, hist,
                              model.level(buf[:ti], dtn, tq))
    return buf
