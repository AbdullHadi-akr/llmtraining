"""A convolutional surrogate for the same field, on the same recurrence.

Why a CNN is applicable here at all
-----------------------------------
The 363 points are a structured ``3 x 11 x 11`` raster (``grid.py`` derives and
checks that from the coordinates). The quantity being predicted is therefore an
image, not a point cloud, and heat conduction -- the process generating it -- is
LOCAL: a point's next temperature depends on its neighbours and on the source at
that point. A 3x3 convolution is exactly that stencil, with the locality built
in rather than learned.

The MLP in ``model.py`` has to discover locality from raw coordinates: it sees
one point at a time and has no way of knowing that two points are adjacent
except by learning the metric from ``xn``. With 16 operating points that is a lot
to ask of the data. A convolution is handed it.

What is deliberately kept identical to ``RecurrentField``
---------------------------------------------------------
So that a measured difference is attributable to the ARCHITECTURE and not to a
richer input, this model is fed the same features, per point, that the MLP gets:
``[xn(3), static(S), config(C), forcing(F), history(k)]``. The scalars (config,
forcing) are broadcast across the image, the per-point ones are channels. No
extra channel -- not even the obvious one, the volumetric source field ``Qsrc``,
which the MLP currently has to reconstruct from ``q_dot`` and the JR1 indicator.
Adding it is a real and probably worthwhile experiment; it is not this one, and
mixing the two would make the comparison unreadable.

Everything about the RECURRENCE is inherited unchanged from ``RecurrentField``:
the history layout (raw or hybrid), ``delta`` / ``delta_grid`` / ``rate_lags`` as
fixed buffers, the causality clamp, the ``rollout_plan`` fast path, the
spatially-constant residual level, and the free-running rollout in
``model.rollout`` -- which works on this class verbatim, because
:meth:`ConvRecurrentField.field` keeps the same signature.

What necessarily changes
------------------------
The MLP is a function of continuous coordinates, so ``physics.py`` can take the
Laplacian by autograd at any point. A convolution cannot be differentiated that
way. It does read ``xn`` -- as three input CHANNELS, so that it sees exactly what
the MLP sees -- and autograd therefore returns a finite, plausible-looking
number. That number is not the spatial Laplacian: it is how the prediction
responds to RELABELLING a pixel's coordinates while its neighbours' temperatures
stay put. The conduction the residual is about lives in the kernel, and this
derivative never touches it. Nothing raises and ``L_phys`` falls like any other
run, which is the worst kind of wrong, so ``physics.py`` refuses a model with a
``grid`` attribute outright (``_reject_grid_model``) and ``physics_grid.py``
differences the lattice instead. Same reason this class refuses
``time_deriv='autograd'``: there is no continuous time input either.

Layout of the network
---------------------
``(B, F, nx, ny, nz)`` is folded to ``(B, nx*F, ny, nz)`` and run through 2-D
convolutions over ``(y, z)``:

* **y and z** get 3x3 kernels. 11 levels each, genuinely local, and after four
  layers the receptive field is 9 of 11 -- close to global without a single
  pooling step (pooling an 11x11 image away would be silly).
* **x** is folded into the channel dimension, so every layer is fully connected
  across the three planes. With three levels there is no locality to exploit and
  a 3-tap kernel in x would be a dense map wearing a stencil's clothes.

Edge padding is ``replicate`` by default: zero padding would assert that the
temperature drops to the normalised zero point one cell outside the cell wall,
which is a fabricated Dirichlet boundary. Replication asserts zero gradient
instead -- still an assumption, but the neutral one.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from grid import GridSpec
from model import LearnableSwish, RecurrentField


class ConvStack(nn.Module):
    """Conv net over the (y, z) raster, x folded into channels.

    Input  ``(B, nx * in_per_plane, ny, nz)`` -> output ``(B, nx, ny, nz)``.

    The activation is the same per-layer learnable swish the MLP uses, so the two
    architectures differ in their connectivity and nothing else.
    """

    def __init__(
        self,
        in_per_plane: int,
        nx: int,
        width: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so padding keeps the shape")
        self.nx = int(nx)
        self.in_per_plane = int(in_per_plane)
        pad = kernel_size // 2

        self.hidden = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev = self.nx * self.in_per_plane
        for _ in range(num_layers):
            self.hidden.append(
                nn.Conv2d(prev, width, kernel_size, padding=pad,
                          padding_mode=padding_mode)
            )
            self.acts.append(LearnableSwish(beta_init))
            prev = width
        # 1x1 head: the last mixing across channels (and so across x-planes)
        # without widening the receptive field, which the stack above has already
        # set on purpose.
        self.out = nn.Conv2d(prev, self.nx, 1)

        for conv in list(self.hidden) + [self.out]:
            # Same initialisation policy as the Modulus FCLayer the MLP uses:
            # xavier weights, zero bias. Not cosmetic -- the untrained rollout's
            # boundedness depends on the per-layer gain, and the default
            # kaiming_uniform(a=sqrt(5)) is markedly less expansive.
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    @property
    def receptive_field(self) -> int:
        """Half-width reach in (y, z), in cells, of the whole stack."""
        return sum((conv.kernel_size[0] - 1) // 2 for conv in self.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.hidden, self.acts):
            x = act(conv(x))
        return self.out(x)

    def betas(self) -> list[float]:
        return [float(a.beta.detach()) for a in self.acts]


class ConvRecurrentField(RecurrentField):
    """``RecurrentField`` with the MLP swapped for :class:`ConvStack`.

    Only the function approximator changes. ``field`` keeps the per-point
    signature so ``model.rollout`` drives this class unchanged, and every history
    method is inherited.

    ``field`` is a WHOLE-FIELD operator: a convolution has no meaning on an
    arbitrary subset of pixels, so the per-point call must cover all ``P`` points
    of one time step, in the canonical point order -- exactly what ``rollout``
    passes. Callers that want several time steps at once (the training loop, and
    ``physics_grid``) should use :meth:`field_batch`, which is the same
    computation without the ``B = P`` reshape dance.
    """

    def __init__(
        self,
        grid: GridSpec | dict,
        n_config: int,
        n_static: int = 0,
        n_forcing: int = 0,
        k_max: int = 2,
        history_mode: str = "raw",
        rate_lags: Sequence[float] = (5.0, 25.0),
        width: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        delta_seconds: float = 1.0,
        dtn: float = 1.0,
        t_span_ref: float = 1.0,
        rate_scale: float = 1.0,
        delta_grid: float | None = None,
        beta_init: float = 1.0,
        residual_output: bool = False,
        learn_gains: bool = False,
    ) -> None:
        # The parent builds the MLP backbone and every buffer the recurrence
        # needs. The buffers are the point; the MLP is replaced immediately
        # below, with ``layer_size=1, num_layers=1`` so the discarded one costs a
        # handful of weights rather than the full 70k.
        super().__init__(
            n_config=n_config, n_static=n_static, n_forcing=n_forcing,
            k_max=k_max, history_mode=history_mode, rate_lags=rate_lags,
            layer_size=1, num_layers=1,
            delta_seconds=delta_seconds, dtn=dtn, t_span_ref=t_span_ref,
            rate_scale=rate_scale, delta_grid=delta_grid, weight_norm=False,
            beta_init=beta_init, use_autograd_time=False,
            residual_output=residual_output, learn_gains=learn_gains,
        )
        del self.mlp
        self.mlp = None
        self.mlp_with_time = None

        # A checkpoint stores the spec as plain data, so that reloading stays
        # the documented one-liner ``ConvRecurrentField(**ckpt["model_config"])``
        # rather than a two-step dance the caller has to remember.
        if isinstance(grid, dict):
            grid = GridSpec.from_dict(grid)
        self.grid = grid
        # Buffers, not plain attributes: the permutation has to follow the model
        # onto the GPU and into the checkpoint's device move.
        self.register_buffer("_to_points", grid.to_points.clone())
        self.register_buffer("_to_raster", grid.to_raster.clone())

        # 3 coords + static + config + forcing + history, per point -- the SAME
        # block ``RecurrentField.field`` concatenates.
        self.in_per_plane = 3 + n_static + n_config + n_forcing + self.k_max
        self.width = int(width)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.padding_mode = str(padding_mode)
        self.net = ConvStack(
            in_per_plane=self.in_per_plane, nx=grid.nx, width=width,
            num_layers=num_layers, kernel_size=kernel_size,
            padding_mode=padding_mode, beta_init=beta_init,
        )

    # -- shape helpers ------------------------------------------------------
    @property
    def n_points(self) -> int:
        return self.grid.n_points

    def _as_grid(self, values: torch.Tensor) -> torch.Tensor:
        """(..., P) -> (..., nx, ny, nz), using the buffered permutation."""
        lead = values.shape[:-1]
        g = self.grid
        return values[..., self._to_points].reshape(*lead, g.nx, g.ny, g.nz)

    def _as_points(self, grid_vals: torch.Tensor) -> torch.Tensor:
        """(..., nx, ny, nz) -> (..., P)."""
        lead = grid_vals.shape[:-3]
        return grid_vals.reshape(*lead, self.n_points)[..., self._to_raster]

    # -- the network --------------------------------------------------------
    def field_batch(
        self,
        xn: torch.Tensor,        # (P, 3)
        static: torch.Tensor,    # (P, n_static)
        cfg: torch.Tensor,       # (B, n_config)   -- one row per time
        forcing: torch.Tensor,   # (B, n_forcing)  -- one row per time
        hist: torch.Tensor,      # (B, P, k_max)
        level: torch.Tensor | None = None,   # (B,) or None
    ) -> torch.Tensor:
        """Predict the whole field at ``B`` times at once. Returns ``(B, P)``.

        This is the natural call for a grid model and the efficient one: the data
        term gets all P labels per forward pass instead of the one pixel a
        pointwise minibatch would use.
        """
        B, P = hist.shape[0], self.n_points
        if xn.shape[0] != P or static.shape[0] != P:
            raise ValueError(
                f"a convolution needs the whole field: expected {P} points, got "
                f"xn={xn.shape[0]}, static={static.shape[0]}"
            )
        if hist.shape[1] != P:
            raise ValueError(f"history must cover all {P} points, got {hist.shape}")

        per_point = torch.cat([xn, static], dim=1).unsqueeze(0).expand(B, -1, -1)
        per_time = torch.cat([cfg, forcing], dim=1).unsqueeze(1).expand(-1, P, -1)
        feats = torch.cat([per_point, per_time, hist], dim=2)   # (B, P, F)

        g = self.grid
        # (B, P, F) -> (B, F, P) -> (B, F, nx, ny, nz) -> (B, nx, F, ny, nz)
        # -> (B, nx*F, ny, nz): the channel axis runs x-plane-major, so each
        # plane's feature block stays contiguous.
        img = self._as_grid(feats.transpose(1, 2))
        img = img.permute(0, 2, 1, 3, 4).reshape(B, g.nx * feats.shape[2], g.ny, g.nz)

        out = self.net(img)                          # (B, nx, ny, nz)
        pred = self._as_points(out)                  # (B, P)
        if level is None:
            return pred
        return level.reshape(B, 1) + pred

    def field(
        self,
        xn: torch.Tensor,
        static: torch.Tensor,
        cfg: torch.Tensor,
        forcing: torch.Tensor,
        hist: torch.Tensor,
        level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One time step over the whole field -- the signature ``rollout`` calls.

        ``cfg``/``forcing``/``level`` arrive with one identical row per point
        (``rollout`` expands them), so row 0 carries the whole information and the
        rest is dropped rather than fed to the net P times over.
        """
        pred = self.field_batch(
            xn, static, cfg[:1], forcing[:1], hist.unsqueeze(0),
            None if level is None else level.reshape(-1)[:1],
        )
        return pred.squeeze(0)

    def field_with_time(self, *args, **kwargs):  # pragma: no cover - guarded path
        raise NotImplementedError(
            "the convolutional model has no continuous-time input: "
            "--time-deriv autograd is not available for --arch cnn. "
            "Use bdf1 or bdf2, which difference the recurrence itself."
        )

    def betas(self) -> list[float]:
        return self.net.betas()

    def extra_repr(self) -> str:  # pragma: no cover - logging only
        g = self.grid
        return (f"grid={g.nx}x{g.ny}x{g.nz} in_per_plane={self.in_per_plane} "
                f"width={self.width} layers={self.num_layers} "
                f"kernel={self.kernel_size} pad={self.padding_mode}")
