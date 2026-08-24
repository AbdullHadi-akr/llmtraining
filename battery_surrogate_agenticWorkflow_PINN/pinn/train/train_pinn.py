"""OP01 PINN trainer (plan 004).

Design decisions (all reflected below and printed at runtime):
- SEPARATE nets: net_T (temperature, recurrent) and net_V (voltage, non-recurrent).
- NO TEACHER FORCING: net_T is rolled out autoregressively on its OWN predictions
  T_hat; the warm-up history is the OFFICIAL measured first sample (t = 0.1 s).
- Truncated BPTT: gradients through the predicted history flow back at most W steps.
- IC: the official initial condition is the MEASURED first sample T[0] at t = 0.1 s
  (per grid point), treated as t~ = 0 (Notion: IC = true temperature from the data).
    * hard IC:  T~ = T~_ic + t~ * N(...)   -> exact at t~=0.
    * soft IC:  penalize (T~(t~=0) - T~_ic)^2.
- Physics: anisotropic solid heat equation (Notion-faithful):
      r = d_t~ T~ - sum_ij Fo_ij d^2_{x~_i x~_j} T~ - Q~
  with the full symmetric lambda tensor; evaluated on sampled collocation timesteps
  using the true (detached) history. (--iso-physics gives an isotropic debug fallback.)
- BC: DISABLED for this run (w_bc = 0). Reported and plotted as 0 so it is visible.
- Normalization: xyz, t -> [0, 1]; outputs T, bc_V (+ config) -> z-score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.preprocess import NormStats
from ..models.net_T import NetT


@dataclass
class LossWeights:
    """Loss weights. BC is 0 for this run (BC not included)."""

    w_data: float = 1.0
    w_phys: float = 0.1
    w_ic: float = 1.0
    w_bc_in: float = 0.0   # BC NOT included
    w_bc_out: float = 0.0  # BC NOT included


@dataclass
class LossHistory:
    """Per-epoch loss component values (for plotting every loss)."""

    data: List[float] = field(default_factory=list)
    phys: List[float] = field(default_factory=list)
    ic: List[float] = field(default_factory=list)
    bc_in: List[float] = field(default_factory=list)
    bc_out: List[float] = field(default_factory=list)
    total: List[float] = field(default_factory=list)


class TemperaturePINN:
    """net_T trainer with autoregressive rollout (no teacher forcing) + physics."""

    def __init__(
        self,
        t: np.ndarray,
        xyz: np.ndarray,
        T_labels: np.ndarray,
        q_dot: np.ndarray,
        rho: np.ndarray,
        Cp: np.ndarray,
        lambda_tensor: np.ndarray,
        region: np.ndarray,
        stats: NormStats,
        weights: LossWeights,
        depth: int = 4,
        width: int = 128,
        k: int = 2,
        bptt_window: int = 8,
        hard_ic: bool = True,
        lr: float = 1e-3,
        iso_physics: bool = False,
        phys_batch: int = 24,
        phys_per_win: int = 2,
        phys_points: int = 160,
        split_t: int = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.k = k
        self.W = bptt_window
        self.hard_ic = hard_ic
        self.weights = weights
        self.iso_physics = iso_physics
        self.phys_batch = phys_batch
        self.phys_per_win = phys_per_win
        self.phys_points = phys_points

        self.n_t = len(t)
        self.n_points = xyz.shape[0]
        self.split_t = split_t if split_t is not None else int(0.8 * self.n_t)

        # ---- tensors ----
        def T(a, dtype=torch.float32):
            return torch.as_tensor(a, dtype=dtype, device=self.device)

        self.stats = stats
        xyz_min = T(stats.xyz_min)
        self.L_axis = T(stats.L_axis)                         # (3,)
        self.T_mu, self.T_sigma = stats.T_mu, stats.T_sigma
        self.t0, self.t_max = stats.t0, stats.t_max
        self.T_span = self.t_max - self.t0                    # duration mapped to t~ in [0,1]

        # coordinates: ISOTROPIC scaling by a single reference length L_ref (geometric mean
        # of the axis extents). Per-axis [0,1] scaling made the diffusion tensor Fo_xx ~ 241
        # (thin x-axis, 1/L_x^2) -> stiff, ill-conditioned physics. With one L_ref, Fo ~ O(1)
        # and anisotropy comes only from the physical conductivity tensor lambda_ij.
        self.L_ref = float(np.prod(np.asarray(stats.L_axis)) ** (1.0 / 3.0))
        self.xn = (T(xyz) - xyz_min) / self.L_ref                     # (n_points, 3)
        self.tn = (T(t) - self.t0) / (self.T_span + 1e-12)            # (n_t,)

        # z-scored labels and official IC
        self.Tn_labels = (T(T_labels) - self.T_mu) / self.T_sigma     # (n_t, n_points)
        self.Tn_ic = (T(np.asarray(stats.T_ic)) - self.T_mu) / self.T_sigma  # (n_points,)

        # config normalized (constant for one OP -> zeros)
        cfg_mu = T(stats.config_mu)
        cfg_sig = T(stats.config_sigma)
        cfg_raw = cfg_mu  # for a single OP the raw config equals its mean
        self.cfg_n = ((cfg_raw - cfg_mu) / cfg_sig).reshape(1, -1)    # (1, 7)

        # physics coefficients (per point)
        self.rho = T(rho)
        self.Cp = T(Cp)
        self.lam = T(lambda_tensor)                                   # (n_points, 3, 3)
        self.q_dot = T(q_dot)                                         # (n_t,)
        # heat source is a JELLY-ROLL quantity (q_dot = Q_JR1 / V_JR1): active ONLY
        # in JR1 (region 1). Cell-center (0) and housing (2) generate no heat -> q=0.
        self.q_mask = T((np.asarray(region) == 1).astype(np.float32)) # (n_points,)

        # Fourier tensor with a SINGLE reference length: Fo_ij = lam_ij * T_span / (rho*Cp*L_ref^2)
        # -> O(1), anisotropy carried only by the physical conductivity tensor lam_ij.
        rc = (self.rho * self.Cp).reshape(-1, 1, 1)
        self.Fo = self.lam * self.T_span / (rc * (self.L_ref ** 2) + 1e-30)
        lam_eff = (self.lam[:, 0, 0] + self.lam[:, 1, 1] + self.lam[:, 2, 2]) / 3.0
        self.Fo_iso = lam_eff * self.T_span / (self.rho * self.Cp * (self.L_ref ** 2) + 1e-30)

        # characteristic dT/dt~ magnitude (from measured labels) to normalize the residual -> O(1)
        dTdt_true = (self.Tn_labels[2:] - self.Tn_labels[:-2]) / (self.tn[2:] - self.tn[:-2]).unsqueeze(1)
        self.phys_scale = float(dTdt_true.pow(2).mean().sqrt().item()) + 1e-6

        # ---- model ----
        self.net = NetT(depth=depth, width=width, k=k, n_config=7).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.history = LossHistory()

    # ------------------------------------------------------------------ helpers
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward with optional hard-IC anchoring. Returns (n_points,) normalized T."""
        raw = self.net(inp).squeeze(-1)
        if self.hard_ic:
            # T~ = T~_ic + t~ * N  -> exact IC at t~=0 (t~ is input column 3)
            return self.Tn_ic + inp[:, 3] * raw
        return raw

    def _assemble(self, ti: int, hist: torch.Tensor) -> torch.Tensor:
        """Build input row (n_points, 4+7+k) for time index ti and history (n_points,k)."""
        x = self.xn[:, 0:1]
        y = self.xn[:, 1:2]
        z = self.xn[:, 2:3]
        tt = self.tn[ti].expand(self.n_points, 1)
        cfg = self.cfg_n.expand(self.n_points, -1)
        return torch.cat([x, y, z, tt, cfg, hist], dim=1)

    # ------------------------------------------------------------------ physics
    def _field_value(self, ti: int, idx: torch.Tensor) -> torch.Tensor:
        """Predicted normalized T at time index ti on point subset idx (no spatial grad).

        Keeps grad w.r.t. net parameters so the finite-difference time derivative trains net.
        """
        m = idx.shape[0]
        cfg = self.cfg_n.expand(m, -1)
        tt = self.tn[ti].expand(m, 1)
        hist = self._true_history(ti)[idx].detach()
        xn = self.xn[idx]
        inp = torch.cat([xn[:, 0:1], xn[:, 1:2], xn[:, 2:3], tt, cfg, hist], dim=1)
        raw = self.net(inp).squeeze(-1)
        return self.Tn_ic[idx] + tt.squeeze(-1) * raw if self.hard_ic else raw

    def _physics_residual(self, ti: int) -> torch.Tensor:
        """Non-dimensional PDE residual at collocation time ti on a random point subset.

        Time derivative is a FINITE DIFFERENCE along the trajectory (total dT/dt, consistent
        with the discrete recurrent model), NOT autograd on the explicit t input (which would be
        a partial derivative that ignores the history channels). Space uses autograd on the
        isotropically-scaled coordinates so the diffusion tensor Fo is O(1). Subsampling points
        keeps the second-order autograd cheap (works the same on CPU or GPU).
        """
        m = min(self.phys_points, self.n_points)
        idx = torch.randperm(self.n_points, device=self.device)[:m]
        xn = self.xn[idx]
        x = xn[:, 0:1].clone().requires_grad_(True)
        y = xn[:, 1:2].clone().requires_grad_(True)
        z = xn[:, 2:3].clone().requires_grad_(True)
        tt = self.tn[ti].expand(m, 1)
        cfg = self.cfg_n.expand(m, -1)
        hist = self._true_history(ti)[idx].detach()
        inp = torch.cat([x, y, z, tt, cfg, hist], dim=1)

        raw = self.net(inp).squeeze(-1)
        Tn = self.Tn_ic[idx] + tt.squeeze(-1) * raw if self.hard_ic else raw

        ones = torch.ones_like(Tn)
        Tn_x = torch.autograd.grad(Tn, x, ones, create_graph=True)[0]
        Tn_y = torch.autograd.grad(Tn, y, ones, create_graph=True)[0]
        Tn_z = torch.autograd.grad(Tn, z, ones, create_graph=True)[0]
        o = torch.ones_like(Tn_x)
        Tn_xx = torch.autograd.grad(Tn_x, x, o, create_graph=True)[0].squeeze(-1)
        Tn_yy = torch.autograd.grad(Tn_y, y, o, create_graph=True)[0].squeeze(-1)
        Tn_zz = torch.autograd.grad(Tn_z, z, o, create_graph=True)[0].squeeze(-1)

        if self.iso_physics:
            div = self.Fo_iso[idx] * (Tn_xx + Tn_yy + Tn_zz)
        else:
            Tn_xy = torch.autograd.grad(Tn_x, y, o, create_graph=True)[0].squeeze(-1)
            Tn_xz = torch.autograd.grad(Tn_x, z, o, create_graph=True)[0].squeeze(-1)
            Tn_yz = torch.autograd.grad(Tn_y, z, o, create_graph=True)[0].squeeze(-1)
            Fo = self.Fo[idx]
            div = (
                Fo[:, 0, 0] * Tn_xx
                + Fo[:, 1, 1] * Tn_yy
                + Fo[:, 2, 2] * Tn_zz
                + 2.0 * (Fo[:, 0, 1] * Tn_xy + Fo[:, 0, 2] * Tn_xz + Fo[:, 1, 2] * Tn_yz)
            )

        # finite-difference TOTAL time derivative (central where possible), same point subset
        tp = min(ti + 1, self.n_t - 1)
        tm = max(ti - 1, 0)
        dTdt = (self._field_value(tp, idx) - self._field_value(tm, idx)) / (self.tn[tp] - self.tn[tm] + 1e-12)

        # dimensionless heat source on the subset
        Q = self.q_dot[ti] * self.q_mask[idx] * self.T_span / (self.rho[idx] * self.Cp[idx] * self.T_sigma)

        return (dTdt - div - Q) / self.phys_scale

    def _true_history(self, ti: int) -> torch.Tensor:
        """(n_points, k) z-scored measured history for step ti; warm-up = official IC."""
        cols = []
        for j in range(1, self.k + 1):
            idx = ti - j
            cols.append(self.Tn_labels[idx] if idx >= 0 else self.Tn_ic)
        return torch.stack(cols, dim=1)

    # ------------------------------------------------------------------ epoch
    def train_epoch(self) -> Dict[str, float]:
        self.net.train()

        # DATA rollout (truncated BPTT) with PHYSICS folded into every window update, so
        # data and physics share one optimizer step and cannot fight across separate steps.
        hist = self.Tn_ic.unsqueeze(1).repeat(1, self.k)   # warm-up = official IC
        self.opt.zero_grad()
        win_loss = torch.zeros((), device=self.device)
        win = 0
        data_sum, data_cnt = 0.0, 0
        phys_sum, phys_cnt = 0.0, 0

        def phys_minibatch() -> torch.Tensor:
            lp = torch.zeros((), device=self.device)
            for _ in range(self.phys_per_win):
                tj = int(np.random.randint(1, self.split_t))
                lp = lp + torch.mean(self._physics_residual(tj) ** 2)
            return lp / self.phys_per_win

        def flush(win_loss, win):
            l_phys = phys_minibatch()
            loss = self.weights.w_data * win_loss / win + self.weights.w_phys * l_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()
            return float(l_phys.item())

        for ti in range(1, self.split_t):
            inp = self._assemble(ti, hist)
            Tn = self._forward(inp)
            l = torch.mean((Tn - self.Tn_labels[ti]) ** 2)
            win_loss = win_loss + l
            data_sum += l.item()
            data_cnt += 1
            win += 1
            hist = torch.cat([Tn.unsqueeze(1), hist[:, : self.k - 1]], dim=1)
            if win == self.W:
                phys_sum += flush(win_loss, win)
                phys_cnt += 1
                hist = hist.detach()
                win_loss = torch.zeros((), device=self.device)
                win = 0
        if win > 0:
            phys_sum += flush(win_loss, win)
            phys_cnt += 1

        # IC term: exact under hard IC (~0) -> only take a step for soft IC.
        inp0 = self._assemble(0, self.Tn_ic.unsqueeze(1).repeat(1, self.k))
        l_ic = torch.mean((self._forward(inp0) - self.Tn_ic) ** 2)
        if not self.hard_ic:
            self.opt.zero_grad()
            (self.weights.w_ic * l_ic).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()

        comp = {
            "data": data_sum / max(1, data_cnt),
            "phys": phys_sum / max(1, phys_cnt),
            "ic": float(l_ic.item()),
            "bc_in": 0.0,   # BC not included (w_bc_in = 0)
            "bc_out": 0.0,  # BC not included (w_bc_out = 0)
        }
        comp["total"] = (
            self.weights.w_data * comp["data"]
            + self.weights.w_phys * comp["phys"]
            + self.weights.w_ic * comp["ic"]
        )
        for key in ("data", "phys", "ic", "bc_in", "bc_out", "total"):
            getattr(self.history, key).append(comp[key])
        return comp

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def rollout_predict(self) -> np.ndarray:
        """Free-running rollout over ALL timesteps -> predicted T (physical), (n_t, n_points)."""
        self.net.eval()
        hist = self.Tn_ic.unsqueeze(1).repeat(1, self.k)
        preds = [self.Tn_ic.clone()]     # t index 0 = official IC
        for ti in range(1, self.n_t):
            inp = self._assemble(ti, hist)
            Tn = self._forward(inp)
            preds.append(Tn)
            hist = torch.cat([Tn.unsqueeze(1), hist[:, : self.k - 1]], dim=1)
        Tn_all = torch.stack(preds, dim=0)                 # (n_t, n_points)
        return (Tn_all * self.T_sigma + self.T_mu).cpu().numpy()


def train_temperature(
    trainer: TemperaturePINN,
    epochs: int,
    log_interval: int = 10,
    print_fn: Callable[[str], None] = print,
) -> None:
    for ep in range(1, epochs + 1):
        c = trainer.train_epoch()
        if ep == 1 or ep % log_interval == 0 or ep == epochs:
            print_fn(
                f"Epoch {ep:4d} | L_data={c['data']:.4e} L_phys={c['phys']:.4e} "
                f"L_IC={c['ic']:.4e} L_BCin={c['bc_in']:.1e} L_BCout={c['bc_out']:.1e} "
                f"| Total={c['total']:.4e}"
            )


# ---------------------------------------------------------------------- net_V
class VoltageNet(nn.Module):
    """Non-recurrent voltage net: inputs (t~, config[7]) -> bc_V (normalized)."""

    def __init__(self, depth: int = 2, width: int = 64, n_config: int = 7):
        super().__init__()
        from modulus.models.mlp import FullyConnected

        self.net = FullyConnected(
            in_features=1 + n_config,
            out_features=1,
            layer_size=width,
            num_layers=depth,
            activation_fn="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_voltage(
    t: np.ndarray,
    bc_V: np.ndarray,
    stats: NormStats,
    split_t: int,
    depth: int = 2,
    width: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    w_ic: float = 1.0,
    device: Optional[torch.device] = None,
    print_fn: Callable[[str], None] = print,
) -> Tuple[nn.Module, List[float], np.ndarray]:
    """Train net_V (data-only + soft IC at t0). Returns (model, loss_hist, V_pred_phys)."""
    device = device or torch.device("cpu")
    tn = torch.as_tensor((t - stats.t0) / (stats.t_max - stats.t0 + 1e-12), dtype=torch.float32, device=device)
    Vn = torch.as_tensor((bc_V - stats.V_mu) / stats.V_sigma, dtype=torch.float32, device=device)
    cfg = torch.zeros(1, 7, device=device)  # constant config -> zeros
    Vn_ic = torch.tensor((stats.V_ic - stats.V_mu) / stats.V_sigma, dtype=torch.float32, device=device)

    model = VoltageNet(depth=depth, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr = slice(0, split_t)
    inp_tr = torch.cat([tn[tr].unsqueeze(1), cfg.expand(split_t, -1)], dim=1)
    tgt_tr = Vn[tr]
    inp0 = torch.cat([tn[0:1].unsqueeze(1), cfg], dim=1)

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(inp_tr).squeeze(-1)
        l_data = torch.mean((pred - tgt_tr) ** 2)
        l_ic = torch.mean((model(inp0).squeeze(-1) - Vn_ic) ** 2)
        loss = l_data + w_ic * l_ic
        loss.backward()
        opt.step()
        hist.append(float(l_data.item()))
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print_fn(f"[net_V] Epoch {ep:4d} | L_data={l_data.item():.4e} L_IC={l_ic.item():.4e}")

    model.eval()
    with torch.no_grad():
        inp_all = torch.cat([tn.unsqueeze(1), cfg.expand(len(t), -1)], dim=1)
        V_pred = model(inp_all).squeeze(-1) * stats.V_sigma + stats.V_mu
    return model, hist, V_pred.cpu().numpy()
