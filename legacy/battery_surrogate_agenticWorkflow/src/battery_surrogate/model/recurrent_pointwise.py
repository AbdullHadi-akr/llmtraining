"""Recurrent point-wise surrogate model (GRU/LSTM).

Note: Recurrent models with per-target history (k_T != k_V) use a grouped history layout
([T-block | V-block] instead of interleaved). Existing checkpoints must be retrained.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .features_sequence import resolve_history_lengths


class RecurrentPointwise(nn.Module):
    """Sequence model that predicts [T, bc_V] from known inputs plus lag history.
    
    Supports per-target history lengths via config["model"]["history_length"] = {"T": k_T, "bc_V": k_V}
    or scalar k interpreted as k_T = k_V = k.
    """

    def __init__(self, config: dict[str, Any], n_sensors: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))

        self.n_sensors = int(n_sensors)
        model_cfg = config.get("model", {})
        
        # Resolve per-target history lengths
        self.k_T, self.k_V = resolve_history_lengths(model_cfg)
        
        # For backward compatibility and warm-up reference
        self.history_length = max(self.k_T, self.k_V)
        
        self.rnn_type = str(model_cfg.get("rnn_type", "gru")).lower()
        self.n_layers = int(model_cfg.get("n_layers", 2))
        self.hidden_size = int(model_cfg.get("hidden_size", 128))

        # Input: 11 features + grouped history [T-block | V-block]
        input_size = 11 + self.k_T + self.k_V
        
        if self.rnn_type == "gru":
            self.rnn: nn.Module = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.head = nn.Linear(self.hidden_size, 2)

    def forward(self, features_seq: torch.Tensor, history_seq: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass returning (batch, seq, 2)."""

        if features_seq.ndim != 3:
            raise ValueError("features_seq must have shape (batch, seq, 11)")
        if history_seq.ndim != 3:
            raise ValueError(f"history_seq must have shape (batch, seq, {self.k_T + self.k_V})")

        x = torch.cat([features_seq, history_seq], dim=-1)
        rnn_out, _state = self.rnn(x)
        return self.head(rnn_out)

    def rollout(self, features_seq: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout with known input features and predicted target history.

        The model always receives known per-step inputs (features_seq). Only the target
        history is autoregressive. Uses grouped history layout [T-block | V-block].

        Parameters
        ----------
        features_seq : torch.Tensor
            Shape (seq_len, 11) — known input features
        y0 : torch.Tensor
            Shape (2,) — initial condition [T_0, V_0]

        Returns
        -------
        torch.Tensor
            Shape (seq_len, 2) — predicted [T, bc_V] at each time step
        """

        if features_seq.ndim != 2 or features_seq.shape[-1] != 11:
            raise ValueError("features_seq must have shape (seq, 11)")

        if y0.ndim != 1 or y0.shape[0] != 2:
            raise ValueError("y0 must have shape (2,)")

        seq_len = int(features_seq.shape[0])
        k_max = max(self.k_T, self.k_V)

        preds: list[torch.Tensor] = []
        hist_T: list[torch.Tensor] = [y0[0:1]] * self.k_T  # k_T copies of y0[0]
        hist_V: list[torch.Tensor] = [y0[1:2]] * self.k_V  # k_V copies of y0[1]

        for t in range(seq_len):
            # Build grouped history: [T-block | V-block]
            h_T = torch.cat(hist_T[-self.k_T:], dim=0)  # shape (k_T,)
            h_V = torch.cat(hist_V[-self.k_V:], dim=0)  # shape (k_V,)
            h_t = torch.cat([h_T, h_V], dim=0).unsqueeze(0).unsqueeze(0)  # shape (1, 1, k_T + k_V)

            f_t = features_seq[t].unsqueeze(0).unsqueeze(0)  # shape (1, 1, 11)
            y_t = self.forward(f_t, h_t).squeeze(0).squeeze(0)  # shape (2,)

            preds.append(y_t)
            
            # Update histories
            hist_T.append(y_t[0:1])
            hist_V.append(y_t[1:2])

        return torch.stack(preds, dim=0)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
