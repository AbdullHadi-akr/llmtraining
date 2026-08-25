"""Voltage (bc_V) training - data-only fit.

Simple supervised learning on bc_V(t) with no physics constraints.
Non-recurrent for now (TODO: add recurrence later).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..models.net_V import create_net_V


@dataclass
class VLossComponents:
    """Loss components for voltage fit."""
    data: float = 0.0


class VoltageFitter:
    """Simple data-only fitter for bc_V."""
    
    def __init__(
        self,
        t: np.ndarray,
        bc_V: np.ndarray,
        config: Dict[str, float],
        depth: int = 2,
        width: int = 64,
        lr: float = 1e-3,
        device: torch.device = None,
    ):
        """Initialize the voltage fitter.
        
        Args:
            t: Time array (N_t,)
            bc_V: Voltage labels (N_t,)
            config: Config scalar dict
            depth: Network depth
            width: Network width
            lr: Learning rate
            device: Torch device
        """
        self.device = device or torch.device("cpu")
        
        # Store data
        self.t = torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(-1)  # (N_t, 1)
        self.bc_V = torch.tensor(bc_V, dtype=torch.float32, device=self.device)
        
        # Config tensor
        config_values = torch.tensor([
            config["c_rate"],
            config["cell_current"],
            config["fluid_initial_temp"],
            config["fluid_inlet_temp"],
            config["fluid_mass_flow"],
            config["soc_start"],
            config["solid_initial_temp"],
        ], dtype=torch.float32, device=self.device)
        self.config_tensor = config_values.unsqueeze(0).expand(len(t), -1)  # (N_t, 7)
        
        # Build inputs: (t, config)
        self.inputs = torch.cat([self.t, self.config_tensor], dim=-1)  # (N_t, 8)
        
        # Create model
        self.model = create_net_V(
            depth=depth,
            width=width,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss history
        self.loss_history: List[VLossComponents] = []
    
    def train_step(self) -> VLossComponents:
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        V_pred = self.model(self.inputs).squeeze(-1)  # (N_t,)
        
        # Data loss
        L_data = torch.mean((V_pred - self.bc_V) ** 2)
        
        # Backward
        L_data.backward()
        self.optimizer.step()
        
        losses = VLossComponents(data=L_data.item())
        self.loss_history.append(losses)
        
        return losses
    
    def save_loss_history(self, path: str):
        """Save loss history to CSV."""
        df = pd.DataFrame([
            {"epoch": i, "L_bcV": l.data}
            for i, l in enumerate(self.loss_history)
        ])
        df.to_csv(path, index=False)


def train_voltage_network(
    t: np.ndarray,
    bc_V: np.ndarray,
    config: Dict[str, float],
    epochs: int = 500,
    depth: int = 2,
    width: int = 64,
    lr: float = 1e-3,
    log_interval: int = 50,
    output_dir: str = None,
) -> Tuple[nn.Module, List[VLossComponents]]:
    """Train voltage network (data-only).
    
    Args:
        t: Time array (N_t,)
        bc_V: Voltage labels (N_t,)
        config: Config scalar dict
        epochs: Number of epochs
        depth: Network depth
        width: Network width
        lr: Learning rate
        log_interval: Print interval
        output_dir: Directory to save loss history
        
    Returns:
        Trained model and loss history
    """
    device = torch.device("cpu")
    
    fitter = VoltageFitter(
        t=t,
        bc_V=bc_V,
        config=config,
        depth=depth,
        width=width,
        lr=lr,
        device=device,
    )
    
    print(f"Training Voltage Network: depth={depth}, width={width}")
    print(f"  Timesteps: {len(t)}, Epochs: {epochs}")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        losses = fitter.train_step()
        
        if epoch == 1 or epoch % log_interval == 0:
            print(f"Epoch {epoch:4d} | L_bcV={losses.data:.4e}")
    
    # Save loss history
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fitter.save_loss_history(str(output_path / "loss_history_V.csv"))
        print(f"\nLoss history saved to {output_path / 'loss_history_V.csv'}")
    
    return fitter.model, fitter.loss_history


if __name__ == "__main__":
    # Test with dummy data
    print("Testing VoltageFitter with dummy data...")
    
    n_t = 100
    t = np.linspace(0.1, 100.0, n_t).astype(np.float32)
    bc_V = np.sin(t / 10) * 0.5 + 4.0  # Fake voltage data
    config = {
        "c_rate": 2.0,
        "cell_current": 316.0,
        "fluid_initial_temp": 25.0,
        "fluid_inlet_temp": 25.0,
        "fluid_mass_flow": 0.0013,
        "soc_start": 10.0,
        "solid_initial_temp": 25.0,
    }
    
    model, history = train_voltage_network(
        t=t,
        bc_V=bc_V,
        config=config,
        epochs=100,
        log_interval=20,
    )
    
    print(f"\nFinal L_bcV: {history[-1].data:.4e}")
