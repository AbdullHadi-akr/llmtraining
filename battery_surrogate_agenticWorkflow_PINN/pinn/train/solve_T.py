"""Temperature PINN solver with grid-based constraints.

Implements:
- L_data: Data loss (predicted T vs labeled T on grid points)
- L_phys: Physics loss (PDE residual on grid points)
- L_IC: Initial condition loss (T(t=0) = T_init)
- L_BCin: Inlet boundary condition (T at inlet faces)
- L_BCout: Outlet boundary condition (T at outlet faces)

Uses autoregressive predicted feedback with truncated BPTT.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..physics.anisotropic_heat import AnisotropicHeatTransient
from ..models.net_T import create_net_T, AutoregressiveRollout


@dataclass
class LossComponents:
    """Container for individual loss components."""
    data: float = 0.0
    phys: float = 0.0
    ic: float = 0.0
    bc_in: float = 0.0
    bc_out: float = 0.0
    total: float = 0.0


class TemperaturePINNTrainer:
    """Trainer for temperature PINN with per-branch loss tracking."""
    
    def __init__(
        self,
        # Data
        t: np.ndarray,
        xyz: np.ndarray,
        T_labels: np.ndarray,
        layer: np.ndarray,
        config: Dict[str, float],
        q_dot: np.ndarray,
        # Material properties
        rho: np.ndarray,
        Cp: np.ndarray,
        lambda_tensor: np.ndarray,
        # Boundary faces
        inlet_xyz: np.ndarray,
        outlet_xyz: np.ndarray,
        T_inlet: float,
        T_outlet: float,
        # Model config
        depth: int = 4,
        width: int = 128,
        k: int = 2,
        hard_ic: bool = True,
        # Loss weights
        w_data: float = 1.0,
        w_phys: float = 0.1,
        w_ic: float = 1.0,
        w_bc_in: float = 1.0,
        w_bc_out: float = 1.0,
        # Training
        lr: float = 1e-3,
        bptt_window: int = 8,
        device: torch.device = None,
    ):
        """Initialize the trainer.
        
        Args:
            t: Time array (N_t,)
            xyz: Grid coordinates (363, 3)
            T_labels: Temperature labels (N_t, 363)
            layer: Layer labels (363,)
            config: Config scalar dict
            q_dot: Heat source (N_t,)
            rho: Density (363,)
            Cp: Specific heat (363,)
            lambda_tensor: Conductivity (363, 3, 3)
            inlet_xyz: Inlet face coords (N_in, 3)
            outlet_xyz: Outlet face coords (N_out, 3)
            T_inlet: Inlet temperature (scalar)
            T_outlet: Outlet temperature (scalar)
            depth: Network depth
            width: Network width
            k: History length
            hard_ic: Use hard IC enforcement
            w_data, w_phys, w_ic, w_bc_in, w_bc_out: Loss weights
            lr: Learning rate
            bptt_window: Truncated BPTT window
            device: Torch device
        """
        self.device = device or torch.device("cpu")
        
        # Store data
        self.t = torch.tensor(t, dtype=torch.float32, device=self.device)
        self.xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        self.T_labels = torch.tensor(T_labels, dtype=torch.float32, device=self.device)
        self.q_dot = torch.tensor(q_dot, dtype=torch.float32, device=self.device)
        
        # Config tensor (repeated for all points)
        config_values = torch.tensor([
            config["c_rate"],
            config["cell_current"],
            config["fluid_initial_temp"],
            config["fluid_inlet_temp"],
            config["fluid_mass_flow"],
            config["soc_start"],
            config["solid_initial_temp"],
        ], dtype=torch.float32, device=self.device)
        self.config_tensor = config_values.unsqueeze(0)  # (1, 7)
        
        # Boundary data
        self.inlet_xyz = torch.tensor(inlet_xyz, dtype=torch.float32, device=self.device)
        self.outlet_xyz = torch.tensor(outlet_xyz, dtype=torch.float32, device=self.device)
        self.T_inlet = T_inlet
        self.T_outlet = T_outlet
        
        # Material properties
        self.rho = torch.tensor(rho, dtype=torch.float32, device=self.device)
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=self.device)
        
        # Extract lambda components
        self.lambda_xx = torch.tensor(lambda_tensor[:, 0, 0], dtype=torch.float32, device=self.device)
        self.lambda_yy = torch.tensor(lambda_tensor[:, 1, 1], dtype=torch.float32, device=self.device)
        self.lambda_zz = torch.tensor(lambda_tensor[:, 2, 2], dtype=torch.float32, device=self.device)
        self.lambda_xy = torch.tensor(lambda_tensor[:, 0, 1], dtype=torch.float32, device=self.device)
        self.lambda_xz = torch.tensor(lambda_tensor[:, 0, 2], dtype=torch.float32, device=self.device)
        self.lambda_yz = torch.tensor(lambda_tensor[:, 1, 2], dtype=torch.float32, device=self.device)
        
        # Create PDE
        self.pde = AnisotropicHeatTransient(
            rho=self.rho,
            Cp=self.Cp,
            lambda_xx=self.lambda_xx,
            lambda_yy=self.lambda_yy,
            lambda_zz=self.lambda_zz,
            lambda_xy=self.lambda_xy,
            lambda_xz=self.lambda_xz,
            lambda_yz=self.lambda_yz,
        )
        
        # Create model
        T_init = config["solid_initial_temp"]
        self.k = k
        self.hard_ic = hard_ic
        self.model = create_net_T(
            depth=depth,
            width=width,
            k=k,
            hard_ic=hard_ic,
            T_init=T_init,
        ).to(self.device)
        
        # Loss weights
        self.w_data = w_data
        self.w_phys = w_phys
        self.w_ic = w_ic
        self.w_bc_in = w_bc_in
        self.w_bc_out = w_bc_out
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Rollout helper
        self.n_points = xyz.shape[0]
        self.T_init = T_init
        self.bptt_window = bptt_window
        
        # Loss history
        self.loss_history: List[LossComponents] = []
    
    def _build_inputs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t_val: torch.Tensor,
        T_history: torch.Tensor,
    ) -> torch.Tensor:
        """Build input tensor for the model.
        
        Args:
            x: X coordinates (N, 1) with requires_grad=True for physics
            y: Y coordinates (N, 1) with requires_grad=True for physics
            z: Z coordinates (N, 1) with requires_grad=True for physics
            t_val: Time scalar or (N,) or (N, 1)
            T_history: Past predictions (N, k)
            
        Returns:
            inputs: (N, 4 + 7 + k)
        """
        N = x.shape[0]
        
        # Expand time if scalar
        if t_val.dim() == 0:
            t_val = t_val.expand(N, 1)
        elif t_val.dim() == 1 and t_val.shape[0] == 1:
            t_val = t_val.expand(N, 1)
        elif t_val.dim() == 1:
            t_val = t_val.unsqueeze(-1)
        
        # Expand config
        config = self.config_tensor.expand(N, -1)  # (N, 7)
        
        # Concatenate: x, y, z, t, config, T_history
        inputs = torch.cat([
            x,          # (N, 1)
            y,          # (N, 1)
            z,          # (N, 1)
            t_val,      # (N, 1)
            config,     # (N, 7)
            T_history,  # (N, k)
        ], dim=-1)
        
        return inputs
    
    def compute_losses(
        self,
        t_idx: int,
        T_history: torch.Tensor,
    ) -> LossComponents:
        """Compute all loss components for a single timestep.
        
        Args:
            t_idx: Time index
            T_history: Past predictions (n_points, k)
            
        Returns:
            LossComponents with individual losses
        """
        losses = LossComponents()
        
        t_val = self.t[t_idx]
        T_label = self.T_labels[t_idx]  # (n_points,)
        q_dot_t = self.q_dot[t_idx]
        
        # Build inputs for grid points (x, y, z require grad for physics)
        x = self.xyz[:, 0:1].clone().requires_grad_(True)
        y = self.xyz[:, 1:2].clone().requires_grad_(True)
        z = self.xyz[:, 2:3].clone().requires_grad_(True)
        t_tensor = t_val.expand(self.n_points, 1).clone().requires_grad_(True)
        
        inputs = self._build_inputs(x, y, z, t_tensor, T_history)
        
        # Forward pass
        T_pred = self.model(inputs).squeeze(-1)  # (n_points,)
        
        # L_data: MSE vs labels
        losses.data = torch.mean((T_pred - T_label) ** 2).item()
        
        # L_phys: PDE residual (need gradients)
        q_dot_expanded = torch.full((self.n_points,), q_dot_t, device=self.device)
        
        # Re-do forward with gradient tracking for PDE
        T_for_pde = self.model(inputs).squeeze(-1)
        residual = self.pde.residual(
            T_for_pde, x, y, z, t_tensor, q_dot_expanded
        )
        losses.phys = torch.mean(residual ** 2).item()
        
        # L_IC: Only at first timestep (t_idx == 0)
        if t_idx == 0:
            if self.hard_ic:
                # Hard IC enforced exactly, loss should be ~0
                losses.ic = torch.mean((T_pred - self.T_init) ** 2).item()
            else:
                losses.ic = torch.mean((T_pred - self.T_init) ** 2).item()
        
        # L_BCin: Inlet boundary
        inlet_x = self.inlet_xyz[:, 0:1]
        inlet_y = self.inlet_xyz[:, 1:2]
        inlet_z = self.inlet_xyz[:, 2:3]
        T_history_bc = torch.full((len(self.inlet_xyz), self.k), self.T_init, device=self.device)
        inputs_inlet = self._build_inputs(inlet_x, inlet_y, inlet_z, t_tensor[:len(self.inlet_xyz)], T_history_bc)
        T_inlet_pred = self.model(inputs_inlet).squeeze(-1)
        losses.bc_in = torch.mean((T_inlet_pred - self.T_inlet) ** 2).item()
        
        # L_BCout: Outlet boundary
        outlet_x = self.outlet_xyz[:, 0:1]
        outlet_y = self.outlet_xyz[:, 1:2]
        outlet_z = self.outlet_xyz[:, 2:3]
        T_history_bc = torch.full((len(self.outlet_xyz), self.k), self.T_init, device=self.device)
        inputs_outlet = self._build_inputs(outlet_x, outlet_y, outlet_z, t_tensor[:len(self.outlet_xyz)], T_history_bc)
        T_outlet_pred = self.model(inputs_outlet).squeeze(-1)
        losses.bc_out = torch.mean((T_outlet_pred - self.T_outlet) ** 2).item()
        
        # Total weighted loss
        losses.total = (
            self.w_data * losses.data
            + self.w_phys * losses.phys
            + self.w_ic * losses.ic
            + self.w_bc_in * losses.bc_in
            + self.w_bc_out * losses.bc_out
        )
        
        return losses, T_pred
    
    def train_step(self, epoch: int, batch_size: int = None) -> LossComponents:
        """Perform one training step with autoregressive rollout.
        
        Args:
            epoch: Current epoch number
            batch_size: Number of timesteps to sample (None = all)
            
        Returns:
            Average loss components over the batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        n_t = len(self.t)
        
        # Sample timesteps (or use all)
        if batch_size is not None and batch_size < n_t:
            t_indices = np.random.choice(n_t, batch_size, replace=False)
            t_indices = np.sort(t_indices)  # Sort for sequential rollout
        else:
            t_indices = np.arange(n_t)
        
        # Initialize rollout
        rollout = AutoregressiveRollout(
            k=self.k,
            T_init=self.T_init,
            n_points=self.n_points,
            bptt_window=self.bptt_window,
            device=self.device,
        )
        
        # Accumulate losses
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        avg_losses = LossComponents()
        n_steps = 0
        
        for t_idx in t_indices:
            T_history = rollout.get_history()
            
            # Compute losses
            t_val = self.t[t_idx]
            T_label = self.T_labels[t_idx]
            q_dot_t = self.q_dot[t_idx]
            
            # Build inputs (x, y, z require grad for physics)
            x = self.xyz[:, 0:1].clone().requires_grad_(True)
            y = self.xyz[:, 1:2].clone().requires_grad_(True)
            z = self.xyz[:, 2:3].clone().requires_grad_(True)
            t_tensor = t_val.expand(self.n_points, 1).clone().requires_grad_(True)
            
            inputs = self._build_inputs(x, y, z, t_tensor, T_history)
            
            # Forward pass
            T_pred = self.model(inputs).squeeze(-1)
            
            # L_data
            L_data = torch.mean((T_pred - T_label) ** 2)
            
            # L_phys
            q_dot_expanded = torch.full((self.n_points,), q_dot_t, device=self.device)
            residual = self.pde.residual(T_pred, x, y, z, t_tensor, q_dot_expanded)
            L_phys = torch.mean(residual ** 2)
            
            # L_IC (only first timestep)
            if t_idx == 0:
                L_ic = torch.mean((T_pred - self.T_init) ** 2)
            else:
                L_ic = torch.tensor(0.0, device=self.device)
            
            # L_BC (don't need gradients for BC)
            inlet_x = self.inlet_xyz[:, 0:1]
            inlet_y = self.inlet_xyz[:, 1:2]
            inlet_z = self.inlet_xyz[:, 2:3]
            T_history_bc = torch.full((len(self.inlet_xyz), self.k), self.T_init, device=self.device)
            t_bc = t_val.expand(len(self.inlet_xyz), 1)
            inputs_inlet = self._build_inputs(inlet_x, inlet_y, inlet_z, t_bc, T_history_bc)
            T_inlet_pred = self.model(inputs_inlet).squeeze(-1)
            L_bc_in = torch.mean((T_inlet_pred - self.T_inlet) ** 2)
            
            outlet_x = self.outlet_xyz[:, 0:1]
            outlet_y = self.outlet_xyz[:, 1:2]
            outlet_z = self.outlet_xyz[:, 2:3]
            T_history_bc = torch.full((len(self.outlet_xyz), self.k), self.T_init, device=self.device)
            t_bc = t_val.expand(len(self.outlet_xyz), 1)
            inputs_outlet = self._build_inputs(outlet_x, outlet_y, outlet_z, t_bc, T_history_bc)
            T_outlet_pred = self.model(inputs_outlet).squeeze(-1)
            L_bc_out = torch.mean((T_outlet_pred - self.T_outlet) ** 2)
            
            # Total loss
            step_loss = (
                self.w_data * L_data
                + self.w_phys * L_phys
                + self.w_ic * L_ic
                + self.w_bc_in * L_bc_in
                + self.w_bc_out * L_bc_out
            )
            
            total_loss = total_loss + step_loss
            
            # Accumulate for averaging
            avg_losses.data += L_data.item()
            avg_losses.phys += L_phys.item()
            avg_losses.ic += L_ic.item()
            avg_losses.bc_in += L_bc_in.item()
            avg_losses.bc_out += L_bc_out.item()
            n_steps += 1
            
            # Update rollout (detach for truncated BPTT)
            detach = rollout.should_detach()
            rollout.update(T_pred, detach=detach)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Average losses
        if n_steps > 0:
            avg_losses.data /= n_steps
            avg_losses.phys /= n_steps
            avg_losses.ic /= n_steps
            avg_losses.bc_in /= n_steps
            avg_losses.bc_out /= n_steps
        avg_losses.total = (
            self.w_data * avg_losses.data
            + self.w_phys * avg_losses.phys
            + self.w_ic * avg_losses.ic
            + self.w_bc_in * avg_losses.bc_in
            + self.w_bc_out * avg_losses.bc_out
        )
        
        self.loss_history.append(avg_losses)
        
        return avg_losses
    
    def save_loss_history(self, path: str):
        """Save loss history to CSV."""
        df = pd.DataFrame([
            {
                "epoch": i,
                "L_data": l.data,
                "L_phys": l.phys,
                "L_IC": l.ic,
                "L_BCin": l.bc_in,
                "L_BCout": l.bc_out,
                "total": l.total,
            }
            for i, l in enumerate(self.loss_history)
        ])
        df.to_csv(path, index=False)


def train_temperature_pinn(
    # Data
    t: np.ndarray,
    xyz: np.ndarray,
    T_labels: np.ndarray,
    layer: np.ndarray,
    config: Dict[str, float],
    q_dot: np.ndarray,
    # Material properties
    rho: np.ndarray,
    Cp: np.ndarray,
    lambda_tensor: np.ndarray,
    # Boundary
    inlet_xyz: np.ndarray,
    outlet_xyz: np.ndarray,
    T_inlet: float,
    T_outlet: float,
    # Training params
    epochs: int = 500,
    depth: int = 4,
    width: int = 128,
    k: int = 2,
    hard_ic: bool = True,
    lr: float = 1e-3,
    batch_size: int = 32,
    log_interval: int = 50,
    output_dir: str = None,
) -> Tuple[nn.Module, List[LossComponents]]:
    """Train temperature PINN and return model + loss history.
    
    This is the main training function.
    """
    device = torch.device("cpu")
    
    trainer = TemperaturePINNTrainer(
        t=t,
        xyz=xyz,
        T_labels=T_labels,
        layer=layer,
        config=config,
        q_dot=q_dot,
        rho=rho,
        Cp=Cp,
        lambda_tensor=lambda_tensor,
        inlet_xyz=inlet_xyz,
        outlet_xyz=outlet_xyz,
        T_inlet=T_inlet,
        T_outlet=T_outlet,
        depth=depth,
        width=width,
        k=k,
        hard_ic=hard_ic,
        lr=lr,
        device=device,
    )
    
    print(f"Training Temperature PINN: depth={depth}, width={width}, k={k}, hard_ic={hard_ic}")
    print(f"  Grid points: {xyz.shape[0]}, Timesteps: {len(t)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        losses = trainer.train_step(epoch, batch_size=batch_size)
        
        if epoch == 1 or epoch % log_interval == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"L_data={losses.data:.4e} L_phys={losses.phys:.4e} "
                f"L_IC={losses.ic:.4e} L_BCin={losses.bc_in:.4e} L_BCout={losses.bc_out:.4e} | "
                f"Total={losses.total:.4e}"
            )
    
    # Save loss history
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_loss_history(str(output_path / "loss_history.csv"))
        print(f"\nLoss history saved to {output_path / 'loss_history.csv'}")
    
    return trainer.model, trainer.loss_history


if __name__ == "__main__":
    # Test with dummy data
    print("Testing TemperaturePINNTrainer with dummy data...")
    
    n_t = 10
    n_points = 363
    
    t = np.linspace(0.1, 10.0, n_t).astype(np.float32)
    xyz = np.random.rand(n_points, 3).astype(np.float32)
    T_labels = np.random.rand(n_t, n_points).astype(np.float32) * 10 + 20
    layer = np.array(["cc"] * 121 + ["jr1c"] * 121 + ["g"] * 121)
    config = {
        "c_rate": 2.0,
        "cell_current": 316.0,
        "fluid_initial_temp": 25.0,
        "fluid_inlet_temp": 25.0,
        "fluid_mass_flow": 0.0013,
        "soc_start": 10.0,
        "solid_initial_temp": 25.0,
    }
    q_dot = np.random.rand(n_t).astype(np.float32) * 1000
    
    rho = np.full(n_points, 2500.0, dtype=np.float32)
    Cp = np.full(n_points, 900.0, dtype=np.float32)
    lambda_tensor = np.zeros((n_points, 3, 3), dtype=np.float32)
    lambda_tensor[:, 0, 0] = 20.0
    lambda_tensor[:, 1, 1] = 20.0
    lambda_tensor[:, 2, 2] = 20.0
    
    inlet_xyz = np.random.rand(20, 3).astype(np.float32)
    outlet_xyz = np.random.rand(20, 3).astype(np.float32)
    
    model, history = train_temperature_pinn(
        t=t,
        xyz=xyz,
        T_labels=T_labels,
        layer=layer,
        config=config,
        q_dot=q_dot,
        rho=rho,
        Cp=Cp,
        lambda_tensor=lambda_tensor,
        inlet_xyz=inlet_xyz,
        outlet_xyz=outlet_xyz,
        T_inlet=25.0,
        T_outlet=30.0,
        epochs=5,
        depth=2,
        width=32,
        log_interval=1,
    )
    
    print("\nTraining complete!")
    print(f"Final losses: L_data={history[-1].data:.4e}, L_phys={history[-1].phys:.4e}")
