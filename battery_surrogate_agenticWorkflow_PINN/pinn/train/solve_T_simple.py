#!/usr/bin/env python3
"""Simplified training with proper normalization and physics loss.

Non-dimensionalized PDE:
  ∂T̃/∂t̃ = Fo·∇²T̃ + Q̃
  
Where:
  T̃ = (T - T_init) / ΔT      (normalized temperature)
  t̃ = t / t_max              (normalized time)
  x̃ = (x - x_min) / L        (normalized space)
  Fo = λ·t_max / (ρ·Cp·L²)   (Fourier number)
  Q̃ = q̇·t_max / (ρ·Cp·ΔT)   (dimensionless heat source)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

from ..models.net_T import NetT


@dataclass  
class LossComponents:
    """Container for individual loss components."""
    data: float = 0.0
    phys: float = 0.0
    ic: float = 0.0
    bc_in: float = 0.0
    bc_out: float = 0.0
    total: float = 0.0


class NormalizedTrainer:
    """Trainer with proper normalization and non-dimensional physics."""
    
    def __init__(
        self,
        t: np.ndarray,
        xyz: np.ndarray,
        T_labels: np.ndarray,
        config: Dict[str, float],
        q_dot: np.ndarray = None,
        rho: np.ndarray = None,
        Cp: np.ndarray = None,
        lambda_eff: np.ndarray = None,
        depth: int = 4,
        width: int = 128,
        k: int = 2,
        lr: float = 1e-3,
        w_data: float = 1.0,
        w_phys: float = 0.1,
        w_ic: float = 1.0,
        device: torch.device = None,
    ):
        """Initialize with normalization and optional physics."""
        self.device = device or torch.device("cpu")
        self.k = k
        self.w_data = w_data
        self.w_phys = w_phys
        self.w_ic = w_ic
        
        # Convert to tensors
        self.t = torch.tensor(t, dtype=torch.float32, device=self.device)
        self.xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        self.T_labels = torch.tensor(T_labels, dtype=torch.float32, device=self.device)
        
        # Normalization scales
        self.xyz_min = self.xyz.min(dim=0).values
        self.xyz_max = self.xyz.max(dim=0).values
        self.L = (self.xyz_max - self.xyz_min).max().item()  # Characteristic length
        self.t_max = self.t.max().item()
        self.T_init = config["solid_initial_temp"]
        self.T_scale = max(10.0, (T_labels.max() - T_labels.min()))  # Temperature rise scale
        
        # Normalized coordinates
        self.xyz_norm = (self.xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-6)
        self.t_norm = self.t / self.t_max
        
        # Physics parameters (optional)
        self.has_physics = (q_dot is not None and rho is not None and Cp is not None and lambda_eff is not None)
        
        if self.has_physics:
            # Store raw physics data
            self.q_dot = torch.tensor(q_dot, dtype=torch.float32, device=self.device)
            self.rho = torch.tensor(rho, dtype=torch.float32, device=self.device)
            self.Cp = torch.tensor(Cp, dtype=torch.float32, device=self.device)
            self.lambda_eff = torch.tensor(lambda_eff, dtype=torch.float32, device=self.device)
            
            # Compute characteristic non-dimensional numbers
            # Fo = λ·t_max / (ρ·Cp·L²) - per grid point
            self.Fo = (self.lambda_eff * self.t_max) / (self.rho * self.Cp * self.L**2)
            
            # Q̃ scale = t_max / (ρ·Cp·ΔT) - per grid point  
            self.Q_scale = self.t_max / (self.rho * self.Cp * self.T_scale)
        
        # Config tensor (normalized)
        self.config_tensor = torch.tensor([
            config["c_rate"] / 3,
            config["cell_current"] / 1000,
            (config["fluid_initial_temp"] - 25) / 10,
            (config["fluid_inlet_temp"] - 25) / 10,
            config["fluid_mass_flow"] * 1000,
            config["soc_start"] / 100,
            (config["solid_initial_temp"] - 25) / 10,
        ], dtype=torch.float32, device=self.device)
        
        # Grid info
        self.n_points = xyz.shape[0]
        self.n_t = len(t)
        
        # Create model (NO hard IC wrapper for now - let's keep it simple)
        self.model = NetT(
            depth=depth,
            width=width,
            k=k,
            n_config=7,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # History
        self.loss_history: List[LossComponents] = []
    
    def compute_physics_residual(
        self,
        T_norm: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        ti: int,
    ) -> torch.Tensor:
        """Compute non-dimensional PDE residual.
        
        Non-dim PDE: ∂T̃/∂t̃ - Fo·∇²T̃ - Q̃ = 0
        
        Args:
            T_norm: Normalized temperature output (n_points,)
            x, y, z, t: Normalized coordinates with requires_grad=True
            ti: Time index for q_dot
            
        Returns:
            residual: (n_points,) - should be ~0 if PDE satisfied
        """
        if not self.has_physics:
            return torch.zeros(T_norm.shape[0], device=self.device)
        
        # Compute gradients
        grad_outputs = torch.ones_like(T_norm)
        
        # Time derivative: ∂T̃/∂t̃
        T_t = torch.autograd.grad(T_norm, t, grad_outputs=grad_outputs, create_graph=True)[0]
        if T_t.dim() == 2:
            T_t = T_t.squeeze(-1)
        
        # Spatial derivatives (for Laplacian)
        T_x = torch.autograd.grad(T_norm, x, grad_outputs=grad_outputs, create_graph=True)[0]
        T_y = torch.autograd.grad(T_norm, y, grad_outputs=grad_outputs, create_graph=True)[0]
        T_z = torch.autograd.grad(T_norm, z, grad_outputs=grad_outputs, create_graph=True)[0]
        
        if T_x.dim() == 2:
            T_x = T_x.squeeze(-1)
            T_y = T_y.squeeze(-1)
            T_z = T_z.squeeze(-1)
        
        # Second derivatives
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z), create_graph=True)[0]
        
        if T_xx.dim() == 2:
            T_xx = T_xx.squeeze(-1)
            T_yy = T_yy.squeeze(-1)
            T_zz = T_zz.squeeze(-1)
        
        # Laplacian (isotropic approximation)
        laplacian = T_xx + T_yy + T_zz
        
        # Dimensionless heat source: Q̃ = q_dot · Q_scale
        q_dot_t = self.q_dot[ti]
        Q_tilde = q_dot_t * self.Q_scale  # (n_points,)
        
        # Non-dimensional residual: ∂T̃/∂t̃ - Fo·∇²T̃ - Q̃
        residual = T_t - self.Fo * laplacian - Q_tilde
        
        return residual
        
    def train_epoch(self, batch_size: int = 32) -> LossComponents:
        """Train one epoch over sampled timesteps with physics loss."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample timesteps (skip first k for history)
        valid_indices = list(range(self.k, self.n_t))
        n_sample = min(batch_size, len(valid_indices))
        t_indices = np.random.choice(valid_indices, n_sample, replace=False)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_data = 0.0
        total_phys = 0.0
        total_ic = 0.0
        
        for ti in t_indices:
            # Build normalized inputs with gradients for physics
            x = self.xyz_norm[:, 0:1].clone().requires_grad_(True)
            y = self.xyz_norm[:, 1:2].clone().requires_grad_(True)
            z = self.xyz_norm[:, 2:3].clone().requires_grad_(True)
            t_val = self.t_norm[ti].expand(self.n_points, 1).clone().requires_grad_(True)
            cfg = self.config_tensor.unsqueeze(0).expand(self.n_points, -1)
            
            # History from labels (normalized)
            hist = (self.T_labels[ti-self.k:ti].T - self.T_init) / self.T_scale
            
            inputs = torch.cat([x, y, z, t_val, cfg, hist], dim=-1)
            
            # Forward
            T_pred = self.model(inputs).squeeze(-1)
            
            # Target (normalized)
            T_target = (self.T_labels[ti] - self.T_init) / self.T_scale
            
            # L_data: MSE on normalized temperature
            L_data = torch.mean((T_pred - T_target) ** 2)
            total_data += L_data.item()
            
            # L_phys: Non-dimensional PDE residual
            if self.has_physics:
                residual = self.compute_physics_residual(T_pred, x, y, z, t_val, ti)
                L_phys = torch.mean(residual ** 2)
                total_phys += L_phys.item()
            else:
                L_phys = torch.tensor(0.0, device=self.device)
            
            # L_IC: Initial condition (only at t=0)
            if ti == self.k:  # First valid timestep
                L_ic = torch.mean(T_pred ** 2)  # Should be 0 at t=0
                total_ic += L_ic.item()
            else:
                L_ic = torch.tensor(0.0, device=self.device)
            
            # Weighted loss
            step_loss = self.w_data * L_data + self.w_phys * L_phys + self.w_ic * L_ic
            total_loss = total_loss + step_loss
        
        # Average and backward
        total_loss = total_loss / n_sample
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record losses
        losses = LossComponents(
            data=total_data / n_sample,
            phys=total_phys / n_sample,
            ic=total_ic / max(1, sum(1 for ti in t_indices if ti == self.k)),
            bc_in=0.0,
            bc_out=0.0,
            total=total_loss.item(),
        )
        self.loss_history.append(losses)
        
        return losses
    
    def get_final_losses(self) -> LossComponents:
        """Evaluate on all timesteps and return final losses."""
        self.model.eval()
        
        total_data = 0.0
        n_eval = 0
        
        with torch.no_grad():
            for ti in range(self.k, self.n_t):  # Skip first k timesteps
                x = self.xyz_norm[:, 0:1]
                y = self.xyz_norm[:, 1:2]
                z = self.xyz_norm[:, 2:3]
                t_val = self.t_norm[ti].expand(self.n_points, 1)
                cfg = self.config_tensor.unsqueeze(0).expand(self.n_points, -1)
                hist = (self.T_labels[ti-self.k:ti].T - self.T_init) / self.T_scale
                
                inputs = torch.cat([x, y, z, t_val, cfg, hist], dim=-1)
                T_pred = self.model(inputs).squeeze(-1)
                T_target = (self.T_labels[ti] - self.T_init) / self.T_scale
                
                L_data = torch.mean((T_pred - T_target) ** 2).item()
                total_data += L_data
                n_eval += 1
        
        return LossComponents(
            data=total_data / n_eval,
            phys=0.0,
            ic=0.0,
            bc_in=0.0,
            bc_out=0.0,
            total=total_data / n_eval,
        )


def train_simplified(
    t: np.ndarray,
    xyz: np.ndarray,
    T_labels: np.ndarray,
    config: Dict[str, float],
    depth: int = 4,
    width: int = 128,
    k: int = 2,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    log_interval: int = 10,
    print_fn=print,
    # Physics parameters (optional - pass all or none)
    q_dot: np.ndarray = None,
    rho: np.ndarray = None,
    Cp: np.ndarray = None,
    lambda_eff: np.ndarray = None,
    w_phys: float = 0.1,
) -> Tuple[nn.Module, List[LossComponents], LossComponents]:
    """Simplified training function with optional physics loss.
    
    Args:
        t, xyz, T_labels: Training data
        config: Operating point config
        q_dot: Heat source (n_t, n_points)
        rho, Cp, lambda_eff: Material properties (n_points,)
        w_phys: Physics loss weight
        
    Returns:
        (model, loss_history, final_losses)
    """
    has_physics = q_dot is not None
    mode = "Physics+Data" if has_physics else "Data-only"
    print_fn(f"Training Temperature PINN: depth={depth}, width={width}, k={k}, mode={mode}")
    print_fn(f"  Grid points: {xyz.shape[0]}, Timesteps: {len(t)}")
    print_fn(f"  Epochs: {epochs}, Batch size: {batch_size}, w_phys={w_phys}")
    print_fn("-" * 80)
    
    trainer = NormalizedTrainer(
        t=t,
        xyz=xyz,
        T_labels=T_labels,
        config=config,
        q_dot=q_dot,
        rho=rho,
        Cp=Cp,
        lambda_eff=lambda_eff,
        depth=depth,
        width=width,
        k=k,
        lr=lr,
        w_phys=w_phys,
    )
    
    # Training loop
    for epoch in range(1, epochs + 1):
        losses = trainer.train_epoch(batch_size=batch_size)
        
        if epoch == 1 or epoch % log_interval == 0:
            print_fn(
                f"Epoch {epoch:4d} | L_data={losses.data:.4e} L_phys={losses.phys:.4e} "
                f"L_IC={losses.ic:.4e} L_BCin={losses.bc_in:.4e} L_BCout={losses.bc_out:.4e} | "
                f"Total={losses.total:.4e}"
            )
    
    # Final evaluation
    final = trainer.get_final_losses()
    
    return trainer.model, trainer.loss_history, final
