"""Anisotropic transient heat equation for battery PINN.

PDE: ρ·Cp·∂T/∂t = ∇·(λ·∇T) + q̇

For anisotropic symmetric λ tensor:
  ∇·(λ·∇T) = Σ_{i,j} ∂/∂x_i (λ_ij · ∂T/∂x_j)
           = λ_xx·T_xx + λ_yy·T_yy + λ_zz·T_zz
             + 2·λ_xy·T_xy + 2·λ_xz·T_xz + 2·λ_yz·T_yz
             + (∂λ_xx/∂x)·T_x + (∂λ_xy/∂x)·T_y + (∂λ_xz/∂x)·T_z
             + (∂λ_xy/∂y)·T_x + (∂λ_yy/∂y)·T_y + (∂λ_yz/∂y)·T_z
             + (∂λ_xz/∂z)·T_x + (∂λ_yz/∂z)·T_y + (∂λ_zz/∂z)·T_z

For grid-based approach with constant coefficients per point:
  Spatial derivatives of λ are approximated as zero (coefficients are piecewise constant).
  
Residual: heat = ρ·Cp·T_t - (λ_xx·T_xx + λ_yy·T_yy + λ_zz·T_zz
                             + 2·λ_xy·T_xy + 2·λ_xz·T_xz + 2·λ_yz·T_yz) - q̇
"""

import torch
from typing import Dict, Optional


class AnisotropicHeatTransient:
    """Anisotropic transient heat equation PDE residual.
    
    Computes: heat = ρ·Cp·∂T/∂t - ∇·(λ·∇T) - q̇
    
    For the symmetric tensor λ with components λ_xx, λ_xy, λ_xz, λ_yy, λ_yz, λ_zz.
    """
    
    def __init__(
        self,
        rho: torch.Tensor,
        Cp: torch.Tensor,
        lambda_xx: torch.Tensor,
        lambda_yy: torch.Tensor,
        lambda_zz: torch.Tensor,
        lambda_xy: Optional[torch.Tensor] = None,
        lambda_xz: Optional[torch.Tensor] = None,
        lambda_yz: Optional[torch.Tensor] = None,
    ):
        """Initialize with per-point material properties.
        
        Args:
            rho: Density (N,) [kg/m^3]
            Cp: Specific heat (N,) [J/(kg·K)]
            lambda_xx, lambda_yy, lambda_zz: Diagonal conductivity components (N,) [W/(m·K)]
            lambda_xy, lambda_xz, lambda_yz: Off-diagonal components (N,) [W/(m·K)], default 0
        """
        self.rho = rho
        self.Cp = Cp
        self.lambda_xx = lambda_xx
        self.lambda_yy = lambda_yy
        self.lambda_zz = lambda_zz
        self.lambda_xy = lambda_xy if lambda_xy is not None else torch.zeros_like(rho)
        self.lambda_xz = lambda_xz if lambda_xz is not None else torch.zeros_like(rho)
        self.lambda_yz = lambda_yz if lambda_yz is not None else torch.zeros_like(rho)
    
    def residual(
        self,
        T: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        q_dot: torch.Tensor,
        point_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the PDE residual.
        
        Args:
            T: Temperature output from network (N,) or (N, 1)
            x, y, z, t: Spatial and temporal coordinates, requires_grad=True
            q_dot: Volumetric heat source (N,) [W/m^3]
            point_idx: Optional indices into material property arrays (N,)
            
        Returns:
            heat: PDE residual (N,) — should be 0 when PDE is satisfied
        """
        if T.dim() == 2:
            T = T.squeeze(-1)
        
        # Get material properties for these points
        if point_idx is not None:
            rho = self.rho[point_idx]
            Cp = self.Cp[point_idx]
            lxx = self.lambda_xx[point_idx]
            lyy = self.lambda_yy[point_idx]
            lzz = self.lambda_zz[point_idx]
            lxy = self.lambda_xy[point_idx]
            lxz = self.lambda_xz[point_idx]
            lyz = self.lambda_yz[point_idx]
        else:
            rho = self.rho
            Cp = self.Cp
            lxx = self.lambda_xx
            lyy = self.lambda_yy
            lzz = self.lambda_zz
            lxy = self.lambda_xy
            lxz = self.lambda_xz
            lyz = self.lambda_yz
        
        # Compute gradients
        grad_outputs = torch.ones_like(T)
        
        # First derivatives
        T_t = torch.autograd.grad(T, t, grad_outputs=grad_outputs, create_graph=True)[0]
        T_x = torch.autograd.grad(T, x, grad_outputs=grad_outputs, create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=grad_outputs, create_graph=True)[0]
        T_z = torch.autograd.grad(T, z, grad_outputs=grad_outputs, create_graph=True)[0]
        
        # Squeeze if needed
        if T_t.dim() == 2:
            T_t = T_t.squeeze(-1)
            T_x = T_x.squeeze(-1)
            T_y = T_y.squeeze(-1)
            T_z = T_z.squeeze(-1)
        
        # Second derivatives (diagonal)
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z), create_graph=True)[0]
        
        if T_xx.dim() == 2:
            T_xx = T_xx.squeeze(-1)
            T_yy = T_yy.squeeze(-1)
            T_zz = T_zz.squeeze(-1)
        
        # Mixed second derivatives (for off-diagonal λ terms)
        T_xy = torch.autograd.grad(T_x, y, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_xz = torch.autograd.grad(T_x, z, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yz = torch.autograd.grad(T_y, z, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        
        if T_xy.dim() == 2:
            T_xy = T_xy.squeeze(-1)
            T_xz = T_xz.squeeze(-1)
            T_yz = T_yz.squeeze(-1)
        
        # Diffusion term: ∇·(λ·∇T) with constant λ per point
        # = λ_xx·T_xx + λ_yy·T_yy + λ_zz·T_zz + 2·(λ_xy·T_xy + λ_xz·T_xz + λ_yz·T_yz)
        diffusion = (
            lxx * T_xx + lyy * T_yy + lzz * T_zz
            + 2.0 * (lxy * T_xy + lxz * T_xz + lyz * T_yz)
        )
        
        # PDE residual: ρ·Cp·T_t - diffusion - q_dot = 0
        heat = rho * Cp * T_t - diffusion - q_dot
        
        return heat


def compute_pde_loss(
    model: torch.nn.Module,
    pde: AnisotropicHeatTransient,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    q_dot: torch.Tensor,
    T_history: Optional[torch.Tensor] = None,
    config_tensor: Optional[torch.Tensor] = None,
    point_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute physics loss (PDE residual MSE).
    
    Args:
        model: Neural network that predicts T given inputs
        pde: AnisotropicHeatTransient instance
        x, y, z, t: Coordinates with requires_grad=True
        q_dot: Heat source (N,)
        T_history: Past predicted temperatures (N, k) for autoregressive
        config_tensor: Config scalars (N, 7)
        point_idx: Indices into material arrays (N,)
        
    Returns:
        L_phys: Mean squared PDE residual
    """
    # Build model input
    inputs = torch.cat([x, y, z, t], dim=-1)
    if config_tensor is not None:
        inputs = torch.cat([inputs, config_tensor], dim=-1)
    if T_history is not None:
        inputs = torch.cat([inputs, T_history], dim=-1)
    
    # Forward pass
    T_pred = model(inputs)
    
    # Compute PDE residual
    residual = pde.residual(T_pred, x, y, z, t, q_dot, point_idx)
    
    # MSE loss
    L_phys = torch.mean(residual ** 2)
    
    return L_phys


if __name__ == "__main__":
    # Test PDE class
    import numpy as np
    
    N = 100
    device = torch.device("cpu")
    
    # Create dummy material properties
    rho = torch.full((N,), 2500.0, device=device)
    Cp = torch.full((N,), 900.0, device=device)
    lxx = torch.full((N,), 20.0, device=device)
    lyy = torch.full((N,), 20.0, device=device)
    lzz = torch.full((N,), 20.0, device=device)
    
    pde = AnisotropicHeatTransient(rho, Cp, lxx, lyy, lzz)
    
    # Test coordinates
    x = torch.rand(N, 1, requires_grad=True, device=device)
    y = torch.rand(N, 1, requires_grad=True, device=device)
    z = torch.rand(N, 1, requires_grad=True, device=device)
    t = torch.rand(N, 1, requires_grad=True, device=device)
    q_dot = torch.zeros(N, device=device)
    
    # Test with simple function T = x + y + z + t (so T_xx = T_yy = T_zz = 0, T_t = 1)
    T = x + y + z + t
    T = T.squeeze(-1)
    
    residual = pde.residual(T, x, y, z, t, q_dot)
    
    # Expected: ρ·Cp·1 - 0 - 0 = ρ·Cp = 2500 * 900 = 2.25e6
    expected = rho * Cp
    
    print(f"PDE residual test:")
    print(f"  Residual mean: {residual.mean().item():.2e}")
    print(f"  Expected: {expected.mean().item():.2e}")
    print(f"  Match: {torch.allclose(residual, expected, rtol=1e-3)}")
