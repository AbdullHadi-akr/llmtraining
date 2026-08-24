"""Hard constraint wrapper for initial conditions.

Implements hard IC enforcement: T = T_init + t · N(x, y, z, t, ...)

This ensures T(t=0) = T_init exactly, reducing error propagation
in the autoregressive rollout.
"""

import torch
import torch.nn as nn
from typing import Optional


class HardICWrapper(nn.Module):
    """Wrapper that enforces T(t=0) = T_init exactly.
    
    Output: T = T_init + t · N(inputs)
    
    At t=0: T = T_init (exact)
    At t>0: T = T_init + t·N (network learns the deviation from IC)
    """
    
    def __init__(
        self,
        base_network: nn.Module,
        T_init: float,
        t_idx: int = 3,  # Index of t in the input tensor
    ):
        """Initialize the hard IC wrapper.
        
        Args:
            base_network: The underlying neural network N(inputs)
            T_init: Initial temperature (scalar, applied to all points)
            t_idx: Index of the time coordinate in the input tensor
        """
        super().__init__()
        self.base_network = base_network
        self.register_buffer("T_init", torch.tensor(T_init, dtype=torch.float32))
        self.t_idx = t_idx
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with hard IC enforcement.
        
        Args:
            inputs: Input tensor (..., D) where D includes (x, y, z, t, ...)
            
        Returns:
            T: Temperature with hard IC (..., 1)
        """
        # Extract time
        t = inputs[..., self.t_idx:self.t_idx+1]  # (..., 1)
        
        # Get network output
        N_out = self.base_network(inputs)  # (..., 1)
        
        # Apply hard constraint: T = T_init + t * N
        T = self.T_init + t * N_out
        
        return T
    
    def set_T_init(self, T_init: float):
        """Update the initial temperature."""
        self.T_init.fill_(T_init)


class HardICBCWrapper(nn.Module):
    """Wrapper that enforces both IC and BC (more complex).
    
    For BC: Uses distance functions to boundaries.
    T = g(x,t) + h(x,t) · N(inputs)
    
    Where:
    - g satisfies the BC and IC
    - h = 0 at boundaries and t=0
    
    For our case (inlet/outlet BC):
    - This is complex because BC points are outside the grid
    - Recommend using soft BC instead (default)
    """
    
    def __init__(
        self,
        base_network: nn.Module,
        T_init: float,
        T_inlet: float,
        T_outlet: float,
        y_inlet: float = -0.1265,
        y_outlet: float = 0.14605,
        t_idx: int = 3,
        y_idx: int = 1,
    ):
        """Initialize hard IC+BC wrapper.
        
        This is experimental and may not work well for external BC points.
        """
        super().__init__()
        self.base_network = base_network
        self.register_buffer("T_init", torch.tensor(T_init, dtype=torch.float32))
        self.register_buffer("T_inlet", torch.tensor(T_inlet, dtype=torch.float32))
        self.register_buffer("T_outlet", torch.tensor(T_outlet, dtype=torch.float32))
        self.y_inlet = y_inlet
        self.y_outlet = y_outlet
        self.t_idx = t_idx
        self.y_idx = y_idx
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with hard IC+BC."""
        t = inputs[..., self.t_idx:self.t_idx+1]
        y = inputs[..., self.y_idx:self.y_idx+1]
        
        # Distance functions (normalized)
        y_range = self.y_outlet - self.y_inlet
        d_inlet = (y - self.y_inlet) / y_range  # 0 at inlet, 1 at outlet
        d_outlet = (self.y_outlet - y) / y_range  # 1 at inlet, 0 at outlet
        
        # h = 0 at t=0 and at boundaries
        # Simple: h = t * d_inlet * d_outlet (0 at IC and both BCs)
        h = t * d_inlet * d_outlet
        
        # g satisfies IC and BCs via interpolation
        # At t=0: g = T_init
        # At inlet (d_inlet=0): g = T_inlet
        # At outlet (d_outlet=0): g = T_outlet
        # This is tricky... use weighted blend
        t_scale = torch.clamp(t, min=0.0, max=1.0)  # Normalize time
        g = (1 - t_scale) * self.T_init + t_scale * (d_outlet * self.T_inlet + d_inlet * self.T_outlet)
        
        # Network output
        N_out = self.base_network(inputs)
        
        # Final: T = g + h * N
        T = g + h * N_out
        
        return T


if __name__ == "__main__":
    # Test HardICWrapper
    from modulus.models.mlp import FullyConnected
    
    # Create base network
    base_net = FullyConnected(
        in_features=11,  # x, y, z, t, 7 config
        out_features=1,
        layer_size=64,
        num_layers=2,
        activation_fn="silu",
    )
    
    T_init = 25.0  # Initial temperature
    
    # Wrap with hard IC
    model = HardICWrapper(base_net, T_init=T_init, t_idx=3)
    
    # Test: at t=0, output should be T_init
    N = 10
    inputs = torch.rand(N, 11)
    inputs[:, 3] = 0.0  # t = 0
    
    T_out = model(inputs)
    print(f"Hard IC test at t=0:")
    print(f"  T_init = {T_init}")
    print(f"  T_out mean = {T_out.mean().item():.4f}")
    print(f"  All equal to T_init: {torch.allclose(T_out, torch.full_like(T_out, T_init))}")
    
    # Test: at t>0, output differs from T_init
    inputs[:, 3] = 1.0  # t = 1
    T_out = model(inputs)
    print(f"\nHard IC test at t=1:")
    print(f"  T_out mean = {T_out.mean().item():.4f}")
    print(f"  Different from T_init: {not torch.allclose(T_out, torch.full_like(T_out, T_init))}")
