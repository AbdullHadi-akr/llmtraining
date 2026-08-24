"""Temperature prediction network with autoregressive predicted history.

Input Keys: (x, y, z, t) + config scalars (7) + T̂_{t-1..t-k} (k past predictions)
Output: T (temperature at the given point and time)

The network uses autoregressive PREDICTED feedback (not teacher forcing):
- Feed the model's own past predictions T̂_{t-1..t-k}
- Warm-up first k steps from IC (T̂_{<0} := T_solid_init)
- Roll out sequentially, computing loss vs labeled T at each step
"""

import torch
import torch.nn as nn
from typing import Optional
from modulus.models.mlp import FullyConnected


class NetT(nn.Module):
    """Temperature prediction network with configurable depth/width.
    
    Wraps Modulus FullyConnected with proper input/output handling.
    """
    
    def __init__(
        self,
        depth: int = 4,
        width: int = 128,
        activation: str = "silu",
        k: int = 2,
        n_config: int = 7,
    ):
        """Initialize the temperature network.
        
        Args:
            depth: Number of hidden layers
            width: Number of neurons per hidden layer
            activation: Activation function ("silu", "tanh", "relu")
            k: Number of past T̂ predictions to include as input
            n_config: Number of config scalar inputs (default 7)
        """
        super().__init__()
        
        # Input features: x, y, z, t (4) + config (7) + T_history (k)
        self.in_features = 4 + n_config + k
        self.k = k
        self.n_config = n_config
        
        # Build the network
        self.net = FullyConnected(
            in_features=self.in_features,
            out_features=1,
            layer_size=width,
            num_layers=depth,
            activation_fn=activation,
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Shape (..., in_features) = (x, y, z, t, config[7], T_hist[k])
            
        Returns:
            T: Predicted temperature (..., 1)
        """
        return self.net(inputs)


def create_net_T(
    depth: int = 4,
    width: int = 128,
    activation: str = "silu",
    k: int = 2,
    n_config: int = 7,
    hard_ic: bool = True,
    T_init: float = 25.0,
) -> nn.Module:
    """Factory function to create temperature network.
    
    Args:
        depth: Number of hidden layers
        width: Neurons per hidden layer
        activation: Activation function
        k: Number of history steps
        n_config: Number of config scalars
        hard_ic: If True, wrap with HardICWrapper
        T_init: Initial temperature for hard IC
        
    Returns:
        Network (optionally wrapped for hard IC)
    """
    from ..physics.hard_constraint import HardICWrapper
    
    base_net = NetT(
        depth=depth,
        width=width,
        activation=activation,
        k=k,
        n_config=n_config,
    )
    
    if hard_ic:
        # Wrap with hard IC: T = T_init + t * N(...)
        # t_idx = 3 (after x, y, z)
        return HardICWrapper(base_net, T_init=T_init, t_idx=3)
    else:
        return base_net


class AutoregressiveRollout:
    """Helper class for autoregressive rollout with truncated BPTT.
    
    Manages the predicted history T̂_{t-1..t-k} during sequential rollout.
    """
    
    def __init__(
        self,
        k: int,
        T_init: float,
        n_points: int,
        bptt_window: int = 8,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize rollout state.
        
        Args:
            k: Number of history steps
            T_init: Initial temperature for warm-up
            n_points: Number of grid points
            bptt_window: Truncated BPTT window size
            device: Torch device
        """
        self.k = k
        self.T_init = T_init
        self.n_points = n_points
        self.bptt_window = bptt_window
        self.device = device
        
        # Initialize history buffer with IC
        # Shape: (k, n_points) - most recent at index 0
        self.history = torch.full(
            (k, n_points), T_init, dtype=torch.float32, device=device
        )
        
        # Track step count for BPTT
        self.step_count = 0
    
    def get_history(self) -> torch.Tensor:
        """Get current history as input tensor.
        
        Returns:
            T_history: Shape (n_points, k) for concatenation with other inputs
        """
        return self.history.T  # (n_points, k)
    
    def update(self, T_pred: torch.Tensor, detach: bool = False):
        """Update history with new prediction.
        
        Args:
            T_pred: New predicted temperature (n_points,) or (n_points, 1)
            detach: If True, detach from computation graph (for BPTT)
        """
        if T_pred.dim() == 2:
            T_pred = T_pred.squeeze(-1)
        
        if detach:
            T_pred = T_pred.detach()
        
        # Shift history: drop oldest, add newest at front
        self.history = torch.cat([
            T_pred.unsqueeze(0),
            self.history[:-1],
        ], dim=0)
        
        self.step_count += 1
    
    def should_detach(self) -> bool:
        """Check if we should detach for truncated BPTT."""
        return self.step_count > 0 and self.step_count % self.bptt_window == 0
    
    def reset(self):
        """Reset history to IC."""
        self.history.fill_(self.T_init)
        self.step_count = 0


if __name__ == "__main__":
    # Test network creation
    device = torch.device("cpu")
    
    # Test NetT
    net = NetT(depth=4, width=128, k=2, n_config=7)
    print(f"NetT created: in_features={net.in_features}")
    
    # Test forward pass
    N = 100
    inputs = torch.rand(N, net.in_features, device=device)
    T_out = net(inputs)
    print(f"Forward pass: input={inputs.shape}, output={T_out.shape}")
    
    # Test with hard IC
    net_hard = create_net_T(depth=4, width=128, k=2, hard_ic=True, T_init=25.0)
    
    # At t=0, should return T_init
    inputs_t0 = torch.rand(N, 4 + 7 + 2, device=device)
    inputs_t0[:, 3] = 0.0  # t = 0
    T_out_t0 = net_hard(inputs_t0)
    print(f"Hard IC at t=0: T={T_out_t0.mean().item():.4f} (should be 25.0)")
    
    # Test autoregressive rollout
    rollout = AutoregressiveRollout(k=2, T_init=25.0, n_points=363, bptt_window=8)
    T_hist = rollout.get_history()
    print(f"Initial history shape: {T_hist.shape}")
    print(f"Initial history values: [{T_hist[0,0]:.1f}, {T_hist[0,1]:.1f}]")
    
    # Simulate update
    T_pred = torch.full((363,), 26.0)
    rollout.update(T_pred)
    T_hist = rollout.get_history()
    print(f"After update: [{T_hist[0,0]:.1f}, {T_hist[0,1]:.1f}]")
