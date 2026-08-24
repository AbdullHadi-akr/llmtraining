"""Voltage (bc_V) prediction network.

Small MLP for bc_V prediction from time and config scalars.
Data-only fit (no physics), non-recurrent for now.

TODO (later):
- Add recurrence: bc_V_{t-1..t-k}
- Add bc_V IC: bc_V(0) from config
"""

import torch
import torch.nn as nn
from modulus.models.mlp import FullyConnected


class NetV(nn.Module):
    """Voltage prediction network (data-only, non-recurrent).
    
    Input: (t, config[7])
    Output: bc_V
    """
    
    def __init__(
        self,
        depth: int = 2,
        width: int = 64,
        activation: str = "silu",
        n_config: int = 7,
    ):
        """Initialize the voltage network.
        
        Args:
            depth: Number of hidden layers
            width: Neurons per hidden layer
            activation: Activation function
            n_config: Number of config scalars
        """
        super().__init__()
        
        # Input: t (1) + config (7)
        self.in_features = 1 + n_config
        
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
            inputs: Shape (..., 8) = (t, config[7])
            
        Returns:
            bc_V: Predicted voltage (..., 1)
        """
        return self.net(inputs)


def create_net_V(
    depth: int = 2,
    width: int = 64,
    activation: str = "silu",
    n_config: int = 7,
) -> nn.Module:
    """Factory function for voltage network.
    
    Args:
        depth: Number of hidden layers
        width: Neurons per hidden layer  
        activation: Activation function
        n_config: Number of config scalars
        
    Returns:
        NetV instance
    """
    return NetV(
        depth=depth,
        width=width,
        activation=activation,
        n_config=n_config,
    )


if __name__ == "__main__":
    # Test network
    device = torch.device("cpu")
    
    net = create_net_V(depth=2, width=64)
    print(f"NetV created: in_features={net.in_features}")
    
    # Test forward pass
    N = 100
    inputs = torch.rand(N, net.in_features, device=device)
    V_out = net(inputs)
    print(f"Forward pass: input={inputs.shape}, output={V_out.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {n_params}")
