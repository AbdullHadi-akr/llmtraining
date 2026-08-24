"""Neural network models for PINN."""
from .net_T import create_net_T, NetT
from .net_V import create_net_V, NetV

__all__ = ["create_net_T", "NetT", "create_net_V", "NetV"]
