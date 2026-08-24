"""Physics modules for PINN."""
from .anisotropic_heat import AnisotropicHeatTransient
from .hard_constraint import HardICWrapper

__all__ = ["AnisotropicHeatTransient", "HardICWrapper"]
