"""Data loading modules for PINN."""
from .load_op01 import load_op01_data
from .load_properties import load_material_properties
from .load_faces import load_inlet_outlet_faces

__all__ = ["load_op01_data", "load_material_properties", "load_inlet_outlet_faces"]
