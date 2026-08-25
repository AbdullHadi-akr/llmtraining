#!/usr/bin/env python3
"""Simple test of architecture sweep functionality."""

import sys
from pathlib import Path

# Add package to path
PINN_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PINN_ROOT))

print("Testing architecture sweep...", flush=True)
print(f"PINN_ROOT: {PINN_ROOT}", flush=True)

# Test imports
print("Importing torch...", flush=True)
import torch
print(f"PyTorch version: {torch.__version__}", flush=True)

print("Importing numpy...", flush=True)
import numpy as np
print(f"NumPy version: {np.__version__}", flush=True)

print("Importing data loaders...", flush=True)
from pinn.data.load_op01 import load_op01_data
from pinn.data.load_properties import load_material_properties
from pinn.data.load_faces import load_inlet_outlet_faces
print("Data loaders imported!", flush=True)

print("Importing models...", flush=True)
from pinn.models.net_T import create_net_T
from pinn.models.net_V import create_net_V
print("Models imported!", flush=True)

print("Importing trainer...", flush=True)
from pinn.train.solve_T import TemperaturePINNTrainer, LossComponents
print("Trainer imported!", flush=True)

print("All imports successful!", flush=True)

# Quick architecture test
print("\nCreating test networks:", flush=True)

configs = [
    ("small", 2, 64),
    ("medium", 4, 128),
    ("large", 6, 256),
]

for name, depth, width in configs:
    net = create_net_T(depth=depth, width=width, k=2, hard_ic=True, T_init=298.0)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  {name}: depth={depth}, width={width}, params={n_params:,}", flush=True)

print("\nDone!", flush=True)
