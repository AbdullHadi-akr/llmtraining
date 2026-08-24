#!/usr/bin/env python3
"""Quick diagnostic to identify NaN source."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

print("Loading data...", flush=True)
from pinn.data.load_op01 import load_op01_data
from pinn.data.load_properties import load_material_properties
from pinn.data.load_faces import load_inlet_outlet_faces

project_root = Path(__file__).parent.parent.parent.parent
op01_data = load_op01_data(
    npz_path=str(project_root / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
    heat_source_csv=str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/OP01_raw/OP01/OP1_Heat Source.csv"),
    subsample_time=10,
)
props = load_material_properties(
    layer=op01_data["layer"],
    props_cc_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/Cell Center"),
    props_jr1_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/JR1 Center"),
)

print(f"\nData ranges:", flush=True)
print(f"  t: [{op01_data['t'].min():.2f}, {op01_data['t'].max():.2f}] s", flush=True)
print(f"  x: [{op01_data['xyz'][:, 0].min():.4f}, {op01_data['xyz'][:, 0].max():.4f}] m", flush=True)
print(f"  y: [{op01_data['xyz'][:, 1].min():.4f}, {op01_data['xyz'][:, 1].max():.4f}] m", flush=True)
print(f"  z: [{op01_data['xyz'][:, 2].min():.4f}, {op01_data['xyz'][:, 2].max():.4f}] m", flush=True)
print(f"  T: [{op01_data['T'].min():.2f}, {op01_data['T'].max():.2f}] °C", flush=True)
print(f"  q_dot: [{op01_data['q_dot'].min():.2e}, {op01_data['q_dot'].max():.2e}] W/m³", flush=True)

print(f"\nMaterial property ranges:", flush=True)
print(f"  rho: [{props['rho'].min():.1f}, {props['rho'].max():.1f}] kg/m³", flush=True)
print(f"  Cp: [{props['Cp'].min():.1f}, {props['Cp'].max():.1f}] J/(kg·K)", flush=True)
print(f"  lambda_tensor diag: [{props['lambda_tensor'][:, 0, 0].min():.1f}, {props['lambda_tensor'][:, 0, 0].max():.1f}] W/(m·K)", flush=True)

# The issue: rho*Cp ~ 2700*900 ~ 2.4e6
# T_t ~ dT/dt ~ 10°C / 1000s ~ 0.01 K/s
# rho*Cp*T_t ~ 2.4e6 * 0.01 ~ 2.4e4
# lambda*T_xx ~ 200 * T_xx
# To balance: T_xx ~ 2.4e4 / 200 ~ 120 K/m²
# But spatial scale is ~0.02m, so T_xx ~ T / (0.02)² ~ T / 4e-4
# If T ~ 25°C, T_xx ~ 25 / 4e-4 ~ 6e4 -- way too large for balance!
# This explains the NaN: numerical instability from scale mismatch.

print(f"\nScale analysis (why NaN):", flush=True)
rho_Cp = props['rho'].mean() * props['Cp'].mean()
print(f"  rho*Cp ~ {rho_Cp:.2e}", flush=True)
dx = 0.02  # ~spatial scale
print(f"  dx ~ {dx} m", flush=True)
print(f"  T_xx ~ T/dx² ~ 25/{dx**2:.4f} ~ {25/dx**2:.2e}", flush=True)
print(f"  lambda*T_xx ~ 200 * {25/dx**2:.2e} ~ {200*25/dx**2:.2e}", flush=True)
print(f"  But rho*Cp*T_t ~ {rho_Cp:.2e} * 0.01 ~ {rho_Cp*0.01:.2e}", flush=True)
print(f"  Mismatch: {200*25/dx**2 / (rho_Cp*0.01):.1f}x -- causes NaN!", flush=True)

print("\nRecommendation: normalize inputs/outputs, use characteristic scales.", flush=True)
