"""Load OP01 data from the cached npz file and heat source CSV.

Provides coordinates (363×3), temperature labels T(t, 363), bc_V(t),
config scalars, and q̇(t) from heat source CSV.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def load_op01_data(
    npz_path: str,
    heat_source_csv: str,
    subsample_time: int = 10,
    v_jr1: float = 4.394793e-04,
) -> Dict[str, Any]:
    """Load OP01 data for PINN training.

    Args:
        npz_path: Path to OP01.npz file.
        heat_source_csv: Path to heat source CSV (OP1_Heat Source.csv).
        subsample_time: Subsample every N timesteps (e.g., 10 -> 1s effective dt from 0.1s raw).
        v_jr1: JR1 volume in m^3 for q̇ calculation.

    Returns:
        Dictionary with keys:
            - t: time array (N_t,)
            - dt: time step in seconds
            - xyz: coordinates (363, 3)
            - T: temperature labels (N_t, 363)
            - bc_V: voltage (N_t,)
            - bc_OCV: open circuit voltage (N_t,)
            - bc_I: current (N_t,)
            - layer: layer labels (363,) - 'cc', 'jr1c', 'g'
            - config: dict of config scalar names to values
            - q_dot: heat source per unit volume (N_t,) in W/m^3 for JR1+CC region
    """
    # Load npz
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract arrays
    t_full = data["t_fast"]  # (14450,)
    T_full = data["T"]       # (14450, 363)
    bc_V_full = data["bc_V"]
    bc_OCV_full = data["bc_OCV"]
    bc_I_full = data["bc_I"]
    xyz = data["xyz"]        # (363, 3)
    layer = data["layer"]    # (363,) with values 'cc', 'jr1c', 'g'
    
    # Config scalars
    config_names = eval(data["sim_config_scalar_names_json"].item())
    config_values = data["sim_config_scalar"]
    config = dict(zip(config_names, config_values))
    
    # Subsample time
    if subsample_time > 1:
        t = t_full[::subsample_time]
        T = T_full[::subsample_time]
        bc_V = bc_V_full[::subsample_time]
        bc_OCV = bc_OCV_full[::subsample_time]
        bc_I = bc_I_full[::subsample_time]
    else:
        t = t_full
        T = T_full
        bc_V = bc_V_full
        bc_OCV = bc_OCV_full
        bc_I = bc_I_full
    
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # Load heat source CSV and compute q̇
    q_dot = _load_heat_source(heat_source_csv, t, v_jr1)
    
    return {
        "t": t.astype(np.float32),
        "dt": float(dt),
        "xyz": xyz.astype(np.float32),
        "T": T.astype(np.float32),
        "bc_V": bc_V.astype(np.float32),
        "bc_OCV": bc_OCV.astype(np.float32),
        "bc_I": bc_I.astype(np.float32),
        "layer": layer,
        "config": config,
        "q_dot": q_dot.astype(np.float32),
    }


def _load_heat_source(
    csv_path: str,
    t_target: np.ndarray,
    v_jr1: float,
    n_jr1_points: int = 121,
) -> np.ndarray:
    """Load heat source from CSV and interpolate to target times.
    
    q̇ = Heat Source JR1 / (V_JR1 · N_JR1_points) [W/m^3]
    
    Per the Notion "Gleichverteilung" spec the total JR1 heat source is spread
    equally over the 121 JR1 grid points, so we divide by the JR1 volume AND by
    the number of JR1 points exactly once.
    
    Args:
        csv_path: Path to OP1_Heat Source.csv
        t_target: Target time array to interpolate to
        v_jr1: JR1 volume in m^3
        n_jr1_points: number of JR1 grid points for the Gleichverteilung (121)
        
    Returns:
        q_dot array (N_t,) in W/m^3
    """
    df = pd.read_csv(csv_path)
    
    # Extract time and heat source JR1
    t_csv = df["Physical Time (s)"].values
    q_jr1 = df["Heat Source JR1 Monitor (W)"].values
    
    # Interpolate to target times
    q_jr1_interp = np.interp(t_target, t_csv, q_jr1)
    
    # Convert to volumetric heat source, equally distributed over the JR1 points
    q_dot = q_jr1_interp / (v_jr1 * n_jr1_points)
    
    return q_dot


def get_config_tensor(config: Dict[str, float], n_points: int) -> np.ndarray:
    """Expand config scalars to match grid points.
    
    Args:
        config: Dict of config name -> value
        n_points: Number of grid points
        
    Returns:
        Array (n_points, n_config) with config values repeated for each point
    """
    config_values = np.array([
        config["c_rate"],
        config["cell_current"],
        config["fluid_initial_temp"],
        config["fluid_inlet_temp"],
        config["fluid_mass_flow"],
        config["soc_start"],
        config["solid_initial_temp"],
    ], dtype=np.float32)
    
    # Repeat for each point
    return np.tile(config_values, (n_points, 1))


if __name__ == "__main__":
    # Test loading
    import os
    project_root = Path(__file__).parent.parent.parent.parent
    
    data = load_op01_data(
        npz_path=str(project_root / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
        heat_source_csv=str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/OP01_raw/OP01/OP1_Heat Source.csv"),
        subsample_time=10,
    )
    
    print("OP01 Data loaded:")
    print(f"  t shape: {data['t'].shape}, dt={data['dt']:.2f}s")
    print(f"  xyz shape: {data['xyz'].shape}")
    print(f"  T shape: {data['T'].shape}")
    print(f"  bc_V shape: {data['bc_V'].shape}")
    print(f"  layer unique: {np.unique(data['layer'])}")
    print(f"  config: {data['config']}")
    print(f"  q_dot shape: {data['q_dot'].shape}, range: [{data['q_dot'].min():.2f}, {data['q_dot'].max():.2f}] W/m^3")
