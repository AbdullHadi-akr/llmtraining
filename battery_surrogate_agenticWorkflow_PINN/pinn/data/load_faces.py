"""Load inlet and outlet face coordinates from CSV.

The faces CSV has columns: label, type, x, y, z
Types are 'inlet' and 'outlet'.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


def load_inlet_outlet_faces(csv_path: str) -> Dict[str, np.ndarray]:
    """Load inlet and outlet face center coordinates.

    Args:
        csv_path: Path to inlet_outlet_faces.csv.

    Returns:
        Dictionary with keys:
            - inlet_xyz: inlet face centers (N_in, 3)
            - outlet_xyz: outlet face centers (N_out, 3)
            - inlet_labels: inlet face labels (N_in,)
            - outlet_labels: outlet face labels (N_out,)
    """
    df = pd.read_csv(csv_path)
    
    # Split by type
    inlet_df = df[df["type"] == "inlet"]
    outlet_df = df[df["type"] == "outlet"]
    
    # Extract coordinates
    inlet_xyz = inlet_df[["x", "y", "z"]].values.astype(np.float32)
    outlet_xyz = outlet_df[["x", "y", "z"]].values.astype(np.float32)
    
    # Extract labels
    inlet_labels = inlet_df["label"].values
    outlet_labels = outlet_df["label"].values
    
    return {
        "inlet_xyz": inlet_xyz,
        "outlet_xyz": outlet_xyz,
        "inlet_labels": inlet_labels,
        "outlet_labels": outlet_labels,
    }


if __name__ == "__main__":
    # Test loading
    project_root = Path(__file__).parent.parent.parent.parent
    
    faces = load_inlet_outlet_faces(
        str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/inlet_outlet_faces.csv")
    )
    
    print("Inlet/Outlet faces loaded:")
    print(f"  inlet_xyz shape: {faces['inlet_xyz'].shape}")
    print(f"  outlet_xyz shape: {faces['outlet_xyz'].shape}")
    print(f"  inlet y-coord (should be ~-0.1265): {faces['inlet_xyz'][:, 1].mean():.4f}")
    print(f"  outlet y-coord (should be ~0.14605): {faces['outlet_xyz'][:, 1].mean():.4f}")
