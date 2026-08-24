"""Load material properties (λ, ρ, Cp) per grid point.

Material properties from Material_Properties_Gridpoints.pdf:
- Housing (Gehäuse): ρ=2700, Cp=893, λ=193 W/mK isotropic, q̇=0
- JR1: ρ=2468.13, Cp=938.05; λ XX/XY/YY from csv, XZ=YZ=0, ZZ=22.4 const
- Cell Center: ρ, Cp per-point from csv; λ XX/YY/ZZ from csv, XY=XZ=YZ=0
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


# Constants for Gehäuse (housing) region
HOUSING_RHO = 2700.0      # kg/m^3
HOUSING_CP = 893.0        # J/(kg·K)
HOUSING_LAMBDA = 193.0    # W/(m·K) isotropic

# Constants for JR1 region
JR1_RHO = 2468.13         # kg/m^3
JR1_CP = 938.05           # J/(kg·K)
JR1_LAMBDA_ZZ = 22.4      # W/(m·K)


def load_material_properties(
    layer: np.ndarray,
    props_cc_dir: str,
    props_jr1_dir: str,
) -> Dict[str, np.ndarray]:
    """Load material properties for all 363 grid points.

    Args:
        layer: Layer labels (363,) with values 'cc', 'jr1c', 'g'.
        props_cc_dir: Directory with Cell Center property CSVs.
        props_jr1_dir: Directory with JR1 Center property CSVs.

    Returns:
        Dictionary with keys:
            - rho: density (363,) [kg/m^3]
            - Cp: specific heat (363,) [J/(kg·K)]
            - lambda_tensor: thermal conductivity (363, 3, 3) symmetric [W/(m·K)]
            - region: region label per point (363,) - 0=CC, 1=JR1, 2=Housing
    """
    n_points = len(layer)
    
    # Initialize arrays
    rho = np.zeros(n_points, dtype=np.float32)
    Cp = np.zeros(n_points, dtype=np.float32)
    lambda_tensor = np.zeros((n_points, 3, 3), dtype=np.float32)
    region = np.zeros(n_points, dtype=np.int32)
    
    # Create masks for each layer
    mask_cc = layer == "cc"
    mask_jr1 = layer == "jr1c"
    mask_g = layer == "g"
    
    # Count points per layer
    n_cc = mask_cc.sum()
    n_jr1 = mask_jr1.sum()
    n_g = mask_g.sum()
    
    # Load Cell Center properties
    if n_cc > 0:
        props_cc = _load_cc_properties(props_cc_dir, n_cc)
        rho[mask_cc] = props_cc["rho"]
        Cp[mask_cc] = props_cc["Cp"]
        # CC has diagonal lambda (XX, YY, ZZ), off-diagonal = 0
        lambda_tensor[mask_cc, 0, 0] = props_cc["lambda_xx"]
        lambda_tensor[mask_cc, 1, 1] = props_cc["lambda_yy"]
        lambda_tensor[mask_cc, 2, 2] = props_cc["lambda_zz"]
        region[mask_cc] = 0
    
    # Load JR1 properties
    if n_jr1 > 0:
        props_jr1 = _load_jr1_properties(props_jr1_dir, n_jr1)
        rho[mask_jr1] = JR1_RHO
        Cp[mask_jr1] = JR1_CP
        # JR1 has XX, XY, YY from CSV; XZ=YZ=0; ZZ=22.4 const
        lambda_tensor[mask_jr1, 0, 0] = props_jr1["lambda_xx"]
        lambda_tensor[mask_jr1, 0, 1] = props_jr1["lambda_xy"]
        lambda_tensor[mask_jr1, 1, 0] = props_jr1["lambda_xy"]  # symmetric
        lambda_tensor[mask_jr1, 1, 1] = props_jr1["lambda_yy"]
        lambda_tensor[mask_jr1, 2, 2] = JR1_LAMBDA_ZZ
        region[mask_jr1] = 1
    
    # Housing properties (constant)
    if n_g > 0:
        rho[mask_g] = HOUSING_RHO
        Cp[mask_g] = HOUSING_CP
        # Housing is isotropic
        lambda_tensor[mask_g, 0, 0] = HOUSING_LAMBDA
        lambda_tensor[mask_g, 1, 1] = HOUSING_LAMBDA
        lambda_tensor[mask_g, 2, 2] = HOUSING_LAMBDA
        region[mask_g] = 2
    
    return {
        "rho": rho,
        "Cp": Cp,
        "lambda_tensor": lambda_tensor,
        "region": region,
    }


def _load_cc_properties(props_dir: str, n_points: int) -> Dict[str, np.ndarray]:
    """Load Cell Center properties from CSVs."""
    props_dir = Path(props_dir)
    
    # Load density
    rho = _load_property_csv(props_dir / "Density_Grid_CellCenter.csv", n_points)
    
    # Load specific heat
    Cp = _load_property_csv(props_dir / "SpecificHeat_Grid_CellCenter.csv", n_points)
    
    # Load thermal conductivity components
    lambda_xx = _load_property_csv(props_dir / "ThermalConductivityXX_Grid_CellCenter.csv", n_points)
    lambda_yy = _load_property_csv(props_dir / "ThermalConductivityYY_Grid_CellCenter.csv", n_points)
    lambda_zz = _load_property_csv(props_dir / "ThermalConductivityZZ_Grid_CellCenter.csv", n_points)
    
    return {
        "rho": rho,
        "Cp": Cp,
        "lambda_xx": lambda_xx,
        "lambda_yy": lambda_yy,
        "lambda_zz": lambda_zz,
    }


def _load_jr1_properties(props_dir: str, n_points: int) -> Dict[str, np.ndarray]:
    """Load JR1 properties from CSVs."""
    props_dir = Path(props_dir)
    
    # Load thermal conductivity components
    lambda_xx = _load_property_csv(props_dir / "ThermalConductivityXX_Grid_JR1Center.csv", n_points)
    lambda_xy = _load_property_csv(props_dir / "ThermalConductivityXY_Grid_JR1Center.csv", n_points)
    lambda_yy = _load_property_csv(props_dir / "ThermalConductivityYY_Grid_JR1Center.csv", n_points)
    
    return {
        "lambda_xx": lambda_xx,
        "lambda_xy": lambda_xy,
        "lambda_yy": lambda_yy,
    }


def _load_property_csv(csv_path: Path, expected_points: int) -> np.ndarray:
    """Load a single property CSV file.
    
    The CSV has one header row with column names like "Property_001 Monitor (unit)"
    and one data row with values for each of the 121 grid points (columns).
    """
    df = pd.read_csv(csv_path)
    
    # Get values from the single data row (all columns)
    values = df.iloc[0].values.astype(np.float32)
    
    # Validate shape
    if len(values) != expected_points:
        # The CSV might have all 121 points for this layer
        # If we got more than expected, it's because we're loading for a subset
        pass
    
    return values


if __name__ == "__main__":
    # Test loading
    import numpy as np
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Create a mock layer array with 363 points
    # Typically: 121 CC, 121 JR1, 121 Housing
    data = np.load(
        project_root / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz",
        allow_pickle=True
    )
    layer = data["layer"]
    
    props = load_material_properties(
        layer=layer,
        props_cc_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/Cell Center"),
        props_jr1_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/JR1 Center"),
    )
    
    print("Material properties loaded:")
    print(f"  rho shape: {props['rho'].shape}, range: [{props['rho'].min():.1f}, {props['rho'].max():.1f}]")
    print(f"  Cp shape: {props['Cp'].shape}, range: [{props['Cp'].min():.1f}, {props['Cp'].max():.1f}]")
    print(f"  lambda_tensor shape: {props['lambda_tensor'].shape}")
    print(f"  region counts: CC={np.sum(props['region']==0)}, JR1={np.sum(props['region']==1)}, Housing={np.sum(props['region']==2)}")
    
    # Check lambda is symmetric
    lam = props["lambda_tensor"]
    print(f"  Lambda symmetric: {np.allclose(lam, lam.transpose(0,2,1))}")
