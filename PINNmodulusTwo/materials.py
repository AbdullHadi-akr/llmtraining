"""Self-contained material-property loader for PINNmodulusTwo.

Reads EVERYTHING from the local ``material_properties/`` folder:
  * scalar constants  <- ``constants.yaml`` (summarised from the PDF), and
  * per-point arrays  <- the ``Cell Center/`` and ``JR1 Center/`` CSVs.

No material numbers are hardcoded in code — the constants live only in
``constants.yaml``. This replaces the earlier dependency on the external
``pinn.data.load_properties`` module (which had constants baked into Python).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

MAT_DIR = Path(__file__).resolve().parent / "material_properties"
CC_DIR = MAT_DIR / "Cell Center"
JR1_DIR = MAT_DIR / "JR1 Center"


def _load_row_csv(path: Path) -> np.ndarray:
    """Property CSVs store one header row + one data row (one column per point)."""
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=np.float64)


def load_constants() -> dict:
    return yaml.safe_load((MAT_DIR / "constants.yaml").read_text())


def load_material_properties(layer: np.ndarray) -> Dict[str, np.ndarray]:
    """Return per-point ``rho``, ``Cp``, ``lambda_tensor`` (n,3,3) and ``region``.

    Args:
        layer: (n_points,) labels with values 'cc', 'jr1c', 'g'.
    """
    c = load_constants()
    n = len(layer)
    rho = np.zeros(n, dtype=np.float64)
    Cp = np.zeros(n, dtype=np.float64)
    lam = np.zeros((n, 3, 3), dtype=np.float64)
    region = np.zeros(n, dtype=np.int64)

    m_cc = layer == "cc"
    m_jr1 = layer == "jr1c"
    m_g = layer == "g"

    # ---- Cell Center: all per-point from CSV, off-diagonals 0 -----------------
    if m_cc.any():
        rho[m_cc] = _load_row_csv(CC_DIR / "Density_Grid_CellCenter.csv")
        Cp[m_cc] = _load_row_csv(CC_DIR / "SpecificHeat_Grid_CellCenter.csv")
        lam[m_cc, 0, 0] = _load_row_csv(CC_DIR / "ThermalConductivityXX_Grid_CellCenter.csv")
        lam[m_cc, 1, 1] = _load_row_csv(CC_DIR / "ThermalConductivityYY_Grid_CellCenter.csv")
        lam[m_cc, 2, 2] = _load_row_csv(CC_DIR / "ThermalConductivityZZ_Grid_CellCenter.csv")
        region[m_cc] = 0

    # ---- JR1: scalar rho/Cp/lambda_zz from constants, lambda_xx/xy/yy from CSV -
    if m_jr1.any():
        jr1 = c["jr1"]
        rho[m_jr1] = jr1["density"]
        Cp[m_jr1] = jr1["specific_heat"]
        lxx = _load_row_csv(JR1_DIR / "ThermalConductivityXX_Grid_JR1Center.csv")
        lxy = _load_row_csv(JR1_DIR / "ThermalConductivityXY_Grid_JR1Center.csv")
        lyy = _load_row_csv(JR1_DIR / "ThermalConductivityYY_Grid_JR1Center.csv")
        lam[m_jr1, 0, 0] = lxx
        lam[m_jr1, 0, 1] = lxy
        lam[m_jr1, 1, 0] = lxy   # symmetric
        lam[m_jr1, 1, 1] = lyy
        lam[m_jr1, 2, 2] = jr1["lambda_zz"]
        region[m_jr1] = 1

    # ---- Housing: all scalar isotropic from constants -------------------------
    if m_g.any():
        h = c["housing"]
        rho[m_g] = h["density"]
        Cp[m_g] = h["specific_heat"]
        lam[m_g, 0, 0] = h["lambda_iso"]
        lam[m_g, 1, 1] = h["lambda_iso"]
        lam[m_g, 2, 2] = h["lambda_iso"]
        region[m_g] = 2

    return {"rho": rho, "Cp": Cp, "lambda_tensor": lam, "region": region}


if __name__ == "__main__":
    import json
    lay = np.array(["cc"] * 121 + ["jr1c"] * 121 + ["g"] * 121)
    p = load_material_properties(lay)
    for r, name in [(0, "CC"), (1, "JR1"), (2, "Housing")]:
        m = p["region"] == r
        print(f"{name:8s} n={m.sum():3d}  rho[{p['rho'][m].min():.1f},{p['rho'][m].max():.1f}]"
              f"  Cp[{p['Cp'][m].min():.1f},{p['Cp'][m].max():.1f}]"
              f"  lam_diag_mean={np.round(p['lambda_tensor'][m][:, [0,1,2], [0,1,2]].mean(0), 2).tolist()}")
    print("constants.yaml:", json.dumps(load_constants(), indent=0)[:200])
