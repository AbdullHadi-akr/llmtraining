"""Testbootstrap: ``GridCNN/`` importierbar machen.

Anders als ``PINNmodulusTwo/tests/conftest.py`` muss hier nichts gestubbt
werden -- ``GridCNN`` braucht kein Modulus, nur numpy und torch. Das ist eine
der wenigen Stellen, an denen dieses Projekt billiger ist als das bestehende,
und sie ist Absicht: die Tests sollen ueberall laufen, nicht nur auf dem
GPU-Server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Die Geometrie des echten Gitters, am 02.09. aus den Koordinaten ausgezaehlt.
# Die Tests bauen sie nach, statt einen Wuerfel zu nehmen: ein aequidistantes
# Testgitter wuerde genau die Eigenschaft nicht treffen, die hier gefaehrlich
# ist -- dass x NICHT aequidistant ist.
X_PLANES = np.array([0.0, 0.010786, 0.021900])
DY, DZ = 0.0198089, 0.0104441
NY = NZ = 11


@pytest.fixture
def layout():
    """Das echte 3 x 11 x 11 mit den echten Abstaenden."""
    import grid as gridmod
    xx, yy, zz = np.meshgrid(
        X_PLANES, np.arange(NY) * DY, np.arange(NZ) * DZ, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return gridmod.derive_layout(xyz)


@pytest.fixture
def random_field(layout):
    """Ein zufaelliges Feld -- die Randzusagen duerfen nicht von T abhaengen."""
    import torch
    torch.manual_seed(0)
    return torch.randn(layout.shape, dtype=torch.float64)
