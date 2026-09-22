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

# Die Geometrie des echten Gitters. Die Tests bauen sie nach, statt einen
# Wuerfel zu nehmen: ein aequidistantes Testgitter wuerde genau die Eigenschaft
# nicht treffen, die hier gefaehrlich ist -- dass x NICHT aequidistant ist.
#
# ⚠ KORRIGIERT am 22.09. Bis dahin standen hier 0.0198089 / 0.0104441 und
# x = 0.010786 -- am 02.09. von Hand ausgezaehlt und um 5 bis 9 um daneben.
# Weil ``test_layout_trifft_die_echte_geometrie`` dieselben Konstanten prueft,
# aus denen diese Fixture das Gitter baut, war der Test zirkulaer: er konnte
# nicht fallen, und gegen die ECHTEN Koordinaten waeren alle vier Zusagen
# durchgefallen (dx um 4.6e-7, die Spannweiten um 5.4e-6 bzw. 9.0e-6).
#
# Die Werte unten sind aus den drei CSVs in
# ``legacy/battery_surrogate_agenticWorkflow/coordinates/`` gemessen -- den
# Dateien, die auch im Repo liegen. ``test_layout_aus_den_echten_koordinaten``
# leitet das Gitter direkt aus ihnen ab und schliesst den Kreis damit dauerhaft:
# faellt diese Fixture wieder von der Wirklichkeit ab, faellt jener Test.
X_PLANES = np.array([0.0, 0.010785542, 0.021900])
DY, DZ = 0.0198094368, 0.0104431991
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
