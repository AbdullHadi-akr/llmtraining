"""Der Reshape und die drei Paddings.

Diese Tests bewachen die **strukturellen** Zusagen des Entwurfs -- die, die
exakt gelten muessen und nicht nur ungefaehr. Deshalb steht hier an mehreren
Stellen ``== 0.0`` und keine Toleranz: waere die Symmetrie nur naeherungsweise
erfuellt, waere sie ein Strafterm mit anderem Namen, und der ganze Grund, x
nicht in die Kanaele zu legen, waere weg.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grid as gridmod
from conftest import DY, DZ, NY, NZ, X_PLANES


# ---------------------------------------------------------------------------
# Ableiten statt raten
# ---------------------------------------------------------------------------
def test_layout_trifft_die_echte_geometrie(layout):
    assert layout.shape == (3, NY, NZ)
    assert layout.n_points == 363
    np.testing.assert_allclose(layout.xu, X_PLANES, atol=1e-12)
    # Die 3 % Unterschied in x sind der Grund fuer die nicht-aequidistante
    # Zweite-Ableitungs-Formel in physics.py. Faellt der Test, ist entweder das
    # Gitter ein anderes oder die Formel unnoetig -- beides will man wissen.
    np.testing.assert_allclose(layout.dx, [0.010786, 0.011114], atol=1e-9)
    assert abs(layout.dx[1] / layout.dx[0] - 1.0) > 0.02

    # Kante auf Kante: das Raster spannt die Zellflaeche, nicht irgendetwas
    # darin. Der legacy-README nennt dy=0.198, dz=0.104.
    assert layout.yu[-1] - layout.yu[0] == pytest.approx(0.198089, abs=1e-6)
    assert layout.zu[-1] - layout.zu[0] == pytest.approx(0.104441, abs=1e-6)


def test_luecken_im_raster_fallen_auf():
    """Ein unvollstaendiges Gitter muss laut scheitern, nicht still verwuerfeln."""
    xx, yy, zz = np.meshgrid(X_PLANES, np.arange(4) * DY, np.arange(4) * DZ,
                             indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)[:-1]
    with pytest.raises(ValueError, match="Kein volles Tensorgitter"):
        gridmod.derive_layout(xyz)


def test_nicht_aequidistantes_y_faellt_auf():
    """y und z muessen aequidistant sein -- der zentrale Stern setzt es voraus."""
    ys = np.array([0.0, 1.0, 3.0])           # absichtlich ungleichmaessig
    xx, yy, zz = np.meshgrid(X_PLANES, ys, np.arange(3) * DZ, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    with pytest.raises(ValueError, match="nicht aequidistant"):
        gridmod.derive_layout(xyz)


# ---------------------------------------------------------------------------
# Hin und zurueck
# ---------------------------------------------------------------------------
def test_reshape_ist_exakt_umkehrbar(layout):
    """Kein ``allclose``: ein Reshape, der rundet, ist keiner."""
    flat = torch.arange(layout.n_points, dtype=torch.float64)
    assert torch.equal(gridmod.to_flat(gridmod.to_field(flat, layout), layout),
                       flat)


def test_reshape_traegt_fuehrende_achsen(layout):
    """Eine ganze Trajektorie muss genauso durchgehen wie ein Schnappschuss."""
    traj = torch.randn(7, layout.n_points, dtype=torch.float64)
    field = gridmod.to_field(traj, layout)
    assert field.shape == (7, *layout.shape)
    assert torch.equal(gridmod.to_flat(field, layout), traj)


def test_reshape_ordnet_nach_koordinaten_nicht_nach_index(layout):
    """``flat_index[i,j,k]`` muss zum Punkt mit (xu[i], yu[j], zu[k]) gehoeren.

    Das ist die eigentliche Zusage: der Reshape wird aus den Koordinaten
    ABGELEITET. Ein Test, der nur die Form prueft, wuerde eine Permutation
    durchgehen lassen -- und eine verdrehte Achse ist genau der Fehler, der
    erst drei Wochen spaeter als schlechte Konvergenz auffaellt.
    """
    xx, yy, zz = np.meshgrid(
        X_PLANES, np.arange(NY) * DY, np.arange(NZ) * DZ, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    for i in (0, 2):
        for j in (0, 5, NY - 1):
            for k in (0, NZ - 1):
                p = layout.flat_index[i, j, k]
                np.testing.assert_allclose(
                    xyz[p], [layout.xu[i], layout.yu[j], layout.zu[k]],
                    atol=1e-12)


# ---------------------------------------------------------------------------
# Die Paddings
# ---------------------------------------------------------------------------
def test_ghost_lo_ist_die_spiegelung(layout, random_field):
    """``ghost_lo`` wird berechnet, nicht uebergeben -- und ist exakt T1."""
    padded = gridmod.pad_x(random_field, random_field[-2])
    assert padded.shape == (layout.shape[0] + 2, *layout.shape[1:])
    assert torch.equal(padded[0], random_field[1])


def test_ghost_hi_wird_durchgereicht(layout, random_field):
    """``ghost_hi`` traegt Physik und kommt von aussen -- unveraendert."""
    ghost = torch.full(layout.shape[1:], 42.0, dtype=torch.float64)
    padded = gridmod.pad_x(random_field, ghost)
    assert torch.equal(padded[-1], ghost)


def test_reflect_spiegelt_um_den_randknoten(random_field):
    """``reflect`` und nicht ``replicate``: Geist := erster INNERER Knoten."""
    p = gridmod.pad_yz(random_field)
    assert torch.equal(p[:, 0, 1:-1], random_field[:, 1, :])     # y unten
    assert torch.equal(p[:, -1, 1:-1], random_field[:, -2, :])   # y oben
    assert torch.equal(p[:, 1:-1, 0], random_field[:, :, 1])     # z unten
    assert torch.equal(p[:, 1:-1, -1], random_field[:, :, -2])   # z oben


def test_zentrale_differenz_am_yz_rand_ist_exakt_null(layout, random_field):
    """Die Neumann-0-Bedingung, algebraisch statt naeherungsweise.

    ``replicate`` (Geist := Randwert) wuerde hier NICHT null liefern -- das ist
    der ganze Unterschied zwischen den beiden, und er haengt daran, dass der
    Randknoten auf der Grenze liegt.
    """
    p = gridmod.pad_yz(random_field)
    dy_edge = (p[:, 2, :] - p[:, 0, :]) / (2 * layout.dy)
    dz_edge = (p[:, :, 2] - p[:, :, 0]) / (2 * layout.dz)
    assert float(dy_edge.abs().max()) == 0.0
    assert float(dz_edge.abs().max()) == 0.0


def test_pad_all_ist_pad_x_dann_pad_yz(layout, random_field):
    ghost = random_field[-2]
    assert torch.equal(
        gridmod.pad_all(random_field, ghost),
        gridmod.pad_yz(gridmod.pad_x(random_field, ghost)))
    assert gridmod.pad_all(random_field, ghost).shape == tuple(
        s + 2 for s in layout.shape)
