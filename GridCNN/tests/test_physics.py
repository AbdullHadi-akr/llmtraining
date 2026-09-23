"""Die Differenzensterne, der Wandterm und die Quelle.

Die Sterne werden gegen **analytische** Felder geprueft und ueber die
Konvergenzordnung, nicht gegen eine Toleranz. Ein Stern mit falscher Ordnung
ist nicht "etwas ungenauer" -- er ist ein anderer Operator, und eine grosszuegig
gewaehlte Toleranz wuerde das durchgehen lassen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grid as gridmod
import physics as phys


# ---------------------------------------------------------------------------
# Die Symmetrie -- die Zusage, die exakt sein muss
# ---------------------------------------------------------------------------
def test_dTdx_an_der_zellmitte_ist_exakt_null(layout, random_field):
    """Der Kern des ganzen Layout-Arguments.

    Kein ``pytest.approx``: im Zaehler steht ``T1 - T1``, und ``a - a`` ist
    null fuer jedes ``a``. Waere hier eine Toleranz noetig, waere die
    Randbedingung nicht strukturell, und x koennte genauso gut in den Kanaelen
    liegen.
    """
    padded = gridmod.pad_all(random_field, random_field[-2])
    dx = phys.d_dx(padded, layout)
    assert float(dx[0].abs().max()) == 0.0


def test_symmetrie_gilt_fuer_jedes_feld(layout):
    """Auch fuer boesartige Felder -- sie haengt nicht an Glattheit."""
    for scale in (1e-8, 1.0, 1e8):
        torch.manual_seed(1)
        f = torch.randn(layout.shape, dtype=torch.float64) * scale
        padded = gridmod.pad_all(f, f[-2])
        assert float(phys.d_dx(padded, layout)[0].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Konvergenzordnung
# ---------------------------------------------------------------------------
def _stretched(n: int):
    """Gitter mit n Knoten je Achse, x absichtlich gestreckt."""
    xs = np.linspace(0.0, 1.0, n) ** 1.3
    ys = zs = np.linspace(0.0, 1.0, n)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return gridmod.derive_layout(xyz), xx, yy, zz


def _analytic(xx, yy, zz):
    f = np.sin(2.0 * xx) * np.cos(1.5 * yy) * np.exp(0.3 * zz)
    return f, -4.0 * f, -2.25 * f, 0.09 * f


def _interior_err(num, ref):
    a = num[1:-1, 1:-1, 1:-1]
    a = a.numpy() if hasattr(a, "numpy") else a
    return float(np.abs(a - ref[1:-1, 1:-1, 1:-1]).max())


@pytest.mark.parametrize(
    "op, which, min_order",
    [(phys.d2_dx2, 1, 0.9),    # nicht aequidistant: formal 1. Ordnung
     (phys.d2_dy2, 2, 1.8),
     (phys.d2_dz2, 3, 1.8)],
)
def test_zweite_ableitungen_konvergieren(op, which, min_order):
    errs = []
    for n in (17, 33):
        layout, xx, yy, zz = _stretched(n)
        vals = _analytic(xx, yy, zz)
        t = torch.as_tensor(vals[0], dtype=torch.float64)
        padded = gridmod.pad_all(t, t[-2])
        errs.append(_interior_err(op(padded, layout), vals[which]))
    order = np.log2(errs[0] / errs[1])
    assert order > min_order, f"Ordnung {order:.2f}, Fehler {errs}"


def test_kreuzterm_trifft_die_analytische_loesung():
    """``lambda_xy`` ist auf JR1 ungleich null -- der Term darf nicht fehlen."""
    layout, xx, yy, _ = _stretched(33)
    f = np.sin(2.0 * xx) * np.cos(1.5 * yy)
    fxy = -3.0 * np.cos(2.0 * xx) * np.sin(1.5 * yy)
    t = torch.as_tensor(f, dtype=torch.float64)
    padded = gridmod.pad_all(t, t[-2])
    rel = _interior_err(phys.d2_dxdy(padded, layout), fxy) / np.abs(fxy).max()
    assert rel < 0.05, f"relativer Fehler {rel:.3g}"


def test_aequidistanter_fall_faellt_auf_den_einfachen_stern_zurueck():
    """Bei a == b muss die nicht-aequidistante Formel identisch werden.

    Sonst haette man sich mit der allgemeinen Form einen Fehler eingehandelt,
    den die einfache nicht hat.
    """
    n = 9
    g = np.linspace(0.0, 1.0, n)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    layout = gridmod.derive_layout(xyz)
    h = layout.dx[0]
    t = torch.as_tensor(np.sin(3.0 * xx) * np.cos(2.0 * yy), dtype=torch.float64)
    padded = gridmod.pad_all(t, t[-2])
    got = phys.d2_dx2(padded, layout)
    p = padded[:, 1:-1, 1:-1]
    naive = (p[:-2] - 2.0 * p[1:-1] + p[2:]) / h ** 2
    assert torch.allclose(got, naive, atol=1e-12)


# ---------------------------------------------------------------------------
# Der anisotrope Operator
# ---------------------------------------------------------------------------
def test_laplacian_ist_isotrop_wenn_fo_es_ist(layout, random_field):
    """Mit Einheits-Fo muss die Summe der drei zweiten Ableitungen herauskommen."""
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    for i in range(3):
        fo[..., i, i] = 1.0
    padded = gridmod.pad_all(random_field, random_field[-2])
    got = phys.anisotropic_laplacian(padded, layout, fo)
    want = (phys.d2_dx2(padded, layout) + phys.d2_dy2(padded, layout)
            + phys.d2_dz2(padded, layout))
    assert torch.allclose(got, want, atol=1e-12)


def test_laplacian_faellt_laut_bei_unbehandelten_komponenten(layout, random_field):
    """``lambda_xz``/``lambda_yz`` sind fuer diesen Datensatz null.

    Waeren sie es nicht, fehlte ein Term. Lieber hier laut als still
    verschluckt -- das ist die Sorte Fehler, die als "das Netz konvergiert
    schlecht" zurueckkommt.
    """
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    fo[..., 0, 2] = 1.0
    padded = gridmod.pad_all(random_field, random_field[-2])
    with pytest.raises(NotImplementedError, match="lambda_xz"):
        phys.anisotropic_laplacian(padded, layout, fo)


def test_konstantes_feld_hat_keine_kruemmung(layout):
    """Ein konstantes Feld darf nirgends diffundieren -- auch nicht am Rand."""
    f = torch.full(layout.shape, 7.5, dtype=torch.float64)
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    for i in range(3):
        fo[..., i, i] = 1.0
    fo[..., 0, 1] = 0.5
    padded = gridmod.pad_all(f, f[-2])
    got = phys.anisotropic_laplacian(padded, layout, fo)
    assert float(got.abs().max()) < 1e-9


def test_ein_gleichmaessiger_versatz_ist_fuer_den_laplace_unsichtbar(
        layout, random_field):
    """Der Diffusionskern sieht keinen Pegelfehler -- er kann ihn also auch
    nicht zurueckholen.

    Am 22.09. hiess es, Arm B sei "die Behandlung" fuer O13, weil der
    dissipative Kern das Leck liefere, das der Delta-Form fehlt. Die Laeufe
    16 und 17 zeigen O13 als **Pegelfehler**: das ganze Feld liegt am Ende
    zu kalt. Ein gleichmaessiger Versatz aendert den Laplace aber nicht --
    auch nicht mit raeumlich wechselndem Fo und Kreuzterm, auch nicht am
    adiabaten Rand, dessen Geisterschicht den Versatz mitspiegelt. Was im
    Physikteil auf den Pegel reagiert, ist allein der Wandterm.
    """
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    fo[..., 0, 0], fo[..., 1, 1], fo[..., 2, 2] = 1.0, 0.7, 0.9
    fo[1] *= 3.0                                  # Schichten wie im Datensatz
    fo[..., 0, 1] = fo[..., 1, 0] = 0.2
    versetzt = random_field + 11.5                # der Endversatz vom 22.09.
    lap = phys.anisotropic_laplacian(
        gridmod.pad_all(random_field, random_field[-2]), layout, fo)
    lap_v = phys.anisotropic_laplacian(
        gridmod.pad_all(versetzt, versetzt[-2]), layout, fo)
    assert torch.allclose(lap, lap_v, atol=1e-9)


# ---------------------------------------------------------------------------
# U(V_dot) -- und die Regel, dass nie extrapoliert wird
# ---------------------------------------------------------------------------
def test_ucurve_interpoliert_zwischen_den_stuetzstellen():
    u = phys.UCurve.from_samples([0.0, 15.0, 30.0], [50.0, 600.0, 1130.0])
    assert u(0.0) == pytest.approx(50.0)
    assert u(30.0) == pytest.approx(1130.0)
    assert 50.0 < u(7.5) < 600.0


def test_ucurve_klemmt_statt_zu_extrapolieren():
    """OP16 faehrt V_dot = 90 und ist ein TEST-OP.

    ``U`` dort zu extrapolieren waere eine Auswahl auf dem Extrapolationstier --
    genau das, wovor ``op_registry.py`` warnt. Geklemmt wird, und der Aufruf
    wird als Extrapolation gemeldet, damit es im Protokoll auftaucht.
    """
    u = phys.UCurve.from_samples([0.0, 15.0, 30.0], [50.0, 600.0, 1130.0])
    assert u(90.0) == pytest.approx(1130.0)
    assert u(-5.0) == pytest.approx(50.0)
    assert u.is_extrapolation(90.0)
    assert not u.is_extrapolation(20.0)


def test_ucurve_sortiert_unsortierte_stuetzstellen():
    a = phys.UCurve.from_samples([30.0, 0.0, 15.0], [1130.0, 50.0, 600.0])
    b = phys.UCurve.from_samples([0.0, 15.0, 30.0], [50.0, 600.0, 1130.0])
    assert a(7.5) == pytest.approx(b(7.5))


# ---------------------------------------------------------------------------
# Der Wandterm
# ---------------------------------------------------------------------------
def _wall(layout, mode="advective", c_fluid=None):
    return phys.WallModel(
        layout, phys.UCurve.from_samples([0.0, 15.0, 30.0], [50.0, 600.0, 1130.0]),
        cp_fluid=3500.0, mode=mode, c_fluid=c_fluid,
        lam_xx_wall=torch.tensor(200.0, dtype=torch.float64))


def test_knotenflaechen_summieren_sich_zur_kalibrierflaeche(layout):
    """``U`` wurde gegen A kalibriert, also muss die Summe wieder A ergeben.

    Sonst traegt der Wandterm systematisch zu viel oder zu wenig Energie ab --
    und zwar genau um den Anteil, den die Randzellen ausmachen.
    """
    w = _wall(layout)
    _, ny, nz = layout.shape
    assert w.node_area * ny * nz == pytest.approx(phys.WALL_AREA_M2)


def test_waermere_wand_gibt_waerme_ab(layout):
    """Vorzeichen: T2 ueber Fluid heisst Fluss nach aussen, Geist unter T2."""
    w = _wall(layout)
    f = torch.full(layout.shape, 60.0, dtype=torch.float64)
    ghost, q, _ = w.ghost(f, t_in=20.0, mdot=0.1, vdot=30.0,
                          state=phys.FluidState(20.0), dt=0.2)
    assert float(q.min()) > 0.0
    assert float(ghost.max()) < 60.0


def test_fluid_erwaermt_sich_entlang_plus_y(layout):
    """Die Enthalpiebilanz laeuft in +y -- bei kleinem y ist es kaelter.

    Damit ist der Wandfluss bei kleinem y groesser: die Wand sieht dort ein
    kaelteres Fluid.
    """
    w = _wall(layout)
    f = torch.full(layout.shape, 60.0, dtype=torch.float64)
    _, q, _ = w.ghost(f, t_in=20.0, mdot=0.02, vdot=15.0,
                      state=phys.FluidState(20.0), dt=0.2)
    row_mean = q.mean(dim=1)
    assert float(row_mean[0]) > float(row_mean[-1])


def test_advektiv_ueberlebt_mdot_null(layout):
    """Kein Fluss darf nicht durch null teilen.

    Die Advektionsform kann diesen Fall physikalisch nicht -- sie setzt
    T_fluid = T_in. Das ist eine untere Schranke, kein Ergebnis; dafuer gibt es
    den Modus 'capacity'.
    """
    w = _wall(layout)
    f = torch.full(layout.shape, 60.0, dtype=torch.float64)
    ghost, q, _ = w.ghost(f, t_in=20.0, mdot=0.0, vdot=0.0,
                          state=phys.FluidState(20.0), dt=0.2)
    assert torch.isfinite(ghost).all()
    assert torch.isfinite(q).all()


def test_capacity_laedt_bei_mdot_null_auf(layout):
    """``mdot = 0`` heisst kein Fluss, nicht kein Fluid.

    Das Kuehlmittel steht im Kanal und nimmt Waerme in seine eigene
    Waermekapazitaet auf -- die Fluidtemperatur muss also steigen.
    """
    w = _wall(layout, mode="capacity", c_fluid=500.0)
    f = torch.full(layout.shape, 60.0, dtype=torch.float64)
    state = phys.FluidState(20.0)
    for _ in range(5):
        _, _, state = w.ghost(f, t_in=20.0, mdot=0.0, vdot=0.0,
                              state=state, dt=0.2)
    assert state.t_fluid > 20.0


def test_capacity_ohne_c_fluid_faellt_laut(layout):
    """``C_fluid`` liegt nicht vor. Raten waere ein freier Parameter."""
    with pytest.raises(ValueError, match="c_fluid"):
        _wall(layout, mode="capacity")


def test_unbekannter_modus_faellt_laut(layout):
    with pytest.raises(ValueError, match="Unbekannter Modus"):
        _wall(layout, mode="magie")


# ---------------------------------------------------------------------------
# Quelle und CFL
# ---------------------------------------------------------------------------
def test_quelle_wirkt_nur_auf_jr1(layout):
    """Die Quelle wirkt NUR auf JR1 -- am 01.09. von der Simulationsseite
    bestaetigt, und bewusst abweichend vom legacy-README."""
    mask = torch.zeros(layout.shape, dtype=torch.float64)
    mask[1] = 1.0
    rho_cp = torch.full(layout.shape, 2.0e6, dtype=torch.float64)
    q = phys.source_term(1.0e5, mask, rho_cp)
    assert float(q[0].abs().max()) == 0.0
    assert float(q[2].abs().max()) == 0.0
    assert float(q[1].min()) > 0.0


def test_cfl_schranke_faellt_mit_groesserem_fo(layout):
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    for i in range(3):
        fo[..., i, i] = 1e-5
    slow = phys.cfl_limit(layout, fo)
    assert phys.cfl_limit(layout, fo * 10.0) < slow
    assert phys.cfl_limit(layout, fo * 0.0) == float("inf")


# ---------------------------------------------------------------------------
# Der exakte Integrator (Schritt 2, PR #51)
# ---------------------------------------------------------------------------
def _diag_fo(layout, xy: float = 0.0):
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    fo[..., 0, 0], fo[..., 1, 1], fo[..., 2, 2] = 3e-3, 2e-3, 2.5e-3
    fo[1] *= 3.0                                   # Schichten wie im Datensatz
    if xy:
        fo[1, ..., 0, 1] = fo[1, ..., 1, 0] = xy
    return fo


def _rk4(layout, fo, t, q, dt, n):
    """Die Referenz: dieselbe Gleichung, klassisches RK4 in n Unterschritten."""
    def f(x):
        b = x.unsqueeze(0)
        return phys.anisotropic_laplacian(
            gridmod.pad_all(b, b[:, -2]), layout, fo)[0] + q
    h = dt / n
    for _ in range(n):
        k1 = f(t)
        k2 = f(t + 0.5 * h * k1)
        k3 = f(t + 0.5 * h * k2)
        k4 = f(t + h * k3)
        t = t + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t


@pytest.mark.parametrize("xy", [0.0, 4e-4])
def test_der_exakte_schritt_trifft_rk4_weit_ueber_der_cfl_schranke(
        layout, random_field, xy):
    """50x ueber der Euler-Schranke -- dort, wo Euler explodiert -- trifft der
    exakte Schritt eine fein aufgeloeste RK4-Rechnung derselben Gleichung."""
    fo = _diag_fo(layout, xy)
    dt = 50.0 * phys.cfl_limit(layout, fo)
    torch.manual_seed(1)
    q = torch.randn(layout.shape, dtype=torch.float64)
    integ = phys.ExpIntegrator(layout, fo, dt)
    got = integ.schritt(random_field.unsqueeze(0).float(),
                        q.unsqueeze(0).float())[0].double()
    ref = _rk4(layout, fo, random_field, q, dt, n=400)
    assert torch.allclose(got, ref, atol=2e-5, rtol=1e-4)


def test_der_exakte_schritt_bleibt_1000x_ueber_der_schranke_beschraenkt(
        layout, random_field):
    """Das Maximumprinzip der Waermeleitung: ohne Quelle waechst kein Wert
    ueber das Anfangsmaximum. Euler waere nach wenigen Schritten bei inf."""
    fo = _diag_fo(layout)
    dt = 1000.0 * phys.cfl_limit(layout, fo)
    integ = phys.ExpIntegrator(layout, fo, dt)
    t = random_field.unsqueeze(0).float()
    null = torch.zeros_like(t)
    oben = float(t.abs().max())
    for _ in range(500):
        t = integ.schritt(t, null)
    assert float(t.abs().max()) <= oben + 1e-5
    # Und Euler mit demselben dt laeuft tatsaechlich weg.
    e = random_field.unsqueeze(0)
    for _ in range(20):
        e = e + dt * phys.anisotropic_laplacian(
            gridmod.pad_all(e, e[:, -2]), layout, fo)
    assert float(e.abs().max()) > 1e6


def test_ein_konstantes_feld_bleibt_ohne_quelle_stehen(layout):
    fo = _diag_fo(layout, 4e-4)
    integ = phys.ExpIntegrator(layout, fo, 100.0 * phys.cfl_limit(layout, fo))
    t = torch.full((2, *layout.shape), 3.25)
    assert torch.allclose(integ.schritt(t, torch.zeros_like(t)), t, atol=1e-5)
    assert abs(integ.groesster_eigenwert()) < 1e-8


def test_die_sekante_ist_die_rate_des_exakten_schritts(layout, random_field):
    """schritt = T + dt * sekante -- die Sekante ist ohne Ausloeschung
    gerechnet, muss aber dieselbe Groesse sein."""
    fo = _diag_fo(layout)
    dt = 20.0 * phys.cfl_limit(layout, fo)
    integ = phys.ExpIntegrator(layout, fo, dt)
    t = random_field.unsqueeze(0)
    q = torch.full_like(t, 0.3)
    integ64 = integ
    integ64.p, integ64.phi = integ.p.double(), integ.phi.double()
    integ64.rate_matrix = integ.rate_matrix.double()
    integ64.quell_matrix = integ.quell_matrix.double()
    assert torch.allclose(integ64.schritt(t, q), t + dt * integ64.sekante(t, q),
                          atol=1e-6)
