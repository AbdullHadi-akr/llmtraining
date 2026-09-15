"""Tests fuer den Faltungsstapel.

Der wichtigste Test ist ``test_bei_init_rechnet_das_modell_exakt_die_physik``.
Alles andere hier sind Formen und Randfaelle; jener eine haelt die Zusage fest,
auf der die ganze Delta-Form steht: **f korrigiert den Loeser, es ersetzt ihn
nicht.** Geht er kaputt, startet das Training bei Rauschen statt bei Physik --
und das waere aus einer Trainingskurve heraus nicht zu erkennen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grid as gridmod
import model as M
import physics as phys


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------
@pytest.fixture
def fo_field(layout):
    """Ein Fourier-Tensor mit ``lam_xy != 0`` genau auf JR1, wie im Datensatz."""
    nx, ny, nz = layout.shape
    fo = torch.zeros(nx, ny, nz, 3, 3, dtype=torch.float32)
    fo[..., 0, 0], fo[..., 1, 1], fo[..., 2, 2] = 3e-3, 2e-3, 2.5e-3
    fo[1, ..., 0, 1] = 4e-4
    return fo


@pytest.fixture
def statics(layout):
    rng = np.random.default_rng(0)
    n = layout.n_points
    return M.build_static_maps(
        layout, lam=rng.random((n, 3, 3)) + 1.0,
        rho=np.full(n, 2500.0), cp=np.full(n, 900.0))


@pytest.fixture
def batch(layout, statics):
    """Ein vollstaendiger Eingang fuer vier Betriebspunkte."""
    torch.manual_seed(0)
    nx, ny, nz = layout.shape
    b = 4
    t0, t1, t2 = (torch.randn(b, nx, ny, nz) for _ in range(3))
    drivers = M.driver_channels(torch.randn(b, 7), torch.randn(b, 11), ny, nz)
    x = M.assemble_input(M.state_channels(t0, t1, t2), statics, drivers)
    qsrc = torch.zeros(b, nx, ny, nz)
    qsrc[:, 1] = 0.05                      # die Quelle wirkt nur auf JR1
    return t0, x, qsrc


# ---------------------------------------------------------------------------
# Die Groesse
# ---------------------------------------------------------------------------
def test_die_voreinstellung_trifft_die_11400_aus_dem_fahrplan(layout):
    """44 -> 16 -> 16 -> 16 -> 3, alles 3x3. Der Fahrplan nennt ~11 400."""
    assert M.GridCNN(layout).n_parameters == 11_427


def test_die_groesse_aus_der_praesentation_ist_eine_sweep_achse(layout):
    """64 x 4 muss ueber Argumente erreichbar sein, nicht ueber einen Umbau.

    ⚠ **137 923, nicht "~100 k".** README Sec. 11.4 und der Fahrplan nennen fuer
    diesen Entwurf ~100 k Parameter; nachgerechnet sind es 137 923, also 38 %
    mehr. Die Schaetzung hat vermutlich den Sprung 44 -> 64 in der ersten
    Schicht und die vierte 64x64-Faltung zu klein angesetzt.

    Das aendert an der Entscheidung fuer 16 x 3 nichts -- es macht sie nur
    deutlicher: das Verhaeltnis zu elf Trajektorien ist noch ungleicher, als
    die Doku behauptet hat.
    """
    big = M.GridCNN(layout, width=64, blocks=4)
    assert big.n_parameters == 137_923
    # und sie darf nichts anderes an der Schnittstelle aendern
    assert big.correction.head.out_channels == layout.shape[0]


def test_die_gegenprobe_aus_route_R5_ist_erreichbar(layout):
    """24 x 3 -- die Gegenprobe, die PR #37 neben die 16 x 3 gestellt hatte."""
    assert M.GridCNN(layout, width=24, blocks=3).n_parameters == 20_595


def test_die_kanalaufteilung_ergibt_44(layout, statics, batch):
    assert M.CH_STATE + M.CH_STATIC + M.CH_DRIVER == M.CH_IN == 44
    _, x, _ = batch
    assert x.shape[1] == 44
    assert statics.maps.shape == (17, *layout.shape[1:])


# ---------------------------------------------------------------------------
# Die Zusage, auf der alles steht
# ---------------------------------------------------------------------------
def test_bei_init_rechnet_das_modell_exakt_die_physik(layout, fo_field, batch):
    """Ohne Toleranz: ``head`` ist null initialisiert, also ist ``g_theta`` null.

    Die Rate muss damit **Bit fuer Bit** der Rate aus ``solve.py`` entsprechen.
    Alles ausser exakt 0.0 heisst, dass das Modell bei Rauschen startet statt
    bei der Physik -- und dann ist die Delta-Form nicht das, was der Fahrplan
    beschreibt.
    """
    t0, x, qsrc = batch
    m = M.GridCNN(layout)
    ghost = M.adiabatic_ghost(t0)
    rate = m.rate(t0, x, fo_field=fo_field, qsrc=qsrc, ghost_hi=ghost)

    padded = gridmod.pad_all(t0, ghost)
    physik = phys.anisotropic_laplacian(padded, layout, fo_field) + qsrc
    assert torch.equal(rate, physik)


def test_nach_dem_lernen_weicht_die_rate_ab(layout, fo_field, batch):
    """Die Gegenprobe: waere ``g_theta`` gar nicht angeschlossen, bliebe es null."""
    t0, x, qsrc = batch
    m = M.GridCNN(layout)
    with torch.no_grad():
        m.correction.head.weight.normal_(0.0, 0.1)
    ghost = M.adiabatic_ghost(t0)
    padded = gridmod.pad_all(t0, ghost)
    physik = phys.anisotropic_laplacian(padded, layout, fo_field) + qsrc
    rate = m.rate(t0, x, fo_field=fo_field, qsrc=qsrc, ghost_hi=ghost)
    assert (rate - physik).abs().max() > 1e-6


def test_konfiguration_A_laesst_die_physik_weg(layout, fo_field, batch):
    """Konfiguration A der Ablation: reine Blackbox, ``f = g_theta``."""
    t0, x, qsrc = batch
    a = M.GridCNN(layout, use_physics=False)
    with torch.no_grad():
        a.correction.head.weight.normal_(0.0, 0.1)
    rate = a.rate(t0, x, fo_field=fo_field, qsrc=qsrc,
                  ghost_hi=M.adiabatic_ghost(t0))
    assert torch.equal(rate, a.correction(x))


def test_konfiguration_A_startet_bei_persistenz_nicht_bei_rauschen(layout, fo_field, batch):
    """Auch die Blackbox startet auf etwas Sinnvollem.

    ``head`` ist null, also ist die Rate null und ``T_{t+1} = T_t``. Das ist
    ``persistence`` -- einer der trivialen Vorhersager, gegen die ohnehin
    gemessen wird. A startet damit bei Persistenz und B bei der Physik; keine
    der beiden startet bei Rauschen, und das haelt den Vergleich fair.
    """
    t0, x, qsrc = batch
    a = M.GridCNN(layout, use_physics=False)
    nxt = a.step(t0, x, dt_n=0.01, fo_field=fo_field, qsrc=qsrc,
                 ghost_hi=M.adiabatic_ghost(t0))
    assert torch.equal(nxt, t0)


# ---------------------------------------------------------------------------
# Das Padding der Faltung
# ---------------------------------------------------------------------------
def test_reflect_padding_erhaelt_ein_konstantes_feld(layout, statics):
    """Der Test, der ``zeros`` als Padding auffliegen liesse.

    Ein ueber (y, z) konstanter Eingang muss einen ueber (y, z) konstanten
    Ausgang geben: ``reflect`` spiegelt denselben Wert in den Rand, der Kern
    sieht also ueberall dieselbe Nachbarschaft. Mit dem Default ``zeros`` saehe
    er am Rand eine erfundene Null -- im z-Score eine Temperatur von ``T_mu``,
    also eine Randbedingung, die niemand beschlossen hat.
    """
    torch.manual_seed(0)
    g = M.ConvCorrection()
    with torch.no_grad():
        g.head.weight.normal_(0.0, 0.5)
        g.head.bias.normal_(0.0, 0.5)

    ny, nz = layout.shape[1:]
    x = torch.randn(1, M.CH_IN, 1, 1).expand(1, M.CH_IN, ny, nz).contiguous()
    out = g(x)
    spanne = (out.amax(dim=(2, 3)) - out.amin(dim=(2, 3))).max()
    assert spanne < 1e-5, f"Rand weicht ab: {spanne} -- padding ist nicht reflect"


def test_mit_zero_padding_waere_der_rand_anders(layout):
    """Gegenprobe zum Test darueber: der Unterschied ist messbar, nicht kosmetisch."""
    torch.manual_seed(0)
    ny, nz = layout.shape[1:]
    conv = torch.nn.Conv2d(M.CH_IN, 4, 3, padding=0)
    x = torch.randn(1, M.CH_IN, 1, 1).expand(1, M.CH_IN, ny, nz).contiguous()
    mit_reflect = conv(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"))
    mit_zeros = conv(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant"))
    assert (mit_reflect - mit_zeros).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# Die Kanaele
# ---------------------------------------------------------------------------
def test_zustandskanaele_sind_anker_plus_zwei_raten(layout):
    nx, ny, nz = layout.shape
    t0 = torch.randn(2, nx, ny, nz)
    t1 = torch.randn(2, nx, ny, nz)
    t2 = torch.randn(2, nx, ny, nz)
    s = M.state_channels(t0, t1, t2)
    assert s.shape == (2, 9, ny, nz)
    assert torch.equal(s[:, 0:3], t0)
    assert torch.equal(s[:, 3:6], t0 - t1)
    assert torch.equal(s[:, 6:9], t0 - t2)


def test_statische_karten_sind_je_groesse_zgescort_nicht_je_ebene(statics):
    """Roh sind lambda O(10) und rho*Cp O(1e6) -- ungescort dominierte rho*Cp.

    Gescort wird **je Materialgroesse ueber alle drei x-Ebenen zusammen**, nicht
    je Kanal. Der Unterschied ist nicht kosmetisch: scorte man jede Ebene
    einzeln, saehen Cell Center, JR1 und Gehaeuse hinterher identisch verteilt
    aus -- und genau der Kontrast zwischen ihnen ist die Information, deretwegen
    die Karten ueberhaupt Kanaele bekommen.
    """
    # Die ersten 15 Kanaele sind 5 Groessen x 3 Ebenen, in dieser Reihenfolge.
    # rho*Cp ist in dieser Fixture konstant und damit tot -- siehe eigener Test.
    namen = ("lam_xx", "lam_yy", "lam_zz", "lam_xy", "rho*Cp")
    for g, name in enumerate(namen):
        gruppe = statics.maps[g * 3:(g + 1) * 3]
        if name in statics.dead:
            continue
        assert abs(float(gruppe.mean())) < 1e-4, f"{name} nicht zentriert"
        assert abs(float(gruppe.std(unbiased=False)) - 1.0) < 1e-3, name

    # Die zwei Koordinatenkarten haengen nicht von x ab und stehen fuer sich.
    for i in (15, 16):
        assert abs(float(statics.maps[i].mean())) < 1e-4


def test_ein_konstanter_kanal_wird_null_und_nicht_plusminus_eins(layout):
    """Die Falle, die float32 hier stellt -- und der Grund fuer den Guard.

    ``(a - a.mean()) / (a.std() + 1e-12)`` gibt fuer konstantes ``a`` in float32
    **nicht** null: ``rho*Cp = 2.25e6`` traegt im Mittelwert einen
    Rundungsfehler von O(0.25), und dieser Rest geteilt durch eine genauso
    winzige Streuung ergibt +-1. Das Netz bekaeme einen Kanal aus reinem
    Rauschen mit Standardabweichung 1 -- ununterscheidbar von echter Struktur.
    """
    n = layout.n_points
    rng = np.random.default_rng(1)
    st = M.build_static_maps(layout, lam=rng.random((n, 3, 3)) + 1.0,
                             rho=np.full(n, 2500.0), cp=np.full(n, 900.0))
    rho_cp = st.maps[12:15]
    assert float(rho_cp.abs().max()) == 0.0, "konstanter Kanal ist nicht null"
    assert "rho*Cp" in st.dead, "toter Kanal wird nicht gemeldet"
    assert "rho*Cp" in st.describe()


def test_ein_variabler_kanal_gilt_nicht_als_tot(layout):
    n = layout.n_points
    rng = np.random.default_rng(2)
    rho = np.where(np.arange(n) % 3 == 1, 2800.0, 2500.0)
    st = M.build_static_maps(layout, lam=rng.random((n, 3, 3)) + 1.0,
                             rho=rho, cp=np.full(n, 900.0))
    assert st.dead == ()
    assert float(st.maps[12:15].abs().max()) > 0.0


def test_der_kontrast_zwischen_den_x_ebenen_ueberlebt_das_scoren(layout):
    """Wenn JR1 ein anderes lambda hat als das Gehaeuse, muss man das noch sehen."""
    n = layout.n_points
    lam = np.zeros((n, 3, 3))
    region = gridmod.to_field(
        torch.arange(n, dtype=torch.float64), layout)          # Punktindex
    # lambda_xx je x-Ebene verschieden setzen: 1, 10, 100
    lam_xx = np.zeros(n)
    for plane, wert in enumerate((1.0, 10.0, 100.0)):
        lam_xx[layout.flat_index[plane].ravel()] = wert
    lam[:, 0, 0] = lam_xx
    st = M.build_static_maps(layout, lam=lam, rho=np.full(n, 2500.0),
                             cp=np.full(n, 900.0))
    ebenen_mittel = [float(st.maps[i].mean()) for i in range(3)]
    assert ebenen_mittel[0] < ebenen_mittel[1] < ebenen_mittel[2]
    assert ebenen_mittel[2] - ebenen_mittel[0] > 0.5


def test_treiber_werden_gebroadcastet_nicht_gekachelt(layout):
    ny, nz = layout.shape[1:]
    cfg, frc = torch.randn(3, 7), torch.randn(3, 11)
    d = M.driver_channels(cfg, frc, ny, nz)
    assert d.shape == (3, 18, ny, nz)
    # jeder Kanal ist ueber das Gitter konstant
    assert torch.equal(d[..., 0, 0], torch.cat([cfg, frc], dim=1))
    assert float((d.amax(dim=(2, 3)) - d.amin(dim=(2, 3))).abs().max()) == 0.0


def test_falsche_treiberbreite_faellt_laut_aus(layout):
    ny, nz = layout.shape[1:]
    with pytest.raises(ValueError, match="Treiber"):
        M.driver_channels(torch.randn(2, 6), torch.randn(2, 11), ny, nz)


def test_statische_karten_mit_falscher_kanalzahl_faellt_laut_aus(layout):
    with pytest.raises(ValueError, match="17"):
        M.StaticMaps(maps=torch.zeros(16, 11, 11))


def test_state_channels_braucht_eine_batchachse(layout):
    nx, ny, nz = layout.shape
    with pytest.raises(ValueError, match="B, nx, ny, nz"):
        M.state_channels(*(torch.randn(nx, ny, nz) for _ in range(3)))


# ---------------------------------------------------------------------------
# Delta-Form und Gradient
# ---------------------------------------------------------------------------
def test_step_ist_T_plus_dt_mal_rate(layout, fo_field, batch):
    t0, x, qsrc = batch
    m = M.GridCNN(layout)
    with torch.no_grad():
        m.correction.head.weight.normal_(0.0, 0.1)
    kw = dict(fo_field=fo_field, qsrc=qsrc, ghost_hi=M.adiabatic_ghost(t0))
    assert torch.allclose(m.step(t0, x, dt_n=0.2, **kw),
                          t0 + 0.2 * m.rate(t0, x, **kw))


def test_der_gradient_erreicht_jedes_gewicht(layout, fo_field, batch):
    """Eine tote Schicht faellt sonst erst nach Stunden Training auf."""
    t0, x, qsrc = batch
    m = M.GridCNN(layout)
    out = m.step(t0, x, dt_n=0.2, fo_field=fo_field, qsrc=qsrc,
                 ghost_hi=M.adiabatic_ghost(t0))
    # quadratisch, damit auch die null-initialisierte head-Schicht Gradient sieht
    (out ** 2).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} hat keinen Gradienten"
        assert torch.isfinite(p.grad).all(), f"{name} hat nicht-endlichen Gradienten"


def test_adiabate_geisterschicht_ist_die_spiegelung(layout):
    nx, ny, nz = layout.shape
    t = torch.randn(2, nx, ny, nz)
    assert torch.equal(M.adiabatic_ghost(t), t[:, -2])


def test_die_wand_veraendert_die_rate(layout, fo_field, batch):
    """Die Geisterschicht ist kein Zierrat: eine andere Wand, eine andere Rate."""
    t0, x, qsrc = batch
    m = M.GridCNN(layout)
    kw = dict(fo_field=fo_field, qsrc=qsrc)
    kalt = m.rate(t0, x, ghost_hi=M.adiabatic_ghost(t0) - 1.0, **kw)
    warm = m.rate(t0, x, ghost_hi=M.adiabatic_ghost(t0) + 1.0, **kw)
    assert (kalt - warm).abs().max() > 1e-6
    # und nur die wandnaechste Ebene sieht den Unterschied direkt
    assert (kalt[:, 0] - warm[:, 0]).abs().max() == 0.0
