"""Der Faltungsstapel in Delta-Form -- die gelernte Korrektur auf der Physik.

Was hier steht
--------------
Die Rate, die ``solve.py`` als reine Physik rechnet, bekommt einen gelernten
Summanden::

    T_{t+1}  =  T_t  +  dt * f( T_t , T_{t-D1} , T_{t-D2} , u_t , S )

    f  =  L(T_t)  +  Qsrc_t  +  g_theta(X_t)
          ^feste Physik         ^gelernte Korrektur, ~11 400 Parameter

``L`` ist ``physics.anisotropic_laplacian`` auf dem gepaddeten Stapel aus
``grid.pad_all`` -- also **exakt derselbe** Operator, den der Loeser ohne Netz
benutzt. Es gibt keine zweite Implementierung; g_theta korrigiert ihn, es
ersetzt ihn nicht.

Die zwei Rollen der x-Achse
---------------------------
Das ist der Punkt, an dem der Entwurf am leichtesten misszuverstehen ist:

======================  ====================================================
im Physik-Term ``L``    x ist eine **echte Achse** mit Geisterschichten.
                        Symmetrie bei x = 0, Kuehlwand bei x = 0.0219, und
                        der nicht-aequidistante Dreipunktstern dazwischen.
in der Korrektur ``g``  x ist in die **Kanaele** gefaltet. g sieht ein
                        11 x 11-Bild mit 44 Kanaelen und gibt 3 Kanaele
                        zurueck -- eine Rate je x-Ebene.
======================  ====================================================

Beides gleichzeitig ist kein Widerspruch: der Stencil braucht die Nachbarn in
x, der Faltungskern nicht. Drei Ebenen sind zu wenig, um darueber zu falten.

Warum die letzte Schicht auf null startet
------------------------------------------
``head`` wird mit Gewicht und Bias **exakt null** initialisiert. Beim ersten
Schritt ist also ``g_theta = 0`` und das Modell rechnet **genau** den Loeser
aus Stufe 3. Das ist kein Trick, sondern die Einloesung des Satzes aus dem
Fahrplan: *f korrigiert den Loeser, es ersetzt ihn nicht.* Ohne diese
Initialisierung startet das Modell bei Rauschen und muss sich die Physik erst
wieder erarbeiten, die es geschenkt bekommen koennte.

Warum reflect und nicht zero padding
-------------------------------------
Jede Faltung padded ``reflect`` in (y, z) -- dieselbe Konvention wie
``grid.pad_yz``. Mit dem Default ``zeros`` saehe der Kern am Rand eine
erfundene Null, also eine Temperatur von ``T_mu`` im z-Score. Das waere eine
Randbedingung, die niemand beschlossen hat, und sie widerspraeche der adiabaten
Annahme aus README Sec. 5 direkt.

Die Groesse
-----------
**16 Kanaele x 3 Bloecke, 11 427 Parameter.** Der Entwurf aus der Praesentation
sah 64 x 4 (~100 k) vor; entschieden wurde die kleinere Variante, weil der
Rangtest vier Moden bei 99.9 % gemessen hat und elf Trajektorien dahinter
stehen. ``ConvCorrection`` nimmt ``width`` und ``blocks`` als Argumente, die
64 x 4 sind also eine Sweep-Achse und kein Umbau -- siehe FAHRPLAN, "Die
Groesse ist eine Entscheidung".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grid as gridmod
import physics as phys
from grid import GridLayout

# Die Kanalaufteilung aus README Sec. 3. Sie steht hier als Konstante, damit ein
# Umbau am Eingang genau eine Stelle hat und nicht drei.
CH_STATE = 9      # T_t (3) + Rate ueber kurzem Lag (3) + ueber langem Lag (3)
CH_STATIC = 17    # lam_xx/yy/zz/xy x 3 Ebenen (12) + rho*Cp x 3 (3) + y/z-Karte (2)
CH_COORD = 2      # davon die zwei Koordinatenkarten -- Ablationsarm D zieht sie
CH_STATIC_OHNE_KOORD = CH_STATIC - CH_COORD   # 15
CH_DRIVER = 18    # 7 config + 11 forcing, gebroadcastet
CH_IN = CH_STATE + CH_STATIC + CH_DRIVER   # 44


# ---------------------------------------------------------------------------
# Die statischen Karten
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StaticMaps:
    """Die 17 ortsfesten Kanaele, einmal je Datensatz gebaut.

    Konstant ueber Zeit **und** ueber alle siebzehn Betriebspunkte -- die
    Materialdaten haengen am Layer, nicht am OP. Einmal bauen, durchreichen.
    """

    maps: torch.Tensor          # (17, ny, nz)
    dead: tuple = ()            # Namen der Groessen, die konstant und damit tot sind

    def __post_init__(self) -> None:
        if self.maps.shape[0] not in (CH_STATIC, CH_STATIC_OHNE_KOORD):
            raise ValueError(
                f"StaticMaps braucht {CH_STATIC} Kanaele (oder "
                f"{CH_STATIC_OHNE_KOORD} ohne die Koordinatenkarten), hat "
                f"{self.maps.shape[0]}")

    @property
    def n_channels(self) -> int:
        return int(self.maps.shape[0])

    @property
    def hat_koordinatenkarten(self) -> bool:
        return self.n_channels == CH_STATIC

    def expand(self, batch: int) -> torch.Tensor:
        return self.maps.unsqueeze(0).expand(batch, -1, -1, -1)

    def describe(self) -> str:
        """Was das Netz an ortsfester Information tatsaechlich bekommt.

        Die tote Liste steht hier, weil sie sonst niemand sieht. ``data.py``
        meldet dasselbe fuer die Treiber (``DEAD -> forced to 0``), und aus
        demselben Grund: ein Kanal, der nichts traegt, ist kein harmloser
        Kanal -- er kostet Parameter in der ersten Faltung und verschleiert,
        wie viel Information wirklich hineingeht.
        """
        lebt = self.n_channels - 3 * len(self.dead)
        zeile = f"{self.n_channels} statische Karten, davon {lebt} mit Struktur"
        if not self.hat_koordinatenkarten:
            zeile += ("\n  ARM D: y/z-Karten GEZOGEN. Ortsstruktur muss aus den "
                      "Materialkarten kommen -- siehe BENCHMARK.md, Route R8.")
        if self.dead:
            zeile += ("\n  TOT (konstant, auf 0 gesetzt): "
                      + ", ".join(self.dead))
        return zeile


def _zscore(a: torch.Tensor) -> torch.Tensor:
    """Gepoolt z-scoren, wie ``data.py`` es fuer alles andere auch tut.

    Roh sind die Waermeleitfaehigkeiten O(1) bis O(100) W/mK und ``rho*Cp``
    O(1e6) J/m^3K. Ungescort wuerde der ``rho*Cp``-Kanal die erste Faltung
    dominieren, und zwar nicht, weil er wichtiger ist, sondern weil seine
    Einheit groesser ist.

    ``unbiased=False`` ist **kein Detail**: ``data.py:468`` scort mit numpys
    ``.std()``, und das ist ddof = 0. Torchs ``.std()`` ist ddof = 1. Beliesse
    man den Default, waeren die Karten hier um ``sqrt(n/(n-1))`` anders skaliert
    als alles, was aus ``data.py`` kommt -- bei n = 363 gerade 0.14 %. Klein
    genug, um nie aufzufallen, und keine Groesse, die man verschieden haben
    will, wenn zwei Projekte gegeneinander gemessen werden.

    ⚠ **Die Falle beim konstanten Kanal.** Die naive Form
    ``(a - a.mean()) / (a.std() + 1e-12)`` gibt fuer ein konstantes ``a``
    NICHT null. In float32 ist ``rho*Cp = 2.25e6`` ueberall, der Mittelwert
    traegt einen Rundungsfehler von O(0.25), und dieser Rest geteilt durch eine
    genauso winzige Streuung ergibt **+-1** -- einen Kanal aus reinem
    Rauschen mit Standardabweichung 1, den das Netz nicht von echter Struktur
    unterscheiden kann. Deshalb wird in float64 gerechnet und der konstante
    Fall ausdruecklich abgefangen: er wird zu einer ehrlichen Null.
    """
    x = a.to(torch.float64)
    mu = x.mean()
    sd = x.std(unbiased=False)
    if float(sd) <= 1e-12 * max(1.0, abs(float(mu))):
        return torch.zeros_like(a)
    return ((x - mu) / sd).to(a.dtype)


def build_static_maps(layout: GridLayout, *, lam: np.ndarray, rho: np.ndarray,
                      cp: np.ndarray, device=None, dtype=torch.float32,
                      coord_maps: bool = True) -> StaticMaps:
    """Baut die 17 Karten aus den Rohgroessen von ``data._grid_arrays``.

    ``lam`` ist ``(n_points, 3, 3)``, ``rho`` und ``cp`` sind ``(n_points,)``.

    **Nicht** aus ``OPData.static_feat`` gebaut: das sind nur drei Kanaele
    (``alpha_z``, ``jr1``, ``x_z``) und damit eine andere Auswahl als die, die
    README Sec. 3c fuer den CNN trifft. ``alpha`` faellt hier weg, weil es aus
    ``lam`` und ``rho*Cp`` ableitbar ist, und ``region``/``q_mask`` fallen weg,
    weil sie mit dem x-Ebenenindex identisch sind -- den traegt schon die
    Kanalaufteilung. Ein redundanter Kanal ist bei elf Trajektorien kein
    harmloser Kanal.
    """
    nx, ny, nz = layout.shape

    def to_planes(flat: np.ndarray) -> torch.Tensor:
        """(n_points,) -> (nx, ny, nz), x wird spaeter zu Kanaelen."""
        t = torch.as_tensor(np.asarray(flat), dtype=dtype, device=device)
        return gridmod.to_field(t, layout)

    lam = np.asarray(lam, dtype=np.float64)
    planes, dead = [], []
    # Vier Tensorkomponenten. XZ und YZ sind fuer diesen Datensatz null
    # (legacy-README) und waeren tote Kanaele; XY ist genau auf JR1 ungleich
    # null und traegt deshalb echte Information.
    roh = [(f"lam_{a}{b}", lam[:, i, j]) for (a, b), (i, j) in
           zip(("xx", "yy", "zz", "xy"), ((0, 0), (1, 1), (2, 2), (0, 1)))]
    roh.append(("rho*Cp", np.asarray(rho) * np.asarray(cp)))
    for name, werte in roh:
        feld = to_planes(werte)
        gescort = _zscore(feld)
        if not gescort.any():
            dead.append(name)
        planes.append(gescort)

    # (n_komponenten, nx, ny, nz) -> x in die Kanalachse falten: 5 * 3 = 15
    stacked = torch.stack(planes, dim=0).reshape(-1, ny, nz)

    # Die zwei Koordinatenkarten. Sie sind der Grund, warum der Kern ueberhaupt
    # etwas ueber Position wissen kann: die Faltung ist ueber (y, z) geteilt,
    # also KANN sie Ort nicht auswendig lernen. Ohne diese beiden waere jede
    # Randzelle von jeder Mittelzelle ununterscheidbar.
    #
    # ⚠ Und genau darin widerspricht diese Datei dem README, Sec. 2b, das den
    # geteilten Kern als VORTEIL fuehrt ("der CNN muss raeumliche Struktur ueber
    # die Materialkarten begruenden"). Beide Texte gehen von derselben Praemisse
    # aus und ziehen den entgegengesetzten Schluss; keiner von beiden ist
    # gemessen. ``coord_maps=False`` ist **Ablationsarm D** und macht die Frage
    # entscheidbar: derselbe Verlust, dieselbe Architektur sonst, zwei Kanaele
    # weniger. Route R8 in BENCHMARK.md.
    #
    # Am 22.09. dazu gemessen, was die Materialkarten in der Ebene ueberhaupt
    # hergeben: region/rho/Cp sind je x-Ebene KONSTANT, und ``lam`` variiert nur
    # in zwei Zeilen am unteren y-Rand (22 von 121 Punkten). D ist damit ein
    # fairer Test und kein Strohmann -- aber ein enger.
    if not coord_maps:
        return StaticMaps(maps=stacked.contiguous(), dead=tuple(dead))

    yv = torch.as_tensor(layout.yu, dtype=dtype, device=device)
    zv = torch.as_tensor(layout.zu, dtype=dtype, device=device)
    y_map = _zscore(yv).reshape(ny, 1).expand(ny, nz)
    z_map = _zscore(zv).reshape(1, nz).expand(ny, nz)

    maps = torch.cat([stacked, y_map.unsqueeze(0), z_map.unsqueeze(0)], dim=0)
    return StaticMaps(maps=maps.contiguous(), dead=tuple(dead))


# ---------------------------------------------------------------------------
# Der Eingang
# ---------------------------------------------------------------------------
def state_channels(tn_now: torch.Tensor, tn_lag1: torch.Tensor,
                   tn_lag2: torch.Tensor) -> torch.Tensor:
    """Die 9 Zustands- und Historienkanaele.

    Alle drei Argumente sind ``(B, nx, ny, nz)``. Zurueck kommt
    ``(B, 9, ny, nz)``.

    Gebaut wird die **Hybrid-Historie aus PINNmodulusTwo/model.py**: ein Anker
    plus eine Rate je Lag, nur feldweise statt punktweise. Absicht -- damit der
    Vergleich der beiden Modelle eine Architekturfrage bleibt und keine
    Featurefrage wird.

    Die Raten sind Differenzen, keine durch dt geteilten Ableitungen: dt ist
    ueber die Trajektorie konstant, der Faktor waere also eine Konstante, die
    die erste Faltung ohnehin absorbiert.
    """
    if tn_now.dim() != 4:
        raise ValueError(f"(B, nx, ny, nz) erwartet, bekam {tuple(tn_now.shape)}")
    return torch.cat(
        [tn_now, tn_now - tn_lag1, tn_now - tn_lag2], dim=1)


def driver_channels(config_feat: torch.Tensor, forcing_feat: torch.Tensor,
                    ny: int, nz: int) -> torch.Tensor:
    """Die 18 globalen Skalare, auf das Gitter gebroadcastet.

    ``config_feat`` ist ``(B, 7)``, ``forcing_feat`` ist ``(B, 11)``; zurueck
    kommt ``(B, 18, ny, nz)``.

    Broadcast und nicht FiLM: bei 121 Pixeln kostet das nichts, und FiLM ist
    eine eigene Sweep-Achse (README Sec. 3d) -- keine Architekturentscheidung,
    die man nebenbei mit hineinnimmt.

    ⚠ Drei dieser Kanaele tragen weniger, als ihre Zahl verspricht:
    ``soc_start`` ist ueber alle sechzehn OPs konstant (O5, ``DEAD -> 0``), und
    ``solid_initial_temp`` / ``fluid_initial_temp`` sind im Training perfekt
    konfundiert, weil in 11 von 11 OPs ``T0 = T_fluid`` gilt. Sie werden
    trotzdem uebergeben, damit die Kanalbreite ueber alle OPs gleich ist --
    aber eine gelernte Abhaengigkeit von ihnen ist nicht interpretierbar.
    Nachzulesen in README_OPS.md.
    """
    drivers = torch.cat([config_feat, forcing_feat], dim=1)   # (B, 18)
    if drivers.shape[1] != CH_DRIVER:
        raise ValueError(
            f"{CH_DRIVER} Treiber erwartet, bekam {drivers.shape[1]} "
            f"({config_feat.shape[1]} config + {forcing_feat.shape[1]} forcing)")
    return drivers[:, :, None, None].expand(-1, -1, ny, nz)


def assemble_input(state: torch.Tensor, static: StaticMaps,
                   drivers: torch.Tensor) -> torch.Tensor:
    """Die Kanaele in der Reihenfolge aus README Sec. 3: Zustand, Karten, Treiber.

    44 mit den Koordinatenkarten, 42 ohne (Ablationsarm D). Die Breite kommt aus
    ``static``, nicht aus :data:`CH_IN` -- sonst muesste man sie an zwei Stellen
    gleichzeitig aendern, und die zweite wuerde vergessen.
    """
    b = state.shape[0]
    x = torch.cat([state, static.expand(b), drivers], dim=1)
    erwartet = CH_STATE + static.n_channels + CH_DRIVER
    if x.shape[1] != erwartet:
        raise ValueError(
            f"{erwartet} Kanaele erwartet ({CH_STATE} Zustand + "
            f"{static.n_channels} statisch + {CH_DRIVER} Treiber), gebaut "
            f"wurden {x.shape[1]}")
    return x


# ---------------------------------------------------------------------------
# Die gelernte Korrektur
# ---------------------------------------------------------------------------
class ConvCorrection(nn.Module):
    """``g_theta``: 44 Kanaele -> 3 Kanaele, 3x3 ueber (y, z).

    Voreingestellt ``width=16``, ``blocks=3`` -> **11 427 Parameter**. Die
    Praesentation sah ``width=64``, ``blocks=4`` vor (~100 k); beide sind ueber
    dieselben zwei Argumente erreichbar, damit die Groesse eine Sweep-Achse ist
    und kein Umbau.

    Aktivierung ist ``SiLU`` -- dieselbe Familie wie der ``LearnableSwish`` in
    ``PINNmodulusTwo/model.py``, aber ohne das lernbare ``beta``. Das beta ist
    ein Skalar je Schicht und haette den Vergleich der beiden Modelle um eine
    Kleinigkeit erweitert, die mit der Architekturfrage nichts zu tun hat.
    """

    def __init__(self, in_ch: int = CH_IN, width: int = 16, blocks: int = 3,
                 out_ch: int = 3) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("mindestens ein Block")
        chans = [in_ch] + [width] * blocks
        self.body = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, padding=0)
            for i in range(blocks))
        self.act = nn.SiLU()
        self.head = nn.Conv2d(width, out_ch, kernel_size=3, padding=0)

        # Start exakt auf der Physik: g_theta(X) = 0 fuer jeden Eingang.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 44, ny, nz)`` -> ``(B, 3, ny, nz)``.

        Gepadded wird vor **jeder** Faltung einzeln, nicht einmal breit am
        Anfang: ein 3x3-Kern verbraucht je Schicht einen Ring, und ein einmal
        angelegter Rand waere nach der zweiten Schicht aufgebraucht.
        """
        for conv in self.body:
            x = self.act(conv(F.pad(x, (1, 1, 1, 1), mode="reflect")))
        return self.head(F.pad(x, (1, 1, 1, 1), mode="reflect"))


# ---------------------------------------------------------------------------
# Das Modell
# ---------------------------------------------------------------------------
class GridCNN(nn.Module):
    """Die Delta-Form: Physik plus gelernte Korrektur.

    ``use_physics`` schaltet den Term ``L(T) + Qsrc`` ab und laesst nur
    ``g_theta`` stehen. Das ist **Konfiguration A** aus der Ablation im
    Fahrplan -- die reine Blackbox, gegen die B und C gemessen werden. Es ist
    kein Debug-Schalter: ohne A ist nicht zu sagen, ob die Physik in der
    Architektur etwas beitraegt oder ob das Netz sie ohnehin gelernt haette.

    ``n_static`` ist :data:`CH_STATIC_OHNE_KOORD` fuer **Konfiguration D** --
    B ohne die zwei Koordinatenkarten. Es muss zu den ``StaticMaps`` passen, mit
    denen das Netz gefuettert wird; passt es nicht, faellt schon die erste
    Faltung, und ``assemble_input`` faellt davor mit einer Zahl im Text.
    """

    def __init__(self, layout: GridLayout, *, width: int = 16, blocks: int = 3,
                 use_physics: bool = True, n_static: int = CH_STATIC) -> None:
        super().__init__()
        if n_static not in (CH_STATIC, CH_STATIC_OHNE_KOORD):
            raise ValueError(
                f"n_static ist {CH_STATIC} oder {CH_STATIC_OHNE_KOORD} "
                f"(Ablationsarm D), nicht {n_static}")
        self.layout = layout
        self.use_physics = use_physics
        self.n_static = n_static
        self.correction = ConvCorrection(
            in_ch=CH_STATE + n_static + CH_DRIVER,
            width=width, blocks=blocks, out_ch=layout.shape[0])

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rate(self, tn_field: torch.Tensor, x_channels: torch.Tensor, *,
             fo_field: torch.Tensor, qsrc: torch.Tensor,
             ghost_hi: torch.Tensor) -> torch.Tensor:
        """``f`` -- die Aenderungsrate, ``(B, nx, ny, nz)``.

        ``ghost_hi`` ist ``(B, ny, nz)`` und kommt von aussen: entweder aus
        ``physics.WallModel`` oder, solange Stufe 2 offen ist, als Spiegelung
        (adiabat). Der Wandterm gehoert nicht in diese Klasse -- er ist Physik
        und steht in ``physics.py``.
        """
        g = self.correction(x_channels)
        if not self.use_physics:
            return g
        padded = gridmod.pad_all(tn_field, ghost_hi)
        lap = phys.anisotropic_laplacian(padded, self.layout, fo_field)
        return lap + qsrc + g

    def step(self, tn_field: torch.Tensor, x_channels: torch.Tensor, *,
             dt_n: float, **kw) -> torch.Tensor:
        """Ein Euler-Schritt in Delta-Form: ``T + dt * f``.

        Die Delta-Form ist eine **Hypothese**, keine Messung (README Sec. 4):
        ``residual_output`` ist in ``PINNmodulusTwo`` aus, weil es den Level
        ohne Leck traegt und jeden Rollout weglaufen laesst. Hier sollte der
        dissipative Diffusionskern dieses Leck liefern. Laeuft der Rollout
        trotzdem weg, war die Hypothese falsch -- und das ist billig zu sehen,
        nicht teuer zu uebertuenchen.
        """
        return tn_field + dt_n * self.rate(tn_field, x_channels, **kw)

    def forward(self, tn_field: torch.Tensor, x_channels: torch.Tensor, *,
                dt_n: float, **kw) -> torch.Tensor:
        return self.step(tn_field, x_channels, dt_n=dt_n, **kw)


def adiabatic_ghost(tn_field: torch.Tensor) -> torch.Tensor:
    """Geisterschicht fuer eine adiabate Gehaeusewand: Spiegelung wie bei x = 0.

    Der Platzhalter, solange Stufe 2 offen ist und ``U(V_dot)`` nicht
    kalibriert werden kann. **Das ist eine Ablation, keine Physik** -- dieselbe
    Notiz, die ``solve.rollout`` mitfuehrt, gilt hier unveraendert.
    """
    return tn_field[:, -2]
