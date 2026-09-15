"""Der Differenzenstern, die Quelle und der Wandterm -- ohne ein Gewicht.

Was hier steht und was nicht
----------------------------
Diese Datei rechnet die rechte Seite der Waermeleitungsgleichung auf dem
gepaddeten ``3 x 11 x 11``-Stapel::

    dT/dt  =  Fo : grad^2 T  +  Qsrc

Kein gelerntes Gewicht, kein Autograd-Hessian. ``solve.py`` rollt das zu einem
Nullmodell, und ``model.py`` wird spaeter genau dieselben Operatoren benutzen
und nur eine Korrektur dazulernen. Zwei Implementierungen desselben Sterns
wuerden auseinanderdriften.

Die drei Stellen, an denen man es falsch machen kann
----------------------------------------------------
1. **Der x-Abstand ist nicht aequidistant.** dx1 = 10.786 mm (Mitte -> JR1),
   dx2 = 11.114 mm (JR1 -> Wand), 3 % Unterschied. Die naive Form
   ``(T0 - 2*T1 + T2)/dx^2`` mischt auf Ebene 1 einen Erste-Ableitungs-Anteil
   von ~3 % ein. Deshalb rechnet ``d2_dx2`` die nicht-aequidistante Formel.
   Auf Ebene 0 und 2 ist der Stern per Konstruktion wieder uniform, weil der
   jeweilige Geist im selben Abstand sitzt.

2. **lambda_xy ist auf JR1 nicht null.** Der legacy-README fuehrt fuer JR1
   ``XX/XY/YY`` aus CSV (nur ``XZ = YZ = 0``), und ``data.py`` kontrahiert
   entsprechend voll. ``T_xy`` ist damit eine gemischte Ableitung ueber die
   kurze x-Achse und eine der beiden langen Achsen -- die eine Stelle, an der die
   FD-Variante fummeliger ist als der Autograd-Hessian, den sie ersetzt. Sie
   wird hier als ``d/dy(dT/dx)`` gebildet, nicht als handgeschriebener
   Kreuzstern: zwei nacheinander angewandte lineare Operatoren sind leichter
   zu pruefen als ein Stern mit acht Koeffizienten.

3. **Bei mdot = 0 hat die Advektionsform eine Polstelle.** ``T_fluid =
   T_in + Q/(mdot*Cp)`` divergiert. Das ist kein Randfall, sondern genau das
   Regime der drei schwierigsten Betriebspunkte (OP06, OP07, OP14). Deshalb
   kennt ``WallModel`` zwei Modi -- siehe dort.

Eine bewusste Vereinfachung, die zum PINN passt
-----------------------------------------------
Gerechnet wird ``Fo : grad^2 T``, die **nicht-konservative** Form: der
Fourier-Tensor wird beim Ableiten als ortskonstant behandelt. Streng richtig
waere ``div(lambda grad T)``, was sich dort unterscheidet, wo ``lambda``
springt -- also an den Grenzen zwischen Cell Center, JR1 und Gehaeuse.

Das ist **Absicht**: ``PINNmodulusTwo/physics.py:189`` rechnet dieselbe Form.
Waehlte man hier die konservative, waere ein Vergleich der beiden Modelle keine
Architekturaussage mehr, sondern eine Diskretisierungsaussage. Wenn die Form
geaendert wird, dann in beiden Projekten gleichzeitig und als eigene Achse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from grid import GridLayout

# Die gekuehlte Flaeche der +x-Gehaeusewand, EINE Seite. 0.198 x 0.104 m laut
# legacy-README, am Gitter zu 0.0207 nachgemessen. Sec. 6 des README: zaehlt der
# Waermestrom-Monitor beide Kuehlplatten, dann ENTWEDER Q halbieren ODER A
# verdoppeln -- nie beides. Entschieden wird das von tools/balance_check.py.
WALL_AREA_M2 = 0.0206


# ---------------------------------------------------------------------------
# Ableitungen auf dem gepaddeten Stapel
# ---------------------------------------------------------------------------
def _pad_spacings(dx: np.ndarray) -> np.ndarray:
    """Abstaende im gepaddeten x-Stapel.

    Der Geist bei ``x = 0`` ist die Spiegelung des ersten inneren Knotens,
    sitzt also im Abstand ``dx[0]``; der Geist hinter der Wand sitzt per
    Konstruktion im Abstand ``dx[-1]``. Damit::

        [dx[0]] + dx + [dx[-1]]
    """
    return np.concatenate([dx[:1], dx, dx[-1:]])


def _as_tensor(a: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(a, dtype=ref.dtype, device=ref.device)


def d2_dx2(padded: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """Zweite Ableitung in x auf dem gepaddeten Stapel, nicht-aequidistant.

    ``padded`` ist ``(..., nx+2, ny+2, nz+2)``, zurueck kommt
    ``(..., nx, ny, nz)`` -- also nur die echten Knoten.

    Fuer einen Knoten mit linkem Abstand ``a`` und rechtem Abstand ``b``::

        T'' ~ 2 * ( b*T_links - (a+b)*T_mitte + a*T_rechts )
                  / ( a*b*(a+b) )

    Fuer ``a == b`` faellt das auf ``(T_l - 2T_m + T_r)/h^2`` zurueck, was auf
    Ebene 0 und 2 auch genau passiert.
    """
    h = _pad_spacings(layout.dx)
    a = _as_tensor(h[:-1], padded).reshape(-1, 1, 1)   # linke Abstaende
    b = _as_tensor(h[1:], padded).reshape(-1, 1, 1)    # rechte Abstaende
    lo = padded[..., :-2, :, :]
    mid = padded[..., 1:-1, :, :]
    hi = padded[..., 2:, :, :]
    out = 2.0 * (b * lo - (a + b) * mid + a * hi) / (a * b * (a + b))
    return out[..., 1:-1, 1:-1]


def d_dx(padded: torch.Tensor, layout: GridLayout, *,
         keep_yz_pad: bool = False) -> torch.Tensor:
    """Erste Ableitung in x, nicht-aequidistant.

    Mit ``keep_yz_pad`` bleibt das y/z-Padding stehen -- das braucht der
    Kreuzterm, der danach noch in y ableiten will.

    Auf Ebene 0 ist das Ergebnis **exakt null**: dort sind ``a == b`` und
    ``T_links == T_rechts`` (die Spiegelung), der Zaehler ist also
    ``h^2*T1 - h^2*T1 = 0``. Das ist die Symmetrierandbedingung, und sie kostet
    keinen Strafterm.
    """
    h = _pad_spacings(layout.dx)
    a = _as_tensor(h[:-1], padded).reshape(-1, 1, 1)
    b = _as_tensor(h[1:], padded).reshape(-1, 1, 1)
    lo = padded[..., :-2, :, :]
    mid = padded[..., 1:-1, :, :]
    hi = padded[..., 2:, :, :]
    out = (a * a * hi - b * b * lo + (b * b - a * a) * mid) / (a * b * (a + b))
    return out if keep_yz_pad else out[..., 1:-1, 1:-1]


def d2_dy2(padded: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """Zweite Ableitung in y; y ist aequidistant, also der uebliche Stern."""
    p = padded[..., 1:-1, :, :]
    out = (p[..., :-2, :] - 2.0 * p[..., 1:-1, :] + p[..., 2:, :]) / layout.dy ** 2
    return out[..., 1:-1]


def d2_dz2(padded: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """Zweite Ableitung in z."""
    p = padded[..., 1:-1, 1:-1, :]
    return (p[..., :-2] - 2.0 * p[..., 1:-1] + p[..., 2:]) / layout.dz ** 2


def d2_dxdy(padded: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """Gemischte Ableitung ``d/dy(dT/dx)``.

    Zwei lineare Operatoren nacheinander statt eines handgeschriebenen
    Kreuzsterns. Der Unterschied ist Pruefbarkeit: jeder der beiden Schritte
    hat seinen eigenen Test, ein Acht-Koeffizienten-Stern haette keinen.

    ``fo01`` ist genau auf der geheizten Ebene ungleich null, dieser Term
    darf also nicht stillschweigend weggelassen werden.
    """
    dx_field = d_dx(padded, layout, keep_yz_pad=True)   # (..., nx, ny+2, nz+2)
    out = (dx_field[..., 2:, :] - dx_field[..., :-2, :]) / (2.0 * layout.dy)
    return out[..., 1:-1]


def anisotropic_laplacian(padded: torch.Tensor, layout: GridLayout,
                          fo: torch.Tensor) -> torch.Tensor:
    """``Fo : grad^2 T`` auf dem gepaddeten Stapel.

    ``fo`` ist der Fourier-Tensor als Feld, ``(nx, ny, nz, 3, 3)`` -- aus
    ``data.OPData.Fo`` ueber ``grid.to_field`` gewonnen. Er wird beim Ableiten
    als ortskonstant behandelt; die Begruendung steht im Modulkopf.

    ``T_xz`` und ``T_yz`` werden nur gerechnet, wenn die zugehoerigen
    Tensorkomponenten irgendwo ungleich null sind. Fuer diesen Datensatz sind
    sie es nicht (``XZ = YZ = 0`` laut legacy-README), und der Test dafuer ist
    billiger als die beiden Ableitungen.
    """
    txx = d2_dx2(padded, layout)
    tyy = d2_dy2(padded, layout)
    tzz = d2_dz2(padded, layout)
    out = (fo[..., 0, 0] * txx + fo[..., 1, 1] * tyy + fo[..., 2, 2] * tzz)

    if torch.any(fo[..., 0, 1] != 0):
        out = out + 2.0 * fo[..., 0, 1] * d2_dxdy(padded, layout)
    if torch.any(fo[..., 0, 2] != 0) or torch.any(fo[..., 1, 2] != 0):
        raise NotImplementedError(
            "lambda_xz oder lambda_yz ist ungleich null. Fuer diesen Datensatz "
            "sind beide null (legacy-README); ein Stern dafuer ist nicht "
            "gebaut. Lieber hier laut als still einen Term verschlucken."
        )
    return out


# ---------------------------------------------------------------------------
# Der Wandterm
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UCurve:
    """``U(V_dot)`` als feste Funktion, aus dem gemessenen Waermestrom.

    ``U`` ist ein **Gesamtdurchgang** von der Gitterebene bis in den Fluidkern,
    kein Filmkoeffizient: die 1.9 mm zwischen Monitorebene (x = 0.0219) und
    Kuehlplatte (x = 0.0238) sind im Repo nicht zerlegt und stecken **in** U.
    Der Schichtwiderstand darf deshalb nicht noch einmal aufgeschlagen werden.

    Kalibriert wird offline::

        U(t) = Q_dot(t) / (A * (T2_mittel(t) - T_fluid_mittel(t)))

    **Nie auf OP13/OP15/OP16 kalibrieren.** Trainierte Flusslevel sind
    0 / 15 / 30 l/min; OP16 faehrt 90 und ist die *Gegenprobe* fuer U(V_dot),
    nie die Stuetze. Ausserhalb der Stuetzstellen wird deshalb geklemmt und
    nicht extrapoliert -- und ``clamped_calls`` zaehlt mit, wie oft das
    passiert ist, damit es im Protokoll auftaucht statt still zu bleiben.
    """

    vdot: np.ndarray    # (k,) Stuetzstellen, aufsteigend [l/min]
    u: np.ndarray       # (k,) zugehoerige U [W/m^2K]

    @staticmethod
    def from_samples(vdot, u) -> "UCurve":
        v = np.asarray(vdot, dtype=np.float64)
        uu = np.asarray(u, dtype=np.float64)
        if v.shape != uu.shape or v.ndim != 1 or v.size < 1:
            raise ValueError("vdot und u muessen gleich lange 1D-Arrays sein")
        order = np.argsort(v)
        return UCurve(vdot=v[order], u=uu[order])

    def __call__(self, vdot: float) -> float:
        """Linear interpoliert, ausserhalb der Stuetzstellen geklemmt."""
        return float(np.interp(vdot, self.vdot, self.u))

    def is_extrapolation(self, vdot: float) -> bool:
        return bool(vdot < self.vdot[0] or vdot > self.vdot[-1])


@dataclass
class FluidState:
    """Der Zustand des Kuehlmittels zwischen zwei Zeitschritten.

    Nur im Modus ``capacity`` von Bedeutung; im Modus ``advective`` ist die
    Fluidtemperatur eine reine Funktion des Augenblicks und dieser Zustand
    wird durchgereicht, ohne benutzt zu werden.
    """

    t_fluid: float   # mittlere Fluidtemperatur [gleiche Einheit wie T]


class WallModel:
    """Der Fluidpfad: von der Wandtemperatur zur Geisterschicht.

    Drei Schritte, alle aus README Sec. 6::

        1  T_fluid(y,t) = T_in(t) + Q_kum(y,t) / (mdot(t) * Cp)
        2  q_wall(y,z,t) = U * ( T2(y,z,t) - T_fluid(y,t) )
        3  ghost_hi = T2 - (dx2 / lam_xx) * q_wall

    Schritt 1 ist ein **Marsch entlang +y**: das Kuehlmittel fliesst in +y und
    erwaermt sich dabei, die Wand sieht bei kleinem y also kaelteres Fluid als
    bei grossem. Weil ``q_wall`` selbst von ``T_fluid`` abhaengt, wird
    stationsweise vorwaerts gerechnet, nicht implizit geloest.

    Zwei Modi
    ---------
    ``advective``
        Die Form oben. Richtig fuer einen Durchfluss, **divergiert aber fuer
        mdot -> 0**.
    ``capacity``
        Die Korrektur vom 09.09. (PR #32): ``mdot = 0`` heisst kein *Fluss*,
        nicht kein *Fluid*. Das Kuehlmittel steht im Kanal und laedt seine
        eigene Waermekapazitaet::

            C_fluid * dT_fluid/dt  =  Q_dot  -  mdot * Cp * (T_fluid - T_in)

        Bei ``mdot = 0`` bleibt reines Aufladen, bei grossem ``mdot`` faellt
        die stationaere Bilanz heraus. Eine Zustandsvariable mehr, keine
        Singularitaet. Die Fluidtemperatur ist dann allerdings **gemittelt**
        und nicht mehr laengs y aufgeloest.

    Welcher Modus besser ist, ist die offene Messung aus dem Fahrplan Stufe 3:
    *verbessert ein Kapazitaetsterm die V_dot = 0-Betriebspunkte, ohne dass
    neue Daten dazukommen?* Deshalb stehen beide hier, und keiner ist
    voreingestellt richtig.
    """

    def __init__(self, layout: GridLayout, u_curve: UCurve, *,
                 cp_fluid: float, area: float = WALL_AREA_M2,
                 mode: str = "advective", c_fluid: float | None = None,
                 lam_xx_wall: torch.Tensor | None = None) -> None:
        if mode not in ("advective", "capacity"):
            raise ValueError(f"Unbekannter Modus: {mode!r}")
        if mode == "capacity" and not c_fluid:
            raise ValueError(
                "Modus 'capacity' braucht c_fluid [J/K] -- die Waermekapazitaet "
                "des Kuehlmittels im Kanal. Ohne sie ist der Term nicht "
                "definiert, und ein geratener Wert waere ein freier Parameter."
            )
        self.layout = layout
        self.u_curve = u_curve
        self.cp_fluid = float(cp_fluid)
        self.area = float(area)
        self.mode = mode
        self.c_fluid = float(c_fluid) if c_fluid else 0.0
        self.lam_xx_wall = lam_xx_wall
        self.clamped_calls = 0

    # -- Flaeche je Gitterknoten --------------------------------------------
    @property
    def node_area(self) -> float:
        """``A`` gleichmaessig auf die Wandknoten verteilt.

        Bewusst ``A / (ny*nz)`` und nicht ``dy*dz`` mit Halbzellen am Rand:
        ``U`` wurde gegen **dieselbe** Gesamtflaeche ``A`` kalibriert, also
        muss die Summe der Knotenflaechen wieder exakt ``A`` ergeben. Sonst
        traegt der Wandterm systematisch zu viel oder zu wenig Energie ab, und
        zwar genau um den Faktor, den die Randzellen ausmachen.
        """
        _, ny, nz = self.layout.shape
        return self.area / (ny * nz)

    # -- Modus advective ----------------------------------------------------
    def _advective(self, t2: torch.Tensor, t_in: float, mdot: float,
                   u: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Marsch entlang +y. ``t2`` ist ``(ny, nz)``, zurueck ``(q_wall, t_fluid)``."""
        ny, _ = t2.shape
        dA = self.node_area
        q_rows, t_rows = [], []
        q_acc = 0.0
        # Bei mdot = 0 waere der Nenner null. Die Advektionsform kann diesen
        # Fall nicht: kein Fluss heisst, es wird nichts abtransportiert, die
        # Erwaermung laengs y ist unbestimmt. Wir setzen T_fluid = T_in und
        # halten im Protokoll fest, dass das eine UNTERE Schranke fuer die
        # Fluidtemperatur ist -- der Modus 'capacity' ist fuer diesen Fall
        # gebaut, nicht dieser hier.
        flow = mdot * self.cp_fluid
        for j in range(ny):
            t_f = t_in + (q_acc / flow if flow > 0 else 0.0)
            q_row = u * (t2[j] - t_f)
            q_rows.append(q_row)
            t_rows.append(t_f)
            q_acc += float(q_row.sum().item()) * dA
        q_wall = torch.stack(q_rows, dim=0)
        t_fluid = torch.tensor(t_rows, dtype=t2.dtype, device=t2.device)
        return q_wall, t_fluid

    # -- Modus capacity -----------------------------------------------------
    def _capacity(self, t2: torch.Tensor, t_in: float, mdot: float, u: float,
                  state: FluidState, dt: float,
                  ) -> tuple[torch.Tensor, torch.Tensor, FluidState]:
        """Gemittelte Fluidtemperatur als eigener Zustand, ohne Polstelle."""
        t_f = state.t_fluid
        q_wall = u * (t2 - t_f)
        q_total = float(q_wall.sum().item()) * self.node_area          # [W]
        dtf = (q_total - mdot * self.cp_fluid * (t_f - t_in)) / self.c_fluid
        new = FluidState(t_fluid=t_f + dt * dtf)
        ny, _ = t2.shape
        t_fluid = torch.full((ny,), t_f, dtype=t2.dtype, device=t2.device)
        return q_wall, t_fluid, new

    # -- oeffentlich ---------------------------------------------------------
    def ghost(self, field: torch.Tensor, *, t_in: float, mdot: float,
              vdot: float, state: FluidState, dt: float,
              ) -> tuple[torch.Tensor, torch.Tensor, FluidState]:
        """Geisterschicht hinter der Wand plus Diagnose.

        ``field`` ist ``(nx, ny, nz)``. Zurueck kommen ``ghost_hi`` mit der
        Form ``(ny, nz)``, das Wandflussfeld ``q_wall`` in W/m^2 und der neue
        Fluidzustand.

        ``q_wall`` wird mit herausgegeben, weil ``L_wall`` es spaeter gegen den
        gemessenen Waermestrom stellt -- und weil die Energiebilanz sonst nicht
        nachrechenbar waere.
        """
        if self.u_curve.is_extrapolation(vdot):
            self.clamped_calls += 1
        u = self.u_curve(vdot)
        t2 = field[-1]                       # aeusserste x-Ebene: die Wand

        if self.mode == "advective":
            q_wall, _ = self._advective(t2, t_in, mdot, u)
            new_state = state
        else:
            q_wall, _, new_state = self._capacity(t2, t_in, mdot, u, state, dt)

        lam = self.lam_xx_wall
        if lam is None:
            raise ValueError(
                "lam_xx an der Wand fehlt. Ohne sie ist die Geisterschicht "
                "nicht berechenbar -- sie uebersetzt einen Waermestrom in ein "
                "Temperaturgefaelle."
            )
        dx2 = float(self.layout.dx[-1])
        ghost_hi = t2 - (dx2 / lam) * q_wall
        return ghost_hi, q_wall, new_state


def source_term(q_dot: float, q_mask_field: torch.Tensor,
                rho_cp_field: torch.Tensor) -> torch.Tensor:
    """``Qsrc = q_dot(t) * q_mask / (rho*Cp)`` als Feld.

    ``q_dot`` ist **ein Skalar je Zeitschritt** (JR1-Gesamtleistung geteilt
    durch ``V_JR1``), ``q_mask`` die feste JR1-Ebene. Das raeumliche Muster ist
    ueber die ganze Trajektorie konstant, nur die Amplitude bewegt sich -- der
    Befund, der den Rangtest in Sec. 9 des README erst scharf gemacht hat.

    Die Quelle wirkt **nur auf JR1**, nicht auf Cell Center und nicht auf das
    Gehaeuse. Das ist am 01.09. von der Simulationsseite bestaetigt worden und
    weicht bewusst vom legacy-README ab, das "alle Wickelpunkte" sagt. Der
    Konvention dort zu folgen wuerde mehr Leistung einspeisen, als die
    Simulation berichtet.
    """
    return q_dot * q_mask_field / rho_cp_field


def cfl_limit(layout: GridLayout, fo: torch.Tensor) -> float:
    """Groesster stabiler Zeitschritt fuer explizites Euler, grob.

    ``dt_max = 1 / (2 * sum_i(Fo_ii / h_i^2))`` am schlechtesten Punkt -- die
    uebliche Schranke fuer den expliziten Diffusionsstern. Der Kreuzterm ist
    darin nicht enthalten; die Schranke ist also eine Richtgroesse und kein
    Beweis. Deshalb wird die Stabilitaet in Stufe 3 **gemessen** und nicht
    geschaetzt.
    """
    hx = float(layout.dx.min())
    inv = (fo[..., 0, 0] / hx ** 2
           + fo[..., 1, 1] / layout.dy ** 2
           + fo[..., 2, 2] / layout.dz ** 2)
    worst = float(inv.max().item())
    return float("inf") if worst <= 0 else 1.0 / (2.0 * worst)
