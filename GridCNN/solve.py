"""Stufe 3: der Loeser ohne Netz.

Was das hier ist
----------------
Ein klassischer expliziter Euler auf dem ``3 x 11 x 11``-Gitter::

    T_{t+1} = T_t + dt * ( Fo : grad^2 T + Qsrc )

Kein gelerntes Gewicht. Der Wert liegt in vier Dingen, und alle vier fallen
weg, wenn man diese Stufe ueberspringt und gleich das Netz baut:

1. **Padding, Stern und Wandterm werden unabhaengig vom Lernen geprueft.** Ist
   der Kreuzterm falsch oder das Padding verdreht, sieht man es hier -- und
   nicht drei Wochen spaeter als "das Netz konvergiert schlecht".
2. **Die CFL-Frage wird empirisch beantwortet.** Laeuft der Loeser bei
   ``subsample_time: 2`` (dt = 0.2 s gegen dt_max ~ 0.241 s) stabil, oder
   nicht? Das ist dann keine Schaetzung mehr.
3. **Es liefert eine Physik-Latte.** Verglichen wird sonst gegen
   ``persistence`` und ``train-mean``; beide sind trivial. "Reine
   Waermeleitung mit kalibrierter Kuehlwand" ist eine ernsthafte Latte, und
   ein Surrogat, das sie nicht schlaegt, hat nichts gelernt, was die Physik
   nicht schon weiss.
4. **Es ist der Rest-Definitionspunkt.** Was der Loeser *nicht* trifft, ist
   genau das, was das Netz lernen muss. Damit ist die Aufgabe definiert statt
   geraten.

Entdimensioniert, wie ``data.py``
---------------------------------
Gerechnet wird in genau den Variablen, die ``data.py`` liefert: ``Tn``
z-gescort, ``Qsrc`` und ``Fo`` bereits passend skaliert, ``tn`` auf ~[0,1] und
``xn`` durch ``L_ref`` geteilt. Der Grund ist Vergleichbarkeit -- die
Physik-Latte muss mit derselben Elle gemessen werden wie ``PINNmodulusTwo``,
sonst vergleicht man Normierungen statt Modelle.

Der Wandterm rechnet dagegen in **physikalischen** Einheiten (W/m^2K, m^2,
J/kgK), weil ``U`` so kalibriert wird und weil man eine Waermestromdichte in
SI nachrechnen kann. Die Bruecke zwischen beiden Welten steht an genau einer
Stelle, ``_wall_ghost_nondim``, und ist dort hergeleitet.

Stand
-----
Der Wandterm ist **gebaut, aber noch nicht benutzbar**: er braucht ``mdot``,
``cp_fluid`` und den gemessenen Waermestrom im Buendel, und die liegen erst
nach Stufe 2 im Cache. Bis dahin laeuft der Loeser mit ``wall=None``, also mit
einer adiabaten Gehaeusewand. Das ist physikalisch falsch und wird im
Protokoll auch so ausgewiesen -- es ist eine Ablation, keine Latte.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import torch

import grid as gridmod
import physics as phys
from grid import GridLayout


@dataclass
class Scaling:
    """Die Konstanten, die zwischen entdimensioniert und SI uebersetzen.

    Alle vier stehen in ``data.NormBundle``; sie werden hier gebuendelt, damit
    die Bruecke ein Argument ist und keine verstreuten Globals.
    """

    t_mu: float        # Mittel der Temperatur [C]
    t_sigma: float     # Streuung der Temperatur [C]
    l_ref: float       # Laengenskala [m], mit der xn entdimensioniert wurde
    t_span_ref: float  # Zeitskala [s], mit der tn entdimensioniert wurde

    def to_physical(self, tn: torch.Tensor) -> torch.Tensor:
        return self.t_mu + self.t_sigma * tn

    def to_nondim(self, t_phys: torch.Tensor) -> torch.Tensor:
        return (t_phys - self.t_mu) / self.t_sigma


@dataclass
class SolveResult:
    """Was ein Rollout hinterlaesst -- Trajektorie plus Diagnose."""

    tn: np.ndarray                  # (n_t, n_points) entdimensioniert
    diverged_at: int | None = None  # Schrittindex, oder None
    max_abs: float = 0.0            # groesstes |Tn| ueber den Lauf
    cfl_dt_max: float = float("nan")
    dt_used: float = float("nan")
    wall_used: bool = False
    wall_clamped: int = 0
    notes: list[str] = dc_field(default_factory=list)

    @property
    def stable(self) -> bool:
        return self.diverged_at is None

    def summary(self) -> str:
        head = ("stabil" if self.stable
                else f"DIVERGIERT bei Schritt {self.diverged_at}")
        cfl = (f"dt {self.dt_used:.4g} gegen dt_max {self.cfl_dt_max:.4g} "
               f"({self.dt_used / self.cfl_dt_max:.2f} x)"
               if np.isfinite(self.cfl_dt_max) else "dt_max nicht endlich")
        wall = "Wandterm an" if self.wall_used else "Wandterm AUS (adiabat)"
        if self.wall_clamped:
            wall += f", {self.wall_clamped} x U geklemmt"
        lines = [f"{head} | max|Tn| {self.max_abs:.4g} | {cfl} | {wall}"]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Felder aus einem Buendel
# ---------------------------------------------------------------------------
def fields_from_flat(layout: GridLayout, *, fo: np.ndarray, q_mask: np.ndarray,
                     rho: np.ndarray, cp: np.ndarray,
                     device=None, dtype=torch.float64):
    """Die ortsfesten Groessen einmal in Gitterform bringen.

    ``fo`` ist ``(n_points, 3, 3)``, der Rest ``(n_points,)``. Zurueck kommen
    ``fo_field`` mit ``(nx, ny, nz, 3, 3)`` und zwei Skalarfelder.

    Das passiert einmal je Betriebspunkt und nicht je Zeitschritt: die
    Materialdaten sind konstant ueber Zeit *und* ueber alle OPs.
    """
    def _f(a, trailing=()):
        t = torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        flat = t.reshape(t.shape[0], -1) if trailing else t
        out = gridmod.to_field(flat.T if trailing else flat, layout)
        if trailing:
            nx, ny, nz = layout.shape
            out = out.reshape(*trailing, nx, ny, nz)
            out = out.permute(2, 3, 4, 0, 1) if len(trailing) == 2 else out
        return out

    fo_field = _f(fo, trailing=(3, 3))
    q_field = _f(q_mask)
    rho_cp_field = _f(np.asarray(rho) * np.asarray(cp))
    return fo_field, q_field, rho_cp_field


def _wall_ghost_nondim(field_n: torch.Tensor, wall: phys.WallModel,
                       scaling: Scaling, *, t_in: float, mdot: float,
                       vdot: float, state: phys.FluidState, dt_n: float):
    """Die eine Stelle, an der entdimensioniert und SI aufeinandertreffen.

    Herleitung, damit sie nachrechenbar ist statt geglaubt::

        T      = t_mu + t_sigma * Tn            (z-Score rueckwaerts)
        x      = l_ref * xn                     (Laenge rueckwaerts)

        ghost  = T2 - (dx2 / lam_xx) * q_wall            [physikalisch]

        ghost_n = (ghost - t_mu) / t_sigma
                = T2n - (dx2_n * l_ref) / (lam_xx * t_sigma) * q_wall

    ``dx2_n`` ist der Abstand im **entdimensionierten** Gitter, also muss er
    mit ``l_ref`` zurueckgerechnet werden, bevor er gegen ``lam_xx`` in SI
    steht. Genau dieser Faktor ist die Sorte Buchhaltung, an der dieses Projekt
    schon einmal einen Faktor 121 verloren hat (FAHRPLAN Sec. 11.1) -- deshalb
    steht die Rechnung hier und nicht in einem Kommentar.
    """
    field_phys = scaling.to_physical(field_n)
    ghost_phys, q_wall, new_state = wall.ghost(
        field_phys, t_in=t_in, mdot=mdot, vdot=vdot, state=state,
        dt=dt_n * scaling.t_span_ref)
    return scaling.to_nondim(ghost_phys), q_wall, new_state


# ---------------------------------------------------------------------------
# Der Rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(*, layout: GridLayout, tn0: torch.Tensor, n_steps: int, dt_n: float,
            fo_field: torch.Tensor, qsrc_n: torch.Tensor,
            wall: phys.WallModel | None = None,
            scaling: Scaling | None = None,
            drivers: dict | None = None,
            blow_up: float = 1e3) -> SolveResult:
    """Rollt ``n_steps`` explizite Euler-Schritte.

    Parameter
    ---------
    tn0
        Anfangsfeld ``(nx, ny, nz)``, entdimensioniert.
    qsrc_n
        ``(n_steps, nx, ny, nz)`` oder ``(nx, ny, nz)`` -- die
        entdimensionierte Quelle je Schritt. ``data.OPData.Qsrc`` ist bereits
        in diesen Einheiten.
    wall, scaling, drivers
        Entweder alle drei oder keines. ``drivers`` braucht die Schluessel
        ``t_in``, ``mdot``, ``vdot``, jeweils als Folge ueber die Zeit oder als
        Skalar. Ohne sie laeuft die Gehaeusewand **adiabat** -- eine Ablation,
        keine Physik.
    blow_up
        Ab welchem ``max |Tn|`` abgebrochen wird. Ein divergierender expliziter
        Loeser laeuft sonst bis ``inf`` und kostet nur Zeit. Der Wert ist
        grosszuegig: z-gescorte Temperaturen liegen bei O(1), also ist 1e3
        bereits zweifelsfrei kaputt und nicht nur ungenau.
    """
    if (wall is None) != (scaling is None):
        raise ValueError("wall und scaling gehoeren zusammen")

    fld = tn0.clone()
    out = torch.empty((n_steps + 1, *layout.shape),
                      dtype=fld.dtype, device=fld.device)
    out[0] = fld

    res = SolveResult(tn=np.empty(0), dt_used=dt_n,
                      cfl_dt_max=phys.cfl_limit(layout, fo_field),
                      wall_used=wall is not None)
    if wall is None:
        res.notes.append(
            "Gehaeusewand adiabat gerechnet -- Stufe 2 (Cache) ist offen, "
            "also fehlen mdot, cp_fluid und der gemessene Waermestrom. "
            "Das ist eine Ablation, keine Physik-Latte.")

    state = phys.FluidState(t_fluid=float(
        drivers.get("t_in_init", 0.0))) if drivers else phys.FluidState(0.0)
    src_is_seq = qsrc_n.dim() == 4

    def _drv(key: str, step: int) -> float:
        v = drivers[key]
        return float(v[step]) if np.ndim(v) else float(v)

    for k in range(n_steps):
        if wall is not None and scaling is not None:
            ghost_hi, _, state = _wall_ghost_nondim(
                fld, wall, scaling, t_in=_drv("t_in", k), mdot=_drv("mdot", k),
                vdot=_drv("vdot", k), state=state, dt_n=dt_n)
        else:
            # Adiabat: Nullgradient an der Wand, also spiegeln wie bei x = 0.
            ghost_hi = fld[-2]

        padded = gridmod.pad_all(fld, ghost_hi)
        rhs = phys.anisotropic_laplacian(padded, layout, fo_field)
        rhs = rhs + (qsrc_n[k] if src_is_seq else qsrc_n)
        fld = fld + dt_n * rhs
        out[k + 1] = fld

        m = float(fld.abs().max().item())
        res.max_abs = max(res.max_abs, m)
        if not np.isfinite(m) or m > blow_up:
            res.diverged_at = k + 1
            out = out[: k + 2]
            break

    res.tn = gridmod.to_flat(out, layout).cpu().numpy()
    if wall is not None:
        res.wall_clamped = wall.clamped_calls
    return res
