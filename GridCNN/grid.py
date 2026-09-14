"""Das Gitter: Reshape 363 -> (3, 11, 11) und die drei Paddings.

Warum diese Datei zuerst kommt
------------------------------
Der ganze Entwurf steht auf einer Behauptung: die 363 Punkte in ``data.OPData``
bilden ein vollstaendiges ``3 x 11 x 11``-Tensorgitter, identisch ueber alle
sechzehn Betriebspunkte. Diese Datei **leitet** den Reshape aus den Koordinaten
ab, statt ihn zu raten -- und faellt laut aus, wenn die Annahme nicht traegt.
Ein stillschweigend verdrehtes Feld waere der teuerste Fehler des Projekts:
er faellt erst drei Wochen spaeter als "das Netz konvergiert schlecht" auf.

Die drei Paddings
-----------------
Gefaltet wird ueber ``(y, z)``; ``x`` ist eine kurze echte Achse mit
Geisterschichten. Daraus folgen drei verschiedene Randbehandlungen, und nur
weil ``x`` NICHT in den Kanaelen liegt, sind zwei davon strukturell:

====================  ======================================================
x = 0  (Zellmitte)    Symmetrieebene des Halbmodells. ``ghost_lo := T1`` --
                      eine Spiegelung. Die zentrale Differenz ueber den
                      Randknoten wird damit ``T1 - T1 = 0``, also *exakt*
                      null, fuer jedes Feld und ohne Strafterm.
x = 0.0219 (Wand)     Domaenengrenze, hier tritt Waerme aus. ``ghost_hi``
                      kommt aus dem konvektiven Fluss und wird von
                      ``physics.wall_ghost`` geliefert, nicht hier.
y/z-Umfang            ``reflect``-Padding. ACHTUNG: das ruht auf einer
                      *Annahme* (adiabat), nicht auf einer Messung -- siehe
                      README Sec. 5. Exakt ist hier die Umsetzung der
                      Bedingung, nicht die Bedingung selbst.
====================  ======================================================

``reflect`` und nicht ``replicate``
-----------------------------------
Weil der Randknoten **auf** der Grenze liegt: das Raster spannt gemessen
0.198089 x 0.104441 m, der legacy-PINN-README nennt fuer die Zellflaeche
``dy=0.198``, ``dz=0.104``. Kante auf Kante.

* ``reflect`` spiegelt *um* den Randknoten, Geist := erster innerer Knoten.
  Die zentrale Differenz ueber den Randknoten wird exakt null -- die uebliche
  Art, eine Neumann-0-Bedingung auf einem knotenzentrierten Gitter zu setzen.
* ``replicate`` setzt Geist := Randwert. Das ist ein *einseitiger*
  Nullgradient, eine groebere und andere Diskretisierung derselben Bedingung.

Torch, nicht numpy
------------------
Die Werkzeuge in ``tools/`` sind bewusst numpy-only. Diese Datei nicht: sie
wird sowohl vom Loeser ohne Netz (``solve.py``) als auch spaeter vom Modell
benutzt, und zwei Implementierungen derselben Indexarithmetik driften
auseinander. Genau dieser Fehler hat das Projekt am 31.08. die Zusammenlegung
zweier Vorgaengerprojekte gekostet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GridLayout:
    """Der aus den Koordinaten abgeleitete Reshape plus die Gitterabstaende.

    ``flat_index[i, j, k]`` ist der Index des Punktes mit ``x=xu[i]``,
    ``y=yu[j]``, ``z=zu[k]`` in den flachen ``(n_points,)``-Arrays von
    ``data.OPData``. Alle Abstaende sind in Metern, in denselben Einheiten wie
    die Koordinaten, die hineingegeben wurden.
    """

    flat_index: np.ndarray   # (nx, ny, nz) int64
    xu: np.ndarray           # (nx,) aufsteigende x-Ebenen
    yu: np.ndarray           # (ny,)
    zu: np.ndarray           # (nz,)
    dx: np.ndarray           # (nx-1,) Abstaende in x -- NICHT aequidistant
    dy: float                # aequidistant, sonst faellt die Ableitung aus
    dz: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.flat_index.shape  # type: ignore[return-value]

    @property
    def n_points(self) -> int:
        return int(self.flat_index.size)

    def describe(self) -> str:
        nx, ny, nz = self.shape
        return (
            f"{nx} x {ny} x {nz} = {self.n_points} Punkte\n"
            f"  x-Ebenen : {', '.join(f'{v:.6f}' for v in self.xu)} m\n"
            f"  dx       : {', '.join(f'{v * 1e3:.3f}' for v in self.dx)} mm"
            f"   ({'aequidistant' if _uniform(self.dx) else 'NICHT aequidistant'})\n"
            f"  dy, dz   : {self.dy * 1e3:.3f} mm, {self.dz * 1e3:.3f} mm\n"
            f"  Spannweite y/z: {self.yu[-1] - self.yu[0]:.6f} x "
            f"{self.zu[-1] - self.zu[0]:.6f} m"
        )


def _uniform(d: np.ndarray, rtol: float = 1e-6) -> bool:
    return bool(d.size <= 1 or np.allclose(d, d[0], rtol=rtol))


def derive_layout(xyz: np.ndarray, *, decimals: int = 9) -> GridLayout:
    """Leitet den Reshape aus den Koordinaten ab.

    ``xyz`` ist ``(n_points, 3)`` in physikalischen Metern -- also
    ``data.OPData.xn * L_ref`` oder die rohen Koordinaten aus dem Cache. Die
    Normierung ist egal, solange sie monoton ist; die Abstaende in der
    Rueckgabe tragen dann dieselbe Einheit.

    Wirft ``ValueError``, wenn das Raster kein volles Tensorgitter ist. Das ist
    Absicht: der Reshape ist die Voraussetzung fuer alles Weitere, und ein
    unvollstaendiges Gitter still zu akzeptieren wuerde die Punkte verwuerfeln.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz muss (n_points, 3) sein, ist {xyz.shape}")

    rounded = np.round(xyz, decimals)
    xu, yu, zu = (np.unique(rounded[:, c]) for c in range(3))
    nx, ny, nz = len(xu), len(yu), len(zu)
    if nx * ny * nz != len(xyz):
        raise ValueError(
            f"Kein volles Tensorgitter: {nx} x {ny} x {nz} = {nx * ny * nz}, "
            f"aber {len(xyz)} Punkte. Der reshape waere falsch."
        )

    ix = np.searchsorted(xu, rounded[:, 0])
    iy = np.searchsorted(yu, rounded[:, 1])
    iz = np.searchsorted(zu, rounded[:, 2])
    flat_index = np.full((nx, ny, nz), -1, dtype=np.int64)
    flat_index[ix, iy, iz] = np.arange(len(xyz))
    if (flat_index < 0).any():
        raise ValueError(
            "Gitterpunkte doppelt oder fehlend -- der reshape waere unsicher."
        )

    dy_all = np.diff(yu)
    dz_all = np.diff(zu)
    if not _uniform(dy_all) or not _uniform(dz_all):
        raise ValueError(
            "y oder z ist nicht aequidistant. Die Faltung und der zentrale "
            "Differenzenstern setzen das voraus.\n"
            f"  dy: {dy_all}\n  dz: {dz_all}"
        )

    return GridLayout(
        flat_index=flat_index, xu=xu, yu=yu, zu=zu,
        dx=np.diff(xu), dy=float(dy_all[0]), dz=float(dz_all[0]),
    )


# ---------------------------------------------------------------------------
# Reshape hin und zurueck
# ---------------------------------------------------------------------------
def to_field(flat: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """``(..., n_points)`` -> ``(..., nx, ny, nz)``.

    Die fuehrenden Achsen bleiben unangetastet, es wird also sowohl ein
    einzelner Schnappschuss ``(363,)`` als auch eine ganze Trajektorie
    ``(n_t, 363)`` akzeptiert.
    """
    idx = torch.as_tensor(layout.flat_index, device=flat.device, dtype=torch.long)
    return flat[..., idx.reshape(-1)].reshape(*flat.shape[:-1], *layout.shape)


def to_flat(field: torch.Tensor, layout: GridLayout) -> torch.Tensor:
    """``(..., nx, ny, nz)`` -> ``(..., n_points)``; die Umkehrung von ``to_field``."""
    nx, ny, nz = layout.shape
    if tuple(field.shape[-3:]) != (nx, ny, nz):
        raise ValueError(
            f"Feld hat {tuple(field.shape[-3:])}, Layout erwartet {(nx, ny, nz)}"
        )
    lead = field.shape[:-3]
    idx = torch.as_tensor(layout.flat_index, device=field.device, dtype=torch.long)
    out = field.new_zeros(*lead, layout.n_points)
    out[..., idx.reshape(-1)] = field.reshape(*lead, -1)
    return out


# ---------------------------------------------------------------------------
# Die drei Paddings
# ---------------------------------------------------------------------------
def pad_yz(field: torch.Tensor) -> torch.Tensor:
    """``reflect``-Padding um eine Zelle in y und z.

    ``(..., nx, ny, nz)`` -> ``(..., nx, ny+2, nz+2)``.

    Gespiegelt wird **um den Randknoten**, der Geist ist also der erste innere
    Knoten. Damit ist die zentrale Differenz ueber den Randknoten exakt null,
    ohne dass irgendwo ein Strafterm dafuer bezahlt werden muss.

    ``torch.nn.functional.pad(mode="reflect")`` will 3D/4D/5D-Eingaben mit
    Batch- und Kanalachse, was hier nur Umformerei waere. Zwei
    ``torch.cat``-Aufrufe tun dasselbe und funktionieren fuer jede fuehrende
    Achsenzahl.
    """
    if field.shape[-2] < 2 or field.shape[-1] < 2:
        raise ValueError("reflect braucht mindestens zwei Knoten je Achse")
    y = torch.cat(
        [field[..., 1:2, :], field, field[..., -2:-1, :]], dim=-2)
    return torch.cat([y[..., 1:2], y, y[..., -2:-1]], dim=-1)


def pad_x(field: torch.Tensor, ghost_hi: torch.Tensor) -> torch.Tensor:
    """Haengt die beiden Geisterschichten in x an.

    ``(..., nx, ny, nz)`` -> ``(..., nx+2, ny, nz)``, also der fuenf tiefe
    Stapel aus README Sec. 5::

        [ ghost_lo , T0(Mitte) , T1(JR1) , T2(Wand) , ghost_hi ]

    ``ghost_lo`` wird hier **berechnet, nicht uebergeben**: es ist per
    Spiegelung an der Symmetrieebene identisch mit ``field[..., 1, :, :]``.
    Genau deshalb ist ``dT/dx = 0`` bei ``x = 0`` exakt und nicht
    naeherungsweise -- im Zaehler der Ableitung steht ``T1 - T1``, und das ist
    null fuer jeden moeglichen Netzausgang, in jedem Trainingsschritt, auch auf
    einem nie gesehenen Betriebspunkt.

    ``ghost_hi`` dagegen traegt Physik und kommt von aussen: aus
    ``physics.wall_ghost``. Es hat die Form ``(..., ny, nz)`` oder ist
    broadcastbar darauf.
    """
    if field.shape[-3] < 2:
        raise ValueError("Die Spiegelung braucht mindestens zwei x-Ebenen")
    ghost_lo = field[..., 1:2, :, :]
    hi = ghost_hi
    if hi.shape[-3:-2] != (1,):
        hi = hi.unsqueeze(-3)
    hi = hi.expand(*field.shape[:-3], 1, field.shape[-2], field.shape[-1])
    return torch.cat([ghost_lo, field, hi], dim=-3)


def pad_all(field: torch.Tensor, ghost_hi: torch.Tensor) -> torch.Tensor:
    """``pad_x`` und danach ``pad_yz``: ``(..., nx+2, ny+2, nz+2)``.

    Die Reihenfolge ist gleichgueltig -- beide Operationen betreffen
    verschiedene Achsen -- aber sie ist festgelegt, damit der Kreuzterm
    ``T_xy`` in ``physics.py`` sich auf eine bekannte Indizierung verlassen
    kann.
    """
    return pad_yz(pad_x(field, ghost_hi))
