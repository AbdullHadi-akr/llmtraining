#!/usr/bin/env python3
"""Die Trainingsschleife -- K Schritte gegen eine eingefrorene Trajektorie.

Die eine Eigenschaft, die man hier nicht kaputtmachen darf
-----------------------------------------------------------
Je Epoche und Betriebspunkt wird die **eigene** Trajektorie einmal unter
``torch.no_grad()`` frei ausgerollt, gesaet nur von der gemessenen
Anfangsbedingung, und dann eingefroren. Erst darauf laufen ``inner_steps``
Updates gegen die Labels -- je Update ein Fenster von ``k`` Schritten, das
**im Puffer startet und danach sich selbst frisst**.

**Das ist kein Teacher Forcing**, und es wird oft falsch erzaehlt. Der
Unterschied zwischen Training und Auswertung ist nicht der Eingangszustand --
beide rollen frei. Er ist::

    ================  ==========================  ===============
                      Training                    Auswertung
    ================  ==========================  ===============
    Trajektorie       eingefroren, je Epoche neu  live
    Labels            ja, als Ziel                nein
    Gradient          ueber k Schritte            keiner
    ================  ==========================  ===============

Stufe 5 (truncated BPTT) ist gebaut -- ``--tbptt``
---------------------------------------------------
Bis zum 22.09. stand hier ``k = 1``, und **daraus folgte der Lauf, der kein
Ergebnis war**. Ein Schritt trainierte, rund 8040 wurden gemessen; nichts in
der Schleife beschraenkte das Verhalten ueber diesen Horizont, und bis
99.8 % aller OP-Zeitschritte lagen am ``--clamp``. Die Analyse steht in
``TRAININGS_BERICHT_2026-09-22_KonfigA.md``.

``k > 1`` schliesst genau diese Luecke: das Fenster wird **mit** Gradient
gerollt, jeder seiner Schritte gegen sein eigenes Label. Damit sieht der
Optimierer erstmals den aufsummierten Fehler statt nur den naechsten Schritt --
der Spaetfehler (O13) ist strukturell erreichbar.

**``k = 1`` ist exakt der alte Pfad**, bis auf die Gleitkommabits: dann ruft
``train_epoch`` weiterhin ``data_loss`` auf. Beides wird von Tests gehalten --
``test_der_gradient_ueberquert_die_historie_nicht`` fuer den Ein-Schritt-Fall,
``test_der_gradient_ueberquert_die_historie_bei_k_groesser_eins_sehr_wohl``
fuer den anderen. Wer hier optimiert, darf **keine** der beiden wegraeumen.

Der Preis ist linear: ein Update kostet ``k`` Vorwaerts- und Rueckwaerts-
schritte. ``inner_steps`` gehoert deshalb gesenkt, wenn ``k`` steigt -- was
zaehlt, ist ``inner_steps x k``.

Warum eingefroren und trotzdem unverzerrt
------------------------------------------
Die Rekurrenz detacht die Historie ohnehin zwischen den Schritten. Der Gradient
zum Zeitpunkt t verlaesst also nie die Feldauswertung dieses Schrittes, auch
ohne das Einfrieren. Der Gradient des vollen ``L_data`` ist damit **exakt** eine
Summe unabhaengiger Einzelgradienten gegen eine Trajektorie, in der die Gewichte
als konstant behandelt werden -- und genau das schaetzt ein Minibatch aus
Zeitpunkten unverzerrt.

Was es bringt: ein ~7000-Schritt-Rollout zahlt sonst **einen** Optimiererschritt.
So traegt derselbe Rollout ``inner_steps`` Updates.

Der Preis: nach einigen Updates ist der Puffer nicht mehr ganz die Trajektorie,
die die aktuellen Gewichte erzeugen wuerden. Er wird je Epoche erneuert,
``inner_steps`` tauscht also Updatezahl gegen Schalheit -- Hunderte, nicht
Zehntausende.

Die drei Verluste
-----------------
``L = w_data * L_data + w_phys * L_phys + w_wall * L_wall``

**Kein ``L_bc``.** Die Randbedingungen sind Padding (README Sec. 5): die
Symmetrie bei x = 0 ist exakt erfuellt, weil im Zaehler ``T1 - T1`` steht, und
die Kuehlwand sitzt in der Geisterschicht. Braucht das Modell einen Strafterm
dafuer, ist das Padding falsch -- dann wird das Padding repariert und nicht ein
Term addiert.

``L_wall`` ist der einzige Term mit einem **gemessenen** Ziel und heute nicht
lauffaehig: er braucht ``q_solid_to_fluid`` im Buendel, und das kommt erst mit
Stufe 2. Bis dahin laeuft er als NaN mit, nicht als 0.0 -- damit eine
Konvergenzkurve eine Luecke zeigt statt einer flachen Linie, die es nie gab.

Die T4, und was auf ihr wirklich hilft
---------------------------------------
Gerechnet wird auf einer **Tesla T4** (g4dn, sm_75, 15.6 GiB). Der GPU-Server-
README von ``PINNmodulusTwo`` hat den Engpass dort gemessen, und er trifft
GridCNN staerker als das MLP::

    ~7000 sequentielle Rollout-Schritte je OP und Epoche, jeder ein winziger
    Kernel: ~5 us Startlatenz gegen ~1.5 us Rechenzeit. Die GPU wartet auf
    Python.

Ein Faltungsschritt auf 11 x 11 mit 16 Kanaelen ist fuer eine T4 **nichts**.
Die Laufzeit ist reine Startlatenz mal Schrittzahl. Daraus folgt genau ein
Hebel, und er steht in ``stack_ops`` / ``rollout``:

> **Alle OPs werden in EINEM Rollout gerollt**, nicht elf nacheinander.
> Dieselbe Zahl sequentieller Schritte, aber elffach Arbeit je Kernelstart --
> also fast dieselbe Zeit fuer elfmal so viel. Das ist die eine Optimierung,
> die hier etwas bringt.

⚠ **Und sie ist gratis, weil sie unter ``no_grad`` passiert.** Die *innere*
Schleife bleibt bewusst **je OP** getrennt: einen groesseren Batch dort zu
nehmen macht den Gradienten leiser, und das waere eine Aenderung am Optimierer,
nicht an der Geschwindigkeit. Der GPU-README von ``PINNmodulusTwo`` warnt
genau davor. Geschwindigkeit ja, Experiment anfassen nein.

**Was ausdruecklich NICHT hilft, damit es niemand versucht:**

* **TF32.** Ist Ampere und neuer (sm_80+). Die T4 ist **Turing, sm_75** --
  ``torch.backends.cuda.matmul.allow_tf32`` ist dort wirkungslos. Der Schalter
  in ``device_utils.enable_tf32`` existiert fuer andere Karten.
* **Ein breiteres Netz, um die Karte zu fuellen.** Der Speicher ist nicht die
  Grenze: gemessen wurden 0.11 GB von 15.6 GB. Die Karte ist nicht voll, weil
  das Problem klein ist, und ein groesseres Netz waere eine Architektur-
  entscheidung mit einer Ausrede.
* **``channels_last``.** Bei 11 x 11 ist die Speicheranordnung bedeutungslos.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid as gridmod          # noqa: E402
import model as M               # noqa: E402
import physics as phys          # noqa: E402
from grid import GridLayout     # noqa: E402


# ---------------------------------------------------------------------------
# Was ein Betriebspunkt fuer das Training braucht
# ---------------------------------------------------------------------------
@dataclass
class OPTensors:
    """Ein Betriebspunkt, schon in Gitterform und auf dem Geraet.

    Einmal je Lauf gebaut. Der Reshape von ``(n_t, 363)`` auf
    ``(n_t, 3, 11, 11)`` kostet nichts, aber ihn je Zeitschritt zu machen
    kostet ueber eine Epoche hinweg messbar.
    """

    op_id: str
    tn_seq: torch.Tensor        # (n_t, nx, ny, nz) Labels, z-gescort
    tn_ic: torch.Tensor         # (nx, ny, nz) gemessene Anfangsbedingung
    qsrc: torch.Tensor          # (n_t, nx, ny, nz)
    fo: torch.Tensor            # (nx, ny, nz, 3, 3)
    config: torch.Tensor        # (n_t, 7)
    forcing: torch.Tensor       # (n_t, 11)
    dtn: float
    split_t: int
    n_t: int
    q_wall_meas: torch.Tensor | None = None   # (n_t,) W -- erst nach Stufe 2


@dataclass
class EpochStats:
    """Was eine Epoche hinterlaesst. Zahlen, die eine Kurve nicht zeigt."""

    data: float = 0.0
    phys: float = float("nan")
    wall: float = float("nan")
    saturated: int = 0
    saturated_max: int = 0      # B x (n_max - 1) -- ohne das ist saturated unlesbar
    spread_space: float = 0.0
    spread_time: float = 0.0
    n_ops: int = 0
    tbptt: int = 1              # Fensterlaenge dieser Epoche (1 = Ein-Schritt)
    grad_norm: float = float("nan")   # mittlere Norm VOR dem Clipping
    uebersprungen: int = 0      # Updates mit nicht-endlichem Verlust/Gradienten
    dauer_s: float = float("nan")     # Wanduhr dieser Epoche
    notes: list = field(default_factory=list)

    def line(self, epoch: int) -> str:
        w = "  --  " if np.isnan(self.wall) else f"{self.wall:.4g}"
        p = "  --  " if np.isnan(self.phys) else f"{self.phys:.4g}"
        s = (f"ep {epoch:>4d} | k {self.tbptt:>3d} "
             f"| data {self.data:.5g} | phys {p} | wall {w} "
             f"| Streuung Ort {self.spread_space:.3f} Zeit {self.spread_time:.3f}")
        if not np.isnan(self.grad_norm):
            # Die Zahl, die am 22.09. gefehlt hat: 'data' schwankte auf 2500,
            # und niemand konnte sagen, ob der Verlust gross war oder der
            # Gradient explodiert ist. Das sind zwei verschiedene Krankheiten.
            s += f" | |g| {self.grad_norm:.3g}"
        if self.uebersprungen:
            s += f" | [UEBERSPRUNGEN] {self.uebersprungen}"
        if not np.isnan(self.dauer_s):
            # Damit sich ein Budget aus dem Log planen laesst statt aus dem
            # Gefuehl: mit --tbptt k kostet eine Epoche rund k-mal so viel.
            s += f" | {self.dauer_s:.1f}s"
        if self.saturated:
            # Als Anteil, nicht als nackte Zahl: "88248" ist ohne die
            # Obergrenze nicht zu lesen, und genau daran ist der Lauf vom
            # 22.09. fast vorbeigegangen. Siehe README_DIAGNOSTIK.md.
            if self.saturated_max:
                s += (f" | [SATURATED] {self.saturated}/{self.saturated_max}"
                      f" = {100.0 * self.saturated / self.saturated_max:.1f}%")
            else:
                s += f" | [SATURATED] {self.saturated}"
        return s


# ---------------------------------------------------------------------------
# Historie aus der eingefrorenen Trajektorie
# ---------------------------------------------------------------------------
def history_at(traj: torch.Tensor, idx: torch.Tensor, lag1: int, lag2: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anker plus zwei Lags, an den Zeitindizes ``idx`` aus ``traj`` gegriffen.

    ``traj`` ist ``(n_t, nx, ny, nz)``, ``idx`` ist ``(B,)``. Die Lags werden
    bei 0 geklemmt: am Anfang der Trajektorie gibt es keine Vergangenheit, und
    die Alternative -- das Fenster zu verschieben -- wuerde die ersten Schritte
    stillschweigend aus dem Training nehmen.

    Geklemmt heisst: fuer ``t < lag`` ist die Rate ``T_t - T_0``, also kleiner
    als sie waere. Das ist eine Naeherung am Rand und keine, die sich verbirgt.
    """
    return (traj[idx],
            traj[(idx - lag1).clamp(min=0)],
            traj[(idx - lag2).clamp(min=0)])


# ---------------------------------------------------------------------------
# Mehrere OPs als eine Batch-Achse -- der T4-Hebel
# ---------------------------------------------------------------------------
@dataclass
class OPBatch:
    """Alle Betriebspunkte gestapelt, damit **ein** Rollout sie alle rollt.

    Der Grund steht im Modulkopf: der Rollout ist startlatenz-gebunden, nicht
    rechengebunden. Elf OPs nacheinander zu rollen kostet elfmal die Latenz;
    sie zusammen zu rollen kostet sie **einmal**, bei elffacher Arbeit je
    Kernel -- und die ist auf einer T4 bei 11 x 11 nicht messbar.

    Die OPs haben verschieden viele Zeitschritte. Gerollt wird bis ``n_max``,
    und die Treiber der kuerzeren werden dafuer mit ihrer letzten Zeile
    aufgefuellt. **Gelesen wird ueber das Ende hinaus nie**: jeder OP traegt
    sein eigenes ``n_t`` und sein eigenes ``split_t``, und die innere Schleife
    zieht ihre Zeitindizes daraus.
    """

    ops: list
    tn_ic: torch.Tensor      # (B, nx, ny, nz)
    config: torch.Tensor     # (B, n_max, 7)
    forcing: torch.Tensor    # (B, n_max, 11)
    qsrc: torch.Tensor       # (B, n_max, nx, ny, nz)
    fo: torch.Tensor         # (nx, ny, nz, 3, 3) -- ueber alle OPs identisch
    dtn: float
    n_max: int

    def __len__(self) -> int:
        return len(self.ops)


def _pad_to(a: torch.Tensor, n: int) -> torch.Tensor:
    """Auf ``n`` Zeitschritte auffuellen, indem die letzte Zeile wiederholt wird.

    Wiederholen und nicht mit Nullen fuellen: eine Null waere im z-Score eine
    Temperatur von ``T_mu`` und ein Treiberwert, den es nie gab. Gelesen wird
    der aufgefuellte Bereich ohnehin nicht -- aber falsch aufzufuellen ist die
    Sorte Fehler, die erst auffaellt, wenn sie einmal doch gelesen wird.
    """
    if a.shape[0] >= n:
        return a[:n]
    letzte = a[-1:].expand(n - a.shape[0], *a.shape[1:])
    return torch.cat([a, letzte], dim=0)


def stack_ops(ops: list) -> OPBatch:
    """Baut die Batch-Achse. Einmal je Lauf, nicht je Epoche."""
    if not ops:
        raise ValueError("leere OP-Liste")
    n_max = max(o.n_t for o in ops)
    dtn = {o.dtn for o in ops}
    if len(dtn) != 1:
        raise ValueError(
            f"Die OPs haben verschiedene Zeitschritte {sorted(dtn)}. Ein "
            f"gemeinsamer Rollout setzt dasselbe dt voraus -- sonst rollen sie "
            f"verschieden weit und der Vergleich waere keiner.")
    return OPBatch(
        ops=list(ops),
        tn_ic=torch.stack([o.tn_ic for o in ops]),
        config=torch.stack([_pad_to(o.config, n_max) for o in ops]),
        forcing=torch.stack([_pad_to(o.forcing, n_max) for o in ops]),
        qsrc=torch.stack([_pad_to(o.qsrc, n_max) for o in ops]),
        fo=ops[0].fo, dtn=ops[0].dtn, n_max=n_max)


# ---------------------------------------------------------------------------
# Der freie Rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_batched(net: M.GridCNN, batch: OPBatch, statics: M.StaticMaps, *,
                    lag1: int, lag2: int, clamp: float = 0.0,
                    wall: phys.WallModel | None = None
                    ) -> tuple[torch.Tensor, int]:
    """Frei laufend, gesaet nur von den gemessenen Anfangsbedingungen.

    Zurueck kommt ``(traj, n_saturated)`` mit ``traj`` der Form
    ``(n_max+1, B, nx, ny, nz)`` -- Zeit vorn, damit ``history_at`` unveraendert
    darauf arbeitet.

    ``clamp`` haelt einen weglaufenden Rollout fest, damit der Verlust endlich
    bleibt. **Das Festhalten wird gezaehlt und gemeldet**: Stille saehe hier
    aus wie langsame Konvergenz, und das ist der teuerste Irrtum, den diese
    Schleife anbieten kann. Der ``[SATURATED]``-Zaehler aus ``PINNmodulusTwo``
    wird genau deshalb mitgenommen.
    """
    n, b = batch.n_max - 1, len(batch)
    traj = torch.empty((n + 1, b, *batch.tn_ic.shape[1:]),
                       dtype=batch.tn_ic.dtype, device=batch.tn_ic.device)
    traj[0] = batch.tn_ic
    ny, nz = batch.tn_ic.shape[2:]
    saturated = 0

    for k in range(n):
        t0 = traj[k]
        t1 = traj[max(k - lag1, 0)]
        t2 = traj[max(k - lag2, 0)]
        x = M.assemble_input(
            M.state_channels(t0, t1, t2), statics,
            M.driver_channels(batch.config[:, k], batch.forcing[:, k], ny, nz))
        ghost = (M.adiabatic_ghost(t0) if wall is None
                 else _wall_ghost(t0, wall, batch, k))
        nxt = net.step(t0, x, dt_n=batch.dtn, fo_field=batch.fo,
                       qsrc=batch.qsrc[:, k], ghost_hi=ghost)
        if clamp > 0.0:
            # Je OP gezaehlt, nicht je Schritt -- sonst haenge die Zahl an der
            # Batchgroesse statt am Verhalten.
            saturated += int((nxt.abs() >= clamp).flatten(1).any(dim=1).sum())
            nxt = nxt.clamp(-clamp, clamp)
        traj[k + 1] = nxt
    return traj, saturated


@torch.no_grad()
def rollout(net: M.GridCNN, op: OPTensors, statics: M.StaticMaps, *,
            lag1: int, lag2: int, clamp: float = 0.0,
            wall: phys.WallModel | None = None) -> tuple[torch.Tensor, int]:
    """Ein einzelner OP -- duenner Aufsatz auf ``rollout_batched``.

    Bewusst keine zweite Implementierung: zwei Fassungen derselben Rekurrenz
    driften auseinander, und diese hier traegt die Eigenschaft, an der Stufe 5
    haengt. Zurueck kommt ``(n_t, nx, ny, nz)``, also ohne Batch-Achse.
    """
    traj, sat = rollout_batched(net, stack_ops([op]), statics, lag1=lag1,
                                lag2=lag2, clamp=clamp, wall=wall)
    return traj[:, 0], sat


def _wall_ghost(t0: torch.Tensor, wall: phys.WallModel, op: OPTensors,
                k: int) -> torch.Tensor:
    """Der Wandterm -- **nicht mehr blockiert, nur nicht verdrahtet**.

    ⚠ Der Text hier hat bis zum 22.09., abends, etwas Falsches behauptet: er
    nannte Stufe 2 als offene Voraussetzung. **Stufe 2 ist durch** (17/17 OPs
    auf Schema v3), ``U(V_dot)`` ist aus drei Flussleveln kalibriert, und
    ``physics.UCurve`` wie ``physics.WallModel.ghost`` sind gebaut und
    getestet. Die Sperre war abgestanden, und ein abgestandener Blocker ist
    schlimmer als ein offener: niemand sieht nach.

    Was **wirklich** noch fehlt, und nur das:

    1. ``op_tensoren`` fuellt ``OPTensors.q_wall_meas`` nie -- es steht auf
       ``None``, also kann auch ``wall_loss`` nicht laufen.
    2. ``WallModel.ghost`` braucht ``t_in`` und ``mdot`` **je Zeitschritt**.
       Beide liegen seit Stufe 2 im Buendel, sind aber noch nicht in
       ``OPTensors`` uebernommen.

    Die Bruecke zwischen entdimensioniert und SI steht in ``solve.py`` und
    wird hier nicht ein zweites Mal geschrieben.
    """
    raise NotImplementedError(
        "Der Wandterm ist kalibrierbar, aber nicht verdrahtet: "
        "OPTensors.q_wall_meas bleibt None, und t_in/mdot je Zeitschritt "
        "fehlen in op_tensoren. Beides liegt seit Stufe 2 im Buendel. "
        "Bis das gemacht ist, laeuft das Training adiabat -- eine Ablation, "
        "keine Physik-Latte. Siehe FAHRPLAN, 'Das Naechste'.")


# ---------------------------------------------------------------------------
# Die Verluste
# ---------------------------------------------------------------------------
def data_loss(net: M.GridCNN, op: OPTensors, statics: M.StaticMaps,
              traj: torch.Tensor, idx: torch.Tensor, *, lag1: int, lag2: int
              ) -> torch.Tensor:
    """Ein-Schritt-MSE gegen die Labels.

    Vorhergesagt wird ``T_{t+1}`` aus dem **eingefrorenen** ``T_t``; verglichen
    wird gegen das Label bei ``t+1``. ``idx`` sind die Zeitindizes von ``t``.

    Die Historie kommt aus ``traj``, also aus dem Puffer -- nicht aus den
    Labels. Genau hier wuerde Teacher Forcing entstehen, wenn man ``tn_seq``
    einsetzte, und genau deshalb steht es hier nicht.
    """
    t0, t1, t2 = history_at(traj, idx, lag1, lag2)
    ny, nz = t0.shape[2:]
    x = M.assemble_input(
        M.state_channels(t0, t1, t2), statics,
        M.driver_channels(op.config[idx], op.forcing[idx], ny, nz))
    pred = net.step(t0, x, dt_n=op.dtn, fo_field=op.fo, qsrc=op.qsrc[idx],
                    ghost_hi=M.adiabatic_ghost(t0))
    return torch.mean((pred - op.tn_seq[idx + 1]) ** 2)


def rollout_loss(net: M.GridCNN, op: OPTensors, statics: M.StaticMaps,
                 traj: torch.Tensor, idx: torch.Tensor, *, lag1: int,
                 lag2: int, k: int) -> torch.Tensor:
    """``k`` Schritte MIT Gradient, gestartet im eingefrorenen Puffer.

    Das ist Stufe 5 -- truncated BPTT -- und die einzige Aenderung, die den
    Spaetfehler erreichen kann. ``data_loss`` bestraft **einen** Schritt;
    gemessen werden Tausende. Ein winziger systematischer Fehler je Schritt
    ist unter ``data_loss`` unsichtbar und ueber den Horizont toedlich.

    Der Ablauf, und warum er kein Teacher Forcing ist:

    1. Der Startzustand ist ``traj[idx]`` -- die **eigene** Trajektorie aus
       dem eingefrorenen Puffer, nicht das Label. Genau wie bei ``data_loss``.
    2. Ab Schritt 1 ist der Anker die **eigene Vorhersage** des vorigen
       Schrittes, nicht der Puffer und erst recht nicht das Label. Das Fenster
       frisst sich selbst, und genau daran wird der Drift sichtbar.
    3. Die Lags greifen in den Puffer zurueck, solange sie vor dem Fenster
       liegen (``j < lag``), und danach in die eigenen Vorhersagen. Der
       Uebergang ist stetig: beide Quellen sind dieselbe Groesse, nur einmal
       eingefroren und einmal frisch.

    Zurueck kommt der **Mittelwert** ueber die ``k`` Schritte, nicht die
    Summe: sonst haenge die Schrittweite an ``k`` und ein Curriculum waere
    zugleich ein LR-Plan.

    ``k <= 1`` faellt auf ``data_loss`` zurueck -- Zeichen fuer Zeichen
    derselbe Pfad, damit ``--tbptt 1`` den Lauf vom 22.09. reproduziert.
    """
    if k <= 1:
        return data_loss(net, op, statics, traj, idx, lag1=lag1, lag2=lag2)

    zustand = [traj[idx]]              # zustand[j] ist der Zustand bei idx + j
    ny, nz = zustand[0].shape[2:]
    verlust = zustand[0].new_zeros(())
    for j in range(k):
        t0 = zustand[j]
        # Vor dem Fenster aus dem Puffer, im Fenster aus der eigenen
        # Vorhersage. Die Klemmung bei 0 ist dieselbe wie in history_at.
        t1 = (zustand[j - lag1] if j >= lag1
              else traj[(idx + j - lag1).clamp(min=0)])
        t2 = (zustand[j - lag2] if j >= lag2
              else traj[(idx + j - lag2).clamp(min=0)])
        x = M.assemble_input(
            M.state_channels(t0, t1, t2), statics,
            M.driver_channels(op.config[idx + j], op.forcing[idx + j], ny, nz))
        pred = net.step(t0, x, dt_n=op.dtn, fo_field=op.fo,
                        qsrc=op.qsrc[idx + j], ghost_hi=M.adiabatic_ghost(t0))
        verlust = verlust + torch.mean((pred - op.tn_seq[idx + j + 1]) ** 2)
        zustand.append(pred)
    return verlust / k


def physics_loss(net: M.GridCNN, op: OPTensors, statics: M.StaticMaps,
                 traj: torch.Tensor, idx: torch.Tensor, *, lag1: int, lag2: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuum der Waermeleitungsgleichung auf der eigenen Trajektorie.

    ``res = (T_{t+1} - T_t)/dt  -  ( L(T_t) + Qsrc_t )``

    Also genau die Groesse, die ``g_theta`` beitraegt -- der Term bestraft die
    gelernte Korrektur dafuer, von der Physik abzuweichen. Er ist ein
    Stuetzrad, kein Naturgesetz: die Physik ist bereits **in** der Architektur,
    und dieser Term sagt zusaetzlich, dass die Abweichung klein bleiben soll.

    Zurueck kommt ``(verlust, verhaeltnis)``. Das Verhaeltnis ist die Streuung
    des Residuums gegen die Streuung des Physik-Terms selbst. **Es gehoert
    daneben**, weil das Residuum auf einem raeumlich konstanten Feld identisch
    verschwindet: ein fallender Physik-Verlust ist nur dann ein Beleg fuer
    Physik, solange dieses Verhaeltnis nicht mit gegen null geht. Faellt es
    mit, hat der Optimierer die triviale Loesung gefunden -- und keine
    Verlustkurve dieses Laufs wuerde das zeigen.
    """
    t0, t1, t2 = history_at(traj, idx, lag1, lag2)
    ny, nz = t0.shape[2:]
    x = M.assemble_input(
        M.state_channels(t0, t1, t2), statics,
        M.driver_channels(op.config[idx], op.forcing[idx], ny, nz))
    ghost = M.adiabatic_ghost(t0)
    padded = gridmod.pad_all(t0, ghost)
    physik = phys.anisotropic_laplacian(padded, net.layout, op.fo) + op.qsrc[idx]
    rate = net.rate(t0, x, fo_field=op.fo, qsrc=op.qsrc[idx], ghost_hi=ghost)
    res = rate - physik
    verhaeltnis = res.std() / (physik.std() + 1e-12)
    return torch.mean(res ** 2), verhaeltnis.detach()


def wall_loss(q_wall: torch.Tensor, q_measured: torch.Tensor,
              node_area: float) -> torch.Tensor:
    """Der Wandfluss gegen den gemessenen Waermestrom.

    ``q_wall`` ist ``(B, ny, nz)`` in W/m^2, ``q_measured`` ist ``(B,)`` in W.
    Aufsummiert ueber die Knotenflaechen muss das eine das andere treffen.

    ⚠ ``Q̇`` ist **Aufsicht, nie Eingang**. Es ist ein Simulationsergebnis und
    zur Laufzeit nicht verfuegbar. Eine val-MAE, die mit ihm als Eingang
    entsteht, ist wertlos -- README Sec. 6.
    """
    summe = q_wall.sum(dim=(1, 2)) * node_area
    return torch.mean((summe - q_measured) ** 2)


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------
@torch.no_grad()
def spread_ratios(traj: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Wie viel Struktur die eigene Trajektorie noch hat, gegen die der Labels.

    Beide Residuenterme verschwinden identisch auf einem Feld, das in Ort und
    Zeit konstant ist. Ein Verhaeltnis nahe 1 heisst, das Modell traegt noch
    Struktur; nahe 0 heisst, es ist flach gelaufen -- die triviale Loesung.
    """
    lab = labels[:traj.shape[0]]
    s = float(traj.std(dim=(1, 2, 3)).mean()) / (
        float(lab.std(dim=(1, 2, 3)).mean()) + 1e-12)
    t = float(traj.std(dim=0).mean()) / (float(lab.std(dim=0).mean()) + 1e-12)
    return s, t


# ---------------------------------------------------------------------------
# Eine Epoche
# ---------------------------------------------------------------------------
def train_epoch(net: M.GridCNN, batch: OPBatch, statics: M.StaticMaps,
                opt: torch.optim.Optimizer, *, inner_steps: int,
                batch_t: int, lag1: int, lag2: int, w_data: float,
                w_phys: float, w_wall: float, clamp: float,
                rng: np.random.Generator, tbptt: int = 1,
                clip_grad: float = 0.0) -> EpochStats:
    """**Einmal** alle OPs ausrollen, einfrieren, dann je OP ``inner_steps`` Updates.

    Die Zweiteilung ist Absicht und steht im Modulkopf:

    * **Der Rollout ist gebatcht.** Er laeuft unter ``no_grad``, aendert also
      nichts am Experiment -- nur an der Zeit, die er auf einer T4 kostet.
    * **Die innere Schleife bleibt je OP.** Alle OPs in einen Optimiererschritt
      zu ziehen machte den Gradienten leiser, und das waere eine Aenderung am
      Optimierer. Geschwindigkeit ja, Experiment anfassen nein.

    ``tbptt`` ist die Fensterlaenge ``k`` je Update. Die Vorgabe **1** ist der
    Pfad vom 22.09.; ``main`` setzt sie hoeher, die Tests nicht. So bleibt
    diese Funktion neutral und die Entscheidung steht an einer Stelle.

    ``clip_grad`` ist die Schranke fuer ``clip_grad_norm_``; ``0`` schaltet
    sie ab. Die Norm wird **immer** gemessen und gemeldet, auch ohne
    Schranke -- am 22.09. war nicht zu entscheiden, ob der Verlust gross oder
    der Gradient explodiert war.

    Ein Update mit nicht-endlichem Verlust oder Gradienten wird **verworfen
    und gezaehlt**, nicht in die Gewichte geschrieben. Ein einziges NaN
    vergiftet sonst den ganzen Lauf, und die Kurve zeigte nur eine flache
    Linie.
    """
    ops = batch.ops
    st = EpochStats()
    st.n_ops = len(ops)
    st.tbptt = max(1, int(tbptt))
    want_phys = w_phys > 0.0
    g_sum, g_n = 0.0, 0
    t_start = time.time()

    alle, sat = rollout_batched(net, batch, statics, lag1=lag1, lag2=lag2,
                                clamp=clamp)
    st.saturated = sat
    # Dieselbe Rechnung wie in rollout_batched: B OPs x (n_max - 1) Schritte.
    st.saturated_max = len(ops) * max(batch.n_max - 1, 0)

    for i, op in enumerate(ops):
        traj = alle[:op.n_t, i]          # nur der eigene, gueltige Teil
        s_o, s_t = spread_ratios(traj, op.tn_seq)
        st.spread_space += s_o / len(ops)
        st.spread_time += s_t / len(ops)

        # Labels nur bis split_t. Dahinter liegt das Fenster, das
        # op_metrics als in-time ausgehalten berichtet -- es zu fitten
        # machte aus einer ausgehaltenen Zahl eine Trainingszahl.
        #
        # Mit einem Fenster der Laenge k ist das letzte Ziel idx + k. Also
        # muss idx um k frueher enden, nicht um 1 -- sonst greift das Fenster
        # ueber split_t hinaus und die Haltemenge waere keine mehr. Bei sehr
        # kurzen OPs wird k gekuerzt statt der OP verworfen.
        k = max(1, min(st.tbptt, int(op.split_t) - 1))
        hoch = max(int(op.split_t) - k, 1)
        d_sum = p_sum = 0.0
        n_gezaehlt = 0
        for _ in range(inner_steps):
            # t startet bei 0, das Ziel ist t+1. Zeile 0 ist die auferlegte
            # Anfangsbedingung und wird nie vorhergesagt, nur benutzt.
            idx = torch.as_tensor(rng.integers(0, hoch, size=batch_t),
                                  device=traj.device)
            if k > 1:
                dv = rollout_loss(net, op, statics, traj, idx,
                                  lag1=lag1, lag2=lag2, k=k)
            else:
                dv = data_loss(net, op, statics, traj, idx,
                               lag1=lag1, lag2=lag2)
            loss = w_data * dv
            wert = float(loss.detach())
            if want_phys:
                lp, _ = physics_loss(net, op, statics, traj, idx,
                                     lag1=lag1, lag2=lag2)
                loss = loss + w_phys * lp
                p_sum += float(lp.detach())
            if not np.isfinite(wert) or not bool(torch.isfinite(loss)):
                st.uebersprungen += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # max_norm=inf skaliert nachweislich nichts (clip_coef wird auf 1
            # geklemmt), liefert aber die Norm. So steht sie auch dann im Log,
            # wenn nicht geklemmt wird.
            norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                clip_grad if clip_grad > 0.0 else float("inf"))
            if not bool(torch.isfinite(norm)):
                st.uebersprungen += 1
                opt.zero_grad(set_to_none=True)
                continue
            g_sum += float(norm)
            g_n += 1
            d_sum += wert
            n_gezaehlt += 1
            opt.step()

        # Durch die Zahl der WIRKLICH gezaehlten Updates, nicht durch
        # inner_steps: sonst sieht eine Epoche, die zur Haelfte verworfen
        # wurde, wie eine halbierte Verlustkurve aus.
        st.data += d_sum / max(n_gezaehlt, 1) / len(ops)
        if want_phys:
            p = p_sum / (inner_steps * len(ops))
            st.phys = p if np.isnan(st.phys) else st.phys + p

    if g_n:
        st.grad_norm = g_sum / g_n
    st.dauer_s = time.time() - t_start
    if w_wall > 0.0:
        st.notes.append(
            "w_wall > 0, aber der Wandterm ist nicht kalibrierbar (Stufe 2 "
            "offen). Der Term laeuft als NaN mit und wird NICHT addiert.")
    return st


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Der Ladepfad -- importiert aus PINNmodulusTwo, nicht nachgebaut
# ---------------------------------------------------------------------------
_PINN_DIR = HERE.parent / "PINNmodulusTwo"


def _pinn_module(name: str):
    """``PINNmodulusTwo/<name>.py`` importieren.

    Genau wie ``resolve_device``: **importiert, nicht kopiert**. Die
    Normierungskonstanten, der Fourier-Tensor und die Treiberkanaele muessen
    dieselben sein wie im Basisprojekt, sonst vergleicht der Benchmark zwei
    verschiedene Datensaetze und nennt das eine Architekturaussage.
    """
    if str(_PINN_DIR) not in sys.path:
        sys.path.insert(0, str(_PINN_DIR))
    return __import__(name)


def layout_aus_bundle(bundle) -> GridLayout:
    """``derive_layout`` aus den Koordinaten des Buendels.

    ⚠ ``bundle.xn`` (float64), **nicht** ``op.xn``. Der ist float32, und die
    Aequidistanzpruefung in ``derive_layout`` laeuft mit ``rtol=1e-6``: gemessen
    bleiben in float32 nur 4e-7 Reserve, also die Haelfte. Das faellt nicht
    heute, sondern an dem Tag, an dem jemand die Koordinaten neu exportiert.
    """
    xyz = np.asarray(bundle.xn, dtype=np.float64) * float(bundle.L_ref)
    return gridmod.derive_layout(xyz)


def statics_aus_bundle(bundle, layout: GridLayout, *, coord_maps: bool,
                       device) -> M.StaticMaps:
    """Die 17 (bzw. 15) ortsfesten Karten aus den Materialdaten.

    ``lam`` steht **nicht** im Buendel -- dort liegt nur ``Fo``, in das es
    zusammen mit ``rho*Cp`` schon eingerechnet ist. Es aus ``Fo`` zurueck-
    zurechnen hiesse, durch ein ``+1e-30`` zu dividieren und das Ergebnis
    Materialdaten zu nennen. Also aus der Quelle geholt; die Zuordnung
    Region -> Schicht steht in ``materials.load_material_properties``
    (0 = cc, 1 = jr1c, 2 = g) und ist genau die, die ``data._grid_arrays``
    hineingeschrieben hat.
    """
    materials = _pinn_module("materials")
    layer = np.array(["cc", "jr1c", "g"], dtype=object)[
        np.asarray(bundle.region, dtype=np.int64)]
    props = materials.load_material_properties(layer=layer)
    return M.build_static_maps(
        layout, lam=props["lambda_tensor"], rho=bundle.rho, cp=bundle.Cp,
        device=device, coord_maps=coord_maps)


def op_tensoren(op, layout: GridLayout, device) -> OPTensors:
    """Ein ``data.OPData`` in die Gitterform bringen, einmal je Lauf.

    ``to_field`` bildet die letzte Achse ``(..., n_points)`` auf
    ``(..., nx, ny, nz)`` ab. ``Fo`` traegt seine 3x3 aber **hinten**
    (``(n_points, 3, 3)``), also wandert die Punktachse erst ans Ende und das
    Ergebnis danach zurueck -- sonst stuende der Tensor auf dem Kopf und
    ``anisotropic_laplacian`` griffe ``fo[..., 0, 0]`` aus einer Ortsachse.
    """
    def feld(a):
        return gridmod.to_field(
            torch.as_tensor(np.asarray(a), dtype=torch.float32), layout)

    fo_flat = torch.as_tensor(np.asarray(op.Fo), dtype=torch.float32)   # (n,3,3)
    fo = gridmod.to_field(fo_flat.permute(1, 2, 0), layout)             # (3,3,nx,ny,nz)
    fo = fo.permute(2, 3, 4, 0, 1).contiguous()                         # (nx,ny,nz,3,3)

    return OPTensors(
        op_id=op.op_id,
        tn_seq=feld(op.Tn).to(device),
        tn_ic=feld(op.Tn_ic).to(device),
        qsrc=feld(op.Qsrc).to(device),
        fo=fo.to(device),
        config=torch.as_tensor(np.asarray(op.config_feat),
                               dtype=torch.float32).to(device),
        forcing=torch.as_tensor(np.asarray(op.forcing_feat),
                                dtype=torch.float32).to(device),
        dtn=float(op.dtn),
        split_t=int(op.split_t),
        n_t=int(op.n_t),
    )


def lade_datensatz(args, device):
    """Buendel laden, Layout ableiten, Karten bauen, OPs in Gitterform.

    Zurueck kommt ``(bundle, train, val, layout, statics)``. Die Haltemenge
    wird mit ``build_op`` gegen die **Trainings**konstanten gebaut -- nichts
    wird nachgefittet, sonst waere sie keine Haltemenge mehr.
    """
    D = _pinn_module("data")
    t0 = time.time()
    bundle = D.load_ops(op_ids=list(args.ops), subsample_time=args.subsample)
    held = [D.build_op(o, bundle, subsample_time=args.subsample)
            for o in args.val_ops]

    layout = layout_aus_bundle(bundle)
    kw = modell_kwargs(args)
    statics = statics_aus_bundle(bundle, layout,
                                 coord_maps=kw["static"]["coord_maps"],
                                 device=device)

    train = [op_tensoren(o, layout, device) for o in bundle.ops]
    val = [op_tensoren(o, layout, device) for o in held]

    print(f"[daten] {len(train)} Trainings-OPs, {len(val)} Halte-OPs, "
          f"subsample={args.subsample} -> dt_n={train[0].dtn:.6g}, "
          f"{time.time() - t0:.1f}s")
    print(f"[gitter] {layout.describe().splitlines()[0]}")
    print(f"[normierung] T_mu={bundle.T_mu:.4g} T_sigma={bundle.T_sigma:.4g} C "
          f"L_ref={bundle.L_ref:.6g} m T_span_ref={bundle.T_span_ref:.6g} s")

    # Die CFL-Schranke VOR dem Lauf, nicht als Raetsel danach. Der FAHRPLAN
    # sagt zu einem weglaufenden Rollout: "entweder CFL (dann subsample_time:
    # 1) oder ein Vorzeichenfehler im Padding" -- diese Zeile entscheidet
    # zwischen den beiden, bevor Stunden verbrannt sind. Sie ist eine
    # RICHTGROESSE: der Kreuzterm steckt nicht drin, und das Netz ist nicht
    # der blanke explizite Stern. Deshalb eine Warnung und kein Abbruch.
    dt_max = phys.cfl_limit(layout, train[0].fo)
    dt_max_s = dt_max * float(bundle.T_span_ref)
    dt_s = train[0].dtn * float(bundle.T_span_ref)
    if train[0].dtn > dt_max:
        print(f"!! [CFL] dt_n={train[0].dtn:.6g} ({dt_s:.4g} s) liegt "
              f"{train[0].dtn / dt_max:.1f}x UEBER der expliziten Schranke "
              f"dt_max_n={dt_max:.6g} ({dt_max_s:.4g} s).\n"
              f"   Laeuft der Rollout weg, ist DAS die erste Erklaerung -- "
              f"nicht das Netz. Kleineres --subsample, oder pruefen, ob die "
              f"Materialdaten echt sind: ein synthetisches "
              f"material_properties/ macht das Problem viel steifer, als es "
              f"ist. Arm A (--no-physics) ist davon nicht betroffen.",
              file=sys.stderr)
    else:
        print(f"[CFL] dt_n={train[0].dtn:.6g} ({dt_s:.4g} s) unter der "
              f"Schranke dt_max_n={dt_max:.6g} ({dt_max_s:.4g} s).")
    if statics.dead:
        print(f"[karten] tot (konstant, auf 0 gezwungen): "
              f"{', '.join(statics.dead)}")
    return bundle, train, val, layout, statics


@torch.no_grad()
def triviale_latten(ops: list, T_sigma: float) -> dict:
    """Was man ohne jedes Modell erreicht. Die Latte unter der Latte.

    Ohne sie ist "8.95 C" nicht einzuordnen, und genau daran ist der Lauf vom
    22.09. fast vorbeigegangen: der beste Seed lag auf dem Niveau von
    "sage ueberall den Trainingsmittelwert".

    * ``mittelwert``  -- im z-normierten Raum ist der Trainingsmittelwert
      exakt 0, die MAE also ``mean|tn| * T_sigma``.
    * ``persistenz``  -- ``T(t) = T(0)``, die Anfangsbedingung eingefroren.
      Fuer ein traeges thermisches System ist das eine ERNSTHAFTE Latte, kein
      Strohmann.

    Ein Modell, das eine der beiden nicht schlaegt, hat nichts gelernt.
    """
    out = {}
    for op in ops:
        out[op.op_id] = {
            "mittelwert": float(op.tn_seq.abs().mean().item() * T_sigma),
            "persistenz": float(
                (op.tn_seq - op.tn_ic.unsqueeze(0)).abs().mean().item()
                * T_sigma),
        }
    return out


SEGMENTE = 6        # in wie viele Abschnitte die Trajektorie zerlegt wird
DRIFT_SCHWELLE = 1.5    # ab hier heisst "der Fehler waechst zum Ende" (O13)


@torch.no_grad()
def val_auswertung(net: M.GridCNN, ops: list, statics: M.StaticMaps, *,
                   lag1: int, lag2: int, clamp: float, T_sigma: float,
                   batch: OPBatch | None = None,
                   segmente: int = SEGMENTE) -> dict:
    """Freilaufender Fehler je Halte-OP -- **ueber die Trajektorie aufgeloest**.

    ``tn_seq`` ist ``(T - T_mu) / T_sigma``, der Versatz ``T_mu`` faellt in der
    Differenz also heraus und ``T_sigma`` ist der ganze Umrechnungsfaktor --
    kein zweiter Weg zu denselben Labels, der auseinanderdriften koennte.

    Freilaufend, nicht ein Schritt: gemessen wird, was das Modell allein
    erzeugt. Ein Ein-Schritt-Fehler sieht immer gut aus.

    Warum das nicht eine Zahl ist
    ------------------------------
    Am 22.09. stand ``OP06 6.57 C`` da, und der Plot ueber dieselbe
    Trajektorie zeigte **1.1 C in der Mitte und 15 C am Ende** -- Faktor 14,
    den der Mittelwert vollstaendig verdeckt. Ein Modell, das die erste
    Haelfte trifft und die zweite verliert, ist etwas ganz anderes als eines,
    das ueberall gleich daneben liegt, und beide haetten dieselbe MAE.

    Zurueck kommt je OP:

    ``mae`` / ``bias``
        ueber die ganze Trajektorie. **``bias`` ist vorzeichenbehaftet**: er
        beantwortet "zu warm oder zu kalt", und ohne ihn ist ein
        feldweiter Pegelfehler nicht von Streuung zu unterscheiden.
    ``mae_segmente`` / ``bias_segmente``
        dasselbe je Abschnitt, in Laufrichtung.
    ``drift``
        ``MAE(letzter Abschnitt) / MAE(gesamt)``. Nahe 1 heisst gleichmaessig
        verteilt; deutlich darueber heisst **O13**, der Fehler waechst zum
        Trajektorienende. Die eine Zahl, die am 22.09. gefehlt hat.

    **Gebatcht**, aus demselben Grund wie ``train_epoch``: der Rollout ist
    startlatenz-gebunden. Derselbe ``rollout_batched`` wie im Training -- keine
    zweite Fassung der Rekurrenz, die wegdriften koennte.
    """
    b = batch if batch is not None else stack_ops(ops)
    alle, _ = rollout_batched(net, b, statics, lag1=lag1, lag2=lag2,
                              clamp=clamp)
    out = {}
    for i, op in enumerate(b.ops):
        # Nur der eigene, gueltige Teil: kuerzere OPs sind bis n_max mit ihrer
        # letzten Treiberzeile aufgefuellt und rollen darueber hinaus weiter.
        traj = alle[:op.n_t, i]
        fehler = (traj - op.tn_seq) * T_sigma        # mit Vorzeichen, in C
        mae = float(fehler.abs().mean())

        grenzen = np.linspace(0, op.n_t, max(1, segmente) + 1)
        grenzen = np.unique(np.round(grenzen).astype(int))
        mae_seg, bias_seg = [], []
        for a, z in zip(grenzen[:-1], grenzen[1:]):
            teil = fehler[int(a):int(z)]
            if teil.numel() == 0:
                continue
            mae_seg.append(float(teil.abs().mean()))
            bias_seg.append(float(teil.mean()))
        out[op.op_id] = {
            "mae": mae, "bias": float(fehler.mean()),
            "mae_segmente": mae_seg, "bias_segmente": bias_seg,
            "drift": (mae_seg[-1] / mae) if (mae_seg and mae > 0)
                     else float("nan"),
        }
    return out


def val_mae(net: M.GridCNN, ops: list, statics: M.StaticMaps, *,
            lag1: int, lag2: int, clamp: float, T_sigma: float,
            batch: OPBatch | None = None) -> dict:
    """Nur die eine Zahl je OP -- duenner Aufsatz auf :func:`val_auswertung`.

    Bewusst keine zweite Rechnung: zwei Wege zu derselben MAE driften
    auseinander, und genau das hat dieses Projekt bei ``rollout`` schon
    einmal bewusst vermieden.
    """
    return {k: v["mae"] for k, v in val_auswertung(
        net, ops, statics, lag1=lag1, lag2=lag2, clamp=clamp,
        T_sigma=T_sigma, batch=batch).items()}


def median_ueber(punkte: list, anteil: float = 1.0 / 3.0) -> dict:
    """Median je OP ueber die letzten Messpunkte -- die berichtete Zahl.

    Der FAHRPLAN verbietet fuer den PINN ausdruecklich, die letzte Zeile eines
    Laufs abzulesen, und am 22.09. war genau das eine Lotterie: derselbe
    ``data``-Verlust, Faktor 5.8 in der val-MAE.

    Warum der Median und **nicht** das Beste: das Beste wird auf der
    Haltemenge ausgewaehlt, und die Haltemenge ist hier die ganze Messung
    (zwei OPs). Auf ihr das Minimum zu ziehen und dieselbe Zahl zu berichten
    ist eine Auswahl auf der Teststatistik -- optimistisch, und bei einer
    Streuung wie der vom 22.09. beliebig optimistisch. Das Beste wird
    trotzdem mitgeschrieben, als Diagnose, sauber so benannt.

    ``punkte`` ist ``[(epoch, {op_id: mae}), ...]`` in Laufreihenfolge.
    """
    if not punkte:
        return {}
    m = max(1, int(round(anteil * len(punkte))))
    letzte = punkte[-m:]
    op_ids = sorted(letzte[-1][1])
    return {k: float(np.median([p[1][k] for p in letzte if k in p[1]]))
            for k in op_ids}


def median_profil(punkte: list, anteil: float = 1.0 / 3.0) -> dict:
    """Dasselbe fuer die vollen Auswertungen aus :func:`val_auswertung`.

    Elementweiser Median ueber dieselben Messpunkte, aus denen der berichtete
    Mittelwert kommt -- damit Profil und Kopfzahl vom selben Stand reden und
    nicht aus zwei verschiedenen Epochen stammen.

    ``punkte`` ist ``[(epoch, {op_id: {...}}), ...]``.
    """
    if not punkte:
        return {}
    m = max(1, int(round(anteil * len(punkte))))
    letzte = punkte[-m:]
    out = {}
    for op_id in sorted(letzte[-1][1]):
        reihen = [p[1][op_id] for p in letzte if op_id in p[1]]
        n_seg = min(len(r["mae_segmente"]) for r in reihen)
        med = lambda schl: float(np.median([r[schl] for r in reihen]))  # noqa: E731
        med_seg = lambda schl: [                                        # noqa: E731
            float(np.median([r[schl][j] for r in reihen]))
            for j in range(n_seg)]
        out[op_id] = {"mae": med("mae"), "bias": med("bias"),
                      "drift": med("drift"),
                      "mae_segmente": med_seg("mae_segmente"),
                      "bias_segmente": med_seg("bias_segmente")}
    return out


def profil_zeilen(profil: dict, *, einzug: str = "") -> list:
    """Das Fehlerprofil als Textzeilen -- die Kurve, die sonst ein Plot waere.

    Am 22.09. musste ein Plot von Hand gebaut werden, um zu sehen, dass der
    Fehler auf OP06 von 1.1 C in der Mitte auf 15 C am Ende laeuft. Diese
    Zeilen stehen jetzt unter jedem Lauf.
    """
    zeilen = []
    for op_id, p in sorted(profil.items()):
        segs = "  ".join(f"{v:5.2f}" for v in p["mae_segmente"])
        zeilen.append(f"{einzug}{op_id}  MAE je Abschnitt: {segs}")
        bias = "  ".join(f"{v:+5.1f}" for v in p["bias_segmente"])
        marke = ("  <- O13, der Fehler waechst zum Ende"
                 if np.isfinite(p["drift"]) and p["drift"] > DRIFT_SCHWELLE
                 else "")
        zeilen.append(f"{einzug}{' ' * len(op_id)}  Bias je Abschnitt: {bias}"
                      f"   (gesamt {p['bias']:+.2f} C)")
        zeilen.append(f"{einzug}{' ' * len(op_id)}  Drift "
                      f"{p['drift']:.2f}x   Mittel {p['mae']:.2f} C{marke}")
    return zeilen


# ---------------------------------------------------------------------------
# Die drei Zahlen, die am 22.09. geraten statt hergeleitet waren
# ---------------------------------------------------------------------------
LAG1_S, LAG2_S = 1.0, 4.0       # die Lags in SEKUNDEN, nicht in Schritten
ROH_DT_S = 0.1                  # Rohabtastung der Daten
# Die Schwelle aus dem FAHRPLAN: "ein Unterschied unter ~1 C ist mit drei
# Seeds nicht lesbar". Liegt die Seed-Streuung darueber, kann ein Lauf nichts
# ranken -- egal wie gut sein Mittelwert aussieht.
LESBARKEIT_C = 1.0


def lags_aufloesen(subsample: int, lag1: int | None, lag2: int | None
                   ) -> tuple[int, int, str]:
    """``--lag1/--lag2`` aus einer **Zeit** ableiten, nicht aus einer Schrittzahl.

    5 und 20 Schritte sind bei ``--subsample 2`` genau 1 s und 4 s. Bei
    ``--subsample 10`` waeren dieselben Zahlen 5 s und 20 s -- also eine ganz
    andere Historie, ohne dass jemand etwas geaendert haette. Das ist die
    Sorte Fehler, die einen Vergleich still ungueltig macht.

    Von Hand gesetzte Werte gewinnen; ``--subsample 2`` reproduziert die
    Vorgabe 5/20 auf die Zahl genau.
    """
    dt_s = ROH_DT_S * max(1, int(subsample))
    l1 = int(lag1) if lag1 is not None else max(1, int(round(LAG1_S / dt_s)))
    l2 = int(lag2) if lag2 is not None else max(l1 + 1,
                                                int(round(LAG2_S / dt_s)))
    wie = "von Hand" if (lag1 is not None or lag2 is not None) else "aus der Zeit"
    return l1, l2, (f"lag1={l1} ({l1 * dt_s:.3g} s), lag2={l2} "
                    f"({l2 * dt_s:.3g} s) bei dt={dt_s:.3g} s [{wie}]")


# Das Fenster des POC vom 22.09. (subsample 10, dt = 1 s), in Sekunden. Es ist
# die einzige Fensterlaenge, fuer die bisher ein Ergebnis vorliegt.
TBPTT_POC_S = (4.0, 16.0)


def fensterwarnung(tbptt_start: int, tbptt: int, lag1: int, lag2: int,
                   subsample: int) -> str | None:
    """Erreicht das TBPTT-Fenster die Lags? Wenn nicht, sagen, was fehlt.

    In :func:`rollout_loss` kommt ``t2`` erst ab ``j >= lag2`` aus der eigenen
    Vorhersage, davor aus dem eingefrorenen Puffer. Ist ``k <= lag2``, laeuft
    **kein einziges Update** durch die Rueckkopplung ueber ``lag2`` -- das Netz
    lernt nie, was seine eigenen Vorhersagen von vor 4 s mit ihm anrichten.

    Genau das ist am 23.09. passiert: die Lags wurden auf Sekunden umgestellt,
    ``k`` nicht. Bei ``--subsample 10`` lag ``lag2 = 4`` im Fenster ``4->16``,
    bei ``--subsample 2`` liegt ``lag2 = 20`` hinter ``k = 16``. Der Befehl
    hiess "nur subsample und epochs aendern sich" -- und das Protokoll war
    trotzdem ein anderes.
    """
    dt_s = ROH_DT_S * max(1, int(subsample))
    k_poc = tuple(max(1, int(round(s / dt_s))) for s in TBPTT_POC_S)
    if tbptt <= lag2:
        was = (f"das Fenster k={tbptt} ({tbptt * dt_s:.3g} s) erreicht "
               f"lag2={lag2} ({lag2 * dt_s:.3g} s) NIE: kein Update laeuft "
               f"durch die Rueckkopplung ueber lag2.")
    elif tbptt_start <= lag1:
        was = (f"das Startfenster k={tbptt_start} erreicht lag1={lag1} "
               f"nicht: die ersten Epochen trainieren ohne die Rueckkopplung "
               f"ueber lag1.")
    else:
        return None
    return (f"!! [fenster] {was}\n   Dasselbe Fenster in Sekunden wie der "
            f"POC vom 22.09. ({TBPTT_POC_S[0]:g}->{TBPTT_POC_S[1]:g} s) waere "
            f"hier --tbptt-start {k_poc[0]} --tbptt {k_poc[1]}.")


def clamp_aufloesen(spec: str, ops: list, *, faktor: float, T_sigma: float
                    ) -> tuple[float, str]:
    """``--clamp`` aufloesen -- ``auto`` heisst: aus den Daten, nicht geerbt.

    ``--clamp 50`` kam aus ``PINNmodulusTwo`` und ist hier **keine harmlose
    Zahl**: bei ``T_sigma = 9.602 C`` sind das rund ±480 C um den Mittelwert.
    Ein Rollout, der dort anliegt, ist nicht ungenau, er ist weg -- und der
    ``[SATURATED]``-Zaehler meldet dann etwas, das laengst nichts mehr
    rettet.

    ``auto`` setzt die Schranke auf ein Vielfaches der groessten Auslenkung,
    die in den Labels ueberhaupt vorkommt. Darueber ist kein Zustand mehr
    physikalisch, und das Festhalten faellt frueh genug auf, um noch eine
    Aussage zu sein.
    """
    s = str(spec).strip().lower()
    if s in ("aus", "off", "none", "nein", "0", "0.0"):
        return 0.0, "AUS -- ein weglaufender Rollout wird nicht festgehalten"
    if s == "auto":
        groesst = max(float(o.tn_seq.abs().max().item()) for o in ops)
        wert = float(np.ceil(faktor * groesst))
        return wert, (f"auto = ceil({faktor:g} x max|tn|={groesst:.3f}) = "
                      f"{wert:g}  (±{wert * T_sigma:.0f} C um T_mu; "
                      f"die Labels reichen bis ±{groesst * T_sigma:.1f} C)")
    return float(s), f"von Hand: {float(s):g}  (±{float(s) * T_sigma:.0f} C um T_mu)"


def tbptt_bei(epoch: int, start: int, ende: int, epochs: int,
              anteil: float = 0.5) -> int:
    """Die Fensterlaenge dieser Epoche -- geometrisch wachsend, dann konstant.

    Kurz anfangen und wachsen lassen, weil ein langes Fenster auf einem noch
    untrainierten Netz nur Rauschen ueber ``k`` Schritte aufsummiert. Nach
    ``anteil`` der Epochen steht ``ende`` -- so laeuft die zweite Haelfte, aus
    der der berichtete Median kommt, vollstaendig auf der vollen Laenge und
    nicht auf einem wandernden Ziel.
    """
    start, ende = max(1, int(start)), max(1, int(ende))
    if start >= ende or epochs <= 1:
        return ende
    n = max(2, int(round(anteil * epochs)))
    if epoch >= n:
        return ende
    f = (epoch - 1) / (n - 1)
    return max(1, int(round(start * (ende / start) ** f)))


def fahre_einen_lauf(args, seed: int, device, daten, out_dir: Path,
                     latten: dict | None = None) -> dict:
    """Ein Seed: Netz bauen, Epochen fahren, Artefakte schreiben."""
    bundle, train, val, layout, statics = daten
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    kw = modell_kwargs(args)
    net = M.GridCNN(layout, **kw["net"]).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"\n[seed {seed}] {n_par} Parameter -> {out_dir}")

    latten = latten or {}
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # Ein LR-Plan, weil die spaeten Stuerme vom 22.09. (Seed 0, ep 38-50) wie
    # ein Optimierer aussehen, der bei konstant 1e-3 ein Becken wieder
    # verlaesst. eta_min > 0: ganz auf null zu fahren macht die letzten
    # Epochen zu einer Wiederholung derselben Zeile.
    plan = None
    if args.lr_plan == "cosine" and args.epochs > 1:
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    batch = stack_ops(train)
    val_batch = stack_ops(val)
    verlauf = []
    val_punkte, detail_punkte = [], []
    bestes = {"epoch": -1, "mittel": float("inf"), "mae": {}}

    for epoch in range(1, args.epochs + 1):
        k = tbptt_bei(epoch, args.tbptt_start, args.tbptt, args.epochs)
        st = train_epoch(net, batch, statics, opt,
                         inner_steps=args.inner_steps, batch_t=args.batch_t,
                         lag1=args.lag1, lag2=args.lag2, w_data=args.w_data,
                         w_phys=args.w_phys, w_wall=args.w_wall,
                         clamp=args.clamp, rng=rng, tbptt=k,
                         clip_grad=args.clip_grad)
        zeile = {"epoch": epoch, "data": st.data, "phys": st.phys,
                 "wall": st.wall, "saturated": st.saturated,
                 "saturated_max": st.saturated_max,
                 "spread_space": st.spread_space,
                 "spread_time": st.spread_time, "tbptt": st.tbptt,
                 "grad_norm": st.grad_norm, "dauer_s": st.dauer_s,
                 "uebersprungen": st.uebersprungen,
                 "lr": opt.param_groups[0]["lr"]}

        # Die val-MAE MITSCHREIBEN, nicht nur am Ende einmal nehmen. Am
        # 22.09. wurde Epoche 60 abgelesen, und das ist eine Lotterie: Seed 0
        # endete mit demselben data-Verlust wie Seed 2 und einer 5.8x
        # schlechteren val-MAE. Der FAHRPLAN verbietet das fuer den PINN
        # ausdruecklich ("Nie die letzte Zeile eines Laufs ablesen").
        #
        # Ein Rollout ueber die Haltemenge kostet, deshalb nicht jede Epoche.
        faellig = (epoch % max(1, args.val_every) == 0
                   or epoch == args.epochs or epoch == 1)
        if faellig:
            detail = val_auswertung(net, val, statics, lag1=args.lag1,
                                    lag2=args.lag2, clamp=args.clamp,
                                    T_sigma=bundle.T_sigma, batch=val_batch)
            mae = {k: v["mae"] for k, v in detail.items()}
            zeile["val_mae_C"] = mae
            zeile["val_detail"] = detail
            val_punkte.append((epoch, mae))
            detail_punkte.append((epoch, detail))
            mittel = float(np.mean(list(mae.values())))
            marke = ""
            if mittel < bestes["mittel"]:
                bestes = {"epoch": epoch, "mittel": mittel, "mae": mae}
                torch.save(net.state_dict(), out_dir / "model_best.pt")
                marke = "  <- bestes bisher"
            print(f"[seed {seed}] {st.line(epoch)}", flush=True)
            print(f"[seed {seed}]      val-MAE "
                  + "  ".join(_mit_latte(k, v, latten)
                              for k, v in sorted(mae.items()))
                  + f"  (Mittel {mittel:.4f} C){marke}", flush=True)
        else:
            print(f"[seed {seed}] {st.line(epoch)}", flush=True)
        verlauf.append(zeile)
        if plan is not None:
            plan.step()

    letzt_detail = val_auswertung(net, val, statics, lag1=args.lag1,
                                  lag2=args.lag2, clamp=args.clamp,
                                  T_sigma=bundle.T_sigma, batch=val_batch)
    letzte = {k: v["mae"] for k, v in letzt_detail.items()}
    if not val_punkte or val_punkte[-1][0] != args.epochs:
        val_punkte.append((args.epochs, letzte))
        detail_punkte.append((args.epochs, letzt_detail))
    median = median_ueber(val_punkte)
    profil = median_profil(detail_punkte)
    m_anz = max(1, int(round(len(val_punkte) / 3.0)))

    # Die berichtete Zahl steht OBEN und heisst, wie sie zustande kam. Die
    # beiden anderen stehen darunter, mit dem Vorbehalt, der zu ihnen gehoert.
    print(f"[seed {seed}] BERICHTET (Median ueber die letzten {m_anz} von "
          f"{len(val_punkte)} Messpunkten): "
          + "  ".join(_mit_latte(k, v, latten)
                      for k, v in sorted(median.items())))
    print(f"[seed {seed}]   bestes  ep {bestes['epoch']}: "
          + "  ".join(f"{k} {v:.4f}" for k, v in sorted(bestes["mae"].items()))
          + "   <- auf der Haltemenge ausgewaehlt, also optimistisch")
    print(f"[seed {seed}]   letztes ep {args.epochs}: "
          + "  ".join(f"{k} {v:.4f}" for k, v in sorted(letzte.items()))
          + "   <- eine Lotterie, nicht ablesen (FAHRPLAN)")

    # Das Profil ueber die Trajektorie. Ohne es ist "6.57 C" ein Mittelwert
    # ueber 1.1 C in der Mitte und 15 C am Ende -- siehe val_auswertung().
    if profil:
        n_seg = len(next(iter(profil.values()))["mae_segmente"])
        print(f"[seed {seed}] FEHLERPROFIL ({n_seg} gleich lange Abschnitte "
              f"in Laufrichtung, Grad C):")
        for z in profil_zeilen(profil, einzug=f"[seed {seed}]   "):
            print(z)

    torch.save(net.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(verlauf, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"seed": seed, "konfiguration": konfigurationsname(args),
         "parameter": n_par, "epochs": args.epochs,
         "protokoll": protokollname(args),
         "val_mae_C_berichtet": median,
         "val_mae_C_median_ueber": m_anz,
         "val_mae_C_letzte": letzte,
         "val_mae_C_bestes": bestes["mae"],
         "bestes_epoch": bestes["epoch"],
         "fehlerprofil": profil,
         "triviale_latten_C": latten}, indent=2))
    # Berichtet wird der MEDIAN. Bestes und letztes stehen in metrics.json --
    # als Diagnose, nicht als Ergebnis. Begruendung in median_ueber().
    return (median or letzte), profil


def _mit_latte(op_id: str, wert: float, latten: dict) -> str:
    """``OP06 8.9500 (1.16x Latte)`` -- die Zahl und ihr Massstab in einem.

    Am 22.09. stand "8.95" allein da und sah brauchbar aus. Neben der Latte
    von 7.70 steht sofort, dass es das nicht war. Die Latte gehoert in
    dieselbe Zeile, nicht in einen Kopf zwanzig Bildschirme weiter oben.
    """
    lat = latten.get(op_id) or {}
    if not lat:
        return f"{op_id} {wert:.4f}"
    beste = min(lat.values())
    if not np.isfinite(beste) or beste <= 0:
        return f"{op_id} {wert:.4f}"
    q = wert / beste
    return f"{op_id} {wert:.4f} ({q:.2f}x Latte{'' if q < 1.0 else ' ⚠'})"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GridCNN trainieren -- Ein-Schritt gegen eine "
                    "eingefrorene, frei laufende Trajektorie.")
    p.add_argument("--cache", type=Path, default=Path("data_cache"),
                   help="Verzeichnis mit den OP-Buendeln")
    p.add_argument("--device", default="ask",
                   help="ask (fragt nach, Vorgabe) | auto | cpu | cuda | "
                        "cuda:N. 'cuda' faellt NICHT still auf die CPU zurueck "
                        "-- ein stiller Rueckfall auf einer GPU-Maschine ist in "
                        "einem Log sehr leicht zu uebersehen und kostet den "
                        "ganzen Lauf an Tempo. Geteilt mit PINNmodulusTwo "
                        "ueber device_utils.py.")
    p.add_argument("--ops", nargs="+", default=None,
                   help="Trainings-OPs. Vorgabe: op_registry.DEFAULT_TRAIN_OPS "
                        "-- die Liste kommt aus der Registry und nicht aus dem "
                        "Gedaechtnis")
    p.add_argument("--val-ops", nargs="+", default=None,
                   help="Halte-OPs, gegen die gemessen wird. Vorgabe: "
                        "op_registry.DEFAULT_VAL_OPS (OP06, OP09)")
    p.add_argument("--subsample", type=int, default=2,
                   help="jeder N-te Rohschritt (Roh-dt = 0.1 s). 2 -> dt = 0.2 s")
    p.add_argument("--artifacts-dir", type=Path, default=Path("GridCNN/artifacts"),
                   help="Wurzel fuer die Artefakte. JEDER Seed bekommt darunter "
                        "ein eigenes Verzeichnis -- ohne das schreiben "
                        "gleichzeitige Laeufe in dieselbe model.pt, und das "
                        "scheitert nicht, es mischt")
    p.add_argument("--val-every", type=int, default=5,
                   help="val-MAE alle N Epochen messen (Epoche 1 und die "
                        "letzte immer). Berichtet wird der MEDIAN ueber das "
                        "letzte Drittel der Messpunkte: der FAHRPLAN "
                        "verbietet 'die letzte Zeile ablesen', und das Beste "
                        "waere eine Auswahl auf der Haltemenge")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--inner-steps", type=int, default=100,
                   help="Updates je OP und Epoche auf der eingefrorenen "
                        "Trajektorie. Hunderte, nicht Zehntausende -- sonst "
                        "wird der Puffer schal. Mit --tbptt k kostet jedes "
                        "Update k Schritte: was konstant gehoert, ist "
                        "inner_steps x k.")
    p.add_argument("--tbptt", type=int, default=16,
                   help="Fensterlaenge k je Update -- Stufe 5, truncated "
                        "BPTT. Die EINE Aenderung, die den Bruch zwischen "
                        "'ein Schritt trainiert' und 'achttausend gemessen' "
                        "schliesst. 1 reproduziert den Lauf vom 22.09.")
    p.add_argument("--tbptt-start", type=int, default=4,
                   help="Fensterlaenge in Epoche 1; waechst geometrisch bis "
                        "--tbptt und steht ab der halben Laufzeit. Ein langes "
                        "Fenster auf einem untrainierten Netz summiert nur "
                        "Rauschen. Gleich --tbptt heisst: kein Curriculum.")
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="Schranke fuer clip_grad_norm_. 0 schaltet ab. Die "
                        "data-Ausschlaege auf 2500 bei 11 k Parametern am "
                        "22.09. sind explodierende Gradienten -- der "
                        "billigste Eingriff, den es gibt. Die Norm wird auch "
                        "ohne Schranke gemessen und gemeldet.")
    p.add_argument("--lr-plan", choices=("cosine", "konstant"),
                   default="cosine",
                   help="cosine faehrt auf 5 %% der Start-LR herunter. "
                        "konstant ist der Zustand vom 22.09., und die spaeten "
                        "Stuerme (Seed 0, ep 38-50) sehen genau danach aus.")
    p.add_argument("--batch-t", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16,
                   help="Kanaele je Block. 16 ist die Voreinstellung "
                        "(11 427 Parameter); die Praesentation sah 64 vor "
                        "(mit --blocks 4 dann 137 923).")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--no-physics", action="store_true",
                   help="Konfiguration A der Ablation: reine Blackbox, "
                        "f = g_theta ohne den Physik-Term in der Architektur.")
    p.add_argument("--no-coord-maps", action="store_true",
                   help="Konfiguration D der Ablation: B ohne die zwei "
                        "Koordinatenkarten (y/z), also 42 statt 44 Kanaele. "
                        "Prueft, ob das Netz seine Ortsstruktur aus den "
                        "Materialkarten begruenden kann -- die Behauptung aus "
                        "README Sec. 2b, der model.py:193 widerspricht. "
                        "Route R8 in BENCHMARK.md.")
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--w-phys", type=float, default=0.0,
                   help="Konfiguration C: der Physik-Strafterm OBENDRAUF. "
                        "0 heisst Konfiguration B.")
    p.add_argument("--w-wall", type=float, default=0.0,
                   help="Braucht Stufe 2; laeuft sonst als NaN mit.")
    p.add_argument("--lag1", type=int, default=None,
                   help="Anker-Lag in SCHRITTEN. Vorgabe: aus %.3g s "
                        "abgeleitet, also 5 bei --subsample 2. Ohne die "
                        "Ableitung waeren 5 Schritte je nach --subsample eine "
                        "andere Historie." % LAG1_S)
    p.add_argument("--lag2", type=int, default=None,
                   help="Zweites Lag in SCHRITTEN. Vorgabe: aus %.3g s "
                        "abgeleitet, also 20 bei --subsample 2." % LAG2_S)
    p.add_argument("--clamp", default="auto",
                   help="Haelt einen weglaufenden Rollout fest. 'auto' leitet "
                        "die Schranke aus den Labels ab (siehe "
                        "--clamp-faktor), 'aus' schaltet ab, eine Zahl setzt "
                        "sie von Hand. Die geerbte 50 sind ±480 C und fangen "
                        "nichts ab, was noch zu retten waere. Jedes "
                        "Festhalten wird als [SATURATED] gemeldet.")
    p.add_argument("--clamp-faktor", type=float, default=3.0,
                   help="Vielfaches der groessten Auslenkung in den Labels, "
                        "das '--clamp auto' als Schranke setzt.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1,
                   help="Zahl der Seeds; die Schleife laeuft. Ein Seed ist "
                        "keine Streuung -- am 22.09. lagen bester und "
                        "schlechtester Seed um Faktor 5.8 auseinander. Unter "
                        "3 wird gewarnt.")
    return p


def profil_aus_checkpoint(pfad: Path, layout, net_kwargs: dict, val: list,
                          statics: M.StaticMaps, *, lag1: int, lag2: int,
                          clamp: float, T_sigma: float, device=None) -> dict:
    """Ein gespeichertes ``model.pt`` nachmessen, ohne nachzutrainieren.

    Am 23.09. lief der Lauf auf voller Aufloesung mit einem ``train.py`` von
    VOR dem Fehlerprofil: drei gueltige Seeds, aber kein ``drift`` und kein
    Bias. Die Gewichte liegen noch da -- das Profil ist eine Minute Rollout,
    nicht drei Stunden Training.

    Derselbe :func:`val_auswertung` wie im Lauf, also dieselbe Zahl: die MAE
    hier muss die Zeile ``letztes ep ...`` aus dem Log treffen. Tut sie das
    nicht, passen Checkpoint und Flags nicht zusammen.
    """
    net = M.GridCNN(layout, **net_kwargs)
    zustand = torch.load(pfad, map_location=device or "cpu")
    net.load_state_dict(zustand)
    if device is not None:
        net = net.to(device)
    return val_auswertung(net, val, statics, lag1=lag1, lag2=lag2,
                          clamp=clamp, T_sigma=T_sigma)


def profil_ueber_seeds(profile: list) -> dict:
    """Die Fehlerprofile mehrerer Seeds zu einem Median zusammenziehen.

    Derselbe Median wie ueberall sonst in diesem Modul, nur eine Achse
    weiter aussen. Ein Seed weniger oder mehr darf das Profil nicht kippen --
    am 22.09. lagen bester und schlechtester Seed um Faktor 5.8 auseinander.
    """
    vorhanden = [p for p in profile if p]
    if not vorhanden:
        return {}
    op_ids = sorted(set().union(*(set(p) for p in vorhanden)))
    out = {}
    for op_id in op_ids:
        reihen = [p[op_id] for p in vorhanden if op_id in p]
        if not reihen:
            continue
        n_seg = min(len(r["mae_segmente"]) for r in reihen)
        out[op_id] = {
            "mae": float(np.median([r["mae"] for r in reihen])),
            "bias": float(np.median([r["bias"] for r in reihen])),
            "drift": float(np.median([r["drift"] for r in reihen])),
            "mae_segmente": [float(np.median([r["mae_segmente"][j]
                                              for r in reihen]))
                             for j in range(n_seg)],
            "bias_segmente": [float(np.median([r["bias_segmente"][j]
                                               for r in reihen]))
                              for j in range(n_seg)],
        }
    return out


def zusammenfassung(alle: dict, latten: dict,
                    profil: dict | None = None) -> tuple[list, dict]:
    """Die Schlusstafel und das Verdikt -- als Text, nicht als ``print``.

    Am 22.09. musste man drei Dokumente lesen, um zu sehen, dass
    ``27.11 ± 22.26 C`` kein Ergebnis ist. Das gehoert unter den Lauf, in
    denselben Bildschirm, und es muss pruefbar sein, ohne dass eine GPU
    dreissig Minuten rechnet -- deshalb eine Funktion und kein Block in
    ``main``.

    Zwei Zahlen entscheiden, und beide stehen in ``zusammenfassung.json``:

    * **Guete** ``= val-MAE / beste triviale Latte``. Unter 1.00 hat das Netz
      etwas gelernt, was ueber Raten hinausgeht; darueber nicht. Am 22.09.
      waeren das 8.95 / 7.7 = 1.16x gewesen, und niemand hat es gesehen.
    * **Lesbarkeit**: liegt die Seed-Streuung ueber ``LESBARKEIT_C``, kann der
      Lauf **nichts ranken** -- auch dann nicht, wenn sein Mittelwert gut
      aussieht. Die Streuung ist der Befund, nicht der Mittelwert.

    ``alle`` ist ``{op_id: [mae je Seed]}``, ``latten`` das Ergebnis von
    :func:`triviale_latten`.
    """
    zeilen, guete, streuungen = [], {}, {}
    for op_id, werte in sorted(alle.items()):
        lat = latten.get(op_id) or {}
        beste_latte = min(lat.values()) if lat else float("inf")
        a = np.asarray(werte, dtype=float)
        # ddof=1: die Streuung EINER Stichprobe, nicht der Grundgesamtheit.
        # Bei drei Seeds ist der Unterschied 22 %, und die Zahl wird gegen
        # eine ~1 C-Latte gelesen.
        sd = float(a.std(ddof=1)) if a.size > 1 else float("nan")
        streuungen[op_id] = sd
        zeilen.append(f"  val-MAE {op_id}: {a.mean():.4f} +- {sd:.4f} C   "
                      f"({', '.join(f'{v:.4f}' for v in werte)})")
        if lat:
            wie_viele = int((a < beste_latte).sum())
            guete[op_id] = (float(a.mean() / beste_latte), wie_viele, a.size)
            zeilen.append(f"      triviale Latte {beste_latte:.4f} C "
                          f"(Mittelwert {lat['mittelwert']:.4f} / Persistenz "
                          f"{lat['persistenz']:.4f}) -- "
                          f"{wie_viele}/{a.size} Seed(s) unterbieten sie"
                          + ("" if wie_viele else "  ⚠ KEINER"))

    zeilen.append("")
    zeilen.append("-" * 70)
    zeilen.append("[verdikt] Guete = val-MAE / beste triviale Latte. "
                  "< 1.00 heisst gelernt.")
    lesbar = bool(guete)
    for op_id, (q, wie_viele, n) in sorted(guete.items()):
        sd = streuungen.get(op_id, float("nan"))
        zeilen.append(f"          {op_id}  {q:.2f}x   {wie_viele}/{n} "
                      f"Seed(s) unter der Latte   Seed-Streuung {sd:.2f} C")
        if np.isfinite(sd) and sd > LESBARKEIT_C:
            lesbar = False
    if not guete:
        zeilen.append("          keine Latten -- ohne sie ist keine val-MAE "
                      "einzuordnen.")
    elif not lesbar:
        zeilen.append(
            f"          ⚠ KEIN ERGEBNIS. Die Seed-Streuung liegt ueber der "
            f"Lesbarkeitsschwelle von ~{LESBARKEIT_C:g} C. Dieser Lauf kann "
            f"NICHTS ranken -- weder B gegen A noch sonst etwas. Erst die "
            f"Streuung herunterbekommen, dann vergleichen.")
    elif all(q < 1.0 for q, _, _ in guete.values()):
        zeilen.append("          ERGEBNIS: alle OPs unter der Latte, und die "
                      "Streuung ist lesbar. Das traegt einen Vergleich.")
    else:
        zeilen.append("          ERGEBNIS, aber ein negatives: mindestens ein "
                      "OP liegt auf oder ueber der Latte. Dort hat das Modell "
                      "nichts gelernt, was ueber Raten hinausgeht -- und die "
                      "Streuung ist lesbar, der Befund also echt.")

    # Das Profil ueber die Trajektorie gehoert NEBEN das Verdikt, nicht in
    # einen Anhang: ein Mittelwert von 6.57 C ueber 1.1 C in der Mitte und
    # 15 C am Ende ist eine andere Aussage als 6.57 C ueberall.
    profil = profil or {}
    driften = {}
    if profil:
        zeilen.append("")
        zeilen.append("[profil] Fehler ueber die Trajektorie (Median ueber "
                      "die Seeds):")
        zeilen.extend(profil_zeilen(profil, einzug="          "))
        driften = {k: v["drift"] for k, v in profil.items()}
        schlimm = {k: v for k, v in driften.items()
                   if np.isfinite(v) and v > DRIFT_SCHWELLE}
        if schlimm:
            zeilen.append(
                f"          ⚠ Der Fehler ist NICHT gleichmaessig verteilt: "
                f"{', '.join(f'{k} {v:.2f}x' for k, v in sorted(schlimm.items()))}. "
                f"Das ist O13 -- der Spaetfehler. Die Kopfzahl oben ist ein "
                f"Mittelwert ueber einen guten Anfang und ein schlechtes Ende, "
                f"und Arm A hat keinen dissipativen Term, der den Pegel "
                f"zurueckholt (README Sec. 4).")
        else:
            zeilen.append("          Der Fehler ist ueber die Trajektorie "
                          "gleichmaessig verteilt -- kein Spaetfehler.")
    return zeilen, {"guete": {k: v[0] for k, v in guete.items()},
                    "seed_streuung_C": streuungen, "lesbar": lesbar,
                    "drift": driften}


def protokollname(args) -> str:
    """Wie trainiert wurde, in einer Zeile -- gehoert in JEDES Log.

    Ohne sie steht in einem halben Jahr eine val-MAE in einer Tabelle und
    niemand weiss, ob sie mit k=1 oder k=16 entstanden ist. Das ist derselbe
    Fehler wie eine nackte ``[SATURATED] 88248`` ohne Bezugsgroesse.
    """
    # k in SEKUNDEN daneben: am 23.09. stand in zwei Logs dasselbe "k=4->16",
    # einmal 4->16 s (subsample 10) und einmal 0.8->3.2 s (subsample 2).
    dt_s = ROH_DT_S * max(1, int(args.subsample))
    k_text = (f"k={args.tbptt_start}->{args.tbptt} "
              f"({args.tbptt_start * dt_s:.3g}->{args.tbptt * dt_s:.3g} s)"
              if args.tbptt_start != args.tbptt
              else f"k={args.tbptt} ({args.tbptt * dt_s:.3g} s)")
    teile = [k_text,
             f"clip={args.clip_grad:g}" if args.clip_grad > 0 else "clip=aus",
             f"lr={args.lr:g}/{args.lr_plan}",
             f"inner={args.inner_steps}x{args.batch_t}",
             f"subsample={args.subsample}"]
    return " ".join(teile)


def resolve_device(spec: str):
    """Geraet waehlen -- **importiert** aus ``PINNmodulusTwo``, nicht kopiert.

    Dieselbe Regel wie fuer ``data.py`` und ``op_metrics.py``: eine Kopie
    driftet weg. Vier Modi, und der Unterschied ist, wer entscheidet -- die
    Begruendung steht in ``device_utils.py`` und gilt hier unveraendert.

    Faellt ``device_utils`` aus (keine Schwesterprojekt-Checkout), wird die CPU
    genommen und das gesagt, statt zu scheitern: die Mechanik dieses Moduls
    laeuft ohne CUDA vollstaendig.
    """
    try:
        sys.path.insert(0, str(HERE.parent / "PINNmodulusTwo"))
        import device_utils
    except ImportError:
        print(f"[device] PINNmodulusTwo/device_utils.py nicht gefunden, "
              f"--device {spec} ignoriert -> cpu", file=sys.stderr)
        return torch.device("cpu")
    return device_utils.resolve_device(spec)


def modell_kwargs(args) -> dict:
    """Die Ablationsflags in die Argumente von ``model.py`` uebersetzen.

    Steht als eigene Funktion da, obwohl ``main`` sie heute nur ausgibt: der
    Ladepfad fehlt noch (FAHRPLAN, „Was fehlt"), und ein Flag, das bis dahin
    nirgends ankommt, ist ein Flag, das beim Anschliessen vergessen wird. So ist
    die Uebersetzung schon jetzt an einer Stelle und pruefbar -- der Ladepfad
    ruft sie spaeter nur noch auf.

    ``--no-coord-maps`` ist **Konfiguration D**: B ohne die zwei y/z-Karten.
    Die Breite muss an ZWEI Stellen zusammenpassen -- an den ``StaticMaps`` und
    an der ersten Faltung -- und genau deshalb kommen beide aus diesem einen
    Aufruf.
    """
    return {
        "static": {"coord_maps": not args.no_coord_maps},
        "net": {
            "width": args.width,
            "blocks": args.blocks,
            "use_physics": not args.no_physics,
            "n_static": (M.CH_STATIC_OHNE_KOORD if args.no_coord_maps
                         else M.CH_STATIC),
        },
    }


def konfigurationsname(args) -> str:
    """A / B / C / D aus der Ablationstabelle im Fahrplan, als ein Wort."""
    if args.no_physics:
        return "A (Blackbox, ohne Physik in der Architektur)"
    if args.no_coord_maps:
        return "D (B ohne die zwei Koordinatenkarten)"
    if args.w_phys > 0.0:
        return f"C (B plus Physik-Strafterm, w_phys={args.w_phys})"
    return "B (Physik in der Architektur, kein Strafterm)"


def main(argv: list | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.seeds < 3:
        print("!! --seeds < 3. Ein Seed ist keine Streuung: die gemessene "
              "Seed-Streuung ist 0.518 / 0.882 C, auf OP06 bis 1.63 C. Ein "
              "Vergleich zweier Konfigurationen braucht eine Seed-Schleife.",
              file=sys.stderr)
    if not args.cache.exists():
        print(f"!! Kein Cache unter {args.cache}. Auf der Rechenmaschine "
              f"liegt er unter data_cache/; hier im Repo liegt er nicht.",
              file=sys.stderr)
        return 2

    # Erst hier, nicht im Argparser: der zieht sonst PINNmodulusTwo auf den
    # sys.path, sobald irgendein Test nur die Flags lesen will.
    reg = _pinn_module("op_registry")
    if args.ops is None:
        args.ops = list(reg.DEFAULT_TRAIN_OPS)
    if args.val_ops is None:
        args.val_ops = list(reg.DEFAULT_VAL_OPS)

    device = resolve_device(args.device)
    if device.type == "cuda":
        # Feste Eingangsformen ueber den ganzen Lauf -- cudnn darf einmal
        # suchen und sich den besten Algorithmus merken. Bei wechselnden Formen
        # waere es schaedlich; hier sind sie (B, 44, 11, 11) und bleiben es.
        torch.backends.cudnn.benchmark = True
        print("[t4] cudnn.benchmark an (feste Eingangsformen). TF32 NICHT "
              "gesetzt: das ist Ampere und neuer, die T4 ist Turing (sm_75).",
              flush=True)

    # Die Lags VOR dem Laden: sie haengen nur an --subsample, und sie gehoeren
    # in den Kopf des Logs, nicht in eine Fussnote.
    args.lag1, args.lag2, lag_text = lags_aufloesen(
        args.subsample, args.lag1, args.lag2)

    kw = modell_kwargs(args)
    print(f"[konfiguration] {konfigurationsname(args)}")
    print(f"[protokoll] {protokollname(args)}")
    if args.tbptt <= 1:
        print("!! --tbptt 1: der Gradient laeuft EINEN Schritt weit, gemessen "
              "werden Tausende. Das ist der Pfad vom 22.09., und er hat kein "
              "Ergebnis geliefert. Absicht? Dann gut -- sonst --tbptt 16.",
              file=sys.stderr)
    print(f"[lags] {lag_text}")
    warnung = fensterwarnung(args.tbptt_start, args.tbptt, args.lag1,
                             args.lag2, args.subsample)
    if warnung and args.tbptt > 1:
        print(warnung, file=sys.stderr)
    print(f"[modell] {kw['net']['n_static']} statische Karten -> "
          f"{M.CH_STATE + kw['net']['n_static'] + M.CH_DRIVER} Eingangskanaele, "
          f"width={kw['net']['width']} blocks={kw['net']['blocks']}")
    if args.w_wall > 0.0:
        print("!! --w-wall > 0, aber der Wandterm haengt an den vier "
              "Cache-Groessen aus Stufe 2. Ist der Cache aelter, wirft "
              "_wall_ghost -- das ist Absicht, ein geratenes U waere "
              "schlimmer.", file=sys.stderr)
    else:
        print("[wand] AUS -- dieser Lauf ist adiabat. Das ist eine ABLATION, "
              "keine Latte, und darf auch nicht als eine zitiert werden "
              "(FAHRPLAN, 'Was fehlt').")

    # Einmal laden, ueber alle Seeds hinweg. Die Daten haengen nicht am Seed,
    # und sie je Seed neu zu bauen kostet Minuten und aendert nichts.
    daten = lade_datensatz(args, device)
    bundle, _train, _val = daten[0], daten[1], daten[2]

    # Die Latte unter der Latte, VOR dem ersten Lauf. Ein Modell, das sie
    # nicht schlaegt, hat nichts gelernt -- und ohne sie ist keine val-MAE
    # einzuordnen.
    latten = triviale_latten(_val, bundle.T_sigma)
    print("[latten] was OHNE Modell erreichbar ist (val-MAE in C):")
    for op_id, v in sorted(latten.items()):
        print(f"           {op_id}: Mittelwert-Vorhersage {v['mittelwert']:.4f}"
              f"   Persistenz T(t)=T(0) {v['persistenz']:.4f}"
              f"   -> Latte {min(v.values()):.4f}")
    print("           Wer die nicht unterbietet, hat nichts gelernt.")

    # Die Schranke aus den Daten, nicht aus PINNmodulusTwo geerbt. Erst hier,
    # weil sie die Labels braucht.
    args.clamp, clamp_text = clamp_aufloesen(
        args.clamp, _train + _val, faktor=args.clamp_faktor,
        T_sigma=bundle.T_sigma)
    print(f"[clamp] {clamp_text}")

    wurzel = args.artifacts_dir / konfigurationsname(args).split()[0]
    alle: dict[str, list] = {}
    profile: list = []
    for n in range(args.seeds):
        seed = args.seed + n
        mae, profil = fahre_einen_lauf(args, seed, device, daten,
                                       wurzel / f"seed{seed}", latten=latten)
        profile.append(profil)
        for op_id, v in mae.items():
            alle.setdefault(op_id, []).append(v)

    ueber_seeds = profil_ueber_seeds(profile)
    zeilen, kennzahlen = zusammenfassung(alle, latten, profil=ueber_seeds)
    print(f"\n{'=' * 70}")
    print(f"{konfigurationsname(args)}  --  {args.seeds} Seed(s), "
          f"{args.epochs} Epochen")
    print(f"Protokoll: {protokollname(args)}")
    print("Berichtet ist je Seed der Median ueber das letzte Drittel der "
          "Messpunkte.")
    for z in zeilen:
        print(z)
    if args.seeds < 3:
        print("  ⚠ unter drei Seeds ist das +- keine Streuung, sondern Zierde.")

    zus = wurzel / "zusammenfassung.json"
    zus.write_text(json.dumps(
        {"konfiguration": konfigurationsname(args),
         "protokoll": protokollname(args), "seeds": args.seeds,
         "epochs": args.epochs, "ops": list(args.ops),
         "val_ops": list(args.val_ops), "val_mae_C": alle,
         "fehlerprofil": ueber_seeds, "triviale_latten_C": latten,
         "clamp": float(args.clamp), "lag1": args.lag1, "lag2": args.lag2,
         "T_sigma": float(bundle.T_sigma), **kennzahlen}, indent=2))
    print(f"  -> {zus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
