#!/usr/bin/env python3
"""Die Trainingsschleife -- Ein-Schritt gegen eine eingefrorene Trajektorie.

Die eine Eigenschaft, die man hier nicht kaputtmachen darf
-----------------------------------------------------------
Je Epoche und Betriebspunkt wird die **eigene** Trajektorie einmal unter
``torch.no_grad()`` frei ausgerollt, gesaet nur von der gemessenen
Anfangsbedingung, und dann eingefroren. Erst darauf laufen ``inner_steps``
Ein-Schritt-Updates gegen die Labels.

**Das ist kein Teacher Forcing**, und es wird oft falsch erzaehlt. Der
Unterschied zwischen Training und Auswertung ist nicht der Eingangszustand --
beide rollen frei. Er ist::

    ================  ==========================  ===============
                      Training                    Auswertung
    ================  ==========================  ===============
    Trajektorie       eingefroren, je Epoche neu  live
    Labels            ja, als Ziel                nein
    Gradient          nur EIN Schritt             keiner
    ================  ==========================  ===============

**Daraus folgt der ganze Hebel von Stufe 5.** Weil der Gradient die Historie
nie ueberquert, kann er den Spaetfehler (O13) strukturell nicht erreichen --
egal wie lange man trainiert. Truncated BPTT ist die einzige Aenderung, die
daran etwas aendert. Wer hier optimiert, darf diese Eigenschaft nicht
versehentlich wegraeumen; ein Test haelt sie fest.

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
    spread_space: float = 0.0
    spread_time: float = 0.0
    n_ops: int = 0
    notes: list = field(default_factory=list)

    def line(self, epoch: int) -> str:
        w = "  --  " if np.isnan(self.wall) else f"{self.wall:.4g}"
        p = "  --  " if np.isnan(self.phys) else f"{self.phys:.4g}"
        s = (f"ep {epoch:>4d} | data {self.data:.5g} | phys {p} | wall {w} "
             f"| Streuung Ort {self.spread_space:.3f} Zeit {self.spread_time:.3f}")
        if self.saturated:
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
    """Platzhalter fuer den kalibrierten Wandterm -- Stufe 2 fehlt noch.

    Die Bruecke zwischen entdimensioniert und SI steht in ``solve.py`` und wird
    hier nicht ein zweites Mal geschrieben. Solange ``U(V_dot)`` nicht
    kalibriert ist, gibt es nichts anzuschliessen.
    """
    raise NotImplementedError(
        "Der Wandterm braucht U(V_dot), und das braucht q_solid_to_fluid, "
        "mdot, cp_fluid und fluid_out_temp im Buendel -- Stufe 2 des "
        "Fahrplans. Bis dahin laeuft das Training adiabat, und das ist eine "
        "Ablation, keine Physik-Latte.")


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
                rng: np.random.Generator) -> EpochStats:
    """**Einmal** alle OPs ausrollen, einfrieren, dann je OP ``inner_steps`` Updates.

    Die Zweiteilung ist Absicht und steht im Modulkopf:

    * **Der Rollout ist gebatcht.** Er laeuft unter ``no_grad``, aendert also
      nichts am Experiment -- nur an der Zeit, die er auf einer T4 kostet.
    * **Die innere Schleife bleibt je OP.** Alle OPs in einen Optimiererschritt
      zu ziehen machte den Gradienten leiser, und das waere eine Aenderung am
      Optimierer. Geschwindigkeit ja, Experiment anfassen nein.
    """
    ops = batch.ops
    st = EpochStats()
    st.n_ops = len(ops)
    want_phys = w_phys > 0.0

    alle, sat = rollout_batched(net, batch, statics, lag1=lag1, lag2=lag2,
                                clamp=clamp)
    st.saturated = sat

    for i, op in enumerate(ops):
        traj = alle[:op.n_t, i]          # nur der eigene, gueltige Teil
        s_o, s_t = spread_ratios(traj, op.tn_seq)
        st.spread_space += s_o / len(ops)
        st.spread_time += s_t / len(ops)

        # Labels nur bis split_t. Dahinter liegt das Fenster, das
        # op_metrics als in-time ausgehalten berichtet -- es zu fitten
        # machte aus einer ausgehaltenen Zahl eine Trainingszahl.
        hoch = max(int(op.split_t) - 1, 1)
        d_sum = p_sum = 0.0
        for _ in range(inner_steps):
            # t startet bei 0, das Ziel ist t+1. Zeile 0 ist die auferlegte
            # Anfangsbedingung und wird nie vorhergesagt, nur benutzt.
            idx = torch.as_tensor(rng.integers(0, hoch, size=batch_t),
                                  device=traj.device)
            loss = w_data * data_loss(net, op, statics, traj, idx,
                                      lag1=lag1, lag2=lag2)
            d_sum += float(loss.detach())
            if want_phys:
                lp, _ = physics_loss(net, op, statics, traj, idx,
                                     lag1=lag1, lag2=lag2)
                loss = loss + w_phys * lp
                p_sum += float(lp.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        st.data += d_sum / (inner_steps * len(ops))
        if want_phys:
            p = p_sum / (inner_steps * len(ops))
            st.phys = p if np.isnan(st.phys) else st.phys + p

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
def val_mae(net: M.GridCNN, ops: list, statics: M.StaticMaps, *,
            lag1: int, lag2: int, clamp: float, T_sigma: float) -> dict:
    """Freilaufende MAE je Halte-OP, in Grad Celsius.

    ``tn_seq`` ist ``(T - T_mu) / T_sigma``, der Versatz ``T_mu`` faellt in der
    Differenz also heraus und ``T_sigma`` ist der ganze Umrechnungsfaktor --
    kein zweiter Weg zu denselben Labels, der auseinanderdriften koennte.

    Freilaufend, nicht ein Schritt: gemessen wird, was das Modell allein
    erzeugt. Ein Ein-Schritt-Fehler sieht immer gut aus.
    """
    out = {}
    for op in ops:
        traj, _ = rollout(net, op, statics, lag1=lag1, lag2=lag2, clamp=clamp)
        out[op.op_id] = float(
            (traj - op.tn_seq).abs().mean().item() * T_sigma)
    return out


def fahre_einen_lauf(args, seed: int, device, daten, out_dir: Path) -> dict:
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

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    batch = stack_ops(train)
    verlauf = []
    for epoch in range(1, args.epochs + 1):
        st = train_epoch(net, batch, statics, opt,
                         inner_steps=args.inner_steps, batch_t=args.batch_t,
                         lag1=args.lag1, lag2=args.lag2, w_data=args.w_data,
                         w_phys=args.w_phys, w_wall=args.w_wall,
                         clamp=args.clamp, rng=rng)
        print(f"[seed {seed}] {st.line(epoch)}", flush=True)
        verlauf.append({"epoch": epoch, "data": st.data, "phys": st.phys,
                        "wall": st.wall, "saturated": st.saturated,
                        "spread_space": st.spread_space,
                        "spread_time": st.spread_time})

    mae = val_mae(net, val, statics, lag1=args.lag1, lag2=args.lag2,
                  clamp=args.clamp, T_sigma=bundle.T_sigma)
    for op_id, v in sorted(mae.items()):
        print(f"[seed {seed}] val-MAE {op_id}: {v:.4f} C")

    torch.save(net.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(verlauf, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"seed": seed, "konfiguration": konfigurationsname(args),
         "parameter": n_par, "val_mae_C": mae,
         "epochs": args.epochs}, indent=2))
    return mae


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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--inner-steps", type=int, default=100,
                   help="Updates je OP und Epoche auf der eingefrorenen "
                        "Trajektorie. Hunderte, nicht Zehntausende -- sonst "
                        "wird der Puffer schal.")
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
    p.add_argument("--lag1", type=int, default=5)
    p.add_argument("--lag2", type=int, default=20)
    p.add_argument("--clamp", type=float, default=50.0,
                   help="Haelt einen weglaufenden Rollout fest. Jedes "
                        "Festhalten wird als [SATURATED] gemeldet.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1,
                   help="NOCH KEINE SCHLEIFE -- heute nur eine Warnschwelle. "
                        "Ein Seed ist keine Streuung: sie liegt bei 0.518 / "
                        "0.882 C, auf OP06 bis 1.63 C. Wer hier 3 tippt, "
                        "bekommt trotzdem EINEN Lauf; die echte Schleife "
                        "kommt mit dem Ladepfad (FAHRPLAN, 'Was fehlt')")
    return p


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

    kw = modell_kwargs(args)
    print(f"[konfiguration] {konfigurationsname(args)}")
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
    bundle = daten[0]

    wurzel = args.artifacts_dir / konfigurationsname(args).split()[0]
    alle: dict[str, list] = {}
    for n in range(args.seeds):
        seed = args.seed + n
        mae = fahre_einen_lauf(args, seed, device, daten, wurzel / f"seed{seed}")
        for op_id, v in mae.items():
            alle.setdefault(op_id, []).append(v)

    print(f"\n{'=' * 60}")
    print(f"{konfigurationsname(args)}  --  {args.seeds} Seed(s), "
          f"{args.epochs} Epochen")
    for op_id, werte in sorted(alle.items()):
        a = np.asarray(werte)
        # ddof=1: die Streuung EINER Stichprobe, nicht der Grundgesamtheit.
        # Bei drei Seeds ist der Unterschied 22 %, und die Zahl wird gegen
        # eine ~1 C-Latte gelesen.
        sd = float(a.std(ddof=1)) if a.size > 1 else float("nan")
        print(f"  val-MAE {op_id}: {a.mean():.4f} +- {sd:.4f} C   "
              f"({', '.join(f'{v:.4f}' for v in werte)})")
    if args.seeds < 3:
        print("  ⚠ unter drei Seeds ist das +- keine Streuung, sondern Zierde.")
    zus = wurzel / "zusammenfassung.json"
    zus.write_text(json.dumps(
        {"konfiguration": konfigurationsname(args), "seeds": args.seeds,
         "epochs": args.epochs, "ops": list(args.ops),
         "val_ops": list(args.val_ops), "val_mae_C": alle,
         "T_sigma": float(bundle.T_sigma)}, indent=2))
    print(f"  -> {zus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
