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
"""

from __future__ import annotations

import argparse
import sys
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
# Der freie Rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(net: M.GridCNN, op: OPTensors, statics: M.StaticMaps, *,
            lag1: int, lag2: int, n_steps: int | None = None,
            clamp: float = 0.0,
            wall: phys.WallModel | None = None) -> tuple[torch.Tensor, int]:
    """Frei laufend, gesaet nur von der gemessenen Anfangsbedingung.

    Zurueck kommt ``(traj, n_saturated)`` mit ``traj`` der Form
    ``(n_steps+1, nx, ny, nz)``.

    ``clamp`` haelt einen weglaufenden Rollout fest, damit der Verlust endlich
    bleibt. **Das Festhalten wird gezaehlt und gemeldet**: Stille saehe hier
    aus wie langsame Konvergenz, und das ist der teuerste Irrtum, den diese
    Schleife anbieten kann. Der ``[SATURATED]``-Zaehler aus ``PINNmodulusTwo``
    wird genau deshalb mitgenommen.
    """
    n = op.n_t - 1 if n_steps is None else n_steps
    traj = torch.empty((n + 1, *op.tn_ic.shape), dtype=op.tn_ic.dtype,
                       device=op.tn_ic.device)
    traj[0] = op.tn_ic
    ny, nz = op.tn_ic.shape[1:]
    saturated = 0

    for k in range(n):
        idx = torch.tensor([k], device=traj.device)
        t0, t1, t2 = history_at(traj[:k + 1], idx.clamp(max=k), lag1, lag2)
        x = M.assemble_input(
            M.state_channels(t0, t1, t2), statics,
            M.driver_channels(op.config[k:k + 1], op.forcing[k:k + 1], ny, nz))
        ghost = (M.adiabatic_ghost(t0) if wall is None
                 else _wall_ghost(t0, wall, op, k))
        nxt = net.step(t0, x, dt_n=op.dtn, fo_field=op.fo,
                       qsrc=op.qsrc[k:k + 1], ghost_hi=ghost)[0]
        if clamp > 0.0:
            over = int((nxt.abs() >= clamp).sum())
            if over:
                saturated += 1
                nxt = nxt.clamp(-clamp, clamp)
        traj[k + 1] = nxt
    return traj, saturated


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
def train_epoch(net: M.GridCNN, ops: list, statics: M.StaticMaps,
                opt: torch.optim.Optimizer, *, inner_steps: int,
                batch_t: int, lag1: int, lag2: int, w_data: float,
                w_phys: float, w_wall: float, clamp: float,
                rng: np.random.Generator) -> EpochStats:
    """Je OP einmal ausrollen, einfrieren, dann ``inner_steps`` Updates."""
    st = EpochStats()
    st.n_ops = len(ops)
    want_phys = w_phys > 0.0

    for op in ops:
        traj, sat = rollout(net, op, statics, lag1=lag1, lag2=lag2, clamp=clamp)
        st.saturated += sat
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
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GridCNN trainieren -- Ein-Schritt gegen eine "
                    "eingefrorene, frei laufende Trajektorie.")
    p.add_argument("--cache", type=Path, default=Path("data_cache"),
                   help="Verzeichnis mit den OP-Buendeln")
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
                   help="Ein Seed ist keine Streuung. Die Seed-Streuung liegt "
                        "bei 0.518 / 0.882 C -- ein Unterschied unter ~1 C ist "
                        "mit drei Seeds nicht lesbar.")
    return p


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
    print("Der Ladepfad ueber PINNmodulusTwo/data.py ist noch nicht "
          "angeschlossen -- siehe FAHRPLAN, 'Was fehlt'. Die Mechanik "
          "(Rollout, Verluste, Epoche) steht und ist getestet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
