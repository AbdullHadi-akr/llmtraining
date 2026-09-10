#!/usr/bin/env python3
"""Erster ROM-Versuch: POD-Basis + Operator Inference, frei laufender Rollout.

Was das hier ist -- und was NICHT
---------------------------------
Das ist der *erste* Wurf, kein Modell im Sinne des Fahrplans. Es ersetzt
Stufe 3 nicht (der Galerkin-Loeser aus dem FD-Operator fehlt), und es ist
nicht Stufe 4 (kein Netz, kein Torch). Es ist das, was man an einem
Nachmittag bauen kann, um zu sehen, ob die ROM-Richtung ueberhaupt traegt.

Drei Dinge kommen dabei heraus, in dieser Reihenfolge der Wichtigkeit:

1. **Stufe 3b, der Projektionsrest.** Die harte Obergrenze: was die Basis
   auf einem ausgehaltenen OP nicht erfasst, kann kein Modell darunter
   holen. Das steht VOR dem Training fest und ist die Zahl, die den
   ganzen Ansatz bestaetigt oder kippt.
2. **Ein frei laufender Rollout** ueber die volle Trajektorie, bewertet mit
   denselben Groessen wie ``op_metrics.py`` (mae / rmse / late_mae /
   peak_err), damit die Zahlen neben denen von PINNmodulusTwo lesbar sind.
3. **Die Trennung Basis gegen Dynamik.** Je OP steht der Projektionsboden
   neben dem ROM-Fehler. Ist der ROM-Fehler nahe am Boden, ist die
   Zeit-Abbildung gut und die Basis das Limit. Ist er weit darueber, ist
   es umgekehrt. Das sagt dir, wo du weiterarbeiten musst.

Warum Operator Inference statt Galerkin
---------------------------------------
Der Galerkin-Weg braucht den diskreten Waermeleitoperator ``L``, und den
hat in diesem Projekt noch niemand aufgeschrieben. Operator Inference
(Peherstorfer & Willcox 2016) lernt die reduzierten Operatoren stattdessen
per Regression aus den Schnappschuessen -- nicht-intrusiv, nur numpy,
Sekunden. Wenn das traegt, ist der Galerkin-Loeser danach die saubere
Version davon; wenn es nicht traegt, hat man das billig erfahren.

Der Zustand
-----------
::

    T(t) = m(t) * 1  +  Phi @ a(t)          m = Ortsmittel, Phi = POD-Basis

    z = [m, a_1 ... a_r]                    r+1 Zahlen statt 363

    z_{t+1} = z_t + dt * ( C @ features(z_t, u_t) )

``features`` ist bewusst nicht nur linear. Drei Terme sind physikalisch
motiviert statt generisch, und zwar aus dem Befund vom 09.09.:

* ``dT_wall = T_wand(z) - T_fluid_in``  -- die treibende Differenz des
  Wandflusses. ``T_wand`` ist eine LINEARE Funktion von ``z`` (die
  Phi-Zeilen auf der aeussersten x-Ebene), also im Modalraum berechenbar,
  ohne das Feld zu rekonstruieren.
* ``mdot * dT_wall``  -- der advektive Abtransport. Bilinear in
  (Zustand, Treiber), genau wie der echte Wandterm.
* ``q_dot``  -- die Quelle.

Was NICHT hineingeht -- und das ist eine harte Regel des Projekts:
``Q_dot(t)`` (Heat Transfer solid->fluid) und ``Tmfavg_fluid_out(t)``.
Beides sind Simulations*ergebnisse* und zur Laufzeit nicht verfuegbar. Sie
duerfen Aufsicht und Gegenprobe sein, niemals Eingang. Dieses Skript liest
sie nicht einmal ein.

Aufruf
------
::

    python3 GridCNN/rom/rom_first_try.py
    python3 GridCNN/rom/rom_first_try.py --r 8 --ridge 1e-4
    python3 GridCNN/rom/rom_first_try.py --cache /pfad/zu/data_cache

Braucht **nur numpy**. Kein Torch, kein Modulus, keine
``material_properties/``. Laeuft auf der CPU in Sekunden bis wenigen Minuten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Konstanten, uebernommen aus PINNmodulusTwo -- NICHT neu geraten.
# ---------------------------------------------------------------------------

# data.py:127. jr1_w ist eine Gesamtleistung in W; q_dot will W/m^3.
# Halbmodell, dieselbe Konvention wie das Gitter.
V_JR1 = 4.394793e-04

# data.py:134. Die Reihenfolge ist Vertrag -- Spaltenindizes haengen daran.
CONFIG_ORDER = [
    "c_rate",
    "cell_current",
    "fluid_initial_temp",
    "fluid_inlet_temp",
    "fluid_mass_flow",
    "soc_start",
    "solid_initial_temp",
]
IDX_T_IN = CONFIG_ORDER.index("fluid_inlet_temp")
IDX_MDOT = CONFIG_ORDER.index("fluid_mass_flow")

# op_registry.py:188. Der Split bleibt identisch, sonst ist der Vergleich
# gegen PINNmodulusTwo wertlos.
DEFAULT_TRAIN_OPS = ("OP01", "OP02", "OP03", "OP04", "OP05", "OP07",
                     "OP08", "OP10", "OP11", "OP12", "OP14")
DEFAULT_VAL_OPS = ("OP06", "OP09")
DEFAULT_TEST_OPS = ("OP13", "OP15", "OP16")

# Dieselbe Suchreihenfolge wie data.py und spatial_rank.py, damit hier nicht
# ein anderer Cache gelesen wird als beim Training.
CACHE_CANDIDATES = (
    "PINNmodulusTwo/data_cache",
    "PINNmodulusTwoExtProfiles/data_cache",
    "data_cache",
    "legacy/battery_surrogate_agenticWorkflow/data_cache",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_cache(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"--cache zeigt auf kein Verzeichnis: {p}")
        return p
    root = repo_root()
    for rel in CACHE_CANDIDATES:
        p = root / rel
        if p.is_dir() and any(p.glob("OP*.npz")):
            return p
    sys.exit("Kein data_cache gefunden. Gesucht relativ zu "
             f"{root}:\n  " + "\n  ".join(CACHE_CANDIDATES) +
             "\nMit --cache einen Pfad angeben.")


# ---------------------------------------------------------------------------
# Laden -- dieselben Schluessel, die data.py._read_raw liest.
# ---------------------------------------------------------------------------

def load_op(cache: Path, op_id: str) -> dict | None:
    """Ein OP-Buendel als reines numpy. ``None``, wenn die Datei fehlt."""
    f = cache / f"{op_id}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)

    t = np.asarray(z["t_fast"], dtype=np.float64)
    T = np.asarray(z["T"], dtype=np.float64)              # (n_t, n_points), Grad C

    # q_source[:, 0] = "Heat Source JR1 Monitor (W)". Eine Division, durch das
    # Volumen, in dem die Waerme deponiert wird -- data.py:424.
    q_dot = np.asarray(z["q_source"], dtype=np.float64)[:, 0] / V_JR1

    # Config: erst die Skalare, dann die Zeitreihen daruebergelegt (die
    # Profil-OPs haben sim_config_ts_<name>_t / _v). Kurzform von
    # data.py._config_timeseries_full.
    names = json.loads(str(z["sim_config_scalar_names_json"].item()))
    scal = dict(zip(names, [float(v) for v in z["sim_config_scalar"]]))
    cfg = np.tile(np.array([float(scal.get(k, np.nan)) for k in CONFIG_ORDER]),
                  (t.shape[0], 1))
    if "sim_config_ts_names_json" in z.files:
        for name in json.loads(str(z["sim_config_ts_names_json"].item())):
            if name not in CONFIG_ORDER:
                continue
            ts = np.asarray(z[f"sim_config_ts_{name}_t"], dtype=np.float64)
            vs = np.asarray(z[f"sim_config_ts_{name}_v"], dtype=np.float64)
            cfg[:, CONFIG_ORDER.index(name)] = np.interp(t, ts, vs)

    n = min(len(t), len(T), len(q_dot))
    return dict(op=op_id, t=t[:n], T=T[:n], q_dot=q_dot[:n], cfg=cfg[:n],
                xyz=np.asarray(z["xyz"], dtype=np.float64),
                synthetic=bool(z["synthetic"]) if "synthetic" in z.files else False)


def grid_index(xyz: np.ndarray, decimals: int = 9):
    """(nx, ny, nz)-Reshape aus den Koordinaten ableiten statt raten.

    Identisch zu ``spatial_rank.grid_index`` -- der Reshape, auf dem das
    Projekt steht, wird an genau einer Stelle definiert und sonst nirgends.
    """
    xr = np.round(xyz, decimals)
    xu, yu, zu = (np.unique(xr[:, c]) for c in range(3))
    nx, ny, nz = len(xu), len(yu), len(zu)
    if nx * ny * nz != len(xyz):
        sys.exit(f"Kein volles Tensorgitter: {nx}x{ny}x{nz} != {len(xyz)}.")
    idx = np.full((nx, ny, nz), -1, dtype=np.int64)
    idx[np.searchsorted(xu, xr[:, 0]),
        np.searchsorted(yu, xr[:, 1]),
        np.searchsorted(zu, xr[:, 2])] = np.arange(len(xyz))
    if (idx < 0).any():
        sys.exit("Gitterpunkte doppelt oder fehlend -- reshape unsicher.")
    return idx, xu, yu, zu


# ---------------------------------------------------------------------------
# Die Basis
# ---------------------------------------------------------------------------

def split_field(T: np.ndarray):
    """``T -> (m, anom)``: Ortsmittel je Zeit, und was danach uebrig bleibt.

    Das ist NICHT das PCA-uebliche Zentrieren je Merkmal, sondern ein
    physikalischer Split: erst "die ganze Zelle wird waermer" abziehen, dann
    den Rest zerlegen. Daher das ``m(t)`` im Zustand.
    """
    m = T.mean(axis=1)
    return m, T - m[:, None]


def build_basis(train: list[dict], r: int, max_times: int):
    """POD-Basis ueber die gepoolten Trainings-OPs. Gibt ``(Phi, spektrum)``."""
    blocks = []
    for op in train:
        _, anom = split_field(op["T"])
        step = max(1, len(anom) // max_times)
        blocks.append(anom[::step])
    X = np.concatenate(blocks, axis=0)          # (n_snapshots, n_points)
    # Eckart-Young: die ersten r Rechtssingulaervektoren sind die beste
    # Rang-r-Naeherung im Least-Squares-Sinn. Kein anderer Rang-r-Ansatz
    # schlaegt sie.
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    energy = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-300)
    return Vt[:r].T, energy                      # Phi: (n_points, r)


def projection_residual(Phi: np.ndarray, op: dict) -> float:
    """Stufe 3b: ``||T - (m + Phi Phi^T a)|| / ||T - m||`` fuer einen OP.

    Die harte Obergrenze. Kein Modell kommt unter diese Zahl, weil der
    Anteil, den die Basis nicht aufspannt, nirgends dargestellt werden kann.
    """
    _, anom = split_field(op["T"])
    rec = (anom @ Phi) @ Phi.T
    denom = float(np.linalg.norm(anom))
    return float(np.linalg.norm(anom - rec)) / max(denom, 1e-300)


# ---------------------------------------------------------------------------
# Die Dynamik: Operator Inference
# ---------------------------------------------------------------------------

def features(z: np.ndarray, q: np.ndarray, t_in: np.ndarray,
             mdot: np.ndarray, w_wall: np.ndarray) -> np.ndarray:
    """Regressor-Matrix. ``z`` (n, r+1), Treiber (n,), ``w_wall`` (r,).

    Die drei physikalisch motivierten Terme stehen im Modul-Docstring. Alles
    hier ist zur Laufzeit verfuegbar -- das ist die Bedingung, an der sich
    entscheidet, ob ein Kanal hineindarf.
    """
    m = z[:, 0]
    a = z[:, 1:]
    # T an der Gehaeusewand, LINEAR aus dem Zustand. Das Feld wird dafuer
    # nicht rekonstruiert -- der Punkt der modalen Darstellung.
    T_wall = m + a @ w_wall
    dT_wall = T_wall - t_in
    return np.column_stack([
        np.ones_like(m),          # konstanter Term
        *z.T,                     # der Zustand selbst (r+1 Spalten)
        q,                        # die Quelle
        dT_wall,                  # treibende Differenz am Wandterm
        mdot * dT_wall,           # advektiver Abtransport, bilinear
        mdot,                     # reiner Treiber
    ])


def fit(train: list[dict], Phi: np.ndarray, w_wall: np.ndarray,
        ridge: float, scale: dict):
    """Least-Squares-Fit von ``dz/dt = C @ features``, ridge-regularisiert.

    Gefittet wird der Vorwaerts-Differenzenquotient, also GENAU der Operator,
    der spaeter integriert wird. Ein Fit gegen eine andere Diskretisierung
    als die des Rollouts ist eine der leiseren Arten, sich selbst zu betruegen.
    """
    F_all, Y_all = [], []
    for op in train:
        z = state_of(op, Phi)
        # Treiber auf DERSELBEN Laenge wie z[:-1] -- der Vorwaertsschritt
        # liest den Treiber am linken Rand des Intervalls.
        q, t_in, mdot = (d[:-1] for d in drivers_of(op, scale))
        f = features(z[:-1], q, t_in, mdot, w_wall)
        dt = np.diff(op["t"])[:, None]
        ok = dt[:, 0] > 0
        F_all.append(f[ok])
        Y_all.append((np.diff(z, axis=0) / np.where(dt > 0, dt, 1.0))[ok])
    F = np.concatenate(F_all)
    Y = np.concatenate(Y_all)

    # Spalten standardisieren, sonst dominiert q_dot (~1e5 W/m^3) alles andere
    # und die Ridge-Strafe trifft die falschen Koeffizienten.
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-12] = 1.0
    mu[0] = 0.0                      # den konstanten Term nicht zentrieren
    sd[0] = 1.0
    Fs = (F - mu) / sd

    G = Fs.T @ Fs + ridge * len(Fs) * np.eye(Fs.shape[1])
    C = np.linalg.solve(G, Fs.T @ Y)                # (n_feat, r+1)
    return dict(C=C, mu=mu, sd=sd)


def state_of(op: dict, Phi: np.ndarray) -> np.ndarray:
    m, anom = split_field(op["T"])
    return np.column_stack([m, anom @ Phi])


def drivers_of(op: dict, scale: dict):
    """``(q, T_in, mdot)`` -- alle drei zur Laufzeit verfuegbar."""
    return (op["q_dot"] / scale["q"],
            op["cfg"][:, IDX_T_IN],
            op["cfg"][:, IDX_MDOT] / scale["mdot"])


def rollout(op: dict, Phi: np.ndarray, w_wall: np.ndarray, model: dict,
            scale: dict, bound: float) -> np.ndarray:
    """Frei laufend ab ``T_0``. Kein Teacher Forcing, wie in PINNmodulusTwo.

    Bricht ab, sobald der Zustand die Schranke reisst -- ein Loeser, der
    weglaeuft, ist ein Befund und wird nicht stillschweigend weitergerechnet.
    """
    t = op["t"]
    q, t_in, mdot = drivers_of(op, scale)
    z0 = state_of(op, Phi)[0]
    n, r1 = len(t), len(z0)
    Z = np.empty((n, r1))
    Z[0] = z0
    C, mu, sd = model["C"], model["mu"], model["sd"]
    T0_ref = float(op["T"][0].mean())
    for i in range(n - 1):
        f = features(Z[i:i + 1], q[i:i + 1], t_in[i:i + 1],
                     mdot[i:i + 1], w_wall)
        dz = ((f - mu) / sd) @ C
        Z[i + 1] = Z[i] + (t[i + 1] - t[i]) * dz[0]
        if not np.isfinite(Z[i + 1]).all() or abs(Z[i + 1, 0] - T0_ref) > bound:
            Z[i + 1:] = Z[i + 1]
            print(f"   !! {op['op']}: Rollout divergiert bei Schritt {i+1} "
                  f"(t = {t[i+1]:.1f} s). NICHT mit mehr Modell uebertuenchen "
                  f"-- das ist ein Befund.")
            break
    return Z[:, 0:1] + Z[:, 1:] @ Phi.T           # zurueck auf (n_t, n_points)


# ---------------------------------------------------------------------------
# Bewertung -- dieselben Groessen wie op_metrics.py
# ---------------------------------------------------------------------------

def metrics(pred: np.ndarray, true: np.ndarray, late_frac: float) -> dict:
    """mae / rmse / late_mae / peak_err, Schritt 0 ausgeschlossen.

    Schritt 0 ist die aufgepraegte Anfangsbedingung: sein Fehler ist
    identisch null und wuerde den Mittelwert nur verduennen. Genau die
    Konvention aus ``op_metrics.op_metrics``.
    """
    resid = (pred - true)[1:]
    err = np.abs(resid)
    split = max(int((1.0 - late_frac) * len(err)), 1)
    return dict(
        mae=float(err.mean()),
        rmse=float(np.sqrt((resid ** 2).mean())),
        late_mae=float(err[split:].mean()) if len(err) > split else float("nan"),
        late_bias=float(resid[split:].mean()) if len(err) > split else float("nan"),
        max_abs=float(err.max()),
        peak_err=float(abs(pred.max() - true.max())),
    )


def trivial_baselines(op: dict, train_mean: float, late_frac: float) -> dict:
    """Die zwei trivialen Vorhersager. Wer die nicht schlaegt, hat nichts gelernt."""
    true = op["T"]
    pers = np.repeat(true[:1], len(true), axis=0)
    tm = np.full_like(true, train_mean)
    return {"persistence": metrics(pers, true, late_frac),
            "train-mean": metrics(tm, true, late_frac)}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--r", type=int, default=6,
                    help="Modenzahl (Default 6 = 99.99 %% laut Rangtest)")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="Ridge-Parameter. Hoeher = stabiler, traeger")
    ap.add_argument("--max-times", type=int, default=1500,
                    help="Zeitschritte je OP fuer die SVD")
    ap.add_argument("--late-frac", type=float, default=0.33,
                    help="Anteil am Ende fuer late_mae (Default letztes Drittel)")
    ap.add_argument("--bound", type=float, default=500.0,
                    help="Abbruchschranke |m - T0| in K")
    args = ap.parse_args()

    cache = find_cache(args.cache)
    print(f"Cache: {cache}")
    print(f"r = {args.r} Moden, ridge = {args.ridge:g}\n")

    def load_many(ids):
        out = []
        for op_id in ids:
            b = load_op(cache, op_id)
            if b is None:
                print(f"  [fehlt] {op_id}")
            else:
                out.append(b)
        return out

    train = load_many(DEFAULT_TRAIN_OPS)
    val = load_many(DEFAULT_VAL_OPS)
    test = load_many(DEFAULT_TEST_OPS)
    if not train:
        sys.exit("Kein einziger Trainings-OP geladen.")
    if any(b["synthetic"] for b in train + val + test):
        print("!! SYNTHETISCHES BUENDEL -- die Zahlen unten sagen nichts ueber "
              "die echten OPs.\n")

    idx, xu, _, _ = grid_index(train[0]["xyz"])
    print(f"== Gitter ==  {idx.shape[0]} x {idx.shape[1]} x {idx.shape[2]} "
          f"= {train[0]['xyz'].shape[0]} Punkte")
    wall_pts = idx[-1].ravel()          # aeusserste x-Ebene = Gehaeusewand
    print(f"   Wandebene: x = {xu[-1]:.6f} m, {len(wall_pts)} Punkte\n")

    # ---- Basis --------------------------------------------------------------
    Phi, energy = build_basis(train, args.r, args.max_times)
    w_wall = Phi[wall_pts].mean(axis=0)          # T_wand ist linear in z
    lvl = {L: int(np.searchsorted(energy, L) + 1)
           for L in (0.90, 0.99, 0.999, 0.9999)}
    print("== Basis (POD ueber die 11 Trainings-OPs) ==")
    print("   Moden fuer 90/99/99.9/99.99 %: "
          + "/".join(str(lvl[L]) for L in (0.90, 0.99, 0.999, 0.9999)))
    print(f"   genommen: r = {args.r}, erfasst "
          f"{energy[args.r-1]*100:.4f} % der Trainings-Energie\n")

    # ---- Stufe 3b: das eigentliche Tor -------------------------------------
    print("== Stufe 3b: Projektionsrest -- die harte Obergrenze ==")
    print("   Was die Basis nicht aufspannt, holt KEIN Modell zurueck.")
    print(f"   {'OP':<6} {'Rolle':<8} {'Rest':>9}  {'~Boden [K]':>11}")
    floors = {}
    for role, group in (("train", train), ("val", val), ("test", test)):
        for op in group:
            res = projection_residual(Phi, op)
            _, anom = split_field(op["T"])
            # Rest * RMS der Ortsstruktur = der Fehler in Kelvin, den die
            # Basis allein schon einbaut.
            floor = res * float(np.sqrt((anom ** 2).mean()))
            floors[op["op"]] = floor
            # Die Schwellen sind die aus FAHRPLAN.md, Stufe 3b -- nicht neu
            # erfunden: "Rest <~ 0.1 % auf allen fuenf -> gruen".
            flag = ("" if res < 0.001 else
                    "  <-- grenzwertig" if res < 0.01 else "  <-- GROSS")
            print(f"   {op['op']:<6} {role:<8} {res*100:>8.4f}% "
                  f"{floor:>11.4f}{flag}")
    print("   Lesart: Rest <~ 1 % auf den ausgehaltenen OPs -> die Basis")
    print("   verallgemeinert. Gross nur auf OP06 -> r erhoehen oder es ist O14.")
    print("   Gross ueberall -> POD auf Trainings-OPs ist der falsche Ansatz.\n")

    # ---- Dynamik ------------------------------------------------------------
    scale = {"q": max(float(np.mean([np.abs(b["q_dot"]).max() for b in train])), 1e-30),
             "mdot": max(float(np.mean([np.abs(b["cfg"][:, IDX_MDOT]).max()
                                        for b in train])), 1e-30)}
    model = fit(train, Phi, w_wall, args.ridge, scale)
    n_par = model["C"].size
    print(f"== Dynamik: Operator Inference ==")
    print(f"   {model['C'].shape[0]} Features x {model['C'].shape[1]} "
          f"Zustaende = {n_par} Parameter")
    print(f"   zum Vergleich: das punktweise MLP faehrt ~50-100 k.\n")

    # ---- Rollout ------------------------------------------------------------
    train_mean = float(np.mean([b["T"].mean() for b in train]))
    print("== Frei laufender Rollout ==  (kein Teacher Forcing)")
    print(f"   {'OP':<6} {'Rolle':<8} {'mae':>8} {'rmse':>8} {'late':>8} "
          f"{'peak':>8} {'Boden':>8} {'persist':>8} {'tr-mean':>8}")
    for role, group in (("train", train), ("val", val), ("test", test)):
        for op in group:
            pred = rollout(op, Phi, w_wall, model, scale, args.bound)
            m = metrics(pred, op["T"], args.late_frac)
            base = trivial_baselines(op, train_mean, args.late_frac)
            print(f"   {op['op']:<6} {role:<8} {m['mae']:>8.3f} "
                  f"{m['rmse']:>8.3f} {m['late_mae']:>8.3f} "
                  f"{m['peak_err']:>8.3f} {floors[op['op']]:>8.3f} "
                  f"{base['persistence']['mae']:>8.3f} "
                  f"{base['train-mean']['mae']:>8.3f}")

    print()
    print("   'Boden' ist der Projektionsrest in Kelvin -- der Fehler, den die")
    print("   BASIS schon einbaut. Liegt mae nahe am Boden, ist die Dynamik gut")
    print("   und die Basis das Limit. Liegt mae weit darueber, ist es umgekehrt")
    print("   und die Arbeit gehoert in die Zeit-Abbildung (Stufe 3/4/5).")
    print()
    print("   Zum Einordnen, PINNmodulusTwo auf OP06: 6.270 Mittel / 13.248 spaet.")
    print("   Wer 'persist' und 'tr-mean' nicht schlaegt, hat nichts gelernt.")


if __name__ == "__main__":
    main()
