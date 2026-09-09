#!/usr/bin/env python3
"""Wie viel raeumliche Struktur steckt ueberhaupt im Temperaturfeld?

Warum dieses Skript zuerst laeuft
---------------------------------
``GridCNN`` faltet ueber ein 11x11-Gitter, um raeumliche Struktur zu lernen.
Bevor dafuer eine Zeile Modell geschrieben wird, gehoert die Gegenfrage
gemessen: **ist die raeumliche Struktur ueberhaupt schwer?**

Zwei Befunde aus dem Entwurf machen den Verdacht konkret, dass sie es nicht ist:

* Das Feld ist diffusiv und wird von *glatten globalen Skalaren* getrieben
  (q_dot, Fluidtemperatur, Volumenstrom).
* ``data.py:594`` baut die Quelle als ``q_dot(t) * q_mask / (rho*Cp)`` -- ein
  Skalar mal einer *festen* Ortskarte. Damit ist ``T`` selbst der einzige
  Input, der raeumlich strukturiert UND zeitabhaengig ist.

Wenn vier POD-Moden 99.9 % der Energie tragen, loest jede Architektur das
raeumliche Problem und der ganze Fehler sitzt in der Zeit-Abbildung. Dann ist
ein 5-Moden-ROM die ehrlichere Antwort als ein CNN. Braucht es dreissig Moden,
hat die Faltung etwas zu holen.

Das Skript entscheidet also nicht, *ob* GridCNN gebaut wird -- es entscheidet,
**wie gross** das Netz sein muss. Bei elf Trainings-Trajektorien ist das keine
Nebenfrage.

Was gerechnet wird
------------------
Drei Zerlegungen, weil sie drei verschiedene Fragen beantworten:

1. ``T - Zeitmittel je Punkt``  -- wie viele Moden beschreiben die Dynamik?
2. ``T - Ortsmittel je Zeit``   -- wie viel bleibt uebrig, wenn man "die ganze
   Zelle wird waermer" abzieht? DAS ist die Zahl, an der die Faltung haengt.
3. dasselbe **gepoolt** ueber mehrere OPs -- traegt EINE Basis alle OPs? Das
   ist die eigentlich relevante Frage, denn das Modell soll ja nicht je OP
   eine eigene Basis lernen.

Dazu zwei Kontrollen, die nichts kosten:

* die **Gitterprobe**: dass die 363 Punkte wirklich ein 3x11x11-Raster bilden
  und in welcher Reihenfolge -- der ``reshape``, auf dem das ganze Projekt
  steht, wird hier abgeleitet statt geraten.
* das **Wandgefaelle je OP** neben dem Volumenstrom: fliesst die Waerme
  wirklich in x-Richtung ab, wie der Wandterm des Entwurfs annimmt?

Braucht nur numpy. Kein Torch, kein pandas, keine ``material_properties/``.

Aufruf
------
    python3 GridCNN/tools/spatial_rank.py
    python3 GridCNN/tools/spatial_rank.py --ops OP01 OP04 OP06 OP09
    python3 GridCNN/tools/spatial_rank.py --cache /pfad/zu/data_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Dieselbe Suchreihenfolge wie data.py, damit hier nicht ein anderer Cache
# gelesen wird als beim Training.
CACHE_CANDIDATES = (
    "PINNmodulusTwo/data_cache",
    "PINNmodulusTwoExtProfiles/data_cache",
    "data_cache",
    "legacy/battery_surrogate_agenticWorkflow/data_cache",
)

DEFAULT_OPS = ("OP01", "OP02", "OP03", "OP04", "OP05", "OP07",
               "OP08", "OP10", "OP11", "OP12", "OP14")

# Anteile, an denen abgelesen wird. 99.9 % ist der interessante: darunter
# beginnt das, was ein Surrogat mit ~1 K Ziel noch sehen muss.
LEVELS = (0.90, 0.99, 0.999, 0.9999)


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
    sys.exit(
        "Kein data_cache gefunden. Gesucht wurde, relativ zu "
        f"{root}:\n  " + "\n  ".join(CACHE_CANDIDATES) +
        "\nMit --cache einen Pfad angeben."
    )


def grid_index(xyz: np.ndarray, decimals: int = 9):
    """Leitet den (nx, ny, nz)-Reshape aus den Koordinaten ab.

    Gibt ``(idx, xu, yu, zu)`` zurueck; ``idx[i,j,k]`` ist der Punktindex mit
    ``x=xu[i], y=yu[j], z=zu[k]``. Faellt aus, wenn das Raster nicht
    vollstaendig ist -- lieber hier laut als spaeter mit einem stillschweigend
    verdrehten Feld.
    """
    xr = np.round(xyz, decimals)
    xu, yu, zu = (np.unique(xr[:, c]) for c in range(3))
    nx, ny, nz = len(xu), len(yu), len(zu)
    if nx * ny * nz != len(xyz):
        raise SystemExit(
            f"Kein volles Tensorgitter: {nx} x {ny} x {nz} = {nx*ny*nz}, "
            f"aber {len(xyz)} Punkte. Der reshape waere falsch."
        )
    ix = np.searchsorted(xu, xr[:, 0])
    iy = np.searchsorted(yu, xr[:, 1])
    iz = np.searchsorted(zu, xr[:, 2])
    idx = np.full((nx, ny, nz), -1, dtype=np.int64)
    idx[ix, iy, iz] = np.arange(len(xyz))
    if (idx < 0).any():
        raise SystemExit("Gitterpunkte doppelt oder fehlend -- reshape unsicher.")
    return idx, xu, yu, zu


def spectrum(X: np.ndarray) -> np.ndarray:
    """Kumulierter Energieanteil der POD-Moden von ``X`` (n_samples, n_points)."""
    # Nur die Singulaerwerte, nicht die Vektoren: gvals ist billiger und wir
    # brauchen hier ausschliesslich das Spektrum.
    s = np.linalg.svd(X, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return np.array([1.0])
    return np.cumsum(e) / tot


def modes_for(cum: np.ndarray, level: float) -> int:
    """Kleinste Modenzahl, die ``level`` der Energie erreicht."""
    return int(np.searchsorted(cum, level) + 1)


def line(cum: np.ndarray) -> str:
    """Modenzahlen fuer LEVELS, kompakt als ``1/2/7/40``."""
    return "/".join(str(modes_for(cum, l)) for l in LEVELS)


def subsample(n: int, cap: int) -> slice:
    """Jede k-te Zeile, damit die SVD auch bei ~8000 Schritten Sekunden bleibt."""
    return slice(None, None, max(1, n // cap))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS),
                    help="welche OPs (Default: die elf Trainings-OPs)")
    ap.add_argument("--cache", default=None, help="Pfad zum data_cache")
    ap.add_argument("--max-times", type=int, default=1500,
                    help="Zeitschritte je OP fuer die SVD (Default 1500)")
    args = ap.parse_args()

    cache = find_cache(args.cache)
    print(f"Cache: {cache}\n")

    fields, meta, synthetic = {}, {}, False
    for op in args.ops:
        f = cache / f"{op}.npz"
        if not f.exists():
            print(f"  [fehlt] {op}")
            continue
        r = np.load(f, allow_pickle=True)
        fields[op] = np.asarray(r["T"], dtype=np.float64)
        meta[op] = {"xyz": np.asarray(r["xyz"], dtype=np.float64)}
        if "layer" in r.files:
            meta[op]["layer"] = np.asarray(r["layer"])
        synthetic |= bool(r["synthetic"]) if "synthetic" in r.files else False

    if not fields:
        sys.exit("Kein einziger OP geladen.")
    if synthetic:
        print("!! SYNTHETISCHES BUENDEL -- die Zahlen unten sagen nichts ueber "
              "die echten OPs.\n")

    # ---- 1. Gitterprobe -----------------------------------------------------
    first = next(iter(meta))
    idx, xu, yu, zu = grid_index(meta[first]["xyz"])
    nx, ny, nz = idx.shape
    print(f"== Gitter ==  {nx} x {ny} x {nz} = {nx*ny*nz} Punkte")
    print(f"   x: {np.round(xu, 6).tolist()}")
    print(f"   dy = {np.diff(yu).mean()*1000:.3f} mm  (Spanne "
          f"{(yu[-1]-yu[0])*1000:.2f} mm)")
    print(f"   dz = {np.diff(zu).mean()*1000:.3f} mm  (Spanne "
          f"{(zu[-1]-zu[0])*1000:.2f} mm)")
    if "layer" in meta[first]:
        lay = meta[first]["layer"]
        for i, xv in enumerate(xu):
            names = np.unique(lay[idx[i].ravel()])
            print(f"   x={xv:<10.6f} -> {', '.join(map(str, names))}")
    # Dass alle OPs dasselbe Gitter haben, ist die Voraussetzung fuer EIN Modell.
    for op, m in meta.items():
        if not np.allclose(m["xyz"], meta[first]["xyz"]):
            print(f"   !! {op} hat ein anderes Gitter als {first}")
    print()

    # ---- 2. Rang je OP ------------------------------------------------------
    print("== POD je OP ==  Modenzahl fuer 90/99/99.9/99.99 % der Energie")
    print(f"{'OP':<6} {'n_t':>6} {'uniform':>9} {'Dynamik':>18} "
          f"{'Ortsstruktur':>18}")
    anomalies = []
    for op, T in fields.items():
        Ts = T[subsample(len(T), args.max_times)]
        dyn = spectrum(Ts - Ts.mean(axis=0, keepdims=True))
        # "uniform" = wie viel der Gesamtvarianz allein daher kommt, dass sich
        # das ORTSMITTEL bewegt -- also "die ganze Zelle wird waermer". Was
        # danach uebrig bleibt, ist das, wofuer man ueberhaupt ein Feldmodell
        # braucht.
        sm = Ts.mean(axis=1, keepdims=True)
        anom = Ts - sm
        var_tot = float(((Ts - Ts.mean()) ** 2).sum())
        var_anom = float(((anom - anom.mean()) ** 2).sum())
        uni = 1.0 - var_anom / max(var_tot, 1e-30)
        spa = spectrum(anom - anom.mean(axis=0, keepdims=True))
        anomalies.append(anom - anom.mean(axis=0, keepdims=True))
        print(f"{op:<6} {len(T):>6} {uni*100:>8.3f}% {line(dyn):>18} "
              f"{line(spa):>18}")

    # ---- 3. Gepoolt ---------------------------------------------------------
    # Die entscheidende Zahl: EINE Basis fuer alle OPs. Ein Modell, das je OP
    # eine eigene Basis braeuchte, kann nicht verallgemeinern.
    pooled = np.concatenate(anomalies, axis=0)
    cum = spectrum(pooled)
    print(f"\n== POD gepoolt ueber {len(anomalies)} OPs (Ortsstruktur) ==")
    print(f"   {line(cum)}")
    print(f"   Moden gesamt moeglich: {min(pooled.shape)}")

    print("\n   Lesart -- die 99.9-%-Spalte der GEPOOLTEN Ortsstruktur ist die Zahl:")
    print("     <= ~5 Moden  -> der Raum ist trivial. Ein ROM auf den Moden ist")
    print("                     die ehrlichere Antwort als ein CNN, und f darf")
    print("                     sehr klein sein.")
    print("     ~30+ Moden   -> echte Ortsstruktur, die Faltung hat etwas zu holen.")
    print()
    print("   Vorsicht bei der 99.99-%-Spalte: sobald der uniform-Anteil nahe")
    print("   100 % liegt, ist die Ortsstruktur selbst klein und die letzten")
    print("   0.01 % sind Messrauschen. Eine dreistellige Modenzahl dort zaehlt")
    print("   Rauschen, nicht Physik -- immer neben dem uniform-Anteil lesen.")

    # ---- 4. Wandgefaelle vs. Volumenstrom -----------------------------------
    # Der Entwurf nimmt an, die Waerme verlasse das Gebiet ueber die
    # Gehaeusewand in x. Dann muss das x-Gefaelle mit dem Volumenstrom wachsen.
    print("\n== Gefaelle Wand gegen Zellmitte (Mittel ueber die Zeit) ==")
    print(f"{'OP':<6} {'T(x=0) - T(Wand)':>18} {'K':>3}")
    for op, T in fields.items():
        g = T[:, idx[0].ravel()].mean(axis=1) - T[:, idx[-1].ravel()].mean(axis=1)
        print(f"{op:<6} {g.mean():>18.3f} {'':>3}")
    print("   Waechst der Wert mit dem Volumenstrom des OP, sitzt die Senke")
    print("   wirklich an der Gehaeusewand -- die Annahme hinter ghost_hi.")


if __name__ == "__main__":
    main()
