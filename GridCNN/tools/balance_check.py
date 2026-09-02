#!/usr/bin/env python3
"""Geht die Waermebilanz auf? -- Stufe 1 des GridCNN-Fahrplans.

Warum vor jedem Modellcode
--------------------------
Der Wandterm ist der einzige Punkt, an dem GridCNN mehr Physik enthaelt als
``PINNmodulusTwo`` (README 11.6). Er steht und faellt damit, dass die
Waermebilanz aufgeht. Das jetzt zu wissen kostet Minuten; es spaeter zu merken
kostet Wochen, und es saehe aus wie ein Modellfehler statt wie ein
Buchhaltungsfehler.

Was geprueft wird
-----------------
1. **Der Halbmodell-Faktor.** ``*_Heat Source.csv`` fuehrt ``jr1_w``, ``jr2_w``
   UND ``total_w``. Ist ``total_w ~ jr1_w + jr2_w`` und ``jr2_w ~ jr1_w``, dann
   hat die Zelle zwei Wickel, das Halbmodell (x = 0 .. 0.0219) enthaelt genau
   einen, und ``jr1_w`` ist die Halbmodell-Leistung. Damit ist die Frage
   beantwortet, ohne jemanden zu fragen -- und die Konvention des
   Waermestrom-Monitors laesst sich daran messen.

2. **Die Fluidbilanz.** Fuer einen Durchfluss gilt stationaer

       dT_fluid = Qdot / (mdot * Cp_fluid)

   Das ist gegen die gemessene Differenz ``Tmfavg_fluid_out - T_fluid_in`` zu
   halten.

   .. note::
      Dokument 030 schreibt dafuer ``integral(Qdot dt) / (mdot * Cp)``. Das ist
      dimensionell K*s, nicht K -- die Formel gilt fuer ein GESCHLOSSENES
      Fluidvolumen, das sich aufheizt, nicht fuer einen Durchfluss. Hier steht
      die Durchflussform. Ein Test gegen die falsche Formel wuerde
      fehlschlagen, ohne dass irgendetwas an der Physik falsch waere.

3. **Die Energiebilanz ueber den ganzen Lauf.** ``integral(Qdot dt)`` gegen
   ``integral(jr1_w dt)``: der Anteil der Quellenergie, der ueber die Kuehlwand
   abfliesst. ``energy_balance_report`` sieht davon heute nur den Fehlbetrag
   (0.5-0.9x, dem Volumenstrom folgend) -- hier steht die andere Seite.

4. **``h_eff`` gegen den Volumenstrom.** Liegt es auf einer Kurve, ist ``h``
   eine feste Funktion von ``V_dot`` und kostet im Modell keinen freien
   Parameter. Braucht die Wandtemperatur, also den ``.npz``-Cache; ohne ihn
   wird der Teil uebersprungen.

Bei ``mdot = 0`` (OP07, OP14) gibt es keinen Durchfluss. Dort ist 2. nicht
definiert, aber 3. ist die schaerfste Probe des Satzes: ohne Kuehlung muss
``Qdot`` ~ 0 sein.

Braucht nur numpy. Kein pandas, kein Torch -- die CSVs werden mit ``csv``
gelesen, in derselben Konvention wie die legacy-Assembly (``cp1252``,
Komma-getrennt, eine Kopfzeile).

Aufruf
------
    python3 GridCNN/tools/balance_check.py
    python3 GridCNN/tools/balance_check.py --ops OP04 OP05 OP07 OP14
    python3 GridCNN/tools/balance_check.py --ops OP04 --list-columns

``--list-columns`` druckt jede Spalte jeder gefundenen Datei. Der erste Lauf
auf einer neuen Maschine sollte damit anfangen: die Monitornamen stammen aus
einem StarCCM+-Export und muessen nicht ueberall gleich heissen.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# numpy < 2.0 kennt nur np.trapz, ab 2.0 heisst es np.trapezoid. Der
# Trainingsrechner und dieser Checkout muessen nicht dieselbe Version haben.
_trapz = getattr(np, "trapezoid", None) or np.trapz

RAW_CANDIDATES = (
    "legacy/battery_surrogate_agenticWorkflow/data_raw",
    "data_raw",
)
ENCODING = "cp1252"          # wie build.yaml: csv_encoding, Default cp1252
DEFAULT_OPS = ("OP04", "OP05", "OP07", "OP14")

TIME = ("Physical Time (s)",)
HEAT_SRC = {"jr1_w": ("Heat Source JR1 Monitor (W)",),
            "jr2_w": ("Heat Source JR2 Monitor (W)",),
            "total_w": ("Heat Source Monitor (W)",)}
Q_WALL = ("Heat Transfer: solid to fluid Monitor (W)",
          "Heat Transfer: solid to fluid (W)")
T_OUT = ("Tmfavg_fluid_out Monitor (C)", "Tmfavg_fluid_out (C)")
T_IN = ("Tmfavg_fluid_in Monitor (C)", "Tmfavg_fluid_in (C)",
        "Fluid Inlet Temperature Monitor (C)")
CP_FLUID = ("Specific Heat Monitor (J/kg-K)", "Specific Heat (J/kg-K)")
MDOT = ("Fluid Mass Flow Monitor (kg/s)",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_raw(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"--raw zeigt auf kein Verzeichnis: {p}")
        return p
    root = repo_root()
    for rel in RAW_CANDIDATES:
        p = root / rel
        if p.is_dir():
            return p
    sys.exit("Kein data_raw/ gefunden. Gesucht wurde, relativ zu "
             f"{root}:\n  " + "\n  ".join(RAW_CANDIDATES) +
             "\nMit --raw einen Pfad angeben.")


def read_csv_cols(path: Path) -> dict[str, np.ndarray]:
    """CSV -> {Spaltenname: float-Array}. Nicht-numerische Spalten fallen weg."""
    with open(path, newline="", encoding=ENCODING, errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {}
    head, body = rows[0], rows[1:]
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(head):
        vals = []
        for r in body:
            if i >= len(r) or r[i].strip() == "":
                vals.append(np.nan)
                continue
            try:
                vals.append(float(r[i]))
            except ValueError:
                vals.append(np.nan)
        a = np.asarray(vals, dtype=np.float64)
        if not np.isnan(a).all():
            out[name.strip()] = a
    return out


def pick(cols: dict, candidates, what: str, path: Path, required=True):
    """Erste passende Spalte, case-insensitiv als Rueckfall.

    Die Namen kommen aus einem StarCCM+-Export; ein KeyError mit der Liste der
    vorhandenen Spalten ist hier deutlich nuetzlicher als einer ohne.
    """
    for c in candidates:
        if c in cols:
            return cols[c]
    low = {k.casefold(): v for k, v in cols.items()}
    for c in candidates:
        if c.casefold() in low:
            return low[c.casefold()]
    if not required:
        return None
    sys.exit(
        f"\n{what} nicht gefunden in {path.name}.\n"
        f"  gesucht : {list(candidates)}\n"
        f"  vorhanden:\n    " + "\n    ".join(sorted(cols)) +
        "\n\n  Mit --list-columns alle Dateien auflisten."
    )


def one_glob(d: Path, pattern: str) -> Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def op_dir(raw: Path, op: str) -> Path | None:
    # Die Rohordner liegen zwei Ebenen tief: OP<NN>/OP<NN>/ (README, kein Tippfehler)
    for cand in (raw / op / op, raw / op):
        if cand.is_dir():
            return cand
    return None


def wall_temp_from_cache(op: str):
    """Mittlere Gehaeusewandtemperatur je Zeitschritt, falls der Cache da ist."""
    root = repo_root()
    for rel in ("PINNmodulusTwo/data_cache", "data_cache",
                "legacy/battery_surrogate_agenticWorkflow/data_cache"):
        f = root / rel / f"{op}.npz"
        if f.exists():
            r = np.load(f, allow_pickle=True)
            T = np.asarray(r["T"], dtype=np.float64)
            x = np.round(np.asarray(r["xyz"], dtype=np.float64)[:, 0], 9)
            wall = x == x.max()          # die aeusserste x-Ebene = Gehaeusewand
            return T[:, wall].mean(axis=1), np.asarray(r["t_fast"], np.float64)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS))
    ap.add_argument("--raw", default=None)
    ap.add_argument("--list-columns", action="store_true",
                    help="nur die Spaltennamen jeder Datei ausgeben")
    ap.add_argument("--area", type=float, default=0.0206,
                    help="gekuehlte Flaeche in m^2 fuer h_eff (Default: die "
                         "0.198 x 0.104 m Zellflaeche)")
    args = ap.parse_args()

    raw = find_raw(args.raw)
    print(f"Rohdaten: {raw}\n")

    rows = []
    for op in args.ops:
        d = op_dir(raw, op)
        if d is None:
            print(f"[fehlt] {op}")
            continue
        files = {
            "heat_source": one_glob(d, "*_Heat Source.csv"),
            "heat_transfer": one_glob(d, "*_Heat Transfer.csv"),
            "temperaturen": one_glob(d, "*_Temperaturen.csv"),
            "fluid": one_glob(d, "*_Fluidstoffwerte.csv"),
            "input": one_glob(d, "*_Input*ignale.csv"),
        }

        if args.list_columns:
            print(f"===== {op}  ({d}) =====")
            for tag, f in files.items():
                if f is None:
                    print(f"  [{tag}] -- keine Datei")
                    continue
                cols = read_csv_cols(f)
                print(f"  [{tag}] {f.name}")
                for c in sorted(cols):
                    print(f"      {c}")
            print()
            continue

        missing = [t for t, f in files.items() if f is None]
        if missing:
            print(f"[{op}] Dateien fehlen: {', '.join(missing)}")
            continue

        hs = read_csv_cols(files["heat_source"])
        ht = read_csv_cols(files["heat_transfer"])
        tm = read_csv_cols(files["temperaturen"])
        fl = read_csv_cols(files["fluid"])
        ins = read_csv_cols(files["input"])

        t = pick(hs, TIME, "Zeitachse", files["heat_source"])
        jr1 = pick(hs, HEAT_SRC["jr1_w"], "JR1-Quelle", files["heat_source"])
        jr2 = pick(hs, HEAT_SRC["jr2_w"], "JR2-Quelle", files["heat_source"],
                   required=False)
        tot = pick(hs, HEAT_SRC["total_w"], "Gesamtquelle", files["heat_source"],
                   required=False)

        t_q = pick(ht, TIME, "Zeitachse", files["heat_transfer"])
        q = pick(ht, Q_WALL, "Waermestrom solid->fluid", files["heat_transfer"])
        t_out = pick(tm, T_OUT, "Fluid-Auslasstemperatur", files["temperaturen"])
        t_in = pick(tm, T_IN, "Fluid-Einlasstemperatur", files["temperaturen"],
                    required=False)
        cp = pick(fl, CP_FLUID, "Cp_fluid", files["fluid"])
        mdot = pick(ins, MDOT, "Massenstrom", files["input"])

        cp_v = float(np.nanmean(cp))
        m_v = float(np.nanmean(mdot))

        rows.append(dict(op=op, t=t, jr1=jr1, jr2=jr2, tot=tot, t_q=t_q, q=q,
                         t_out=t_out, t_in=t_in, cp=cp_v, mdot=m_v))

    if args.list_columns or not rows:
        return

    # ---- 1. Halbmodell-Faktor ----------------------------------------------
    print("== 1. Halbmodell-Faktor: was steht in *_Heat Source.csv? ==")
    print(f"{'OP':<6} {'mean jr1_w':>12} {'mean jr2_w':>12} {'mean total_w':>13} "
          f"{'total/jr1':>10} {'jr2/jr1':>9}")
    for r in rows:
        j1 = float(np.nanmean(r["jr1"]))
        j2 = float(np.nanmean(r["jr2"])) if r["jr2"] is not None else np.nan
        tt = float(np.nanmean(r["tot"])) if r["tot"] is not None else np.nan
        print(f"{r['op']:<6} {j1:>12.4g} {j2:>12.4g} {tt:>13.4g} "
              f"{tt/j1 if j1 else np.nan:>10.4f} "
              f"{j2/j1 if j1 else np.nan:>9.4f}")
    print("   total/jr1 ~ 2 und jr2/jr1 ~ 1  ->  zwei Wickel, das Halbmodell")
    print("   enthaelt einen, jr1_w IST die Halbmodell-Leistung.")
    print("   total/jr1 ~ 1                  ->  ein Wickel; dann sagt Punkt 3,")
    print("   ob der Waermestrom-Monitor dieselbe Konvention hat.\n")

    # ---- 2. Fluidbilanz -----------------------------------------------------
    print("== 2. Fluidbilanz:  dT = Qdot / (mdot * Cp)  gegen  T_out - T_in ==")
    print(f"{'OP':<6} {'mdot':>9} {'Cp':>9} {'dT gerechnet':>14} "
          f"{'dT gemessen':>13} {'Verhaeltnis':>12}")
    for r in rows:
        q = np.interp(r["t"], r["t_q"], r["q"])
        if r["mdot"] <= 0:
            print(f"{r['op']:<6} {r['mdot']:>9.4g} {r['cp']:>9.4g} "
                  f"{'--':>14} {'--':>13} {'kein Durchfluss':>12}")
            continue
        dt_calc = float(np.nanmean(q)) / (r["mdot"] * r["cp"])
        if r["t_in"] is not None:
            dt_meas = float(np.nanmean(r["t_out"] - r["t_in"]))
        else:
            dt_meas = np.nan
        print(f"{r['op']:<6} {r['mdot']:>9.4g} {r['cp']:>9.4g} "
              f"{dt_calc:>14.4f} {dt_meas:>13.4f} "
              f"{dt_calc/dt_meas if dt_meas else np.nan:>12.4f}")
    print("   ~1.0 -> die Bilanz geht auf. ~2.0 oder ~0.5 -> Halbmodell-Faktor,")
    print("   siehe Punkt 1, NICHT ein Physikfehler.\n")

    # ---- 3. Energieanteil ueber die Wand ------------------------------------
    print("== 3. Anteil der Quellenergie, der ueber die Wand abfliesst ==")
    print(f"{'OP':<6} {'int Qdot dt [J]':>17} {'int jr1_w dt [J]':>18} "
          f"{'Anteil':>9}")
    for r in rows:
        e_q = float(_trapz(r["q"], r["t_q"]))
        e_s = float(_trapz(r["jr1"], r["t"]))
        print(f"{r['op']:<6} {e_q:>17.5g} {e_s:>18.5g} "
              f"{e_q/e_s if e_s else np.nan:>9.4f}")
    print("   Gegenstueck zu energy_balance_report (0.5-0.9x, dem Volumenstrom")
    print("   folgend). Bei mdot = 0 muss der Anteil nahe 0 liegen -- tut er es")
    print("   nicht, fliesst Waerme auf einem Weg ab, den der Entwurf nicht hat.\n")

    # ---- 4. h_eff gegen den Fluss ------------------------------------------
    print(f"== 4. h_eff = Qdot / (A * dT),  A = {args.area} m^2 ==")
    print(f"{'OP':<6} {'mdot':>9} {'h_eff [W/m2K]':>15}")
    any_wall = False
    for r in rows:
        Tw, tw = wall_temp_from_cache(r["op"])
        if Tw is None:
            continue
        any_wall = True
        q = np.interp(tw, r["t_q"], r["q"])
        tf = np.interp(tw, r["t"] if r["t_in"] is None else r["t"],
                       r["t_out"] if r["t_in"] is None else r["t_in"])
        dT = Tw - tf
        ok = np.abs(dT) > 1e-6
        h = np.nanmean(q[ok] / (args.area * dT[ok])) if ok.any() else np.nan
        print(f"{r['op']:<6} {r['mdot']:>9.4g} {h:>15.4f}")
    if not any_wall:
        print("   uebersprungen -- kein data_cache gefunden (braucht T an der Wand)")
    else:
        print("   Liegen die Werte auf einer Kurve ueber mdot, ist h eine feste")
        print("   Funktion von V_dot und kostet im Modell keinen freien Parameter.")


if __name__ == "__main__":
    main()
