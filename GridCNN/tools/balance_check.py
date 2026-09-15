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
1. **Der Halbmodell-Faktor des Waermestrom-Monitors.** Die Quellenseite ist
   geklaert (Antwort-MD vom 02.09.): ``q_source[:,0]`` ist
   ``Heat Source JR1 Monitor (W)``, eine Rolle, ueber ``V_JR1 = 4.394793e-04 m^3``
   volumetrisch gemacht -- **Halbmodell**, dieselbe Konvention wie das Gitter.
   Offen ist nur, ueber welche Flaeche StarCCM den Solid-to-Fluid-Monitor
   integriert, und das entscheidet ``Q_ht / JR1`` im **spaeten Fenster**:

   ===================  ====================  ==========================
   ``Q_ht / JR1``       ``Q_ht / (JR1+JR2)``  Lesart
   ===================  ====================  ==========================
   ~ 1                  ~ 0.5                 eine Platte, gleiche Konvention
   ~ 2                  ~ 1                   beide Platten, Faktor 2
   ===================  ====================  ==========================

   .. warning::
      ``Heat Source Monitor (total)`` ist **nicht** ``JR1 + JR2``, sondern
      groesser (Antwort-MD, Punkt 2). Es taugt daher **nicht** als
      Halbmodell-Probe. Was taugt: ``jr2/jr1 ~ 1`` (zwei gleiche Rollen) und
      ``Q_ht/JR1``.

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
   ``integral(jr1_w dt)``.

   .. warning::
      Das ist **nicht** das Gegenstueck zu ``energy_balance_report``. Der Report
      sieht nur JR1: ``0.9x`` bei V_dot = 0 heisst *90 % der JR1-Quelle bleiben
      in JR1*, die anderen 10 % gehen **ins Gehaeuse und ins Cell Center** --
      nicht ins Fluid. ``integral(Q_ht) ~ (1 - 0.9) * integral(JR1)`` zu
      erzwingen verdreht **Speicherung zu Kuehlung** (Antwort-MD, Punkt 4).

   Was hier wirklich zu erwarten ist: bei V_dot = 0 muss ``Q_ht ~ 0`` sein
   (kein konvektiver Abtransport). Bei hohem Fluss gross, aber **kleiner** als
   ``0.5 * integral(JR1)``, weil Waerme im Gehaeuse gespeichert bleibt.

4. **``U`` gegen den Volumenstrom.** Liegt es auf einer Kurve, ist ``U`` eine
   feste Funktion von ``V_dot`` und kostet im Modell keinen freien Parameter.
   Braucht die Wandtemperatur, also den ``.npz``-Cache; ohne ihn wird der Teil
   uebersprungen.

   .. note::
      **``U``, nicht ``h_conv``.** Zwischen der Monitorebene ``x = 0.0219`` und
      der Kuehlplatte ``x = 0.0238`` liegen 1.9 mm, die im Repo nicht zerlegt
      sind (Restwand, TIM, Spalt). Was aus ``Q / (A * (T_wand - T_fluid))``
      faellt, ist damit ein **Gesamtdurchgang von der Gitterebene bis in den
      Fluidkern**, kein Filmkoeffizient an der Kanalwand. Der Widerstand der
      Schicht steckt **in** ``U`` und darf nicht noch einmal aufgeschlagen
      werden. Waeren die 1.9 mm reines Aluminium, waere ``lambda/L ~ 1e5
      W/m^2K`` und ``U ~ h_conv`` -- aber das ist eine Abschaetzung, kein
      Schnittbild.

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
# Alle sieben Konstant-Treiber-OPs AUS DEM TRAINING. Die frueheren Defaults
# (OP04, OP05, OP07, OP14) hatten nur ZWEI Flusslevel -- 30 und 0 -- weil OP04
# und OP05 beide V_dot = 30 fahren. Damit lassen sich zwei Punkte messen, aber
# keine Funktion U(V_dot) anpassen. Diese sieben geben die drei trainierten
# Level 0 / 15 / 30, jedes mehrfach belegt.
#
# OP16 (V_dot = 90) fehlt hier ABSICHTLICH: es ist ein TEST-OP. U daran zu
# kalibrieren waere eine Auswahl auf dem Extrapolationstier -- genau das, wovor
# op_registry.py warnt. OP16 ist die Gegenprobe fuer U(V_dot), nie die Stuetze.
DEFAULT_OPS = ("OP01", "OP02", "OP03", "OP04", "OP05", "OP07", "OP14")

TIME = ("Physical Time (s)",)
HEAT_SRC = {"jr1_w": ("Heat Source JR1 Monitor (W)",),
            "jr2_w": ("Heat Source JR2 Monitor (W)",),
            "total_w": ("Heat Source Monitor (W)",)}
Q_WALL = ("Heat Transfer: solid to fluid Monitor (W)",
          "Heat Transfer: solid to fluid (W)")
T_OUT = ("Tmfavg_fluid_out Monitor (C)", "Tmfavg_fluid_out (C)")
# Die Einlasstemperatur stand am 09.09. unter KEINEM dieser Namen in
# *_Temperaturen.csv -- "dT gemessen" kam als nan heraus. Deshalb eine
# Rueckfallkette statt eines einzelnen Ratens, und das Skript sagt, welche
# Quelle es benutzt hat. Fuer die Konstant-Treiber-OPs ist die Fluidtemperatur
# ohnehin ein Skalar und steht in *_Input Signale.csv.
T_IN = ("Tmfavg_fluid_in Monitor (C)", "Tmfavg_fluid_in (C)",
        "Tmfavg_fluid_inlet Monitor (C)",
        "Fluid Inlet Temperature Monitor (C)",
        "Fluid Inlet Temperature (C)",
        "Fluid Initial Temperature Monitor (C)")
# Rueckfall 2: derselbe Wert als Skalar im Inputsignale-Export.
T_IN_SCALAR = ("Fluid Inlet Temperature Monitor (C)",
               "Fluid Initial Temperature Monitor (C)",
               "Fluid Temperature Monitor (C)")
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
                    help="gekuehlte Flaeche in m^2 fuer U. Default 0.0206 = die "
                         "yz-Flaeche der +x-Gehaeusewand, EINE Seite "
                         "(0.198 x 0.104 m), am Gitter ~0.0207 gemessen")
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
        t_in_src = "Temperaturen.csv"
        if t_in is None:
            t_in = pick(ins, T_IN_SCALAR, "Fluid-Einlasstemperatur",
                        files["input"], required=False)
            t_in_src = "Input Signale.csv" if t_in is not None else "FEHLT"
        cp = pick(fl, CP_FLUID, "Cp_fluid", files["fluid"])
        mdot = pick(ins, MDOT, "Massenstrom", files["input"])

        cp_v = float(np.nanmean(cp))
        m_v = float(np.nanmean(mdot))

        rows.append(dict(op=op, t=t, jr1=jr1, jr2=jr2, tot=tot, t_q=t_q, q=q,
                         t_out=t_out, t_in=t_in, t_in_src=t_in_src,
                         cp=cp_v, mdot=m_v))

    if args.list_columns or not rows:
        return

    # ---- 1. Halbmodell-Faktor ----------------------------------------------
    print("== 1. Halbmodell-Konvention des Waermestrom-Monitors ==")
    print("   Die QUELLE ist geklaert: jr1_w = eine Rolle = Halbmodell.")
    print("   Offen ist die Bezugsflaeche des Solid-to-Fluid-Monitors. Gemessen")
    print("   im SPAETEN Fenster (letztes Drittel), wo Einschwingen vorbei ist.\n")
    print(f"{'OP':<6} {'V_dot':>6} {'jr2/jr1':>8} {'tot/jr1':>8} "
          f"{'Q_ht/jr1':>9} {'Q_ht/tot':>9}  {'Lesart':<22}")
    for r in rows:
        late = slice(int(0.67 * len(r["t_q"])), None)
        q_l = float(np.nanmean(r["q"][late]))
        onq = lambda a: float(np.nanmean(np.interp(r["t_q"], r["t"], a)[late]))
        j1 = onq(r["jr1"])
        j2 = onq(r["jr2"]) if r["jr2"] is not None else np.nan
        tt = onq(r["tot"]) if r["tot"] is not None else np.nan
        r1 = q_l / j1 if j1 else np.nan
        rt = q_l / tt if tt else np.nan
        if r["mdot"] <= 0:
            v = "kein Fluss"
        elif not np.isnan(rt) and 0.85 < rt < 1.15:
            v = "= Gesamtquelle"
        elif 0.75 < r1 < 1.3:
            v = "eine Platte"
        elif 1.6 < r1 < 2.4:
            v = "beide Platten"
        else:
            v = "PASST NICHT"
        print(f"{r['op']:<6} {r['mdot']:>6.4g} {j2/j1 if j1 else np.nan:>8.3f} "
              f"{tt/j1 if j1 else np.nan:>8.3f} {r1:>9.3f} {rt:>9.3f}  {v:<22}")
    print()
    print("   Q_ht/tot ~ 1  -> der Monitor draint die GANZE Erzeugung. Dann ist")
    print("                    Q_ht/jr1 kein Konventionsfehler, sondern sagt, dass")
    print("                    die Zelle mehr erzeugt als 2x JR1 -- und die")
    print("                    Modellquelle q_dot deckt nur JR1 ab.")
    print("   Q_ht/jr1 ~ 1  -> eine Platte, gleiche Konvention wie die Quelle.")
    print("   Q_ht/jr1 ~ 2  -> beide Platten: ENTWEDER Q halbieren ODER A")
    print("                    verdoppeln, nie beides.")
    print("   'Heat Source Monitor (total)' ist NICHT jr1+jr2, sondern groesser")
    print("   -- deshalb steht tot/jr1 hier als eigene Spalte und nicht als Probe.\n")

    # ---- 2. Fluidbilanz -----------------------------------------------------
    print("== 2. Fluidbilanz:  dT = Qdot / (mdot * Cp)  gegen  T_out - T_in ==")
    print(f"{'OP':<6} {'mdot':>9} {'Cp':>9} {'dT gerechnet':>14} "
          f"{'dT gemessen':>13} {'Verh.':>7}  {'T_in aus':<20}")
    for r in rows:
        q = np.interp(r["t"], r["t_q"], r["q"])
        if r["mdot"] <= 0:
            print(f"{r['op']:<6} {r['mdot']:>9.4g} {r['cp']:>9.4g} "
                  f"{'--':>14} {'--':>13} {'--':>7}  "
                  f"kein Durchfluss")
            continue
        dt_calc = float(np.nanmean(q)) / (r["mdot"] * r["cp"])
        if r["t_in"] is not None:
            dt_meas = float(np.nanmean(r["t_out"] - r["t_in"]))
        else:
            dt_meas = np.nan
        print(f"{r['op']:<6} {r['mdot']:>9.4g} {r['cp']:>9.4g} "
              f"{dt_calc:>14.4f} {dt_meas:>13.4f} "
              f"{dt_calc/dt_meas if dt_meas else np.nan:>7.3f}  "
              f"{r['t_in_src']:<20}")
    print("   ~1.0 -> die Bilanz geht auf. ~2.0 oder ~0.5 -> Halbmodell-Faktor,")
    print("   siehe Punkt 1, NICHT ein Physikfehler.")
    print("   'T_in aus FEHLT' -> die Spalte wurde nirgends gefunden; dann einmal")
    print("   mit --list-columns laufen und den Namen nachtragen.\n")

    # ---- 3. Energieanteil ueber die Wand ------------------------------------
    print("== 3. Anteil der Quellenergie, der ueber die Wand abfliesst ==")
    print(f"{'OP':<6} {'int Qdot dt [J]':>17} {'int jr1_w dt [J]':>18} "
          f"{'Anteil':>9}")
    for r in rows:
        e_q = float(_trapz(r["q"], r["t_q"]))
        e_s = float(_trapz(r["jr1"], r["t"]))
        print(f"{r['op']:<6} {e_q:>17.5g} {e_s:>18.5g} "
              f"{e_q/e_s if e_s else np.nan:>9.4f}")
    print("   NICHT mit energy_balance_report gleichsetzen: der sieht nur JR1,")
    print("   und er vergleicht RATEN, nicht Energien.")
    print()
    print("   KORRIGIERT am 09.09.: bei mdot = 0 ist NICHT ~0 zu erwarten.")
    print("   mdot = 0 heisst kein FLUSS, nicht kein FLUID -- das Kuehlmittel")
    print("   steht im Kanal und nimmt Waerme in seine eigene Waermekapazitaet")
    print("   auf. Gemessen wurden ~0.27 auf OP07/OP14, und das ist physikalisch")
    print("   richtig: die Energie wird GESPEICHERT, nicht abtransportiert.")
    print("   Was daraus folgt, ist ein Modellbefund, kein Messfehler --")
    print("   dT = Qdot/(mdot*Cp) divergiert bei mdot -> 0 und braucht einen")
    print("   Kapazitaetsterm. Siehe FAHRPLAN, Stufe 1.\n")

    # ---- 4. h_eff gegen den Fluss ------------------------------------------
    print(f"== 4. U = Qdot / (A * dT),  A = {args.area} m^2 ==")
    print("   U ist ein GESAMTDURCHGANG Gitterebene -> Fluidkern, kein")
    print("   Filmkoeffizient: die 1.9 mm zur Kuehlplatte stecken drin.")
    print("   VORLAEUFIG, solange Punkt 1 offen ist: zaehlt der Monitor beide")
    print("   Platten, ist U hier um Faktor 2 zu hoch. Robust ist das VERHAELTNIS")
    print("   zwischen den Flusslevels, nicht der Absolutwert.")
    print(f"{'OP':<6} {'V_dot':>7} {'mdot':>9} {'U [W/m2K]':>12}")
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
        print(f"{r['op']:<6} {'':>7} {r['mdot']:>9.4g} {h:>12.2f}")
    if not any_wall:
        print("   uebersprungen -- kein data_cache gefunden (braucht T an der Wand)")
    else:
        print("   Liegen die Werte auf einer Kurve ueber mdot, ist U eine feste")
        print("   Funktion von V_dot und kostet im Modell keinen freien Parameter.")
        print("   Nicht 'h_conv' nennen -- siehe Kopf des Skripts.")


if __name__ == "__main__":
    main()
