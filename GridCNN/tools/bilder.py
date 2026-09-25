#!/usr/bin/env python3
"""Bilder zu einem oder mehreren Laeufen -- aus den JSON, ohne Rollout.

Warum es das gibt
-----------------
Am 22.09. musste der Plot, der ``OP06 6.57 C`` als "1.1 C in der Mitte, 15 C
am Ende" entlarvt hat, von Hand gebaut werden. Seitdem stehen Profile als
Textzeilen im Log. Das reicht zum Ablesen, nicht zum Vergleichen von drei
Laeufen mit je drei Seeds. Dieses Werkzeug zeichnet, was die Laeufe und
``tools/nachmessen.py`` ohnehin schreiben, und braucht dafuer weder GPU noch
Daten noch ``torch``.

Aufruf
------
::

    python3 GridCNN/tools/bilder.py GridCNN/artifacts/A \\
        GridCNN/artifacts/B-exp GridCNN/artifacts/A-kompakt-film \\
        --aus GridCNN/laeufe/bilder

Je Laufverzeichnis (``<lauf>`` = Verzeichnisname, z. B. ``A``):

``<lauf>_lernkurven.png``
    ``data``-Verlust, Gradientennorm (der Sturm von Lauf 17 Seed 2), Anteil
    am Clamp, und die Halte-MAE **geteilt durch die Latte** je Seed.
``<lauf>_bias_epochen.png``
    Bias je Abschnitt ueber die Epochen, je Seed und Halte-OP. Rot = zu warm,
    blau = zu kalt. Zeigt, **wann** "anfangs warm, am Ende kalt" entsteht.

Aus jeder ``nachgemessen*.json`` (``<messung>`` = ``halte`` fuer die
Vorgabe, ``insample`` fuer ``nachgemessen_insample.json``), Stand
``model.pt``:

``<lauf>_<messung>_sensoren*.png``
    **der Fehlerbetrag ueber die Zeit, ueber alle 363 Gitterpunkte**: MAE,
    kleinster und groesster Punktfehler, und zwei Linien, zwischen denen die
    Haelfte der Punkte liegt (25 % / 75 %).
``<lauf>_<messung>_vorzeichen*.png``
    dasselbe mit Vorzeichen: zu warm (+) oder zu kalt (-), und ob das ganze
    Feld verschoben ist (Band weg von null) oder nur ein Rand (Band bei null,
    Min/Max weit weg).
``<lauf>_<messung>_temperatur.png``
    mittlere und heisseste Stelle, Daten gegen Modell.
``<lauf>_<messung>_karte.png``
    zeitgemittelter Fehler je Gitterpunkt, drei Ebenen -- **wo** er sitzt.
``<lauf>_<messung>_profil.png``
    Bias je Abschnitt (die Zeilen aus dem Log als Bild).

Grau hinterlegt ist bei ``insample`` der Teil **hinter** ``split_t``: dort
hatte auch ein Trainings-OP kein Label. Bei Halte-OPs gibt es nirgends ein
Label, dort ist nichts grau.

Ueber alle Laeufe:

``vergleich_latte.png``
    berichtete MAE / Latte je Seed und Lauf, mit Median und Streuung.
``vergleich_profil.png``
    Median-Bias je Abschnitt, ein Strich je Lauf -- die Frage aus dem
    FAHRPLAN, ob ``Qsrc`` (Arm B) das "spaeter zu kalt" zurueckholt.
``vergleich_zeit.png``
    MAE und Bias ueber die Zeit, Median ueber die Seeds, ein Strich je Lauf
    (braucht ``nachgemessen.json`` in jedem Lauf).

Die Zahlen in den Bildern sind dieselben wie im Log (dieselben JSON). Ein
Bild ist eine Lesehilfe, keine neue Messung.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except ImportError:                                     # pragma: no cover
    print("!! matplotlib fehlt: pip install matplotlib", file=sys.stderr)
    raise SystemExit(2)

# Kategoriale Farben in fester Reihenfolge (Seeds bzw. Laeufe) -- nie nach
# Rang, damit Seed 0 in jedem Bild dieselbe Farbe hat. Die ersten drei sind
# auch fuer Farbfehlsichtige paarweise trennbar.
FARBEN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948")
TINTE = "#0b0b0b"
TINTE_2 = "#52514e"
GITTER = "#dcdbd6"
HINTER_SPLIT = "#efeeea"
# Innerhalb eines Sensor-Bildes sind die Linien Kennzahlen, keine Seeds.
F_MAX, F_MIN, F_BAND = "#eb6834", "#1baf7a", "#2a78d6"
SPLIT_ANTEIL = 0.8          # data.load_ops(train_frac=0.8)
EINZELN_BIS = 4             # bis zu so vielen OPs: Seeds als Zeilen, ein Bild

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": GITTER, "axes.labelcolor": TINTE_2,
    "axes.titlesize": 10, "axes.titlecolor": TINTE,
    "xtick.color": TINTE_2, "ytick.color": TINTE_2,
    "axes.grid": True, "grid.color": GITTER, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


# ---------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------
def _lies(pfad: Path):
    try:
        return json.loads(pfad.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"!! {pfad}: {e}", file=sys.stderr)
        return None


def lauf_lesen(verz: Path) -> dict:
    """``{"name", "seeds": [{"seed", "history", "metrics"}], "messungen"}``."""
    seeds = []
    for d in sorted(verz.glob("seed*")):
        h = _lies(d / "history.json") if (d / "history.json").exists() else None
        m = _lies(d / "metrics.json") if (d / "metrics.json").exists() else None
        if h is None and m is None:
            continue
        seeds.append({"seed": d.name, "history": h or [], "metrics": m or {}})
    messungen = {}
    for j in sorted(verz.glob("nachgemessen*.json")):
        inhalt = _lies(j)
        if inhalt:
            tag = j.stem.replace("nachgemessen", "").strip("_") or "halte"
            messungen[tag] = inhalt
    return {"name": verz.name, "seeds": seeds, "messungen": messungen}


def latte(latten: dict, op_id: str) -> float:
    """Die beste triviale Latte eines OPs -- dieselbe Regel wie im Log."""
    werte = [v for v in (latten.get(op_id) or {}).values()
             if v is not None and np.isfinite(v) and v > 0]
    return min(werte) if werte else float("nan")


def _latten(lauf: dict) -> dict:
    out = {}
    for s in lauf["seeds"]:
        out.update(s["metrics"].get("triviale_latten_C") or {})
    return out


def je_seed(messung: dict) -> dict:
    """``{seed: {op: profil}}`` fuer ``model.pt`` -- der Stand der letzten
    Epoche. ``model_best.pt`` ist auf der Haltemenge ausgewaehlt und bleibt
    deshalb aus den Bildern heraus."""
    out = {}
    for schl, profil in (messung.get("je_checkpoint") or {}).items():
        seed, _, name = schl.partition("/")
        if name == "model.pt":
            out[seed] = profil
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Rahmen
# ---------------------------------------------------------------------------
def _farbe(i: int) -> str:
    return FARBEN[i % len(FARBEN)]


def _gitter(n: int, breite: int = 4, *, sharex: bool = True,
            hoehe: float = 2.5):
    """``n`` Achsen in Zeilen zu ``breite``; ueberzaehlige verschwinden, und
    wer darunter nichts mehr hat, bekommt seine x-Beschriftung zurueck."""
    spalten = max(1, min(n, breite))
    zeilen = int(np.ceil(n / spalten))
    fig, achsen = plt.subplots(zeilen, spalten,
                               figsize=(3.4 * spalten, hoehe * zeilen + 1.0),
                               squeeze=False, sharex=sharex)
    for k, a in enumerate(achsen.flat):
        if k >= n:
            a.set_visible(False)
        elif k + spalten >= n:
            a.xaxis.set_tick_params(labelbottom=True)
    return fig, list(achsen.flat[:n])


def _speichern(fig, pfad: Path, titel: str, griffe: list | None = None,
               xlabel: str | None = None) -> Path:
    """Titel oben links, Legende darunter in einer Zeile, dann speichern.

    Die Legende steht nicht rechts oben, weil sie dort bei schmalen Bildern
    in den Titel laeuft."""
    fig.suptitle(titel, x=0.01, y=0.995, ha="left", va="top", color=TINTE,
                 fontsize=11)
    h = fig.get_size_inches()[1]
    oben = 1.0 - 0.42 / h
    if griffe:
        fig.legend(handles=griffe, loc="upper left", ncol=len(griffe),
                   fontsize=8, bbox_to_anchor=(0.01, oben))
        oben -= 0.30 / h
    if xlabel:
        fig.supxlabel(xlabel, color=TINTE_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, oben))
    fig.savefig(pfad, facecolor="white")
    plt.close(fig)
    print(f"  -> {pfad}")
    return pfad


def _seed_griffe(seeds: list) -> list:
    return [plt.Line2D([], [], color=_farbe(i), label=s)
            for i, s in enumerate(seeds)]


SPLIT_GRIFF = plt.Rectangle((0, 0), 1, 1, color=HINTER_SPLIT,
                            label="hinter split_t (ohne Label)")


def _grau(a, start: float, ende: float) -> None:
    if ende > start:
        a.axvspan(start, ende, color=HINTER_SPLIT, zorder=0, lw=0)


# ---------------------------------------------------------------------------
# Je Lauf, aus history.json / metrics.json
# ---------------------------------------------------------------------------
def bild_lernkurven(lauf: dict, aus: Path) -> Path | None:
    seeds = [s for s in lauf["seeds"] if s["history"]]
    if not seeds:
        return None
    latten = _latten(lauf)
    val_ops = sorted({op for s in seeds for z in s["history"]
                      for op in (z.get("val_mae_C") or {})})
    oben = [("data", "data-Verlust (Training)", True),
            ("grad_norm", "|g| vor dem Clip", True),
            ("clamp", "Anteil am Clamp", False)]
    spalten = max(len(oben), len(val_ops), 1)
    fig, achsen = plt.subplots(2, spalten, figsize=(3.4 * spalten, 6.2),
                               squeeze=False)
    for j, (schl, titel, log) in enumerate(oben):
        a = achsen[0][j]
        for i, s in enumerate(seeds):
            ep = [z["epoch"] for z in s["history"]]
            if schl == "clamp":
                y = [z.get("saturated", 0) / max(1, z.get("saturated_max", 1))
                     for z in s["history"]]
            else:
                y = [z.get(schl, np.nan) for z in s["history"]]
            y = np.asarray(y, dtype=float)
            if log:
                y = np.where(y > 0, y, np.nan)
            a.plot(ep, y, color=_farbe(i))
        if log:
            a.set_yscale("log")
        a.set_title(titel)
        a.set_xlabel("Epoche")
    for j in range(len(oben), spalten):
        achsen[0][j].set_visible(False)
    for j, op in enumerate(val_ops):
        a = achsen[1][j]
        lat = latte(latten, op)
        for i, s in enumerate(seeds):
            pkt = [(z["epoch"], z["val_mae_C"][op]) for z in s["history"]
                   if op in (z.get("val_mae_C") or {})]
            if not pkt:
                continue
            ep, v = zip(*pkt)
            v = np.asarray(v) / lat if np.isfinite(lat) else np.asarray(v)
            a.plot(ep, v, color=_farbe(i), marker="o", markersize=2.5)
        if np.isfinite(lat):
            a.axhline(1.0, color=TINTE_2, lw=1, ls="--")
            a.set_yscale("log")
            a.set_title(f"{op}: Halte-MAE / Latte ({lat:.2f} C)")
            a.set_ylabel("x Latte (unter 1 = gelernt)")
        else:
            a.set_title(f"{op}: Halte-MAE (C)")
        a.set_xlabel("Epoche")
    for j in range(len(val_ops), spalten):
        achsen[1][j].set_visible(False)
    return _speichern(fig, aus / f"{lauf['name']}_lernkurven.png",
                      f"{lauf['name']}: Lernkurven je Seed",
                      _seed_griffe([s["seed"] for s in seeds]))


def bild_bias_epochen(lauf: dict, aus: Path) -> Path | None:
    seeds = [s for s in lauf["seeds"]
             if any(z.get("val_detail") for z in s["history"])]
    if not seeds:
        return None
    val_ops = sorted({op for s in seeds for z in s["history"]
                      for op in (z.get("val_detail") or {})})
    felder = {}
    for s in seeds:
        for op in val_ops:
            reihe = [(z["epoch"], z["val_detail"][op]["bias_segmente"])
                     for z in s["history"]
                     if op in (z.get("val_detail") or {})]
            if reihe:
                ep, b = zip(*reihe)
                n = min(len(x) for x in b)
                felder[(s["seed"], op)] = (np.asarray(ep),
                                           np.asarray([x[:n] for x in b]).T)
    if not felder:
        return None
    # Die Skala aus der zweiten Haelfte: die ersten Epochen stehen am Clamp
    # (Bias bis ~90 C) und wuerden sonst alles Spaetere weiss waschen. Was
    # darueber liegt, wird in der kraeftigsten Farbe gezeigt, nicht versteckt.
    spaet = np.concatenate([f[:, f.shape[1] // 2:].ravel()
                            for _, f in felder.values()])
    grenze = float(np.nanpercentile(np.abs(spaet), 98)) or 1.0
    norm = TwoSlopeNorm(vmin=-grenze, vcenter=0.0, vmax=grenze)
    fig, achsen = plt.subplots(len(seeds), len(val_ops),
                               figsize=(4.8 * len(val_ops),
                                        1.9 * len(seeds) + 1.0),
                               squeeze=False)
    bild = None
    for r, s in enumerate(seeds):
        for c, op in enumerate(val_ops):
            a = achsen[r][c]
            a.grid(False)
            if (s["seed"], op) not in felder:
                a.set_visible(False)
                continue
            ep, f = felder[(s["seed"], op)]
            n_seg = f.shape[0]
            bild = a.imshow(f, aspect="auto", cmap="RdBu_r", norm=norm,
                            extent=(ep[0] - 0.5, ep[-1] + 0.5,
                                    n_seg + 0.5, 0.5),
                            interpolation="nearest")
            a.set_yticks(range(1, n_seg + 1))
            a.set_title(f"{s['seed']} · {op}")
            if c == 0:
                a.set_ylabel("Abschnitt (1 = Anfang)")
            if r == len(seeds) - 1:
                a.set_xlabel("Epoche")
    if bild is not None:
        cb = fig.colorbar(bild, ax=achsen, shrink=0.85, pad=0.01,
                          extend="both")
        cb.set_label("Bias C  (rot = zu warm, blau = zu kalt)")
    fig.suptitle(f"{lauf['name']}: Bias je Abschnitt ueber die Epochen "
                 f"(Skala aus der 2. Haelfte, +-{grenze:.1f} C)",
                 x=0.01, ha="left", color=TINTE, fontsize=11)
    pfad = aus / f"{lauf['name']}_bias_epochen.png"
    fig.savefig(pfad, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {pfad}")
    return pfad


# ---------------------------------------------------------------------------
# Je Messung (nachgemessen*.json)
# ---------------------------------------------------------------------------
def _panel_sensoren(a, k: dict, *, vorzeichen: bool, grau: bool) -> None:
    """Eine Achse: Kennzahlen ueber die Gitterpunkte, je Zeitschritt."""
    t = np.asarray(k["t_s"])
    if vorzeichen:
        mitte, lo, hi, mn, mx = (k["bias"], k["fehler_q25"], k["fehler_q75"],
                                 k["fehler_min"], k["fehler_max"])
    else:
        mitte, lo, hi, mn, mx = (k["mae"], k["abs_q25"], k["abs_q75"],
                                 k["abs_min"], k["abs_max"])
    a.fill_between(t, lo, hi, color=F_BAND, alpha=0.14, lw=0)
    a.plot(t, lo, color=F_BAND, lw=1.0)
    a.plot(t, hi, color=F_BAND, lw=1.0)
    a.plot(t, mx, color=F_MAX, lw=1.1)
    a.plot(t, mn, color=F_MIN, lw=1.1)
    a.plot(t, mitte, color=TINTE, lw=2.2)
    if vorzeichen:
        a.axhline(0, color=TINTE_2, lw=0.8, ls="--")
    if grau and "split_t_s" in k:
        _grau(a, k["split_t_s"], t[-1])
    a.set_xlim(t[0], t[-1])


def _sensor_griffe(vorzeichen: bool, grau: bool) -> list:
    mitte = "Bias (Feldmittel)" if vorzeichen else "MAE (Feldmittel)"
    g = [plt.Line2D([], [], color=TINTE, lw=2.2, label=mitte),
         plt.Line2D([], [], color=F_MAX, label="Max ueber Punkte"),
         plt.Line2D([], [], color=F_MIN, label="Min ueber Punkte"),
         plt.Line2D([], [], color=F_BAND, label="25 % / 75 % (die halbe "
                                                "Punktmenge liegt dazwischen)")]
    return g + ([SPLIT_GRIFF] if grau else [])


def bilder_sensoren(lauf: dict, tag: str, js: dict, aus: Path, *,
                    vorzeichen: bool) -> list:
    seeds = list(js)
    ops = sorted({op for p in js.values() for op in p
                  if "kurve" in p[op]})
    if not ops:
        return []
    grau = tag == "insample"
    datei = "vorzeichen" if vorzeichen else "sensoren"
    ylabel = ("Fehler Modell - Daten (C)" if vorzeichen
              else "|Fehler| (C)")
    was = ("Fehler mit Vorzeichen ueber die Zeit (+ = zu warm)" if vorzeichen
           else "Fehlerbetrag ueber die Zeit, ueber alle Gitterpunkte")
    griffe = _sensor_griffe(vorzeichen, grau)
    pfade = []
    if len(ops) <= EINZELN_BIS:
        # Seeds als Zeilen, OPs als Spalten: ein Bild, alles nebeneinander.
        fig, achsen = plt.subplots(len(seeds), len(ops),
                                   figsize=(5.2 * len(ops),
                                            2.5 * len(seeds) + 1.4),
                                   squeeze=False, sharex="col")
        for r, s in enumerate(seeds):
            for c, op in enumerate(ops):
                a = achsen[r][c]
                k = (js[s].get(op) or {}).get("kurve")
                if not k:
                    a.set_visible(False)
                    continue
                _panel_sensoren(a, k, vorzeichen=vorzeichen, grau=grau)
                a.set_title(f"{s} · {op}   (MAE {js[s][op]['mae']:.2f} C)")
                if c == 0:
                    a.set_ylabel(ylabel)
        pfade.append(_speichern(
            fig, aus / f"{lauf['name']}_{tag}_{datei}.png",
            f"{lauf['name']} · {tag}: {was}", griffe, "Zeit (s)"))
        return pfade
    # Viele OPs (insample): ein Bild je Seed, ein Feld je OP.
    for s in seeds:
        fig, achsen = _gitter(len(ops), sharex=False)
        for a, op in zip(achsen, ops):
            k = (js[s].get(op) or {}).get("kurve")
            if not k:
                continue
            _panel_sensoren(a, k, vorzeichen=vorzeichen, grau=grau)
            a.set_title(f"{op}   (MAE {js[s][op]['mae']:.2f} C)")
        achsen[0].set_ylabel(ylabel)
        pfade.append(_speichern(
            fig, aus / f"{lauf['name']}_{tag}_{datei}_{s}.png",
            f"{lauf['name']} · {tag} · {s}: {was}", griffe, "Zeit (s)"))
    return pfade


def bild_temperatur(lauf: dict, tag: str, js: dict, aus: Path) -> Path | None:
    seeds = list(js)
    ops = sorted({op for p in js.values() for op in p if "kurve" in p[op]})
    if not ops:
        return None
    grau = tag == "insample"
    fig, achsen = _gitter(len(ops), sharex=False,
                          hoehe=3.0 if len(ops) <= EINZELN_BIS else 2.5)
    for a, op in zip(achsen, ops):
        ende = None
        for i, s in enumerate(seeds):
            k = (js[s].get(op) or {}).get("kurve")
            if not k:
                continue
            if ende is None:
                a.plot(k["t_s"], k["T_wahr_max"], color=TINTE, lw=1.6,
                       ls="--")
                a.plot(k["t_s"], k["T_wahr_mittel"], color=TINTE, lw=2.4)
                ende = k["t_s"][-1]
                if grau:
                    _grau(a, k["split_t_s"], ende)
            a.plot(k["t_s"], k["T_modell_max"], color=_farbe(i), lw=1.0,
                   ls="--")
            a.plot(k["t_s"], k["T_modell_mittel"], color=_farbe(i), lw=1.2)
        if ende is not None:
            a.set_xlim(0, ende)
        a.set_title(op)
    achsen[0].set_ylabel("Temperatur (C)")
    griffe = ([plt.Line2D([], [], color=TINTE, lw=2.4, label="Daten Mittel"),
               plt.Line2D([], [], color=TINTE, lw=1.6, ls="--",
                          label="Daten heissester Punkt")]
              + _seed_griffe(seeds)
              + [plt.Line2D([], [], color=TINTE_2, lw=1.0, ls="--",
                            label="gestrichelt = heissester Punkt")]
              + ([SPLIT_GRIFF] if grau else []))
    return _speichern(fig, aus / f"{lauf['name']}_{tag}_temperatur.png",
                      f"{lauf['name']} · {tag}: Daten gegen Modell "
                      f"(Mittel und Hotspot), model.pt je Seed",
                      griffe, "Zeit (s)")


def bild_karte(lauf: dict, tag: str, js: dict, aus: Path) -> Path | None:
    """Zeitgemittelter |Fehler| je Gitterpunkt, Median ueber die Seeds."""
    ops = sorted({op for p in js.values() for op in p if "kurve" in p[op]})
    karten = {}
    for op in ops:
        reihe = [np.asarray(p[op]["kurve"]["karte_mae"], dtype=float)
                 .reshape(p[op]["kurve"]["karte_form"])
                 for p in js.values() if op in p and "kurve" in p[op]
                 and "karte_mae" in p[op]["kurve"]]
        if reihe:
            karten[op] = np.median(np.stack(reihe), axis=0)
    if not karten:
        return None
    ops = list(karten)
    nx = next(iter(karten.values())).shape[0]
    vmax = float(max(k.max() for k in karten.values())) or 1.0
    fig, achsen = plt.subplots(len(ops), nx,
                               figsize=(2.3 * nx + 1.2, 2.1 * len(ops) + 0.9),
                               squeeze=False)
    bild = None
    for r, op in enumerate(ops):
        for x in range(nx):
            a = achsen[r][x]
            a.grid(False)
            feld = karten[op][x]
            bild = a.imshow(feld.T, origin="lower", cmap="Blues", vmin=0,
                            vmax=vmax, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{op} · Ebene x{x}   max {feld.max():.1f}",
                        fontsize=8)
            if x == 0:
                a.set_ylabel("z")
            if r == len(ops) - 1:
                a.set_xlabel("y")
    cb = fig.colorbar(bild, ax=achsen, shrink=0.8, pad=0.02)
    cb.set_label("|Fehler| zeitgemittelt (C)")
    fig.suptitle(f"{lauf['name']} · {tag}: wo der Fehler sitzt "
                 f"(Median ueber Seeds)", x=0.01, ha="left", color=TINTE,
                 fontsize=11)
    pfad = aus / f"{lauf['name']}_{tag}_karte.png"
    fig.savefig(pfad, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {pfad}")
    return pfad


def bild_profil(lauf: dict, tag: str, js: dict, aus: Path) -> Path | None:
    seeds = list(js)
    ops = sorted({op for p in js.values() for op in p})
    if not ops:
        return None
    grau = tag == "insample"
    fig, achsen = _gitter(len(ops))
    for a, op in zip(achsen, ops):
        reihen = []
        for i, s in enumerate(seeds):
            p = js[s].get(op)
            if not p:
                continue
            b = np.asarray(p["bias_segmente"], dtype=float)
            a.plot((np.arange(len(b)) + 0.5) / len(b) * 100, b,
                   color=_farbe(i), marker="o", markersize=3, lw=1.1)
            reihen.append(b)
        if reihen:
            n = min(len(r) for r in reihen)
            med = np.median([r[:n] for r in reihen], axis=0)
            a.plot((np.arange(n) + 0.5) / n * 100, med, color=TINTE, lw=2.4)
        if grau:
            _grau(a, SPLIT_ANTEIL * 100, 100)
        a.axhline(0, color=TINTE_2, lw=0.8, ls="--")
        a.set_xlim(0, 100)
        a.set_title(op)
    achsen[0].set_ylabel("Bias C (+ = zu warm)")
    griffe = (_seed_griffe(seeds)
              + [plt.Line2D([], [], color=TINTE, lw=2.4, label="Median")]
              + ([SPLIT_GRIFF] if grau else []))
    return _speichern(fig, aus / f"{lauf['name']}_{tag}_profil.png",
                      f"{lauf['name']} · {tag}: Bias je Sechstel der "
                      f"Trajektorie, model.pt", griffe, "Trajektorie (%)")


def bilder_messung(lauf: dict, tag: str, messung: dict, aus: Path) -> list:
    js = je_seed(messung)
    if not js:
        return []
    pfade = [bild_profil(lauf, tag, js, aus)]
    if not any("kurve" in p[op] for p in js.values() for op in p):
        print(f"  (keine Zeitreihen in nachgemessen_{tag} -- mit dem "
              f"aktuellen nachmessen.py neu messen)")
        return pfade
    pfade += bilder_sensoren(lauf, tag, js, aus, vorzeichen=False)
    pfade += bilder_sensoren(lauf, tag, js, aus, vorzeichen=True)
    pfade.append(bild_temperatur(lauf, tag, js, aus))
    pfade.append(bild_karte(lauf, tag, js, aus))
    return [p for p in pfade if p]


# ---------------------------------------------------------------------------
# Ueber alle Laeufe
# ---------------------------------------------------------------------------
def bild_vergleich_latte(laeufe: list, aus: Path) -> Path | None:
    daten = []
    for lauf in laeufe:
        werte = {}
        for s in lauf["seeds"]:
            for op, v in (s["metrics"].get("val_mae_C_berichtet")
                          or {}).items():
                werte.setdefault(op, []).append((s["seed"], float(v)))
        if werte:
            daten.append((lauf["name"], _latten(lauf), werte))
    if not daten:
        return None
    ops = sorted({op for _, _, w in daten for op in w})
    seeds_alle = sorted({s for _, _, w in daten for r in w.values()
                         for s, _ in r})
    fig, achsen = plt.subplots(1, len(ops),
                               figsize=(max(4.5, 2.0 * len(daten) + 1.8)
                                        * len(ops), 4.0), squeeze=False)
    for a, op in zip(achsen[0], ops):
        for x, (name, latten, werte) in enumerate(daten):
            reihe = werte.get(op) or []
            lat = latte(latten, op)
            if not reihe:
                continue
            ys = []
            for s, v in reihe:
                y = v / lat if np.isfinite(lat) else v
                ys.append(y)
                i = seeds_alle.index(s)
                a.plot(x + (i - (len(seeds_alle) - 1) / 2) * 0.09, y, "o",
                       markersize=8, color=_farbe(i),
                       markeredgecolor="white", markeredgewidth=1.5)
            med = float(np.median(ys))
            a.plot([x - 0.25, x + 0.25], [med, med], color=TINTE, lw=2.4)
            streu = max(v for _, v in reihe) - min(v for _, v in reihe)
            a.annotate(f"{med:.2f}x\nSpanne {streu:.1f} C",
                       (x + 0.28, med), fontsize=8, color=TINTE_2,
                       va="center")
        a.axhline(1.0, color=TINTE_2, lw=1, ls="--")
        a.set_xticks(range(len(daten)))
        a.set_xticklabels([d[0] for d in daten])
        a.set_xlim(-0.5, len(daten) - 0.5 + 0.7)
        a.set_title(f"{op}: berichtete MAE / Latte")
        a.grid(axis="x", visible=False)
    achsen[0][0].set_ylabel("x Latte (unter 1 = gelernt)")
    griffe = (_seed_griffe(seeds_alle)
              + [plt.Line2D([], [], color=TINTE, lw=2.4, label="Median")])
    return _speichern(fig, aus / "vergleich_latte.png",
                      "Laeufe im Vergleich, je Punkt ein Seed. Ein Ergebnis "
                      "erst bei Streuung unter ~1 C", griffe)


def bild_vergleich_profil(laeufe: list, aus: Path) -> Path | None:
    med = {}
    for lauf in laeufe:
        je_op = {}
        for s in lauf["seeds"]:
            for op, p in (s["metrics"].get("fehlerprofil") or {}).items():
                je_op.setdefault(op, []).append(
                    np.asarray(p["bias_segmente"], dtype=float))
        for op, reihen in je_op.items():
            n = min(len(r) for r in reihen)
            med[(lauf["name"], op)] = np.median([r[:n] for r in reihen],
                                                axis=0)
    if not med:
        return None
    namen = [l["name"] for l in laeufe if any(k[0] == l["name"] for k in med)]
    ops = sorted({k[1] for k in med})
    fig, achsen = plt.subplots(1, len(ops), figsize=(4.6 * len(ops), 3.8),
                               squeeze=False, sharey=True)
    for a, op in zip(achsen[0], ops):
        for i, name in enumerate(namen):
            b = med.get((name, op))
            if b is None:
                continue
            a.plot((np.arange(len(b)) + 0.5) / len(b) * 100, b,
                   color=_farbe(i), marker="o", markersize=4)
        a.axhline(0, color=TINTE_2, lw=0.8, ls="--")
        a.set_xlim(0, 100)
        a.set_title(op)
    achsen[0][0].set_ylabel("Median-Bias C (+ = zu warm)")
    griffe = [plt.Line2D([], [], color=_farbe(i), marker="o", label=n)
              for i, n in enumerate(namen)]
    return _speichern(fig, aus / "vergleich_profil.png",
                      "Bias je Sechstel der Trajektorie, Median ueber Seeds "
                      "(metrics.json, letztes Drittel der Epochen)", griffe,
                      "Trajektorie (%)")


def bild_vergleich_zeit(laeufe: list, aus: Path) -> Path | None:
    """MAE und Bias ueber die Zeit, Median ueber Seeds, ein Strich je Lauf."""
    reihen = {}
    for lauf in laeufe:
        js = je_seed(lauf["messungen"].get("halte") or {})
        for op in sorted({op for p in js.values() for op in p}):
            ks = [p[op]["kurve"] for p in js.values()
                  if op in p and "kurve" in p[op]]
            if not ks:
                continue
            n = min(len(k["t_s"]) for k in ks)
            reihen[(lauf["name"], op)] = {
                "t": np.asarray(ks[0]["t_s"][:n]),
                "mae": np.median([k["mae"][:n] for k in ks], axis=0),
                "bias": np.median([k["bias"][:n] for k in ks], axis=0),
                "n": len(ks)}
    if not reihen:
        return None
    namen = [l["name"] for l in laeufe
             if any(k[0] == l["name"] for k in reihen)]
    ops = sorted({k[1] for k in reihen})
    fig, achsen = plt.subplots(2, len(ops), figsize=(5.4 * len(ops), 6.2),
                               squeeze=False, sharex="col")
    for c, op in enumerate(ops):
        for i, name in enumerate(namen):
            r = reihen.get((name, op))
            if r is None:
                continue
            achsen[0][c].plot(r["t"], r["mae"], color=_farbe(i), lw=1.6)
            achsen[1][c].plot(r["t"], r["bias"], color=_farbe(i), lw=1.6)
        lat = next((latte(_latten(l), op) for l in laeufe
                    if np.isfinite(latte(_latten(l), op))), float("nan"))
        if np.isfinite(lat):
            achsen[0][c].axhline(lat, color=TINTE_2, lw=1, ls=":")
        achsen[1][c].axhline(0, color=TINTE_2, lw=0.8, ls="--")
        achsen[0][c].set_title(f"{op}: MAE ueber die Zeit")
        achsen[1][c].set_title(f"{op}: Bias ueber die Zeit (+ = zu warm)")
    achsen[0][0].set_ylabel("MAE (C)")
    achsen[1][0].set_ylabel("Bias (C)")
    griffe = [plt.Line2D([], [], color=_farbe(i), lw=1.6, label=n)
              for i, n in enumerate(namen)]
    griffe.append(plt.Line2D([], [], color=TINTE_2, ls=":",
                             label="Latte (MAE ueber alles)"))
    return _speichern(fig, aus / "vergleich_zeit.png",
                      "Halte-OPs ueber die Zeit, Median ueber die Seeds "
                      "(model.pt)", griffe, "Zeit (s)")


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("laeufe", type=Path, nargs="+",
                   help="Laufverzeichnisse, z. B. GridCNN/artifacts/A")
    p.add_argument("--aus", type=Path, default=Path("GridCNN/laeufe/bilder"),
                   help="Zielverzeichnis fuer die PNGs")
    args = p.parse_args(argv)
    args.aus.mkdir(parents=True, exist_ok=True)

    laeufe = []
    for verz in args.laeufe:
        if not verz.is_dir():
            print(f"!! {verz} ist kein Verzeichnis -- uebersprungen",
                  file=sys.stderr)
            continue
        lauf = lauf_lesen(verz)
        print(f"[{lauf['name']}] {len(lauf['seeds'])} Seed(s), Messungen: "
              f"{', '.join(lauf['messungen']) or 'keine'}")
        bild_lernkurven(lauf, args.aus)
        bild_bias_epochen(lauf, args.aus)
        for tag, messung in lauf["messungen"].items():
            bilder_messung(lauf, tag, messung, args.aus)
        laeufe.append(lauf)
    if not laeufe:
        return 2
    print("[vergleich]")
    bild_vergleich_latte(laeufe, args.aus)
    bild_vergleich_profil(laeufe, args.aus)
    bild_vergleich_zeit(laeufe, args.aus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
