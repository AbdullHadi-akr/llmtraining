#!/usr/bin/env python3
"""Der fortlaufende Benchmark -- Werkzeug zum lebenden Dokument.

Warum das kein Skript mit einer Endzahl ist
-------------------------------------------
Ein Benchmark, der am Ende eine Zahl ausspuckt, ist erst nach Wochen Bauzeit
ueberhaupt lauffaehig -- und bis dahin weiss niemand, ob die Werkzeugkette
traegt. Deshalb ist dieser hier eine **Leiter von Stufen**, und die erste
Stufe prueft ausdruecklich noch gar keine Physik, sondern nur: *tut der
Workflow?*

Jeder Lauf schreibt einen datierten Eintrag in [`BENCHMARK.md`](BENCHMARK.md)
-- oben, damit das Neueste zuerst steht -- und eine Zeile nach
``benchmark_runs.jsonl`` fuer Verlaeufe. Das Dokument ist gemeinsam gepflegt:
das Werkzeug traegt Messungen ein, der Abschnitt "Offene Routen" wird von Hand
fortgeschrieben. Was gemessen wurde, gehoert der Maschine; was daraus folgt,
gehoert uns beiden.

Die Stufen
----------
====  ==================  ===========  ================================
  0   ``workflow``        ohne Daten   laeuft die Kette ueberhaupt?
  1   ``grid``            Cache        ist das Gitter 3 x 11 x 11?
  2   ``stencil``         ohne Daten   stimmt die Ordnung der Ableitungen?
  3   ``solver``          Cache        ist der Loeser stabil, schlaegt er
                                       die trivialen Vorhersager?
====  ==================  ===========  ================================

Stufen ohne Daten laufen ueberall und in Sekunden. Das ist Absicht: sie sind
der Rauchtest, den man vor jedem Commit laufen laesst.

Aufruf
------
    python3 GridCNN/benchmark.py                  # alle lauffaehigen Stufen
    python3 GridCNN/benchmark.py --stage 0        # nur der Workflow-Test
    python3 GridCNN/benchmark.py --stage 0 2      # mehrere
    python3 GridCNN/benchmark.py --dry-run        # nichts ins Protokoll
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DOC = HERE / "BENCHMARK.md"
JSONL = HERE / "benchmark_runs.jsonl"
MARKER = "<!-- LAUF-PROTOKOLL -->"

# Die x-Ebenen und Abstaende des echten Gitters, am 02.09. aus den Koordinaten
# ausgezaehlt. Die Stufen ohne Daten bauen damit ein Gitter derselben Geometrie
# nach -- ein Rauchtest auf einem Wuerfel wuerde genau die Eigenschaft nicht
# treffen, die hier gefaehrlich ist: dass x NICHT aequidistant ist.
X_PLANES = np.array([0.0, 0.010786, 0.021900])
DY, DZ = 0.0198089, 0.0104441
NY = NZ = 11


# ---------------------------------------------------------------------------
# Ergebnisstruktur
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str          # ok | warn | fail | skip
    detail: str = ""
    value: float | None = None

    SYM = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "--  "}

    def line(self) -> str:
        v = "" if self.value is None else f"  [{self.value:.6g}]"
        return f"  {self.SYM[self.status]}  {self.name}{v}" + (
            f"\n        {self.detail}" if self.detail else "")


@dataclass
class StageResult:
    stage: int
    name: str
    checks: list[Check] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)

    @property
    def worst(self) -> str:
        for s in ("fail", "warn", "skip", "ok"):
            if any(c.status == s for c in self.checks):
                return s
        return "skip"

    def add(self, name, status, detail="", value=None) -> None:
        self.checks.append(Check(name, status, detail, value))


# ---------------------------------------------------------------------------
# Stufe 0 -- tut der Workflow?
# ---------------------------------------------------------------------------
def stage_workflow() -> StageResult:
    """Keine Physik, keine Daten. Nur: haelt die Kette zusammen?

    Geprueft werden die Zusagen, auf denen alles Weitere steht -- und zwar die
    *strukturellen*, die exakt gelten muessen und nicht nur ungefaehr:

    * der Reshape laesst sich aus Koordinaten ableiten und ist umkehrbar,
    * ``dT/dx`` an der Symmetrieebene ist **exakt** null,
    * die zentrale Differenz am y/z-Rand ist **exakt** null,
    * ein Euler-Schritt laeuft durch und bleibt endlich.

    Schlaegt eine dieser vier fehl, ist jede Zahl aus den spaeteren Stufen
    wertlos -- deshalb stehen sie hier ganz vorn und nicht am Ende.
    """
    r = StageResult(0, "workflow")

    try:
        import torch
        import grid as gridmod
        import physics as phys
        import solve
    except Exception as exc:                      # pragma: no cover
        r.add("Importe", "fail", f"{type(exc).__name__}: {exc}")
        r.routes.append("Erst die Umgebung reparieren -- ohne Importe geht nichts.")
        return r
    r.add("Importe (torch, grid, physics, solve)", "ok",
          f"torch {torch.__version__}, numpy {np.__version__}")

    # -- Gitter ableiten ----------------------------------------------------
    xx, yy, zz = np.meshgrid(
        X_PLANES, np.arange(NY) * DY, np.arange(NZ) * DZ, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    try:
        layout = gridmod.derive_layout(xyz)
    except Exception as exc:
        r.add("Reshape aus Koordinaten", "fail", f"{type(exc).__name__}: {exc}")
        return r
    ok_shape = layout.shape == (3, NY, NZ)
    r.add("Reshape aus Koordinaten abgeleitet", "ok" if ok_shape else "fail",
          layout.describe())

    # -- Hin und zurueck ----------------------------------------------------
    flat = torch.arange(layout.n_points, dtype=torch.float64)
    back = gridmod.to_flat(gridmod.to_field(flat, layout), layout)
    exact = bool(torch.equal(flat, back))
    r.add("to_field / to_flat ist exakt umkehrbar", "ok" if exact else "fail",
          "" if exact else "Die Punkte werden verwuerfelt -- alles Weitere ist Schrott.")

    # -- Die Symmetrie, exakt ----------------------------------------------
    torch.manual_seed(0)
    fld = torch.randn(layout.shape, dtype=torch.float64)
    padded = gridmod.pad_all(fld, fld[-2])
    dx_all = phys.d_dx(padded, layout)
    sym_err = float(dx_all[0].abs().max().item())
    r.add("dT/dx an der Zellmitte ist exakt null", "ok" if sym_err == 0.0 else "fail",
          "Der Zaehler ist T1 - T1; alles ausser 0.0 heisst, das Padding ist falsch.",
          value=sym_err)

    # -- Der y/z-Rand, exakt ------------------------------------------------
    py = gridmod.pad_yz(fld)
    dy_edge = float(((py[:, 2, :] - py[:, 0, :]) / (2 * layout.dy)).abs().max().item())
    r.add("zentrale Differenz am y-Rand ist exakt null",
          "ok" if dy_edge == 0.0 else "fail",
          "reflect spiegelt um den Randknoten -- sonst waere es replicate.",
          value=dy_edge)

    # -- Ein Schritt --------------------------------------------------------
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    for i in range(3):
        fo[..., i, i] = 1e-5
    qsrc = torch.zeros(layout.shape, dtype=torch.float64)
    dt = 0.2 * phys.cfl_limit(layout, fo)
    res = solve.rollout(layout=layout, tn0=fld, n_steps=5, dt_n=dt,
                        fo_field=fo, qsrc_n=qsrc)
    r.add("fuenf Euler-Schritte laufen durch", "ok" if res.stable else "fail",
          res.summary(), value=res.max_abs)

    # -- Diffusion glaettet -------------------------------------------------
    rough0 = float((fld - fld.mean()).pow(2).mean().item())
    end = torch.as_tensor(res.tn[-1], dtype=torch.float64)
    rough1 = float((end - end.mean()).pow(2).mean().item())
    smooths = rough1 < rough0
    r.add("Diffusion glaettet (Varianz faellt)", "ok" if smooths else "fail",
          f"Varianz {rough0:.4g} -> {rough1:.4g}. Steigt sie, hat der Stern ein "
          f"falsches Vorzeichen.", value=rough1 / rough0)

    # -- Bilanztreue: eine bekannte Naeherung, die beziffert gehoert ---------
    # Der Stern ist NICHT konservativ -- Folge der nicht-konservativen Form
    # ``Fo : grad^2 T`` und des knotenzentrierten reflect-Paddings ohne
    # Halbzellgewichte. Beides ist Absicht. Die Drift des Mittels muss deshalb
    # klein sein und vor allem SAETTIGEN: sie laeuft, solange das Feld
    # ungleichmaessig ist, und hoert auf, sobald es flach ist. Waechst sie
    # weiter, leckt ein Rand echte Energie -- und das waere ein Fehler statt
    # einer Naeherung.
    m0, s0 = float(fld.mean()), float(fld.std())
    d = {}
    for n in (400, 1600):
        rr = solve.rollout(layout=layout, tn0=fld, n_steps=n, dt_n=dt,
                           fo_field=fo, qsrc_n=qsrc)
        d[n] = abs(float(rr.tn[-1].mean()) - m0)
    rel, grows = d[1600] / s0, d[1600] > 1.1 * d[400]
    r.add("Drift des Mittels saettigt (Stern ist nicht bilanztreu)",
          "fail" if grows else ("warn" if rel > 0.05 else "ok"),
          f"Drift {d[400]:.4g} -> {d[1600]:.4g} bei 400 -> 1600 Schritten, "
          f"{rel * 100:.1f} % von std. Bekannte Naeherung, siehe Route R6.",
          value=rel)

    if r.worst == "ok":
        r.routes.append(
            "Workflow traegt. Naechste sinnvolle Stufe ist 2 (Stencil-Ordnung) "
            "-- sie braucht ebenfalls keine Daten.")
        r.routes.append(
            "Fuer Stufe 1 und 3 wird ein data_cache gebraucht. Der liegt nicht "
            "im Repo; auf der Rechenmaschine liegt er unter data_cache/.")
    return r


# ---------------------------------------------------------------------------
# Stufe 2 -- stimmt die Ordnung der Ableitungen?
# ---------------------------------------------------------------------------
def stage_stencil() -> StageResult:
    """Die Differenzensterne gegen analytische Felder.

    Ein Stern, der die falsche Ordnung hat, ist nicht "etwas ungenauer" -- er
    ist ein anderer Operator. Deshalb wird hier nicht auf eine Toleranz
    geprueft, sondern auf **Konvergenzordnung**: das Gitter wird verfeinert und
    nachgesehen, ob der Fehler wie erwartet faellt.

    Geprueft wird auf einem *nicht*-aequidistanten x-Gitter, weil genau dort
    die naive Formel einen Erste-Ableitungs-Anteil einmischt.
    """
    import torch
    import grid as gridmod
    import physics as phys

    r = StageResult(2, "stencil")

    def build(n: int):
        """Gitter mit n Knoten je Achse, x absichtlich gestreckt."""
        xs = np.linspace(0.0, 1.0, n) ** 1.3          # nicht aequidistant
        ys = np.linspace(0.0, 1.0, n)
        zs = np.linspace(0.0, 1.0, n)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        return gridmod.derive_layout(xyz), xx, yy, zz

    # Ein glattes Feld mit bekannter zweiter Ableitung. Absichtlich ohne
    # Symmetrie in x, damit die Geisterschichten das Ergebnis nicht schoenen:
    # gemessen wird nur im INNEREN, wo kein Geist hineinreicht.
    def exact(xx, yy, zz):
        f = np.sin(2.0 * xx) * np.cos(1.5 * yy) * np.exp(0.3 * zz)
        fxx = -4.0 * f
        fyy = -2.25 * f
        fzz = 0.09 * f
        return f, fxx, fyy, fzz

    errs = {"xx": [], "yy": [], "zz": []}
    sizes = (17, 33)
    for n in sizes:
        layout, xx, yy, zz = build(n)
        f, fxx, fyy, fzz = exact(xx, yy, zz)
        t = torch.as_tensor(f, dtype=torch.float64)
        padded = gridmod.pad_all(t, t[-2])
        num = {"xx": phys.d2_dx2(padded, layout),
               "yy": phys.d2_dy2(padded, layout),
               "zz": phys.d2_dz2(padded, layout)}
        ref = {"xx": fxx, "yy": fyy, "zz": fzz}
        for k in errs:
            a = num[k][1:-1, 1:-1, 1:-1].numpy()
            b = ref[k][1:-1, 1:-1, 1:-1]
            errs[k].append(float(np.abs(a - b).max()))

    for k, name in (("xx", "d2/dx2 (nicht aequidistant)"),
                    ("yy", "d2/dy2"), ("zz", "d2/dz2")):
        e0, e1 = errs[k]
        order = float(np.log2(e0 / e1)) if e1 > 0 else float("inf")
        # Auf einem gestreckten Gitter ist der nicht-aequidistante
        # Dreipunktstern formal erster Ordnung und in der Praxis zwischen 1 und
        # 2. Unter 0.9 stimmt der Stern nicht.
        status = "ok" if order > 0.9 else "fail"
        r.add(f"Konvergenzordnung {name}", status,
              f"Fehler {e0:.3g} -> {e1:.3g} bei {sizes[0]} -> {sizes[1]} Knoten",
              value=order)

    # Der Kreuzterm gegen sein analytisches Gegenstueck.
    layout, xx, yy, zz = build(33)
    f = np.sin(2.0 * xx) * np.cos(1.5 * yy)
    fxy = -2.0 * 1.5 * np.cos(2.0 * xx) * np.sin(1.5 * yy)
    t = torch.as_tensor(f, dtype=torch.float64)
    padded = gridmod.pad_all(t, t[-2])
    num = phys.d2_dxdy(padded, layout)[1:-1, 1:-1, 1:-1].numpy()
    err = float(np.abs(num - fxy[1:-1, 1:-1, 1:-1]).max())
    scale = float(np.abs(fxy).max())
    rel = err / scale
    r.add("Kreuzterm d2/dxdy", "ok" if rel < 0.05 else "fail",
          "lambda_xy ist auf JR1 ungleich null -- dieser Term darf nicht fehlen.",
          value=rel)

    if r.worst == "ok":
        r.routes.append(
            "Die Sterne stimmen. Damit ist physics.py so weit belastbar, wie es "
            "ohne echte Materialdaten geht.")
        r.routes.append(
            "Offen bleibt die konservative Form div(lambda grad T): sie wird "
            "bewusst NICHT gerechnet, damit der Vergleich mit PINNmodulusTwo "
            "eine Architekturaussage bleibt. Wenn geaendert, dann in beiden "
            "Projekten gleichzeitig und als eigene Achse.")
    return r


# ---------------------------------------------------------------------------
# Stufen mit Daten
# ---------------------------------------------------------------------------
CACHE_CANDIDATES = ("data_cache", "../data_cache", "PINNmodulusTwo/data_cache")


def find_cache(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    root = HERE.parent
    for c in CACHE_CANDIDATES:
        p = (root / c).resolve()
        if p.is_dir():
            return p
    return None


def stage_grid(cache: Path | None) -> StageResult:
    """Das echte Gitter aus dem Cache -- 3 x 11 x 11, ueber alle OPs gleich."""
    r = StageResult(1, "grid")
    if cache is None:
        r.add("data_cache gefunden", "skip",
              "Kein Cache. Auf der Rechenmaschine liegt er unter data_cache/; "
              "mit --cache einen Pfad angeben.")
        r.routes.append(
            "Stufe 1 und 3 auf der Maschine mit den Daten laufen lassen. Sie "
            "sind schnell und brauchen kein GPU.")
        return r
    r.add("data_cache gefunden", "ok", str(cache))
    r.add("Gitterprobe", "skip",
          "Noch nicht implementiert -- wird gebaut, sobald ein Cache zum "
          "Gegenlesen da ist. tools/spatial_rank.py macht die Probe heute "
          "bereits nebenbei.")
    return r


def stage_solver(cache: Path | None) -> StageResult:
    """Stabilitaet und Physik-Latte -- das eigentliche Tor von Stufe 3."""
    r = StageResult(3, "solver")
    if cache is None:
        r.add("data_cache gefunden", "skip", "siehe Stufe 1")
        return r
    r.add("Wandterm kalibrierbar", "skip",
          "Braucht q_solid_to_fluid, mdot, cp_fluid und fluid_out_temp im "
          "Buendel -- das ist Stufe 2 des Fahrplans und noch offen. Ohne sie "
          "laeuft der Loeser adiabat, und das ist eine Ablation, keine Latte.")
    r.routes.append(
        "Route A: Stufe 2 zuerst (Cache erweitern), dann die echte Physik-Latte.")
    r.routes.append(
        "Route B: den Loeser vorher adiabat ueber alle OPs rollen. Das misst "
        "keine Latte, findet aber Stabilitaets- und Vorzeichenfehler, solange "
        "sie billig zu finden sind.")
    return r


# ---------------------------------------------------------------------------
# Protokoll
# ---------------------------------------------------------------------------
def render_entry(results: list[StageResult], stamp: str) -> str:
    worst = "ok"
    for s in ("fail", "warn", "skip"):
        if any(r.worst == s for r in results):
            worst = s
            break
    head = {"ok": "gruen", "warn": "gelb", "fail": "ROT", "skip": "teilweise"}[worst]

    lines = [f"### {stamp} — {head}", ""]
    for r in results:
        counts = {}
        for c in r.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        tally = ", ".join(f"{v}x {k}" for k, v in sorted(counts.items()))
        lines.append(f"**Stufe {r.stage} · `{r.name}`** — {tally}")
        lines.append("")
        for c in r.checks:
            v = "" if c.value is None else f" `{c.value:.6g}`"
            lines.append(f"- `{Check.SYM[c.status].strip() or '--'}` {c.name}{v}")
            if c.detail:
                first = c.detail.strip().splitlines()[0]
                lines.append(f"  - {first}")
        lines.append("")
    routes = [x for r in results for x in r.routes]
    if routes:
        lines.append("**Vorgeschlagene Routen**")
        lines.append("")
        lines += [f"{i}. {t}" for i, t in enumerate(routes, 1)]
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def append_to_doc(entry: str) -> None:
    if not DOC.exists():
        raise SystemExit(
            f"{DOC} fehlt. Das lebende Dokument gehoert ins Repo -- "
            f"ohne es hat ein Lauf keinen Ort.")
    text = DOC.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"Marker {MARKER} fehlt in {DOC}.")
    head, tail = text.split(MARKER, 1)
    DOC.write_text(head + MARKER + "\n\n" + entry + tail.lstrip("\n"),
                   encoding="utf-8")


def append_jsonl(results: list[StageResult], stamp: str) -> None:
    row = {
        "stamp": stamp,
        "python": platform.python_version(),
        "stages": {
            str(r.stage): {
                "name": r.name,
                "worst": r.worst,
                "checks": [
                    {"name": c.name, "status": c.status, "value": c.value}
                    for c in r.checks
                ],
            } for r in results
        },
    }
    with JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fortlaufender Benchmark fuer GridCNN.")
    ap.add_argument("--stage", type=int, nargs="*", default=None,
                    help="Nur diese Stufen (Default: alle lauffaehigen).")
    ap.add_argument("--cache", default=None, help="Pfad zum data_cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Nichts ins Protokoll schreiben.")
    args = ap.parse_args()

    cache = find_cache(args.cache)
    all_stages = {
        0: lambda: stage_workflow(),
        1: lambda: stage_grid(cache),
        2: lambda: stage_stencil(),
        3: lambda: stage_solver(cache),
    }
    wanted = args.stage if args.stage else sorted(all_stages)

    results: list[StageResult] = []
    print("=" * 72)
    print("GridCNN — fortlaufender Benchmark")
    print("=" * 72)
    for s in wanted:
        if s not in all_stages:
            raise SystemExit(f"Unbekannte Stufe: {s}. Bekannt: {sorted(all_stages)}")
        res = all_stages[s]()
        results.append(res)
        print(f"\n-- Stufe {res.stage}: {res.name} " + "-" * (50 - len(res.name)))
        for c in res.checks:
            print(c.line())

    routes = [x for r in results for x in r.routes]
    if routes:
        print("\n-- Vorgeschlagene Routen " + "-" * 46)
        for i, t in enumerate(routes, 1):
            print(f"  {i}. {t}")

    failed = any(r.worst == "fail" for r in results)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if args.dry_run:
        print(f"\n[dry-run] {DOC.name} nicht angefasst.")
    else:
        append_to_doc(render_entry(results, stamp))
        append_jsonl(results, stamp)
        print(f"\nProtokoll ergaenzt: {DOC.relative_to(HERE.parent)}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
