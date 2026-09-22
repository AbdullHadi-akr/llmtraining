#!/usr/bin/env python3
"""Build the .npz OP bundles PINNmodulusTwo trains on.

Usage:
    python3 PINNmodulusTwo/generate_cache.py --all           # ALLE SIEBZEHN
    python3 PINNmodulusTwo/generate_cache.py --all --check   # nur pruefen
    python3 PINNmodulusTwo/generate_cache.py --all -j 4      # vier gleichzeitig
    python3 PINNmodulusTwo/generate_cache.py OP08 OP09       # einzelne OPs

Writes to the top-level ``data_cache/``. The raw-CSV assembly still comes from
the legacy workflow -- this is the only place the active code depends on it.

⚠ ES SIND SIEBZEHN OPs, NICHT SECHZEHN.
    Im Cache liegen OP01-OP16 **plus OP19** (der Report-only-OP aus O11). Wer
    sechzehn neu baut, laesst OP19 auf dem alten ``schema_version`` zurueck --
    und merkt es erst, wenn Stufe 6 ihn mitrollen will. Deshalb gibt es
    ``--all``: die Liste kommt aus ``op_registry`` und nicht aus dem
    Gedaechtnis. Gemessen am 22.09.

⚠ VORHER PRUEFEN, NICHT NACH ZWANZIG MINUTEN MERKEN.
    ``--check`` liest keine Daten, sondern sieht nur nach, ob jeder OP alle
    Dateien hat, die ``assemble_op`` braucht -- inklusive der zwei, die mit
    Schema v3 dazugekommen sind (``*_Heat Transfer.csv``,
    ``*_Temperaturen.csv``). Das dauert Sekunden.

``-j``: die siebzehn OPs sind unabhaengig
-------------------------------------------
Jeder OP liest seine eigenen CSVs und schreibt seine eigene ``.npz``. Es gibt
keinen gemeinsamen Zustand, also ist das der billigste Parallelismus im ganzen
Projekt -- und anders als beim Training auch der wirksamste: das hier ist
pandas, das CSVs parst, also CPU und Platte, nicht die GPU. ``-j`` ist damit
**nicht** dasselbe wie das ``-j`` von ``sweep.py``: hier ist weder CUDA noch
MPS im Spiel, MPS aendert an diesem Lauf nichts.

**Die Vorgabe bleibt seriell (``-j 1``), mit Absicht.** Der Rebuild vom
22.09. ist der Torlauf von Stufe 2: ``profile_report``, ``coverage_report``
und ``energy_balance_report`` muessen danach exakt dieselben Zahlen liefern
wie vorher. Ein Tor prueft man nicht und aendert gleichzeitig, wie gebaut
wird. Wer ``-j`` setzt, bekommt dieselben Dateien -- die Bundles haengen
nicht voneinander ab -- aber die Entscheidung gehoert getippt und nicht
geerbt.

⚠ ``-j`` kostet Arbeitsspeicher: jeder Prozess haelt ein **volles** Bundle,
  bevor er es schreibt. Auf einer kleinen Box ist ``-j 2`` das Vernuenftige;
  ``-j 0`` waehlt ``min(4, Kerne/2)``, wie ``sweep.py``.
"""

import argparse
import dataclasses
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# The raw-CSV assembly still lives in the legacy workflow; this is the one place
# the active code depends on it. Both layouts are accepted so a checkout that has
# not been restructured keeps working.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_CANDIDATES = (
    _PROJECT_ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "src",
    _PROJECT_ROOT / "battery_surrogate_agenticWorkflow" / "src",
)
for _src in _SRC_CANDIDATES:
    if _src.exists():
        sys.path.insert(0, str(_src))
        break
else:
    raise SystemExit(
        "cannot find the legacy battery_surrogate sources needed to build the cache.\n"
        "  searched:\n"
        + "".join(f"    {c}\n" for c in _SRC_CANDIDATES)
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))

from battery_surrogate.data.assemble import assemble_op
from battery_surrogate.data.cache import save_bundle, compute_cache_key
from battery_surrogate.data.paths import data_raw_dir

import op_registry


# Die siebzehn. OP01-OP16 aus ALL_OPS plus OP19 aus MEASUREMENT_OPS_AVAILABLE --
# aus der Registry gelesen, damit die Liste nicht an zwei Stellen gepflegt wird.
ALLE_OPS: tuple[str, ...] = tuple(op_registry.ALL_OPS) + tuple(
    op_registry.MEASUREMENT_OPS_AVAILABLE
)

# Was assemble_op je OP oeffnet. Die letzten zwei sind mit Schema v3
# dazugekommen und fehlen einem aelteren Datenstand moeglicherweise.
BENOETIGTE_DATEIEN: tuple[tuple[str, str], ...] = (
    ("*_Batemo FMU1.csv", "v1"),
    ("*_Heat Source.csv", "v1"),
    ("*_Fluidstoffwerte.csv", "v1"),
    ("*_T_grid_cc_i.csv", "v1"),
    ("*_T_grid_g_i.csv", "v1"),
    ("*_T_grid_jr1c_i.csv", "v1"),
    ("*_Heat Transfer.csv", "v3"),
    ("*_Temperaturen.csv", "v3"),
)


def pruefe_ops(op_ids) -> dict[str, list[str]]:
    """Welche Datei fehlt welchem OP? Liest nichts, oeffnet nichts."""
    root = data_raw_dir()
    fehlend: dict[str, list[str]] = {}
    for op_id in op_ids:
        op_dir = root / op_id / op_id
        if not op_dir.exists():
            fehlend[op_id] = [f"Ordner fehlt: {op_dir}"]
            continue
        luecken = [f"{pat}  (seit {seit})"
                   for pat, seit in BENOETIGTE_DATEIEN
                   if not sorted(op_dir.glob(pat))]
        if luecken:
            fehlend[op_id] = luecken
    return fehlend


def _bericht_pruefung(op_ids) -> bool:
    """True, wenn alles da ist."""
    fehlend = pruefe_ops(op_ids)
    if not fehlend:
        print(f"✓ Vorpruefung: alle {len(op_ids)} OPs haben alle "
              f"{len(BENOETIGTE_DATEIEN)} benoetigten Dateien.")
        return True
    print(f"✗ Vorpruefung: {len(fehlend)} von {len(op_ids)} OPs sind unvollstaendig.\n")
    for op_id, luecken in sorted(fehlend.items()):
        print(f"  {op_id}")
        for luecke in luecken:
            print(f"      fehlt: {luecke}")
    print("\n  Ein OP ohne '*_Heat Transfer.csv' oder '*_Temperaturen.csv' laesst")
    print("  sich nicht auf Schema v3 bauen -- der Wandterm haengt an genau diesen")
    print("  zwei Dateien. Siehe GridCNN/FAHRPLAN.md, Stufe 2.")
    return False


def _baue_einen(op_id: str, output_dir: Path) -> tuple[str, bool, str]:
    """Einen OP bauen und schreiben. Gibt ``(op_id, ok, Ausgabetext)`` zurueck.

    Gibt den Text **zurueck**, statt ihn zu drucken, damit er unter ``-j`` am
    Stueck herauskommt. Vier Prozesse, die gleichzeitig in dasselbe stdout
    schreiben, verschraenken ihre Zeilen -- und ein halber Traceback zwischen
    zwei fremden Zahlen ist genau die Ausgabe, die man nach zwanzig Minuten
    nicht mehr lesen kann.
    """
    zeilen = [f"\n=== Processing {op_id} ==="]
    try:
        # Assemble the OP from raw CSVs
        bundle = assemble_op(op_id)

        # Compute cache key (bundle is frozen; use dataclasses.replace)
        cache_key = compute_cache_key(op_id)
        bundle = dataclasses.replace(bundle, cache_key=cache_key)

        # Save to .npz
        path = save_bundle(bundle, target_dir=output_dir)
        zeilen += [
            f"\u2713 Created: {path}",
            f"  T shape: {bundle.T.shape}",
            f"  xyz shape: {bundle.xyz.shape}",
            f"  t_fast points: {len(bundle.t_fast)}",
            f"  schema_version: {bundle.schema_version}",
            # Schema v3: die zwei neuen Reihen mit IHRER Achse, und ob die
            # Achse zufaellig t_slow ist. Nicht angleichen -- nur zeigen.
            f"  fluid_props: "
            f"{dict(zip(bundle.fluid_props_names, bundle.fluid_props[0]))}",
        ]
        achsen = bundle.meta.get("wall_ts_axes", {})
        for name, (times, values) in sorted(bundle.wall_ts.items()):
            info = achsen.get(name, {})
            gleich = "= t_slow" if info.get("gleich_t_slow") else "EIGENE Achse"
            zeilen.append(f"  wall_ts[{name}]: {values.shape[0]} Punkte, {gleich}"
                          f"  (t_slow: {len(bundle.t_slow)})")
        return op_id, True, "\n".join(zeilen)

    except Exception as e:                        # noqa: BLE001 - absichtlich
        zeilen.append(f"\u2717 Error processing {op_id}: {e}")
        zeilen.append(traceback.format_exc())
        return op_id, False, "\n".join(zeilen)


def generate_cache_for_ops(op_ids, jobs: int = 1):
    """Generate .npz cache files for the given OP IDs.

    ``jobs=1`` baut seriell und druckt jeden OP, sobald er fertig ist -- bei
    zehn bis dreissig Minuten Laufzeit will man sehen, dass es vorangeht.
    ``jobs>1`` verteilt auf Prozesse; jeder Block wird vom Elternprozess am
    Stueck gedruckt, sobald sein OP fertig ist.
    """
    # Preferred location: shared and top-level, matching data._CACHE_CANDIDATES.
    output_dir = _PROJECT_ROOT / "data_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    op_ids = list(op_ids)
    jobs = max(1, min(int(jobs), len(op_ids)))

    gebaut, gescheitert = [], []

    def verbuchen(op_id: str, ok: bool, text: str) -> None:
        print(text, flush=True)
        (gebaut if ok else gescheitert).append(op_id)

    n_ges = len(op_ids)
    if jobs == 1:
        for n, op_id in enumerate(op_ids, start=1):
            # VOR dem Bau, nicht danach: ein OP kostet Minuten, und ein
            # stummes Terminal ist bei zehn bis dreissig Minuten Laufzeit
            # nicht von einem haengenden Prozess zu unterscheiden.
            print(f"\n[{n}/{n_ges}] {op_id} ...", flush=True)
            verbuchen(*_baue_einen(op_id, output_dir))
    else:
        print(f"-j {jobs}: {n_ges} OPs, {jobs} gleichzeitig. Die Bloecke kommen "
              f"in der Reihenfolge, in der sie fertig werden.", flush=True)
        # Prozesse, nicht Threads: assemble_op ist pandas und numpy, und ein
        # eigener Interpreter je OP ist die einzige Trennung, die auch den
        # Zustand der Legacy-Module auseinanderhaelt.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_baue_einen, op_id, output_dir)
                       for op_id in op_ids]
            for n, fut in enumerate(futures, start=1):
                print(f"\n[{n}/{n_ges}] fertig:", flush=True)
                verbuchen(*fut.result())

    # In OP-Reihenfolge, nicht in Fertigstellungsreihenfolge: die Zusammen-
    # fassung soll zwischen -j 1 und -j 4 vergleichbar bleiben.
    reihe = {op_id: n for n, op_id in enumerate(op_ids)}
    gebaut.sort(key=reihe.get)
    gescheitert.sort(key=reihe.get)
    return gebaut, gescheitert


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Baut die .npz-Bundles. --all sind SIEBZEHN OPs, nicht sechzehn.",
    )
    ap.add_argument("ops", nargs="*", help="OP-Ids, z.B. OP08 OP09")
    ap.add_argument("--all", action="store_true",
                    help=f"alle {len(ALLE_OPS)} OPs: {' '.join(ALLE_OPS)}")
    ap.add_argument("--check", action="store_true",
                    help="nur pruefen, ob alle Dateien da sind -- baut nichts")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="OPs gleichzeitig. Vorgabe 1 (seriell), weil der "
                         "Rebuild ein Torlauf ist. 0 = min(4, Kerne/2). Jeder "
                         "Prozess haelt ein volles Bundle -- das kostet RAM, "
                         "nicht GPU: MPS spielt hier keine Rolle")
    args = ap.parse_args()

    if args.all and args.ops:
        ap.error("--all und eine OP-Liste schliessen sich aus")
    if args.all:
        ops_to_process = list(ALLE_OPS)
    elif args.ops:
        ops_to_process = list(args.ops)
    else:
        ap.error(
            "keine OPs angegeben.\n"
            f"  --all baut alle {len(ALLE_OPS)}: {' '.join(ALLE_OPS)}\n"
            "  Der alte stille Default (OP05 OP06 OP07) ist weg: er hat beim\n"
            "  Rebuild vierzehn OPs auf dem alten schema_version zurueckgelassen."
        )

    print(f"OPs: {' '.join(ops_to_process)}  ({len(ops_to_process)} Stueck)")
    if not _bericht_pruefung(ops_to_process):
        return 1
    if args.check:
        print("\n--check: es wurde nichts gebaut.")
        return 0

    jobs = args.jobs or max(1, min(4, (os.cpu_count() or 2) // 2))
    gebaut, gescheitert = generate_cache_for_ops(ops_to_process, jobs=jobs)
    print(f"\n{'='*60}")
    print(f"gebaut:      {len(gebaut)}/{len(ops_to_process)}  {' '.join(gebaut)}")
    if gescheitert:
        print(f"GESCHEITERT: {len(gescheitert)}  {' '.join(gescheitert)}")
        print("\n⚠ Ein unvollstaendiger Cache ist schlimmer als keiner: die")
        print("  gescheiterten OPs liegen jetzt auf dem ALTEN schema_version")
        print("  neben den neuen. Erst reparieren, dann die Reports vergleichen.")
        return 1
    print(f"✓ Done! Cache files created in {_PROJECT_ROOT / 'data_cache'}")
    print("\nNaechstes: die drei Reports gegen die Zahlen von VOR dem Umbau")
    print("halten -- profile_report, coverage_report, energy_balance_report")
    print("muessen EXAKT dasselbe liefern. Das ist das Tor von Stufe 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
