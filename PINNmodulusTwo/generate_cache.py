#!/usr/bin/env python3
"""Build the .npz OP bundles PINNmodulusTwo trains on.

Usage:
    python3 PINNmodulusTwo/generate_cache.py --all          # ALLE SIEBZEHN
    python3 PINNmodulusTwo/generate_cache.py --all --check   # nur pruefen
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
"""

import argparse
import dataclasses
import sys
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


def generate_cache_for_ops(op_ids):
    """Generate .npz cache files for the given OP IDs."""
    # Preferred location: shared and top-level, matching data._CACHE_CANDIDATES.
    output_dir = _PROJECT_ROOT / "data_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    gebaut, gescheitert = [], []
    for op_id in op_ids:
        print(f"\n=== Processing {op_id} ===")
        try:
            # Assemble the OP from raw CSVs
            bundle = assemble_op(op_id)

            # Compute cache key (bundle is frozen; use dataclasses.replace)
            cache_key = compute_cache_key(op_id)
            bundle = dataclasses.replace(bundle, cache_key=cache_key)

            # Save to .npz
            path = save_bundle(bundle, target_dir=output_dir)
            print(f"✓ Created: {path}")
            print(f"  T shape: {bundle.T.shape}")
            print(f"  xyz shape: {bundle.xyz.shape}")
            print(f"  t_fast points: {len(bundle.t_fast)}")
            print(f"  schema_version: {bundle.schema_version}")
            # Schema v3: die zwei neuen Reihen mit IHRER Achse, und ob die
            # Achse zufaellig t_slow ist. Nicht angleichen -- nur zeigen.
            print(f"  fluid_props: {dict(zip(bundle.fluid_props_names, bundle.fluid_props[0]))}")
            achsen = bundle.meta.get("wall_ts_axes", {})
            for name, (times, values) in sorted(bundle.wall_ts.items()):
                info = achsen.get(name, {})
                gleich = "= t_slow" if info.get("gleich_t_slow") else "EIGENE Achse"
                print(f"  wall_ts[{name}]: {values.shape[0]} Punkte, {gleich}"
                      f"  (t_slow: {len(bundle.t_slow)})")
            gebaut.append(op_id)

        except Exception as e:
            print(f"✗ Error processing {op_id}: {e}")
            import traceback
            traceback.print_exc()
            gescheitert.append(op_id)

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

    gebaut, gescheitert = generate_cache_for_ops(ops_to_process)
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
