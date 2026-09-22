"""Schema v3 -- der Wandpfad im Cache, ohne Rohdaten pruefbar.

Warum diese Datei hier liegt und nicht in ``legacy/.../tests``: die CI faehrt
``PINNmodulusTwo/tests`` und ``GridCNN/tests``, den Legacy-Baum nicht. Ein Test
dort waere geschrieben und nie gelaufen.

Was v3 dazugibt, und warum -- GridCNN/FAHRPLAN.md, Stufe 2:

* ``wall_ts["q_solid_to_fluid"]`` (W) und ``wall_ts["fluid_out_temp"]`` (C).
  Ohne die beiden ist ``L_wall`` nicht anschliessbar, der einzige Verlustterm
  mit einem GEMESSENEN Ziel.
* ``fluid_props`` wird benannt gelesen statt positionell, und die Namen fahren
  in ``fluid_props_names`` mit.

Der rote Faden durch alle Tests ist derselbe Fehler in drei Gestalten: **eine
Annahme ueber Zeitachsen, die niemand prueft.** Sie ist am 22.09. in
``GridCNN/tools/balance_check.py`` aufgeflogen (``fp and xp are not of the same
length``), nachdem sie an anderer Stelle still durchgelaufen war -- NumPy hatte
dort eine Laenge-1-Reihe ueber eine Laenge-N gebroadcastet und eine
richtig aussehende Zahl aus dem falschen Grund geliefert.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_SRC = _ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "src"
if _LEGACY_SRC.exists() and str(_LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(_LEGACY_SRC))

pytest.importorskip("pandas", reason="die Rohleser bauen auf pandas")

from battery_surrogate.data import cache as C                    # noqa: E402
from battery_surrogate.data.models import OpBundle               # noqa: E402
from battery_surrogate.data.raw_readers import (                 # noqa: E402
    read_fluidstoffwerte,
    read_heat_transfer,
    read_temperaturen,
)

# Die Spaltennamen sind nicht erfunden: genau diese haben am 22.09. in
# balance_check.py gegen die echten Exporte aller sieben Konstant-Treiber-OPs
# getroffen.
HT_CSV = (
    "Physical Time (s),Heat Transfer: solid to fluid Monitor (W)\n"
    "0.0,1.5\n0.4,2.5\n0.8,3.5\n"
)
TM_CSV = (
    "Physical Time (s),Tmfavg_fluid_out Monitor (C),NTC Monitor (C)\n"
    "0.0,20.0\n0.2,21.0\n0.4,22.0\n0.6,23.0\n0.8,24.0\n"
)
# Reihenfolge wie im echten Export -- NICHT die, die der v2-Vertrag vermutete.
FL_CSV = (
    "Physical Time (s),Conductivity Monitor (W/m-K),Density Monitor (kg/m^3),"
    "Specific Heat Monitor (J/kg-K)\n0.0,0.42,1050.0,3374.0\n"
)


@pytest.fixture
def rohdateien(tmp_path: Path) -> Path:
    (tmp_path / "X_Heat Transfer.csv").write_text(HT_CSV, encoding="utf-8")
    (tmp_path / "X_Temperaturen.csv").write_text(TM_CSV, encoding="utf-8")
    (tmp_path / "X_Fluidstoffwerte.csv").write_text(FL_CSV, encoding="utf-8")
    return tmp_path


def _bundle(**kw) -> OpBundle:
    basis = dict(
        op_id="OP01", schema_version=3, cache_key="k",
        t_fast=np.arange(5, dtype=np.float32),
        t_slow=np.arange(3, dtype=np.float32),
        bc_V=np.zeros(5, np.float32), bc_OCV=np.zeros(5, np.float32),
        bc_I=np.zeros(5, np.float32), pe_P_loss=np.zeros(5, np.float32),
        T=np.zeros((5, 2), np.float32), q_source=np.zeros((3, 3), np.float32),
        xyz=np.zeros((2, 3), np.float32), layer=np.array(["cc", "g"]),
        sensor_id=np.array(["0", "1"]),
        fluid_props=np.zeros((1, 3), np.float32),
        sim_config_scalar=np.zeros(1, np.float32),
        sim_config_scalar_names=("c_rate",),
        sim_config_ts={}, meta={"op_id": "OP01"},
    )
    basis.update(kw)
    return OpBundle(**basis)


# --- Die Leser ---------------------------------------------------------


def test_die_zwei_neuen_leser_treffen_die_echten_spaltennamen(rohdateien):
    ht = read_heat_transfer(rohdateien / "X_Heat Transfer.csv", encoding="utf-8")
    tm = read_temperaturen(rohdateien / "X_Temperaturen.csv", encoding="utf-8")
    assert list(ht) == ["q_solid_to_fluid"]
    assert list(tm) == ["fluid_out_temp"]
    assert ht["q_solid_to_fluid"][1].tolist() == [1.5, 2.5, 3.5]


def test_jede_reihe_behaelt_ihre_eigene_zeitachse(rohdateien):
    """Der Kern von v3. Die zwei Dateien sind verschieden lang, und das bleibt so.

    Genau hier ist balance_check.py am 22.09. gefallen. Wer die Achsen
    gleichsetzt, bekommt entweder einen Absturz oder -- schlimmer -- eine
    stille Fehlzuordnung.
    """
    ht = read_heat_transfer(rohdateien / "X_Heat Transfer.csv", encoding="utf-8")
    tm = read_temperaturen(rohdateien / "X_Temperaturen.csv", encoding="utf-8")
    t_ht, _ = ht["q_solid_to_fluid"]
    t_tm, _ = tm["fluid_out_temp"]
    assert t_ht.shape == (3,) and t_tm.shape == (5,)
    assert t_ht.shape != t_tm.shape
    # und die Achse gehoert zur Reihe, nicht neben sie
    for reihe in (ht["q_solid_to_fluid"], tm["fluid_out_temp"]):
        assert reihe[0].shape == reihe[1].shape


def test_eine_fehlende_spalte_nennt_gesuchtes_und_vorhandenes(tmp_path):
    p = tmp_path / "Z_Heat Transfer.csv"
    p.write_text("Physical Time (s),Irgendwas\n0,1\n", encoding="utf-8")
    with pytest.raises(KeyError) as ex:
        read_heat_transfer(p, encoding="utf-8")
    text = str(ex.value)
    assert "q_solid_to_fluid" in text and "Irgendwas" in text


def test_eine_fehlende_zeitspalte_ist_ein_fehler_kein_index(tmp_path):
    p = tmp_path / "Z_Temperaturen.csv"
    p.write_text("Tmfavg_fluid_out Monitor (C)\n20.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="no time column"):
        read_temperaturen(p, encoding="utf-8")


# --- fluid_props: benannt statt geraten --------------------------------


def test_fluid_props_wird_benannt_gelesen(rohdateien):
    props, names = read_fluidstoffwerte(
        rohdateien / "X_Fluidstoffwerte.csv", encoding="utf-8")
    assert names == ("conductivity", "density", "cp_fluid")
    werte = dict(zip(names, props[0]))
    assert werte["cp_fluid"] == pytest.approx(3374.0)
    assert werte["density"] == pytest.approx(1050.0)
    assert werte["conductivity"] == pytest.approx(0.42)


def test_die_alte_positionelle_lesart_war_nachweislich_falsch(rohdateien):
    """Der v2-Vertrag sagte "typically density, specific heat, thermal
    conductivity". Der Export steht anders herum, und die erste Spalte ist
    die Leitfaehigkeit -- wer sie als Dichte liest, rechnet mit 0.42 kg/m^3.
    """
    props, names = read_fluidstoffwerte(
        rohdateien / "X_Fluidstoffwerte.csv", encoding="utf-8")
    positionell_density = float(props[0][0])
    assert positionell_density == pytest.approx(0.42)
    assert names[0] == "conductivity"


def test_ohne_treffer_faellt_es_positionell_zurueck_und_sagt_es(tmp_path):
    p = tmp_path / "Y_Fluidstoffwerte.csv"
    p.write_text("a,b,c\n1.0,2.0,3.0\n", encoding="utf-8")
    row, names = read_fluidstoffwerte(p, encoding="utf-8")
    assert row.shape == (1, 3)
    assert all(n.startswith("unbenannt") for n in names)


# --- Der Rundlauf durch npz --------------------------------------------


def test_wall_ts_kommt_bitgleich_durch_das_npz(rohdateien, tmp_path):
    ht = read_heat_transfer(rohdateien / "X_Heat Transfer.csv", encoding="utf-8")
    tm = read_temperaturen(rohdateien / "X_Temperaturen.csv", encoding="utf-8")
    props, names = read_fluidstoffwerte(
        rohdateien / "X_Fluidstoffwerte.csv", encoding="utf-8")

    b = _bundle(wall_ts={**ht, **tm}, fluid_props=props, fluid_props_names=names)
    zurueck = C.load_bundle(C.save_bundle(b, target_dir=tmp_path))

    assert sorted(zurueck.wall_ts) == ["fluid_out_temp", "q_solid_to_fluid"]
    assert zurueck.fluid_props_names == names
    for schluessel, (t, v) in b.wall_ts.items():
        t2, v2 = zurueck.wall_ts[schluessel]
        assert np.array_equal(t, t2), schluessel
        assert np.array_equal(v, v2), schluessel


def test_ein_v2_bundle_laedt_weiter_und_sagt_leer_statt_null(rohdateien, tmp_path):
    """Rueckwaertskompatibilitaet in die richtige Richtung.

    Ein Bundle von vor dem Umbau hat die neuen Schluessel nicht. Es muss laden
    -- sonst ist zwischen Schema-Bump und Rebuild gar nichts mehr benutzbar --
    und ``wall_ts == {}`` heisst dann genau "noch nicht neu gebaut". Eine
    stille Null waere die schlechtere Antwort: sie sieht nach Messung aus.
    """
    ht = read_heat_transfer(rohdateien / "X_Heat Transfer.csv", encoding="utf-8")
    p3 = C.save_bundle(_bundle(wall_ts=dict(ht)), target_dir=tmp_path)

    with np.load(p3, allow_pickle=False) as z:
        v2 = {k: z[k] for k in z.files
              if not k.startswith("wall_ts") and k != "fluid_props_names_json"}
    p2 = tmp_path / "OP02.npz"
    np.savez_compressed(p2, **v2)

    alt = C.load_bundle(p2)
    assert alt.wall_ts == {}
    assert alt.fluid_props_names == ()


def test_opbundle_ohne_die_neuen_felder_baut_weiter():
    """Die zwei Felder haben Defaults. Ohne sie braeche jeder bestehende
    Aufruf von ``OpBundle(...)`` -- auch die in den Legacy-Tests.
    """
    b = _bundle()
    assert b.wall_ts == {}
    assert b.fluid_props_names == ()


def test_save_bundle_verweigert_eine_unverzeichnete_schemaversion(tmp_path):
    """``schema_versions.md`` ist keine Zierde: ohne Eintrag kein Schreiben.

    Das ist der Mechanismus, der verhindert, dass eine Version still im Cache
    landet, die niemand beschrieben hat.
    """
    with pytest.raises(Exception, match="99"):
        C.save_bundle(_bundle(schema_version=99), target_dir=tmp_path)


def test_die_gebaute_version_ist_in_schema_versions_verzeichnet():
    """Und die Gegenrichtung: was build.yaml sagt, muss beschrieben sein."""
    from battery_surrogate.data.paths import build_config_path, schema_versions_path
    import yaml

    version = int(yaml.safe_load(build_config_path().read_text(encoding="utf-8"))
                  ["schema_version"])
    assert version >= 3, "Stufe 2 hebt das Schema auf mindestens 3"
    text = schema_versions_path().read_text(encoding="utf-8")
    assert f"## v{version}" in text
