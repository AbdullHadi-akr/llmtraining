"""Dataclasses used by the workflow package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpBundle:
    """One fully assembled OP bundle ready for cache writing or loading."""

    op_id: str
    schema_version: int
    cache_key: str
    t_fast: np.ndarray
    t_slow: np.ndarray
    bc_V: np.ndarray
    bc_OCV: np.ndarray
    bc_I: np.ndarray
    pe_P_loss: np.ndarray
    T: np.ndarray
    q_source: np.ndarray
    xyz: np.ndarray
    layer: np.ndarray
    sensor_id: np.ndarray
    fluid_props: np.ndarray
    sim_config_scalar: np.ndarray
    sim_config_scalar_names: tuple[str, ...]
    sim_config_ts: dict[str, tuple[np.ndarray, np.ndarray]]
    meta: dict[str, Any]

    # --- Schema v3: der Wandpfad ---------------------------------------
    # Jede Reihe traegt IHRE EIGENE Zeitachse, wie ``sim_config_ts`` auch.
    # Das ist kein Stilentscheid: ``*_Heat Transfer.csv`` und
    # ``*_Temperaturen.csv`` haben andere Achsen als ``*_Heat Source.csv``,
    # und die stillschweigende Annahme, alles liege auf einer, ist am 22.09.
    # in ``GridCNN/tools/balance_check.py`` aufgeflogen.
    #
    #   ``q_solid_to_fluid``  W    Waermestrom Festkoerper -> Fluid
    #   ``fluid_out_temp``    C    Fluid-Auslasstemperatur
    #
    # Beide zusammen machen ``L_wall`` anschliessbar -- den einzigen
    # Verlustterm mit einem GEMESSENEN Ziel. Siehe GridCNN/FAHRPLAN.md,
    # Stufe 2.
    #
    # Default leer, damit ein Bundle aus Schema v2 weiterhin laedt und
    # konstruierbar bleibt: der Rebuild fuellt es, das Fehlen ist kein
    # Fehler, sondern "noch nicht neu gebaut".
    wall_ts: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # Namen der Spalten in ``fluid_props``, in derselben Reihenfolge.
    # Bis v2 war das positionell und der Vertrag riet die Bedeutung falsch
    # ("typically density, specific heat, thermal conductivity"; gemessen ist
    # es Conductivity, Density, Specific Heat). Leer = altes Bundle, Bedeutung
    # unbekannt -- und dann darf sie auch niemand annehmen.
    fluid_props_names: tuple[str, ...] = ()
