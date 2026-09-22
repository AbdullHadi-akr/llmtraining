"""Canonical column names and file-name constants for the workflow."""

from __future__ import annotations

CANONICAL_CHANNELS: tuple[str, ...] = (
    "c_rate",
    "cell_current",
    "fluid_initial_temp",
    "fluid_inlet_temp",
    "fluid_mass_flow",
    "soc_start",
    "solid_initial_temp",
)

BATEMO_FMU1_KEEP_COLUMNS: dict[str, str] = {
    "bc_V Monitor": "bc_V",
    "bc_OCV Monitor": "bc_OCV",
    "bc_I Monitor": "bc_I",
    "pe_P_loss Monitor": "pe_P_loss",
}

INPUTSIGNALE_COLUMN_ALIASES: dict[str, str] = {
    "C-Rate Monitor (1/h)": "c_rate",
    "C-Rate Monitor": "c_rate",
    "C-Rate": "c_rate",
    "Cell Current Monitor (A)": "cell_current",
    "Cell Current Monitor": "cell_current",
    "Fluid Initial Temperature Monitor (C)": "fluid_initial_temp",
    "Fluid Initial Temperature Monitor": "fluid_initial_temp",
    "Fluid Inlet Temperature Monitor (C)": "fluid_inlet_temp",
    "Fluid Inlet Temperature Monitor": "fluid_inlet_temp",
    "Fluid Mass Flow Monitor (kg/s)": "fluid_mass_flow",
    "Fluid Mass Flow Monitor": "fluid_mass_flow",
    "SOC Start Monitor (%)": "soc_start",
    "SOC Start Monitor": "soc_start",
    "Solid Initial Temperature Monitor (C)": "solid_initial_temp",
    "Solid Initial Temperature Monitor": "solid_initial_temp",
}

# --- Schema v3: der Wandpfad -------------------------------------------------
# Die Namen sind nicht geraten. Sie stammen aus GridCNN/tools/balance_check.py,
# das am 22.09. gegen die echten Exporte aller sieben Konstant-Treiber-OPs
# gelaufen ist -- dieselben Kandidatenlisten, dieselbe Reihenfolge. Jeder
# Eintrag hat dort getroffen; die zweite Variante ist der Export ohne
# "Monitor" im Namen.
#
# WARUM sie in den Cache muessen: ohne q_solid_to_fluid und fluid_out_temp ist
# der Wandterm nicht an die gemessene Waerme anschliessbar -- L_wall ist der
# einzige Verlustterm mit einem gemessenen Ziel. Siehe GridCNN/FAHRPLAN.md,
# Stufe 2.
TIME_COLUMN: tuple[str, ...] = ("Physical Time (s)",)

HEAT_TRANSFER_COLUMNS: dict[str, tuple[str, ...]] = {
    "q_solid_to_fluid": (
        "Heat Transfer: solid to fluid Monitor (W)",
        "Heat Transfer: solid to fluid (W)",
    ),
}

TEMPERATUREN_COLUMNS: dict[str, tuple[str, ...]] = {
    "fluid_out_temp": (
        "Tmfavg_fluid_out Monitor (C)",
        "Tmfavg_fluid_out (C)",
    ),
}

# ``*_Fluidstoffwerte.csv`` wurde bis v2 POSITIONELL gelesen -- die ersten drei
# numerischen Spalten, und der Vertrag riet ihre Bedeutung ("typically density,
# specific heat, thermal conductivity"). Gemessen am 22.09. stehen sie in der
# Reihenfolge Conductivity, Density, Specific Heat; die Vermutung war also
# falsch. Ab v3 werden die Spalten benannt mitgeschrieben, damit niemand mehr
# raten muss, welche Zahl welche ist.
FLUID_PROP_COLUMNS: dict[str, tuple[str, ...]] = {
    "conductivity": (
        "Conductivity Monitor (W/m-K)", "Conductivity (W/m-K)",
    ),
    "density": (
        "Density Monitor (kg/m^3)", "Density (kg/m^3)",
    ),
    "cp_fluid": (
        "Specific Heat Monitor (J/kg-K)", "Specific Heat (J/kg-K)",
    ),
}


PROFILE_CHANNELS: tuple[str, ...] = (
    "cell_current",
    "fluid_inlet_temp",
    "fluid_mass_flow",
)

MODULE_TEST_CHANNELS: tuple[str, ...] = (
    "cell_current",
    "fluid_mass_flow",
    "fluid_inlet_temp",
)

COORDINATE_FILES: tuple[tuple[str, str], ...] = (
    ("cc", "Coordinates - Grid Cell Center.csv"),
    ("g", "Coordinates - Grid Gehäusewand.csv"),
    ("jr1c", "Coordinates - Grid JR1 Center.csv"),
)

T_GRID_LAYER_COLUMNS: dict[str, str] = {
    "cc": "T_grid_cc",
    "g": "T_grid_g",
    "jr1c": "T_grid_jr1c",
}
