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
    "Cell Current Monitor (A)": "cell_current",
    "Fluid Initial Temperature Monitor (C)": "fluid_initial_temp",
    "Fluid Inlet Temperature Monitor (C)": "fluid_inlet_temp",
    "Fluid Mass Flow Monitor (kg/s)": "fluid_mass_flow",
    "SOC Start Monitor (%)": "soc_start",
    "SOC Start Monitor": "soc_start",
    "Solid Initial Temperature Monitor (C)": "solid_initial_temp",
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
