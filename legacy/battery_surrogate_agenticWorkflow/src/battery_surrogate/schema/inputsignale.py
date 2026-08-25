"""Inputsignale parsing and sentinel handling."""

from __future__ import annotations

from enum import Enum


class InputSentinel(Enum):
    """Special values that appear in Inputsignale.csv instead of numbers."""

    FILE_TABLE = "file_table"
    TEMPERATURPROFIL = "temperaturprofil"
    VOLUMENSTROMPROFIL = "volumenstromprofil"
    MODUL_TEST_DATA = "modul_test_data"
    DERIVE_FROM_OCV = "derive_from_ocv"


_SENTINEL_ALIASES: dict[str, set[str]] = {
    "cell_current": {"file table"},
    "fluid_inlet_temp": {"temperaturprofil"},
    "fluid_mass_flow": {"volumenstromprofil"},
    "soc_start": {"nicht def. -> ocv", "nicht def.", "nicht definiert -> ocv"},
}


def parse_inputsignale_value(raw_value: object, channel: str) -> float | InputSentinel:
    """Parse one raw Inputsignale cell into a number or a known sentinel."""

    if raw_value is None:
        raise ValueError(f"Inputsignale cell for {channel} is empty")

    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"Inputsignale cell for {channel} is empty")

    try:
        return float(text.replace(",", "."))
    except ValueError:
        pass

    lowered = text.casefold()
    if lowered in _SENTINEL_ALIASES.get(channel, set()):
        if channel == "cell_current":
            return InputSentinel.FILE_TABLE
        if channel == "fluid_inlet_temp":
            return InputSentinel.TEMPERATURPROFIL
        if channel == "fluid_mass_flow":
            return InputSentinel.VOLUMENSTROMPROFIL
        if channel == "soc_start":
            return InputSentinel.DERIVE_FROM_OCV

    if lowered in {
        "siehe modultestdata",
        "siehe moduletestdata",
        "test data",
        "siehe modultest daten",
    }:
        return InputSentinel.MODUL_TEST_DATA

    raise ValueError(f"Unknown Inputsignale value for {channel}: {text!r}")
