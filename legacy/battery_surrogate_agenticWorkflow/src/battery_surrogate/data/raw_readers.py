"""Parsers for the raw workflow input files."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..schema.columns import (
    BATEMO_FMU1_KEEP_COLUMNS,
    FLUID_PROP_COLUMNS,
    HEAT_TRANSFER_COLUMNS,
    INPUTSIGNALE_COLUMN_ALIASES,
    T_GRID_LAYER_COLUMNS,
    TEMPERATUREN_COLUMNS,
    TIME_COLUMN,
)
from ..schema.inputsignale import InputSentinel, parse_inputsignale_value


def _read_table(path: Path, encoding: str) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, encoding=encoding)


def _first_matching_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    columns = list(frame.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lowered = {str(column).casefold(): column for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.casefold())
        if match is not None:
            return match
    raise KeyError(f"No matching column found. Expected one of: {candidates!r}")


def read_batemo_fmu1(path: Path, encoding: str) -> pd.DataFrame:
    """Read FMU1 and keep only the columns used by the surrogate workflow."""

    frame = _read_table(path, encoding=encoding)
    renamed: dict[str, str] = {"Physical Time (s)": "physical_time_s"}
    renamed.update(BATEMO_FMU1_KEEP_COLUMNS)
    keep = [column for column in frame.columns if column in renamed]
    result = frame[keep].rename(columns=renamed)
    return result


def read_heat_source(path: Path, encoding: str) -> pd.DataFrame:
    """Read the heat-source time series."""

    frame = _read_table(path, encoding=encoding)
    renamed = {
        "Physical Time (s)": "physical_time_s",
        "Heat Source JR1 Monitor (W)": "jr1_w",
        "Heat Source JR2 Monitor (W)": "jr2_w",
        "Heat Source Monitor (W)": "total_w",
    }
    keep = [column for column in frame.columns if column in renamed]
    return frame[keep].rename(columns=renamed)


def read_t_grid(path: Path, layer: str, encoding: str) -> pd.DataFrame:
    """Read one thermal grid file and give the columns stable names."""

    frame = _read_table(path, encoding=encoding)
    prefix = T_GRID_LAYER_COLUMNS[layer]
    renamed: dict[str, str] = {"Physical Time (s)": "physical_time_s"}
    for column in frame.columns:
        text = str(column)
        if text.startswith(prefix):
            parts = text.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                renamed[column] = f"{prefix}_{parts[2]}"
            else:
                renamed[column] = text.replace(" Monitor (C)", "")
    keep = [column for column in frame.columns if column in renamed]
    return frame[keep].rename(columns=renamed)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """First candidate that is present, case-insensitively as a fallback."""

    for name in candidates:
        if name in frame.columns:
            return name
    lowered = {str(c).casefold(): c for c in frame.columns}
    for name in candidates:
        hit = lowered.get(name.casefold())
        if hit is not None:
            return hit
    return None


def _read_named_series(
    path: Path,
    encoding: str,
    wanted: dict[str, tuple[str, ...]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (time, values)}`` -- every series with ITS OWN time axis.

    Jede dieser CSVs bringt eine eigene Zeitachse mit, und sie sind nicht gleich
    lang. Genau diese Annahme ist am 22.09. in ``balance_check.py`` aufgeflogen
    (``fp and xp are not of the same length``), nachdem sie an anderer Stelle
    still durchgelaufen war. Deshalb wird die Achse hier MITGESCHRIEBEN und
    nicht auf ``t_slow`` gelegt: wer sie braucht, interpoliert sichtbar.
    """
    frame = _read_table(path, encoding=encoding)
    time_column = _pick_column(frame, TIME_COLUMN)
    if time_column is None:
        raise KeyError(
            f"{path.name} has no time column; looked for {list(TIME_COLUMN)}, "
            f"found {list(frame.columns)}"
        )
    times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=np.float32)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, candidates in wanted.items():
        column = _pick_column(frame, candidates)
        if column is None:
            raise KeyError(
                f"{path.name} has no column for {name!r}; looked for "
                f"{list(candidates)}, found {list(frame.columns)}"
            )
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        out[name] = (times, values)
    return out


def read_heat_transfer(path: Path, encoding: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``q_solid_to_fluid`` mit eigener Zeitachse, aus ``*_Heat Transfer.csv``."""

    return _read_named_series(path, encoding, HEAT_TRANSFER_COLUMNS)


def read_temperaturen(path: Path, encoding: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``fluid_out_temp`` mit eigener Zeitachse, aus ``*_Temperaturen.csv``."""

    return _read_named_series(path, encoding, TEMPERATUREN_COLUMNS)


def read_fluidstoffwerte(
    path: Path, encoding: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fluidstoffwerte als 1xk-Zeile PLUS die Namen ihrer Spalten.

    Bis Schema v2 wurden hier die ersten drei numerischen Spalten genommen und
    der Vertrag riet ihre Bedeutung. Ab v3 werden sie benannt gesucht; die
    Reihenfolge des Exports entscheidet nichts mehr. Die Rueckgabeform bleibt
    ``(1, 3)``, damit alles, was ``fluid_props`` heute liest, weiterlaeuft --
    aber die Namen stehen jetzt daneben.
    """
    frame = _read_table(path, encoding=encoding)
    names: list[str] = []
    values: list[float] = []
    for name, candidates in FLUID_PROP_COLUMNS.items():
        column = _pick_column(frame, candidates)
        if column is None:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            continue
        names.append(name)
        values.append(float(finite.mean()))

    if not names:
        # Kein benannter Treffer: die alte positionelle Lesart als Rueckfall,
        # damit ein aelterer Export nicht den ganzen Rebuild anhaelt. Die Namen
        # sagen dann ausdruecklich, dass geraten wurde.
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] < 3:
            numeric = frame.apply(pd.to_numeric, errors="coerce")
        row = numeric.iloc[0, :3].to_numpy(dtype=np.float32).reshape(1, 3)
        return row, ("unbenannt_0", "unbenannt_1", "unbenannt_2")

    return np.asarray(values, dtype=np.float32).reshape(1, -1), tuple(names)


def read_inputsignale(path: Path, encoding: str) -> dict[str, float | InputSentinel]:
    """Read Inputsignale and convert every cell to a number or sentinel."""

    frame = pd.read_csv(path, encoding=encoding, dtype=str)
    frame = frame.rename(columns=INPUTSIGNALE_COLUMN_ALIASES)
    row = frame.iloc[0].to_dict()
    result: dict[str, float | InputSentinel] = {}
    for channel, raw_value in row.items():
        result[channel] = parse_inputsignale_value(raw_value, channel)
    return result


def read_time_series_input(
    path: Path,
    channel: str,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a profile file and return its time axis plus values."""

    frame = _read_table(path, encoding=encoding)
    time_column = _first_matching_column(frame, ["Physical Time (s)", "Time (s)"])
    value_candidates = {
        "c_rate": ["C-Rate Monitor (1/h)", "C-Rate (1/h)", "C-Rate", "c_rate", "crate"],
        "cell_current": ["Cell Current (A)", "Cell Current Monitor (A)"],
        "fluid_mass_flow": ["Fluid Mass Flow (kg/s)", "Fluid Mass Flow Monitor (kg/s)"],
        "fluid_inlet_temp": ["Fluid Inlet Temperature (C)", "Fluid Inlet Temperatur (C)"],
    }
    value_column = _first_matching_column(frame, value_candidates[channel])
    times = frame[time_column].to_numpy(dtype=np.float32)
    values = frame[value_column].to_numpy(dtype=np.float32)
    return times, values


def read_module_test_data(path: Path, encoding: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read module-test data and return time series for the channels it provides."""

    frame = _read_table(path, encoding=encoding)
    time_column = _first_matching_column(frame, ["Physical Time (s)"])
    channel_candidates = {
        "c_rate": ["C-Rate Monitor (1/h)", "C-Rate (1/h)", "C-Rate", "c_rate", "crate"],
        "cell_current": ["Cell Current (A)"],
        "fluid_mass_flow": ["Fluid Mass Flow (kg/s)"],
        "fluid_inlet_temp": ["Fluid Inlet Temperature (C)", "Fluid Inlet Temperatur (C)"],
    }
    times = frame[time_column].to_numpy(dtype=np.float32)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel, candidates in channel_candidates.items():
        try:
            value_column = _first_matching_column(frame, candidates)
        except KeyError:
            continue
        values = frame[value_column].to_numpy(dtype=np.float32)
        result[channel] = (times, values)
    return result
