"""Parsers for the raw workflow input files."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..schema.columns import (
    BATEMO_FMU1_KEEP_COLUMNS,
    INPUTSIGNALE_COLUMN_ALIASES,
    T_GRID_LAYER_COLUMNS,
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


def read_fluidstoffwerte(path: Path, encoding: str) -> np.ndarray:
    """Read the fluid property row and return a 1x3 array."""

    frame = _read_table(path, encoding=encoding)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] < 3:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.iloc[0, :3].to_numpy(dtype=np.float32)
    return values.reshape(1, 3)


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
